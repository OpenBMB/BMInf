import torch
from typing import List, Optional
from cpm_kernels.library import cudart

def calc_fixed_layers(total_layers : int, max_fixed : int):
    max_fixed = min(max_fixed, total_layers)
    scheduled_layers = total_layers - max_fixed
    vals = [(i + 1) * scheduled_layers // total_layers for i in range(total_layers)]
    ret = []
    last_v = 0
    for i, v in enumerate(vals):
        if v == last_v:
            ret.append(i)
        else:
            last_v = v
    return ret

def pin_layer(m : torch.nn.Module):
    for param in m.parameters():
        with torch.no_grad():
            param.data = param.data.pin_memory()
    return m

def transfer_layers(m_src : torch.nn.Module, m_dst : dict):
    with torch.no_grad():
        for name, param in m_src.named_parameters():
            assert name in m_dst
            # copy to device buffer
            m_dst[name].copy_(param, non_blocking=True)

def swap_params(m_src : torch.nn.Module, m_dst : dict):
    with torch.no_grad():
        for name, param in m_src.named_parameters():
            assert name in m_dst

            # swap memory info
            tmp = m_dst[name].data
            m_dst[name].data = param.data
            param.data = tmp

class OpDeviceLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self : 'DeviceLayerScheduler', hidden_state, num_kw, *args):
        tensors = []
        others = []
        for arg in args:
            if torch.is_tensor(arg):
                tensors.append(arg)
                others.append(None)
            else:
                tensors.append(None)
                others.append(arg)
    
        ctx.nontensor_inputs = others
        ctx.num_kw = num_kw
        ctx.self = self
        layer_inputs = []
        cuda_rng_state = []

        if num_kw > 0:
            kw_arg_names = args[-num_kw*2::2]
            kw_arg_vals = args[-num_kw*2+1::2]
            kwargs = {kw : val for kw, val in zip(kw_arg_names, kw_arg_vals)}
            args = args[:-num_kw*2]
        else:
            kwargs = {}
            args = args

        with torch.no_grad():
            for i in range(len(self)):
                layer_inputs.append(hidden_state)
                cuda_rng_state.append( torch.cuda.get_rng_state() )
                
                self._look_ahead(range(i, len(self)))
                if (i not in self._fixed_layers) and (i not in self._active_layers):
                    raise RuntimeError("Layer %d is not on device" % i)
                with self._device:
                    if i in self._fixed_layers:
                        hidden_state = self._layers[i](hidden_state, *args, **kwargs)
                    else:
                        buf_id = self._active_layers[i]
                        torch.cuda.current_stream().wait_event(self._sched_layers[buf_id]["evt"])

                        # swap device parameters and cpu parameters
                        swap_params(self._layers[i], self._sched_layers[buf_id]["parameters"])
                        try:
                            hidden_state = self._layers[i](hidden_state, *args, **kwargs)
                        finally:
                            swap_params(self._layers[i], self._sched_layers[buf_id]["parameters"])
                            # record event after calc
                            self._sched_layers[buf_id]["evt"].record(torch.cuda.current_stream())
                            self._sched_layers[buf_id]["unused"] = True

        ctx.cuda_rng_state = cuda_rng_state        
        ctx.save_for_backward(*layer_inputs, *tensors)
        ctx.num_save_needed = len(layer_inputs)
        return hidden_state


    @staticmethod
    def backward(ctx, grad_hidden_state : torch.Tensor):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")
        all_inputs = []
        input_requires_grad = []
        
        layer_inputs = ctx.saved_tensors[:ctx.num_save_needed]
        save_args = ctx.saved_tensors[ctx.num_save_needed:]
        for tensor, other in zip(save_args, ctx.nontensor_inputs):
            if tensor is None:
                all_inputs.append(other)
                input_requires_grad.append(False)
            else:
                # detach for tensor inputs
                input_requires_grad.append( tensor.requires_grad )
                nw_tensor = tensor.detach()
                nw_tensor.requires_grad = tensor.requires_grad
                all_inputs.append(nw_tensor)
        
        if ctx.num_kw > 0:
            kw_arg_names = all_inputs[-ctx.num_kw*2::2]
            kw_arg_vals = all_inputs[-ctx.num_kw*2+1::2]
            kwargs = {kw : val for kw, val in zip(kw_arg_names, kw_arg_vals) }
            args = all_inputs[:-ctx.num_kw*2]
        else:
            kwargs = {}
            args = all_inputs
        self : DeviceLayerScheduler = ctx.self
        
        with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=True):
            with torch.enable_grad():
                # overlap load and scatter here
                for i in reversed(range(len(self))):
                    self._look_ahead( range(i, -1, -1) )
                    torch.cuda.set_rng_state(ctx.cuda_rng_state[i])
                    ipt = layer_inputs[i].detach().requires_grad_()

                    if (i not in self._fixed_layers) and (i not in self._active_layers):
                        raise RuntimeError("Layer %d is not on device" % i)
                    with self._device:
                        if i in self._fixed_layers:
                            output = self._layers[i](ipt, *args, **kwargs)
                        else:
                            buf_id = self._active_layers[i]
                            torch.cuda.current_stream().wait_event(self._sched_layers[buf_id]["evt"])

                            # swap device parameters and cpu parameters
                            swap_params(self._layers[i], self._sched_layers[buf_id]["parameters"])
                            try:
                                output = self._layers[i](ipt, *args, **kwargs)
                            finally:
                                swap_params(self._layers[i], self._sched_layers[buf_id]["parameters"])
                                # record event after calc
                                self._sched_layers[buf_id]["evt"].record(torch.cuda.current_stream())
                                self._sched_layers[buf_id]["unused"] = True
                            
                    torch.autograd.backward(
                        [output],
                        [grad_hidden_state]
                    )
                    grad_hidden_state = ipt.grad

        grads = []
        for inp, requires_grad in zip(all_inputs, input_requires_grad):
            if requires_grad:
                grads.append(inp.grad)
            else:
                grads.append(None)
        return (None, grad_hidden_state, None) + tuple(grads)

class DeviceLayerScheduler:
    def __init__(self, layers : List[torch.nn.Module], device_id : int, memory_limit : Optional[int] = None):
        self._device = torch.cuda.device(device_id)
        self._num_layers = len(layers)
        
        self._fixed_layers = set()
        self._sched_layers = []
        self._layers = []
        self._active_layers = {}

        with self._device:
            with torch.no_grad():
                self._load_stream = torch.cuda.stream(torch.cuda.Stream())
                if self._num_layers > 0:
                    if memory_limit is None:
                        free_mem = cudart.cudaMemGetInfo()[0]
                    else:
                        free_mem = memory_limit

                    total_size = 0
                    for param in layers[0].parameters():
                        total_size += param.numel() * param.storage().element_size()
                    
                    total_layers = free_mem // total_size
                    if total_layers < 2:
                        raise OSError("CUDA out of memory on device %d" % device_id)
                    
                    sched_layers = 2
                    if total_layers >= self._num_layers:
                        sched_layers = 0
                    fixed_layers = total_layers - sched_layers

                    layer_id_to_fix = calc_fixed_layers(self._num_layers, fixed_layers)
                    self._fixed_layers = set(layer_id_to_fix)
                    for i in range(self._num_layers):
                        if i in self._fixed_layers:
                            self._layers.append( layers[i].cuda() )
                        else:
                            self._layers.append( pin_layer(layers[i].cpu()) )
                    
                    for i in range(self._num_layers):
                        if len(self._sched_layers) >= sched_layers:
                            break
                        if i not in self._fixed_layers:
                            self._active_layers[i] = len(self._sched_layers)
                            self._sched_layers.append({
                                "parameters": { name: param.cuda() for name, param in layers[i].named_parameters()},
                                "evt": torch.cuda.Event(),
                                "id": i,
                                "unused": True
                            })
    
    def __len__(self):
        return self._num_layers

    def _get_unused_buffer_id(self):
        for i, buf in enumerate(self._sched_layers):
            if buf["unused"]:
                return i
        return None
    
    def _look_ahead(self, layer_ids : List[int]):
        for j in layer_ids:
            if j not in self._fixed_layers:
                # need sched
                if j in self._active_layers:
                    # already in buffer, just mark as used
                    buf_id = self._active_layers[j]
                    assert self._sched_layers[buf_id]["id"] == j
                    self._sched_layers[buf_id]["unused"] = False
                    continue

                # else not in buffer, get an unused buffer id
                buf_id = self._get_unused_buffer_id()
                if buf_id is None:
                    # no available buffer
                    break

                # remove old id from active layers and set new id
                del self._active_layers[self._sched_layers[buf_id]["id"]]
                self._sched_layers[buf_id]["id"] = j
                self._active_layers[j] = buf_id

                with self._device:
                    with self._load_stream:
                        # wait for calc stream
                        torch.cuda.current_stream().wait_event(self._sched_layers[buf_id]["evt"])

                        # copy to buffer async
                        transfer_layers(self._layers[j], self._sched_layers[buf_id]["parameters"])

                        # event record after async load
                        self._sched_layers[buf_id]["evt"].record(torch.cuda.current_stream())
                        self._sched_layers[buf_id]["unused"] = False

    def __iter__(self):
        with torch.no_grad():
            try:
                for i in range(self._num_layers):
                    # layer prefetch
                    self._look_ahead(range(i, self._num_layers))

                    if (i not in self._fixed_layers) and (i not in self._active_layers):
                        raise RuntimeError("Layer %d is not on device" % i)
                    with self._device:
                        if i in self._fixed_layers:
                            yield self._layers[i]
                        else:
                            buf_id = self._active_layers[i]
                            torch.cuda.current_stream().wait_event(self._sched_layers[buf_id]["evt"])

                            # swap device parameters and cpu parameters
                            swap_params(self._layers[i], self._sched_layers[buf_id]["parameters"])
                            try:
                                yield self._layers[i]
                            finally:
                                swap_params(self._layers[i], self._sched_layers[buf_id]["parameters"])
                                # record event after calc
                                self._sched_layers[buf_id]["evt"].record(torch.cuda.current_stream())
                                self._sched_layers[buf_id]["unused"] = True
                        
            finally:
                with self._device:
                    # clear buffer
                    for buf in self._sched_layers:
                        buf["unused"] = True
                        buf["evt"].record(torch.cuda.current_stream())

    def forward(self, x, *args, **kwargs):
        lst = list(args)
        for k, v in kwargs.items():
            lst.append(k)
            lst.append(v)
        with self._device: 
            x = x.cuda(non_blocking=True)
            cuda_lst = []
            for it in lst:
                if torch.torch.is_tensor(it):
                    cuda_lst.append( it.cuda(non_blocking=True) )
                else:
                    cuda_lst.append(it)
            return OpDeviceLayer.apply(self, x, len(kwargs), *cuda_lst)

class TransformerBlockList(torch.nn.Module):
    def __init__(self, layers : List[torch.nn.Module], gpus : List[int]):
        super().__init__()

        self.layers = layers

        self._parameter_info = {}
        self._scheds = []

        if len(layers) == 0:
            raise ValueError("No layers in list")

        # get parameter info of layer 0
        for name, v in layers[0].named_parameters():
            self._parameter_info[name] = {
                "shape": v.size(),
                "type": v.dtype,
            }
        
        # check parameters are the same in all layers
        for i, layer in enumerate(layers):
            p_in_layer = 0  # number of parameters in this layer
            for name, v in layer.named_parameters():
                if name not in self._parameter_info:
                    raise ValueError("Unknown parameter `%s`" % name)
                else:
                    info = self._parameter_info[name]
                    if info["shape"] != v.size() or info["type"] != v.dtype:
                        raise ValueError("Parameter `%s` not the same" % name)
                    p_in_layer += 1
                v.requires_grad_(False) # parameters in layers can't be updated
            if p_in_layer != len(self._parameter_info):
                raise ValueError("Missing some parameters in layer %d" % i)
        
        # devide into GPUs
        num_gpus = len(gpus)
        layers_per_gpu = (len(layers) + num_gpus - 1) // num_gpus   # round_up
        for i in range(num_gpus):
            self._scheds.append(
                DeviceLayerScheduler(
                    layers[i * layers_per_gpu: (i + 1) * layers_per_gpu],
                    gpus[i]
                )
            )
    
        for i, layer in enumerate(layers):
            self.add_module("%d" % i, layer)
    
    def __getitem__(self, key):
        return self.layers[key]
    
    def __iter__(self):
        for sched in self._scheds:
            for layer in sched:
                yield layer

    def forward(self, x, *args, **kwargs):
        for sched in self._scheds:
            x = sched.forward(x, *args, **kwargs)
        return x
