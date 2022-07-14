import torch
from typing import List

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

def copy_layer(m_src : torch.nn.Module, m_dst : torch.nn.Module):
    for (n1, p1), (n2, p2) in zip(m_src.named_parameters(), m_dst.named_parameters()):
        if n1 != n2:
            raise RuntimeError("Parameter `%s` != `%s`" % (n1, n2))
        p2.copy_(p1, non_blocking=True)

class DeviceLayerScheduler:
    def __init__(self, layers : List[torch.nn.Module], device_id):
        self._device = torch.cuda.device(device_id)
        self._num_layers = len(layers)
        
        self._fixed_layers = set()
        self._active_layers = {}
        self._sched_layers = []
        self._layers = []

        with self._device:
            self._load_stream = torch.cuda.stream(torch.cuda.Stream())
            if self._num_layers > 0:
                free_mem = torch.cuda.mem_get_info()[0]

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
                    elif len(self._sched_layers) < sched_layers:
                        self._layers.append( pin_layer(layers[i]) )
                        self._active_layers[i] = len(self._sched_layers)
                        self._sched_layers.append({
                            "layer": layers[i].cuda(),
                            "evt": torch.cuda.Event(),
                            "unused": True
                        })
                    else:
                        self._layers.append( pin_layer(layers[i]) )

    def _get_unused_buffer_id(self):
        for i, buf in enumerate(self._sched_layers):
            if buf["unused"]:
                return i
        return None

    def __iter__(self):
        try:
            for i in range(self._num_layers):
                # layer prefetch
                for j in range(i + 1, self._num_layers):
                    if (j not in self._fixed_layers) and (j not in self._active_layers):
                        buf_id = self._get_unused_buffer_id()
                        if buf_id is None:
                            # no available buffer
                            break
                        with self._device:
                            with self._load_stream:
                                torch.cuda.current_stream().wait_event(self._sched_layers[buf_id]["evt"])
                                # copy to buffer async
                                copy_layer(self._layers[j], self._sched_layers[buf_id]["layer"])

                                # event record after async load
                                self._sched_layers[buf_id]["evt"].record(torch.cuda.current_stream())
                                self._sched_layers[buf_id]["unused"] = False


                if (i not in self._fixed_layers) and (i not in self._active_layers):
                    raise RuntimeError("Layer %d is not on device" % i)
                with self._device:
                    if i in self._fixed_layers:
                        yield self._layers[i]
                    else:
                        buf_id = self._active_layers[i]
                        torch.cuda.current_stream().wait_event(self._sched_layers[buf_id]["evt"])

                        yield self._sched_layers[buf_id]["layer"]
                        
                        # record event after calc
                        self._sched_layers[buf_id]["evt"].record(torch.cuda.current_stream())
                        self._sched_layers[buf_id]["unused"] = True
                    
        finally:
            with self._device:
                # clear buffer
                for buf in self._sched_layers:
                    buf["unused"] = True
                    buf["evt"].record(torch.cuda.current_stream())


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
    
    def __getitem__(self, key):
        return self.layers[key]
    
    def __iter__(self):
        for sched in self._scheds:
            for layer in sched:
                yield layer
