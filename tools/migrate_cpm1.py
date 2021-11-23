import torch
import numpy as np
from bminf.core import Parameter
from bminf.arch import GPT2Model, GPTConfiguration 
from bminf.layers.transformer_block import DecoderBlock
import cpm_kernels.kernels as ck

device = torch.device("cuda:0")


def build_parameter(tensor, parameter : Parameter):
    tp = parameter.dtype
    v = tensor.cpu().numpy().astype(tp)
    shape = v.shape
    parameter.put_data(shape, v.tobytes(), tp)

def scale_build_parameter(tensor : torch.Tensor, value : Parameter, scale : Parameter):
    tensor = tensor.to(device).half()
    n, m = tensor.size()

    scale_v = torch.empty((n,), dtype=torch.float16, device=device)
    ck.gemm_calc_scale(
        1, n, m,
        tensor.data_ptr(),
        scale_v.data_ptr(),
        torch.cuda.current_stream()
    )
    quant_v = torch.empty((n, m), dtype=torch.int8, device=device)
    ck.gemm_round(
        1, n, m,
        tensor.data_ptr(),
        scale_v.data_ptr(),
        quant_v.data_ptr(),
        torch.cuda.current_stream()
    )

    scale_v = scale_v.cpu().numpy().astype(np.float16)
    qv = quant_v.cpu().numpy().astype(np.int8)

    value.put_data(qv.shape, qv.tobytes(), qv.dtype)
    scale.put_data(scale_v.shape, scale_v.tobytes(), scale_v.dtype)

def split(x, s):
    sizes = []
    for it in x.size():
        sizes.append(it)
    assert sizes[0] % s == 0
    sizes = [s, sizes[0] // s ] + sizes[1:]
    return x.reshape(*sizes)

def build_block(ckpt, model : DecoderBlock, prefix):
    build_parameter(ckpt[f"{prefix}.input_layernorm.weight"], model.ln_attn.weight)
    build_parameter(ckpt[f"{prefix}.input_layernorm.bias"], model.ln_attn.bias)

    split_attn_weight = split(ckpt[f"{prefix}.attention.query_key_value.weight"], 3)
    split_attn_bias = split(ckpt[f"{prefix}.attention.query_key_value.bias"], 3)
    scale_build_parameter(split_attn_weight[0], model.self_attn.project_q.weight, model.self_attn.project_q.scale)
    scale_build_parameter(split_attn_weight[1], model.self_attn.project_k.weight, model.self_attn.project_k.scale)
    scale_build_parameter(split_attn_weight[2], model.self_attn.project_v.weight, model.self_attn.project_v.scale)
    build_parameter(split_attn_bias[0], model.self_attn.project_q.bias)
    build_parameter(split_attn_bias[1], model.self_attn.project_k.bias)
    build_parameter(split_attn_bias[2], model.self_attn.project_v.bias)

    scale_build_parameter(ckpt[f"{prefix}.attention.dense.weight"], model.self_attn.linear_out.weight, model.self_attn.linear_out.scale)
    build_parameter(ckpt[f"{prefix}.attention.dense.bias"], model.self_attn.linear_out.bias)

    build_parameter(ckpt[f"{prefix}.post_attention_layernorm.weight"], model.ln_ff.weight)
    build_parameter(ckpt[f"{prefix}.post_attention_layernorm.bias"], model.ln_ff.bias)
    scale_build_parameter(ckpt[f"{prefix}.mlp.dense_h_to_4h.weight"], model.ff.linear_in.weight, model.ff.linear_in.scale)
    build_parameter(ckpt[f"{prefix}.mlp.dense_h_to_4h.bias"], model.ff.linear_in.bias)
    scale_build_parameter(ckpt[f"{prefix}.mlp.dense_4h_to_h.weight"], model.ff.linear_out.weight, model.ff.linear_out.scale)
    build_parameter(ckpt[f"{prefix}.mlp.dense_4h_to_h.bias"], model.ff.linear_out.bias)

def build_layers(ckpt, model : GPT2Model):
    for i in range(model.num_layers):
        build_block(ckpt, model.layers[i], f"transformer.layers.{i}")


def build_model(ckpt, model : GPT2Model):
    build_parameter(ckpt["word_embeddings.weight"], model.token_embedding.weight) 
    build_parameter(ckpt["position_embeddings.weight"], model.position_embedding.weight)
    build_parameter(ckpt["transformer.final_layernorm.weight"], model.layernorm.weight)
    build_parameter(ckpt["transformer.final_layernorm.bias"], model.layernorm.bias)

    build_layers(ckpt, model)

def main():
    model = torch.load("merge.pt")
    config = GPTConfiguration()
    config.MODEL_NAME = None
    cpm1 = GPT2Model(config=config)
    build_model(model, cpm1)
    cpm1.dump(open("checkpoint.pt", "wb"))

if __name__ == "__main__":
    main()
    