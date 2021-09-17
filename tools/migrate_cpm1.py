import torch
import numpy as np
from bminference.models.cpm1 import CPM1, CPM1Configuration
from bminference.parameter import Parameter
from bminference.layers.transformer_block import TransformerBlockGPT

device = torch.device("cuda:0")

def build_parameter(name, parameter : Parameter, ckpt):
    tensor = ckpt[name]
    tp = parameter.dtype
    v = tensor.cpu().numpy().astype(tp)
    shape = v.shape
    if np.issubdtype(parameter.dtype, np.integer):
        raise TypeError("%s has low precision" % name)
    parameter.put_data(shape, v.tobytes(), tp)

def scale_build_parameter(name, value : Parameter, scale : Parameter, axis, ckpt):
    tensor = ckpt[name].to(device)
    # v = tensor.numpy().astype(np.float16)
    scale_v = torch.max(tensor.abs(), dim=axis, keepdim=True)[0] / 127

    qv = torch.round(tensor / scale_v).type(torch.int8)
    scale_v = scale_v.type(torch.float16)

    qv = qv.cpu().numpy().astype(np.int8)
    scale_v = scale_v.cpu().numpy().astype(np.float16)

    value.put_data(qv.shape, qv.tobytes(), qv.dtype)
    scale.put_data(scale_v.shape, scale_v.tobytes(), scale_v.dtype)

def split(x, s):
    sizes = []
    for it in x.size():
        sizes.append(it)
    assert sizes[0] % s == 0
    sizes = [s, sizes[0] // s ] + sizes[1:]
    return x.reshape(*sizes)

def build_block(ckpt, model : TransformerBlockGPT, prefix):
    build_parameter(f"{prefix}.input_layernorm.weight", model.layer_nrom_before_self_attn.weight, ckpt)
    build_parameter(f"{prefix}.input_layernorm.bias", model.layer_nrom_before_self_attn.bias, ckpt)

    ckpt[f"{prefix}.attention.query_key_value.weight"] = split(ckpt[f"{prefix}.attention.query_key_value.weight"], 3)
    ckpt[f"{prefix}.attention.query_key_value.bias"] = split(ckpt[f"{prefix}.attention.query_key_value.bias"], 3)
    scale_build_parameter(f"{prefix}.attention.query_key_value.weight", model.self_attention.w_project_qkv, model.self_attention.w_project_qkv_scale, -1, ckpt)
    build_parameter(f"{prefix}.attention.query_key_value.bias", model.self_attention.w_project_bias, ckpt)
    scale_build_parameter(f"{prefix}.attention.dense.weight", model.self_attention.w_out, model.self_attention.w_out_scale, 1, ckpt)
    build_parameter(f"{prefix}.attention.dense.bias", model.self_attention.w_out_bias, ckpt)

    build_parameter(f"{prefix}.post_attention_layernorm.weight", model.layer_nrom_before_ff.weight, ckpt)
    build_parameter(f"{prefix}.post_attention_layernorm.bias", model.layer_nrom_before_ff.bias, ckpt)
    scale_build_parameter(f"{prefix}.mlp.dense_h_to_4h.weight", model.dense_gelu_dense.wi.weight, model.dense_gelu_dense.wi.weight_scale, 1, ckpt)
    build_parameter(f"{prefix}.mlp.dense_h_to_4h.bias", model.dense_gelu_dense.wi.weight_bias, ckpt)
    scale_build_parameter(f"{prefix}.mlp.dense_4h_to_h.weight", model.dense_gelu_dense.wo.weight, model.dense_gelu_dense.wo.weight_scale, 1, ckpt)
    build_parameter(f"{prefix}.mlp.dense_4h_to_h.bias", model.dense_gelu_dense.wo.weight_bias, ckpt)

def build_layers(ckpt, model : CPM1):
    for i in range(model.num_layers):
        build_block(ckpt, model.layers[i], f"transformer.layers.{i}")


def build_model(ckpt, model : CPM1):
    build_parameter("word_embeddings.weight", model.input_embedding.weight, ckpt) 
    build_parameter("position_embeddings.weight", model.position_embedding.weight, ckpt)
    build_parameter("transformer.final_layernorm.weight", model.encoder_final_layer_nrom.weight, ckpt)
    build_parameter("transformer.final_layernorm.bias", model.encoder_final_layer_nrom.bias, ckpt)

    build_layers(ckpt, model)

def main():
    model = torch.load("merge.pt")
    config = CPM1Configuration()
    config.MODEL_NAME = None
    cpm1 = CPM1(config=config)
    build_model(model, cpm1)
    cpm1.dump(open("checkpoint.pt", "wb"))

if __name__ == "__main__":
    main()
    