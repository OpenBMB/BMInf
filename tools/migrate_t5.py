from bminference.models.eva2 import EVA2Configuration
import torch
import numpy as np
from bminference.models.cpm2 import CPM2, CPM2Configuration
from bminference.parameter import Parameter
from bminference.layers.transformer_block import TransformerBlockDecoder

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


def build_block(ckpt, model : TransformerBlockDecoder, prefix, has_cross_attn):
    build_parameter(f"{prefix}.self_attn.layer_norm.weight", model.layer_nrom_before_self_attn.weight, ckpt)
    scale_build_parameter(f"{prefix}.self_attn.self_attn.project.weight", model.self_attention.w_project_qkv, model.self_attention.w_project_qkv_scale, 1, ckpt)
    scale_build_parameter(f"{prefix}.self_attn.self_attn.dense.weight", model.self_attention.w_out, model.self_attention.w_out_scale, 1, ckpt)


    if has_cross_attn:
        build_parameter(f"{prefix}.cross_attn.layer_norm.weight", model.layer_nrom_before_cross_attn.weight, ckpt)
        scale_build_parameter(f"{prefix}.cross_attn.cross_attn.project_q.weight", model.cross_attention.w_project_q, model.cross_attention.w_project_q_scale, 1, ckpt)
        scale_build_parameter(f"{prefix}.cross_attn.cross_attn.dense.weight", model.cross_attention.w_out, model.cross_attention.w_out_scale, 1, ckpt)
    
    build_parameter(f"{prefix}.ff.layer_norm.weight", model.layer_nrom_before_ff.weight, ckpt)
    scale_build_parameter(f"{prefix}.ff.dense_relu_dense.wi_0.weight", model.dense_gelu_dense.wi_0.weight, model.dense_gelu_dense.wi_0.weight_scale, 1, ckpt)
    scale_build_parameter(f"{prefix}.ff.dense_relu_dense.wi_1.weight", model.dense_gelu_dense.wi_1.weight, model.dense_gelu_dense.wi_1.weight_scale, 1, ckpt)
    scale_build_parameter(f"{prefix}.ff.dense_relu_dense.wo.weight", model.dense_gelu_dense.wo.weight, model.dense_gelu_dense.wo.weight_scale, 1, ckpt)

def build_encoder(ckpt, model : CPM2):
    for i in range(24):
        build_block(ckpt, model.encoder[i], f"encoder.blocks.{i}", False)

def build_decoder(ckpt, model : CPM2):
    for i in range(24):
        build_block(ckpt, model.decoder[i], f"decoder.blocks.{i}", True) 

def build_model(ckpt, model : CPM2):
    build_parameter("word_embeds.weight", model.input_embedding.weight, ckpt) 
    build_parameter("encoder.blocks.0.self_attn.self_attn.relative_attention_bias.weight", model.encoder_position_bias.embedding.weight, ckpt)
    build_parameter("encoder.final_layernorm.weight", model.encoder_final_layer_nrom.weight, ckpt)
    build_parameter("decoder.blocks.0.self_attn.self_attn.relative_attention_bias.weight", model.decoder_position_bias.embedding.weight, ckpt)
    build_parameter("decoder.final_layernorm.weight", model.decoder_final_layer_nrom.weight, ckpt)
    build_parameter("lm_head.weight", model.lm_head.weight, ckpt)
    
    ret = []
    for i in range(24):
        ret.append(ckpt[f"decoder.blocks.{i}.cross_attn.cross_attn.project_kv.weight"].cpu().numpy())
    v = np.stack(ret)
    ckpt["encoder_kv.weight"] = torch.from_numpy(v)
    scale_build_parameter("encoder_kv.weight", model.encoder_kv.w_project_kv, model.encoder_kv.w_project_kv_scale, -1, ckpt)
    build_encoder(ckpt, model)
    build_decoder(ckpt, model)
    
def main():
    model = torch.load("merge.pt")
    config = CPM2Configuration()
    config.MODEL_NAME = None
    cpm2 = CPM2(config=config)
    build_model(model, cpm2)
    cpm2.dump(open("checkpoint.pt", "wb"))

if __name__ == "__main__":
    main()
    