import sys
sys.path.insert(0, "/root/toolkit")
from typing import Union
import torch
import numpy as np
from bminf.core import Parameter
from bminf.arch import T5Model, T5Configuration
from bminf.layers import EncoderBlock, DecoderBlockWithCrossAttention
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

def build_block(ckpt, model : Union[EncoderBlock, DecoderBlockWithCrossAttention], prefix, has_cross_attn):
    build_parameter(ckpt[f"{prefix}.self_attn.layer_norm.weight"], model.ln_attn.weight)

    split_attn = split(ckpt[f"{prefix}.self_attn.self_attn.project.weight"], 3)
    scale_build_parameter(split_attn[0], model.self_attn.project_q.weight, model.self_attn.project_q.scale)
    scale_build_parameter(split_attn[1], model.self_attn.project_k.weight, model.self_attn.project_k.scale)
    scale_build_parameter(split_attn[2], model.self_attn.project_v.weight, model.self_attn.project_v.scale)
    scale_build_parameter(ckpt[f"{prefix}.self_attn.self_attn.dense.weight"], model.self_attn.linear_out.weight, model.self_attn.linear_out.scale)


    if has_cross_attn:
        build_parameter(ckpt[f"{prefix}.cross_attn.layer_norm.weight"], model.ln_cross_attn.weight)
        split_attn = split(ckpt[f"{prefix}.cross_attn.cross_attn.project_kv.weight"], 2)
        cross_attn = [ckpt[f"{prefix}.cross_attn.cross_attn.project_q.weight"], split_attn[0], split_attn[1]]
        scale_build_parameter(cross_attn[0], model.cross_attn.project_q.weight, model.cross_attn.project_q.scale)
        scale_build_parameter(cross_attn[1], model.cross_attn.project_k.weight, model.cross_attn.project_k.scale)
        scale_build_parameter(cross_attn[2], model.cross_attn.project_v.weight, model.cross_attn.project_v.scale)
        scale_build_parameter(ckpt[f"{prefix}.cross_attn.cross_attn.dense.weight"], model.cross_attn.linear_out.weight, model.cross_attn.linear_out.scale)
    
    build_parameter(ckpt[f"{prefix}.ff.layer_norm.weight"], model.ln_ff.weight)
    scale_build_parameter(ckpt[f"{prefix}.ff.dense_relu_dense.wi_0.weight"], model.ff.linear_in.weight, model.ff.linear_in.scale)
    scale_build_parameter(ckpt[f"{prefix}.ff.dense_relu_dense.wi_1.weight"], model.ff.linear_gated.weight, model.ff.linear_gated.scale)
    scale_build_parameter(ckpt[f"{prefix}.ff.dense_relu_dense.wo.weight"], model.ff.linear_out.weight, model.ff.linear_out.scale)

def build_encoder(ckpt, model : T5Model):
    for i in range(24):
        print("Encoder %d" % i)
        build_block(ckpt, model.enc_layers[i], f"encoder.blocks.{i}", False)

def build_decoder(ckpt, model : T5Model):
    for i in range(24):
        print("Decoder %d" % i)
        build_block(ckpt, model.dec_layers[i], f"decoder.blocks.{i}", True) 

def build_model(ckpt, model : T5Model):
    build_parameter(ckpt["word_embeds.weight"], model.input_embedding.weight) 
    build_parameter(ckpt["encoder.blocks.0.self_attn.self_attn.relative_attention_bias.weight"].transpose(0, 1), model.position_bias_enc.weight)
    build_parameter(ckpt["encoder.final_layernorm.weight"], model.ln_enc.weight)
    build_parameter(ckpt["decoder.blocks.0.self_attn.self_attn.relative_attention_bias.weight"].transpose(0, 1), model.position_bias_dec.weight)
    build_parameter(ckpt["decoder.final_layernorm.weight"], model.ln_dec.weight)
    build_parameter(ckpt["lm_head.weight"], model.output_embedding.weight)
    
    build_encoder(ckpt, model)
    build_decoder(ckpt, model)
    
def main():
    model = torch.load("merge.pt")
    config = T5Configuration()
    cpm2 = T5Model(config=config)
    build_model(model, cpm2)
    cpm2.dump(open("checkpoint.pt", "wb"))

if __name__ == "__main__":
    main()
    