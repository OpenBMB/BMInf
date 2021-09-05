import time
import bminference
import numpy as np
import cupy
from tqdm import tqdm
import argparse
from bminference.arch.t5.context import T5InferenceContext

def get_args():
    parser = argparse.ArgumentParser(description='CPM2 encoder benchmark')
    parser.add_argument('--device', type=int, default=0, help="Device index")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size")
    parser.add_argument('--encode-length', type=int, default=256, help="Encoder input length")
    parser.add_argument('--decode-length', type=int, default=255, help="Decoder output length ( < 256)")
    parser.add_argument('--cases', type=int, default=10, help="Num test cases")
    return parser.parse_args()

def test(model : bminference.models.CPM2, ctx : T5InferenceContext, batch_size, decode_length):
    for _ in range(decode_length):
        model.decode_step( ctx, [0] * batch_size)

def main():
    args = get_args()

    print("Loading model")
    device_idx = args.device
    cpm2 = bminference.models.CPM2(device=device_idx)
    device = cupy.cuda.Device(device_idx)

    with device:
        ctx = T5InferenceContext( cupy.random.randn(args.batch_size, cpm2.dim_model, args.encode_length, dtype=cupy.float32).astype(cupy.float16), [args.encode_length] * args.batch_size )
        cpm2.init_decoder_context( ctx )
        device.synchronize()
        
    test(cpm2, ctx, args.batch_size, args.decode_length)

    d = []
    for i in tqdm(range(args.cases)):
        with device:
            ctx = T5InferenceContext( cupy.random.randn(args.batch_size, cpm2.dim_model, args.encode_length, dtype=cupy.float32).astype(cupy.float16), [args.encode_length] * args.batch_size )
            cpm2.init_decoder_context( ctx )
            device.synchronize()
        st = time.perf_counter()
        test(cpm2, ctx, args.batch_size, args.decode_length)
        device.synchronize()
        st = time.perf_counter() - st
        d.append(st)
    
    print(f"""
Average {sum(d) / len(d)}s
Max {max(d)}s
Min {min(d)}s

Throughput { (args.batch_size * args.decode_length) * len(d) / sum(d) } tokens/s
""")

if __name__ == "__main__":
    main()