import time
import bigmodels
import numpy as np
import cupy
from tqdm import tqdm
import argparse
from bigmodels.arch.t5.context import T5InferenceContext

def get_args():
    parser = argparse.ArgumentParser(description='CPM2 encoder benchmark')
    parser.add_argument('--device', type=int, default=0, help="Device index")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size")
    parser.add_argument('--encode-length', type=int, default=256, help="Encoder input length")
    parser.add_argument('--cases', type=int, default=20, help="Num test cases")
    return parser.parse_args()

def test(model : bigmodels.models.CPM2, ctx : T5InferenceContext):
    model.init_decoder_context( ctx )

def main():
    args = get_args()

    print("Loading model")
    device_idx = args.device
    cpm2 = bigmodels.models.CPM2(device=device_idx)
    device = cupy.cuda.Device(device_idx)

    with device:
        ctx = T5InferenceContext( cupy.random.randn(args.batch_size, cpm2.dim_model, args.encode_length, dtype=cupy.float32).astype(cupy.float16), [args.encode_length] * args.batch_size )
    test(cpm2, ctx)
    device.synchronize()
    del ctx
        

    d = []
    for i in tqdm(range(args.cases)):
        with device:
            ctx = T5InferenceContext( cupy.random.randn(args.batch_size, cpm2.dim_model, args.encode_length, dtype=cupy.float32).astype(cupy.float16), [args.encode_length] * args.batch_size )
        st = time.perf_counter()
        test(cpm2, ctx)
        device.synchronize()
        st = time.perf_counter() - st
        d.append(st)
        del ctx
    
    print(f"""
Average {sum(d) / len(d)}s
Max {max(d)}s
Min {min(d)}s

Throughput { (args.batch_size * args.encode_length) * len(d) / sum(d) } tokens/s
""")

if __name__ == "__main__":
    main()