import time
import bminf
import numpy as np
from tqdm import tqdm
import argparse
from cpm_kernels.library import cudart

def get_args():
    parser = argparse.ArgumentParser(description='CPM2 encoder benchmark')
    parser.add_argument('--device', type=int, default=0, help="Device index")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size")
    parser.add_argument('--decode-length', type=int, default=255, help="Decoder output length ( < 256)")
    parser.add_argument('--cases', type=int, default=10, help="Num test cases")
    return parser.parse_args()

def test(
        model : bminf.models.CPM1, 
        decoder_input,
        decode_length,
        buf_k,
        buf_v,
    ):
    ctx = model._ctx
    for i in range(decode_length):
        model._model.step(
            ctx,
            decoder_input,
            buf_k,
            buf_v,
            i
        )

def main():
    args = get_args()

    print("Loading model")
    device_idx = args.device
    cpm1 = bminf.models.CPM1(device_idx=device_idx)
    
    ctx = cpm1._ctx
    decoder_input = ctx.allocate((args.batch_size, cpm1._config.DIM_MODEL), np.half)


    buf_k = cpm1._model.allocate_decode_buffer(ctx, args.batch_size, args.decode_length)
    buf_v = cpm1._model.allocate_decode_buffer(ctx, args.batch_size, args.decode_length)

    test(
        cpm1,
        decoder_input,
        args.decode_length,
        buf_k,
        buf_v
    )

    d = []
    for i in tqdm(range(args.cases)):
        cudart.cudaStreamSynchronize(ctx.current_stream)
        st = time.perf_counter()
        test(
            cpm1,
            decoder_input,
            args.decode_length,
            buf_k,
            buf_v
        )
        cudart.cudaStreamSynchronize(ctx.current_stream)
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