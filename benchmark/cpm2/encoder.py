import time

from cpm_kernels.library import cudart
import bminf
import numpy as np
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='CPM2 encoder benchmark')
    parser.add_argument('--device', type=int, default=0, help="Device index")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size")
    parser.add_argument('--input-length', type=int, default=256, help="Input length")
    parser.add_argument('--cases', type=int, default=20, help="Num test cases")
    return parser.parse_args()

def test(model : bminf.models.CPM2, input_tensor, mask):
    model._model.encode(
        model._ctx,
        input_tensor,
        mask
    )

def main():
    args = get_args()

    print("Loading model")
    cpm2 = bminf.models.CPM2(device_idx=args.device)

    ctx = cpm2._ctx
    encoder_input = ctx.allocate((args.batch_size, cpm2._config.DIM_MODEL, args.input_length), np.half)
    encoder_mask = np.ones((args.batch_size, args.input_length), np.bool8)
    test(cpm2, encoder_input, encoder_mask)

    d = []
    for i in tqdm(range(args.cases)):
        cudart.cudaStreamSynchronize(ctx.current_stream)
        st = time.perf_counter()
        test(cpm2, encoder_input, encoder_mask)
        cudart.cudaStreamSynchronize(ctx.current_stream)
        st = time.perf_counter() - st
        d.append(st)
    
    print(f"""
Average {sum(d) / len(d)}s
Max {max(d)}s
Min {min(d)}s

Throughput { (args.batch_size * args.input_length) * len(d) / sum(d) } tokens/s
""")

if __name__ == "__main__":
    main()