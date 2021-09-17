import time
import bminf
import numpy as np
import cupy
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='CPM2 encoder benchmark')
    parser.add_argument('--device', type=int, default=0, help="Device index")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size")
    parser.add_argument('--input-length', type=int, default=256, help="Input length")
    parser.add_argument('--cases', type=int, default=20, help="Num test cases")
    return parser.parse_args()

def test(model : bminf.models.CPM2, batch_size, input_length):
    input_idx = np.arange(input_length)[np.newaxis].repeat(batch_size, axis=0)
    input_len = [input_length] * batch_size
    model.encode( input_idx, input_len )

def main():
    args = get_args()

    print("Loading model")
    device_idx = args.device
    cpm2 = bminf.models.CPM2(device=device_idx)
    device = cupy.cuda.Device(device_idx)

    test(cpm2, args.batch_size, args.input_length)

    d = []
    for i in tqdm(range(args.cases)):
        device.synchronize()
        st = time.perf_counter()
        test(cpm2, args.batch_size, args.input_length)
        device.synchronize()
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