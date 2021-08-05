import torch
import time

def main():
    torch.randn(3).to("cuda:0")

    v = torch.randn(1024 * 1024 * 1024).type(torch.int8)
    lst = []
    for i in range(10):
        st = time.perf_counter()
        lst.append(v.to("cuda:0"))
        print(time.perf_counter() - st)

if __name__ == "__main__":
    main()