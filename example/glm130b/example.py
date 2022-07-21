from model import GLM130B
import torch
import bminf

def main():
    m = bminf.wrapper(GLM130B().cuda(), quantization=False)
    print(list(m.state_dict().keys()))

    x = torch.LongTensor([[1, 2, 3, 4]]).cuda()
    position = torch.LongTensor([[0, 1, 2, 3]]).cuda()
    mask = torch.BoolTensor([[
        [True, True, True, True],
        [True, True, True, True],
        [True, True, True, True],
        [True, True, True, True],
    ]]).cuda()
    print(m(x, position, mask))

if __name__ == "__main__":
    main()
