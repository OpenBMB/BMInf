import torch
import bminf.torch as bt

def main():
    model = bt.CPM1(
        memory_limit= 8 * 1024 * 1024 * 1024, # 8GB
    )
    ids = torch.LongTensor([[1, 2, 3, 4]])
    attention_mask = torch.FloatTensor([[1, 1, 1, 1]])

    output = model(
        ids,
        attention_mask,
    )
    print("Logits: ", output.logits)

if __name__ == "__main__":
    main()