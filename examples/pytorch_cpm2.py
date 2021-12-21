import torch
import bminf.torch as bt

def main():
    model = bt.CPM2(
        memory_limit= 8 * 1024 * 1024 * 1024, # 8GB
    )
    ids = torch.LongTensor([[1, 2, 3, 4]])
    attention_mask = torch.FloatTensor([[1, 1, 1, 1]])
    decoder_input = torch.LongTensor([[123, 124, 125, 126]])
    decoder_attention_mask = torch.FloatTensor([[1, 1, 1, 0]])
    output = model(
        input_ids=ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input,
        decoder_attention_mask=decoder_attention_mask
    )
    print("Logits: ", output.logits)

if __name__ == "__main__":
    main()