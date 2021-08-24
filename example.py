import sys
import numpy as np
import bigmodels
import time
from tqdm import tqdm

def generate_span(model : bigmodels.models.CPM2, sentence):
    idx = [model.text_to_id(sentence) + [ model.get_token_id("<s_0>") ]]
    input_length = [len(idx[0])]

    hidden_state = model.encode(np.array(idx, dtype=np.int64), input_length)
    
    tokens = []
    begin_token = model.get_token_id("<s_0>")
    end_token = model.get_token_id("<s_1>")
    record = False

    for token_id in model.decode( hidden_state, input_length, sampler="greedy"):
        token_id = token_id[0]

        if token_id == begin_token:
            record = True
        elif token_id == end_token:
            break
        elif record:
            tokens.append(token_id)
        if len(tokens) > 10:
            break
    return model.id_to_text(tokens)

def generate(model : bigmodels.models.CPM2, sentence):
    with tqdm() as progress_bar:
        progress_bar.write(sentence)
        while True:
            span = generate_span(model, sentence)
            sentence += span
            progress_bar.write(sentence)
            progress_bar.update(1)
            if span.find( "<eod>") != -1:
                break

input_text = """问题：游戏《绝地求生》中有哪些鲜为人知的技巧？\n描述：\n"""

def main():
    print("Loading model")
    cpm2 = bigmodels.models.CPM2()
    print("Start")
    generate(cpm2, input_text)

if __name__ == "__main__":
    main()