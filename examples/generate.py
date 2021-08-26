import numpy as np
import bigmodels
from tqdm import tqdm
import logging

def generate_span(model : bigmodels.models.CPM2, sentence):
    idx = [model.text_to_id(sentence) + [ model.get_token_id("<s_0>") ]]
    input_length = [len(idx[0])]

    hidden_state = model.encode(np.array(idx, dtype=np.int64), input_length)
    begin_token = model.get_token_id("<s_0>")
    end_token = model.get_token_id("<s_1>")
    record = False

    first_token = True
    for token_id in model.decode( hidden_state, input_length, sampler="greedy"):
        token_id = token_id[0]
        if first_token:
            if token_id != begin_token:
                raise RuntimeError("Decoder error")
            first_token = False
        if token_id == begin_token:
            record = True
        elif token_id == end_token:
            break
        elif record:
            yield model.get_id_token(token_id) 

def generate(model : bigmodels.models.CPM2, sentence):
    with tqdm() as progress_bar:
        progress_bar.write(sentence)
        while True:
            for token in generate_span(model, sentence):
                sentence += token
                progress_bar.write(sentence)
                progress_bar.update(1)
            
                if token.find( "<eod>") != -1:
                    break

input_text = """7月底，四川此轮疫情中与天府机场相关的本土病例曾引发广泛关注。媒体"""

def main():
    print("Loading model")
    cpm2 = bigmodels.models.CPM2()
    print("Start")
    generate(cpm2, input_text)

if __name__ == "__main__":
    main()
