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

    for token_id in model.decode( hidden_state, input_length ):
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

def main():
    print("Loading model")
    cpm2 = bigmodels.models.CPM2()
    print("Start")
    generate(cpm2, "习近平总书记23日下午来到塞罕坝机械林场尚海纪念林。纪念林位于原马蹄坑造林会战区，是塞罕坝精神发源地、百万亩林海起源地。习近平同林场职工代表亲切交流，他强调，你们做的事非常有示范意义，对全国生态文明建设具有激励作用和深远影响。")

if __name__ == "__main__":
    main()