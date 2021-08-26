import numpy as np
import bigmodels

TOKEN_BLANK = "<blank>"

def generate_blank(model : bigmodels.models.CPM2, sentence : str):
    st = 0
    idx = []
    num_spans = 0
    while True:
        pos = sentence.find(TOKEN_BLANK, st)
        if pos == -1:
            idx += model.text_to_id(sentence[st:])
            break
        idx += model.text_to_id( sentence[st:pos] )
        idx += [ model.get_token_id("<s_%d>" % num_spans) ]
        num_spans += 1
        st = pos + len(TOKEN_BLANK)
    
    if num_spans == 0:
        return ""
        
    idx = [idx]
    input_length = [len(idx[0])]

    hidden_state = model.encode(np.array(idx, dtype=np.int64), input_length)

    
    blanks = []

    next_span = 0
    
    for token_id in model.decode( hidden_state, input_length):
        token_id = token_id[0]
        if token_id == model.get_token_id("<s_%d>" % next_span):
            next_span += 1
            if next_span > num_spans:
                break
            blanks.append([])
        else:
            blanks[-1].append(token_id)
    return [model.id_to_text(tokens) for tokens in blanks]

def fill_blank(model : bigmodels.models.CPM2, sentence : str):
    print("Input: ", sentence.replace(TOKEN_BLANK,  "\033[4m    \033[0m") )
    result = sentence
    for blank in generate_blank(model, sentence):
        result = result.replace(TOKEN_BLANK, "\033[0;32m" + blank + "\033[0m", 1)
    print("Output:", result)

input_text = """7月底，四川此轮疫情中与天府机场相关的<blank>病例曾引发<blank>关注。"""

def main():
    print("Loading model")
    cpm2 = bigmodels.models.CPM2()
    print("Start")

    fill_blank(cpm2, input_text)
    

if __name__ == "__main__":
    main()