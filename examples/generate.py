import numpy as np
import bigmodels
from tqdm import tqdm

def generate(model : bigmodels.models.CPM2, sentence):
    with tqdm() as progress_bar:
        progress_bar.write(sentence)
        while True:
            value = model.generate(sentence + "<span>", 
                temperature=1.1, 
                top_p=0.9, 
                top_n=20, 
                frequency_penalty=5,
                presence_penalty=0
            )[0]["text"]
            sentence += value
            progress_bar.write(sentence)
            progress_bar.update(1)
            if value.find("<eod>") != -1:
                break
            

input_text = """天空是蔚蓝色，窗外有"""

def main():
    print("Loading model")
    cpm2 = bigmodels.models.CPM2()
    print("Start")
    generate(cpm2, input_text)

if __name__ == "__main__":
    main()
