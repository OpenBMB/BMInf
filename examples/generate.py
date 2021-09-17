import bminference
from tqdm import tqdm

def generate(model : bminference.models.CPM1, sentence):
    with tqdm() as progress_bar:
        progress_bar.write(sentence)
        while True:
            result = model.generate(
                sentence, 
                max_tokens=8,
                top_n=5,
                top_p=None,
                temperature=0.85,
                frequency_penalty=0,
                presence_penalty=0
            )
            sentence += result
            progress_bar.write(sentence)
            progress_bar.update(1)
            if result.find("<eod>") != -1:
                break
            

input_text = """天空是蔚蓝色，窗外有"""

def main():
    print("Loading model")
    cpm2 = bminference.models.CPM1()
    print("Start")
    generate(cpm2, input_text)

if __name__ == "__main__":
    main()
