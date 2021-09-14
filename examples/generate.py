import bminference
from tqdm import tqdm

def generate(model : bminference.models.CPM2, sentence):
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
    config = bminference.models.CPM2Configuration()
    config.MODEL_NAME = "file:///root/.cache/bigmodels/cpm2/"
    cpm2 = bminference.models.CPM2(config=config)
    print("Start")
    generate(cpm2, input_text)

if __name__ == "__main__":
    main()
