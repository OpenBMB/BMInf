import bminf
from tqdm import tqdm

def generate(model : bminf.models.CPM1, sentence):
    with tqdm() as progress_bar:
        progress_bar.write(sentence)
        stoped = False
        while not stoped:
            result, stoped = model.generate(
                sentence, 
                max_tokens=8,
                top_n=5,
                top_p=None,
                temperature=0.85,
                frequency_penalty=0,
                presence_penalty=1
            )
            sentence += result
            progress_bar.write("=" * 20 + "\n" + sentence)
            progress_bar.update(1)
            

input_text = """天空是蔚蓝色，窗外有"""

def main():
    print("Loading model")
    cpm1 = bminf.models.CPM1()
    print("Start")
    generate(cpm1, input_text)

if __name__ == "__main__":
    main()
