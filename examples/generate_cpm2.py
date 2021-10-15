import bminf
import sys

input_text = """天空是蔚蓝色，窗外有"""

def generate(model : bminf.models.CPM2, text):
    print("Input: ", text)
    sys.stdout.write("Output: %s" % text)
    stoped = False
    while not stoped:
        value, stoped = model.generate(
            input_sentence = text[-32:], 
            max_tokens=32,
            top_n=5,
            top_p=None,
            temperature=0.85,
            frequency_penalty=0,
            presence_penalty=0,
        )
        text += value
        sys.stdout.write(value)
        sys.stdout.flush()
    sys.stdout.write("\n")
    

def main():
    print("Loading model")
    cpm2_1 = bminf.models.CPM2()
    generate(cpm2_1, input_text)

if __name__ == "__main__":
    main()