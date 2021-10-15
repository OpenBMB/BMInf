import bminf
from tqdm import tqdm

TOKEN_SPAN = "<span>"

input_fill_blank_text = "北京环球度假区相关负责人介绍，北京环球影城指定单日门票将采用<span>制度，即推出淡季日、平季日、旺季日和特定日门票。<span>价格为418元，<span>价格为528元，<span>价格为638元，<span>价格为<span>元。北京环球度假区将提供90天滚动价格日历，以方便游客提前规划行程。"

def fill_blank(cpm2 : bminf.models.CPM2, text):
    print("Input: ", text.replace(TOKEN_SPAN,  "\033[4m____\033[0m") )
    for result in cpm2.fill_blank(text, 
            top_p=1.0,
            top_n=5, 
            temperature=0.5,
            frequency_penalty=0,
            presence_penalty=0
        ):
        value = result["text"]
        text = text.replace(TOKEN_SPAN, "\033[0;32m" + value + "\033[0m", 1)
    print("Output:", text)

input_generate_text = """天空是蔚蓝色，窗外有"""

def generate(model : bminf.models.CPM2, text):
    print("Input: ", text)
    for i in range(3):
        value = model.generate(
            input_sentence = text[-32:] + "<span>", 
            max_tokens=32,
            top_n=5,
            top_p=None,
            temperature=0.85,
            frequency_penalty=0,
            presence_penalty=0,
            stop_words = ['，', '。', '！']
        )[0]["text"]
        text += value
        if text.find("<eod>") != -1:
            break
    print("Output:", text)

def main():
    print("Loading model")
    cpm2_1 = bminf.models.CPM2()
    fill_blank(cpm2_1, input_fill_blank_text)
    generate(cpm2_1, input_generate_text)

if __name__ == "__main__":
    main()