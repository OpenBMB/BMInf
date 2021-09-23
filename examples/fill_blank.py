import bminf
from tqdm import tqdm

TOKEN_SPAN = "<span>"

input_1 = "北京环球度假区相关负责人介绍，北京环球影城指定单日门票将采用<span>制度，即推出淡季日、平季日、旺季日和特定日门票。<span>价格为418元，<span>价格为528元，<span>价格为638元，<span>价格为<span>元。北京环球度假区将提供90天滚动价格日历，以方便游客提前规划行程。"

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

input_2 = "天空是蔚蓝色，窗外有"

def generate(model : bminf.models.CPM2, text):
    with tqdm() as progress_bar:
        progress_bar.write(text)
        while True:
            value = model.generate(
                input_sentence = text + "<span>", 
                max_tokens=20,
                top_n=5,
                top_p=None,
                temperature=0.85,
                frequency_penalty=0,
                presence_penalty=0
            )[0]["text"]
            text += value
            progress_bar.write(text)
            progress_bar.update(1)
            if value.find("<eod>") != -1:
                break

def main():
    print("Loading model")
    cpm2 = bminf.models.CPM2()
    print("Start")
    fill_blank(cpm2, input_1)
    generate(cpm2, input_2)

if __name__ == "__main__":
    main()