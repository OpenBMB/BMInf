import bminference

TOKEN_SPAN = "<span>"
input_text = "北京环球度假区相关负责人介绍，北京环球影城指定单日门票将采用<span>制度，即推出淡季日、平季日、旺季日和特定日门票。<span>价格为418元，<span>价格为528元，<span>价格为638元，<span>价格为<span>元。北京环球度假区将提供90天滚动价格日历，以方便游客提前规划行程。"
def fill_blank(cpm2 : bminference.models.CPM2, text : str):
    print("Input: ", text.replace(TOKEN_SPAN,  "\033[4m____\033[0m") )
    for result in cpm2.generate(input_text, 
            top_p=1.0,
            top_n=10, 
            temperature=0.9,
            frequency_penalty=0,
            presence_penalty=0
        ):
        value = result["text"]
        text = text.replace(TOKEN_SPAN, "\033[0;32m" + value + "\033[0m", 1)
    print("Output:", text)

def main():
    print("Loading model")
    config = bminference.models.CPM2Configuration()
    config.MODEL_NAME = "file:///root/.cache/bigmodels/cpm2/"
    cpm2 = bminference.models.CPM2(config=config)
    print("Start")
    fill_blank(cpm2, input_text)

if __name__ == "__main__":
    main()