import bminference

input_text = """中国的首都是北京
日本的首都是东京
法国的首都是巴黎
德国的首都是柏林"""

def main():
    config = bminference.models.CPM1Configuration()
    config.MODEL_NAME = "file:///root/.cache/bigmodels/cpm1"
    cpm1 = bminference.models.CPM1(config=config)
    print(cpm1.generate(input_text, top_n=1))

if __name__ == "__main__":
    main()