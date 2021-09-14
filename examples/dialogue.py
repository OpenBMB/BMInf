import bminference

def main():
    print("Loading model")
    config = bminference.models.EVA2Configuration()
    config.MODEL_NAME = "file:///root/.cache/bigmodels/eva2/"
    eva2 = bminference.models.EVA2(config=config)
    
    context = []
    while True:
        you = input("你：")
        context.append(you)
        computer = eva2.dialogue(context)
        context.append(computer)
        print("EVA2：", computer)

if __name__ == "__main__":
    main()