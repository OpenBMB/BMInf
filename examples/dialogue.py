import bminf

def main():
    print("Loading model")
    eva = bminf.models.EVA()
    
    context = []
    while True:
        you = input("你：")
        context.append(you)
        if you == "你好":
            computer = "你好，我是EVA"
        else:
            computer, _ = eva.dialogue(context)
        context.append(computer)
        print("EVA：", computer)

if __name__ == "__main__":
    main()