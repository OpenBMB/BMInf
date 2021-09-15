import bminference

def main():
    print("Loading model")
    eva2 = bminference.models.EVA2()
    
    context = []
    while True:
        you = input("你：")
        context.append(you)
        computer = eva2.dialogue(context)
        context.append(computer)
        print("EVA2：", computer)

if __name__ == "__main__":
    main()