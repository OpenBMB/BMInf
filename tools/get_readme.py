from pathlib import Path

def get_readme():
    ret = ""
    with open(Path(__file__).parent.parent.joinpath("README.md"), encoding="utf-8") as frd:
        ret = frd.read()
    return ret