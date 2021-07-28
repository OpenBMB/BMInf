from pathlib import Path

def get_requirements():
    ret = []
    with open(Path(__file__).parent.parent.joinpath("requirements.txt"), encoding="utf-8") as freq:
        for line in freq.readlines():
            ret.append( line.strip() )
    return ret