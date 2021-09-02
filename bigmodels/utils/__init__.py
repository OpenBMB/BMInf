from .sampler import GenerateSampler
from .jieba import jieba
def round_up(x, d):
    return (x + d - 1) // d * d