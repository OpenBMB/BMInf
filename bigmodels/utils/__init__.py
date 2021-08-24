from .sampler import greedy_sampler, random_sampler

def round_up(x, d):
    return (x + d - 1) // d * d