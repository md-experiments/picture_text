import itertools
import copy

def flatten_list(res):
    res=copy.deepcopy(res)
    return list(itertools.chain.from_iterable(res))
