import numpy as np


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def partition(pred, prefix, iterable):
    "Use a predicate to partition entries into false entries and true entries"
    trues = []
    falses = []
    [
        (trues.append((key, val)) if pred(key, prefix) else falses.append((key, val)))
        for (key, val) in iterable
    ]
    return trues, falses


def startswith(s, *args):
    return s.startswith(args)
