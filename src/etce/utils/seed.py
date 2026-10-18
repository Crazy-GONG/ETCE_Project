import random


def set_seed(seed):
    """Only stdlib random is used; temporal splitting never samples."""
    random.seed(seed)
