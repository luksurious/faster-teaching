import numpy as np
from numpy.random import Generator


class RandomWrapper:
    rg: Generator

    def __init__(self):
        self.seed()

    def seed(self, seed=None):
        self.rg = np.random.default_rng(seed=seed)


rand_ng = RandomWrapper()
