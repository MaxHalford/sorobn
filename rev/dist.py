"""Probability distribution used within each node."""
import collections
import random

import numpy as np


__all__ = ['Bernoulli', 'Multinomial']


class Bernoulli:

    def __init__(self, p, seed=None):
        self.p = p
        self._rng = random.Random(seed)

    def P(self, value):
        return self.p if bool(value) else (1 - self.p)

    def sample(self):
        return self._rng.random() < self.p

    def fit(self, samples, weights=None):

        if weights is None:
            weights = np.ones_like(samples, dtype=float)
        else:
            weights = np.asarray(weights)
        weights /= weights.sum()

        self.p = np.dot(samples, weights)
        return self

    def __repr__(self):
        return f'Bernoulli({self.p:.3f})'


class Multinomial(collections.UserDict):

    def P(self, value):
        return self.get(value, 0)

    def sample(self):
        return np.random.choice(list(self), p=list(self.values()))

    def fit(self, samples):
        for k in list(self):
            del self[k]
        super().update(samples.value_counts(normalize=True).to_dict())
        return self
