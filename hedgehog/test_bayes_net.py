import copy

import numpy as np
import pandas as pd

import hedgehog as hh


def test_partial_fit():
    """Checks that partial_fit produces the same result as fit."""

    bn = hh.load_asia()
    bn2 = hh.load_asia()

    samples = bn.sample(1000)

    # Fit the parameters of the first parameters in one go
    bn.fit(samples)

    # Fit the parameters of the second BN incrementally
    bn2.cpts = {}
    bn2._cpt_sizes = {}
    for chunk in np.array_split(samples, 10):
        bn2.partial_fit(chunk)

    # Check that the obtained parameters are identical
    for node in bn.cpts:
        pd.testing.assert_series_equal(bn.cpts[node], bn2.cpts[node])
