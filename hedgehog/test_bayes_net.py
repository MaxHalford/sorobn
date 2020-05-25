import copy
import importlib
import inspect

import numpy as np
import pandas as pd
import pytest

import hedgehog as hh


def check_partial_fit(bn):
    """Checks that partial_fit produces the same result as fit."""

    bn2 = copy.deepcopy(bn)

    samples = bn.sample(500)

    # Fit the parameters of the first parameters in one go
    bn.fit(samples)

    # Fit the parameters of the second BN incrementally
    bn2.P = {}
    bn2._P_sizes = {}
    for chunk in np.array_split(samples, 5):
        bn2.partial_fit(chunk)

    # Check that the obtained parameters are identical
    for node in bn.P:
        pd.testing.assert_series_equal(bn.P[node], bn2.P[node])


def check_sample_many(bn):
    for n in (2, 3, 100):
        sample = bn.sample(n)
        assert len(sample) == n
        assert sorted(sample.columns) == sorted(bn.nodes)


def check_sample_one(bn):
    sample = bn.sample()
    assert isinstance(sample, dict)
    assert sorted(sample.keys()) == sorted(bn.nodes)


def check_full_joint_dist(bn):
    fjd = bn.full_joint_dist()
    assert fjd.sum() == 1
    assert sorted(fjd.index.names) == sorted(bn.nodes)


def check_Ps(bn):

    for child, parents in bn.parents.items():
        P = bn.P[child]
        assert P.index.names[-1] == child
        assert P.index.names[:-1] == parents
        assert P.groupby(parents).sum().eq(1).all()

    for orphan in set(bn.nodes) - set(bn.parents):
        P = bn.P[orphan]
        assert P.index.name == orphan
        assert P.sum() == 1


@pytest.mark.parametrize('bn, check', [
    pytest.param(bn(), check, id=f"{'_'.join(bn.__name__.split('_')[1:])}:{check.__name__}")
    for _, bn in inspect.getmembers(importlib.import_module('hedgehog.examples'), inspect.isfunction)
    for check in (
        check_partial_fit,
        check_sample_many,
        check_sample_one,
        check_full_joint_dist,
        check_Ps
    )
])
def test(bn, check):
    check(bn)
