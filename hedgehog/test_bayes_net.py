import copy
import importlib
import inspect
import math
import random

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
    assert math.isclose(fjd.sum(), 1)
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


def check_query(bn):
    """Checks that the query function works for every algorithm."""

    fjd = bn.full_joint_dist()
    event = dict(zip(fjd.index.names, fjd.index[0]))
    query = random.choice(list(event))
    del event[query]

    for algorithm in ('exact', 'gibbs', 'likelihood', 'rejection'):
        bn.query(query, event=event, algorithm=algorithm)


def naive():
    bn = hh.BayesNet('A', 'B', 'C')
    bn.P['A'] = pd.Series({True: .1, False: .9})
    bn.P['B'] = pd.Series({True: .3, False: .7})
    bn.P['C'] = pd.Series({True: .5, False: .5})
    bn.prepare()
    return bn


@pytest.mark.parametrize('bn, check', [
    pytest.param(
        example(),
        check,
        id=f"{example.__name__}:{check.__name__}"
    )
    for example in (
        *dict(inspect.getmembers(
            importlib.import_module('hedgehog.examples'),
            inspect.isfunction)
        ).values(),
        naive
    )
    for check in (
        check_partial_fit,
        check_sample_many,
        check_sample_one,
        check_full_joint_dist,
        check_Ps,
        check_query
    )
])
def test(bn, check):
    check(bn)


def test_indep_vars():
    """This doctest checks that querying with independent variables works as expected.

    >>> bn = hh.BayesNet()
    >>> bn.P['A'] = pd.Series({1: .2, 2: .3, 3: .5})
    >>> bn.P['B'] = pd.Series({1: .4, 2: .2, 3: .4})
    >>> bn.prepare()

    >>> bn.full_joint_dist()
    A  B
    1  1    0.08
       2    0.04
       3    0.08
    2  1    0.12
       2    0.06
       3    0.12
    3  1    0.20
       2    0.10
       3    0.20
    Name: P(A, B), dtype: float64

    >>> bn.query('A', event={'B': 1})
    A
    1    0.2
    2    0.3
    3    0.5
    Name: P(A), dtype: float64

    >>> bn.query('A', event={'B': 2})
    A
    1    0.2
    2    0.3
    3    0.5
    Name: P(A), dtype: float64

    >>> bn.query('A', event={'B': 3})
    A
    1    0.2
    2    0.3
    3    0.5
    Name: P(A), dtype: float64

    """
