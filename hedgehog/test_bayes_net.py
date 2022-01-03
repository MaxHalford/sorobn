import copy
import importlib
import inspect
import itertools
import math
import random

import numpy as np
import pandas as pd
import pytest

import hedgehog as hh


def check_partial_fit(bn):
    """Checks that partial_fit produces the same result as fit."""

    bn_partial = copy.deepcopy(bn)

    # Fit the parameters of the first BN in one go
    samples = bn.sample(500)
    bn.fit(samples)

    # Fit the parameters of the second BN incrementally
    bn_partial.P = {}
    for chunk in np.array_split(samples, 5):
        bn_partial.partial_fit(chunk)

    # Check that the obtained parameters are identical
    for node in bn.P:
        pd.testing.assert_series_equal(bn.P[node], bn_partial.P[node])


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

def test_cpt_with_index_names():
    """

    From https://github.com/MaxHalford/hedgehog/issues/19

    """

    edges = pd.DataFrame({"parent": ["A", "B"], "child": "C"})
    bn = hh.BayesNet(*edges.itertuples(index=False, name=None))

    bn.P['A'] = pd.Series({True: 0.7, False: 0.3})
    bn.P['B'] = pd.Series({True: 0.4, False: 0.6})

    PC = pd.DataFrame(
        {
            "B": [True, True, True, True, False, False, False, False],
            "A": [True, True, False, False, True, True, False, False],
            "C": [
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
            ],
            "p": [1, 0, 0, 1, 0.5, 0.5, 0.001, 0.999],
        }
    )
    bn.P["C"] = PC.set_index(["B", "A", "C"])["p"]
    bn.prepare()

    pd.testing.assert_series_equal(
        bn.query("C", event={"B": False, "A": True}),
        pd.Series([0.5, 0.5], name="P(C)", index=pd.Index([False, True], name="C"))
    )

def test_predict_proba_order_doesnt_matter():

    bn = hh.examples.alarm()
    event = {
        'Alarm': False,
        'Burglary': False,
        'Earthquake': True,
        'John calls': False,
        'Mary calls': False
    }

    for order in itertools.permutations(event.keys()):
        ordered_event = {var: event[var] for var in order}
        assert math.isclose(bn.predict_proba(ordered_event), bn.predict_proba(event))
