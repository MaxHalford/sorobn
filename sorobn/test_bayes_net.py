import copy
import importlib
import inspect
import itertools
import math
import random

import numpy as np
import pandas as pd
import pytest

import sorobn


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
    assert isinstance(sample, pd.Series)
    assert sorted(sample.index) == sorted(bn.nodes)


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


def check_distribution(bn):
    """Checks that the distribution method works for every algorithm."""

    fjd = bn.full_joint_dist()
    given = dict(zip(fjd.index.names, fjd.index[0]))
    variable = random.choice(list(given))
    del given[variable]

    for algorithm in ("exact", "gibbs", "likelihood", "rejection"):
        bn.distribution(variable, given=given, algorithm=algorithm)


def naive():
    bn = sorobn.BayesNet("A", "B", "C")
    bn.P["A"] = pd.Series({True: 0.1, False: 0.9})
    bn.P["B"] = pd.Series({True: 0.3, False: 0.7})
    bn.P["C"] = pd.Series({True: 0.5, False: 0.5})
    bn.prepare()
    return bn


@pytest.mark.parametrize(
    "bn, check",
    [
        pytest.param(example(), check, id=f"{example.__name__}:{check.__name__}")
        for example in (
            *dict(
                inspect.getmembers(
                    importlib.import_module("sorobn.examples"), inspect.isfunction
                )
            ).values(),
            naive,
        )
        for check in (
            check_partial_fit,
            check_sample_many,
            check_sample_one,
            check_full_joint_dist,
            check_Ps,
            check_distribution,
        )
    ],
)
def test(bn, check):
    check(bn)


def test_indep_vars():
    """This doctest checks that querying with independent variables works as expected.

    >>> bn = sorobn.BayesNet()
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

    >>> bn.distribution('A', given={'B': 1})
    A
    1    0.2
    2    0.3
    3    0.5
    Name: P(A), dtype: float64

    >>> bn.distribution('A', given={'B': 2})
    A
    1    0.2
    2    0.3
    3    0.5
    Name: P(A), dtype: float64

    >>> bn.distribution('A', given={'B': 3})
    A
    1    0.2
    2    0.3
    3    0.5
    Name: P(A), dtype: float64

    """


def test_cpt_with_index_names():
    """

    From https://github.com/MaxHalford/sorobn/issues/19

    """

    edges = pd.DataFrame({"parent": ["A", "B"], "child": "C"})
    bn = sorobn.BayesNet(*edges.itertuples(index=False, name=None))

    bn.P["A"] = pd.Series({True: 0.7, False: 0.3})
    bn.P["B"] = pd.Series({True: 0.4, False: 0.6})

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
        bn.distribution("C", given={"B": False, "A": True}),
        pd.Series([0.5, 0.5], name="P(C)", index=pd.Index([False, True], name="C")),
    )


def test_cpt_dataframe():
    """Test that CPTs can be specified as DataFrames with a 'p' column."""

    bn = sorobn.BayesNet(
        ("A", "C"),
        ("B", "C"),
    )

    bn.P["A"] = pd.Series({True: 0.7, False: 0.3})
    bn.P["B"] = pd.Series({True: 0.4, False: 0.6})
    bn.P["C"] = pd.DataFrame(
        {
            "A": [True, True, True, True, False, False, False, False],
            "B": [True, True, False, False, True, True, False, False],
            "C": [True, False, True, False, True, False, True, False],
            "p": [1, 0, 0.5, 0.5, 0.5, 0.5, 0.001, 0.999],
        }
    )
    bn.prepare()

    # Verify the CPT was properly converted to a Series
    P = bn.P["C"]
    assert isinstance(P, pd.Series)
    assert P.index.names == ["A", "B", "C"]
    assert P.groupby(["A", "B"]).sum().eq(1).all()

    pd.testing.assert_series_equal(
        bn.distribution("C", given={"A": True, "B": False}),
        pd.Series([0.5, 0.5], name="P(C)", index=pd.Index([False, True], name="C")),
    )


def test_cpt_dataframe_column_order_doesnt_matter():
    """Test that DataFrame column order doesn't affect the result."""

    bn1 = sorobn.BayesNet(("A", "C"), ("B", "C"))
    bn1.P["A"] = pd.Series({True: 0.7, False: 0.3})
    bn1.P["B"] = pd.Series({True: 0.4, False: 0.6})
    # Columns in order: A, B, C
    bn1.P["C"] = pd.DataFrame(
        {
            "A": [True, True, False, False],
            "B": [True, False, True, False],
            "C": [True, True, True, True],
            "p": [0.9, 0.8, 0.7, 0.1],
        }
    )
    bn1.prepare()

    bn2 = sorobn.BayesNet(("A", "C"), ("B", "C"))
    bn2.P["A"] = pd.Series({True: 0.7, False: 0.3})
    bn2.P["B"] = pd.Series({True: 0.4, False: 0.6})
    # Columns in different order: B, C, A
    bn2.P["C"] = pd.DataFrame(
        {
            "B": [True, False, True, False],
            "C": [True, True, True, True],
            "A": [True, True, False, False],
            "p": [0.9, 0.8, 0.7, 0.1],
        }
    )
    bn2.prepare()

    pd.testing.assert_series_equal(bn1.P["C"], bn2.P["C"])


def test_cpt_dataframe_missing_p_column():
    """Test that a DataFrame without a 'p' column raises an error."""

    bn = sorobn.BayesNet(("A", "B"))
    bn.P["A"] = pd.Series({True: 0.5, False: 0.5})
    bn.P["B"] = pd.DataFrame(
        {
            "A": [True, True, False, False],
            "B": [True, False, True, False],
            "prob": [0.9, 0.1, 0.4, 0.6],  # wrong column name
        }
    )
    with pytest.raises(ValueError, match="must have a 'p' column"):
        bn.prepare()


def test_cpt_dataframe_wrong_columns():
    """Test that a DataFrame with wrong variable columns raises an error."""

    bn = sorobn.BayesNet(("A", "B"))
    bn.P["A"] = pd.Series({True: 0.5, False: 0.5})
    bn.P["B"] = pd.DataFrame(
        {
            "A": [True, True, False, False],
            "X": [True, False, True, False],  # wrong column name
            "p": [0.9, 0.1, 0.4, 0.6],
        }
    )
    with pytest.raises(ValueError, match="has columns"):
        bn.prepare()


def test_cpt_dataframe_with_string_values():
    """Test DataFrame CPT format with non-boolean values."""

    bn = sorobn.BayesNet(("Weather", "Mood"))

    bn.P["Weather"] = pd.Series({"Sunny": 0.7, "Rainy": 0.3})
    bn.P["Mood"] = pd.DataFrame(
        {
            "Weather": ["Sunny", "Sunny", "Rainy", "Rainy"],
            "Mood": ["Happy", "Sad", "Happy", "Sad"],
            "p": [0.9, 0.1, 0.4, 0.6],
        }
    )
    bn.prepare()

    result = bn.distribution("Mood", given={"Weather": "Sunny"})
    assert math.isclose(result["Happy"], 0.9)
    assert math.isclose(result["Sad"], 0.1)


def test_cpt_dataframe_rows_format():
    """Test DataFrame CPT using row-based initialization with explicit columns."""

    bn = sorobn.BayesNet(("A", "C"), ("B", "C"))
    bn.P["A"] = pd.Series({True: 0.5, False: 0.5})
    bn.P["B"] = pd.Series({True: 0.5, False: 0.5})
    bn.P["C"] = pd.DataFrame(
        [
            [True, True, True, 0.9],
            [True, True, False, 0.1],
            [True, False, True, 0.6],
            [True, False, False, 0.4],
            [False, True, True, 0.7],
            [False, True, False, 0.3],
            [False, False, True, 0.2],
            [False, False, False, 0.8],
        ],
        columns=["A", "B", "C", "p"],
    )
    bn.prepare()

    P = bn.P["C"]
    assert isinstance(P, pd.Series)
    assert P.index.names == ["A", "B", "C"]
    assert P.groupby(["A", "B"]).sum().eq(1).all()


def test_conditional_reversal():
    """Test that _conditional correctly reverses CPDs via Bayes' theorem.

    In a network (A,B) -> C learned from data where only (T,T,T) and (F,F,F) exist,
    P(B | A=T, C=T) should be 1.0 for B=T (not 0.5 as the marginal P(B) would suggest).

    """
    X = pd.DataFrame(
        [[True, True, True], [False, False, False]],
        columns=["A", "B", "C"],
    )
    bn = sorobn.BayesNet((["A", "B"], "C"))
    bn.fit(X)

    # P(B | A=True, C=True) should be deterministic: B=True
    P = bn._conditional("B", given={"A": True, "C": True})
    assert math.isclose(P[True], 1.0)
    assert False not in P.index or math.isclose(P[False], 0.0)

    # P(B | A=False, C=False) should be deterministic: B=False
    P = bn._conditional("B", given={"A": False, "C": False})
    assert math.isclose(P[False], 1.0)
    assert True not in P.index or math.isclose(P[True], 0.0)


def test_conditional_marginalizes_hidden():
    """Test that _conditional correctly marginalizes unobserved variables.

    In the network A->B, B->C, (A,C)->D with data [(1,1,1,1), (2,1,2,1)],
    P(C | A=1, B=1) should be 1.0 for C=1, because D's CPD only has entries for
    (A=1,C=1) and (A=2,C=2).

    """
    X = pd.DataFrame(
        [[1, 1, 1, 1], [2, 1, 2, 1]],
        columns=["A", "B", "C", "D"],
    )
    bn = sorobn.BayesNet(("A", "B"), ("B", "C"), (["A", "C"], "D"))
    bn.fit(X)

    # Without considering D, P(C|B=1) would be 0.5/0.5.
    # But D's CPD constrains (A=1,C=2) to be impossible, so P(C=1|A=1,B=1) = 1
    P = bn._conditional("C", given={"A": 1, "B": 1})
    assert math.isclose(P[1], 1.0)


def test_conditional_no_evidence():
    """Test that _conditional with no event returns the marginal."""
    bn = sorobn.examples.sprinkler()
    P = bn._conditional("Cloudy", given={})
    assert math.isclose(P[True], 0.5)
    assert math.isclose(P[False], 0.5)


def test_dfs_order_visits_all_nodes():
    """Test that _dfs_order visits every node exactly once."""
    for example_fn in (sorobn.examples.sprinkler, sorobn.examples.asia, sorobn.examples.alarm):
        bn = example_fn()
        order = bn._dfs_order()
        assert sorted(order) == sorted(bn.nodes)
        assert len(order) == len(set(order))


def test_path_sample_only_valid():
    """Test that path sampling never produces invalid (zero-probability) samples.

    This is the core property: forward sampling can fail on these networks,
    but path sampling should always produce valid samples.

    """
    # Example 1: (A,B) -> C with only (T,T,T) and (F,F,F)
    X = pd.DataFrame(
        [[True, True, True], [False, False, False]],
        columns=["A", "B", "C"],
    )
    bn = sorobn.BayesNet((["A", "B"], "C"), seed=42)
    bn.fit(X)

    for _ in range(50):
        s = bn.sample(method="path")
        assert (s["A"] == s["B"] == s["C"]), f"Invalid sample: {dict(s)}"

    # Example 2: A->B, B->C, (A,C)->D
    X = pd.DataFrame(
        [[1, 1, 1, 1], [2, 1, 2, 1]],
        columns=["A", "B", "C", "D"],
    )
    bn = sorobn.BayesNet(("A", "B"), ("B", "C"), (["A", "C"], "D"), seed=42)
    bn.fit(X)

    fjd = bn.full_joint_dist()
    for _ in range(50):
        s = bn.sample(method="path")
        key = tuple(s[col] for col in fjd.index.names)
        assert key in fjd.index, f"Invalid sample: {dict(s)}"


def test_path_sample_distribution():
    """Test that path sampling converges to the correct distribution."""
    bn = sorobn.examples.sprinkler(seed=42)
    fjd = bn.full_joint_dist()

    df = bn.sample(n=5000, method="path")
    empirical = df.value_counts(normalize=True)
    # Reindex to match fjd
    empirical = empirical.reindex(fjd.index, fill_value=0)

    # Allow some sampling noise, but should be close
    max_err = (fjd - empirical).abs().max()
    assert max_err < 0.03, f"Max error {max_err} too large"


def test_path_sample_with_init():
    """Test that path sampling respects init (fixed values)."""
    bn = sorobn.examples.sprinkler(seed=42)

    for _ in range(20):
        s = bn.sample(method="path", init={"Cloudy": True})
        assert s["Cloudy"] == True


def test_junction_tree_graph_algorithms():
    """Test the junction tree graph algorithm primitives."""
    from sorobn.sampling import moralize, triangulate, find_cliques, build_junction_tree

    # Sprinkler: Cloudy -> Rain, Cloudy -> Sprinkler, (Rain, Sprinkler) -> Wet grass
    bn = sorobn.examples.sprinkler()
    adj = moralize(bn.parents, bn.children, bn.nodes)

    # Rain and Sprinkler should be married (co-parents of Wet grass)
    assert "Rain" in adj["Sprinkler"]
    assert "Sprinkler" in adj["Rain"]

    tri_adj, order = triangulate(adj)
    assert set(order) == set(bn.nodes)

    cliques = find_cliques(tri_adj, order)
    # Every node must appear in at least one clique
    all_vars = set().union(*cliques)
    assert all_vars == set(bn.nodes)

    tree_adj = build_junction_tree(cliques)
    # Junction tree has |cliques| - 1 edges
    n_edges = sum(len(nbs) for nbs in tree_adj.values()) // 2
    assert n_edges == len(cliques) - 1


def test_junction_sample_only_valid():
    """Test that junction tree sampling never produces invalid samples."""
    # Example 1
    X = pd.DataFrame(
        [[True, True, True], [False, False, False]],
        columns=["A", "B", "C"],
    )
    bn = sorobn.BayesNet((["A", "B"], "C"), seed=42)
    bn.fit(X)

    for _ in range(50):
        s = bn.sample(method="junction")
        assert (s["A"] == s["B"] == s["C"]), f"Invalid sample: {dict(s)}"

    # Example 2
    X = pd.DataFrame(
        [[1, 1, 1, 1], [2, 1, 2, 1]],
        columns=["A", "B", "C", "D"],
    )
    bn = sorobn.BayesNet(("A", "B"), ("B", "C"), (["A", "C"], "D"), seed=42)
    bn.fit(X)

    fjd = bn.full_joint_dist()
    for _ in range(50):
        s = bn.sample(method="junction")
        key = tuple(s[col] for col in fjd.index.names)
        assert key in fjd.index, f"Invalid sample: {dict(s)}"


def test_junction_sample_distribution():
    """Test that junction tree sampling converges to the correct distribution."""
    bn = sorobn.examples.sprinkler(seed=42)
    fjd = bn.full_joint_dist()

    df = bn.sample(n=5000, method="junction")
    empirical = df.value_counts(normalize=True).reindex(fjd.index, fill_value=0)

    max_err = (fjd - empirical).abs().max()
    assert max_err < 0.03, f"Max error {max_err} too large"


def test_junction_sample_with_init():
    """Test that junction tree sampling respects init (fixed values)."""
    bn = sorobn.examples.sprinkler(seed=42)

    for _ in range(20):
        s = bn.sample(method="junction", init={"Cloudy": True})
        assert s["Cloudy"] == True


def test_junction_sample_asia():
    """Test junction tree sampling on the larger Asia network (8 nodes)."""
    bn = sorobn.examples.asia(seed=42)
    fjd = bn.full_joint_dist()

    df = bn.sample(n=3000, method="junction")
    empirical = df.value_counts(normalize=True).reindex(fjd.index, fill_value=0)

    max_err = (fjd - empirical).abs().max()
    assert max_err < 0.03, f"Max error {max_err} too large"


def test_predict_proba_order_doesnt_matter():
    bn = sorobn.examples.alarm()
    event = {
        "Alarm": False,
        "Burglary": False,
        "Earthquake": True,
        "John calls": False,
        "Mary calls": False,
    }

    for order in itertools.permutations(event.keys()):
        ordered_event = {var: event[var] for var in order}
        assert math.isclose(bn.predict_proba(ordered_event), bn.predict_proba(event))
