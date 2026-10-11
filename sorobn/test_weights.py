"""Fitting from counts must match fitting from the repeated rows."""
import numpy as np
import pandas as pd
import pytest

import sorobn


def rows_and_counts():
    rows = sorobn.examples.asia().sample(2_000)
    rows["amount"] = np.random.default_rng(0).integers(0, 1_000, len(rows))
    counts = rows.groupby(list(rows.columns), observed=True).size().rename("n").reset_index()
    return rows, counts


def edges(net):
    return [(p, c) for c, parents in net.parents.items() for p in sorted(parents)]


def assert_same_tables(a, b):
    assert a.P.keys() == b.P.keys()
    for node in a.P:
        left, right = a.P[node].sort_index(), b.P[node].sort_index()
        assert left.index.equals(right.index)
        np.testing.assert_allclose(left.to_numpy(), right.to_numpy())


def make_net(edges):
    return sorobn.BayesNet(
        *edges,
        discretizers={"amount": sorobn.Discretizer(n_bins=4)},
        compactors={"Smoker": sorobn.Compactor(max_categories=2)},
    )


def test_fit_with_counts_matches_rows():
    rows, counts = rows_and_counts()
    structure = edges(sorobn.examples.asia()) + [("Smoker", "amount")]
    by_rows = make_net(structure).fit(rows)
    by_counts = make_net(structure).fit(counts, weights="n")
    assert_same_tables(by_rows, by_counts)
    assert by_rows.discretizers["amount"].edges_.tolist() == by_counts.discretizers["amount"].edges_.tolist()


def test_partial_fit_with_counts_matches_rows():
    rows, counts = rows_and_counts()
    structure = edges(sorobn.examples.asia())
    by_rows = sorobn.BayesNet(*structure).fit(rows.drop(columns="amount"))
    by_counts = sorobn.BayesNet(*structure)
    counts = counts.drop(columns="amount").groupby(list(rows.columns[:-1]), observed=True)["n"].sum().reset_index()
    for chunk in np.array_split(np.arange(len(counts)), 3):
        by_counts.partial_fit(counts.iloc[chunk], weights="n")
    assert_same_tables(by_rows, by_counts)


def test_chow_liu_with_counts_matches_rows():
    rows, counts = rows_and_counts()
    rows = rows.drop(columns="amount")
    counts = counts.drop(columns="amount").groupby(list(rows.columns), observed=True)["n"].sum().reset_index()
    assert sorobn.structure.chow_liu(rows) == sorobn.structure.chow_liu(
        counts.drop(columns="n"), weights=counts["n"]
    )


def test_weights_validation():
    data = pd.DataFrame({"a": [0, 1], "n": [1, -1]})
    with pytest.raises(ValueError, match="weights"):
        sorobn.BayesNet("a").fit(data, weights="n")
    with pytest.raises(ValueError, match="weights"):
        sorobn.BayesNet("a").fit(data[["a"]], weights=[1, 2, 3])
