import numpy as np
import pandas as pd
import pytest

import sorobn


@pytest.mark.parametrize("values", [
    pd.Series([1, 2], dtype=int),
    pd.Series([1.0, 2.0], dtype=float),
    pd.Series([1, 2], dtype="Int64"),
])
def test_discretizer_accepts_numeric_dtypes(values):
    sorobn.Discretizer(n_bins=2).fit(values)


@pytest.mark.parametrize("values", [
    pd.Series(["1", "2"], dtype="string"),
    pd.Series(["1", "2"], dtype=object),
    pd.Series([True, False], dtype=bool),
    pd.Series([1, 2], dtype="category"),
])
def test_discretizer_rejects_categorical_dtypes(values):
    with pytest.raises(TypeError, match="numeric column"):
        sorobn.Discretizer(n_bins=2).fit(values)


def test_interval_categories_keep_empty_bins_and_nulls():
    values = pd.Series([0., 0.5, None, 3.], index=list("abcd"), name="amount")
    original = values.copy()
    scheme = sorobn.Discretizer(edges=[0, 1, 2, 3])
    result = scheme.fit_transform(values)

    assert isinstance(result.dtype, pd.CategoricalDtype)
    assert result.cat.ordered
    assert list(result.cat.categories) == [
        pd.Interval(0, 1, closed="left"),
        pd.Interval(1, 2, closed="left"),
        pd.Interval(2, 3, closed="both"),
        sorobn.MISSING,
    ]
    assert result.cat.codes.tolist() == [0, 0, 3, 2]
    assert result.eq(sorobn.MISSING).tolist() == [False, False, True, False]
    assert result.iloc[0].left == 0
    assert result.iloc[0].right == 1
    assert result.iloc[-1].closed == "both"
    pd.testing.assert_index_equal(result.iloc[:1].cat.categories, result.cat.categories)
    pd.testing.assert_series_equal(values, original)
    assert result.index.equals(values.index)
    assert result.name == values.name


def test_interval_membership_at_every_boundary():
    edges = [0., 1., 2., 3.]
    values = [0., np.nextafter(1., 0.), 1., np.nextafter(1., 2.), 2., 3.]
    scheme = sorobn.Discretizer(edges=edges).fit(values)
    binned = scheme.transform(values)
    assert binned.cat.codes.tolist() == [0, 0, 1, 1, 2, 2]
    for value, interval in zip(values, binned):
        assert value in interval
        assert sum(value in category for category in binned.cat.categories[:-1]) == 1


def test_constant_column_is_a_closed_point_interval():
    result = sorobn.Discretizer().fit_transform(pd.Series([7., 7., None]))
    assert isinstance(result.dtype, pd.CategoricalDtype)
    assert list(result.cat.categories) == [
        pd.Interval(7., 7., closed="both"), sorobn.MISSING,
    ]
    assert result.cat.codes.tolist() == [0, 0, 1]
    assert 7 in result.iloc[0]


def test_interval_endpoints_are_not_rounded_for_interpolation():
    edges = [0.123456789, 0.123456799, 0.123456809]
    scheme = sorobn.Discretizer(edges=edges)
    model = sorobn.BayesNet("X", discretizers={"X": scheme}).fit(
        pd.DataFrame({"X": [edges[0], edges[-1]]})
    )
    categories = model.P["X"].index.categories
    assert categories[0].left == edges[0]
    assert categories[0].right == edges[1]
    threshold = edges[0] + (edges[1] - edges[0]) / 2
    expected = 0.5 * (threshold - edges[0]) / (edges[1] - edges[0])
    assert model.probability({"X": sorobn.Lt(threshold)}) == pytest.approx(expected)


def test_posterior_and_incremental_fit_retain_interval_categories():
    data = pd.DataFrame({"X": [0., None, 3., 0.5], "Y": [0, 1, 1, 0]})
    schemes = {"X": sorobn.Discretizer(edges=[0, 1, 2, 3])}
    full = sorobn.BayesNet(("X", "Y"), discretizers=schemes).fit(data)
    partial = sorobn.BayesNet(("X", "Y"), discretizers=schemes)
    for i in range(len(data)):
        partial.partial_fit(data.iloc[i:i + 1])
    for node in full.P:
        pd.testing.assert_series_equal(full.P[node], partial.P[node])
    for model in (full, partial):
        posterior = model.distribution("X", given={"Y": 1})
        assert isinstance(posterior.index, pd.CategoricalIndex)
        assert len(posterior.index.categories) == 4
        assert posterior[pd.Interval(2., 3., closed="both")] == 0.5
        assert posterior[sorobn.MISSING] == 0.5
        assert model.probability({"X": sorobn.Between(1, 2)}) == 0
        assert model.probability({"X": sorobn.IsNull()}) == 0.25


@pytest.mark.parametrize("method", ["forward", "path", "junction"])
@pytest.mark.parametrize("missing", [False, True])
def test_sampling_native_bins_with_unobserved_categories(method, missing):
    data = pd.DataFrame({"X": [0., 3.], "Y": [0, 1]})
    if missing:
        data.loc[2] = [np.nan, 2]
    model = sorobn.BayesNet(
        ("X", "Y"), discretizers={"X": sorobn.Discretizer(edges=[0, 1, 2, 3])}, seed=42,
    ).fit(data)
    samples = model.sample(20, method=method)
    assert samples.X.dtype == model.discretizers["X"].dtype_
    for x, y in samples.itertuples(index=False, name=None):
        if x is sorobn.MISSING:
            assert missing and y == 2
            continue
        assert isinstance(x, pd.Interval)
        assert (x.left, y) in {(0., 0), (2., 1)}


def test_all_null_input_with_explicit_bins():
    model = sorobn.BayesNet(
        "X", discretizers={"X": sorobn.Discretizer(edges=[0, 1, 2])}
    ).fit(pd.DataFrame({"X": [sorobn.MISSING, pd.NA]}))
    assert isinstance(model.P["X"].index, pd.CategoricalIndex)
    assert len(model.P["X"].index.categories) == 3
    assert model.probability({"X": sorobn.IsNull()}) == 1
    assert model.probability({"X": sorobn.IsNotNull()}) == 0


def test_structure_learning_preserves_categorical_nulls_and_empty_bins():
    data = pd.DataFrame({"X": [0., None, 3., None], "Y": [0, 1, 0, 1]})
    binned = data.copy()
    binned["X"] = sorobn.Discretizer(edges=[0, 1, 2, 3]).fit_transform(data.X)
    original = binned.copy(deep=True)
    assert sorobn.structure.chow_liu(binned, root="Y") == [("Y", "X")]
    pd.testing.assert_frame_equal(binned, original)
