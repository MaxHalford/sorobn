import copy
import pickle

import pandas as pd
import pytest

import sorobn


def test_max_categories_groups_least_frequent_values_everywhere():
    data = pd.DataFrame({
        "city": ["Paris"] * 4 + ["London"] * 3 + ["Lyon"] * 2 + ["Nice"],
        "bought": [True] * 4 + [False] * 3 + [True, False] + [False],
    })
    model = sorobn.BayesNet(
        ("city", "bought"),
        compactors={"city": sorobn.Compactor(max_categories=3)},
    ).fit(data)

    assert model.compactors["city"].frequent_values_ == ("Paris", "London")
    assert model.compactors["city"].n_members_ == 2
    assert model.P["city"]["Paris"] == pytest.approx(0.4)
    assert model.P["city"]["London"] == pytest.approx(0.3)
    assert model.P["city"][sorobn.OTHER] == pytest.approx(0.3)
    assert model.P["bought"].loc[sorobn.OTHER, True] == pytest.approx(1 / 3)
    assert model.P["bought"].loc[sorobn.OTHER, False] == pytest.approx(2 / 3)

    # Grouped values are assumed equally frequent: 1 / n_members_ of OTHER each.
    assert model.probability({"city": sorobn.OTHER}) == pytest.approx(0.3)
    assert model.probability({"city": "Lyon"}) == pytest.approx(0.15)
    assert model.probability({"city": sorobn.Eq("Nice")}) == pytest.approx(0.15)
    assert model.probability({"city": sorobn.In(["Paris", "Nice"])}) == pytest.approx(0.55)
    assert model.probability({"city": sorobn.Ne("Nice")}) == pytest.approx(0.85)
    # An unseen value is assumed to be a grouped value too.
    assert model.probability({"city": "Berlin"}) == pytest.approx(0.15)
    # Patterns take their share of OTHER from the frequent values: London
    # matches one of two, so OTHER contributes (1 + 1) / (2 + 2) of its 0.3.
    assert model.probability({"city": sorobn.Glob("L*")}) == pytest.approx(0.3 + 0.15)
    assert model.probability({"city": sorobn.Glob("Z*")}) == pytest.approx(0.3 / 4)
    assert model.distribution("bought", given={"city": "Nice"})[True] == pytest.approx(1 / 3)


def test_memory_does_not_grow_with_distinct_values():
    values = [f"v{i}" for i in range(10_000)] + ["common"] * 10_000
    compactor = sorobn.Compactor(max_categories=2).fit(values)
    assert compactor.n_members_ == 10_000
    assert len(pickle.dumps(compactor)) < 1_000


def test_ranges_take_their_share_from_frequent_values():
    data = pd.DataFrame({"x": pd.Series([1, 1, 2, 2, 9, 10, 11]).astype("category")})
    model = sorobn.BayesNet(
        "x", compactors={"x": sorobn.Compactor(max_categories=3)}
    ).fit(data)
    # Frequent: 1 and 2, both below 5. OTHER holds 3 of 7 rows.
    assert model.probability({"x": sorobn.Lt(5)}) == pytest.approx(4 / 7 + 3 / 7 * 3 / 4)


def test_max_categories_includes_reserved_other_and_breaks_ties_by_appearance():
    compactor = sorobn.Compactor(max_categories=3).fit(
        ["second", "first", "third", "second", "first", "third"]
    )
    assert compactor.frequent_values_ == ("second", "first")
    assert compactor.transform(["second", "third", "new"]).tolist() == [
        "second", sorobn.OTHER, sorobn.OTHER
    ]
    assert list(compactor.dtype_.categories) == [
        "second", "first", sorobn.OTHER, sorobn.MISSING,
    ]


def test_categorical_input_ignores_unobserved_levels_and_uses_row_order_for_ties():
    values = pd.Categorical(
        ["second", "first", "third"],
        categories=["unused", "third", "first", "second"],
    )
    compactor = sorobn.Compactor(max_categories=3).fit(values)
    assert compactor.frequent_values_ == ("second", "first")
    assert compactor.n_members_ == 1


@pytest.mark.parametrize("values", [
    pd.Series(["a", "b"], dtype="string"),
    pd.Series(["a", "b"], dtype=object),
    pd.Series([True, False], dtype=bool),
    pd.Series([1, 2], dtype="category"),
])
def test_compactor_accepts_categorical_dtypes(values):
    sorobn.Compactor(max_categories=2).fit(values)


@pytest.mark.parametrize("values", [
    pd.Series([1, 2], dtype=int),
    pd.Series([1.0, 2.0], dtype=float),
    pd.Series([1, 2], dtype="Int64"),
])
def test_compactor_rejects_numeric_dtypes(values):
    with pytest.raises(TypeError, match="categorical column"):
        sorobn.Compactor(max_categories=2).fit(values)


def test_compactor_repr_is_stable_and_concise():
    assert repr(sorobn.Compactor(max_categories=3)) == (
        "Compactor(max_categories=3)"
    )
    assert repr(sorobn.Compactor(coverage=0.9)) == (
        "Compactor(coverage=0.9)"
    )


def test_coverage_keeps_values_until_their_share_of_rows_is_reached():
    # Shares use non-null observations: a covers 6 of 10, a and b cover 9 of 10.
    compactor = sorobn.Compactor(coverage=0.9).fit(
        ["a"] * 6 + ["b"] * 3 + ["c"] + [None] * 2
    )
    assert compactor.frequent_values_ == ("a", "b")
    assert compactor.n_members_ == 1
    assert compactor.transform([None]).iloc[0] is sorobn.MISSING


def test_partial_fit_freezes_groups_and_never_grows_past_limit():
    model = sorobn.BayesNet(
        "kind",
        compactors={
            "kind": sorobn.Compactor(max_categories=3)
        },
    )
    model.partial_fit(pd.DataFrame({"kind": ["a", "a", "b"]}))
    model.partial_fit(pd.DataFrame({"kind": ["c", "c", "d"]}))

    assert model.compactors["kind"].frequent_values_ == ("a", "b")
    assert model.P["kind"]["a"] == pytest.approx(2 / 6)
    assert model.P["kind"]["b"] == pytest.approx(1 / 6)
    assert model.P["kind"][sorobn.OTHER] == pytest.approx(3 / 6)
    assert model.compactors["kind"].n_members_ == 0
    # The first batch had no grouped values: OTHER counts as one member.
    assert model.probability({"kind": "c"}) == pytest.approx(3 / 6)
    assert len(model.P["kind"][model.P["kind"] > 0]) <= 3
    assert model.P["kind"][sorobn.MISSING] == 0


def test_refit_relearns_groups_and_configuration_is_copied():
    config = {"kind": sorobn.Compactor(max_categories=2)}
    model = sorobn.BayesNet("kind", compactors=config)
    model.fit(pd.DataFrame({"kind": ["a", "a", "b"]}))
    assert model.compactors["kind"].frequent_values_ == ("a",)
    model.fit(pd.DataFrame({"kind": ["b", "b", "a"]}))
    assert model.compactors["kind"].frequent_values_ == ("b",)
    assert config["kind"].frequent_values_ is None


def test_grouped_samples_use_the_fitted_categorical_dtype():
    model = sorobn.BayesNet(
        "kind", seed=42,
        compactors={"kind": sorobn.Compactor(max_categories=2)},
    ).fit(pd.DataFrame({"kind": ["a", "a", "b"]}))
    samples = model.sample(10, init={"kind": "b"})
    assert samples["kind"].dtype == model.compactors["kind"].dtype_
    assert samples["kind"].eq(sorobn.OTHER).all()


@pytest.mark.parametrize("kwargs", [
    {},
    {"max_categories": 0},
    {"max_categories": True},
    {"coverage": 0},
    {"coverage": 1.1},
    {"coverage": True},
    {"coverage": "0.9"},
])
def test_validation(kwargs):
    with pytest.raises(ValueError):
        sorobn.Compactor(**kwargs)

    with pytest.raises(ValueError, match="Unknown compacted"):
        sorobn.BayesNet("x", compactors={"missing": sorobn.Compactor(max_categories=2)})
    with pytest.raises(TypeError, match="Compactor"):
        sorobn.BayesNet("x", compactors={"x": object()})
    with pytest.raises(ValueError, match="both"):
        sorobn.BayesNet(
            "x",
            discretizers={"x": sorobn.Discretizer()},
            compactors={"x": sorobn.Compactor(max_categories=2)},
        )


def test_other_is_reserved():
    with pytest.raises(ValueError, match="reserved"):
        sorobn.Compactor(max_categories=2).fit(["a", sorobn.OTHER])


def test_deepcopy_preserves_other_label_equality():
    assert copy.deepcopy(sorobn.OTHER) is sorobn.OTHER


def test_missing_is_a_stable_public_singleton():
    assert repr(sorobn.MISSING) == "<MISSING>"
    assert copy.deepcopy(sorobn.MISSING) is sorobn.MISSING
    assert pickle.loads(pickle.dumps(sorobn.MISSING)) is sorobn.MISSING
    assert sorobn.IsNull()(sorobn.MISSING)
    assert not sorobn.IsNotNull()(sorobn.MISSING)
    model = sorobn.BayesNet("value").fit(
        pd.DataFrame({"value": [sorobn.MISSING, "present"]})
    )
    assert model.P["value"][sorobn.MISSING] == pytest.approx(0.5)


def test_max_categories_caps_coverage():
    values = ["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["d"]
    # Coverage alone keeps a, b, c: 12 of 13 rows. The cap keeps the two most frequent.
    assert sorobn.Compactor(coverage=0.9).fit(values).frequent_values_ == ("a", "b", "c")
    capped = sorobn.Compactor(coverage=0.9, max_categories=3).fit(values)
    assert capped.frequent_values_ == ("a", "b")
    assert capped.n_members_ == 2
    # A loose cap changes nothing.
    loose = sorobn.Compactor(coverage=0.9, max_categories=10).fit(values)
    assert loose.frequent_values_ == ("a", "b", "c")
    assert repr(capped) == "Compactor(max_categories=3, coverage=0.9)"
