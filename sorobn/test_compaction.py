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
    assert model.compactors["city"].infrequent_values_ == ("Lyon", "Nice")
    assert model.compactors["city"].infrequent_counts_ == {
        "Lyon": 2, "Nice": 1
    }
    assert model.P["city"]["Paris"] == pytest.approx(0.4)
    assert model.P["city"]["London"] == pytest.approx(0.3)
    assert model.P["city"][sorobn.OTHER] == pytest.approx(0.3)
    assert model.P["bought"].loc[sorobn.OTHER, True] == pytest.approx(1 / 3)
    assert model.P["bought"].loc[sorobn.OTHER, False] == pytest.approx(2 / 3)

    # Individual bucket members receive a uniform share of the aggregate mass.
    assert model.probability({"city": sorobn.OTHER}) == pytest.approx(0.3)
    assert model.probability({"city": "Lyon"}) == pytest.approx(0.15)
    assert model.probability({"city": sorobn.Eq("Nice")}) == pytest.approx(0.15)
    assert model.probability({"city": sorobn.In(["Paris", "Nice"])}) == pytest.approx(0.55)
    assert model.probability({"city": sorobn.Glob("L*")}) == pytest.approx(0.45)
    assert model.probability({"city": "Berlin"}) == 0
    assert model.distribution("bought", given={"city": "Nice"})[True] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "disaggregation,alpha,lyon,nice",
    [
        ("uniform", 1, 0.15, 0.15),
        ("empirical", 1, 0.2, 0.1),
        ("dirichlet", 1, 0.18, 0.12),
        ("dirichlet", 2, 6 / 35, 9 / 70),
    ],
)
def test_disaggregation_strategies(disaggregation, alpha, lyon, nice):
    data = pd.DataFrame({
        "city": ["Paris"] * 4 + ["London"] * 3 + ["Lyon"] * 2 + ["Nice"]
    })
    model = sorobn.BayesNet(
        "city",
        compactors={
            "city": sorobn.Compactor(
                max_categories=3,
                disaggregation=disaggregation,
                alpha=alpha,
            )
        },
    ).fit(data)

    assert model.probability({"city": "Lyon"}) == pytest.approx(lyon)
    assert model.probability({"city": "Nice"}) == pytest.approx(nice)
    assert model.probability({
        "city": sorobn.In(["Lyon", "Nice"])
    }) == pytest.approx(0.3)
    assert model.probability({"city": sorobn.Glob("L*")}) == pytest.approx(
        0.3 + lyon
    )


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
    assert "unused" not in compactor.infrequent_values_


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
    assert repr(sorobn.Compactor(
        min_frequency=0.1, disaggregation="dirichlet", alpha=2
    )) == (
        "Compactor(min_frequency=0.1, disaggregation='dirichlet', alpha=2)"
    )


@pytest.mark.parametrize(
    "min_frequency,expected",
    [
        (2, ("a", "b")),
        (0.3, ("a",)),
    ],
)
def test_min_frequency_accepts_counts_and_fractions(min_frequency, expected):
    # Fractions use all observations as their denominator; null stays separate.
    compactor = sorobn.Compactor(min_frequency=min_frequency).fit(
        ["a", "a", "a", "b", "b", "c", None, None]
    )
    assert compactor.frequent_values_ == expected
    assert compactor.transform([None]).iloc[0] is sorobn.MISSING


def test_partial_fit_freezes_groups_and_never_grows_past_limit():
    model = sorobn.BayesNet(
        "kind",
        compactors={
            "kind": sorobn.Compactor(
                max_categories=3, disaggregation="empirical"
            )
        },
    )
    model.partial_fit(pd.DataFrame({"kind": ["a", "a", "b"]}))
    model.partial_fit(pd.DataFrame({"kind": ["c", "c", "d"]}))

    assert model.compactors["kind"].frequent_values_ == ("a", "b")
    assert model.P["kind"]["a"] == pytest.approx(2 / 6)
    assert model.P["kind"]["b"] == pytest.approx(1 / 6)
    assert model.P["kind"][sorobn.OTHER] == pytest.approx(3 / 6)
    assert model.compactors["kind"].infrequent_values_ == ("c", "d")
    assert model.compactors["kind"].infrequent_counts_ == {"c": 2, "d": 1}
    assert model.probability({"kind": "c"}) == pytest.approx(1 / 3)
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
    {"max_categories": 2, "min_frequency": 2},
    {"max_categories": 0},
    {"max_categories": True},
    {"min_frequency": 0},
    {"min_frequency": 1.1},
    {"min_frequency": True},
    {"max_categories": 2, "disaggregation": "unknown"},
    {"max_categories": 2, "alpha": 0},
    {"max_categories": 2, "alpha": float("nan")},
    {"max_categories": 2, "alpha": True},
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
