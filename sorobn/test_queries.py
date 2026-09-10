"""Check inference against enumeration, SQL filters, and analytic histograms."""

import itertools
import sqlite3

import numpy as np
import pandas as pd
import pytest

import sorobn as sb


@pytest.fixture
def cities():
    bn = sb.BayesNet(("City", "Purchased"))
    bn.P["City"] = pd.Series({"Paris": 0.6, "Parma": 0.1, "London": 0.3})
    bn.P["Purchased"] = pd.DataFrame([
        ["Paris", True, 0.8], ["Paris", False, 0.2],
        ["Parma", True, 0.1], ["Parma", False, 0.9],
        ["London", True, 0.5], ["London", False, 0.5],
    ], columns=["City", "Purchased", "p"])
    bn.prepare()
    return bn


@pytest.mark.parametrize("predicate", [
    sb.In(["Paris", "Parma"]), sb.Glob("Par*"), sb.Like("Par%"), sb.Regex("^Par"),
    sb.Eq("Paris") | sb.Eq("Parma"), ~sb.Eq("London"),
])
def test_set_evidence_preserves_relative_mass(cities, predicate):
    assert cities.probability({"City": predicate}) == pytest.approx(0.7)
    assert cities.distribution("Purchased", given={"City": predicate})[True] == pytest.approx(0.7)
    assert cities.probability(
        {"Purchased": True}, given={"City": predicate}
    ) == pytest.approx(0.7)


def test_same_variable_in_target_and_evidence(cities):
    assert cities.probability(
        {"City": "Paris"}, given={"City": sb.Glob("Par*")}
    ) == pytest.approx(6 / 7)
    posterior = cities.distribution("City", given={"City": sb.Glob("Par*")})
    assert posterior["Paris"] == pytest.approx(6 / 7)
    assert posterior.sum() == pytest.approx(1)
    assert cities.probability({"City": sb.Eq("Paris") | sb.Glob("Par*")}) == pytest.approx(0.7)


def test_exact_against_joint_enumeration():
    bn = sb.examples.asia()
    joint = bn.full_joint_dist()
    for names in itertools.combinations(joint.index.names, 2):
        for values in itertools.product([True, False], repeat=2):
            event = dict(zip(names, values))
            selected = joint
            for name, value in event.items():
                selected = selected[selected.index.get_level_values(name) == value]
            assert bn.probability(event) == pytest.approx(selected.sum())


def test_disconnected_scalar_factors():
    bn = sb.BayesNet("A", "B")
    bn.P["A"] = pd.Series({0: 0.2, 1: 0.8})
    bn.P["B"] = pd.Series({0: 0.3, 1: 0.7})
    bn.prepare()
    assert bn.probability({"A": 0, "B": 1}) == pytest.approx(0.14)
    pd.testing.assert_series_equal(
        bn.distribution("A", given={"B": 1}), bn.distribution("A")
    )


def test_zero_and_empty_events(cities):
    assert cities.probability({}) == 1
    assert cities.probability({"City": sb.In([])}) == 0
    assert cities.probability({"City": "Atlantis"}) == 0
    assert cities.probability({"City": "Paris"}, given={"City": "Parma"}) == 0
    for call in (
        lambda: cities.distribution("Purchased", given={"City": "Atlantis"}),
        lambda: cities.probability({}, given={"City": "Atlantis"}),
    ):
        with pytest.raises(ValueError, match="zero probability"):
            call()
    with pytest.raises(ValueError, match="Unknown"):
        cities.probability({"Typo": True})


def test_probability_does_not_materialize_joint(cities, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Scalar selectivity must use variable elimination")

    monkeypatch.setattr(cities, "full_joint_dist", fail)
    assert cities.predict_proba({"City": "London"}) == pytest.approx(0.3)
    result = cities.predict_proba(pd.DataFrame({"City": ["London", "Paris", "London"]}))
    assert list(result) == pytest.approx([0.3, 0.6, 0.3])


@pytest.mark.parametrize("algorithm", ["gibbs", "likelihood", "rejection"])
def test_unsupported_sampler_is_explicit(cities, algorithm):
    with pytest.raises(ValueError, match="require algorithm='exact'"):
        cities.distribution("Purchased", given={"City": sb.Glob("Par*")}, algorithm=algorithm)


@pytest.mark.parametrize("predicate,sql", [
    (sb.Like("Star%"), "title LIKE 'Star%'"),
    (~sb.Like("%Star%"), "title NOT LIKE '%Star%'"),
    (sb.Like(r"100\%%"), "title LIKE '100\\%%' ESCAPE '\\'"),
    (sb.Like("Sta_ %"), "title LIKE 'Sta_ %'"),
    (sb.In(["Alien", "Star Wars"]), "title IN ('Alien', 'Star Wars')"),
    (~sb.In(["Alien", None]), "title NOT IN ('Alien', NULL)"),
    (sb.IsNull(), "title IS NULL"),
    (sb.IsNotNull(), "title IS NOT NULL"),
    (sb.IsNull() | sb.Like("Star%"), "title IS NULL OR title LIKE 'Star%'"),
])
def test_title_filters_against_sql(predicate, sql):
    data = pd.DataFrame({
        "title": ["Star Wars", "Star Trek", "Alien", None, "100% real", "star wars"],
        "year": [1977, 2009, 1979, 2000, None, 1977],
    })
    bn = sb.BayesNet(("title", "year")).fit(data)
    with sqlite3.connect(":memory:") as connection:
        connection.execute("PRAGMA case_sensitive_like=ON")
        data.to_sql("movies", connection, index=False)
        count = connection.execute(f"SELECT count(*) FROM movies WHERE {sql}").fetchone()[0]
    assert bn.probability({"title": predicate}) == pytest.approx(count / len(data))
    # Include the child so inference must join a null parent state across factors.
    mask = data.title.map(predicate) & data.year.notna()
    assert bn.probability({"title": predicate, "year": sb.IsNotNull()}) == pytest.approx(mask.mean())


def test_nulls_and_partial_fit():
    data = pd.DataFrame({"A": [None, "x", None, "x"], "B": [1, None, None, 2]})
    full = sb.BayesNet(("A", "B")).fit(data)
    partial = sb.BayesNet(("A", "B"))
    for row in range(len(data)):
        partial.partial_fit(data.iloc[row:row + 1])
    for model in (full, partial):
        assert model.probability({"A": sb.IsNull(), "B": sb.IsNull()}) == pytest.approx(0.25)
        assert model.probability({"B": sb.IsNull()}) == pytest.approx(0.5)
        assert model.distribution("A", given={"B": sb.IsNull()}).sum() == pytest.approx(1)


def test_prepare_canonicalizes_missing_states_in_manual_tables():
    model = sb.BayesNet(("A", "B"))
    model.P["A"] = pd.Series([0.5, 0.5], index=pd.Index(["x", None]))
    model.P["B"] = pd.DataFrame({
        "A": ["x", "x", pd.NA, pd.NA],
        "B": [0, 1, 0, 1],
        "p": [0.25, 0.75, 0.8, 0.2],
    })
    model.prepare()
    assert model.P["A"].index.tolist() == ["x", sb.MISSING]
    assert model.probability(
        {"B": 0}, given={"A": sb.IsNull()}
    ) == pytest.approx(0.8)


@pytest.mark.parametrize("marker", [None, np.nan, pd.NA, pd.NaT, sb.MISSING])
def test_impute_accepts_every_pandas_missing_scalar(marker):
    numeric = sb.BayesNet("X").fit(pd.DataFrame({"X": [10, 10, None]}))
    assert numeric.impute({"X": marker})["X"] == 10

    text = sb.BayesNet("X").fit(pd.DataFrame({"X": ["long", "long", None]}))
    assert text.impute({"X": marker})["X"] == "long"


def test_impute_is_a_noop_without_missing_values_and_preserves_order():
    model = sb.BayesNet(("A", "B")).fit(pd.DataFrame({
        "A": ["x", "x", "y"],
        "B": [1, None, 2],
    }))
    complete = {"B": 2, "A": "y"}
    result = model.impute(complete)
    assert result.to_dict() == complete
    assert result.index.tolist() == ["B", "A"]


def test_impute_never_chooses_the_missing_state():
    model = sb.BayesNet(("A", "B")).fit(pd.DataFrame({
        "A": [None, None, "x"],
        "B": [None, None, 1],
    }))
    result = model.impute({"A": None, "B": None})
    assert result.to_dict() == {"A": "x", "B": 1.0}

    all_null = sb.BayesNet("X").fit(pd.DataFrame({"X": [None, pd.NA]}))
    with pytest.raises(ValueError, match="non-null"):
        all_null.impute({"X": None})


@pytest.mark.parametrize("algorithm", ["gibbs", "likelihood", "rejection"])
def test_approximate_inference_retains_missing_target_mass(algorithm):
    model = sb.BayesNet("X", seed=42).fit(
        pd.DataFrame({"X": ["a", "a", None]})
    )
    result = model.distribution("X", algorithm=algorithm, n_iterations=3_000)
    assert result.sum() == pytest.approx(1)
    assert result[sb.MISSING] == pytest.approx(1 / 3, abs=0.05)


@pytest.mark.parametrize("algorithm", ["gibbs", "likelihood", "rejection"])
@pytest.mark.parametrize("marker", [None, np.nan, pd.NA, pd.NaT, sb.MISSING])
def test_approximate_inference_accepts_missing_evidence(algorithm, marker):
    data = pd.DataFrame({
        "A": [None, None, "x", "x"],
        "B": [0, 0, 1, 1],
    })
    model = sb.BayesNet(("A", "B"), seed=42).fit(data)
    result = model.distribution(
        "B", given={"A": marker}, algorithm=algorithm, n_iterations=100
    )
    assert result.to_dict() == {0: 1.0}


@pytest.mark.parametrize("method", ["forward", "path", "junction"])
@pytest.mark.parametrize("marker", [None, np.nan, pd.NA, pd.NaT, sb.MISSING])
def test_conditioned_sampling_accepts_missing_values(method, marker):
    data = pd.DataFrame({
        "A": [None, None, "x", "x"],
        "B": [0, 0, 1, 1],
    })
    model = sb.BayesNet(("A", "B"), seed=42).fit(data)
    sample = model.sample(init={"A": marker}, method=method)
    assert sample["A"] is sb.MISSING
    assert sample["B"] == 0


def test_predict_proba_treats_missing_cells_as_observed_missing_states():
    model = sb.BayesNet("X").fit(pd.DataFrame({"X": ["a", None]}))
    for marker in (None, np.nan, pd.NA, pd.NaT, sb.MISSING):
        assert model.probability({"X": marker}) == pytest.approx(0.5)
        assert model.predict_proba({"X": marker}) == pytest.approx(0.5)
        assert model.probability({"X": sb.Eq(marker)}) == 0
    result = model.predict_proba(pd.DataFrame({"X": ["a", None]}))
    assert result.tolist() == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("predicate,values,expected", [
    (sb.Between(1, 2), [0, 1, 2, 3, None], [False, True, True, False, False]),
    (sb.Glob("a?*"), ["a", "ab", "abc", None, 12], [False, True, True, False, False]),
    (sb.Regex("b"), ["abc", "b", "aaa", None], [True, True, False, False]),
    (sb.Like("a%b"), ["a\nb", "ab", "xab", "abx"], [True, True, False, False]),
    (sb.Like("AB", case_sensitive=False), ["ab", "Ab", "abc"], [True, True, False]),
    (~sb.In(["x", None]), ["x", "y", None], [False, False, False]),
    (~(sb.IsNull() | sb.Eq("x")), ["x", "y", None], [False, True, False]),
])
def test_predicate_semantics(predicate, values, expected):
    assert list(map(predicate, values)) == expected


def test_invalid_predicate_usage():
    with pytest.raises(ValueError, match="escape"):
        sb.Like("oops\\")
    with pytest.raises(ValueError, match="escape"):
        sb.Like("%", escape="xx")
    with pytest.raises(TypeError, match="Combine predicates"):
        bool(sb.Gt(1))


@pytest.fixture
def histogram():
    return sb.BayesNet("X", discretizers={"X": sb.Discretizer(edges=[0, 10, 20])}).fit(
        pd.DataFrame({"X": [0, 5, 10, 20]})
    )


def test_analytic_histogram_ranges(histogram):
    assert histogram.probability({"X": sb.Between(5, 15)}) == pytest.approx(0.5)
    assert histogram.probability({"X": sb.Lt(5)}) == pytest.approx(0.25)
    assert histogram.probability({"X": sb.Gt(15)}) == pytest.approx(0.25)
    assert histogram.probability({"X": sb.Between(-10, 100)}) == 1
    assert histogram.probability({"X": sb.Between(30, 40)}) == 0
    assert histogram.probability({"X": sb.Between(10, 0)}) == 0
    assert histogram.probability({"X": 5}) == 0
    assert histogram.probability({"X": sb.In([5, 10])}) == 0
    assert histogram.probability({"X": ~sb.Eq(5)}) == 1


def test_continuous_intersections_before_interpolation(histogram):
    assert histogram.probability(
        {"X": sb.Between(2, 8)}, given={"X": sb.Between(0, 5)}
    ) == pytest.approx(3 / 5)
    assert histogram.probability(
        {"X": sb.Between(6, 9)}, given={"X": sb.Between(0, 5)}
    ) == 0
    assert histogram.probability({
        "X": sb.Between(0, 8) | sb.Between(2, 10)
    }) == pytest.approx(0.5)
    assert histogram.probability({
        "X": ~(sb.Between(0, 8) | sb.Between(2, 10))
    }) == pytest.approx(0.5)


def test_fractional_evidence_applied_once():
    bn = sb.BayesNet(
        ("X", ["A", "B"]), discretizers={"X": sb.Discretizer(edges=[0, 10, 20])}
    ).fit(pd.DataFrame({"X": [0, 5, 10, 20], "A": [0, 0, 1, 1], "B": [0, 0, 1, 1]}))
    assert bn.probability({"X": sb.Between(5, 15), "A": 0, "B": 0}) == pytest.approx(0.25)
    posterior = bn.distribution("A", "B", given={"X": sb.Between(5, 15)})
    assert posterior.loc[0, 0] == pytest.approx(0.5)
    assert posterior.loc[1, 1] == pytest.approx(0.5)


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_discretization_strategies_and_boundaries(strategy):
    values = pd.Series([0, 0, 0, 1, 5, 10], name="X")
    d = sb.Discretizer(n_bins=2, strategy=strategy).fit(values)
    expected = [0, 5, 10] if strategy == "uniform" else [0, 0.5, 10]
    assert d.edges_ == pytest.approx(expected)
    assert d.transform(pd.Series(expected)).cat.codes.tolist() == [0, 1, 1]
    transformed = d.transform(values)
    assert transformed.index.equals(values.index)
    assert transformed.name == values.name


def test_constant_and_duplicate_quantile_edges():
    constant = sb.BayesNet("X", discretizers={"X": sb.Discretizer()}).fit(
        pd.DataFrame({"X": [3, 3, None]})
    )
    assert constant.probability({"X": 3}) == pytest.approx(2 / 3)
    assert constant.probability({"X": sb.Gt(3)}) == 0
    assert constant.probability({"X": sb.IsNull()}) == pytest.approx(1 / 3)
    assert constant.probability({"X": sb.IsNotNull()}) == pytest.approx(2 / 3)
    duplicate = sb.Discretizer(4).fit([0, 0, 0, 0, 1])
    assert list(duplicate.edges_) == [0, 1]


def test_fit_partial_fit_and_input_ownership():
    data = pd.DataFrame({"X": [0, 3, 7, 10], "Y": [0, 0, 1, 1]})
    original = data.copy()
    config = {"X": sb.Discretizer(edges=[0, 5, 10])}
    full = sb.BayesNet(("X", "Y"), discretizers=config).fit(data)
    partial = sb.BayesNet(("X", "Y"), discretizers=config)
    partial.partial_fit(data.iloc[:2]).partial_fit(data.iloc[2:])
    for node in full.P:
        pd.testing.assert_series_equal(full.P[node], partial.P[node])
    assert config["X"].edges_ is None
    pd.testing.assert_frame_equal(data, original)
    with pytest.raises(ValueError, match="outside"):
        partial.partial_fit(pd.DataFrame({"X": [20], "Y": [1]}))
    assert partial.probability({"X": sb.Lt(5)}) == pytest.approx(0.5)


def test_refit_relearns_bins():
    bn = sb.BayesNet("X", discretizers={"X": sb.Discretizer(2, "uniform")})
    bn.fit(pd.DataFrame({"X": [0, 10]}))
    bn.fit(pd.DataFrame({"X": [100, 200]}))
    assert list(bn.discretizers["X"].edges_) == [100, 150, 200]
    assert bn.probability({"X": sb.Lt(150)}) == pytest.approx(0.5)


def test_discretization_validation(histogram):
    for edges in ([0, 0], [1, 0], [0, np.inf], [0, np.nan], [0]):
        with pytest.raises(ValueError, match="edges"):
            sb.Discretizer(edges=edges)
    for n_bins in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="n_bins"):
            sb.Discretizer(n_bins=n_bins)
    with pytest.raises(ValueError, match="strategy"):
        sb.Discretizer(strategy="typo")
    with pytest.raises(ValueError, match="all-null"):
        sb.Discretizer().fit([None, None])
    with pytest.raises(ValueError, match="finite"):
        sb.Discretizer().fit([0, np.inf])
    with pytest.raises(ValueError, match="Fit"):
        sb.Discretizer().transform([0])
    with pytest.raises(TypeError, match="discretized"):
        histogram.probability({"X": sb.Glob("*")})
