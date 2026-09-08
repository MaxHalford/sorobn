import numpy as np
import pandas as pd
import pytest

import sorobn


@pytest.mark.parametrize("discretizer", [
    sorobn.Discretizer(n_bins=2, strategy="uniform"),
    sorobn.Discretizer(n_bins=2, strategy="quantile"),
    sorobn.Discretizer(edges=[0, 3.5, 7]),
])
def test_discretize_learn_structure_fit_raw_data(discretizer):
    # Raw A uniquely identifies every row. Its bins, however, are independent of
    # B and C, which are perfectly correlated with each other.
    data = pd.DataFrame({"A": range(8), "B": [0, 1] * 4, "C": [0, 1] * 4})
    original = data.copy(deep=True)
    schemes = {"A": discretizer}
    expected = [("A", "B"), ("B", "C")]

    binned = data.copy()
    binned["A"] = discretizer.fit_transform(data["A"])
    edges = sorobn.structure.chow_liu(binned, root="A")
    assert edges == expected
    assert sorobn.structure.chow_liu(data, root="A") != expected

    bn = sorobn.BayesNet(*edges, discretizers=schemes).fit(data)
    assert bn.probability({"A": sorobn.Lt(3.5)}, given={"B": 0}) == pytest.approx(0.5)
    assert bn.probability({"B": 0, "C": 1}) == 0
    pd.testing.assert_frame_equal(data, original)
    np.testing.assert_array_equal(bn.discretizers["A"].edges_, discretizer.edges_)
    pd.testing.assert_series_equal(bn.discretizers["A"].transform(data["A"]), binned["A"])


def test_spanning_tree_connects_strongly_correlated_pairs():
    data = pd.DataFrame({
        "A": [0, 0, 1, 1], "B": [0, 0, 1, 1],
        "C": [0, 1, 0, 1], "D": [0, 1, 0, 1],
    })
    edges = sorobn.structure.chow_liu(data)
    assert edges == [("A", "B"), ("A", "C"), ("C", "D")]
    bn = sorobn.BayesNet(*edges).fit(data)
    assert set(bn.nodes) == set(data.columns)
    assert bn.probability({"A": 0, "C": 0}) == pytest.approx(0.25)


def test_structure_learning_null_dependencies():
    # Missingness itself carries the dependency between A and B.
    data = pd.DataFrame({
        "A": [None, "x", None, "x"],
        "B": [0, 1, 0, 1],
        "C": [0., 0.1, 9., 10.],
    })
    schemes = {"C": sorobn.Discretizer(2, "uniform")}
    binned = data.copy()
    binned["C"] = schemes["C"].fit_transform(data["C"])
    edges = sorobn.structure.chow_liu(binned)
    assert edges == [("A", "B"), ("A", "C")]
    bn = sorobn.BayesNet(*edges, discretizers=schemes).fit(data)
    assert bn.probability({"B": 0}, given={"A": sorobn.IsNull()}) == 1
    assert bn.probability({"C": sorobn.Lt(5)}, given={"A": sorobn.IsNull()}) == 0.5


def test_structure_is_deterministic_under_row_and_column_permutation():
    # Independent variables create tied edge weights. Fix the root so changing
    # the first column does not intentionally change the orientation.
    data = pd.DataFrame({
        "A": [0, 0, 1, 1], "B": [0, 1, 0, 1], "C": [1., 0., 0., 1.],
    })
    edges = sorobn.structure.chow_liu(data, root="A")
    assert edges == [("A", "B"), ("A", "C")]
    assert sorobn.structure.chow_liu(
        data.iloc[::-1][["C", "B", "A"]], root="A"
    ) == edges


def test_structure_learning_validates_data_and_root():
    data = pd.DataFrame({"A": [0, 1], "B": [1, 0]})
    with pytest.raises(ValueError, match="Unknown root"):
        sorobn.structure.chow_liu(data, root="typo")
    with pytest.raises(ValueError, match="nonempty"):
        sorobn.structure.chow_liu(data.iloc[:0])
    with pytest.raises(ValueError, match="unique"):
        sorobn.structure.chow_liu(data.rename(columns={"B": "A"}))
