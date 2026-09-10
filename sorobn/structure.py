import collections
import itertools

import numpy as np
import pandas as pd

from .predicates import is_null, normalize_missing

__all__ = ["chow_liu"]


def chow_liu(X, root=None):
    """Return a Chow-Liu tree.

    A Chow-Liu tree takes three steps to build:

    1. Compute the mutual information between each pair of variables. The values are organised in
        a fully connected graph.
    2. Extract the maximum spanning tree from the graph.
    3. Orient the edges of the tree by picking a root.

    TODO: the current implementation uses Kruskal's algorithm to extract the MST. According to
        Wikipedia, faster algorithms exist for fully connected graphs.

    Parameters
    ----------
    X
        A nonempty DataFrame of discrete observations. Discretize continuous
        columns before calling this function. Missing values are a separate state.
    root
        Root of the directed tree. Defaults to the first column.

    Examples
    --------
    >>> import pandas as pd
    >>> import sorobn
    >>> data = pd.DataFrame({"amount": [0., 1., 9., 10.], "kind": [0, 0, 1, 1]})
    >>> scheme = sorobn.Discretizer(n_bins=2, strategy="uniform")
    >>> binned = data.copy()
    >>> binned["amount"] = scheme.fit_transform(data["amount"])
    >>> edges = sorobn.structure.chow_liu(binned, root="kind")
    >>> edges
    [('kind', 'amount')]
    >>> bn = sorobn.BayesNet(*edges, discretizers={"amount": scheme}).fit(data)
    >>> bn.probability({"amount": sorobn.Lt(5)}, given={"kind": 0})
    1.0

    References
    ----------

    1. Chow, C. and Liu, C., 1968. Approximating discrete probability distributions with
        dependence trees. IEEE transactions on Information Theory, 14(3), pp.462-467.
    2. https://www.wikiwand.com/en/Chow-Liu_tree

    """

    if X.empty:
        raise ValueError("Structure learning requires nonempty data")
    if not X.columns.is_unique:
        raise ValueError("Column names must be unique")
    if root is None:
        root = X.columns[0]
    if root not in X.columns:
        raise ValueError(f"Unknown root variable: {root!r}")

    X = X.copy()
    # Match parameter learning: normalize every external null representation to
    # the stable MISSING state before aligning marginal and joint tables.
    for node in X.columns:
        missing = np.fromiter(
            (is_null(value) for value in X[node]), dtype=bool, count=len(X)
        )
        if missing.any():
            X[node] = X[node].astype(object)
            X.loc[missing, node] = [
                normalize_missing(value) for value in X.loc[missing, node]
            ]

    # Compute the mutual information between each pair of variables
    marginals = {v: X[v].value_counts(normalize=True, dropna=False) for v in X.columns}
    edge = collections.namedtuple("edge", ["u", "v", "mi"])
    mis = (
        edge(
            u,
            v,
            mutual_info(
                puv=X.groupby([u, v], dropna=False, observed=True).size() / len(X),
                pu=marginals[u], pv=marginals[v],
            ),
        )
        for u, v in itertools.combinations(sorted(X.columns), 2)
    )
    edges = ((e.u, e.v) for e in sorted(mis, key=lambda e: e.mi, reverse=True))

    # Extract the maximum spanning tree
    neighbors = kruskal(vertices=X.columns, edges=edges)

    return list(orient_tree(neighbors, root, visited=set()))


def mutual_info(puv, pu, pv):
    """Return the mutual information between variables u and v."""

    # We first align pu and pv with puv so that we can vectorise the MI computation
    # TODO: maybe there's a faster way to align pu and pv with respect to puv
    pu = pu.reindex(puv.index.get_level_values(pu.index.name)).values
    pv = pv.reindex(puv.index.get_level_values(pv.index.name)).values

    return (puv * np.log(puv / (pv * pu))).sum()


class DisjointSet:
    """Disjoint-set data structure.

    References
    ----------

    1. Tarjan, R.E. and Van Leeuwen, J., 1984. Worst-case analysis of set union algorithms.
        Journal of the ACM (JACM), 31(2), pp.245-281.
    2. https://www.wikiwand.com/en/Disjoint-set_data_structure

    """

    def __init__(self, *values):
        self.parents = {x: x for x in values}
        self.sizes = {x: 1 for x in values}

    def find(self, x):
        while self.parents[x] != x:
            x, self.parents[x] = self.parents[x], self.parents[self.parents[x]]
        return x

    def union(self, x, y):
        if self.sizes[x] < self.sizes[y]:
            x, y = y, x
        self.parents[y] = x
        self.sizes[x] += self.sizes[y]


def kruskal(vertices, edges):
    """Find the Maximum Spanning Tree of a dense graph using Kruskal's algorithm.

    The provided edges are assumed to be sorted in descending order.

    References
    ----------

    1. Kruskal, J.B., 1956. On the shortest spanning subtree of a graph and the traveling
        salesman problem. Proceedings of the American Mathematical society, 7(1), pp.48-50.

    """

    ds = DisjointSet(*vertices)
    neighbors = collections.defaultdict(set)
    n_edges = 0

    for u, v in edges:
        if ds.find(u) != ds.find(v):
            neighbors[u].add(v)
            neighbors[v].add(u)
            ds.union(ds.find(u), ds.find(v))
            n_edges += 1

        if n_edges == len(vertices) - 1:
            break

    return neighbors


def orient_tree(neighbors, root, visited):
    """Return tree edges that originate from the given root."""

    for neighbor in sorted(neighbors[root] - visited):
        yield root, neighbor
        yield from orient_tree(neighbors, root=neighbor, visited={root})
