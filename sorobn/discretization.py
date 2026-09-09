"""Finite histograms with uniform density within each bin."""

import numpy as np
import pandas as pd

from .predicates import as_predicate, is_null

__all__ = ["Discretizer"]


class Discretizer:
    """Learn equal-width or quantile bins, or use explicit finite edges.

    Bins include their left edge; the final bin also includes its right edge.
    Duplicate quantile edges are removed. A constant column is represented by a
    single point mass. Nulls remain a separate state. Values outside fitted edges
    are rejected during transformation, rather than silently clipped or dropped.

    Examples
    --------
    >>> from sorobn import Discretizer
    >>> width = Discretizer(n_bins=2, strategy="uniform").fit([0, 1, 2, 10])
    >>> width.edges_.tolist()
    [0.0, 5.0, 10.0]
    >>> frequency = Discretizer(n_bins=2, strategy="quantile").fit([0, 1, 2, 10])
    >>> frequency.edges_.tolist()
    [0.0, 1.5, 10.0]
    >>> binned = width.transform([0, 5, 10])
    >>> binned
    0     [0.0, 5.0)
    1    [5.0, 10.0]
    2    [5.0, 10.0]
    dtype: category
    Categories (2, object): [[0.0, 5.0) < [5.0, 10.0]]
    >>> binned.cat.codes.tolist()
    [0, 1, 1]
    >>> custom = Discretizer(edges=[0, 2, 10]).fit([0, 1, 2, 10])
    >>> custom.transform([1, 3]).tolist()
    [Interval(0.0, 2.0, closed='left'), Interval(2.0, 10.0, closed='both')]
    """

    def __init__(self, n_bins=10, strategy="quantile", *, edges=None):
        if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
            raise ValueError("n_bins must be a positive integer")
        if strategy not in ("uniform", "quantile"):
            raise ValueError("strategy must be 'uniform' or 'quantile'")
        self.n_bins = n_bins
        self.strategy = strategy
        self.edges = None if edges is None else tuple(edges)
        self.edges_ = None
        self.dtype_ = None
        if self.edges is not None:
            values = np.asarray(self.edges, dtype=float)
            if (
                values.ndim != 1 or len(values) < 2
                or not np.isfinite(values).all() or not (np.diff(values) > 0).all()
            ):
                raise ValueError("edges must be finite and strictly increasing")

    @staticmethod
    def _as_numeric_series(values):
        series = pd.Series(values)
        dtype = series.dtype
        if (
            not pd.api.types.is_any_real_numeric_dtype(dtype)
            and not series.isna().all()
        ):
            raise TypeError(
                f"Discretizer requires a numeric column; got dtype {dtype!r}"
            )
        return series

    def fit(self, values):
        values = self._as_numeric_series(values).to_numpy(
            dtype=float, na_value=np.nan
        )
        observed = values[~np.isnan(values)]
        if not np.isfinite(observed).all():
            raise ValueError("Discretization requires finite values or nulls")
        if self.edges is not None:
            edges = np.asarray(self.edges, dtype=float)
        elif not len(observed):
            raise ValueError("Cannot learn bins from an empty or all-null column; supply edges")
        elif self.strategy == "uniform":
            edges = np.linspace(observed.min(), observed.max(), self.n_bins + 1)
        else:
            edges = np.quantile(observed, np.linspace(0, 1, self.n_bins + 1))
        edges = np.unique(edges)
        if len(edges) == 1:
            edges = np.repeat(edges, 2)
        if len(observed) and (observed.min() < edges[0] or observed.max() > edges[-1]):
            raise ValueError("Values fall outside discretization edges")
        self.edges_ = edges
        # Mixed closure keeps every endpoint exact: no padding or rounding of
        # labels is needed to include the maximum (or a constant point mass).
        # Pandas stores these native Interval scalars once, in the category index.
        categories = pd.Index([
            pd.Interval(left, right, closed="both" if i == len(edges) - 2 else "left")
            for i, (left, right) in enumerate(zip(edges, edges[1:]))
        ], dtype=object)
        self.dtype_ = pd.CategoricalDtype(categories=categories, ordered=True)
        return self

    def transform(self, values):
        """Return ordered interval categoricals, preserving the index and nulls.

        All fitted bins remain in ``.cat.categories``, including unobserved bins.
        ``.cat.codes`` exposes the compact integer representation; -1 denotes null.
        """
        if self.edges_ is None:
            raise ValueError("Fit the discretizer before transforming values")
        series = self._as_numeric_series(values)
        values = series.to_numpy(dtype=float, na_value=np.nan)
        missing = np.isnan(values)
        if (
            (~np.isfinite(values) & ~missing).any()
            or (values < self.edges_[0]).any() or (values > self.edges_[-1]).any()
        ):
            raise ValueError("Values fall outside discretization edges")
        bins = np.searchsorted(self.edges_, values, side="right") - 1
        bins = np.minimum(bins, len(self.edges_) - 2)
        bins[missing] = -1
        binned = pd.Categorical.from_codes(bins, dtype=self.dtype_)
        return pd.Series(binned, index=series.index, name=series.name)

    def fit_transform(self, values):
        return self.fit(values).transform(values)

    def weights(self, predicate, states):
        """Return the fraction of each bin selected by a predicate.

        Range probabilities interpolate the CDF linearly within bins. Equality
        has zero mass in a nondegenerate bin. Interval endpoints retain their full
        precision independently of pandas' display formatting.
        """
        if self.edges_ is None:
            raise ValueError("Fit the discretizer before querying it")
        predicate = as_predicate(predicate)
        predicate._boundaries()  # Validate even when all states happen to be null.
        return np.asarray([
            float(predicate(state)) if is_null(state) else predicate._interval_weight(
                state.left, state.right
            )
            for state in states
        ])
