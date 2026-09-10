"""Frequency-based grouping for high-cardinality categorical variables."""

import numpy as np
import pandas as pd

from .predicates import (
    MISSING, Eq, In, Ne, _Combined, _Negated, as_predicate, is_null,
)

__all__ = ["Compactor", "OTHER"]


class _Other:
    """Collision-free label for values grouped by a :class:`Compactor`."""

    def __repr__(self):
        return "<OTHER>"

    def __str__(self):
        return "<OTHER>"

    def __eq__(self, other):
        return isinstance(other, _Other)

    def __hash__(self):
        return hash(_Other)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_get_other, ())


OTHER = _Other()


def _get_other():
    return OTHER


class Compactor:
    """Group infrequent non-null values into :data:`OTHER`.

    Exactly one strategy must be selected. ``max_categories`` places an upper
    bound on the number of non-null model states, including ``OTHER``.
    ``min_frequency`` keeps values occurring at least that many times, or at
    least that fraction of all observations when given a float.

    ``disaggregation`` controls how the probability of ``OTHER`` is assigned
    when a query addresses original bucket members: ``"uniform"`` gives every
    member equal weight, ``"empirical"`` uses their observed counts, and
    ``"dirichlet"`` uses the posterior mean under a symmetric Dirichlet prior
    with concentration ``alpha`` per member.

    Nulls are never grouped: every external null representation becomes the
    explicit :data:`~sorobn.MISSING` state. Frequency ties at a
    ``max_categories`` boundary are resolved by first appearance. The ``OTHER``
    state is always reserved, allowing later calls to :meth:`transform` to
    handle values not seen during fitting without growing the state space.

    Examples
    --------
    >>> from sorobn import Compactor, OTHER
    >>> compactor = Compactor(max_categories=3).fit(["a", "a", "b", "c"])
    >>> compactor.frequent_values_
    ('a', 'b')
    >>> compactor.transform(["a", "c", "new"]).tolist()
    ['a', <OTHER>, <OTHER>]
    >>> compactor.transform_value("new") == OTHER
    True
    """

    def __init__(
        self, *, max_categories=None, min_frequency=None,
        disaggregation="uniform", alpha=1.0,
    ):
        if (max_categories is None) == (min_frequency is None):
            raise ValueError(
                "Specify exactly one of max_categories or min_frequency"
            )
        if max_categories is not None and (
            not isinstance(max_categories, int)
            or isinstance(max_categories, bool)
            or max_categories < 1
        ):
            raise ValueError("max_categories must be a positive integer")
        if min_frequency is not None:
            valid_count = (
                isinstance(min_frequency, int)
                and not isinstance(min_frequency, bool)
                and min_frequency >= 1
            )
            valid_fraction = (
                isinstance(min_frequency, float) and 0 < min_frequency <= 1
            )
            if not (valid_count or valid_fraction):
                raise ValueError(
                    "min_frequency must be a positive integer or a float in (0, 1]"
                )
        if disaggregation not in ("uniform", "empirical", "dirichlet"):
            raise ValueError(
                "disaggregation must be 'uniform', 'empirical', or 'dirichlet'"
            )
        if (
            not isinstance(alpha, (int, float))
            or isinstance(alpha, bool)
            or not np.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError("alpha must be a positive finite number")

        self.max_categories = max_categories
        self.min_frequency = min_frequency
        self.disaggregation = disaggregation
        self.alpha = alpha
        self.frequent_values_ = None
        self.infrequent_values_ = None
        self.infrequent_counts_ = None
        self.dtype_ = None
        self._frequent = None

    def __repr__(self):
        limit = (
            f"max_categories={self.max_categories!r}"
            if self.max_categories is not None
            else f"min_frequency={self.min_frequency!r}"
        )
        options = [limit]
        if self.disaggregation != "uniform":
            options.append(f"disaggregation={self.disaggregation!r}")
        if self.disaggregation == "dirichlet":
            options.append(f"alpha={self.alpha!r}")
        return f"Compactor({', '.join(options)})"

    @staticmethod
    def _as_categorical_series(values):
        series = pd.Series(values)
        dtype = series.dtype
        if not (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            raise TypeError(
                "Compactor requires a categorical column (string, object, boolean, "
                f"or category); got dtype {dtype!r}. Cast numeric labels to "
                "'category' explicitly."
            )
        return series

    def fit(self, values):
        """Learn which values to retain from a one-dimensional collection."""
        series = self._as_categorical_series(values)
        missing = np.fromiter(
            (is_null(value) for value in series), dtype=bool, count=len(series)
        )
        observed = series[~missing]
        if observed.map(lambda value: value == OTHER).any():
            raise ValueError("OTHER is reserved and cannot occur in training data")

        # ``sort=False`` preserves first appearance. The stable frequency sort
        # consequently gives deterministic behavior for ties without requiring
        # heterogeneous values to be mutually orderable.
        counts = observed.groupby(observed, sort=False, observed=True).size()
        ranked = counts.sort_values(ascending=False, kind="stable")

        if self.max_categories is not None:
            kept = ranked.index[: max(0, self.max_categories - 1)]
        else:
            threshold = (
                self.min_frequency * len(series)
                if isinstance(self.min_frequency, float)
                else self.min_frequency
            )
            kept = ranked[ranked >= threshold].index

        self.frequent_values_ = tuple(kept)
        kept_set = set(kept)
        self.infrequent_values_ = tuple(
            value for value in ranked.index if value not in kept_set
        )
        self.infrequent_counts_ = {
            value: int(ranked[value]) for value in self.infrequent_values_
        }
        self._frequent = set(self.frequent_values_)
        self.dtype_ = pd.CategoricalDtype(
            categories=[*self.frequent_values_, OTHER, MISSING], ordered=False
        )
        return self

    def partial_fit(self, values):
        """Update bucket counts without changing the retained categories."""
        if self._frequent is None:
            return self.fit(values)
        series = self._as_categorical_series(values)
        missing = np.fromiter(
            (is_null(value) for value in series), dtype=bool, count=len(series)
        )
        observed = series[~missing]
        counts = observed.groupby(observed, sort=False, observed=True).size()
        additions = []
        for value, count in counts.items():
            if isinstance(value, _Other):
                raise ValueError("OTHER is reserved and cannot occur in training data")
            if value in self._frequent:
                continue
            if value not in self.infrequent_counts_:
                additions.append(value)
                self.infrequent_counts_[value] = 0
            self.infrequent_counts_[value] += int(count)
        self.infrequent_values_ += tuple(additions)
        return self

    def transform_value(self, value):
        """Map one value into the fitted state space."""
        if self._frequent is None:
            raise ValueError("Fit the compactor before transforming values")
        if is_null(value):
            return MISSING
        return value if value in self._frequent else OTHER

    def transform(self, values):
        """Group infrequent and unseen values, preserving the index and nulls."""
        if self._frequent is None:
            raise ValueError("Fit the compactor before transforming values")
        series = self._as_categorical_series(values)
        grouped = [
            MISSING if is_null(value) else self.transform_value(value)
            for value in series
        ]
        return pd.Series(
            pd.Categorical(grouped, dtype=self.dtype_),
            index=series.index,
            name=series.name,
        )

    def fit_transform(self, values):
        return self.fit(values).transform(values)

    def _references_other(self, predicate):
        """Whether a predicate explicitly addresses the aggregate state."""
        if isinstance(predicate, (Eq, Ne)):
            return isinstance(predicate.value, _Other)
        if isinstance(predicate, In):
            return any(isinstance(value, _Other) for value in predicate.values)
        if isinstance(predicate, _Combined):
            return (
                self._references_other(predicate.left)
                or self._references_other(predicate.right)
            )
        if isinstance(predicate, _Negated):
            return self._references_other(predicate.predicate)
        return False

    def weights(self, predicate, states):
        """Return whether each grouped state is selected by a predicate.

        A predicate that explicitly references ``OTHER`` operates on the
        aggregate state. Otherwise, each known bucket member matched by the
        predicate contributes according to the configured disaggregation rule.
        """
        predicate = as_predicate(predicate)
        references_other = self._references_other(predicate)
        matched = [
            value for value in self.infrequent_values_ if predicate(value)
        ]
        if not matched:
            bucket_weight = 0.0
        elif self.disaggregation == "uniform":
            bucket_weight = len(matched) / len(self.infrequent_values_)
        elif self.disaggregation == "empirical":
            bucket_weight = (
                sum(self.infrequent_counts_[value] for value in matched)
                / sum(self.infrequent_counts_.values())
            )
        else:
            bucket_weight = (
                sum(self.infrequent_counts_[value] + self.alpha for value in matched)
                / (
                    sum(self.infrequent_counts_.values())
                    + self.alpha * len(self.infrequent_values_)
                )
            )
        weights = []
        for state in states:
            if not isinstance(state, _Other):
                weights.append(float(predicate(state)))
            elif references_other:
                weights.append(float(predicate(OTHER)))
            else:
                weights.append(bucket_weight)
        return np.asarray(weights)
