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

    ``min_frequency`` keeps values occurring at least that many times, or at
    least that fraction of all observations when given a float.
    ``max_categories`` is a hard cap on the number of non-null model states,
    including ``OTHER``: the most frequent values are kept first. Set either,
    or both to keep frequent values but never more than the cap.

    Only the frequent values and the number of grouped values (``n_members_``)
    are kept, so memory does not grow with the number of distinct values.
    Queries on grouped values use heuristics:

    - ``Eq``, ``In``, and ``Ne`` assume every grouped value is equally frequent:
      each gets ``1 / n_members_`` of ``OTHER``.
    - Other predicates (patterns, ranges) take their share of ``OTHER`` from
      the frequent values: ``(matches + 1) / (len(frequent_values_) + 2)``.

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
    >>> compactor.n_members_
    1
    >>> compactor.transform(["a", "c", "new"]).tolist()
    ['a', <OTHER>, <OTHER>]
    >>> compactor.transform_value("new") == OTHER
    True
    """

    def __init__(self, *, max_categories=None, min_frequency=None):
        if max_categories is None and min_frequency is None:
            raise ValueError(
                "Specify max_categories, min_frequency, or both"
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

        self.max_categories = max_categories
        self.min_frequency = min_frequency
        self.frequent_values_ = None
        self.n_members_ = None
        self.dtype_ = None
        self._frequent = None

    def __repr__(self):
        options = []
        if self.max_categories is not None:
            options.append(f"max_categories={self.max_categories!r}")
        if self.min_frequency is not None:
            options.append(f"min_frequency={self.min_frequency!r}")
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

        kept = ranked.index
        if self.min_frequency is not None:
            threshold = (
                self.min_frequency * len(series)
                if isinstance(self.min_frequency, float)
                else self.min_frequency
            )
            kept = ranked[ranked >= threshold].index
        if self.max_categories is not None:
            kept = kept[: max(0, self.max_categories - 1)]

        self.frequent_values_ = tuple(kept)
        self.n_members_ = len(ranked) - len(kept)
        self._frequent = set(self.frequent_values_)
        self.dtype_ = pd.CategoricalDtype(
            categories=[*self.frequent_values_, OTHER, MISSING], ordered=False
        )
        return self

    def partial_fit(self, values):
        """Fit on the first batch. Later batches change nothing here.

        The retained values and ``n_members_`` stay as learned from the first
        batch, so accumulated counts keep a stable meaning.
        """
        if self._frequent is None:
            return self.fit(values)
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

    def _point_share(self, predicate):
        """Share of OTHER matched by Eq, In, or Ne, or None for other predicates.

        Each non-null value outside the frequent ones counts as one member.
        OTHER has at least one member: values first seen by partial_fit land there.
        """
        if isinstance(predicate, Ne):
            return 1 - self._point_share(Eq(predicate.value))
        if isinstance(predicate, _Negated) and isinstance(predicate.predicate, (Eq, In)):
            return 1 - self._point_share(predicate.predicate)
        if isinstance(predicate, Eq):
            values = (predicate.value,)
        elif isinstance(predicate, In):
            values = predicate.values
        else:
            return None

        n = max(self.n_members_, 1)
        members = {v for v in values if not is_null(v) and v not in self._frequent}
        return min(len(members), n) / n

    def _bucket_share(self, predicate):
        """Share of OTHER selected by a predicate that does not name OTHER.

        e.g. with frequent values ('Paris', 'London'), Glob('L*') matches one
        of two, so it selects (1 + 1) / (2 + 2) = 0.5 of OTHER.
        """
        point = self._point_share(predicate)
        if point is not None:
            return point
        matches = sum(bool(predicate(value)) for value in self.frequent_values_)
        return (matches + 1) / (len(self.frequent_values_) + 2)

    def weights(self, predicate, states):
        """Return how much of each grouped state a predicate selects.

        A predicate that explicitly references ``OTHER`` operates on the
        aggregate state. Otherwise ``OTHER`` gets a heuristic share (see the
        class docstring).
        """
        predicate = as_predicate(predicate)
        references_other = self._references_other(predicate)
        weights = []
        for state in states:
            if not isinstance(state, _Other):
                weights.append(float(predicate(state)))
            elif references_other:
                weights.append(float(predicate(OTHER)))
            else:
                weights.append(self._bucket_share(predicate))
        return np.asarray(weights)
