"""Predicates on a single variable, including SQL-style three-valued null logic."""

import fnmatch
import math
import operator
import re
from dataclasses import dataclass

import pandas as pd

__all__ = [
    "Predicate", "Eq", "Ne", "Lt", "Le", "Gt", "Ge", "In", "Between",
    "Like", "Regex", "Glob", "IsNull", "IsNotNull",
]


def is_null(value):
    missing = pd.isna(value)
    return bool(missing) if pd.api.types.is_scalar(missing) else False


def as_predicate(value):
    return value if isinstance(value, Predicate) else Eq(value)


class Predicate:
    """A condition on one variable. Combine conditions with ``&``, ``|``, and ``~``.

    Calling a predicate returns whether a value matches. Internally, ``None`` denotes
    SQL UNKNOWN, so negating a comparison does not make null values match.

    Examples
    --------
    >>> from sorobn import Ge, Lt
    >>> adult = Ge(18) & Lt(65)
    >>> adult(30)
    True
    >>> adult(70)
    False
    >>> (~adult)(None)
    False
    """

    def evaluate(self, value):
        raise NotImplementedError

    def __call__(self, value):
        result = self.evaluate(value)
        return result is not None and bool(result)

    def __and__(self, other):
        return _Combined(self, as_predicate(other), "and")

    def __or__(self, other):
        return _Combined(self, as_predicate(other), "or")

    def __invert__(self):
        return _Negated(self)

    def __bool__(self):
        raise TypeError("Combine predicates with &, |, and ~, not and, or, and not")

    def _boundaries(self):
        raise TypeError(f"{type(self).__name__} cannot be used on a discretized variable")

    def _interval_weight(self, left, right):
        # Splitting at every comparison boundary also handles overlapping unions and
        # intersections correctly: integrate the combined predicate, not its weights.
        boundaries = self._boundaries()
        if left == right:
            return float(self(left))
        cuts = [left, *sorted({x for x in boundaries if left < x < right}), right]
        return sum(
            (b - a) / (right - left)
            for a, b in zip(cuts, cuts[1:])
            if self(a + (b - a) / 2)
        )


@dataclass(frozen=True)
class _Comparison(Predicate):
    """Shared comparison behavior, including SQL UNKNOWN for missing operands.

    Examples
    --------
    >>> Eq(1).evaluate(None) is None
    True
    """

    value: object

    def evaluate(self, value):
        if is_null(value) or is_null(self.value):
            return None
        return bool(self._operator(value, self.value))

    def _boundaries(self):
        if is_null(self.value):
            return []
        return [float(self.value)]


class Eq(_Comparison):
    """Equality (ordinary values in an event are also treated as equality).

    Examples
    --------
    >>> from sorobn import Eq
    >>> Eq("movie")("movie")
    True
    >>> Eq("movie")("series")
    False
    """

    _operator = staticmethod(operator.eq)


class Ne(_Comparison):
    """Inequality; null values do not match.

    Examples
    --------
    >>> from sorobn import Ne
    >>> Ne("movie")("series")
    True
    >>> Ne("movie")(None)
    False
    """

    _operator = staticmethod(operator.ne)


class Lt(_Comparison):
    """Strictly less than.

    Examples
    --------
    >>> from sorobn import Lt
    >>> Lt(2000)(1999)
    True
    >>> Lt(2000)(2000)
    False
    """

    _operator = staticmethod(operator.lt)


class Le(_Comparison):
    """Less than or equal to.

    Examples
    --------
    >>> from sorobn import Le
    >>> Le(2000)(2000)
    True
    >>> Le(2000)(2001)
    False
    """

    _operator = staticmethod(operator.le)


class Gt(_Comparison):
    """Strictly greater than.

    Examples
    --------
    >>> from sorobn import Gt
    >>> Gt(2000)(2001)
    True
    >>> Gt(2000)(2000)
    False
    """

    _operator = staticmethod(operator.gt)


class Ge(_Comparison):
    """Greater than or equal to.

    Examples
    --------
    >>> from sorobn import Ge
    >>> Ge(2000)(2000)
    True
    >>> Ge(2000)(1999)
    False
    """

    _operator = staticmethod(operator.ge)


@dataclass(frozen=True)
class In(Predicate):
    """Membership in a collection of values, with SQL IN semantics for nulls.

    Examples
    --------
    >>> from sorobn import In
    >>> In(["movie", "series"])("movie")
    True
    >>> (~In(["movie", "series"]))("short")
    True
    >>> (~In(["movie", None]))("short")
    False
    """

    values: tuple

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(self.values))

    def evaluate(self, value):
        unknown = is_null(value)
        for candidate in self.values:
            match = Eq(candidate).evaluate(value)
            if match:
                return True
            unknown |= match is None
        return None if unknown else False

    def _boundaries(self):
        return [float(x) for x in self.values if not is_null(x)]


@dataclass(frozen=True)
class Between(Predicate):
    """An inclusive range. Use comparisons for open or half-open ranges.

    Examples
    --------
    >>> from sorobn import Between
    >>> Between(1990, 2000)(2000)
    True
    >>> Between(1990, 2000)(2001)
    False
    """

    lower: float = -math.inf
    upper: float = math.inf

    def evaluate(self, value):
        return (Ge(self.lower) & Le(self.upper)).evaluate(value)

    def _boundaries(self):
        return [float(self.lower), float(self.upper)]


class Regex(Predicate):
    """Python regular-expression search (use anchors for a full-string match).

    Examples
    --------
    >>> from sorobn import Regex
    >>> Regex(r"[0-9]{4}")("Released in 1999")
    True
    >>> Regex(r"^[0-9]{4}$")("Released in 1999")
    False
    """

    def __init__(self, pattern, flags=0):
        self.pattern = pattern
        self.flags = flags
        self._compiled = re.compile(pattern, flags)

    def evaluate(self, value):
        if is_null(value):
            return None
        return isinstance(value, str) and self._compiled.search(value) is not None


class Glob(Regex):
    """Case-sensitive, full-string glob matching: *, ?, and character classes.

    Examples
    --------
    >>> from sorobn import Glob
    >>> Glob("Star *")("Star Wars")
    True
    >>> Glob("Star *")("A Star Is Born")
    False
    """

    def __init__(self, pattern):
        super().__init__(r"\A" + fnmatch.translate(pattern))
        self.pattern = pattern


class Like(Regex):
    """SQL LIKE: % matches any string and _ matches one character.

    Matches the entire string, including newlines. ``escape`` defaults to backslash;
    use an empty string to disable escaping. ``case_sensitive=False`` provides
    case-insensitive matching using Python's Unicode regular-expression rules.

    Examples
    --------
    >>> from sorobn import Like
    >>> Like("%Star%")("A Star Is Born")
    True
    >>> Like("Star _")("Star Wars")
    False
    >>> Like("star%", case_sensitive=False)("Star Wars")
    True
    """

    def __init__(self, pattern, escape="\\", case_sensitive=True):
        if len(escape) > 1:
            raise ValueError("escape must be empty or a single character")
        parts = []
        chars = iter(pattern)
        for char in chars:
            if char == escape:
                literal = next(chars, None)
                if literal is None:
                    raise ValueError("LIKE pattern ends with an escape character")
                parts.append(re.escape(literal))
            else:
                parts.append({"%": ".*", "_": "."}.get(char, re.escape(char)))
        super().__init__(
            r"\A" + "".join(parts) + r"\Z",
            re.DOTALL | (0 if case_sensitive else re.IGNORECASE),
        )
        self.pattern = pattern


class IsNull(Predicate):
    """Match None, NaN, pandas.NA, and NaT.

    Examples
    --------
    >>> from sorobn import IsNull
    >>> IsNull()(None)
    True
    >>> IsNull()(float("nan"))
    True
    >>> IsNull()("")
    False
    """

    def evaluate(self, value):
        return is_null(value)

    def _boundaries(self):
        return []


class IsNotNull(IsNull):
    """Match non-null values.

    Examples
    --------
    >>> from sorobn import IsNotNull
    >>> IsNotNull()(None)
    False
    >>> IsNotNull()("")
    True
    """

    def evaluate(self, value):
        return not is_null(value)


@dataclass(frozen=True)
class _Combined(Predicate):
    """A conjunction or disjunction of predicates on the same variable.

    Examples
    --------
    >>> (Lt(0) | Gt(10))(15)
    True
    """

    left: Predicate
    right: Predicate
    operation: str

    def evaluate(self, value):
        left, right = self.left.evaluate(value), self.right.evaluate(value)
        if self.operation == "and":
            if left is False or right is False:
                return False
            return None if left is None or right is None else True
        if left is True or right is True:
            return True
        return None if left is None or right is None else False

    def _boundaries(self):
        return [*self.left._boundaries(), *self.right._boundaries()]


@dataclass(frozen=True)
class _Negated(Predicate):
    """Negation preserving SQL UNKNOWN.

    Examples
    --------
    >>> (~Eq(1))(2)
    True
    >>> (~Eq(1))(None)
    False
    """
    predicate: Predicate

    def evaluate(self, value):
        result = self.predicate.evaluate(value)
        return None if result is None else not result

    def _boundaries(self):
        return self.predicate._boundaries()
