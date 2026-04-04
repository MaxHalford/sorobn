"""
Parse PostgreSQL WHERE predicates into evaluable conditions.

Supports:
  - Equality: col = value
  - Comparison: col >, >=, <, <=
  - BETWEEN: col BETWEEN lo AND hi
  - IN: col IN (val1, val2, ...)
  - LIKE/~~: col LIKE 'pattern' (converted to regex)
  - NOT LIKE: NOT col LIKE 'pattern'
  - IS NULL / IS NOT NULL
  - <> (not equal)
  - Boolean combinations: AND, OR, NOT
"""

import re


def sql_like_to_regex(pattern):
    """Convert a SQL LIKE pattern to a Python regex.

    SQL LIKE:
      %  -> .*  (any sequence of characters)
      _  -> .   (any single character)
      Everything else is literal.
    """
    # Remove surrounding quotes if present
    pattern = pattern.strip("'\"")
    # Escape regex special chars (except % and _ which we handle)
    escaped = ""
    for ch in pattern:
        if ch == "%":
            escaped += ".*"
        elif ch == "_":
            escaped += "."
        elif ch in r"\.^$+?{}[]|()":
            escaped += "\\" + ch
        else:
            escaped += ch
    return "^" + escaped + "$"


# Condition types
class Eq:
    def __init__(self, col, val): self.col, self.val = col, val
    def __repr__(self): return f"{self.col} = {self.val!r}"

class Neq:
    def __init__(self, col, val): self.col, self.val = col, val
    def __repr__(self): return f"{self.col} <> {self.val!r}"

class Cmp:
    def __init__(self, col, op, val): self.col, self.op, self.val = col, op, val
    def __repr__(self): return f"{self.col} {self.op} {self.val!r}"

class Between:
    def __init__(self, col, lo, hi): self.col, self.lo, self.hi = col, lo, hi
    def __repr__(self): return f"{self.col} BETWEEN {self.lo!r} AND {self.hi!r}"

class In:
    def __init__(self, col, vals): self.col, self.vals = col, vals
    def __repr__(self): return f"{self.col} IN ({', '.join(repr(v) for v in self.vals)})"

class Like:
    def __init__(self, col, pattern, negated=False):
        self.col, self.pattern, self.negated = col, pattern, negated
        self.regex = re.compile(sql_like_to_regex(pattern), re.IGNORECASE)
    def __repr__(self):
        neg = "NOT " if self.negated else ""
        return f"{neg}{self.col} LIKE '{self.pattern}'"

class IsNull:
    def __init__(self, col, negated=False): self.col, self.negated = col, negated
    def __repr__(self):
        return f"{self.col} IS {'NOT ' if self.negated else ''}NULL"


def parse_pg_filter(pg_filter):
    """Parse a PostgreSQL EXPLAIN filter string into a list of condition objects.

    Handles the PostgreSQL-specific syntax from EXPLAIN output:
      - ~~ for LIKE
      - ::text casts
      - ANY('{...}'::text[]) for IN
    """
    if not pg_filter:
        return []

    # Clean up PostgreSQL-specific syntax
    s = pg_filter
    s = re.sub(r"::text\[\]", "", s)
    s = re.sub(r"::text", "", s)
    s = re.sub(r"::character varying", "", s)
    # Remove casts like (col)
    s = re.sub(r"\((\w+)\)", r"\1", s)

    conditions = []

    # IS NOT NULL: NOT col IS NULL
    for m in re.finditer(r"NOT\s+(\w+(?:\.\w+)?)\s+IS\s+NULL", s):
        conditions.append(IsNull(m.group(1), negated=True))

    # IS NULL
    for m in re.finditer(r"(?<!NOT\s)(\w+(?:\.\w+)?)\s+IS\s+NULL", s):
        conditions.append(IsNull(m.group(1)))

    # BETWEEN
    for m in re.finditer(
        r"\((\w+(?:\.\w+)?)\s*>=\s*(\d+)\)\s*AND\s*\(\1\s*<=\s*(\d+)\)", s
    ):
        col = m.group(1)
        lo, hi = m.group(2), m.group(3)
        # Try int, fall back to string
        try:
            lo, hi = int(lo), int(hi)
        except ValueError:
            pass
        conditions.append(Between(col, lo, hi))

    # IN via ANY('{...}')
    for m in re.finditer(
        r"(\w+(?:\.\w+)?)\s*=\s*ANY\s*\(\s*'\{([^}]+)\}'", s
    ):
        col = m.group(1)
        vals = [v.strip().strip('"').strip("'") for v in m.group(2).split(",")]
        conditions.append(In(col, vals))

    # IN via col IN (...)
    for m in re.finditer(
        r"(\w+(?:\.\w+)?)\s+IN\s*\(\s*([^)]+)\s*\)", s
    ):
        col = m.group(1)
        vals = [v.strip().strip("'\"") for v in m.group(2).split(",")]
        conditions.append(In(col, vals))

    # NOT LIKE / NOT ~~
    for m in re.finditer(
        r"NOT\s+\(?(\w+(?:\.\w+)?)\s*~~\s*'([^']+)'\)?", s
    ):
        conditions.append(Like(m.group(1), m.group(2), negated=True))

    # LIKE / ~~ (positive, skip already-matched NOT ~~)
    for m in re.finditer(r"(?<!NOT\s)\(?(\w+(?:\.\w+)?)\s*~~\s*'([^']+)'\)?", s):
        conditions.append(Like(m.group(1), m.group(2)))

    # Comparison: col >, >=, <, <=, =, <>
    for m in re.finditer(
        r"\(?(\w+(?:\.\w+)?)\s*(>=|<=|<>|>|<|=)\s*'?([^')]+?)'?\)?(?:\s|$|AND|OR)",
        s,
    ):
        col, op, val = m.group(1), m.group(2), m.group(3).strip("'\" ")
        # Skip if already captured as BETWEEN, IN, LIKE, or IS NULL
        if any(
            isinstance(c, (Between, In, Like, IsNull)) and c.col == col
            for c in conditions
        ):
            continue
        # Try numeric conversion
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass

        if op == "=":
            conditions.append(Eq(col, val))
        elif op == "<>":
            conditions.append(Neq(col, val))
        else:
            conditions.append(Cmp(col, op, val))

    return conditions


def evaluate_condition(df, cond):
    """Evaluate a single condition on a DataFrame, return boolean mask."""
    col = cond.col
    if col not in df.columns:
        # Try without table alias prefix
        col_short = col.split(".")[-1] if "." in col else col
        if col_short in df.columns:
            col = col_short
        else:
            # Column not in DataFrame — return all True (can't evaluate)
            return None

    if isinstance(cond, Eq):
        return df[col] == cond.val
    elif isinstance(cond, Neq):
        return df[col] != cond.val
    elif isinstance(cond, Cmp):
        if cond.op == ">":  return df[col] > cond.val
        elif cond.op == ">=": return df[col] >= cond.val
        elif cond.op == "<":  return df[col] < cond.val
        elif cond.op == "<=": return df[col] <= cond.val
    elif isinstance(cond, Between):
        return df[col].between(cond.lo, cond.hi)
    elif isinstance(cond, In):
        return df[col].isin(cond.vals)
    elif isinstance(cond, Like):
        matches = df[col].astype(str).str.match(cond.regex.pattern, case=False)
        return ~matches if cond.negated else matches
    elif isinstance(cond, IsNull):
        is_null = df[col].isna()
        return ~is_null if cond.negated else is_null
    return None


def evaluate_all(df, conditions):
    """Evaluate all conditions on a DataFrame, return combined boolean mask."""
    mask = None
    for cond in conditions:
        m = evaluate_condition(df, cond)
        if m is None:
            continue
        mask = m if mask is None else (mask & m)
    return mask


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test LIKE to regex conversion
    tests = [
        ("%Money%", "test Money here", True),
        ("%Money%", "no cash here", False),
        ("Kung Fu Panda%", "Kung Fu Panda 2", True),
        ("Kung Fu Panda%", "Not Kung Fu Panda", False),
        ("%murder%", "The murder mystery", True),
        ("Saw%", "Saw III", True),
        ("Saw%", "I Saw You", False),
    ]
    for pattern, text, expected in tests:
        regex = sql_like_to_regex(pattern)
        result = bool(re.match(regex, text, re.IGNORECASE))
        status = "OK" if result == expected else "FAIL"
        print(f"  {status}: LIKE '{pattern}' vs '{text}' -> {result} (expected {expected})")

    # Test filter parsing
    print("\nParsing PostgreSQL filters:")
    filters = [
        "(production_year > 2010)",
        "((production_year >= 2005) AND (production_year <= 2008))",
        "((production_year > 2000) AND ((title ~~ '%Freddy%'::text) OR (title ~~ '%Jason%'::text)))",
        "((kind)::text = ANY ('{movie,episode}'::text[]))",
        "(episode_nr >= 50) AND (episode_nr < 100)",
        "((title <> ''::text) AND ((title ~~ '%Champion%'::text) OR (title ~~ '%Loser%'::text)))",
    ]
    for f in filters:
        conds = parse_pg_filter(f)
        print(f"  {f[:70]}")
        for c in conds:
            print(f"    -> {c}")
