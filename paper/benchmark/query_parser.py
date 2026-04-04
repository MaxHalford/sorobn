"""
Parse benchmark queries to extract table references and per-table predicates.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ParsedQuery:
    """A parsed benchmark query."""
    name: str
    sql: str
    true_cardinality: int | None = None
    tables: dict = field(default_factory=dict)  # alias -> table_name
    join_conditions: list = field(default_factory=list)
    filter_predicates: dict = field(default_factory=dict)  # alias -> [predicate_str, ...]


def parse_stats_queries(path):
    """Parse STATS-CEB query file. Format: true_card||SQL per line."""
    queries = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("--"):
                continue
            parts = line.split("||", 1)
            if len(parts) != 2:
                continue
            true_card = int(parts[0])
            sql = parts[1].rstrip(";")

            pq = _parse_sql(sql, name=f"q{i+1:03d}")
            pq.true_cardinality = true_card
            queries.append(pq)
    return queries


def parse_job_queries(job_dir):
    """Parse all JOB .sql files."""
    import pathlib
    queries = []
    for qf in sorted(pathlib.Path(job_dir).glob("*.sql")):
        if qf.stem in ("schema", "fkindexes"):
            continue
        sql = qf.read_text().strip().rstrip(";")
        pq = _parse_sql(sql, name=qf.stem)
        queries.append(pq)
    return queries


def _parse_sql(sql, name=""):
    """Parse a SQL query to extract tables, joins, and filter predicates."""
    pq = ParsedQuery(name=name, sql=sql)

    # Extract table aliases from FROM clause
    # Handles: "table AS alias" and "table alias"
    for m in re.finditer(
        r'\b(\w+)\s+(?:AS\s+)?(\w+)\b(?=\s*(?:,|WHERE|JOIN|$))', sql, re.IGNORECASE
    ):
        table, alias = m.group(1).lower(), m.group(2).lower()
        # Skip SQL keywords
        if table in ("select", "from", "where", "and", "or", "not", "in",
                      "between", "like", "is", "null", "count", "min", "max",
                      "avg", "sum", "as", "join", "on", "inner", "left",
                      "right", "outer", "cross", "natural", "group", "order",
                      "by", "having", "limit", "offset", "union", "exists"):
            continue
        pq.tables[alias] = table

    # Extract WHERE clause
    where_match = re.search(r'\bWHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|$)',
                            sql, re.IGNORECASE | re.DOTALL)
    if not where_match:
        return pq

    where_clause = where_match.group(1).strip()

    # Split on AND (simple split — doesn't handle nested OR correctly but
    # works for the flat AND structure of STATS-CEB and most JOB queries)
    conditions = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)

    for cond in conditions:
        cond = cond.strip()
        if not cond:
            continue

        # Determine which aliases are referenced
        aliases_in_cond = set()
        for alias in pq.tables:
            if re.search(rf'\b{re.escape(alias)}\.\w+', cond):
                aliases_in_cond.add(alias)

        if len(aliases_in_cond) == 2:
            # Join condition
            pq.join_conditions.append(cond)
        elif len(aliases_in_cond) == 1:
            # Filter predicate on a single table
            alias = aliases_in_cond.pop()
            pq.filter_predicates.setdefault(alias, []).append(cond)
        elif len(aliases_in_cond) == 0:
            # Might reference unaliased table — try to match column names
            pass

    return pq
