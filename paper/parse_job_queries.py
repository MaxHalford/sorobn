"""
Parse the 113 JOB queries to extract per-table filter predicates.

For each query, identifies which tables have local filter predicates
(as opposed to join conditions) and what those predicates are.

This is the foundation for comparing BN estimates against DB optimizer estimates.
"""

import pathlib
import sqlglot
from sqlglot import exp

JOB_DIR = pathlib.Path(__file__).resolve().parent / "job"


def parse_query(sql):
    """Parse a JOB query and extract per-table filter predicates.

    Returns a dict: {alias: {"table": real_table_name, "predicates": [str, ...]}}
    """
    tree = sqlglot.parse_one(sql)

    # Build alias -> table name mapping from the FROM clause
    aliases = {}
    for table in tree.find_all(exp.Table):
        name = table.name
        alias = table.alias or name
        aliases[alias] = name

    # Extract WHERE conditions
    where = tree.find(exp.Where)
    if not where:
        return {}

    # Flatten ANDed conditions
    conditions = []
    def collect_conditions(node):
        if isinstance(node, exp.And):
            collect_conditions(node.left)
            collect_conditions(node.right)
        else:
            conditions.append(node)
    collect_conditions(where.this)

    # Classify each condition as a local filter or a join condition.
    # A local filter references columns from only one table.
    # A join condition references columns from two or more tables.
    table_predicates = {}
    for alias in aliases:
        table_predicates[alias] = {
            "table": aliases[alias],
            "predicates": [],
        }

    for cond in conditions:
        # Find all column references in this condition
        columns = list(cond.find_all(exp.Column))
        tables_referenced = set()
        for col in columns:
            if col.table:
                tables_referenced.add(col.table)

        # If exactly one table is referenced, it's a local filter
        if len(tables_referenced) == 1:
            alias = tables_referenced.pop()
            if alias in table_predicates:
                table_predicates[alias]["predicates"].append(cond.sql())

    # Only return tables that have local predicates
    return {
        alias: info
        for alias, info in table_predicates.items()
        if info["predicates"]
    }


def main():
    query_files = sorted(JOB_DIR.glob("*.sql"))
    query_files = [f for f in query_files if f.stem not in ("schema", "fkindexes")]

    print(f"Parsing {len(query_files)} JOB queries...\n")

    # Collect all predicates by table
    all_table_preds = {}  # table_name -> set of predicate strings
    query_details = []

    for qf in query_files:
        sql = qf.read_text()
        try:
            filters = parse_query(sql)
        except Exception as e:
            print(f"  WARN: {qf.stem}: parse error: {e}")
            continue

        details = {"query": qf.stem, "filters": {}}
        for alias, info in filters.items():
            table = info["table"]
            preds = info["predicates"]
            details["filters"][table] = preds

            if table not in all_table_preds:
                all_table_preds[table] = set()
            for p in preds:
                all_table_preds[table].add(p)

        query_details.append(details)

    # Summary
    print(f"{'Table':<20} {'#Queries with predicates':>25} {'#Distinct predicates':>22}")
    print("-" * 70)

    # Count how many queries have predicates on each table
    table_query_counts = {}
    for detail in query_details:
        for table in detail["filters"]:
            table_query_counts[table] = table_query_counts.get(table, 0) + 1

    for table in sorted(all_table_preds, key=lambda t: -table_query_counts.get(t, 0)):
        n_queries = table_query_counts.get(table, 0)
        n_preds = len(all_table_preds[table])
        print(f"{table:<20} {n_queries:>25} {n_preds:>22}")

    # Detail: show predicates per table
    print("\n\nPredicates per table:")
    print("=" * 70)
    for table in sorted(all_table_preds, key=lambda t: -table_query_counts.get(t, 0)):
        print(f"\n{table} ({table_query_counts[table]} queries):")
        for pred in sorted(all_table_preds[table]):
            print(f"  {pred}")

    return query_details, all_table_preds


if __name__ == "__main__":
    main()
