"""
Main benchmark orchestrator.

Runs the STATS-CEB benchmark comparing:
  1. PostgreSQL estimates
  2. Independence assumption
  3. Single-table BN (one Chow-Liu BN per table)
  4. Multi-table BN (one Chow-Liu BN per join combination)

Usage:
    python -m paper.benchmark.run
"""

import sys
import time
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from paper.benchmark.data_loader import setup_stats_db, load_table, run_psql
from paper.benchmark.query_parser import parse_stats_queries
from paper.benchmark.bn_estimator import build_single_table_bns, build_join_bns
from paper.benchmark.estimators import (
    q_error, pg_standalone_estimate, independence_estimate,
    bn_single_table_estimate, bn_join_estimate,
    _eval_predicate_selectivity,
)

PAPER_DIR = pathlib.Path(__file__).resolve().parents[1]
STATS_QUERIES = PAPER_DIR / "stats-ceb" / "workloads" / "stats_CEB" / "stats_CEB.sql"
RESULTS_PATH = PAPER_DIR / "stats_results.csv"


def main():
    print("=" * 80)
    print("STATS-CEB Benchmark: PostgreSQL vs Independence vs BN")
    print("=" * 80)
    print()

    # --- Setup database ---
    print("Setting up database...")
    setup_stats_db()
    print()

    # --- Parse queries ---
    print("Parsing queries...")
    queries = parse_stats_queries(STATS_QUERIES)
    print(f"  {len(queries)} queries parsed")

    # Analyze which tables/columns have predicates
    table_columns = {}  # table -> set of columns
    for q in queries:
        for alias, preds in q.filter_predicates.items():
            table = q.tables.get(alias)
            if not table:
                continue
            if table not in table_columns:
                table_columns[table] = set()
            for p in preds:
                import re
                for m in re.finditer(rf'{re.escape(alias)}\.(\w+)', p):
                    table_columns[table].add(m.group(1))

    print("\n  Tables with predicates:")
    for table, cols in sorted(table_columns.items(),
                               key=lambda x: -len(x[1])):
        n_queries = sum(1 for q in queries
                       for a in q.filter_predicates
                       if q.tables.get(a) == table)
        print(f"    {table:<15} {len(cols):>2} columns, {n_queries:>3} query predicates: "
              f"{sorted(cols)}")

    # --- Load table data for selectivity evaluation ---
    print("\nLoading table data...")
    all_data = {}
    for table, cols in table_columns.items():
        all_cols = list(cols | {"Id"})  # always include Id
        # Get all columns that exist in the table
        existing = run_psql("stats",
            f"SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{table.lower()}'")
        existing_cols = set(existing.split("\n"))
        valid_cols = [c for c in all_cols if c.lower() in {e.lower() for e in existing_cols}]
        if not valid_cols:
            continue
        df = load_table("stats", table.lower(), valid_cols)
        for col in valid_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        all_data[table] = df
        print(f"  {table}: {len(df):,} rows, columns={valid_cols}")

    # --- Build single-table BNs ---
    print("\nBuilding single-table BNs...")
    # Only use columns that appear in predicates
    single_models = build_single_table_bns("stats", table_columns, {})

    # --- Build multi-table (join) BNs ---
    print("\nBuilding multi-table BNs...")
    join_models = build_join_bns("stats", queries)

    # --- Get table row counts ---
    table_counts = {}
    for table in table_columns:
        n = int(run_psql("stats", f"SELECT COUNT(*) FROM {table.lower()}"))
        table_counts[table] = n

    # --- Run estimation ---
    print(f"\nRunning {len(queries)} queries...\n")
    results = []

    for q in queries:
        if not q.filter_predicates:
            continue

        # Determine the primary table (largest one with predicates)
        primary_alias = max(
            q.filter_predicates.keys(),
            key=lambda a: table_counts.get(q.tables.get(a, ""), 0),
        )
        primary_table = q.tables.get(primary_alias, "")

        # Build the FROM clause for PG estimate
        from_tables = ", ".join(
            f"{q.tables[a]} AS {a}" for a in q.tables
        )
        # Combine all conditions (joins + filters)
        all_conds = list(q.join_conditions)
        for alias, preds in q.filter_predicates.items():
            all_conds.extend(preds)
        where_clause = " AND ".join(all_conds)

        # 1. True cardinality (from the query file)
        true_card = q.true_cardinality or 0
        if true_card == 0:
            continue

        # 2. PostgreSQL estimate
        pg_est = pg_standalone_estimate("stats", where_clause, from_tables)

        # 3. Independence estimate
        # Merge all table data for this query
        merged_df = None
        for alias in q.filter_predicates:
            table = q.tables.get(alias)
            if table and table in all_data:
                df = all_data[table].copy()
                # Rename columns with alias prefix
                df = df.rename(columns={c: f"{alias}.{c}" if "." not in c else c
                                       for c in df.columns})
                # For independence, we don't need to actually join —
                # we just need per-column selectivities
                if merged_df is None:
                    merged_df = df
                else:
                    # Just concatenate for column access
                    # (independence doesn't care about row alignment)
                    for col in df.columns:
                        merged_df[col] = df[col].values[:len(merged_df)] if len(df) >= len(merged_df) else None

        n_total = true_card  # for scaling
        # Actually, independence should use the full join size
        # Approximate: product of table sizes / join selectivity
        # For simplicity, use the true cardinality as the base
        # (this matches what BayesCard and other papers do)

        # Compute per-predicate selectivities on actual data
        indep_sel = 1.0
        for alias, preds in q.filter_predicates.items():
            table = q.tables.get(alias)
            if table and table in all_data:
                table_df = all_data[table]
                for pred_str in preds:
                    # Rename alias to match column names
                    s = _eval_predicate_selectivity(table_df, pred_str, alias)
                    if s is not None:
                        indep_sel *= s

        # Scale by total rows of the base join
        # For proper comparison, we use the same base as PG
        base_rows = 1
        for alias in q.tables:
            table = q.tables[alias]
            if table in table_counts:
                base_rows = max(base_rows, table_counts[table])
        indep_est = max(1, round(indep_sel * base_rows))

        # 4. BN single-table estimate
        bn_single_est = None
        if single_models:
            bn_sel = 1.0
            has_bn = False
            for alias, preds in q.filter_predicates.items():
                table = q.tables.get(alias)
                if table in single_models:
                    has_bn = True
                    model = single_models[table]
                    for pred_str in preds:
                        import re
                        col_match = re.search(rf'{re.escape(alias)}\.(\w+)', pred_str)
                        if col_match and col_match.group(1) in model.fjd_df.columns:
                            from paper.benchmark.estimators import _eval_predicate_on_fjd
                            s = _eval_predicate_on_fjd(
                                model.fjd, model.fjd_df, pred_str, alias,
                                col_match.group(1))
                            if s is not None:
                                bn_sel *= s
                                continue
                        # Fallback to data
                        if table in all_data:
                            s = _eval_predicate_selectivity(all_data[table], pred_str, alias)
                            if s is not None:
                                bn_sel *= s
                elif table in all_data:
                    for pred_str in preds:
                        s = _eval_predicate_selectivity(all_data[table], pred_str, alias)
                        if s is not None:
                            bn_sel *= s
            if has_bn:
                bn_single_est = max(1, round(bn_sel * base_rows))

        # 5. BN join estimate (placeholder — uses single-table for now)
        bn_join_est = bn_single_est  # TODO: use join_models

        # Compute Q-errors
        qe_pg = q_error(pg_est, true_card) if pg_est else None
        qe_indep = q_error(indep_est, true_card)
        qe_bn_s = q_error(bn_single_est, true_card) if bn_single_est else None
        qe_bn_j = q_error(bn_join_est, true_card) if bn_join_est else None

        n_tables = len(q.filter_predicates)
        n_preds = sum(len(p) for p in q.filter_predicates.values())

        results.append({
            "query": q.name,
            "n_tables": n_tables,
            "n_preds": n_preds,
            "true": true_card,
            "pg": pg_est,
            "indep": indep_est,
            "bn_single": bn_single_est,
            "bn_join": bn_join_est,
            "qe_pg": qe_pg,
            "qe_indep": qe_indep,
            "qe_bn_single": qe_bn_s,
            "qe_bn_join": qe_bn_j,
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No results!")
        return

    # --- Print results ---
    print(f"{'Query':<8} {'#T':>2} {'#P':>2} {'True':>10} {'PG':>10} {'Indep':>10} "
          f"{'BN-s':>10} | {'qPG':>6} {'qI':>6} {'qBNs':>6}")
    print("-" * 95)
    for _, r in results_df.head(40).iterrows():
        pg_s = f"{r['pg']:>10,.0f}" if pd.notna(r['pg']) else "       N/A"
        bns = f"{r['bn_single']:>10,.0f}" if pd.notna(r['bn_single']) else "       N/A"
        qpg = f"{r['qe_pg']:>6.1f}" if pd.notna(r['qe_pg']) else "   N/A"
        qbns = f"{r['qe_bn_single']:>6.1f}" if pd.notna(r['qe_bn_single']) else "   N/A"
        print(f"{r['query']:<8} {r['n_tables']:>2} {r['n_preds']:>2} {r['true']:>10,} "
              f"{pg_s} {r['indep']:>10,} {bns} | "
              f"{qpg} {r['qe_indep']:>6.1f} {qbns}")
    if len(results_df) > 40:
        print(f"  ... ({len(results_df) - 40} more rows)")

    # --- Summary ---
    def gmean(s):
        v = s.replace([np.inf, -np.inf], np.nan).dropna()
        return np.exp(np.log(v).mean()) if len(v) else None

    print()
    print("=" * 60)
    print("Geometric Mean Q-error")
    print("=" * 60)
    print(f"{'Group':<20} {'PG':>8} {'Indep':>8} {'BN-single':>10}  {'n':>4}")
    print("-" * 55)

    for nt in sorted(results_df["n_tables"].unique()):
        sub = results_df[results_df["n_tables"] == nt]
        gp = gmean(sub["qe_pg"])
        gi = gmean(sub["qe_indep"])
        gs = gmean(sub["qe_bn_single"])
        gp_s = f"{gp:>8.2f}" if gp else "     N/A"
        gs_s = f"{gs:>10.2f}" if gs else "       N/A"
        print(f"{f'{nt}-table predicates':<20} {gp_s} {gi:>8.2f} {gs_s}  {len(sub):>4}")

    gp = gmean(results_df["qe_pg"])
    gi = gmean(results_df["qe_indep"])
    gs = gmean(results_df["qe_bn_single"])
    gp_s = f"{gp:>8.2f}" if gp else "     N/A"
    gs_s = f"{gs:>10.2f}" if gs else "       N/A"
    print("-" * 55)
    print(f"{'OVERALL':<20} {gp_s} {gi:>8.2f} {gs_s}  {len(results_df):>4}")

    # Save
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
