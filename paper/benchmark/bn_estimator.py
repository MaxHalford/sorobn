"""
Bayesian network cardinality estimator.

Two modes:
  1. Single-table BN: one Chow-Liu BN per table
  2. Multi-table BN: one Chow-Liu BN per join combination that appears in queries

The multi-table mode pre-joins the relevant tables, builds a BN over the
combined columns, and uses it to estimate the joint selectivity of predicates
that span multiple tables.
"""

import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BNModel:
    """A trained BN model for cardinality estimation."""
    name: str
    tables: set  # which tables this BN covers
    bn: object  # sorobn.BayesNet
    fjd: pd.Series  # full joint distribution
    fjd_df: pd.DataFrame  # FJD as DataFrame for predicate evaluation
    n_rows: int  # number of rows in the training data
    columns: list  # BN column names
    structure: list  # Chow-Liu edges


def discretize_column(series, max_bins=50):
    """Discretize a column into bins if it has too many distinct values."""
    n_distinct = series.nunique()
    if n_distinct <= max_bins:
        return series  # already low cardinality

    # Only bin numeric columns
    if not pd.api.types.is_numeric_dtype(series):
        # For strings with too many values, take the top N most frequent
        top = series.value_counts().head(max_bins - 1).index
        return series.where(series.isin(top), other="__other__")

    try:
        binned = pd.qcut(series, q=max_bins, duplicates="drop")
        return binned.astype(str)
    except (ValueError, TypeError):
        try:
            binned = pd.cut(series, bins=max_bins, duplicates="drop")
            return binned.astype(str)
        except (ValueError, TypeError):
            return series.astype(str)


def build_single_table_bns(db, tables_with_predicates, predicate_columns):
    """Build one Chow-Liu BN per table.

    Args:
        db: database name
        tables_with_predicates: dict of {table_name: [column_names]}
        predicate_columns: dict of {table_name: {col_name: col_type}}

    Returns:
        dict of {table_name: BNModel}
    """
    import sorobn
    from sorobn.structure import chow_liu
    from .data_loader import load_table

    models = {}
    for table, columns in tables_with_predicates.items():
        columns = sorted(columns)  # ensure list, not set
        if len(columns) < 2:
            continue  # need at least 2 columns for a BN

        print(f"  Building BN for {table} ({len(columns)} columns)...", end=" ")
        t0 = time.time()

        df = load_table(db, table.lower(), columns)
        if df.empty:
            print("EMPTY")
            continue

        # Convert types — handle timestamps and numerics
        df_bn = df.copy()
        for col in columns:
            numeric = pd.to_numeric(df_bn[col], errors="coerce")
            if numeric.notna().mean() > 0.5:
                df_bn[col] = numeric
            else:
                # Try timestamp -> epoch day for discretization
                ts = pd.to_datetime(df_bn[col], errors="coerce")
                if ts.notna().mean() > 0.5:
                    # Convert to year-month bucket
                    df_bn[col] = ts.dt.to_period("M").astype(str)
                # else keep as string
        df_bn = df_bn.dropna()

        # Discretize high-cardinality columns
        for col in columns:
            df_bn[col] = discretize_column(df_bn[col])

        if len(df_bn) < 10:
            print(f"too few rows ({len(df_bn)})")
            continue

        # Learn structure
        try:
            structure = chow_liu(df_bn)
        except Exception as e:
            print(f"chow_liu failed: {e}")
            continue

        # Build and fit BN
        bn = sorobn.BayesNet(*structure, seed=42)
        bn.fit(df_bn)

        fjd = bn.full_joint_dist()
        fjd_df = fjd.reset_index()

        models[table] = BNModel(
            name=f"bn_{table}",
            tables={table},
            bn=bn,
            fjd=fjd,
            fjd_df=fjd_df,
            n_rows=len(df_bn),
            columns=list(columns),
            structure=structure,
        )
        print(f"{time.time()-t0:.2f}s, {len(fjd):,} FJD entries, "
              f"structure={structure}")

    return models


def build_join_bns(db, queries, max_columns=6, max_fjd_size=500_000):
    """Build one Chow-Liu BN per unique join combination that appears in queries.

    Only builds BNs for join combinations where:
    1. The join involves 2+ tables with filter predicates
    2. The combined column count is manageable (≤ max_columns)
    3. The expected FJD size is tractable (≤ max_fjd_size)

    Args:
        db: database name
        queries: list of ParsedQuery objects
        max_columns: max columns per BN
        max_fjd_size: max FJD entries before we skip

    Returns:
        dict of {frozenset(table_names): BNModel}
    """
    import sorobn
    from sorobn.structure import chow_liu
    from .data_loader import load_joined

    # Discover which table combinations appear with multi-table predicates
    join_combos = {}  # frozenset(tables) -> {join_conds, filter_cols}
    for q in queries:
        if len(q.filter_predicates) < 2:
            continue  # single-table or no predicates

        # Collect tables that have filter predicates
        pred_tables = set()
        pred_cols = {}  # alias -> [col_names]
        for alias, preds in q.filter_predicates.items():
            table = q.tables.get(alias)
            if table:
                pred_tables.add(table)
                cols = set()
                for p in preds:
                    for m in re.finditer(rf'{re.escape(alias)}\.(\w+)', p):
                        cols.add(m.group(1))
                pred_cols[table] = pred_cols.get(table, set()) | cols

        if len(pred_tables) < 2:
            continue

        key = frozenset(pred_tables)
        if key not in join_combos:
            join_combos[key] = {
                "tables": pred_tables,
                "columns": {},
                "join_conds": [],
                "query_count": 0,
            }
        for t, cols in pred_cols.items():
            join_combos[key]["columns"][t] = (
                join_combos[key]["columns"].get(t, set()) | cols
            )
        join_combos[key]["join_conds"].extend(q.join_conditions)
        join_combos[key]["query_count"] += 1

    print(f"\n  Found {len(join_combos)} unique join combinations with multi-table predicates")

    # Build BNs for the most common combinations
    models = {}
    for key, info in sorted(join_combos.items(),
                            key=lambda x: -x[1]["query_count"]):
        tables = info["tables"]
        all_cols = info["columns"]
        total_cols = sum(len(c) for c in all_cols.values())

        if total_cols > max_columns:
            print(f"  Skipping {tables}: {total_cols} columns > {max_columns}")
            continue
        if total_cols < 2:
            continue

        print(f"  Building join BN for {tables} "
              f"({total_cols} cols, {info['query_count']} queries)...",
              end=" ", flush=True)

        # Build the JOIN SQL
        # We need to figure out the join conditions from the queries
        join_conds = list(set(info["join_conds"]))
        if not join_conds:
            print("no join conditions found")
            continue

        # Build SELECT with prefixed column names
        select_parts = []
        for table, cols in all_cols.items():
            for col in sorted(cols):
                select_parts.append(f"{table}.{col} AS {table}_{col}")

        # Build FROM with joins
        table_list = sorted(tables)
        from_clause = ", ".join(table_list)
        where_clause = " AND ".join(join_conds)

        sql = f"SELECT {', '.join(select_parts)} FROM {from_clause} WHERE {where_clause}"

        t0 = time.time()
        try:
            df = load_joined(db, sql, timeout=120)
        except Exception as e:
            print(f"query failed: {e}")
            continue

        if df.empty or len(df) < 10:
            print(f"too few rows ({len(df)})")
            continue

        # Convert and discretize
        df_bn = df.copy()
        for col in df_bn.columns:
            df_bn[col] = pd.to_numeric(df_bn[col], errors="coerce")
        df_bn = df_bn.dropna()

        for col in df_bn.columns:
            df_bn[col] = discretize_column(df_bn[col])

        # Check FJD size estimate
        fjd_size_est = 1
        for col in df_bn.columns:
            fjd_size_est *= df_bn[col].nunique()
        if fjd_size_est > max_fjd_size:
            print(f"FJD too large (~{fjd_size_est:,})")
            continue

        # Learn structure and build BN
        try:
            structure = chow_liu(df_bn)
            bn = sorobn.BayesNet(*structure, seed=42)
            bn.fit(df_bn)
            fjd = bn.full_joint_dist()
            fjd_df = fjd.reset_index()
        except Exception as e:
            print(f"BN build failed: {e}")
            continue

        models[key] = BNModel(
            name=f"bn_{'_'.join(sorted(tables))}",
            tables=tables,
            bn=bn,
            fjd=fjd,
            fjd_df=fjd_df,
            n_rows=len(df_bn),
            columns=list(df_bn.columns),
            structure=structure,
        )
        print(f"{time.time()-t0:.1f}s, {len(df_bn):,} rows, "
              f"{len(fjd):,} FJD entries")

    return models
