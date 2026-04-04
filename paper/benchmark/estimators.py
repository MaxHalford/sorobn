"""
Cardinality estimation methods.

Each estimator takes a query's filter predicates and returns an estimated
row count.
"""

import json
import re
import subprocess

import numpy as np
import pandas as pd


def q_error(estimated, actual):
    """Q-error: max(est/act, act/est). Always >= 1."""
    if actual == 0 or estimated == 0:
        return float("inf")
    return max(estimated / actual, actual / estimated)


def pg_standalone_estimate(db, sql_where, table_clause=""):
    """Get PostgreSQL's estimate for a standalone query.

    Args:
        db: database name
        sql_where: the WHERE clause (without WHERE keyword)
        table_clause: FROM clause (without FROM keyword)
    """
    if not table_clause:
        return None
    full_sql = f"EXPLAIN (FORMAT JSON) SELECT * FROM {table_clause} WHERE {sql_where}"
    try:
        r = subprocess.run(
            ["psql", db, "-t", "-A", "-c", full_sql],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return None
        plan = json.loads(r.stdout)[0]["Plan"]
        return plan.get("Plan Rows", None)
    except Exception:
        return None


def independence_estimate(df, predicates_by_alias, alias_to_table, n_total):
    """Estimate cardinality assuming independence between all columns.

    Computes per-predicate selectivity on the actual data and multiplies.
    """
    sel = 1.0
    for alias, preds in predicates_by_alias.items():
        for pred_str in preds:
            s = _eval_predicate_selectivity(df, pred_str, alias)
            if s is not None:
                sel *= s
    return max(1, round(sel * n_total))


def bn_single_table_estimate(models, df, predicates_by_alias,
                              alias_to_table, n_total):
    """Estimate using single-table BNs.

    For predicates on tables that have a BN, use the BN's FJD.
    For others, fall back to data selectivity.
    """
    bn_prob = 1.0
    non_bn_sel = 1.0

    for alias, preds in predicates_by_alias.items():
        table = alias_to_table.get(alias)
        if table and table in models:
            model = models[table]
            # Try to evaluate predicates against the FJD
            for pred_str in preds:
                col_match = re.search(rf'{re.escape(alias)}\.(\w+)', pred_str)
                if col_match:
                    col = col_match.group(1)
                    if col in model.fjd_df.columns:
                        s = _eval_predicate_on_fjd(model.fjd, model.fjd_df,
                                                    pred_str, alias, col)
                        if s is not None:
                            bn_prob *= s
                            continue
            # Fallback for predicates not in BN
            for pred_str in preds:
                s = _eval_predicate_selectivity(df, pred_str, alias)
                if s is not None:
                    non_bn_sel *= s
        else:
            for pred_str in preds:
                s = _eval_predicate_selectivity(df, pred_str, alias)
                if s is not None:
                    non_bn_sel *= s

    return max(1, round(bn_prob * non_bn_sel * n_total))


def bn_join_estimate(join_models, single_models, df,
                     predicates_by_alias, alias_to_table, n_total):
    """Estimate using multi-table (join) BNs where available.

    Checks if a join BN covers the queried tables; if so, uses it.
    Falls back to single-table BNs, then to data selectivity.
    """
    # Determine which tables have predicates
    pred_tables = set()
    for alias in predicates_by_alias:
        t = alias_to_table.get(alias)
        if t:
            pred_tables.add(t)

    # Find the best-covering join BN
    best_model = None
    best_coverage = 0
    for key, model in join_models.items():
        coverage = len(pred_tables & model.tables)
        if coverage > best_coverage:
            best_coverage = coverage
            best_model = model

    if best_model and best_coverage >= 2:
        # Use the join BN for covered tables
        covered = best_model.tables & pred_tables
        bn_prob = 1.0
        non_bn_sel = 1.0

        for alias, preds in predicates_by_alias.items():
            table = alias_to_table.get(alias)
            if table in covered:
                for pred_str in preds:
                    col_match = re.search(rf'{re.escape(alias)}\.(\w+)', pred_str)
                    if col_match:
                        col = col_match.group(1)
                        bn_col = f"{table}_{col}"
                        if bn_col in best_model.fjd_df.columns:
                            s = _eval_predicate_on_fjd(
                                best_model.fjd, best_model.fjd_df,
                                pred_str, alias, bn_col,
                            )
                            if s is not None:
                                bn_prob *= s
                                continue
                    # Fallback
                    s = _eval_predicate_selectivity(df, pred_str, alias)
                    if s is not None:
                        non_bn_sel *= s
            else:
                for pred_str in preds:
                    s = _eval_predicate_selectivity(df, pred_str, alias)
                    if s is not None:
                        non_bn_sel *= s

        return max(1, round(bn_prob * non_bn_sel * n_total))
    else:
        # No join BN available, fall back to single-table BNs
        return bn_single_table_estimate(
            single_models, df, predicates_by_alias, alias_to_table, n_total
        )


# ---------------------------------------------------------------------------
# Predicate evaluation helpers
# ---------------------------------------------------------------------------

def _eval_predicate_selectivity(df, pred_str, alias):
    """Evaluate a predicate string on a DataFrame, return selectivity (0-1)."""
    mask = _eval_predicate_mask(df, pred_str, alias)
    if mask is None:
        return None
    return mask.mean()


def _eval_predicate_mask(df, pred_str, alias):
    """Evaluate a predicate string on a DataFrame, return boolean mask."""
    # Pattern: alias.col OP value
    m = re.match(
        rf"{re.escape(alias)}\.(\w+)\s*(>=|<=|<>|!=|>|<|=)\s*'?([^']*?)'?\s*$",
        pred_str.strip(),
    )
    if m:
        col, op, val = m.group(1), m.group(2), m.group(3)
        if col not in df.columns:
            return None
        # Try numeric
        try:
            val_num = float(val)
            series = pd.to_numeric(df[col], errors="coerce")
        except ValueError:
            val_num = None
            series = df[col]

        if val_num is not None:
            if op == "=":  return series == val_num
            if op == ">":  return series > val_num
            if op == ">=": return series >= val_num
            if op == "<":  return series < val_num
            if op == "<=": return series <= val_num
            if op in ("<>", "!="): return series != val_num
        else:
            if op == "=":  return series == val
            if op in ("<>", "!="): return series != val

    # Pattern: alias.col BETWEEN val1 AND val2
    m = re.match(
        rf"{re.escape(alias)}\.(\w+)\s+BETWEEN\s+'?([^']+?)'?\s+AND\s+'?([^']+?)'?$",
        pred_str.strip(), re.IGNORECASE,
    )
    if m:
        col, lo, hi = m.group(1), m.group(2), m.group(3)
        if col not in df.columns:
            return None
        try:
            lo_num, hi_num = float(lo), float(hi)
            series = pd.to_numeric(df[col], errors="coerce")
            return series.between(lo_num, hi_num)
        except ValueError:
            return df[col].between(lo, hi)

    # Pattern: alias.col LIKE 'pattern'
    m = re.match(
        rf"{re.escape(alias)}\.(\w+)\s+(?:~~|LIKE)\s+'([^']+)'",
        pred_str.strip(), re.IGNORECASE,
    )
    if m:
        col, pattern = m.group(1), m.group(2)
        if col not in df.columns:
            return None
        regex = pattern.replace("%", ".*").replace("_", ".")
        return df[col].astype(str).str.match(f"^{regex}$", case=False)

    # Timestamp: alias.col OP 'timestamp'::timestamp
    m = re.match(
        rf"{re.escape(alias)}\.(\w+)\s*(>=|<=|>|<|=)\s*'([^']+)'::timestamp",
        pred_str.strip(),
    )
    if m:
        col, op, val = m.group(1), m.group(2), m.group(3)
        if col not in df.columns:
            return None
        series = pd.to_datetime(df[col], errors="coerce")
        val_ts = pd.Timestamp(val)
        if op == "=":  return series == val_ts
        if op == ">":  return series > val_ts
        if op == ">=": return series >= val_ts
        if op == "<":  return series < val_ts
        if op == "<=": return series <= val_ts

    return None


def _eval_predicate_on_fjd(fjd, fjd_df, pred_str, alias, col):
    """Evaluate a predicate against the FJD, return probability mass."""
    mask = _eval_predicate_mask(fjd_df, pred_str, alias)
    if mask is None:
        # Try with column directly (for join BNs where col is prefixed)
        # Replace alias.X with the BN column name
        modified = re.sub(rf'{re.escape(alias)}\.(\w+)', col, pred_str)
        mask = _eval_predicate_mask(fjd_df, modified, col.split(".")[0] if "." in col else "")
    if mask is None:
        return None
    return fjd.values[mask.values].sum()
