"""
Compare cardinality estimates: PostgreSQL vs BN vs Independence.

For each title-table scan with a filter in the 113 JOB queries:
1. Get PostgreSQL's plan_rows estimate (from pg_estimates.json)
2. Get true cardinality (via COUNT(*) on PostgreSQL)
3. Compute BN estimate (from full joint distribution)
4. Compute independence estimate

Focuses on production_year and episode_nr predicates (which the BN can model).
LIKE predicates on title text are excluded since a BN can't model string patterns.
"""

import json
import re
import subprocess
import sys
import time
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import sorobn

PAPER_DIR = pathlib.Path(__file__).resolve().parent
PG_ESTIMATES_PATH = PAPER_DIR / "pg_estimates.json"
RESULTS_PATH = PAPER_DIR / "job_comparison.csv"


# ---------------------------------------------------------------------------
# 1. Load data and build BN
# ---------------------------------------------------------------------------

def load_title_data():
    """Load title table columns from PostgreSQL."""
    result = subprocess.run(
        ["psql", "imdb", "-t", "-A", "-F", ",", "-c",
         "SELECT kind_id, production_year, episode_nr FROM title"],
        capture_output=True, text=True, timeout=30,
    )
    rows = [line.split(",") for line in result.stdout.strip().split("\n") if line]
    records = []
    for row in rows:
        try:
            kind_id = int(row[0]) if row[0] else None
            prod_year = int(row[1]) if row[1] else None
            ep_nr = int(row[2]) if row[2] else None
            records.append({"kind_id": kind_id, "production_year": prod_year,
                           "episode_nr": ep_nr})
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(records)


def build_title_bn(df):
    """Build a BN for the title table."""
    # Structure: production_year -> kind_id, kind_id -> episode_nr
    # (episodes have episode numbers; movies don't)
    bn = sorobn.BayesNet(
        ("production_year", "kind_id"),
        ("kind_id", "episode_nr"),
        seed=42,
    )

    # Fill NaN episode_nr with a sentinel for "no episode number"
    df_clean = df.copy()
    df_clean["episode_nr"] = df_clean["episode_nr"].fillna(-1).astype(int)
    df_clean = df_clean.dropna(subset=["kind_id", "production_year"])
    df_clean["kind_id"] = df_clean["kind_id"].astype(int)
    df_clean["production_year"] = df_clean["production_year"].astype(int)

    bn.fit(df_clean)
    return bn, df_clean


# ---------------------------------------------------------------------------
# 2. Parse filter predicates into evaluable conditions
# ---------------------------------------------------------------------------

def parse_title_filter(pg_filter):
    """Parse a PostgreSQL filter string into a list of (column, op, value) triples.

    Returns None if the filter contains LIKE/pattern predicates we can't model.
    """
    if not pg_filter:
        return None

    # Skip filters with LIKE, ~~, title patterns
    if "~~" in pg_filter or "LIKE" in pg_filter or "title" in pg_filter:
        # But if there are also numeric predicates, extract those
        pass

    conditions = []

    # Match patterns like: production_year > 2010, production_year >= 2005,
    # production_year <= 2010, production_year = 1998, episode_nr >= 50, etc.
    for match in re.finditer(
        r'(production_year|episode_nr)\s*(>=|<=|>|<|=)\s*(\d+)', pg_filter
    ):
        col, op, val = match.group(1), match.group(2), int(match.group(3))
        conditions.append((col, op, val))

    return conditions if conditions else None


def evaluate_conditions(df, conditions):
    """Evaluate a list of (col, op, val) conditions on a DataFrame, return mask."""
    mask = pd.Series(True, index=df.index)
    for col, op, val in conditions:
        if op == "=":
            mask &= df[col] == val
        elif op == ">":
            mask &= df[col] > val
        elif op == ">=":
            mask &= df[col] >= val
        elif op == "<":
            mask &= df[col] < val
        elif op == "<=":
            mask &= df[col] <= val
    return mask


def independence_estimate(df, conditions, n_total):
    """Product of individual column selectivities."""
    sel = 1.0
    for col, op, val in conditions:
        if op == "=":
            s = (df[col] == val).mean()
        elif op == ">":
            s = (df[col] > val).mean()
        elif op == ">=":
            s = (df[col] >= val).mean()
        elif op == "<":
            s = (df[col] < val).mean()
        elif op == "<=":
            s = (df[col] <= val).mean()
        else:
            continue
        sel *= s
    return max(1, round(sel * n_total))


def bn_exact_estimate(fjd, fjd_df, conditions, n_total):
    """Estimate from the BN's full joint distribution."""
    mask = evaluate_conditions(fjd_df, conditions)
    prob = fjd.values[mask.values].sum()
    return max(1, round(prob * n_total))


def q_error(estimated, actual):
    if actual == 0 or estimated == 0:
        return float("inf")
    return max(estimated / actual, actual / estimated)


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("JOB Benchmark: PostgreSQL vs BN Cardinality Estimates (title table)")
    print("=" * 80)
    print()

    # Load PG estimates
    with open(PG_ESTIMATES_PATH) as f:
        pg_data = json.load(f)
    print(f"Loaded PG estimates for {len(pg_data)} queries")

    # Load title data and build BN
    print("Loading title data from PostgreSQL...", end=" ", flush=True)
    t0 = time.time()
    df_raw = load_title_data()
    print(f"{len(df_raw):,} rows in {time.time()-t0:.1f}s")

    print("Building BN...", end=" ", flush=True)
    t0 = time.time()
    bn, df = build_title_bn(df_raw)
    print(f"{time.time()-t0:.2f}s")

    n_total = len(df)
    print(f"Training rows: {n_total:,}")

    # Compute FJD once
    print("Computing full joint distribution...", end=" ", flush=True)
    t0 = time.time()
    fjd = bn.full_joint_dist()
    fjd_df = fjd.reset_index()
    print(f"{time.time()-t0:.2f}s ({len(fjd):,} entries)")
    print()

    # Process all title scans
    results = []
    skipped_like = 0
    skipped_no_filter = 0

    for query_name in sorted(pg_data.keys()):
        info = pg_data[query_name]
        for scan in info["scans"]:
            if scan["relation"] != "title":
                continue
            if not scan["filter"]:
                skipped_no_filter += 1
                continue

            conditions = parse_title_filter(scan["filter"])
            if conditions is None:
                skipped_like += 1
                continue

            # PostgreSQL estimate
            pg_est = scan["plan_rows"]

            # True cardinality
            true_card = int(evaluate_conditions(df, conditions).sum())
            if true_card == 0:
                continue

            # Independence estimate
            indep_est = independence_estimate(df, conditions, n_total)

            # BN exact estimate
            bn_est = bn_exact_estimate(fjd, fjd_df, conditions, n_total)

            results.append({
                "query": query_name,
                "filter": scan["filter"],
                "n_conditions": len(conditions),
                "columns": "+".join(sorted(set(c[0] for c in conditions))),
                "true": true_card,
                "pg_est": pg_est,
                "indep_est": indep_est,
                "bn_est": bn_est,
                "qe_pg": q_error(pg_est, true_card),
                "qe_indep": q_error(indep_est, true_card),
                "qe_bn": q_error(bn_est, true_card),
            })

    print(f"Processed {len(results)} title scans with numeric predicates")
    print(f"Skipped: {skipped_like} with LIKE/pattern-only, {skipped_no_filter} with no filter")
    print()

    if not results:
        print("No results!")
        return

    results_df = pd.DataFrame(results)

    # Print per-query results
    print(f"{'Query':<8} {'Filter':<55} {'True':>8} {'PG':>8} {'Indep':>8} {'BN':>8} | "
          f"{'qe-PG':>6} {'qe-I':>6} {'qe-BN':>6}")
    print("-" * 130)

    for _, r in results_df.iterrows():
        filt_short = str(r["filter"])[:53]
        print(f"{r['query']:<8} {filt_short:<55} {r['true']:>8,} {r['pg_est']:>8,} "
              f"{r['indep_est']:>8,} {r['bn_est']:>8,} | "
              f"{r['qe_pg']:>6.1f} {r['qe_indep']:>6.2f} {r['qe_bn']:>6.2f}")

    # Summary by predicate type
    print()
    print("=" * 80)
    print("Geometric Mean Q-error")
    print("=" * 80)

    def gmean(s):
        v = s.replace([np.inf], np.nan).dropna()
        return np.exp(np.log(v).mean()) if len(v) else None

    # By column combination
    print(f"\n{'Columns':<30} {'PG':>8} {'Indep':>8} {'BN':>8}  {'n':>4}")
    print("-" * 65)
    for cols in sorted(results_df["columns"].unique()):
        sub = results_df[results_df["columns"] == cols]
        gp = gmean(sub["qe_pg"])
        gi = gmean(sub["qe_indep"])
        gb = gmean(sub["qe_bn"])
        gp_s = f"{gp:>8.2f}" if gp else "     N/A"
        gi_s = f"{gi:>8.2f}" if gi else "     N/A"
        gb_s = f"{gb:>8.2f}" if gb else "     N/A"
        print(f"{cols:<30} {gp_s} {gi_s} {gb_s}  {len(sub):>4}")

    # Overall
    gp = gmean(results_df["qe_pg"])
    gi = gmean(results_df["qe_indep"])
    gb = gmean(results_df["qe_bn"])
    print("-" * 65)
    print(f"{'OVERALL':<30} {gp:>8.2f} {gi:>8.2f} {gb:>8.2f}  {len(results_df):>4}")

    # By number of conditions (1 = single column, 2+ = correlated)
    print(f"\n{'# Conditions':<30} {'PG':>8} {'Indep':>8} {'BN':>8}  {'n':>4}")
    print("-" * 65)
    for nc in sorted(results_df["n_conditions"].unique()):
        sub = results_df[results_df["n_conditions"] == nc]
        gp = gmean(sub["qe_pg"])
        gi = gmean(sub["qe_indep"])
        gb = gmean(sub["qe_bn"])
        print(f"{f'{nc} condition(s)':<30} {gp:>8.2f} {gi:>8.2f} {gb:>8.2f}  {len(sub):>4}")

    # Save
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
