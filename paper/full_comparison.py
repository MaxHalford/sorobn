"""
Full JOB benchmark comparison: PostgreSQL vs BN vs Independence.

Evaluates ALL title-table predicates from the 113 JOB queries,
including LIKE patterns on the title column.

For numeric columns (production_year, episode_nr, kind_id), uses a BN.
For string LIKE patterns, evaluates directly on the data.
"""

import json
import subprocess
import sys
import time
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import sorobn
from paper.predicates import parse_pg_filter, evaluate_all, evaluate_condition

PAPER_DIR = pathlib.Path(__file__).resolve().parent
PG_ESTIMATES_PATH = PAPER_DIR / "pg_estimates.json"
RESULTS_PATH = PAPER_DIR / "full_comparison.csv"


def load_title():
    """Load title table from PostgreSQL with all columns used in JOB predicates."""
    cols = "id, title, kind_id, production_year, episode_nr"
    r = subprocess.run(
        ["psql", "imdb", "-t", "-A", "-F", "\t", "-c",
         f"SELECT {cols} FROM title"],
        capture_output=True, text=True, timeout=60,
    )
    rows = []
    for line in r.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        rows.append(parts)

    df = pd.DataFrame(rows, columns=["id", "title", "kind_id",
                                      "production_year", "episode_nr"])
    # Convert types
    for col in ["id", "kind_id", "production_year", "episode_nr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_bn(df):
    """Build BN on title columns using Chow-Liu tree structure learning."""
    from sorobn.structure import chow_liu

    bn_cols = ["kind_id", "production_year", "episode_nr"]
    df_bn = df[bn_cols].copy()
    # Bin episode_nr into buckets for tractable BN (raw values too many)
    df_bn["episode_nr"] = pd.cut(
        df_bn["episode_nr"], bins=[-1, 0, 5, 50, 100, 500, float("inf")],
        labels=["null", "1-5", "6-50", "51-100", "101-500", "500+"],
    ).astype(str)
    df_bn["episode_nr"] = df_bn["episode_nr"].fillna("null")
    df_bn = df_bn.dropna(subset=["kind_id", "production_year"])
    df_bn["kind_id"] = df_bn["kind_id"].astype(int)
    df_bn["production_year"] = df_bn["production_year"].astype(int)

    # Learn structure automatically via Chow-Liu tree
    structure = chow_liu(df_bn)
    print(f"  Chow-Liu structure: {structure}")

    bn = sorobn.BayesNet(*structure, seed=42)
    bn.fit(df_bn)
    return bn, df_bn


def get_pg_standalone_estimate(where_clause):
    """Get PostgreSQL's standalone estimate for a WHERE clause on title."""
    r = subprocess.run(
        ["psql", "imdb", "-t", "-A", "-c",
         f"EXPLAIN (FORMAT JSON) SELECT * FROM title WHERE {where_clause}"],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0:
        return None
    try:
        plan = json.loads(r.stdout)[0]["Plan"]
        return plan["Plan Rows"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def q_error(est, actual):
    if actual == 0 or est == 0:
        return float("inf")
    return max(est / actual, actual / est)


def main():
    print("=" * 80)
    print("Full JOB Comparison: PostgreSQL vs BN vs Independence (title table)")
    print("=" * 80)
    print()

    # Load data
    print("Loading title table from PostgreSQL...", end=" ", flush=True)
    t0 = time.time()
    df = load_title()
    print(f"{len(df):,} rows in {time.time()-t0:.1f}s")

    # Build BN
    print("Building BN...", end=" ", flush=True)
    t0 = time.time()
    bn, df_bn = build_bn(df)
    print(f"{time.time()-t0:.2f}s, {len(df_bn):,} training rows")

    # Compute FJD
    print("Computing full joint distribution...", end=" ", flush=True)
    t0 = time.time()
    fjd = bn.full_joint_dist()
    fjd_df = fjd.reset_index()
    print(f"{time.time()-t0:.2f}s ({len(fjd):,} entries)")

    n_total = len(df)
    n_bn = len(df_bn)

    # Load PG plan estimates
    with open(PG_ESTIMATES_PATH) as f:
        pg_data = json.load(f)

    # Collect unique title filters from JOB queries
    seen_filters = set()
    scan_entries = []
    for query_name in sorted(pg_data.keys()):
        for scan in pg_data[query_name]["scans"]:
            if scan["relation"] != "title" or not scan["filter"]:
                continue
            filt = scan["filter"]
            if filt in seen_filters:
                continue
            seen_filters.add(filt)
            scan_entries.append({
                "query": query_name,
                "pg_filter": filt,
                "pg_plan_rows": scan["plan_rows"],
            })

    print(f"\n{len(scan_entries)} unique title filters from JOB queries\n")

    # Process each filter
    results = []
    for entry in scan_entries:
        pg_filter = entry["pg_filter"]
        conditions = parse_pg_filter(pg_filter)
        if not conditions:
            continue

        # True cardinality: evaluate all conditions on the full DataFrame
        mask = evaluate_all(df, conditions)
        if mask is None:
            continue
        true_card = int(mask.sum())
        if true_card == 0:
            continue

        # Build a SQL WHERE clause for PostgreSQL standalone estimate
        # (we need to reconstruct it from the parsed conditions)
        # Simpler: just use the PG filter directly (strip the PG-specific syntax)
        pg_where = pg_filter
        pg_where = pg_where.replace("::text[]", "").replace("::text", "")
        pg_where = pg_where.replace("::character varying", "")
        pg_where = pg_where.replace("~~", "LIKE")
        pg_standalone = get_pg_standalone_estimate(pg_where)

        # Independence estimate: multiply individual condition selectivities
        indep_sel = 1.0
        for cond in conditions:
            m = evaluate_condition(df, cond)
            if m is not None:
                indep_sel *= m.mean()
        indep_est = max(1, round(indep_sel * n_total))

        # BN estimate: use FJD for numeric conditions, multiply by LIKE
        # selectivity from data (hybrid approach).
        from paper.predicates import Like, Neq, IsNull, Eq, Cmp, Between, In
        bn_conds = [c for c in conditions
                    if not isinstance(c, (Like,))
                    and c.col in fjd_df.columns]
        like_conds = [c for c in conditions if isinstance(c, Like)]
        other_conds = [c for c in conditions
                       if not isinstance(c, Like)
                       and c.col not in fjd_df.columns]

        # BN probability for numeric conditions
        if bn_conds:
            # Map episode_nr numeric conditions to bin labels
            bn_fjd_conds = []
            for c in bn_conds:
                if c.col == "episode_nr":
                    # episode_nr is binned in the BN; evaluate on raw data instead
                    like_conds.append(c)  # treat as "non-BN" condition
                else:
                    bn_fjd_conds.append(c)

            if bn_fjd_conds:
                bn_mask = evaluate_all(fjd_df, bn_fjd_conds)
                bn_prob = fjd.values[bn_mask.values].sum() if bn_mask is not None else 1.0
            else:
                bn_prob = 1.0
        else:
            bn_prob = 1.0

        # Multiply by selectivity of non-BN conditions (LIKE, episode_nr, etc.)
        non_bn_conds = like_conds + other_conds
        for c in non_bn_conds:
            m = evaluate_condition(df, c)
            if m is not None:
                bn_prob *= m.mean()

        bn_est = max(1, round(bn_prob * n_total))

        # Classify the filter
        has_like = "~~" in pg_filter or "LIKE" in pg_filter
        has_numeric = any(c in pg_filter for c in ["production_year", "episode_nr"])
        if has_like and has_numeric:
            group = "mixed"
        elif has_like:
            group = "like_only"
        elif has_numeric:
            group = "numeric"
        else:
            group = "other"

        n_conds = len(conditions)

        pg_s_str = pg_standalone if pg_standalone else None
        qe_pg = q_error(pg_standalone, true_card) if pg_standalone else None
        qe_ind = q_error(indep_est, true_card)
        qe_bn = q_error(bn_est, true_card) if bn_est else None

        results.append({
            "filter": pg_filter[:80],
            "group": group,
            "n_conds": n_conds,
            "true": true_card,
            "pg_standalone": pg_s_str,
            "indep": indep_est,
            "bn": bn_est,
            "qe_pg": qe_pg,
            "qe_indep": qe_ind,
            "qe_bn": qe_bn,
        })

    results_df = pd.DataFrame(results)

    # Print results
    print(f"{'Filter':<60} {'Grp':<6} {'True':>9} {'PG':>9} {'Indep':>9} {'BN':>9} | {'qPG':>5} {'qI':>5} {'qBN':>5}")
    print("-" * 135)
    for _, r in results_df.iterrows():
        pg_s = f"{r['pg_standalone']:>9,}" if r['pg_standalone'] else "      N/A"
        bn_s = f"{r['bn']:>9,}" if r['bn'] else "      N/A"
        qpg = f"{r['qe_pg']:>5.2f}" if r['qe_pg'] else "  N/A"
        qbn = f"{r['qe_bn']:>5.2f}" if r['qe_bn'] else "  N/A"
        print(f"{str(r['filter'])[:58]:<60} {r['group']:<6} {r['true']:>9,} {pg_s} {r['indep']:>9,} {bn_s} | {qpg} {r['qe_indep']:>5.2f} {qbn}")

    # Summary
    print()
    print("=" * 80)
    print("Geometric Mean Q-error by Group")
    print("=" * 80)

    def gmean(s):
        v = s.replace([np.inf], np.nan).dropna()
        return np.exp(np.log(v).mean()) if len(v) else None

    print(f"{'Group':<12} {'PG':>8} {'Indep':>8} {'BN':>8}  {'n':>4}")
    print("-" * 48)
    for group in ["numeric", "like_only", "mixed"]:
        sub = results_df[results_df["group"] == group]
        if len(sub) == 0:
            continue
        gp = gmean(sub["qe_pg"])
        gi = gmean(sub["qe_indep"])
        gb = gmean(sub["qe_bn"])
        gp_s = f"{gp:>8.2f}" if gp else "     N/A"
        gi_s = f"{gi:>8.2f}" if gi else "     N/A"
        gb_s = f"{gb:>8.2f}" if gb else "     N/A"
        print(f"{group:<12} {gp_s} {gi_s} {gb_s}  {len(sub):>4}")

    gp = gmean(results_df["qe_pg"])
    gi = gmean(results_df["qe_indep"])
    gb = gmean(results_df["qe_bn"])
    gp_s = f"{gp:>8.2f}" if gp else "     N/A"
    gb_s = f"{gb:>8.2f}" if gb else "     N/A"
    print("-" * 48)
    print(f"{'OVERALL':<12} {gp_s} {gi:>8.2f} {gb_s}  {len(results_df):>4}")

    # Save
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
