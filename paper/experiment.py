"""
Bayesian network selectivity estimation on the IMDB/JOB benchmark.

Uses the `title` table from the Join Order Benchmark (Leis et al., 2015),
the standard single-table cardinality estimation target with known
correlations (kind_id <-> production_year).

Compares three cardinality estimation methods:
  1. True cardinality   (pandas groupby)
  2. Independence assumption  (product of per-column selectivities)
  3. BN estimate via path sampling

Usage:
    # First, download the IMDB data:
    mkdir -p paper/data && cd paper/data
    curl -OL https://bonsai.cedardb.com/job/imdb.tgz
    tar -zxf imdb.tgz

    # Then run:
    python paper/experiment.py
"""

import sys
import time
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import sorobn

PAPER_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = PAPER_DIR / "data"
RESULTS_PATH = PAPER_DIR / "results.csv"

# Columns we use — these have the strongest correlations in the title table
COLUMNS = ["kind_id", "production_year"]


# ---------------------------------------------------------------------------
# 1.  Load IMDB title table
# ---------------------------------------------------------------------------

def load_data():
    """Load the title table from JOB CSV and return a pandas DataFrame."""
    csv_path = DATA_DIR / "title.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Download the JOB data first:")
        print("  mkdir -p paper/data && cd paper/data")
        print("  curl -OL https://bonsai.cedardb.com/job/imdb.tgz")
        print("  tar -zxf imdb.tgz")
        sys.exit(1)

    col_names = [
        "id", "title", "imdb_index", "kind_id", "production_year",
        "imdb_id", "phonetic_code", "episode_of_id", "season_nr",
        "episode_nr", "series_years", "md5sum",
    ]

    t0 = time.time()
    df = pd.read_csv(
        csv_path, header=None, names=col_names,
        usecols=COLUMNS, low_memory=False,
    )
    load_time = time.time() - t0

    # Drop nulls and cast
    df = df.dropna(subset=COLUMNS)
    for col in COLUMNS:
        df[col] = df[col].astype(int)
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df):,} titles in {load_time:.2f}s")
    return df


# ---------------------------------------------------------------------------
# 2.  Build the Bayesian network
# ---------------------------------------------------------------------------

def build_bn(df):
    """Build and fit a BN on the title columns."""
    bn = sorobn.BayesNet(
        ("production_year", "kind_id"),
        seed=42,
    )
    t0 = time.time()
    bn.fit(df)
    fit_time = time.time() - t0

    t0 = time.time()
    bn._path_sampler._build()
    build_time = time.time() - t0

    n_tables = len(bn._path_sampler._tables)
    total_rows = sum(len(c) for _, _, c in bn._path_sampler._tables)

    print(f"BN fit: {fit_time:.3f}s, path sampler build: {build_time:.3f}s")
    print(f"  {n_tables} tables, {total_rows:,} total rows")
    return bn


# ---------------------------------------------------------------------------
# 3.  Define queries
# ---------------------------------------------------------------------------

def make_queries(df):
    """Return a list of (name, group, predicate_fn, predicate_desc) tuples.

    predicate_fn takes a DataFrame and returns a boolean mask.
    """
    queries = []

    # --- Group A: Single-column equality ---
    kind_names = {1: "movie", 2: "tv_series", 3: "tv_movie", 4: "video_movie",
                  5: "tv_mini_series", 6: "video_game", 7: "episode"}
    for kid in [1, 2, 6, 7]:
        queries.append((
            f"A: kind={kind_names[kid]}",
            "single",
            lambda d, k=kid: d["kind_id"] == k,
            f"kind_id = {kid}",
        ))

    # --- Group B: Year ranges ---
    for lo, hi in [(1950, 1955), (1970, 1975), (1990, 1995),
                   (2000, 2005), (2005, 2010), (2010, 2013)]:
        queries.append((
            f"B: year {lo}-{hi}",
            "range",
            lambda d, a=lo, b=hi: d["production_year"].between(a, b),
            f"production_year BETWEEN {lo} AND {hi}",
        ))

    # --- Group C: Correlated (kind + year) ---
    # TV episodes barely existed before 2000; movies peak earlier
    corr = [
        ("movie & year<1980",   1, None, 1979),
        ("movie & year>=2000",  1, 2000, None),
        ("episode & year<2000", 7, None, 1999),
        ("episode & year>=2005", 7, 2005, None),
        ("tv_series & year>=2010", 2, 2010, None),
        ("video_game & year<2000", 6, None, 1999),
        ("movie & 1990-1999",   1, 1990, 1999),
        ("episode & 2000-2005", 7, 2000, 2005),
        ("tv_series & 1980-1999", 2, 1980, 1999),
        ("movie & 2010-2013",   1, 2010, 2013),
    ]
    for desc, kid, lo, hi in corr:
        def make_fn(k, lo, hi):
            def fn(d):
                m = d["kind_id"] == k
                if lo is not None:
                    m &= d["production_year"] >= lo
                if hi is not None:
                    m &= d["production_year"] <= hi
                return m
            return fn

        # Build SQL WHERE that matches the DuckDB estimates JSON keys
        sql_parts = [f"kind_id = {kid}"]
        if lo is not None and hi is not None:
            sql_parts.append(f"production_year BETWEEN {lo} AND {hi}")
        elif lo is not None:
            sql_parts.append(f"production_year >= {lo}")
        elif hi is not None:
            sql_parts.append(f"production_year <= {hi}")
        sql_where = " AND ".join(sql_parts)

        queries.append((
            f"C: {desc}",
            "correlated",
            make_fn(kid, lo, hi),
            sql_where,
        ))

    return queries


# ---------------------------------------------------------------------------
# 4.  Estimation methods
# ---------------------------------------------------------------------------

def true_cardinality(df, mask_fn):
    """Exact count."""
    return mask_fn(df).sum()


def independence_estimate(df, mask_fn, n_total, predicate_desc):
    """Estimate cardinality assuming column independence.

    Parse the predicate description to apply each column's selectivity
    independently.
    """
    # Apply the actual predicate to get individual column selectivities
    mask = mask_fn(df)
    # For multi-predicate queries, compute independence estimate by
    # multiplying individual column selectivities.
    # Handle BETWEEN...AND specially before splitting on AND.
    import re
    desc_normalized = re.sub(
        r'BETWEEN\s+(\d+)\s+AND\s+(\d+)', r'BETWEEN_\1_\2', predicate_desc
    )
    parts = desc_normalized.split(" AND ")
    sel = 1.0
    for part in parts:
        part = part.strip()
        if "kind_id" in part and "=" in part:
            kid = int(part.split("=")[1])
            sel *= (df["kind_id"] == kid).mean()
        elif "production_year >=" in part or "year>=" in part:
            yr = int(re.search(r'>=\s*(\d+)', part).group(1))
            sel *= (df["production_year"] >= yr).mean()
        elif "production_year <=" in part or "year<=" in part:
            yr = int(re.search(r'<=\s*(\d+)', part).group(1))
            sel *= (df["production_year"] <= yr).mean()
        elif "BETWEEN_" in part:
            # Was normalized from "BETWEEN X AND Y" to "BETWEEN_X_Y"
            nums = part.split("BETWEEN_")[1].split("_")
            lo, hi = int(nums[0]), int(nums[1])
            sel *= df["production_year"].between(lo, hi).mean()
        elif "kind_id" in part:
            # single equality
            kid = int(part.split("=")[1].strip())
            sel *= (df["kind_id"] == kid).mean()
    return max(1, round(sel * n_total))


def bn_sample_estimate(bn_samples, mask_fn, n_total):
    """Estimate cardinality from pre-generated BN samples."""
    frac = mask_fn(bn_samples).mean()
    return max(1, round(frac * n_total))


def bn_exact_estimate(bn, mask_fn, n_total):
    """Estimate cardinality using the full joint distribution."""
    fjd = bn.full_joint_dist()
    fjd_df = fjd.reset_index()
    frac = fjd[mask_fn(fjd_df)].sum()
    return max(1, round(frac * n_total))


# ---------------------------------------------------------------------------
# 5.  Q-error
# ---------------------------------------------------------------------------

def q_error(estimated, actual):
    if actual == 0:
        return float("inf") if estimated > 0 else 1.0
    if estimated == 0:
        return float("inf")
    return max(estimated / actual, actual / estimated)


# ---------------------------------------------------------------------------
# 6.  Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("JOB Cardinality Estimation: BN vs Independence")
    print("=" * 70)
    print()

    # Load
    df = load_data()
    n_total = len(df)
    print()

    # Build BN
    bn = build_bn(df)
    print()

    # Generate samples
    N_SAMPLES = 500_000
    print(f"Generating {N_SAMPLES:,} BN samples via path sampling...", end=" ", flush=True)
    t0 = time.time()
    bn_samples = bn.sample(n=N_SAMPLES, method="path")
    sample_time = time.time() - t0
    print(f"{sample_time:.2f}s ({N_SAMPLES / sample_time:,.0f} samples/s)")

    # Exact BN joint (precompute once)
    print("Computing exact BN joint distribution...", end=" ", flush=True)
    t0 = time.time()
    fjd = bn.full_joint_dist()
    fjd_df = fjd.reset_index()
    fjd_time = time.time() - t0
    print(f"{fjd_time:.3f}s ({len(fjd):,} entries)")
    print()

    # Load DuckDB estimates (pre-computed using lea's venv)
    import json
    duckdb_path = PAPER_DIR / "duckdb_estimates.json"
    if duckdb_path.exists():
        with open(duckdb_path) as f:
            duckdb_raw = json.load(f)
        duckdb_data = {k: v["estimate"] for k, v in duckdb_raw.items()}
        print(f"Loaded DuckDB estimates for {len(duckdb_data)} queries")
    else:
        duckdb_data = {}
        print("No DuckDB estimates found (run duckdb_estimates.py to generate)")
    print()

    # Run queries
    queries = make_queries(df)
    results = []

    print(f"{'Query':<30} {'True':>8} {'DuckDB':>8} {'Indep':>8} {'BN-samp':>8} {'BN-exact':>8} | "
          f"{'qe-D':>6} {'qe-I':>6} {'qe-Bs':>6} {'qe-Be':>6}")
    print("-" * 118)

    for name, group, mask_fn, desc in queries:
        true_card = int(true_cardinality(df, mask_fn))
        duck_est = duckdb_data.get(desc)
        indep_est = independence_estimate(df, mask_fn, n_total, desc)
        bn_samp_est = bn_sample_estimate(bn_samples, mask_fn, n_total)

        # Exact BN estimate
        frac_exact = fjd.values[mask_fn(fjd_df).values].sum()
        bn_ex_est = max(1, round(frac_exact * n_total))

        qe_d = q_error(duck_est, true_card) if duck_est else None
        qe_i = q_error(indep_est, true_card)
        qe_bs = q_error(bn_samp_est, true_card)
        qe_be = q_error(bn_ex_est, true_card)

        d_str = f"{duck_est:>8,}" if duck_est else "     N/A"
        qd_str = f"{qe_d:>6.2f}" if qe_d else "   N/A"

        print(f"{name:<30} {true_card:>8,} {d_str} {indep_est:>8,} "
              f"{bn_samp_est:>8,} {bn_ex_est:>8,} | "
              f"{qd_str} {qe_i:>6.2f} {qe_bs:>6.2f} {qe_be:>6.2f}")

        results.append({
            "query": name, "group": group,
            "true": true_card,
            "duckdb": duck_est,
            "independence": indep_est,
            "bn_sample": bn_samp_est,
            "bn_exact": bn_ex_est,
            "qe_duckdb": qe_d,
            "qe_independence": qe_i,
            "qe_bn_sample": qe_bs,
            "qe_bn_exact": qe_be,
        })

    # Summary
    results_df = pd.DataFrame(results)
    print()
    print("=" * 75)
    print("Geometric Mean Q-error by Group")
    print("=" * 75)
    print(f"{'Group':<15} {'DuckDB':>10} {'Indep':>10} {'BN-sample':>10} {'BN-exact':>10}  {'n':>4}")
    print("-" * 65)

    def gmean(series):
        vals = series.dropna()
        return np.exp(np.log(vals).mean()) if len(vals) else None

    for group in ["single", "range", "correlated"]:
        sub = results_df[results_df["group"] == group]
        gd = gmean(sub["qe_duckdb"])
        gi = gmean(sub["qe_independence"])
        gs = gmean(sub["qe_bn_sample"])
        ge = gmean(sub["qe_bn_exact"])
        gd_s = f"{gd:>10.2f}" if gd else "       N/A"
        print(f"{group:<15} {gd_s} {gi:>10.2f} {gs:>10.2f} {ge:>10.2f}  {len(sub):>4}")

    gd = gmean(results_df["qe_duckdb"])
    gi = gmean(results_df["qe_independence"])
    gs = gmean(results_df["qe_bn_sample"])
    ge = gmean(results_df["qe_bn_exact"])
    gd_s = f"{gd:>10.2f}" if gd else "       N/A"
    print("-" * 65)
    print(f"{'OVERALL':<15} {gd_s} {gi:>10.2f} {gs:>10.2f} {ge:>10.2f}  {len(results_df):>4}")

    # Timing
    print()
    print("Timing")
    print("-" * 40)
    print(f"  Path sampling:  {sample_time:.2f}s for {N_SAMPLES:,} samples")
    print(f"  Exact FJD:      {fjd_time:.3f}s")
    print()

    # Save
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
