"""
Load benchmark datasets into PostgreSQL and provide DataFrames.
"""

import subprocess
import pathlib

import pandas as pd

PAPER_DIR = pathlib.Path(__file__).resolve().parents[1]


def run_psql(db, sql, timeout=120):
    """Run a SQL command against PostgreSQL, return stdout."""
    r = subprocess.run(
        ["psql", db, "-t", "-A", "-c", sql],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"psql error: {r.stderr[:500]}")
    return r.stdout.strip()


def run_psql_file(db, path, timeout=120):
    """Run a SQL file against PostgreSQL."""
    r = subprocess.run(
        ["psql", db, "-f", str(path)],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"psql error: {r.stderr[:500]}")
    return r.stdout


def db_exists(db):
    """Check if a PostgreSQL database exists."""
    r = subprocess.run(
        ["psql", "-lqt"], capture_output=True, text=True, timeout=10
    )
    return any(db in line.split("|")[0] for line in r.stdout.split("\n"))


def setup_stats_db(force=False):
    """Create and populate the stats database if it doesn't exist."""
    if db_exists("stats") and not force:
        n = int(run_psql("stats", "SELECT COUNT(*) FROM users"))
        print(f"  stats database already exists ({n} users)")
        return

    data_dir = PAPER_DIR / "stats-ceb" / "datasets" / "stats_simplified"
    schema_path = data_dir / "stats.sql"

    if not schema_path.exists():
        raise FileNotFoundError(f"Stats schema not found at {schema_path}")

    # Create database
    subprocess.run(["dropdb", "--if-exists", "stats"],
                   capture_output=True, timeout=10)
    subprocess.run(["createdb", "stats"], capture_output=True, timeout=10)

    # Load schema
    run_psql_file("stats", schema_path)

    # Load CSVs
    tables = ["users", "posts", "postLinks", "postHistory",
              "comments", "badges", "tags", "votes"]
    for table in tables:
        csv_path = data_dir / f"{table}.csv"
        if csv_path.exists():
            run_psql("stats",
                     f"\\COPY {table} FROM '{csv_path}' WITH (FORMAT csv, DELIMITER ',', HEADER true)",
                     timeout=60)

    run_psql("stats", "ANALYZE")
    n = int(run_psql("stats", "SELECT COUNT(*) FROM users"))
    print(f"  stats database created ({n} users)")


def setup_imdb_db():
    """Check that the imdb database exists (should already be set up)."""
    if not db_exists("imdb"):
        raise RuntimeError("imdb database not found. Load it first.")
    n = int(run_psql("imdb", "SELECT COUNT(*) FROM title"))
    print(f"  imdb database exists ({n:,} titles)")


def load_table(db, table, columns=None):
    """Load a table (or subset of columns) from PostgreSQL into a DataFrame."""
    cols = ", ".join(columns) if columns else "*"
    sql = f"SELECT {cols} FROM {table}"
    r = subprocess.run(
        ["psql", db, "-t", "-A", "-F", "\t", "-c", sql],
        capture_output=True, text=True, timeout=120,
    )
    if not r.stdout.strip():
        return pd.DataFrame()
    rows = [line.split("\t") for line in r.stdout.strip().split("\n")]
    col_names = columns or [f"col{i}" for i in range(len(rows[0]))]
    return pd.DataFrame(rows, columns=col_names)


def load_joined(db, sql, timeout=300):
    """Execute a JOIN query and return the result as a DataFrame."""
    r = subprocess.run(
        ["psql", db, "-F", "\t", "-A", "-c", sql],
        capture_output=True, text=True, timeout=timeout,
    )
    if not r.stdout.strip():
        return pd.DataFrame()
    lines = r.stdout.strip().split("\n")
    header = lines[0].split("\t")
    # Last line might be a count marker like "(N rows)"
    data_lines = [l for l in lines[1:] if not l.startswith("(")]
    rows = [line.split("\t") for line in data_lines]
    return pd.DataFrame(rows, columns=header)
