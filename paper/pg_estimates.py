"""
Extract per-table cardinality estimates from PostgreSQL's EXPLAIN output
for all 113 JOB queries.

Saves results to pg_estimates.json.
"""

import json
import pathlib
import subprocess

JOB_DIR = pathlib.Path(__file__).resolve().parent / "job"
OUTPUT = pathlib.Path(__file__).resolve().parent / "pg_estimates.json"


def run_explain(sql, db="imdb"):
    """Run EXPLAIN (FORMAT JSON) and return the parsed plan."""
    result = subprocess.run(
        ["psql", db, "-t", "-A", "-c", f"EXPLAIN (FORMAT JSON) {sql}"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return None, result.stderr.strip()
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError:
        return None, f"JSON parse error: {result.stdout[:200]}"


def extract_scans(plan, scans=None):
    """Recursively extract all Seq Scan / Index Scan nodes from an EXPLAIN plan."""
    if scans is None:
        scans = []

    node_type = plan.get("Node Type", "")
    if "Scan" in node_type:
        scans.append({
            "node_type": node_type,
            "relation": plan.get("Relation Name", plan.get("Alias", "")),
            "alias": plan.get("Alias", ""),
            "plan_rows": plan.get("Plan Rows", 0),
            "filter": plan.get("Filter", ""),
            "index_cond": plan.get("Index Cond", ""),
        })

    for child in plan.get("Plans", []):
        extract_scans(child, scans)

    return scans


def main():
    query_files = sorted(JOB_DIR.glob("*.sql"))
    query_files = [f for f in query_files if f.stem not in ("schema", "fkindexes")]

    print(f"Running EXPLAIN on {len(query_files)} JOB queries...\n")

    results = {}
    errors = []

    for qf in query_files:
        sql = qf.read_text().strip()
        plan_json, err = run_explain(sql)
        if err:
            errors.append((qf.stem, err))
            continue

        plan = plan_json[0]["Plan"]
        scans = extract_scans(plan)

        results[qf.stem] = {
            "total_estimate": plan.get("Plan Rows", 0),
            "scans": scans,
        }
        print(f"  {qf.stem}: {len(scans)} scans, total_est={plan.get('Plan Rows', 0)}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for name, err in errors:
            print(f"  {name}: {err[:100]}")

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} query plans to {OUTPUT}")


if __name__ == "__main__":
    main()
