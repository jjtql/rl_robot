import argparse
import csv
import re
from pathlib import Path

import numpy as np


SEED_RE = re.compile(r"(.+)_seed(\d+)$")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_run_metadata(input_dir, summary_path):
    rel_parent = summary_path.parent.relative_to(input_dir)
    parts = rel_parent.parts
    suite_group = parts[0] if parts else "root"
    run_name = parts[-1] if parts else "root"
    method = run_name
    train_seed = ""

    if run_name == "baselines":
        method = "baseline"
    else:
        match = SEED_RE.match(run_name)
        if match:
            method = match.group(1)
            train_seed = match.group(2)

    return {
        "suite_group": suite_group,
        "suite_run": str(rel_parent),
        "method": method,
        "train_seed": train_seed,
        "source_summary": str(summary_path),
    }


def collect(input_dir):
    input_dir = Path(input_dir)
    rows = []
    for summary_path in sorted(input_dir.glob("**/summary.csv")):
        metadata = infer_run_metadata(input_dir, summary_path)
        for row in read_csv(summary_path):
            out = dict(metadata)
            out.update(row)
            rows.append(out)
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        key = (
            row.get("method", ""),
            row.get("policy", ""),
            row.get("stage", ""),
            row.get("max_steams", ""),
        )
        groups.setdefault(key, []).append(row)

    out_rows = []
    for (method, policy, stage, max_steams), group_rows in sorted(groups.items()):
        numeric_fields = []
        for field in group_rows[0].keys():
            if field in {"train_seed", "seed", "episodes"}:
                continue
            if not field.endswith("_mean"):
                continue
            if any(as_float(row.get(field)) is not None for row in group_rows):
                numeric_fields.append(field)

        summary = {
            "method": method,
            "policy": policy,
            "stage": stage,
            "max_steams": max_steams,
            "rows": len(group_rows),
            "train_seeds": ",".join(sorted({row.get("train_seed", "") for row in group_rows if row.get("train_seed", "")})),
            "eval_seeds": ",".join(sorted({row.get("seed", "") for row in group_rows if row.get("seed", "")})),
        }
        for field in numeric_fields:
            values = [as_float(row.get(field)) for row in group_rows]
            values = np.array([value for value in values if value is not None], dtype=np.float64)
            if values.size == 0:
                continue
            summary[f"{field}_across_runs"] = float(values.mean())
            summary[f"{field}_std_across_runs"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        out_rows.append(summary)
    return out_rows


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect run_matrix summary.csv files into paper-ready tables.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--combined-output", required=True)
    parser.add_argument("--paper-output", required=True)
    return parser


def main():
    args = build_arg_parser().parse_args()
    rows = collect(args.input_dir)
    if not rows:
        raise FileNotFoundError(f"No summary.csv files found under {args.input_dir}")
    write_csv(args.combined_output, rows)
    write_csv(args.paper_output, summarize(rows))
    print(f"Wrote {len(rows)} combined rows to {args.combined_output}", flush=True)
    print(f"Wrote paper summary to {args.paper_output}", flush=True)


if __name__ == "__main__":
    main()
