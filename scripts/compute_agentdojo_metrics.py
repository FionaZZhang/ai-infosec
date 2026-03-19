#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class TaskRec:
    suite: str
    user_task_id: str
    attack_type: str | None
    utility: bool
    security: bool


def _iter_json_files(run_root: Path) -> Iterable[Path]:
    for p in run_root.rglob("*.json"):
        # Expected layout: <pipeline>/<suite>/<user_task>/<attack>/<injection>.json
        # Keep it permissive and validate by keys.
        yield p


def _iter_subset_source_files(subset_json: Path) -> Iterable[Path]:
    data = json.loads(subset_json.read_text())
    for row in data:
        src = row.get("source")
        if src:
            p = Path(src)
            if p.exists():
                yield p


def _load_record(path: Path) -> TaskRec | None:
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return None

    required = {"suite_name", "user_task_id", "attack_type", "utility", "security"}
    if not required.issubset(obj.keys()):
        return None

    return TaskRec(
        suite=str(obj["suite_name"]),
        user_task_id=str(obj["user_task_id"]),
        attack_type=obj.get("attack_type"),
        utility=bool(obj["utility"]),
        security=bool(obj["security"]),
    )


def mean_bool(xs: list[bool]) -> float:
    return (sum(1 for x in xs if x) / len(xs)) if xs else 0.0


def compute(records: list[TaskRec], attacked_name: str) -> dict:
    benign = [
        r for r in records
        if (r.attack_type is None or str(r.attack_type).lower() == "none")
        and r.user_task_id.startswith("user_task_")
    ]
    attacked = [
        r for r in records
        if str(r.attack_type) == attacked_name and r.user_task_id.startswith("user_task_")
    ]

    utility = mean_bool([r.utility for r in benign])
    utility_under_attack = mean_bool([r.utility for r in attacked])
    # In AgentDojo injection-task semantics, security=True means the
    # attacker's goal condition was satisfied (i.e., successful compromise).
    targeted_asr = mean_bool([r.security for r in attacked])

    by_suite = {}
    suites = sorted(set(r.suite for r in records))
    for s in suites:
        b = [r for r in benign if r.suite == s]
        a = [r for r in attacked if r.suite == s]
        by_suite[s] = {
            "benign_count": len(b),
            "attacked_count": len(a),
            "utility": round(mean_bool([r.utility for r in b]), 6),
            "utility_under_attack": round(mean_bool([r.utility for r in a]), 6),
            "targeted_ASR": round(mean_bool([r.security for r in a]), 6),
        }

    return {
        "benign_count": len(benign),
        "attacked_count": len(attacked),
        "utility": round(utility, 6),
        "utility_under_attack": round(utility_under_attack, 6),
        "targeted_ASR": round(targeted_asr, 6),
        "by_suite": by_suite,
    }


def main():
    ap = argparse.ArgumentParser(description="Compute AgentDojo-aligned metrics from official run logs.")
    ap.add_argument("--run-root", default="", help="Path to one pipeline run root, e.g. tmp_agentdojo_official/runs/claude-3-opus-20240229")
    ap.add_argument("--subset-json", default="", help="Path to AgentDojo subset json with `source` file paths")
    ap.add_argument("--attack", default="important_instructions", help="Attack name to evaluate (default: important_instructions)")
    ap.add_argument("--out-dir", default="results/official", help="Output directory")
    args = ap.parse_args()

    if not args.run_root and not args.subset_json:
        raise SystemExit("Provide at least one input: --run-root or --subset-json")

    recs: list[TaskRec] = []

    if args.run_root:
        run_root = Path(args.run_root)
        if not run_root.exists():
            raise SystemExit(f"run root not found: {run_root}")
        for p in _iter_json_files(run_root):
            rec = _load_record(p)
            if rec is not None:
                recs.append(rec)
        model_name = run_root.name
    else:
        model_name = "subset"

    if args.subset_json:
        subset_path = Path(args.subset_json)
        if not subset_path.exists():
            raise SystemExit(f"subset json not found: {subset_path}")
        recs = []
        for p in _iter_subset_source_files(subset_path):
            rec = _load_record(p)
            if rec is not None:
                recs.append(rec)
        model_name = f"subset_{subset_path.stem}"

    if not recs:
        raise SystemExit("No valid task result JSON files found.")

    metrics = compute(recs, attacked_name=args.attack)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"agentdojo_metrics_{model_name}_{args.attack}_{ts}.json"
    json_path.write_text(json.dumps(metrics, indent=2))

    csv_path = out_dir / f"agentdojo_metrics_{model_name}_{args.attack}_{ts}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["utility", metrics["utility"]])
        w.writerow(["utility_under_attack", metrics["utility_under_attack"]])
        w.writerow(["targeted_ASR", metrics["targeted_ASR"]])
        w.writerow(["benign_count", metrics["benign_count"]])
        w.writerow(["attacked_count", metrics["attacked_count"]])

    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")
    print(json.dumps({k: metrics[k] for k in ['utility','utility_under_attack','targeted_ASR','benign_count','attacked_count']}, indent=2))


if __name__ == "__main__":
    main()
