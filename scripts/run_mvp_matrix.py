#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from agent_security.config import RunConfig
from agent_security.runner import run as run_once


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser(description="Run baseline/defended matrix over multiple seeds and summarize.")
    ap.add_argument("--subset", default="subset50mix")
    ap.add_argument("--attack", choices=["on", "off"], default="on")
    ap.add_argument("--max-tasks", type=int, default=20)
    ap.add_argument("--seeds", default="41,42,43", help="Comma-separated seeds")
    ap.add_argument("--backend", choices=["heuristic", "gemini"], default="heuristic")
    ap.add_argument("--model-name", default="gemini-2.0-flash")
    ap.add_argument("--policy-backend", choices=["heuristic", "gemini"], default="heuristic")
    ap.add_argument("--policy-threshold", type=float, default=0.6)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("No valid seeds provided")

    rows = []
    metrics_by_mode: dict[str, list[dict]] = {"baseline": [], "defended": []}

    for seed in seeds:
        for mode in ["baseline", "defended"]:
            cfg = RunConfig(
                mode=mode,
                subset=args.subset,
                attack=args.attack,
                max_tasks=args.max_tasks,
                seed=seed,
                out_dir=args.out_dir,
                model_backend=args.backend,
                model_name=args.model_name,
                policy_backend=args.policy_backend,
                policy_threshold=args.policy_threshold,
            )
            out_path, payload = run_once(cfg)
            m = payload["metrics"]
            metrics_by_mode[mode].append(m)
            rows.append({
                "mode": mode,
                "seed": seed,
                "run_file": str(out_path),
                **m,
            })

    # Aggregate key metrics
    keys = [
        "benign_utility",
        "utility_under_attack",
        "targeted_ASR",
        "benign_keyword_coverage",
        "attacked_keyword_coverage",
        "targeted_ASR_soft",
    ]

    summary_rows = [["metric", "baseline_mean", "baseline_std", "defended_mean", "defended_std"]]
    for k in keys:
        b = [x.get(k, 0.0) for x in metrics_by_mode["baseline"]]
        d = [x.get(k, 0.0) for x in metrics_by_mode["defended"]]
        summary_rows.append([k, round(mean(b), 4), round(std(b), 4), round(mean(d), 4), round(std(d), 4)])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_tables = Path(args.out_dir) / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    per_run_path = out_tables / f"matrix_runs_{ts}.json"
    per_run_path.write_text(json.dumps(rows, indent=2))

    summary_csv = out_tables / f"matrix_summary_{ts}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(summary_rows)

    print(f"saved: {per_run_path}")
    print(f"saved: {summary_csv}")


if __name__ == "__main__":
    main()
