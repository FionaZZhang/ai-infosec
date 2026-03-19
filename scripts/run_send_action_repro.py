#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from agent_security.config import RunConfig
from agent_security.runner import run


def _table_rows(metrics_by_name: dict[str, dict]) -> list[list[str]]:
    cols = [
        "benign_utility",
        "utility_under_attack",
        "targeted_ASR",
        "benign_keyword_coverage",
        "attacked_keyword_coverage",
        "targeted_ASR_soft",
    ]
    order = ["baseline", "defended_heuristic", "defended_heuristic_gemini"]
    rows: list[list[str]] = []
    for name in order:
        m = metrics_by_name[name]
        rows.append([
            name,
            f"{m['benign_utility']:.4f}",
            f"{m['utility_under_attack']:.4f}",
            f"{m['targeted_ASR']:.4f}",
            f"{m['benign_keyword_coverage']:.4f}",
            f"{m['attacked_keyword_coverage']:.4f}",
            f"{m['targeted_ASR_soft']:.4f}",
        ])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce baseline vs defended(send-action) comparison")
    ap.add_argument(
        "--subset",
        default="src/agent_security/data/agentdojo_subset_send_action_20.json",
        help="Subset alias/path to run",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tasks", type=int, default=20)
    ap.add_argument("--model-name", default="gemini-2.5-flash")
    ap.add_argument("--policy-threshold", type=float, default=0.6)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    # 1) baseline
    p_baseline, r_baseline = run(
        RunConfig(
            mode="baseline",
            subset=args.subset,
            attack="on",
            max_tasks=args.max_tasks,
            seed=args.seed,
            out_dir=args.out_dir,
            model_backend="heuristic",
            policy_backend="heuristic",
            model_name=args.model_name,
            policy_threshold=args.policy_threshold,
        )
    )

    # 2) defended heuristic
    p_def_h, r_def_h = run(
        RunConfig(
            mode="defended",
            subset=args.subset,
            attack="on",
            max_tasks=args.max_tasks,
            seed=args.seed,
            out_dir=args.out_dir,
            model_backend="heuristic",
            policy_backend="heuristic",
            model_name=args.model_name,
            policy_threshold=args.policy_threshold,
        )
    )

    # 3) defended heuristic + gemini guard
    p_def_g, r_def_g = run(
        RunConfig(
            mode="defended",
            subset=args.subset,
            attack="on",
            max_tasks=args.max_tasks,
            seed=args.seed,
            out_dir=args.out_dir,
            model_backend="heuristic",
            policy_backend="gemini",
            model_name=args.model_name,
            policy_threshold=args.policy_threshold,
        )
    )

    metrics_by_name = {
        "baseline": r_baseline["metrics"],
        "defended_heuristic": r_def_h["metrics"],
        "defended_heuristic_gemini": r_def_g["metrics"],
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_dir = Path(args.out_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    csv_path = table_dir / f"send_action_repro_{ts}.csv"
    md_path = table_dir / f"send_action_repro_{ts}.md"
    manifest_path = table_dir / f"send_action_repro_{ts}.json"

    header = [
        "variant",
        "benign_utility",
        "utility_under_attack",
        "targeted_ASR",
        "benign_keyword_coverage",
        "attacked_keyword_coverage",
        "targeted_ASR_soft",
    ]
    rows = _table_rows(metrics_by_name)

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    name_map = {
        "baseline": "baseline",
        "defended_heuristic": "defended (heuristic)",
        "defended_heuristic_gemini": "defended (heuristic + gemini)",
    }
    md_lines = [
        "| Variant | BU | UAA | ASR | Benign KC | Attacked KC | ASR_soft |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ["baseline", "defended_heuristic", "defended_heuristic_gemini"]:
        m = metrics_by_name[k]
        md_lines.append(
            f"| {name_map[k]} | {m['benign_utility']:.2f} | {m['utility_under_attack']:.2f} | {m['targeted_ASR']:.2f} | {m['benign_keyword_coverage']:.4f} | {m['attacked_keyword_coverage']:.4f} | {m['targeted_ASR_soft']:.2f} |"
        )

    md_lines += [
        "",
        "Run files:",
        f"- baseline: `{p_baseline}`",
        f"- defended (heuristic): `{p_def_h}`",
        f"- defended (heuristic + gemini): `{p_def_g}`",
    ]
    md_path.write_text("\n".join(md_lines) + "\n")

    manifest = {
        "subset": args.subset,
        "seed": args.seed,
        "max_tasks": args.max_tasks,
        "model_name": args.model_name,
        "policy_threshold": args.policy_threshold,
        "runs": {
            "baseline": str(p_baseline),
            "defended_heuristic": str(p_def_h),
            "defended_heuristic_gemini": str(p_def_g),
        },
        "metrics": metrics_by_name,
        "outputs": {
            "csv": str(csv_path),
            "md": str(md_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"saved: {csv_path}")
    print(f"saved: {md_path}")
    print(f"saved: {manifest_path}")
    print(json.dumps(metrics_by_name, indent=2))


if __name__ == "__main__":
    main()
