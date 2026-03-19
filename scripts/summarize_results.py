#!/usr/bin/env python3
from pathlib import Path
import json
import csv
import argparse
from datetime import datetime

root = Path(__file__).resolve().parents[1]
res = root / "results"


def load_run(path: Path) -> dict:
    return json.loads(path.read_text())


def latest(mode: str):
    files = sorted((res / mode).glob("run_*.json"))
    return files[-1] if files else None


def latest_matching(bj: dict):
    files = sorted((res / "defended").glob("run_*.json"), reverse=True)
    bc = bj.get("config", {})
    for p in files:
        dj = load_run(p)
        dc = dj.get("config", {})
        if (
            dc.get("subset") == bc.get("subset")
            and dc.get("attack") == bc.get("attack")
            and dc.get("max_tasks") == bc.get("max_tasks")
            and dc.get("seed") == bc.get("seed")
        ):
            return p, dj
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="", help="Path to baseline run json")
    ap.add_argument("--defended", default="", help="Path to defended run json")
    args = ap.parse_args()

    if args.baseline:
        b = Path(args.baseline)
    else:
        b = latest("baseline")

    if not b or not b.exists():
        raise SystemExit("Baseline run file not found.")

    bj = load_run(b)

    if args.defended:
        d = Path(args.defended)
        if not d.exists():
            raise SystemExit("Defended run file not found.")
        dj = load_run(d)
    else:
        d, dj = latest_matching(bj)
        if not d:
            raise SystemExit("No defended run matching baseline config. Pass --defended explicitly.")

    rows = [
        ["metric", "baseline", "defended"],
        ["benign_utility", bj["metrics"].get("benign_utility"), dj["metrics"].get("benign_utility")],
        ["utility_under_attack", bj["metrics"].get("utility_under_attack"), dj["metrics"].get("utility_under_attack")],
        ["targeted_ASR", bj["metrics"].get("targeted_ASR"), dj["metrics"].get("targeted_ASR")],
    ]

    out = res / "tables" / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"baseline: {b}")
    print(f"defended: {d}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
