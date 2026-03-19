from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path

from .config import RunConfig, subset_path
from .baseline_agent import BaselineAgent
from .defended_agent import DefendedAgent
from .eval.agentdojo_adapter import load_subset, default_tools, MockAgentDojoEnv, evaluate_task
from .eval.metrics import compute_metrics
from .utils.jsonio import dump_json


def run(cfg: RunConfig):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(cfg.out_dir) / cfg.mode
    run_dir.mkdir(parents=True, exist_ok=True)

    spath = subset_path(cfg.subset)
    tasks = load_subset(spath, attack=cfg.attack, max_tasks=cfg.max_tasks, seed=cfg.seed)
    tools = default_tools()
    tool_names = [t.name for t in tools]

    if cfg.mode == "baseline":
        agent = BaselineAgent(max_steps=cfg.max_steps)
    else:
        backend = "gemini" if cfg.model_backend == "gemini" else "deterministic"
        agent = DefendedAgent(
            max_steps=cfg.max_steps,
            model_backend=backend,
            model_name=cfg.model_name,
            policy_backend=cfg.policy_backend,
            policy_threshold=cfg.policy_threshold,
        )

    traces, outcomes = [], []

    for task in tasks:
        env = MockAgentDojoEnv(task)
        if cfg.mode == "defended":
            trace = agent.run(task, tool_names, env, audit_path=str(run_dir / f"audit_{ts}.jsonl"))
        else:
            trace = agent.run(task, tool_names, env)
        outcome = evaluate_task(task, trace, env)
        traces.append(trace)
        outcomes.append(outcome)

    metrics = compute_metrics(outcomes)
    payload = {
        "config": {
            "mode": cfg.mode,
            "subset": cfg.subset,
            "attack": cfg.attack,
            "max_tasks": cfg.max_tasks,
            "seed": cfg.seed,
            "model_backend": cfg.model_backend,
            "model_name": cfg.model_name,
            "policy_backend": cfg.policy_backend,
            "policy_threshold": cfg.policy_threshold,
            "temperature": cfg.temperature,
        },
        "metrics": metrics,
        "outcomes": outcomes,
        "traces": traces,
    }

    out_path = run_dir / f"run_{ts}.json"
    dump_json(out_path, payload)
    print(f"saved: {out_path}")
    print(metrics)
    return out_path, payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "defended"], required=True)
    ap.add_argument("--subset", default="small")
    ap.add_argument("--attack", choices=["on", "off"], default="on")
    ap.add_argument("--max_tasks", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--model_backend", default="heuristic", choices=["heuristic", "gemini"])
    ap.add_argument("--model_name", default="gemini-2.0-flash")
    ap.add_argument("--policy_backend", default="heuristic", choices=["heuristic", "gemini"])
    ap.add_argument("--policy_threshold", type=float, default=0.6)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    cfg = RunConfig(
        mode=args.mode,
        subset=args.subset,
        attack=args.attack,
        max_tasks=args.max_tasks,
        max_steps=args.max_steps,
        seed=args.seed,
        out_dir=args.out_dir,
        model_backend=args.model_backend,
        model_name=args.model_name,
        policy_backend=args.policy_backend,
        policy_threshold=args.policy_threshold,
        temperature=args.temperature,
    )
    run(cfg)


if __name__ == "__main__":
    main()
