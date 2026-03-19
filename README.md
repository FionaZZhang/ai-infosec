# Agent Security MVP (March 13)

MVP for **Hardening Tool-Using LLM Agents Against Prompt Injection**.

## What this repo demonstrates

- Baseline tool-using agent (naive raw tool-output feedback).
- Defended agent with **Policy & Isolation Layer**:
  - Quarantine all tool outputs.
  - Data-only summarizer step (deterministic by default; optional Gemini call).
  - Least-privilege tool allowlist per task.
  - Sensitive-tool gating (HIGH-risk tools denied by default).
  - Dry-run argument checks for exfil/scope violations.
  - Minimal provenance tracking and JSONL audit logs.
- Evaluation on a small AgentDojo-style subset.
- Quantitative metrics:
  - benign_utility
  - utility_under_attack
  - targeted_ASR

## Repo layout

```text
agent-security-mvp/
  README.md
  pyproject.toml
  src/agent_security/
    config.py
    runner.py
    baseline_agent.py
    defended_agent.py
    policy/
    isolation/
    eval/
    utils/
    data/agentdojo_small_subset.json
  scripts/
    run_baseline.sh
    run_defended.sh
    summarize_results.py
  notebooks/
    demo_march13.ipynb
  results/
```

## Quickstart

```bash
cd agent-security-mvp
bash scripts/run_baseline.sh
bash scripts/run_defended.sh
python3 scripts/summarize_results.py
```

Outputs:
- `results/baseline/run_<timestamp>.json`
- `results/defended/run_<timestamp>.json`
- `results/defended/audit_<timestamp>.jsonl`
- `results/tables/summary_<timestamp>.csv`

## Official AgentDojo-aligned evaluation (repo/paper method)

To avoid MVP toy-metric artifacts, run the official AgentDojo benchmark script and compute metrics from official task logs:

```bash
# 1) run official benchmark (requires credentials in tmp_agentdojo_official)
MODEL=gemini-2.0-flash-001 ATTACK=important_instructions DEFENSE=none \
  bash scripts/run_agentdojo_official.sh

# 2) compute Utility / Utility-under-attack / Targeted-ASR from official logs
python3 scripts/compute_agentdojo_metrics.py \
  --run-root ../tmp_agentdojo_official/runs/<pipeline_name> \
  --attack important_instructions \
  --out-dir results/official

# 3) credential-free subset mode (metrics on official run artifacts listed in subset JSON)
python3 scripts/compute_agentdojo_metrics.py \
  --subset-json src/agent_security/data/agentdojo_subset_50_mixed_official.json \
  --attack important_instructions \
  --out-dir results/official
```

Notes:
- `Targeted ASR` follows AgentDojo injection-task semantics (`security=True` means attacker goal achieved).
- Official comparison is only valid when benchmark version, attack name, defense, suites/tasks, and model setup match.

## Optional: Gemini summarizer mode

For faster iteration, use `gemini-2.0-flash`; for final checks, switch to pro.

```bash
PYTHONPATH=src python3 -m agent_security.runner \
  --mode defended \
  --attack on \
  --model_backend gemini \
  --model_name gemini-2.0-flash
```

## CLI

```bash
PYTHONPATH=src python3 -m agent_security.runner \
  --mode baseline|defended \
  --subset small|subset50|subset50mix|<path_to_json> \
  --attack on|off \
  --max_tasks 30 \
  --seed 42 \
  --out_dir results
```

### Repro script: baseline vs defended(heuristic) vs defended(heuristic+gemini)

```bash
PYTHONPATH=src python3 scripts/run_send_action_repro.py \
  --subset src/agent_security/data/agentdojo_subset_send_action_20.json \
  --seed 42 \
  --max-tasks 20 \
  --model-name gemini-2.5-flash
```

Outputs:
- `results/baseline/run_<ts>.json`
- `results/defended/run_<ts>.json` (heuristic)
- `results/defended/run_<ts>.json` (heuristic+gemini)
- `results/tables/send_action_repro_<ts>.{csv,md,json}`

### Multi-seed matrix (recommended for non-binary view)

```bash
PYTHONPATH=src python3 scripts/run_mvp_matrix.py \
  --subset subset50mix \
  --attack on \
  --max-tasks 20 \
  --seeds 41,42,43 \
  --backend heuristic
```

Outputs:
- `results/tables/matrix_runs_<ts>.json`
- `results/tables/matrix_summary_<ts>.csv` (mean/std across seeds)

50-task subset file:
- `src/agent_security/data/agentdojo_subset_50.json`
- This subset is derived from official AgentDojo run artifacts in the upstream repository (`runs/claude-3-opus-20240229/workspace/*`).

## Acceptance checklist

- [x] baseline run command works end-to-end on subset
- [x] defended run command works end-to-end on subset
- [x] defended agent never feeds raw tool output into planner prompt
- [x] policy layer produces allow/deny with reasons
- [x] results include metrics and summary table
- [x] demo notebook or demo log exists

## Error analysis (MVP)

1. **Utility drop under strict gating**: denying all high-risk tools can block legitimate tasks that actually require write actions.
2. **Heuristic summarizer misses fields**: deterministic extraction may miss nuanced facts; enabling Gemini summarizer improves quality.
3. **Prompt-injection paraphrases**: simple regex detection can miss indirect malicious instructions; stronger classifier/AST-style parser is future work.
4. **Task-type mapping is coarse**: allowlist by keyword can be brittle; richer intent classification is future work.
5. **Fallback dataset**: if full AgentDojo integration is unavailable locally, this MVP runs a compatible small subset to keep the pipeline reproducible.
