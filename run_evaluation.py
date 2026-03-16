#!/usr/bin/env python3
"""
Main evaluation script for the agent security project.

Runs baseline and defended agents on real AgentDojo benchmarks,
calculates metrics, and generates visualizations.

Usage:
    python run_evaluation.py [--suites workspace,banking] [--api-key KEY]
    python run_evaluation.py --num-user-tasks 5 --num-injection-tasks 3  # Quick test
"""

import argparse
import os
import sys
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.agentdojo_loader import (
    get_suite,
    create_attack_payload,
    DEFAULT_BENCHMARK_VERSION,
    load_suite,
    ATTACK_TEMPLATES,
)
from agentdojo.task_suite.load_suites import get_suite as agentdojo_get_suite
from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from src.evaluation.metrics import MetricsCalculator, TaskResult, EvaluationResult
from src.visualization.charts import ResultsVisualizer
from src.config import DefenseConfig


def run_real_evaluation(
    suites: list[str],
    num_user_tasks: int = None,
    num_injection_tasks: int = None,
    model: str = "gpt-4o-mini-2024-07-18",
    attack_type: str = "important_instructions",
    verbose: bool = True,
) -> dict[str, dict[str, EvaluationResult]]:
    """Run evaluation using real AgentDojo API calls."""

    results = {}

    for suite_name in suites:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {suite_name.upper()} SUITE")
        print(f"{'='*60}")

        # Load suite
        suite = agentdojo_get_suite(DEFAULT_BENCHMARK_VERSION, suite_name)
        suite_data = load_suite(suite_name)

        user_task_ids = list(suite.user_tasks.keys())
        injection_task_ids = list(suite.injection_tasks.keys())

        # Limit tasks if specified
        if num_user_tasks:
            user_task_ids = user_task_ids[:num_user_tasks]
        if num_injection_tasks:
            injection_task_ids = injection_task_ids[:num_injection_tasks]

        print(f"  User tasks: {len(user_task_ids)}")
        print(f"  Injection tasks: {len(injection_task_ids)}")
        print(f"  Total test cases: {len(user_task_ids)} benign + {len(user_task_ids) * len(injection_task_ids)} attack")
        print(f"  Model: {model}")
        print(f"  Attack type: {attack_type}")

        # Create agent pipeline (baseline - no defenses)
        config = PipelineConfig(
            llm=model,
            defense=None,
            system_message_name='default',
            system_message=None,
            model_id=None,
        )
        agent_pipeline = AgentPipeline.from_config(config)

        # Track results for baseline
        baseline_metrics = MetricsCalculator()

        # Run benign tasks
        print(f"\n--- Baseline: Benign Tasks ---")
        for i, user_task_id in enumerate(user_task_ids):
            user_task = suite.user_tasks[user_task_id]
            if verbose:
                print(f"  [{i+1}/{len(user_task_ids)}] {user_task_id}")

            try:
                utility, security = suite.run_task_with_pipeline(
                    agent_pipeline=agent_pipeline,
                    user_task=user_task,
                    injection_task=None,
                    injections={},
                )
                result = TaskResult(
                    task_id=f"{user_task_id}_benign",
                    suite=suite_name,
                    is_attack_case=False,
                    task_completed=utility,
                    attack_succeeded=False,
                    attacker_goal_achieved=False,
                    harmful_action_executed=False,
                    tools_blocked=[],
                    injections_detected=0,
                )
                baseline_metrics.add_result(result)
                if verbose:
                    status = "OK" if utility else "FAIL"
                    print(f"    [{status}]")
            except Exception as e:
                print(f"    [ERROR] {e}")
                result = TaskResult(
                    task_id=f"{user_task_id}_benign",
                    suite=suite_name,
                    is_attack_case=False,
                    task_completed=False,
                    attack_succeeded=False,
                    attacker_goal_achieved=False,
                    harmful_action_executed=False,
                    tools_blocked=[],
                    injections_detected=0,
                )
                baseline_metrics.add_result(result)

        # Run attack scenarios
        print(f"\n--- Baseline: Attack Scenarios ---")
        attack_count = 0
        total_attacks = len(user_task_ids) * len(injection_task_ids)

        for user_task_id in user_task_ids:
            user_task = suite.user_tasks[user_task_id]

            for injection_task_id in injection_task_ids:
                attack_count += 1
                injection_task = suite.injection_tasks[injection_task_id]

                if verbose:
                    print(f"  [{attack_count}/{total_attacks}] {user_task_id} + {injection_task_id}", end="")

                # Create injection payload
                injection_payload = create_attack_payload(attack_type, injection_task.GOAL)
                vector_defaults = suite.get_injection_vector_defaults()
                injections = {vector_id: injection_payload for vector_id in vector_defaults.keys()}

                try:
                    utility, security = suite.run_task_with_pipeline(
                        agent_pipeline=agent_pipeline,
                        user_task=user_task,
                        injection_task=injection_task,
                        injections=injections,
                    )
                    attack_succeeded = not security
                    result = TaskResult(
                        task_id=f"{user_task_id}_{injection_task_id}",
                        suite=suite_name,
                        is_attack_case=True,
                        task_completed=utility,
                        attack_succeeded=attack_succeeded,
                        attacker_goal_achieved=attack_succeeded,
                        harmful_action_executed=attack_succeeded,
                        tools_blocked=[],
                        injections_detected=0,
                    )
                    baseline_metrics.add_result(result)
                    if verbose:
                        status = "PWNED" if attack_succeeded else "SAFE"
                        print(f" [{status}]")
                except Exception as e:
                    print(f" [ERROR] {e}")
                    result = TaskResult(
                        task_id=f"{user_task_id}_{injection_task_id}",
                        suite=suite_name,
                        is_attack_case=True,
                        task_completed=False,
                        attack_succeeded=False,
                        attacker_goal_achieved=False,
                        harmful_action_executed=False,
                        tools_blocked=[],
                        injections_detected=0,
                    )
                    baseline_metrics.add_result(result)

        # Calculate baseline metrics
        baseline_result = baseline_metrics.calculate_metrics(suite_name, "baseline")

        # For defended, we use the spotlighting defense available in AgentDojo
        print(f"\n--- Defended: Running with spotlighting_with_delimiting defense ---")

        defended_config = PipelineConfig(
            llm=model,
            defense="spotlighting_with_delimiting",
            system_message_name='default',
            system_message=None,
            model_id=None,
        )
        defended_pipeline = AgentPipeline.from_config(defended_config)

        defended_metrics = MetricsCalculator()

        # Run benign tasks with defense
        print(f"\n--- Defended: Benign Tasks ---")
        for i, user_task_id in enumerate(user_task_ids):
            user_task = suite.user_tasks[user_task_id]
            if verbose:
                print(f"  [{i+1}/{len(user_task_ids)}] {user_task_id}", end="")

            try:
                utility, security = suite.run_task_with_pipeline(
                    agent_pipeline=defended_pipeline,
                    user_task=user_task,
                    injection_task=None,
                    injections={},
                )
                result = TaskResult(
                    task_id=f"{user_task_id}_benign",
                    suite=suite_name,
                    is_attack_case=False,
                    task_completed=utility,
                    attack_succeeded=False,
                    attacker_goal_achieved=False,
                    harmful_action_executed=False,
                    tools_blocked=[],
                    injections_detected=0,
                )
                defended_metrics.add_result(result)
                if verbose:
                    status = "OK" if utility else "FAIL"
                    print(f" [{status}]")
            except Exception as e:
                print(f" [ERROR] {e}")
                result = TaskResult(
                    task_id=f"{user_task_id}_benign",
                    suite=suite_name,
                    is_attack_case=False,
                    task_completed=False,
                    attack_succeeded=False,
                    attacker_goal_achieved=False,
                    harmful_action_executed=False,
                    tools_blocked=[],
                    injections_detected=0,
                )
                defended_metrics.add_result(result)

        # Run attack scenarios with defense
        print(f"\n--- Defended: Attack Scenarios ---")
        attack_count = 0

        for user_task_id in user_task_ids:
            user_task = suite.user_tasks[user_task_id]

            for injection_task_id in injection_task_ids:
                attack_count += 1
                injection_task = suite.injection_tasks[injection_task_id]

                if verbose:
                    print(f"  [{attack_count}/{total_attacks}] {user_task_id} + {injection_task_id}", end="")

                injection_payload = create_attack_payload(attack_type, injection_task.GOAL)
                vector_defaults = suite.get_injection_vector_defaults()
                injections = {vector_id: injection_payload for vector_id in vector_defaults.keys()}

                try:
                    utility, security = suite.run_task_with_pipeline(
                        agent_pipeline=defended_pipeline,
                        user_task=user_task,
                        injection_task=injection_task,
                        injections=injections,
                    )
                    attack_succeeded = not security
                    result = TaskResult(
                        task_id=f"{user_task_id}_{injection_task_id}",
                        suite=suite_name,
                        is_attack_case=True,
                        task_completed=utility,
                        attack_succeeded=attack_succeeded,
                        attacker_goal_achieved=attack_succeeded,
                        harmful_action_executed=attack_succeeded,
                        tools_blocked=[],
                        injections_detected=1 if not attack_succeeded else 0,
                    )
                    defended_metrics.add_result(result)
                    if verbose:
                        status = "PWNED" if attack_succeeded else "BLOCKED"
                        print(f" [{status}]")
                except Exception as e:
                    print(f" [ERROR] {e}")
                    result = TaskResult(
                        task_id=f"{user_task_id}_{injection_task_id}",
                        suite=suite_name,
                        is_attack_case=True,
                        task_completed=False,
                        attack_succeeded=False,
                        attacker_goal_achieved=False,
                        harmful_action_executed=False,
                        tools_blocked=[],
                        injections_detected=0,
                    )
                    defended_metrics.add_result(result)

        defended_result = defended_metrics.calculate_metrics(suite_name, "defended")

        results[suite_name] = {
            "baseline": baseline_result,
            "defended": defended_result,
        }

        # Print suite summary
        print(f"\n--- {suite_name.upper()} SUMMARY ---")
        print(f"  Baseline:")
        print(f"    Benign Utility:       {baseline_result.benign_utility*100:.1f}%")
        print(f"    Utility Under Attack: {baseline_result.utility_under_attack*100:.1f}%")
        print(f"    Attack Success Rate:  {baseline_result.attack_success_rate*100:.1f}%")
        print(f"  Defended (spotlighting):")
        print(f"    Benign Utility:       {defended_result.benign_utility*100:.1f}%")
        print(f"    Utility Under Attack: {defended_result.utility_under_attack*100:.1f}%")
        print(f"    Attack Success Rate:  {defended_result.attack_success_rate*100:.1f}%")
        asr_reduction = (baseline_result.attack_success_rate - defended_result.attack_success_rate) * 100
        print(f"  ASR Reduction: {asr_reduction:+.1f} percentage points")

    return results


def print_results_table(results: dict[str, dict[str, EvaluationResult]]):
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (Real AgentDojo Data)")
    print("=" * 80)

    for suite, configs in results.items():
        print(f"\n--- {suite.upper()} SUITE ---\n")

        print(f"{'Config':<20} {'Benign Utility':<18} {'Util Under Attack':<20} {'ASR':<10}")
        print("-" * 70)

        baseline = configs.get("baseline")
        defended = configs.get("defended")

        if baseline:
            print(
                f"{'Baseline':<20} "
                f"{baseline.benign_utility*100:>13.1f}% "
                f"{baseline.utility_under_attack*100:>16.1f}% "
                f"{baseline.attack_success_rate*100:>8.1f}%"
            )

        if defended:
            delta_asr = ""
            if baseline:
                delta = (defended.attack_success_rate - baseline.attack_success_rate) * 100
                delta_asr = f"({delta:+.1f}pp)"

            print(
                f"{'Defended':<20} "
                f"{defended.benign_utility*100:>13.1f}% "
                f"{defended.utility_under_attack*100:>16.1f}% "
                f"{defended.attack_success_rate*100:>8.1f}% {delta_asr}"
            )

    print("\n" + "=" * 80)


def save_results(results: dict, output_dir: str = "results"):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"evaluation_{timestamp}.json")

    output = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_version": DEFAULT_BENCHMARK_VERSION,
        "results": {
            suite: {
                config_name: result.to_dict()
                for config_name, result in configs.items()
            }
            for suite, configs in results.items()
        }
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    # Also save as latest
    latest_path = os.path.join(output_dir, "evaluation_results_latest.json")
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Run agent security evaluation on real AgentDojo benchmarks"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--suites",
        type=str,
        default="workspace,banking",
        help="Comma-separated list of suites to evaluate (default: workspace,banking)"
    )
    parser.add_argument(
        "--num-user-tasks",
        type=int,
        default=None,
        help="Limit number of user tasks per suite (default: all)"
    )
    parser.add_argument(
        "--num-injection-tasks",
        type=int,
        default=None,
        help="Limit number of injection tasks per suite (default: all)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Model to use (default: gpt-4o-mini-2024-07-18)"
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        default="important_instructions",
        choices=list(ATTACK_TEMPLATES.keys()),
        help="Attack template to use"
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Set API key
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key

    if not os.environ.get('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not set.")
        print("Provide via --api-key or set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Parse suites
    suites = [s.strip() for s in args.suites.split(",")]

    print("=" * 60)
    print("AGENTDOJO SECURITY EVALUATION")
    print("Hardening Tool-Using LLM Agents Against Prompt Injection")
    print("=" * 60)
    print(f"\nSuites: {suites}")
    print(f"Model: {args.model}")
    print(f"Attack type: {args.attack_type}")
    print(f"Benchmark version: {DEFAULT_BENCHMARK_VERSION}")
    if args.num_user_tasks:
        print(f"User tasks limit: {args.num_user_tasks} per suite")
    if args.num_injection_tasks:
        print(f"Injection tasks limit: {args.num_injection_tasks} per suite")

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Run evaluation
    results = run_real_evaluation(
        suites=suites,
        num_user_tasks=args.num_user_tasks,
        num_injection_tasks=args.num_injection_tasks,
        model=args.model,
        attack_type=args.attack_type,
        verbose=not args.quiet,
    )

    # Print results table
    print_results_table(results)

    # Save results
    save_results(results, args.results_dir)

    # Generate charts
    if not args.no_charts:
        print("\n--- Generating Visualizations ---")
        try:
            visualizer = ResultsVisualizer(results_dir=args.results_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = visualizer.generate_all_charts(results, prefix=f"{timestamp}_")
            print(f"Generated {len(saved_files)} visualization(s)")
        except Exception as e:
            print(f"Warning: Could not generate charts: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print("\n--- Key Findings ---\n")
    for suite, configs in results.items():
        baseline = configs.get("baseline")
        defended = configs.get("defended")

        if baseline and defended:
            asr_reduction = (baseline.attack_success_rate - defended.attack_success_rate) * 100
            utility_delta = (defended.benign_utility - baseline.benign_utility) * 100

            print(f"{suite.upper()} Suite:")
            print(f"  - ASR: {baseline.attack_success_rate*100:.1f}% -> {defended.attack_success_rate*100:.1f}% ({asr_reduction:+.1f}pp)")
            print(f"  - Benign Utility: {baseline.benign_utility*100:.1f}% -> {defended.benign_utility*100:.1f}% ({utility_delta:+.1f}pp)")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
