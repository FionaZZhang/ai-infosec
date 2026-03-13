#!/usr/bin/env python3
"""
Main evaluation script for the agent security project.

Runs baseline and defended agents on AgentDojo-style benchmarks,
calculates metrics, and generates visualizations.

Usage:
    python run_evaluation.py [--suites workspace,banking] [--no-charts]
"""

import argparse
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.runner import EvaluationRunner
from src.visualization.charts import ResultsVisualizer
from src.config import DefenseConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run agent security evaluation on AgentDojo benchmarks"
    )
    parser.add_argument(
        "--suites",
        type=str,
        default="workspace,banking",
        help="Comma-separated list of suites to evaluate (default: workspace,banking)"
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
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )

    args = parser.parse_args()

    # Parse suites
    suites = [s.strip() for s in args.suites.split(",")]

    print("=" * 60)
    print("AGENT SECURITY EVALUATION")
    print("Hardening Tool-Using LLM Agents Against Prompt Injection")
    print("=" * 60)
    print(f"\nSuites to evaluate: {suites}")
    print(f"Results directory: {args.results_dir}")
    print()

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Initialize runner
    runner = EvaluationRunner(results_dir=args.results_dir)

    # Run evaluation
    print("\n--- Running Evaluation ---\n")
    results = runner.run_full_evaluation(suites=suites)

    # Print results table
    runner.print_results_table(results)

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.results_dir, f"evaluation_results_{timestamp}.json")
    runner.save_results(results, filename=os.path.basename(json_path))

    # Also save as latest
    runner.save_results(results, filename="evaluation_results_latest.json")

    # Generate charts
    if not args.no_charts:
        print("\n--- Generating Visualizations ---\n")
        visualizer = ResultsVisualizer(results_dir=args.results_dir)
        saved_files = visualizer.generate_all_charts(results, prefix=f"{timestamp}_")

        print(f"\nGenerated {len(saved_files)} visualization(s):")
        for f in saved_files:
            print(f"  - {f}")

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
            print(f"  - ASR reduced from {baseline.attack_success_rate*100:.1f}% to {defended.attack_success_rate*100:.1f}% ({asr_reduction:+.1f}pp)")
            print(f"  - Benign utility: {baseline.benign_utility*100:.1f}% -> {defended.benign_utility*100:.1f}% ({utility_delta:+.1f}pp)")
            print(f"  - Utility under attack improved from {baseline.utility_under_attack*100:.1f}% to {defended.utility_under_attack*100:.1f}%")
            print()

    print("Defense mechanisms applied:")
    print("  1. Tool Permission Scoping (least privilege)")
    print("  2. Sensitive Tool Gating (confirmation for write tools)")
    print("  3. Untrusted Content Quarantine (summarize before returning)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
