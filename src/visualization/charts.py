"""
Visualization module for evaluation results.

Creates charts and plots for:
- Baseline vs defended comparison
- Metrics across suites
- ASR reduction analysis
"""

import os
import json
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from ..evaluation.metrics import EvaluationResult


class ResultsVisualizer:
    """
    Creates visualizations for evaluation results.
    """

    def __init__(
        self,
        results_dir: str = "results",
        style: str = "seaborn-v0_8-whitegrid",
    ):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn" in plt.style.available else "ggplot")

        # Color scheme
        self.colors = {
            "baseline": "#E74C3C",  # Red
            "defended": "#27AE60",  # Green
            "benign_utility": "#3498DB",  # Blue
            "utility_under_attack": "#9B59B6",  # Purple
            "asr": "#E74C3C",  # Red
        }

    def plot_metrics_comparison(
        self,
        results: Dict[str, Dict[str, EvaluationResult]],
        save_path: Optional[str] = None,
    ):
        """
        Create bar chart comparing baseline vs defended metrics.

        Shows all three metrics side by side for each suite.
        """
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))
        if len(results) == 1:
            axes = [axes]

        for idx, (suite, configs) in enumerate(results.items()):
            ax = axes[idx]

            baseline = configs.get("baseline")
            defended = configs.get("defended")

            metrics = ["Benign Utility", "Utility Under Attack", "ASR"]
            x = np.arange(len(metrics))
            width = 0.35

            baseline_values = [
                baseline.benign_utility * 100 if baseline else 0,
                baseline.utility_under_attack * 100 if baseline else 0,
                baseline.attack_success_rate * 100 if baseline else 0,
            ]

            defended_values = [
                defended.benign_utility * 100 if defended else 0,
                defended.utility_under_attack * 100 if defended else 0,
                defended.attack_success_rate * 100 if defended else 0,
            ]

            bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline',
                          color=self.colors["baseline"], alpha=0.8)
            bars2 = ax.bar(x + width/2, defended_values, width, label='Defended',
                          color=self.colors["defended"], alpha=0.8)

            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_title(f'{suite.replace("_", " ").title()} Suite', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, fontsize=10)
            ax.legend(fontsize=10)
            ax.set_ylim(0, 100)

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_asr_reduction(
        self,
        results: Dict[str, Dict[str, EvaluationResult]],
        save_path: Optional[str] = None,
    ):
        """
        Create a chart showing ASR reduction from baseline to defended.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        suites = list(results.keys())
        x = np.arange(len(suites))
        width = 0.35

        baseline_asr = []
        defended_asr = []
        reductions = []

        for suite in suites:
            baseline = results[suite].get("baseline")
            defended = results[suite].get("defended")

            b_asr = baseline.attack_success_rate * 100 if baseline else 0
            d_asr = defended.attack_success_rate * 100 if defended else 0

            baseline_asr.append(b_asr)
            defended_asr.append(d_asr)
            reductions.append(b_asr - d_asr)

        # Bar chart
        bars1 = ax.bar(x - width/2, baseline_asr, width, label='Baseline ASR',
                      color=self.colors["baseline"], alpha=0.8)
        bars2 = ax.bar(x + width/2, defended_asr, width, label='Defended ASR',
                      color=self.colors["defended"], alpha=0.8)

        ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
        ax.set_title('ASR Reduction: Baseline vs Defended', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", " ").title() for s in suites], fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 100)

        # Add reduction annotations
        for i, (b, d, r) in enumerate(zip(baseline_asr, defended_asr, reductions)):
            ax.annotate(f'{b:.1f}%', xy=(i - width/2, b + 2), ha='center', fontsize=10)
            ax.annotate(f'{d:.1f}%', xy=(i + width/2, d + 2), ha='center', fontsize=10)

            # Draw arrow showing reduction
            if r > 0:
                mid_x = i
                ax.annotate('', xy=(mid_x, d + 8), xytext=(mid_x, b - 2),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
                ax.annotate(f'-{r:.0f}pp', xy=(mid_x + 0.15, (b + d) / 2),
                           fontsize=11, fontweight='bold', color='green')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_security_utility_tradeoff(
        self,
        results: Dict[str, Dict[str, EvaluationResult]],
        save_path: Optional[str] = None,
    ):
        """
        Create a scatter plot showing security/utility tradeoff.

        X-axis: Benign Utility (higher is better)
        Y-axis: ASR (lower is better - inverted)
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        markers = {'workspace': 'o', 'banking': 's'}
        colors = {'baseline': self.colors["baseline"], 'defended': self.colors["defended"]}

        for suite, configs in results.items():
            for config_name, result in configs.items():
                x = result.benign_utility * 100
                y = (1 - result.attack_success_rate) * 100  # Inverted: higher is better (more secure)

                ax.scatter(
                    x, y,
                    s=200,
                    marker=markers.get(suite, 'o'),
                    c=colors.get(config_name, 'gray'),
                    alpha=0.8,
                    edgecolors='black',
                    linewidth=1.5,
                    label=f'{suite.title()} - {config_name.title()}',
                )

                # Add label
                offset = (5, 5) if config_name == "defended" else (-5, -10)
                ax.annotate(
                    f'{config_name[:3].upper()}',
                    xy=(x, y),
                    xytext=offset,
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                )

        ax.set_xlabel('Benign Utility (%)', fontsize=12)
        ax.set_ylabel('Security (100 - ASR) (%)', fontsize=12)
        ax.set_title('Security vs Utility Tradeoff', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        # Add ideal region shading
        ax.fill_between([60, 100], [60, 60], [100, 100], alpha=0.1, color='green')
        ax.annotate('Ideal Region', xy=(80, 90), fontsize=10, color='green', alpha=0.7)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=10)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_detailed_results_table(
        self,
        results: Dict[str, Dict[str, EvaluationResult]],
        save_path: Optional[str] = None,
    ):
        """
        Create a formatted table image of results.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')

        # Prepare data
        table_data = []
        for suite, configs in results.items():
            baseline = configs.get("baseline")
            defended = configs.get("defended")

            if baseline and defended:
                delta_asr = (defended.attack_success_rate - baseline.attack_success_rate) * 100

                table_data.append([
                    suite.replace("_", " ").title(),
                    f"{baseline.benign_utility*100:.1f}%",
                    f"{baseline.utility_under_attack*100:.1f}%",
                    f"{baseline.attack_success_rate*100:.1f}%",
                    f"{defended.benign_utility*100:.1f}%",
                    f"{defended.utility_under_attack*100:.1f}%",
                    f"{defended.attack_success_rate*100:.1f}%",
                    f"{delta_asr:+.1f}pp",
                ])

        # Create table
        columns = [
            'Suite',
            'BU (Base)', 'UUA (Base)', 'ASR (Base)',
            'BU (Def)', 'UUA (Def)', 'ASR (Def)',
            'Delta ASR'
        ]

        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#2C3E50')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        # Color code delta ASR
        for i, row in enumerate(table_data, start=1):
            delta = float(row[-1].replace('pp', '').replace('+', ''))
            if delta < 0:
                table[(i, 7)].set_facecolor('#27AE60')
                table[(i, 7)].set_text_props(color='white', fontweight='bold')
            else:
                table[(i, 7)].set_facecolor('#E74C3C')
                table[(i, 7)].set_text_props(color='white')

        ax.set_title('Evaluation Results Summary\n(BU=Benign Utility, UUA=Utility Under Attack, ASR=Attack Success Rate)',
                    fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def generate_all_charts(
        self,
        results: Dict[str, Dict[str, EvaluationResult]],
        prefix: str = "",
    ) -> List[str]:
        """
        Generate all charts and save them.

        Returns list of saved file paths.
        """
        saved_files = []

        # Metrics comparison
        path = os.path.join(self.results_dir, f"{prefix}metrics_comparison.png")
        self.plot_metrics_comparison(results, save_path=path)
        saved_files.append(path)

        # ASR reduction
        path = os.path.join(self.results_dir, f"{prefix}asr_reduction.png")
        self.plot_asr_reduction(results, save_path=path)
        saved_files.append(path)

        # Security-utility tradeoff
        path = os.path.join(self.results_dir, f"{prefix}security_utility_tradeoff.png")
        self.plot_security_utility_tradeoff(results, save_path=path)
        saved_files.append(path)

        # Results table
        path = os.path.join(self.results_dir, f"{prefix}results_table.png")
        self.plot_detailed_results_table(results, save_path=path)
        saved_files.append(path)

        plt.close('all')

        return saved_files

    def load_results_from_json(self, filepath: str) -> Dict[str, Dict[str, EvaluationResult]]:
        """Load results from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        results = {}
        for suite, configs in data.items():
            results[suite] = {}
            for config_name, metrics in configs.items():
                results[suite][config_name] = EvaluationResult(
                    suite=metrics["suite"],
                    config_name=metrics["config_name"],
                    benign_utility=metrics["benign_utility"],
                    utility_under_attack=metrics["utility_under_attack"],
                    attack_success_rate=metrics["attack_success_rate"],
                    total_benign_tasks=metrics["total_benign_tasks"],
                    benign_tasks_completed=metrics["benign_tasks_completed"],
                    total_attack_cases=metrics["total_attack_cases"],
                    attack_cases_task_completed=metrics["attack_cases_task_completed"],
                    attack_cases_succeeded=metrics["attack_cases_succeeded"],
                    total_tools_blocked=metrics.get("total_tools_blocked", 0),
                    total_injections_detected=metrics.get("total_injections_detected", 0),
                )

        return results
