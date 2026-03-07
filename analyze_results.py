#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Result Analysis Script for Fairness-aware Federated Learning
=============================================================

This script analyzes and visualizes the experimental results from:
  - FedAvg
  - FedALA
  - FairFed
  - FedALAFair

It generates comparison plots and statistical summaries.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def load_results(algorithm, metric, goal="fairness_experiment", dataset="Adult"):
    """
    Load experiment results from HDF5 files

    Args:
        algorithm: Algorithm name (e.g., 'FedAvg', 'FedALA', 'FairFed', 'FedALAFair')
        metric: Metric to load (e.g., 'test_acc', 'train_acc', 'train_loss')
        goal: Experiment goal string
        dataset: Dataset name

    Returns:
        numpy array of results across multiple runs
    """
    results_dir = Path("system/results")
    file_pattern = f"{dataset}_{algorithm}_{goal}_{metric}.h5"
    file_path = results_dir / file_pattern

    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return None

    try:
        with h5py.File(file_path, "r") as f:
            # Load all runs
            data = []
            for key in f.keys():
                data.append(f[key][()])
            return np.array(data)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def plot_comparison(
    algorithms, metric="test_acc", goal="fairness_experiment", dataset="Adult"
):
    """
    Plot comparison of algorithms for a specific metric
    """
    plt.figure(figsize=(12, 7))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, algo in enumerate(algorithms):
        data = load_results(algo, metric, goal, dataset)

        if data is None:
            continue

        # Compute mean and std across runs
        mean_values = np.mean(data, axis=0)
        std_values = np.std(data, axis=0)
        rounds = np.arange(len(mean_values))

        # Plot with confidence interval
        plt.plot(rounds, mean_values, label=algo, color=colors[i], linewidth=2)
        plt.fill_between(
            rounds,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.2,
            color=colors[i],
        )

    plt.xlabel("Communication Round", fontsize=14)

    if metric == "test_acc":
        plt.ylabel("Test Accuracy", fontsize=14)
        plt.title(
            "Test Accuracy Comparison on Adult Dataset", fontsize=16, fontweight="bold"
        )
    elif metric == "train_acc":
        plt.ylabel("Training Accuracy", fontsize=14)
        plt.title(
            "Training Accuracy Comparison on Adult Dataset",
            fontsize=16,
            fontweight="bold",
        )
    elif metric == "train_loss":
        plt.ylabel("Training Loss", fontsize=14)
        plt.title(
            "Training Loss Comparison on Adult Dataset", fontsize=16, fontweight="bold"
        )

    plt.legend(fontsize=12, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{metric}_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / f"{metric}_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / f'{metric}_comparison.png'}")

    plt.show()


def compute_statistics(
    algorithms, metric="test_acc", goal="fairness_experiment", dataset="Adult"
):
    """
    Compute and display statistical summary
    """
    print("\n" + "=" * 80)
    print(f"Statistical Summary: {metric.upper()}")
    print("=" * 80)
    print(f"{'Algorithm':<20} {'Mean':<15} {'Std':<15} {'Max':<15} {'Final':<15}")
    print("-" * 80)

    results_summary = {}

    for algo in algorithms:
        data = load_results(algo, metric, goal, dataset)

        if data is None:
            print(f"{algo:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
            continue

        # Statistics across all runs and rounds
        mean_all = np.mean(data)
        std_all = np.std(data)
        max_all = np.max(data)
        final_mean = np.mean(data[:, -1])  # Mean of final round across runs

        results_summary[algo] = {
            "mean": mean_all,
            "std": std_all,
            "max": max_all,
            "final": final_mean,
        }

        print(
            f"{algo:<20} {mean_all:<15.4f} {std_all:<15.4f} {max_all:<15.4f} {final_mean:<15.4f}"
        )

    print("=" * 80 + "\n")

    return results_summary


def plot_final_comparison(algorithms, goal="fairness_experiment", dataset="Adult"):
    """
    Create bar plot comparing final performance
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ["test_acc", "train_acc", "train_loss"]
    metric_names = ["Test Accuracy", "Train Accuracy", "Train Loss"]

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        final_values = []
        final_stds = []
        algo_names = []

        for algo in algorithms:
            data = load_results(algo, metric, goal, dataset)

            if data is None:
                continue

            # Get final round values across all runs
            final_round = data[:, -1]
            final_values.append(np.mean(final_round))
            final_stds.append(np.std(final_round))
            algo_names.append(algo)

        # Create bar plot
        x_pos = np.arange(len(algo_names))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][: len(algo_names)]

        bars = ax.bar(
            x_pos, final_values, yerr=final_stds, color=colors, alpha=0.7, capsize=5
        )

        ax.set_xlabel("Algorithm", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"Final {metric_name}", fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algo_names, rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, (bar, val, std) in enumerate(zip(bars, final_values, final_stds)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()

    # Save figure
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "final_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "final_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'final_comparison.png'}")

    plt.show()


def main():
    """
    Main analysis function
    """
    print("\n" + "=" * 80)
    print("Fairness-aware Federated Learning - Result Analysis")
    print("=" * 80 + "\n")

    # Define algorithms to compare
    algorithms = ["FedAvg", "FedALA", "FairFed", "FedALAFair"]

    # Check if results directory exists
    results_dir = Path("system/results")
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run experiments first using run_experiments.sh")
        return

    print("Available result files:")
    for f in sorted(results_dir.glob("Adult_*_fairness_experiment_*.h5")):
        print(f"  - {f.name}")
    print()

    # Plot comparisons for each metric
    print("Generating comparison plots...")
    print("-" * 80)

    plot_comparison(algorithms, metric="test_acc")
    plot_comparison(algorithms, metric="train_acc")
    plot_comparison(algorithms, metric="train_loss")

    # Plot final comparison
    print("\nGenerating final performance comparison...")
    print("-" * 80)
    plot_final_comparison(algorithms)

    # Compute and display statistics
    for metric in ["test_acc", "train_acc", "train_loss"]:
        compute_statistics(algorithms, metric=metric)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("\nResults saved to: analysis_results/")
    print("\nGenerated files:")
    print("  - test_acc_comparison.png/pdf")
    print("  - train_acc_comparison.png/pdf")
    print("  - train_loss_comparison.png/pdf")
    print("  - final_comparison.png/pdf")
    print("\n")


if __name__ == "__main__":
    main()
