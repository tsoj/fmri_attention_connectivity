#!/usr/bin/env python3
"""
Compare R² distributions between Granger causality and attention-based predictions.

This script loads R² values from two result directories:
1. Granger causality results: subject_id directories containing granger_metrics.json
2. Attention-based results: subject_id directories containing info.json

Creates histogram comparisons of the R² distributions.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from utils import check_assert_enabled


def load_granger_r2_values(granger_dir: Path) -> Dict[str, float]:
    """
    Load R² values from Granger causality results.

    Args:
        granger_dir: Directory containing subject_id subdirectories with granger_metrics.json

    Returns:
        Dict mapping subject_id to R² value
    """
    r2_values = {}

    for subject_dir in granger_dir.iterdir():
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name
        granger_metrics_path = subject_dir / "granger_metrics.json"

        if not granger_metrics_path.exists():
            print(f"Warning: granger_metrics.json not found for subject {subject_id}")
            continue

        with open(granger_metrics_path, 'r') as f:
            metrics = json.load(f)

        assert "r2" in metrics, f"No 'r2' field in granger_metrics.json for subject {subject_id}"
        r2_values[subject_id] = metrics["r2"]

    return r2_values


def load_attention_r2_values(attention_dir: Path) -> Dict[str, float]:
    """
    Load R² values from attention-based results.

    Args:
        attention_dir: Directory containing subject_id subdirectories with info.json

    Returns:
        Dict mapping subject_id to R² value
    """
    r2_values = {}

    for subject_dir in attention_dir.iterdir():
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name
        info_path = subject_dir / "info.json"

        if not info_path.exists():
            print(f"Warning: info.json not found for subject {subject_id}")
            continue

        with open(info_path, 'r') as f:
            info = json.load(f)

        assert "metrics" in info, f"No 'metrics' field in info.json for subject {subject_id}"
        assert "r2" in info["metrics"], f"No 'r2' field in metrics for subject {subject_id}"
        r2_values[subject_id] = info["metrics"]["r2"]

    return r2_values




def create_histogram_comparison(granger_r2: Dict[str, float],
                              attention_r2: Dict[str, float],
                              output_dir: Path) -> None:
    """
    Create a clear scatter plot showing attention R² vs granger R² with color coding.

    Args:
        granger_r2: R² values from Granger causality
        attention_r2: R² values from attention-based predictions
        output_dir: Directory to save plots
    """
    # Find common subjects
    common_subjects = set(granger_r2.keys()) & set(attention_r2.keys())

    if not common_subjects:
        print("Warning: No common subjects found between the two methods")
        return

    print(f"Found {len(common_subjects)} common subjects")

    # Extract R² values for common subjects
    granger_values = np.array([granger_r2[subject] for subject in common_subjects])
    attention_values = np.array([attention_r2[subject] for subject in common_subjects])

    # Calculate overall means
    mean_granger = np.mean(granger_values)
    mean_attention = np.mean(attention_values)
    
    # Calculate percentage where attention outperforms granger
    attention_wins = sum(1 for i in range(len(attention_values)) if attention_values[i] > granger_values[i])
    attention_win_percentage = 100 * attention_wins / len(common_subjects)

    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot colored by which method wins
    colors = ['red' if attention_values[i] > granger_values[i] else 'blue'
              for i in range(len(granger_values))]

    scatter = ax.scatter(granger_values, attention_values,
                        c=colors, alpha=0.4, s=50, edgecolors='black', linewidth=0.5)

    # Add diagonal line (y=x)
    min_val = min(min(granger_values), min(attention_values))
    max_val = max(max(granger_values), max(attention_values))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='y=x (Equal Performance)')

    # Add shaded regions
    ax.fill_between([min_val, max_val], [min_val, max_val], [max_val, max_val],
                    color='red', alpha=0.1, label='Attention Better')
    ax.fill_between([min_val, max_val], [min_val, min_val], [min_val, max_val],
                    color='blue', alpha=0.1, label='Granger Better')

    # Labels and title
    ax.set_xlabel('Granger Causality R²', fontsize=12)
    ax.set_ylabel('Attention-based R²', fontsize=12)
    ax.set_title('Prediction R² Comparison: Attention vs Granger Causality', fontsize=14, fontweight='bold')

    # Add text box with mean R² values and win percentage
    textstr = f'Mean R² Scores:\n'
    textstr += f'Attention: {mean_attention:.4f}\n'
    textstr += f'Granger: {mean_granger:.4f}\n\n'
    textstr += f'Subjects where Attention\noutperforms Granger:\n{attention_wins}/{len(common_subjects)} ({attention_win_percentage:.1f}%)'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # Add legend
    ax.legend(loc='lower right', fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Make axes equal for better comparison
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save plot
    output_path = output_dir / 'r2_scatter_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved R² scatter comparison to: {output_path}")


def print_summary_statistics(granger_r2: Dict[str, float],
                           attention_r2: Dict[str, float]) -> None:
    """
    Print summary statistics for both methods.
    """
    # Find common subjects
    common_subjects = set(granger_r2.keys()) & set(attention_r2.keys())

    if not common_subjects:
        print("No common subjects for statistics comparison")
        return

    granger_values = [granger_r2[subject] for subject in common_subjects]
    attention_values = [attention_r2[subject] for subject in common_subjects]
    differences = [attention_r2[subject] - granger_r2[subject] for subject in common_subjects]

    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Number of common subjects: {len(common_subjects)}")

    print(f"\nGranger Causality R² Statistics:")
    print(f"  Mean: {np.mean(granger_values):.6f}")
    print(f"  Std:  {np.std(granger_values):.6f}")
    print(f"  Min:  {np.min(granger_values):.6f}")
    print(f"  Max:  {np.max(granger_values):.6f}")

    print(f"\nAttention-based R² Statistics:")
    print(f"  Mean: {np.mean(attention_values):.6f}")
    print(f"  Std:  {np.std(attention_values):.6f}")
    print(f"  Min:  {np.min(attention_values):.6f}")
    print(f"  Max:  {np.max(attention_values):.6f}")

    print(f"\nDifference Statistics (Attention - Granger):")
    print(f"  Mean: {np.mean(differences):.6f}")
    print(f"  Std:  {np.std(differences):.6f}")
    print(f"  Min:  {np.min(differences):.6f}")
    print(f"  Max:  {np.max(differences):.6f}")

    # Count improvements
    improvements = sum(1 for d in differences if d > 0)
    print(f"\nSubjects where Attention outperforms Granger: {improvements}/{len(common_subjects)} ({100*improvements/len(common_subjects):.1f}%)")


def main():
    check_assert_enabled()

    parser = argparse.ArgumentParser(
        description='Compare R² distributions between Granger causality and attention-based predictions'
    )
    parser.add_argument('granger_dir', type=str,
                       help='Directory containing Granger causality results (subject_id subdirs with granger_metrics.json)')
    parser.add_argument('attention_dir', type=str,
                       help='Directory containing attention-based results (subject_id subdirs with info.json)')
    parser.add_argument('--output-dir', type=str, default='r2_comparison_plots',
                       help='Directory to save comparison plots (default: r2_comparison_plots)')

    args = parser.parse_args()

    # Convert to Path objects
    granger_dir = Path(args.granger_dir)
    attention_dir = Path(args.attention_dir)
    output_dir = Path(args.output_dir)

    # Validate input directories
    assert granger_dir.exists(), f"Granger directory does not exist: {granger_dir}"
    assert attention_dir.exists(), f"Attention directory does not exist: {attention_dir}"
    assert granger_dir.is_dir(), f"Granger path is not a directory: {granger_dir}"
    assert attention_dir.is_dir(), f"Attention path is not a directory: {attention_dir}"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load R² values
    print(f"Loading Granger causality R² values from: {granger_dir}")
    granger_r2 = load_granger_r2_values(granger_dir)
    print(f"Loaded {len(granger_r2)} Granger subjects")

    print(f"Loading attention-based R² values from: {attention_dir}")
    attention_r2 = load_attention_r2_values(attention_dir)
    print(f"Loaded {len(attention_r2)} attention subjects")

    # Ensure we have data
    assert granger_r2, "No Granger causality R² values loaded"
    assert attention_r2, "No attention-based R² values loaded"

    # Create scatter comparison plot
    print("\nCreating scatter comparison plot...")
    create_histogram_comparison(granger_r2, attention_r2, output_dir)

    # Print summary statistics
    print_summary_statistics(granger_r2, attention_r2)

    print(f"\nComparison complete! Plot saved to: {output_dir}/r2_scatter_comparison.png")


if __name__ == "__main__":
    main()
