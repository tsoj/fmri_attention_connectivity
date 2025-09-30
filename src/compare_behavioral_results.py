#!/usr/bin/env python3
"""
Compare results from multiple behavioral connectivity runs.

This script loads results from different connectivity methods (Pearson, Partial, Granger, Attention)
and creates comparison plots showing:
1. Which method performs best most frequently
2. Performance of each method across all targets
3. Head-to-head comparisons between attention and other methods
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import check_assert_enabled


def load_run_metadata(result_dir: Path) -> Dict:
    """Load metadata.json from a result directory."""
    metadata_path = result_dir / "experiment_metadata.json"
    assert metadata_path.exists(), f"experiment_metadata.json not found in {result_dir}"

    with open(metadata_path, 'r') as f:
        return json.load(f)


def get_method_name(matrix_pattern: str, attention_type: Optional[str] = None) -> str:
    """Convert matrix pattern filename to method name."""
    pattern_to_method = {
        "pearson_correlation.npy": "Pearson",
        "partial_correlation.npy": "Partial",
        "granger_causality.npy": "Granger",
        "mean_attention.npy": "Attention"
    }

    method = pattern_to_method.get(matrix_pattern)
    assert method is not None, f"Unknown matrix pattern: {matrix_pattern}"
    
    # Handle attention type specification
    if method == "Attention" and attention_type:
        if attention_type == "group":
            method = "Group attention"
        elif attention_type == "subject":
            method = "Subject attention"
    
    return method


def load_target_results(result_dir: Path) -> Dict[str, Dict]:
    """Load all target result JSON files from a directory."""
    results = {}

    for json_path in result_dir.glob("*_results.json"):
        # Skip the summary file
        if json_path.name == "all_results_summary.json":
            continue

        target_name = json_path.stem.replace("_results", "")

        with open(json_path, 'r') as f:
            data = json.load(f)
            results[target_name] = data

    return results


def extract_r2_scores(results: Dict[str, Dict[Path, Dict]]) -> Dict[str, Dict[str, float]]:
    """
    Extract r^2 scores from results.

    Returns:
        Dict mapping method_name -> target_name -> r2_score
    """
    r2_scores = defaultdict(dict)

    for method_name, target_results in results.items():
        for target_name, target_data in target_results.items():
            # Extract r^2 from overall_metrics
            r2 = target_data['overall_metrics'].get('r_squared', 0.0)
            r2_scores[method_name][target_name] = r2

    return dict(r2_scores)


def get_method_frequency_and_colors(r2_scores: Dict[str, Dict[str, float]]):
    """Calculate frequency of best performance for each method and assign colors."""
    # Count best method for each target
    best_counts = defaultdict(int)

    # Get all targets (assuming all methods have same targets)
    all_targets = set()
    for method_scores in r2_scores.values():
        all_targets.update(method_scores.keys())

    for target in all_targets:
        best_r2 = -np.inf
        best_method = None

        for method, scores in r2_scores.items():
            if target in scores and scores[target] > best_r2:
                best_r2 = scores[target]
                best_method = method

        if best_method:
            best_counts[best_method] += 1

    # Separate methods into attention and non-attention
    all_methods = list(best_counts.keys())
    attention_methods = [m for m in all_methods if 'attention' in m.lower()]
    other_methods = [m for m in all_methods if 'attention' not in m.lower()]
    
    # Sort by frequency (descending) within each category
    attention_methods.sort(key=lambda m: best_counts[m], reverse=True)
    other_methods.sort(key=lambda m: best_counts[m], reverse=True)
    
    # Assign colors based on ordering within categories
    method_colors = {}
    
    # Attention colors (red shades) - lighter for higher frequency
    if len(attention_methods) >= 2:
        method_colors[attention_methods[0]] = '#FF6B6B'  # Light red for highest frequency
        method_colors[attention_methods[1]] = '#B22222'  # Dark red for lower frequency
    elif len(attention_methods) == 1:
        method_colors[attention_methods[0]] = '#FF6B6B'  # Light red
    
    # Other method colors (blue shades) - lighter for higher frequency
    blue_colors = ['#87CEEB', '#4682B4', '#1e3a5f']  # Light blue, normal blue, dark blue
    for i, method in enumerate(other_methods):
        if i < len(blue_colors):
            method_colors[method] = blue_colors[i]
    
    return best_counts, method_colors


def plot_best_method_frequency(r2_scores: Dict[str, Dict[str, float]], output_dir: Path):
    """Plot how often each method performs best across targets."""
    # Get frequency counts and colors
    best_counts, method_colors = get_method_frequency_and_colors(r2_scores)
    
    # Order methods: attention versions first, then others
    all_methods = list(best_counts.keys())
    attention_methods = [m for m in all_methods if 'attention' in m.lower()]
    other_methods = [m for m in all_methods if 'attention' not in m.lower()]
    
    # Sort attention methods by count (descending), then other methods by count (descending)
    attention_methods.sort(key=lambda m: best_counts[m], reverse=True)
    other_methods.sort(key=lambda m: best_counts[m], reverse=True)
    
    # Final ordering
    ordered_methods = attention_methods + other_methods
    ordered_counts = [best_counts[m] for m in ordered_methods]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(ordered_methods, ordered_counts)

    # Apply colors
    for bar, method in zip(bars, ordered_methods):
        bar.set_color(method_colors.get(method, '#808080'))

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Number of Targets Where Method is Best', fontsize=12)
    ax.set_title('Frequency of Best Performance Across Targets', fontsize=14, fontweight='bold')

    # Add count labels on bars
    for bar, count in zip(bars, ordered_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    # Add vertical grid lines for better readability
    x_pos = range(len(ordered_methods))
    for x in x_pos:
        ax.axvline(x=x, color='gray', linestyle='-', alpha=0.1, linewidth=0.5)

    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'best_method_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved best method frequency plot to {output_dir / 'best_method_frequency.png'}")


def plot_all_methods_grouped_by_best(r2_scores: Dict[str, Dict[str, float]], output_dir: Path):
    """
    Plot all methods grouped by which method performs best for each target.
    Creates separate plots for each best-performing method group.
    """
    # Get frequency counts and colors based on overall performance
    best_counts, method_colors = get_method_frequency_and_colors(r2_scores)
    
    # Get all targets and methods
    all_targets = set()
    methods = list(r2_scores.keys())
    for method_scores in r2_scores.values():
        all_targets.update(method_scores.keys())
    
    if not all_targets or not methods:
        print("No data available for grouped comparison plot")
        return
    
    # Order methods: attention versions first (by frequency), then others (by frequency)
    attention_methods = [m for m in methods if 'attention' in m.lower()]
    other_methods = [m for m in methods if 'attention' not in m.lower()]
    
    # Sort by frequency (descending)
    attention_methods.sort(key=lambda m: best_counts.get(m, 0), reverse=True)
    other_methods.sort(key=lambda m: best_counts.get(m, 0), reverse=True)
    
    # Final order for bar positioning
    ordered_methods = attention_methods + other_methods
    
    # Group targets by best performing method
    method_groups = defaultdict(list)  # best_method -> [(target, scores_dict)]
    
    for target in all_targets:
        best_r2 = -np.inf
        best_method = None
        target_scores = {}
        
        # Get scores for this target from all methods
        for method in methods:
            if target in r2_scores[method]:
                score = r2_scores[method][target]
                target_scores[method] = score
                if score > best_r2:
                    best_r2 = score
                    best_method = method
        
        if best_method and target_scores:
            method_groups[best_method].append((target, target_scores))
    
    # Sort targets within each group by best method's score (descending)
    for method in method_groups:
        method_groups[method].sort(key=lambda x: x[1][method], reverse=True)
    
    # Create a separate plot for each best-performing method group
    for best_method in sorted(method_groups.keys()):
        group_targets = method_groups[best_method]
        
        if not group_targets:
            continue
        
        # Prepare data for this group
        targets = [t[0] for t in group_targets]
        n_targets = len(targets)
        n_methods = len(ordered_methods)
        
        # Create figure
        fig_width = max(12, n_targets * 0.8)
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        # Set up bar positions
        x_pos = np.arange(n_targets)
        width = 0.8 / n_methods  # Total width of 0.8 divided by number of methods
        
        # Plot bars for each method in the specified order
        for i, method in enumerate(ordered_methods):
            method_scores = []
            for target, scores_dict in group_targets:
                score = scores_dict.get(method, 0.0)  # Default to 0 if method doesn't have this target
                method_scores.append(score)
            
            offset = (i - (n_methods - 1) / 2) * width
            
            bars = ax.bar(x_pos + offset, method_scores, width, 
                         label=method, color=method_colors.get(method, '#808080'),
                         alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Target', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title(f'Methods Comparison - Best: {best_method} ({n_targets} targets)', 
                    fontsize=14, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(targets, rotation=45, ha='right')
        
        # Add reference lines
        ax.axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        
        # Add vertical grid lines for better readability
        for x in x_pos:
            ax.axvline(x=x, color='gray', linestyle='-', alpha=0.1, linewidth=0.5)
        
        # Legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Set y-axis limits
        all_scores = []
        for _, scores_dict in group_targets:
            all_scores.extend(scores_dict.values())
        if all_scores:
            ax.set_ylim(bottom=min(0, min(all_scores) - 0.05),
                       top=max(all_scores) + 0.05)
        
        plt.tight_layout()
        
        # Save with method name in filename
        safe_method_name = best_method.replace(' ', '_').lower()
        filename = f'methods_comparison_best_{safe_method_name}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {best_method} group comparison to {output_dir / filename}")


def main():
    check_assert_enabled()

    parser = argparse.ArgumentParser(description='Compare behavioral connectivity results')
    parser.add_argument('result_dirs', nargs='+', type=str,
                       help='Paths to result directories from behavioral_connectivity.py runs')
    parser.add_argument('--output-dir', type=str, default='comparison_plots',
                       help='Directory to save comparison plots (default: comparison_plots)')
    parser.add_argument('--group-level', type=str, default=None,
                       help='Path to result directory for group-level attention')
    parser.add_argument('--subject-level', type=str, default=None,
                       help='Path to result directory for subject-level attention')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load metadata and results from each directory
    all_results = {}  # method_name -> target_name -> result_data

    # Process attention directories with specified types first
    attention_dirs = []
    if args.group_level:
        attention_dirs.append((args.group_level, "group"))
    if args.subject_level:
        attention_dirs.append((args.subject_level, "subject"))
    
    for result_dir_str, attention_type in attention_dirs:
        result_dir = Path(result_dir_str)
        assert result_dir.exists(), f"Result directory not found: {result_dir}"

        print(f"\nLoading results from: {result_dir}")

        # Load metadata to identify method
        metadata = load_run_metadata(result_dir)
        matrix_pattern = metadata.get('matrix_pattern')
        assert matrix_pattern, f"No matrix_pattern found in metadata for {result_dir}"
        assert matrix_pattern == "mean_attention.npy", f"Expected mean_attention.npy for {attention_type}-level, got {matrix_pattern}"

        method_name = get_method_name(matrix_pattern, attention_type=attention_type)
        print(f"  Method: {method_name} (pattern: {matrix_pattern})")

        # Check for duplicate methods
        assert method_name not in all_results, f"Duplicate method {method_name} found. Each method should appear only once."

        # Load target results
        target_results = load_target_results(result_dir)
        print(f"  Loaded {len(target_results)} target results")

        all_results[method_name] = target_results

    # Process other directories
    for result_dir_str in args.result_dirs:
        result_dir = Path(result_dir_str)
        assert result_dir.exists(), f"Result directory not found: {result_dir}"

        print(f"\nLoading results from: {result_dir}")

        # Load metadata to identify method
        metadata = load_run_metadata(result_dir)
        matrix_pattern = metadata.get('matrix_pattern')
        assert matrix_pattern, f"No matrix_pattern found in metadata for {result_dir}"

        # Skip if it's an attention pattern (handled above)
        if matrix_pattern == "mean_attention.npy":
            continue
            
        method_name = get_method_name(matrix_pattern)
        print(f"  Method: {method_name} (pattern: {matrix_pattern})")

        # Check for duplicate methods
        assert method_name not in all_results, f"Duplicate method {method_name} found. Each method should appear only once."

        # Load target results
        target_results = load_target_results(result_dir)
        print(f"  Loaded {len(target_results)} target results")

        all_results[method_name] = target_results

    # Extract r^2 scores
    r2_scores = extract_r2_scores(all_results)

    # Verify we have data
    assert r2_scores, "No r^2 scores extracted from results"

    print(f"\nCreating comparison plots...")

    # 1. Plot frequency of best method
    plot_best_method_frequency(r2_scores, output_dir)

    # 2. Plot all methods grouped by best performing method
    plot_all_methods_grouped_by_best(r2_scores, output_dir)

    print(f"\nAll plots saved to: {output_dir}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
