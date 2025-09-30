#!/usr/bin/env python3
"""
Compare results from multiple fingerprint connectivity runs.

This script loads results from multiple fingerprint_connectivity.py runs and creates:
1. Comparison plots for Cohen's d and identification accuracy
2. Distribution histograms with dual y-axes for same-subject and cross-subject scores
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from utils import check_assert_enabled, get_git_info, get_script_name


def get_matrix_display_name(matrix_pattern: str, attention_type: str = None) -> str:
    """
    Map matrix pattern filename to display name.
    
    Args:
        matrix_pattern: Matrix pattern from metadata (e.g., "mean_attention.npy")
        attention_type: Type of attention if applicable (subject, group, group-subject)
        
    Returns:
        Display name for the matrix type
    """
    # Extract filename if it's a full path
    if '/' in matrix_pattern:
        matrix_pattern = Path(matrix_pattern).name
    
    # Map to display names
    mapping = {
        "mean_attention.npy": "Attention",
        "partial_correlation.npy": "Partial correlation", 
        "pearson_correlation.npy": "Pearson correlation",
        "granger_causality.npy": "Granger causality"
    }
    
    base_name = mapping.get(matrix_pattern, matrix_pattern)
    
    # Add attention type prefix if applicable
    if base_name == "Attention" and attention_type:
        attention_mapping = {
            "subject": "Subject Attention",
            "group": "Group Attention", 
            "group-subject": "Group-Subject Attention"
        }
        return attention_mapping.get(attention_type, base_name)
    
    return base_name


def get_method_display_name(method: str) -> str:
    """
    Map method name to display name.
    
    Args:
        method: Method from metadata (e.g., "pearson", "mutual_info")
        
    Returns:
        Display name for the method
    """
    mapping = {
        "pearson": "Pearson correlation",
        "mutual_info": "Mutual information"
    }
    
    return mapping.get(method, method)


def load_run_results(result_dir: Path, attention_type: str = None) -> Dict:
    """
    Load results from a fingerprint connectivity run.
    
    Args:
        result_dir: Path to the result directory
        attention_type: Type of attention if applicable (subject, group, group-subject)
        
    Returns:
        Dictionary containing the results
    """
    results_file = result_dir / "results.json"
    metadata_file = result_dir / "experiment_metadata.json"
    
    assert results_file.exists(), f"Results file not found: {results_file}"
    assert metadata_file.exists(), f"Metadata file not found: {metadata_file}"
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Add display name from matrix pattern
    matrix_pattern = metadata.get('matrix_pattern', 'unknown')
    results['display_name'] = get_matrix_display_name(matrix_pattern, attention_type)
    results['run_name'] = result_dir.name
    results['metadata'] = metadata
    results['attention_type'] = attention_type  # Store attention type
    results['is_attention'] = 'attention' in matrix_pattern.lower() if attention_type else False
    
    return results


def plot_metric_comparison(
    run_results: List[Dict],
    output_dir: Path
) -> None:
    """
    Create bar plots comparing Cohen's d and accuracy across runs.
    
    Args:
        run_results: List of result dictionaries from each run
        output_dir: Directory to save the plot
    """
    # Separate attention and non-attention methods
    attention_runs = [r for r in run_results if r.get('attention_type') is not None]
    other_runs = [r for r in run_results if r.get('attention_type') is None]
    
    # Sort each group by performance (Cohen's d), descending
    attention_runs.sort(key=lambda x: x['effect_size_cohens_d'], reverse=True)
    other_runs.sort(key=lambda x: x['effect_size_cohens_d'], reverse=True)
    
    # Combine: attention methods first, then others
    sorted_runs = attention_runs + other_runs
    n_runs = len(sorted_runs)
    
    run_names = [r['display_name'] for r in sorted_runs]
    cohens_d_values = [r['effect_size_cohens_d'] for r in sorted_runs]
    accuracy_values = [r['identification_accuracy'] for r in sorted_runs]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare colors for bars - use three fixed shades
    def get_bar_colors(runs):
        colors = []
        # Define fixed color palettes
        red_shades = ['#ff4444', '#cc0000', '#990000']  # light, normal, dark red
        blue_shades = ['#4d79a4', '#2e5984', '#1e3a5f']  # light, normal, dark blue
        
        attention_idx = 0
        other_idx = 0
        
        for r in runs:
            if r.get('attention_type') is not None:
                # Use red shades for attention methods
                color_idx = attention_idx % len(red_shades)
                colors.append(red_shades[color_idx])
                attention_idx += 1
            else:
                # Use blue shades for other methods
                color_idx = other_idx % len(blue_shades)
                colors.append(blue_shades[color_idx])
                other_idx += 1
        return colors
    
    # Cohen's d plot
    x_pos = np.arange(n_runs)
    colors_cohens = get_bar_colors(sorted_runs)
    bars1 = ax1.bar(x_pos, cohens_d_values, color=colors_cohens, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Method')
    ax1.set_ylabel("Cohen's d")
    ax1.set_title("Effect Size Comparison (Cohen's d)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(run_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars1, cohens_d_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Accuracy plot
    colors_acc = get_bar_colors(sorted_runs)
    bars2 = ax2.bar(x_pos, accuracy_values, color=colors_acc, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Identification Accuracy Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(run_names, rotation=45, ha='right')
    ax2.set_ylim([0, 1.05])  # Set y-axis to 0-100%
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars2, accuracy_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'metric_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved metric comparison to: {output_path}")


def plot_combined_distributions(
    run_results: List[Dict],
    output_dir: Path
) -> None:
    """
    Create combined histogram plots with dual y-axes for each run.
    Creates a 2x2 grid with subject-level attention and three other non-attention methods.
    
    Args:
        run_results: List of result dictionaries from each run
        output_dir: Directory to save the plot
    """
    # Separate subject-level attention and non-attention runs
    subject_attention_runs = [r for r in run_results if r.get('attention_type') == 'subject']
    other_runs = [r for r in run_results if r.get('attention_type') is None]
    
    # We want exactly 1 subject attention and up to 3 other methods for 2x2 grid
    assert len(subject_attention_runs) > 0, "Need at least one subject-level attention result"
    
    # Take first subject attention and up to 3 other methods
    runs_to_plot = [subject_attention_runs[0]] + other_runs[:3]
    n_runs = len(runs_to_plot)
    
    # Check that all runs use the same method
    methods = [r['metadata'].get('method', 'unknown') for r in runs_to_plot]
    unique_methods = set(methods)
    assert len(unique_methods) == 1, f"All runs must use the same method. Found: {unique_methods}"
    
    # Get method display name for the overall title
    method_display = get_method_display_name(methods[0])
    
    # Fixed 2x2 grid
    n_cols = 2
    n_rows = 2
    
    # Pre-calculate histograms to get consistent y-axis limits
    all_cross_counts = []
    all_same_counts = []
    
    for results in runs_to_plot:
        same_subject_scores = results['same_subject_scores']
        cross_subject_scores = results['cross_subject_scores']
        
        # Determine bin edges for this run
        all_scores = same_subject_scores + cross_subject_scores
        bins = np.histogram_bin_edges(all_scores, bins=30)
        
        # Calculate frequency histogram values
        cross_counts, _ = np.histogram(cross_subject_scores, bins=bins)
        same_counts, _ = np.histogram(same_subject_scores, bins=bins)
        
        all_cross_counts.append(cross_counts)
        all_same_counts.append(same_counts)
    
    # Determine consistent y-axis limits for frequency plots
    max_cross_count = max(np.max(counts) for counts in all_cross_counts)
    max_same_count = max(np.max(counts) for counts in all_same_counts)
    
    # Add some padding
    cross_ylim = (0, max_cross_count * 1.1)
    same_ylim = (0, max_same_count * 1.1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Flatten axes array for easier iteration
    axes = axes.flatten()
    
    # Define colors - orange/purple instead of blue/red
    cross_color = '#8b4789'  # Purple
    same_color = '#ff8c00'   # Orange
    
    for idx, results in enumerate(runs_to_plot):
        ax_left = axes[idx]
        
        same_subject_scores = results['same_subject_scores']
        cross_subject_scores = results['cross_subject_scores']
        
        # Determine bin edges for this specific run
        all_scores = same_subject_scores + cross_subject_scores
        bins = np.histogram_bin_edges(all_scores, bins=30)
        
        # Plot cross-subject scores on left y-axis with frequency
        counts_cross, _, patches_cross = ax_left.hist(
            cross_subject_scores, 
            bins=bins.tolist(), 
            alpha=0.6, 
            label=f"Cross-subject (n={len(cross_subject_scores)})", 
            color=cross_color,
            edgecolor='black',
            linewidth=0.5
        )
        
        ax_left.set_xlabel('Similarity Score')
        ax_left.set_ylabel('Cross-subject Frequency', color=cross_color)
        ax_left.tick_params(axis='y', labelcolor=cross_color)
        ax_left.set_ylim(cross_ylim)
        
        # Create second y-axis for same-subject scores
        ax_right = ax_left.twinx()
        counts_same, _, patches_same = ax_right.hist(
            same_subject_scores, 
            bins=bins.tolist(), 
            alpha=0.6, 
            label=f"Same-subject (n={len(same_subject_scores)})", 
            color=same_color,
            edgecolor='black',
            linewidth=0.5
        )
        
        ax_right.set_ylabel('Same-subject Frequency', color=same_color)
        ax_right.tick_params(axis='y', labelcolor=same_color)
        ax_right.set_ylim(same_ylim)
        
        # Add title and grid
        display_name = results['display_name']
        ax_left.set_title(f'{display_name}', fontsize=11)
        ax_left.grid(True, alpha=0.3)
        
        # Add vertical lines for means
        cross_mean = np.mean(cross_subject_scores)
        same_mean = np.mean(same_subject_scores)
        
        ax_left.axvline(cross_mean, color='#5b2c5d', linestyle='--', alpha=0.8, 
                       label=f'Cross mean: {cross_mean:.3f}')  # Darker purple
        ax_left.axvline(same_mean, color='#cc5500', linestyle='--', alpha=0.8,
                       label=f'Same mean: {same_mean:.3f}')  # Darker orange
        
        # Add legend
        lines1, labels1 = ax_left.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        ax_left.legend(lines1[:1] + lines2[:1] + lines1[1:], 
                      labels1[:1] + labels2[:1] + labels1[1:], 
                      loc='upper right', fontsize=9)
        
        # Add Cohen's d and accuracy as text
        cohens_d = results['effect_size_cohens_d']
        accuracy = results['identification_accuracy']
        text_str = f"Cohen's d: {cohens_d:.3f}\nAccuracy: {accuracy:.2%}"
        ax_left.text(0.02, 0.98, text_str, transform=ax_left.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide empty subplots if any
    for idx in range(n_runs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Fingerprinting Score Distributions Comparison ({method_display})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'distribution_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved distribution comparison to: {output_path}")


def create_summary_table(
    run_results: List[Dict],
    output_dir: Path
) -> None:
    """
    Create a summary table of all runs.
    
    Args:
        run_results: List of result dictionaries from each run
        output_dir: Directory to save the table
    """
    summary_data = []
    
    for results in run_results:
        metadata = results['metadata']
        summary_data.append({
            'run_name': results['run_name'],
            'display_name': results['display_name'],
            'method': metadata.get('method', 'unknown'),
            'z_normalize': metadata.get('z_normalize', False),
            'n_subjects': metadata.get('n_subjects', 0),
            'identification_accuracy': results['identification_accuracy'],
            'cohens_d': results['effect_size_cohens_d'],
            'same_subject_mean': results['same_subject_mean'],
            'same_subject_std': results['same_subject_std'],
            'cross_subject_mean': results['cross_subject_mean'],
            'cross_subject_std': results['cross_subject_std'],
            'n_same_pairs': results['n_same_subject_pairs'],
            'n_cross_pairs': results['n_cross_subject_pairs']
        })
    
    # Save as JSON
    summary_path = output_dir / 'summary_table.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary table to: {summary_path}")
    
    # Print formatted table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    for data in summary_data:
        print(f"\n{data['display_name']} ({data['run_name']}):")
        print(f"  Method: {data['method']}, Z-normalize: {data['z_normalize']}")
        print(f"  Subjects: {data['n_subjects']}")
        print(f"  Accuracy: {data['identification_accuracy']:.2%}")
        print(f"  Cohen's d: {data['cohens_d']:.3f}")
        print(f"  Same-subject: {data['same_subject_mean']:.3f} ± {data['same_subject_std']:.3f} (n={data['n_same_pairs']})")
        print(f"  Cross-subject: {data['cross_subject_mean']:.3f} ± {data['cross_subject_std']:.3f} (n={data['n_cross_pairs']})")


def main():
    check_assert_enabled()
    
    parser = argparse.ArgumentParser(
        description="Compare results from multiple fingerprint connectivity runs"
    )
    
    # Create a mutually exclusive group for attention types
    parser.add_argument(
        '--subject-attention',
        action='append',
        type=Path,
        help='Path to subject-level attention result directory'
    )
    parser.add_argument(
        '--group-attention',
        action='append',
        type=Path,
        help='Path to group-level attention result directory'
    )
    parser.add_argument(
        '--group-subject-attention',
        action='append',
        type=Path,
        help='Path to group-subject attention result directory'
    )
    parser.add_argument(
        '--other',
        action='append',
        type=Path,
        help='Path to non-attention method result directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for comparison plots (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Collect all result directories with their attention types
    all_dirs = []
    
    # Validate that we have at least one subject attention
    if not args.subject_attention:
        parser.error("At least one --subject-attention result is required")
    
    # Add attention results
    if args.subject_attention:
        for dir_path in args.subject_attention:
            all_dirs.append((dir_path, 'subject'))
    
    if args.group_attention:
        for dir_path in args.group_attention:
            all_dirs.append((dir_path, 'group'))
            
    if args.group_subject_attention:
        for dir_path in args.group_subject_attention:
            all_dirs.append((dir_path, 'group-subject'))
    
    # Add non-attention results
    if args.other:
        for dir_path in args.other:
            all_dirs.append((dir_path, None))
    
    # Validate directories
    for result_dir, _ in all_dirs:
        assert result_dir.exists(), f"Result directory not found: {result_dir}"
        assert result_dir.is_dir(), f"Not a directory: {result_dir}"
    
    print(f"Loading results from {len(all_dirs)} runs...")
    
    # Load all results
    run_results = []
    for result_dir, attention_type in all_dirs:
        try:
            results = load_run_results(result_dir, attention_type)
            run_results.append(results)
            type_str = f" ({attention_type})" if attention_type else ""
            print(f"  Loaded: {result_dir.name}{type_str}")
        except Exception as e:
            print(f"  Failed to load {result_dir}: {e}")
            raise
    
    assert len(run_results) > 0, "No valid results loaded"
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"comparison_{timestamp}"
    else:
        output_dir = args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Save metadata about this comparison
    comparison_metadata = {
        'script_name': get_script_name(),
        'timestamp': datetime.now().isoformat(),
        'input_directories': {
            'subject_attention': [str(d) for d in (args.subject_attention or [])],
            'group_attention': [str(d) for d in (args.group_attention or [])],
            'group_subject_attention': [str(d) for d in (args.group_subject_attention or [])],
            'other': [str(d) for d in (args.other or [])]
        },
        'n_runs': len(run_results),
        'git_info': get_git_info()
    }
    
    with open(output_dir / 'comparison_metadata.json', 'w') as f:
        json.dump(comparison_metadata, f, indent=2)
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_metric_comparison(run_results, output_dir)
    plot_combined_distributions(run_results, output_dir)
    
    # Create summary table
    create_summary_table(run_results, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Comparison complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()