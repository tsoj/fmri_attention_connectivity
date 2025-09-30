#!/usr/bin/env python3
"""
Combined parameter-based training curves visualization in a single 2x3 grid.

This script creates a comprehensive view with:
- 5 training curve plots, each color-coded by a different hyperparameter
- 1 ablation study showing test MSE comparisons for parameter groups

The ablation study identifies groups of configs that differ only in one parameter
and shows the test MSE results for the most informative comparisons.
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

from utils import check_assert_enabled


@dataclass
class ParameterConfig:
    """Configuration for a parameter to analyze."""
    name: str
    display_name: str
    extractor_func: callable
    format_func: callable = lambda x: f"{x:.3f}"


def load_config_data(results_dir: Path) -> Dict[str, Dict]:
    """
    Load training data from all config directories.

    Args:
        results_dir: Directory containing config_01, config_02, etc. subdirectories

    Returns:
        Dictionary mapping config names to their training data
    """
    assert results_dir.exists(), f"Results directory does not exist: {results_dir}"
    assert results_dir.is_dir(), f"Path is not a directory: {results_dir}"

    config_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("config_")])
    assert len(config_dirs) > 0, f"No config directories found in {results_dir}"

    configs_data = {}

    for config_dir in config_dirs:
        info_path = config_dir / "info.json"
        assert info_path.exists(), f"info.json not found in {config_dir}"

        with open(info_path, 'r') as f:
            data = json.load(f)

        assert "training_stats" in data, f"training_stats not found in {info_path}"
        assert "train_config" in data, f"train_config not found in {info_path}"

        training_stats = data["training_stats"]
        train_config = data["train_config"]

        # Validate required fields
        required_fields = ["batch_mse_losses", "test_losses", "epochs"]
        for field in required_fields:
            assert field in training_stats, f"Required field '{field}' not found in training_stats from {info_path}"

        configs_data[config_dir.name] = {
            "training_stats": training_stats,
            "train_config": train_config,
            "metrics": data.get("metrics", {}),
            "config_dir": config_dir
        }

    print(f"Loaded {len(configs_data)} configurations")
    return configs_data


def get_parameter_value(train_config: Dict, param_config: ParameterConfig) -> float:
    """Extract a parameter value from a training configuration."""
    try:
        return param_config.extractor_func(train_config)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Failed to extract {param_config.name} from config: {e}") from e


def smooth_curve(y: List[float], window_size: int = 10) -> np.ndarray:
    """Apply moving average smoothing to a curve."""
    if len(y) < window_size:
        return np.array(y)
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')


def create_config_signature(train_config: Dict, exclude_param: str) -> Tuple:
    """
    Create a signature tuple for a config excluding one parameter.
    This is used to group configs that differ only in the excluded parameter.
    """
    signature_params = [
        "attention_dropout_rate", "bottleneck_dim", "hidden_dim",
        "num_input_layers", "num_prediction_layers", "input_window"
    ]

    signature = []
    for param in signature_params:
        if param == exclude_param:
            continue
        elif param == "total_layers" and exclude_param == "total_layers":
            continue
        elif exclude_param == "total_layers" and param in ["num_input_layers", "num_prediction_layers"]:
            continue
        else:
            if param in train_config:
                signature.append(train_config[param])
            elif param == "total_layers":
                signature.append(train_config["num_input_layers"] + train_config["num_prediction_layers"])

    return tuple(signature)


def find_ablation_groups(configs_data: Dict[str, Dict], parameters: List[ParameterConfig]) -> Dict[str, List[Tuple[List[str], str]]]:
    """
    Find groups of configs that differ only in one parameter.

    Returns:
        Dictionary mapping parameter names to lists of (config_group, group_description) tuples
        where group_description indicates what other parameters are held constant
    """
    ablation_groups = {}

    for param_config in parameters:
        # Group configs by their signature (all params except the current one)
        groups = defaultdict(list)

        for config_name, config_data in configs_data.items():
            try:
                signature = create_config_signature(config_data["train_config"], param_config.name)
                groups[signature].append(config_name)
            except KeyError:
                continue

        # Find all groups with multiple unique values for the target parameter
        valid_groups = []

        for signature, group_configs in groups.items():
            if len(group_configs) < 2:
                continue

            # Check how many unique values this parameter has in this group
            param_values = []
            for config_name in group_configs:
                try:
                    param_value = get_parameter_value(configs_data[config_name]["train_config"], param_config)
                    param_values.append(param_value)
                except ValueError:
                    continue

            unique_values = len(set(param_values))
            if unique_values > 1:
                # Create a description of what's held constant in this group
                sample_config = configs_data[group_configs[0]]["train_config"]
                desc_parts = []

                # Add key distinguishing features
                if param_config.name != "hidden_dim" and "hidden_dim" in sample_config:
                    desc_parts.append(f"h={sample_config['hidden_dim']}")
                if param_config.name not in ["num_input_layers", "num_prediction_layers", "total_layers"]:
                    if "num_input_layers" in sample_config and "num_prediction_layers" in sample_config:
                        desc_parts.append(f"layers={sample_config['num_input_layers']}+{sample_config['num_prediction_layers']}")
                if param_config.name != "bottleneck_dim" and "bottleneck_dim" in sample_config:
                    desc_parts.append(f"bn={sample_config['bottleneck_dim']}")
                if param_config.name != "input_window" and "input_window" in sample_config:
                    desc_parts.append(f"win={sample_config['input_window']}")
                if param_config.name != "attention_dropout_rate" and "attention_dropout_rate" in sample_config:
                    desc_parts.append(f"drop={sample_config['attention_dropout_rate']:g}")

                group_desc = ", ".join(desc_parts[:2])  # Limit to 2 most important features
                if not group_desc:
                    group_desc = "base"

                valid_groups.append((group_configs, group_desc, unique_values))

        # Sort by number of unique values (descending) and take top groups
        valid_groups.sort(key=lambda x: x[2], reverse=True)

        if valid_groups:
            # Take up to 2 best groups to avoid overcrowding
            selected_groups = [(group[0], group[1]) for group in valid_groups[:2]]
            ablation_groups[param_config.name] = selected_groups

            # Ensure we have at most 2 groups per parameter
            assert len(selected_groups) <= 2, f"More than 2 groups found for {param_config.name}"

            # If there are 2 groups, find what differs between them and update descriptions
            if len(selected_groups) == 2:
                # Compare the two groups to find what parameter differs
                group1_configs = selected_groups[0][0]
                group2_configs = selected_groups[1][0]

                sample1 = configs_data[group1_configs[0]]["train_config"]
                sample2 = configs_data[group2_configs[0]]["train_config"]

                # Find the differing parameter
                diff_param = None
                diff_values = []

                # For total_layers, ignore the component layer parameters since they make up the total
                exclude_keys = set()
                if param_config.name == "total_layers":
                    exclude_keys.update(["num_input_layers", "num_prediction_layers"])

                for key in sample1:
                    if key != param_config.name and key in sample2 and key not in exclude_keys:
                        val1 = sample1[key]
                        val2 = sample2[key]
                        if val1 != val2:
                            diff_param = key
                            diff_values = [val1, val2]
                            break

                # Update group descriptions with the differing parameter
                if diff_param:
                    # Create shortened parameter names
                    param_short = {
                        "hidden_dim": "h",
                        "bottleneck_dim": "bn",
                        "input_window": "win",
                        "attention_dropout_rate": "drop",
                        "num_input_layers": "in_layers",
                        "num_prediction_layers": "pred_layers"
                    }

                    short_name = param_short.get(diff_param, diff_param)

                    # Format values appropriately
                    if isinstance(diff_values[0], float):
                        formatted_vals = [f"{v:g}" for v in diff_values]
                    else:
                        formatted_vals = [str(v) for v in diff_values]

                    # Update the group descriptions
                    new_groups = []
                    for i, (group_configs, _) in enumerate(selected_groups):
                        new_desc = f"{short_name}={formatted_vals[i]}"
                        new_groups.append((group_configs, new_desc))
                    selected_groups = new_groups
                    ablation_groups[param_config.name] = selected_groups

            for i, (group_configs, group_desc, unique_vals) in enumerate(valid_groups[:2]):
                print(f"Ablation group {i+1} for {param_config.name}: {group_configs} ({unique_vals} unique values, {group_desc})")

    return ablation_groups


def plot_parameter_training_curves(
    configs_data: Dict[str, Dict],
    param_config: ParameterConfig,
    ax: plt.Axes,
    smooth_training: bool = True
) -> None:
    """Plot training curves for one parameter on a given axis."""
    # Extract parameter values
    param_values = {}
    for config_name, config_data in configs_data.items():
        try:
            param_value = get_parameter_value(config_data["train_config"], param_config)
            param_values[config_name] = param_value
        except ValueError:
            continue

    if len(param_values) == 0:
        ax.text(0.5, 0.5, f"No data for\n{param_config.display_name}",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(param_config.display_name)
        return

    # Get min/max for color mapping
    min_val = min(param_values.values())
    max_val = max(param_values.values())

    # Skip if no variation
    if min_val == max_val:
        ax.text(0.5, 0.5, f"No variation in\n{param_config.display_name}",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(param_config.display_name)
        return

    # Create colormap (red to blue)
    cmap = mcolors.LinearSegmentedColormap.from_list("red_blue", ["red", "blue"])

    # Plot each config
    for config_name, config_data in configs_data.items():
        if config_name not in param_values:
            continue

        param_value = param_values[config_name]
        training_stats = config_data["training_stats"]

        # Color mapping
        norm_value = (param_value - min_val) / (max_val - min_val)
        color = cmap(norm_value)

        # Plot training losses
        batch_losses = training_stats["batch_mse_losses"]
        batch_numbers = training_stats.get("batch_numbers", list(range(1, len(batch_losses) + 1)))

        if smooth_training and len(batch_losses) > 20:
            window_size = max(5, len(batch_losses) // 50)
            smooth_losses = smooth_curve(batch_losses, window_size)
            start_idx = (len(batch_numbers) - len(smooth_losses)) // 2
            smooth_batches = batch_numbers[start_idx:start_idx + len(smooth_losses)]
            ax.plot(smooth_batches, smooth_losses, color=color, linewidth=1.2, alpha=0.8)
        else:
            ax.plot(batch_numbers, batch_losses, color=color, linewidth=1.0, alpha=0.7)

    # Configure plot
    ax.set_title(param_config.display_name, fontsize=12)
    ax.set_xlabel("Batch Number", fontsize=10)
    ax.set_ylabel("Training MSE Loss", fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    # Format y-axis to avoid scientific notation
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.4f}'))

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(param_config.display_name, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def plot_hyperparameter_study(
    configs_data: Dict[str, Dict],
    parameters: List[ParameterConfig],
    ax: plt.Axes
) -> None:
    """Create hyperparameter study bar plot."""
    ablation_groups = find_ablation_groups(configs_data, parameters)

    if not ablation_groups:
        ax.text(0.5, 0.5, "No ablation groups found", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Ablation Study")
        return

    # Collect data for plotting
    plot_data = []
    labels = []
    colors = []
    legend_info = []

    # Create color palette using tab20c (has nice light/dark pairs)
    base_colors = plt.cm.tab20c(np.linspace(0, 1, len(ablation_groups) * 4))  # More colors available
    param_idx = 0

    current_x = 0
    x_positions = []

    for param_name, param_groups in ablation_groups.items():
        param_config = next(p for p in parameters if p.name == param_name)
        # Use tab20c color pairs (every 4th color is a group, with light/dark variants)
        color_base_idx = (param_idx * 4) % len(base_colors)
        light_color = base_colors[color_base_idx]
        dark_color = base_colors[color_base_idx + 2]  # Dark variant

        param_start_x = current_x

        for group_idx, (group_configs, group_desc) in enumerate(param_groups):
            # Get parameter values and test MSE for this group
            group_data = []
            for config_name in group_configs:
                config_data = configs_data[config_name]
                try:
                    param_value = get_parameter_value(config_data["train_config"], param_config)
                    test_mse = config_data["training_stats"]["test_losses"][-1]  # Final test loss
                    group_data.append((param_value, test_mse, config_name))
                except (ValueError, KeyError, IndexError):
                    continue

            if not group_data:
                continue

            # Sort by parameter value
            group_data.sort(key=lambda x: x[0])

            # Use light/dark variants based on group index
            if len(param_groups) == 1:
                # Single group - use normal color
                group_color = light_color
            else:
                # Multiple groups - use light/dark variants from tab20c
                if group_idx == 0:
                    group_color = light_color
                else:
                    group_color = dark_color

            # Add to plot data
            for param_value, test_mse, config_name in group_data:
                plot_data.append(test_mse)
                # Format float values to remove trailing zeros
                if isinstance(param_value, float):
                    formatted_value = f"{param_value:g}"  # Removes trailing zeros
                else:
                    formatted_value = param_config.format_func(param_value)
                labels.append(formatted_value)
                colors.append(group_color)
                x_positions.append(current_x)
                current_x += 1

            # Create legend entry with group description
            group_label = f"{param_config.display_name}"
            if len(param_groups) > 1:
                group_label += f" ({group_desc})"
            legend_info.append((group_color, group_label))

        # Add spacing between major parameter groups
        current_x += 2
        param_idx += 1

    if not plot_data:
        ax.text(0.5, 0.5, "No valid hyperparameter data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Hyperparameter Study")
        return

    # Create bar plot with proper spacing
    bars = ax.bar(x_positions, plot_data, color=colors, edgecolor='black', linewidth=0.5)

    # Configure plot
    ax.set_title("Hyperparameter Study: Final Test MSE", fontsize=12)
    ax.set_ylabel("Final Test MSE", fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=8)

    # Format y-axis to avoid scientific notation
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.4f}'))

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=90, ha='right')

    # Create legend
    legend_elements = []
    for color, label in legend_info:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color,
                                           label=label, edgecolor='black'))

    # Split legend into columns based on number of entries
    ncol = 3 if len(legend_elements) > 6 else (2 if len(legend_elements) > 3 else 1)

    # Adjust subplot position to make room for the legend below
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.85])

    # Place legend below the plot in the created space
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.06), fontsize=7, ncol=ncol,
              frameon=True, fancybox=True)


def create_combined_plot(
    configs_data: Dict[str, Dict],
    output_path: Path,
    smooth_training: bool = True
) -> None:
    """Create the combined 2x3 grid plot."""
    # Define parameters
    parameters = [
        ParameterConfig("attention_dropout_rate", "Attention Dropout",
                       lambda cfg: cfg["attention_dropout_rate"], lambda x: f"{x:g}"),
        ParameterConfig("bottleneck_dim", "Bottleneck Dim",
                       lambda cfg: cfg["bottleneck_dim"], lambda x: f"{int(x)}"),
        ParameterConfig("hidden_dim", "Hidden Dim",
                       lambda cfg: cfg["hidden_dim"], lambda x: f"{int(x)}"),
        ParameterConfig("total_layers", "Total Layers",
                       lambda cfg: cfg["num_input_layers"] + cfg["num_prediction_layers"],
                       lambda x: f"{int(x)}"),
        ParameterConfig("input_window", "Input Window",
                       lambda cfg: cfg["input_window"], lambda x: f"{int(x)}")
    ]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Plot parameter curves
    for i, param_config in enumerate(parameters):
        print(f"Creating plot for {param_config.display_name}...")
        plot_parameter_training_curves(configs_data, param_config, axes[i], smooth_training)

    # Plot hyperparameter study
    print("Creating hyperparameter study...")
    plot_hyperparameter_study(configs_data, parameters, axes[5])

    # Adjust layout - use constrained layout for better spacing with legends
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to: {output_path}")


def main():
    check_assert_enabled()

    parser = argparse.ArgumentParser(
        description="Create combined parameter-based training curves visualization"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing config_01, config_02, etc. subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_curves_combined.png",
        help="Output filename (default: training_curves_combined.png)"
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable smoothing of training curves"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    print(f"Loading training data from: {results_dir}")
    configs_data = load_config_data(results_dir)

    print(f"Creating combined visualization...")
    create_combined_plot(
        configs_data,
        output_path,
        smooth_training=not args.no_smooth
    )

    print("Analysis complete!")


if __name__ == "__main__":
    main()
