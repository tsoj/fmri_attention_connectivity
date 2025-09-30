#!/usr/bin/env python3
"""
Model evaluation script for fMRI directed connectivity neural networks.

This script evaluates trained models on specified subjects and saves comprehensive evaluation results.
Supports both group-level and subject-level trained models based on --model-type flag.

Results are saved with the directory structure:
result_root_dir/evaluate_models/model_train_script_name/current_git_info/current_timestamp/subject_id/
"""

import argparse
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from pydantic import TypeAdapter

from config import TrainConfig
from dataset import load_subject_sessions
from evaluation import evaluate_model
from model import HCPAttentionModel
from training_utils import load_subject_ids
from utils import get_script_name, check_assert_enabled, get_git_info


check_assert_enabled()


def load_experiment_metadata(model_dir: str, model_type: str) -> dict[str, Any]:
    """Load experiment metadata from model directory."""
    metadata_dir = Path(model_dir)
    # For both group-level and subject-level, metadata is now at the same level as model directories
    metadata_file = metadata_dir / "experiment_metadata.json"
    assert metadata_file.exists(), f"Experiment metadata not found: {metadata_file}"

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return metadata


def discover_model_configs(model_dir: str, model_type: str, subject_ids: list[str]) -> dict[str, dict[str, str]]:
    """
    Discover all model configurations and return nested mapping.

    Args:
        model_dir: Base model directory
        model_type: Either "group-level" or "subject-level"
        subject_ids: List of subject IDs to evaluate

    Returns:
        Dictionary mapping config_name -> {subject_id -> model_directory_path}
        For group-level: multiple configs, each pointing to same model path for all subjects
        For subject-level: single "config_01", each subject pointing to its own model path
    """
    model_dir_path = Path(model_dir)
    assert model_dir_path.exists(), f"Model directory does not exist: {model_dir}"

    if model_type == "group-level":
        # Find all config_* directories
        group_configs = {}
        for item in model_dir_path.iterdir():
            if item.is_dir() and item.name.startswith("config_"):
                model_file = item / "model.pt"
                if model_file.exists():
                    # Same model path for all subjects
                    group_configs[item.name] = {subject_id: str(item) for subject_id in subject_ids}
                else:
                    print(f"Warning: Config directory {item.name} found but no model.pt file")

        assert group_configs, f"No valid config directories found in {model_dir}"
        print(f"Found {len(group_configs)} group-level config directories: {sorted(group_configs.keys())}")
        return group_configs

    elif model_type == "subject-level":
        # Single config with subject-specific model paths
        subject_model_paths = {}
        for subject_id in subject_ids:
            subject_model_dir = model_dir_path / subject_id
            subject_model_file = subject_model_dir / "model.pt"
            assert subject_model_file.exists(), f"Subject model not found: {subject_model_file}"
            subject_model_paths[subject_id] = str(subject_model_dir)

        print(f"Found {len(subject_model_paths)} subject-level model directories")
        return {"config_01": subject_model_paths}

    else:
        raise ValueError(f"Invalid model type: {model_type}. Must be 'group-level' or 'subject-level'")


def load_model_from_directory(model_dir: str) -> tuple[HCPAttentionModel, TrainConfig]:
    """Load model and training config from directory."""
    model_dir_path = Path(model_dir)
    model_file = model_dir_path / "model.pt"
    info_file = model_dir_path / "info.json"

    assert model_file.exists(), f"Model file not found: {model_file}"
    assert info_file.exists(), f"Info file not found: {info_file}"

    # Load model info
    with open(info_file, "r") as f:
        model_info = json.load(f)

    train_config_dict = model_info["train_config"]
    normalization_params = model_info["normalization_params"]

    # Create TrainConfig using pydantic
    train_config = TypeAdapter(TrainConfig).validate_python(train_config_dict)

    expected_mean = normalization_params["mean"]
    expected_stddev = normalization_params["stddev"]

    num_regions = len(expected_mean)
    dummy_mean = [0.0] * num_regions
    dummy_stddev = [1.0] * num_regions

    # Create model with dummy normalization params (will be overwritten by state dict)
    model = HCPAttentionModel(
        num_regions=num_regions,
        input_window=train_config.input_window,
        output_window=train_config.output_window,
        hidden_dim=train_config.hidden_dim,
        num_input_layers=train_config.num_input_layers,
        num_prediction_layers=train_config.num_prediction_layers,
        bottleneck_dim=train_config.bottleneck_dim,
        attention_dropout_rate=train_config.attention_dropout_rate,
        mean=dummy_mean,
        stddev=dummy_stddev,
    )

    # Load model weights
    _ = model.load_state_dict(torch.load(model_file, map_location="cpu"))

    # Validate that loaded buffers match the saved normalization parameters
    loaded_mean = model.mean.squeeze().detach().cpu().numpy().tolist()  # Remove broadcasting dims and convert to list
    loaded_stddev = model.stddev.squeeze().detach().cpu().numpy().tolist()

    # Check with small tolerance for floating point precision
    assert np.allclose(loaded_mean, expected_mean, rtol=1e-8), (
        f"Loaded mean doesn't match saved: {loaded_mean} vs {expected_mean}"
    )
    assert np.allclose(loaded_stddev, expected_stddev, rtol=1e-8), (
        f"Loaded stddev doesn't match saved: {loaded_stddev} vs {expected_stddev}"
    )

    print(f"Successfully validated normalization parameters from saved buffers")

    return model, train_config


def create_result_directory(
    result_root_dir: str,
    git_info: str,
    start_time: datetime,
    metadata: dict[str, Any],
    subject_id: str,
    config_name: Optional[str] = None,
) -> Path:
    """Create result directory using current git info and timestamp."""
    eval_script_name = get_script_name()

    # Determine training script name from training_type
    training_type = metadata.get("training_type", "unknown")
    if training_type == "group_level":
        train_script_name = "group_level_training"
    elif training_type == "subject_level" or training_type == "subject_level_summary":
        train_script_name = "subject_level_training"
    else:
        raise ValueError(f"Invalid training type: {training_type}. Must be 'group-level' or 'subject-level'")

    # Use current timestamp for directory name
    timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Include config name in path for group-level models
    if config_name is not None:
        result_dir = (
            Path(result_root_dir)
            / eval_script_name
            / train_script_name
            / git_info
            / timestamp_str
            / config_name
            / subject_id
        )
    else:
        result_dir = (
            Path(result_root_dir) / eval_script_name / train_script_name / git_info / timestamp_str / subject_id
        )
    result_dir.mkdir(parents=True, exist_ok=True)

    return result_dir


def create_evaluation_base_directory(
    result_root_dir: str,
    git_info: str,
    start_time: datetime,
    metadata: dict[str, Any],
    config_name: Optional[str] = None,
) -> Path:
    """Create base evaluation directory (without subject_id) for saving evaluation metadata."""
    eval_script_name = get_script_name()

    # Determine training script name from training_type
    training_type = metadata.get("training_type", "unknown")
    if training_type == "group_level":
        train_script_name = "group_level_training"
    elif training_type == "subject_level" or training_type == "subject_level_summary":
        train_script_name = "subject_level_training"
    else:
        raise ValueError(f"Invalid training type: {training_type}. Must be 'group-level' or 'subject-level'")

    # Use current timestamp for directory name
    timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Include config name in path for group-level models
    if config_name is not None:
        base_dir = Path(result_root_dir) / eval_script_name / train_script_name / git_info / timestamp_str / config_name
    else:
        base_dir = Path(result_root_dir) / eval_script_name / train_script_name / git_info / timestamp_str
    base_dir.mkdir(parents=True, exist_ok=True)

    return base_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate fMRI directed connectivity models")
    _ = parser.add_argument("subject_ids_csv", type=str, help="Path to CSV file with subject IDs")
    _ = parser.add_argument("data_path", type=str, help="Path to HCP data directory")
    _ = parser.add_argument("atlas_file", type=str, help="Path to brain atlas file")
    _ = parser.add_argument("model_dir", type=str, help="Path to directory containing models")
    _ = parser.add_argument("result_root_dir", type=str, help="Root directory for saving evaluation results")
    _ = parser.add_argument(
        "--model-type",
        type=str,
        choices=["group-level", "subject-level"],
        required=True,
        help="Type of model: group-level or subject-level",
    )
    _ = parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")

    args = parser.parse_args()

    start_time = datetime.now()
    print(f"Model evaluation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get git information
    git_info = get_git_info()
    print(f"Git info: {git_info}")

    # Validate input paths
    assert Path(args.subject_ids_csv).exists(), f"Subject IDs CSV not found: {args.subject_ids_csv}"
    assert Path(args.data_path).exists(), f"Data path not found: {args.data_path}"
    assert Path(args.atlas_file).exists(), f"Atlas file not found: {args.atlas_file}"
    assert Path(args.model_dir).exists(), f"Model directory not found: {args.model_dir}"

    # Create result root directory if needed
    Path(args.result_root_dir).mkdir(parents=True, exist_ok=True)

    # Load experiment metadata
    print(f"Loading experiment metadata from: {args.model_dir}")
    metadata = load_experiment_metadata(args.model_dir, args.model_type)

    # Load subject IDs
    print(f"Loading subject IDs from: {args.subject_ids_csv}")
    subject_ids = load_subject_ids(args.subject_ids_csv)
    print(f"Found {len(subject_ids)} subjects to evaluate")

    # Load dataset sessions
    print("Loading dataset sessions...")
    subject_sessions, n_regions = load_subject_sessions(
        root_dir=args.data_path, subject_ids=subject_ids, atlas_file=args.atlas_file
    )
    print(f"Loaded sessions for {len(subject_sessions)} subjects with {n_regions} regions")

    # Discover all model configurations
    model_paths = discover_model_configs(args.model_dir, args.model_type, list(subject_sessions.keys()))
    print(f"Discovered {len(model_paths)} model configurations")

    # Unified evaluation loop for both model types
    total_successful_evaluations = 0
    config_names = sorted(model_paths.keys())

    for config_name in config_names:
        config_subject_paths = model_paths[config_name]
        config_subjects = list(config_subject_paths.keys())

        print(f"\n{'=' * 80}")
        print(f"EVALUATING CONFIG: {config_name}")
        print(f"Subjects to evaluate: {len(config_subjects)}")
        print(f"{'=' * 80}")

        # Create base evaluation directory for this config
        base_eval_dir = create_evaluation_base_directory(
            args.result_root_dir,
            git_info,
            start_time,
            metadata,
            config_name if args.model_type == "group-level" else None,
        )

        # Create evaluation metadata for this config
        eval_metadata = {
            "start_time": start_time.isoformat(),
            "git_info": git_info,
            "n_regions": n_regions,
            "evaluation_type": f"{args.model_type}_evaluation",
            "model_directory": args.model_dir,
            "config_name": config_name,
            "config_subjects": config_subjects,
            "model_type": args.model_type,
            "data_path": args.data_path,
            "atlas_file": args.atlas_file,
            "subject_ids_csv": args.subject_ids_csv,
            "batch_size": args.batch_size,
            "requested_subjects": subject_ids,
            "available_subjects": list(subject_sessions.keys()),
            "n_requested_subjects": len(subject_ids),
            "n_available_subjects": len(subject_sessions),
            "training_metadata": metadata,
        }

        eval_metadata_path = base_eval_dir / "evaluation_metadata.json"
        with open(eval_metadata_path, "w") as f:
            json.dump(eval_metadata, f, indent=2)
        print(f"Saved evaluation metadata to: {eval_metadata_path}")

        # Evaluate each subject assigned to this config
        config_successful_evaluations = []
        current_model = None
        current_model_path = None

        for subject_id in config_subjects:
            if subject_id not in subject_sessions:
                print(f"Warning: Subject {subject_id} not found in loaded sessions, skipping")
                continue

            subject_model_path = config_subject_paths[subject_id]

            # Load model only when the path changes (optimization for group-level)
            if current_model is None or current_model_path != subject_model_path:
                print(f"Loading model from: {subject_model_path}")
                current_model, train_config = load_model_from_directory(subject_model_path)
                current_model_path = subject_model_path

            print(f"\n{'=' * 60}")
            print(f"Evaluating subject {subject_id} with config {config_name}")
            print(f"{'=' * 60}")

            # Get subject session data
            session_data = subject_sessions[subject_id]
            print(f"Session data shape: {session_data.shape}")

            # Evaluate model
            print("Running evaluation...")
            results = evaluate_model(
                model=current_model,
                session_data=session_data,
                input_window=train_config.input_window,
                output_window=train_config.output_window,
                batch_size=args.batch_size,
            )

            # Create result directory
            result_dir = create_result_directory(
                args.result_root_dir, git_info, start_time, metadata, subject_id, config_name
            )
            print(f"Saving results to: {result_dir}")

            # Save metrics as JSON
            metrics = {
                "subject_id": subject_id,
                "config_name": config_name,
                "mse (potentially evaluated on training data)": results["mse"],
                "r2 (potentially evaluated on training data)": results["r2"],
                "n_samples": len(results["inputs"]),
                "evaluation_timestamp": start_time.isoformat(),
                "model_directory": subject_model_path,
                "model_type": args.model_type,
                "train_config": train_config.__dict__,
            }

            metrics_file = result_dir / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            # Save mean attention weights as numpy array
            attention_file = result_dir / "mean_attention.npy"
            np.save(attention_file, results["mean_attention"])

            print(f"Successfully evaluated subject {subject_id} with config {config_name}")
            print(f"  MSE: {results['mse']:.6f}")
            print(f"  R²: {results['r2']:.6f}")
            print(f"  Samples: {len(results['inputs'])}")

            config_successful_evaluations.append(subject_id)

        print(
            f"\nConfig {config_name} summary: {len(config_successful_evaluations)}/{len(config_subjects)} subjects evaluated successfully"
        )
        total_successful_evaluations += len(config_successful_evaluations)

    # Print overall summary
    end_time = datetime.now()
    total_expected_evaluations = sum(len(config_subjects) for config_subjects in model_paths.values())
    print(f"\n{'=' * 80}")
    print("OVERALL EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total configs: {len(config_names)}")
    print(f"Model type: {args.model_type}")
    print(f"Total expected evaluations: {total_expected_evaluations}")
    print(f"Successfully completed: {total_successful_evaluations}")
    print(f"Total time: {end_time - start_time}")
    print(f"Results saved to: {args.result_root_dir}")


if __name__ == "__main__":
    main()
