#!/usr/bin/env python3
"""
Subject-level training launch script for fMRI directed connectivity neural network.

This script:
1. Loads configuration from a config file with validation
2. Gets git information and program start time
3. Creates appropriate save directory structure for each subject
4. Loads subject IDs from CSV file (ignoring group information)
5. For each subject, loads their session and splits into train/test portions
6. Trains a separate model for each subject
7. Focuses on the happy path - crashes on unexpected issues
"""

import argparse
import json
from datetime import datetime

from config import ExperimentConfig
from train import train_model
from dataset import load_subject_sessions
from utils import get_git_info, set_global_seed, get_script_name, check_assert_enabled
from training_utils import (
    load_subject_ids,
    create_save_directory,
    save_experiment_metadata,
    split_session_timeseries,
    print_completion_summary,
)


check_assert_enabled()


def main():
    parser = argparse.ArgumentParser(description="Launch subject-level training for fMRI directed connectivity model")
    parser.add_argument("config_file", type=str, help="Path to configuration JSON file")

    args = parser.parse_args()

    # Get program start time right at the beginning
    start_time = datetime.now()
    print(f"Subject-level training launched at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get git information
    git_info = get_git_info()
    print(f"Git info: {git_info}")

    # Load and validate configuration
    print(f"Loading configuration from: {args.config_file}")
    config = ExperimentConfig.from_json_file(args.config_file)
    if len(config.train_configs) != 1:
        raise ValueError(f"Subject-level training requires exactly 1 training configuration, got {len(config.train_configs)}")

    train_config = config.train_configs[0]
    print(f"Loaded training configuration: hidden_dim={train_config.hidden_dim}, input_layers={train_config.num_input_layers}, prediction_layers={train_config.num_prediction_layers}")
    print(f"Test size: {config.test_size}")

    # Set global random seed
    set_global_seed(config.random_seed)
    print(f"Set global random seed to: {config.random_seed}")

    # Load subject IDs (ignoring group information)
    print(f"Loading subject IDs from: {config.subject_ids_csv_path}")
    subject_ids = load_subject_ids(config.subject_ids_csv_path)
    print(f"Found {len(subject_ids)} subjects: {subject_ids[:3]}{'...' if len(subject_ids) > 3 else ''}")

    # Load all subject sessions
    print("Loading subject sessions...")
    subject_sessions, n_regions = load_subject_sessions(
        root_dir=config.hcp_data_path,
        subject_ids=subject_ids,
        atlas_file=config.atlas_file_path,
        num_cortical_vertices=config.num_cortical_vertices,
    )

    print(f"Successfully loaded sessions for {len(subject_sessions)} subjects")
    print(f"Number of brain regions: {n_regions}")

    # Ensure we have data for all requested subjects
    missing_subjects = set(subject_ids) - set(subject_sessions.keys())
    if missing_subjects:
        raise ValueError(f"Missing session data for subjects: {missing_subjects}")

    # Get script name for directory structure
    script_name = get_script_name()

    # Train a model for each subject
    successful_subjects = []
    r2_scores = []  # Collect R² scores for each subject

    for i, subject_id in enumerate(subject_sessions.keys()):
        print(f"\n{'=' * 60}")
        print(f"Training model for subject {subject_id} ({i + 1}/{len(subject_sessions)})")
        print(f"{'=' * 60}")

        # Get subject's session data
        session_data = subject_sessions[subject_id]
        print(f"Session data shape: {session_data.shape}")

        # Split session into train/test portions
        train_data, test_data = split_session_timeseries(session_data, config.test_size)
        print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")

        # Create train/test session dictionaries (with single subject)
        train_sessions = {subject_id: train_data}
        test_sessions = {subject_id: test_data}

        # Create subject-specific save directory
        subject_save_dir = create_save_directory(config.root_save_dir, script_name, git_info, start_time, subject_id)
        print(f"Subject model will be saved to: {subject_save_dir}")

        # Save subject-specific experiment metadata
        subject_metadata = {
            "training_type": "subject_level",
            "subject_id": subject_id,
            "test_size": config.test_size,
            "session_shape": list(session_data.shape),
            "train_shape": list(train_data.shape),
            "test_shape": list(test_data.shape),

        }

        metadata_path = save_experiment_metadata(
            subject_save_dir, start_time, git_info, config, n_regions, subject_metadata
        )
        print(f"Saved subject metadata to: {metadata_path}")

        # Train the model - fail fast on any issues
        print(f"Starting model training for subject {subject_id}...")
        final_metrics = train_model(
            train_config=train_config,
            train_sessions=train_sessions,
            test_sessions=test_sessions,
            n_regions=n_regions,
            save_dir=subject_save_dir,
        )

        # Collect the R² score
        subject_r2 = final_metrics["r2"]
        r2_scores.append(subject_r2)
        print(f"Successfully completed training for subject {subject_id} (R² = {subject_r2:.6f})")
        successful_subjects.append(subject_id)

    # Print final summary
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total subjects completed: {len(successful_subjects)}")
    print(f"Successful subjects: {successful_subjects}")

    # Compute and print R² statistics
    mean_r2 = sum(r2_scores) / len(r2_scores)
    min_r2 = min(r2_scores)
    max_r2 = max(r2_scores)
    print(f"Mean R² score: {mean_r2:.6f}")
    print(f"R² range: [{min_r2:.6f}, {max_r2:.6f}]")
    print(f"Individual R² scores: {[f'{r2:.6f}' for r2 in r2_scores]}")

    # Save overall summary
    base_save_dir = create_save_directory(config.root_save_dir, script_name, git_info, start_time, exists_ok=True)

    summary_metadata = {
        "training_type": "subject_level_summary",
        "total_subjects_completed": len(successful_subjects),
        "successful_subjects": successful_subjects,
        "test_size": config.test_size,
        "all_subject_ids": subject_ids,
        "r2_statistics": {
            "mean_r2": mean_r2,
            "min_r2": min_r2,
            "max_r2": max_r2,
            "individual_r2_scores": r2_scores,
            "num_subjects_with_r2": len(r2_scores)
        }
    }

    summary_path = save_experiment_metadata(base_save_dir, start_time, git_info, config, n_regions, summary_metadata)
    print(f"Saved overall training summary to: {summary_path}")
    print_completion_summary(start_time, base_save_dir)


if __name__ == "__main__":
    main()
