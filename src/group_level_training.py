#!/usr/bin/env python3
"""
Group-level training launch script for fMRI directed connectivity neural network.

This script:
1. Loads configuration from a config file with validation
2. Gets git information and program start time
3. Creates appropriate save directory structure
4. Loads subject IDs and groups from CSV file
5. Validates test_size compatibility with CSV group split
6. Loads HCP sessions for train/test subjects
7. Trains a single model on all training subjects
"""

import argparse
from datetime import datetime

from config import ExperimentConfig
from train import train_model
from dataset import load_subject_sessions
from utils import get_git_info, set_global_seed, get_script_name, check_assert_enabled
from training_utils import (
    load_subject_groups,
    validate_group_test_size,
    create_save_directory,
    save_experiment_metadata,
    print_completion_summary,
)


check_assert_enabled()


def main():
    parser = argparse.ArgumentParser(description="Launch group-level training for fMRI directed connectivity model")
    parser.add_argument("config_file", type=str, help="Path to configuration JSON file")

    args = parser.parse_args()

    # Get program start time right at the beginning
    start_time = datetime.now()
    print(f"Group-level training launched at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get git information
    git_info = get_git_info()
    print(f"Git info: {git_info}")

    # Load and validate configuration
    print(f"Loading configuration from: {args.config_file}")
    config = ExperimentConfig.from_json_file(args.config_file)
    print(f"Loaded {len(config.train_configs)} training configurations")
    for i, train_config in enumerate(config.train_configs):
        print(f"  Config {i+1}: hidden_dim={train_config.hidden_dim}, input_layers={train_config.num_input_layers}, prediction_layers={train_config.num_prediction_layers}")

    # Set global random seed
    set_global_seed(config.random_seed)
    print(f"Set global random seed to: {config.random_seed}")

    # Load subject groups
    print(f"Loading subject groups from: {config.subject_ids_csv_path}")
    train_subject_ids, test_subject_ids = load_subject_groups(config.subject_ids_csv_path)

    print(
        f"Train subjects ({len(train_subject_ids)}): {train_subject_ids[:3]}{'...' if len(train_subject_ids) > 3 else ''}"
    )
    print(
        f"Test subjects ({len(test_subject_ids)}): {test_subject_ids[:3]}{'...' if len(test_subject_ids) > 3 else ''}"
    )

    # Validate test_size compatibility with CSV split
    validate_group_test_size(config, train_subject_ids, test_subject_ids)
    print(f"Validated config test_size ({config.test_size:.3f}) is compatible with CSV group split")

    # Create save directory
    script_name = get_script_name()
    save_dir = create_save_directory(config.root_save_dir, script_name, git_info, start_time)
    print(f"Model will be saved to: {save_dir}")

    # Load training sessions
    print("Loading training sessions...")
    train_sessions, n_regions = load_subject_sessions(
        root_dir=config.hcp_data_path,
        subject_ids=train_subject_ids,
        atlas_file=config.atlas_file_path,
        num_cortical_vertices=config.num_cortical_vertices,
    )

    # Load test sessions
    print("Loading test sessions...")
    test_sessions, _ = load_subject_sessions(
        root_dir=config.hcp_data_path,
        subject_ids=test_subject_ids,
        atlas_file=config.atlas_file_path,
        num_cortical_vertices=config.num_cortical_vertices,
    )

    print(f"Loaded {len(train_sessions)} training sessions and {len(test_sessions)} test sessions")
    print(f"Number of brain regions: {n_regions}")

    # Ensure we have data for all requested subjects
    missing_train_subjects = set(train_subject_ids) - set(train_sessions.keys())
    missing_test_subjects = set(test_subject_ids) - set(test_sessions.keys())

    if missing_train_subjects:
        raise ValueError(f"Missing training data for subjects: {missing_train_subjects}")
    if missing_test_subjects:
        raise ValueError(f"Missing test data for subjects: {missing_test_subjects}")

    # Save experiment metadata
    group_metadata = {
        "training_type": "group_level",
        "train_subjects": list(train_sessions.keys()),
        "test_subjects": list(test_sessions.keys()),
        "n_train_sessions": len(train_sessions),
        "n_test_sessions": len(test_sessions),
        "missing_train_subjects": list(missing_train_subjects),
        "missing_test_subjects": list(missing_test_subjects),
        "num_train_configs": len(config.train_configs),
    }

    metadata_path = save_experiment_metadata(save_dir, start_time, git_info, config, n_regions, group_metadata)
    print(f"Saved experiment metadata to: {metadata_path}")

    # Train models with each configuration
    print(f"Starting training with {len(config.train_configs)} configurations...")
    
    for config_idx, train_config in enumerate(config.train_configs):
        print(f"\n=== Training Configuration {config_idx + 1}/{len(config.train_configs)} ===")
        print(f"Config: hidden_dim={train_config.hidden_dim}, input_layers={train_config.num_input_layers}, prediction_layers={train_config.num_prediction_layers}")
        
        # Create subdirectory for this configuration
        config_save_dir = save_dir / f"config_{config_idx + 1:02d}"
        config_save_dir.mkdir(exist_ok=True)
        
        print(f"Saving to: {config_save_dir}")
        
        _ = train_model(
            train_config=train_config,
            train_sessions=train_sessions,
            test_sessions=test_sessions,
            n_regions=n_regions,
            save_dir=config_save_dir,
        )
        
        print(f"Completed training configuration {config_idx + 1}/{len(config.train_configs)}")

    print_completion_summary(start_time, save_dir)


if __name__ == "__main__":
    main()
