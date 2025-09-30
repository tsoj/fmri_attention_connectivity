#!/usr/bin/env python3
"""
Shared utilities for training scripts.

This module contains common functionality used by both group-level and subject-level
training scripts. Focuses on the happy path - crashes on unexpected issues rather
than trying to handle all edge cases.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

from config import ExperimentConfig


def validate_group_test_size(config: ExperimentConfig, train_subjects: List[str], test_subjects: List[str]) -> None:
    """
    Validate that the config test_size is compatible with the actual test ratio from CSV.
    For group-level training, we need to ensure test_size <= actual_test_ratio to avoid
    having too few test cases.
    """
    assert len(train_subjects) > 0, "No training subjects provided"
    assert len(test_subjects) > 0, "No test subjects provided"
    assert 0 < config.test_size < 1, f"test_size must be between 0 and 1, got {config.test_size}"

    total_subjects = len(train_subjects) + len(test_subjects)
    actual_test_ratio = len(test_subjects) / total_subjects

    if config.test_size > actual_test_ratio:
        raise ValueError(
            f"Config test_size ({config.test_size:.3f}) is larger than actual test ratio "
            f"from CSV ({actual_test_ratio:.3f}). This would result in insufficient test data. "
            f"Either reduce test_size in config or add more test subjects to CSV."
        )


def load_subject_groups(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Load subject IDs and groups from CSV file for group-level training.

    Args:
        csv_path: Path to CSV file with columns: subject_id, group

    Returns:
        Tuple of (train_subject_ids, test_subject_ids)
    """
    assert Path(csv_path).exists(), f"CSV file does not exist: {csv_path}"

    df = pd.read_csv(csv_path)

    assert "subject_id" in df.columns, f"CSV file must contain 'subject_id' column. Found columns: {list(df.columns)}"
    assert "group" in df.columns, f"CSV file must contain 'group' column. Found columns: {list(df.columns)}"
    assert len(df.columns) == 2, f"CSV file should contain exactly 2 columns. Found {len(df.columns)}"
    assert len(df) > 0, "CSV file is empty"

    # Split by group
    train_subjects = df[df["group"] == "train"]["subject_id"].astype(str).tolist()
    test_subjects = df[df["group"].isin(["test-retest", "test"])]["subject_id"].astype(str).tolist()

    assert len(train_subjects) > 0, "No subjects found with group 'train'"
    assert len(test_subjects) > 0, "No subjects found with group 'test-retest' or 'test'"

    return train_subjects, test_subjects


def load_subject_ids(csv_path: str) -> List[str]:
    """
    Load subject IDs from CSV file for subject-level training.
    Ignores group information if present.

    Args:
        csv_path: Path to CSV file with subject_id column

    Returns:
        List of all subject IDs
    """
    assert Path(csv_path).exists(), f"CSV file does not exist: {csv_path}"

    df = pd.read_csv(csv_path)

    assert "subject_id" in df.columns, f"CSV file must contain 'subject_id' column. Found columns: {list(df.columns)}"
    assert len(df) > 0, "CSV file is empty"

    subject_ids = df["subject_id"].astype(str).tolist()
    assert len(subject_ids) > 0, "No subject IDs found in CSV file"

    return subject_ids


def create_save_directory(
    root_save_dir: str,
    script_name: str,
    git_info: str,
    start_time: datetime,
    subject_id: Optional[str] = None,
    exists_ok=False,
) -> Path:
    """
    Create save directory with the structure:
    root_save_dir/script_name/git_info/date_str/[subject_id/]

    Args:
        root_save_dir: Root directory for saving results
        script_name: Name of the main script (without .py extension)
        git_info: Git information string
        start_time: Program start time
        subject_id: Optional subject ID for subject-level training

    Returns:
        Path object for the created directory
    """
    date_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    assert Path(root_save_dir).exists(), f"Root save directory does not exist: {root_save_dir}"
    assert script_name, "Script name cannot be empty"

    save_path = Path(root_save_dir) / script_name / git_info / date_str

    if subject_id is not None:
        assert subject_id, "Subject ID cannot be empty string"
        save_path = save_path / subject_id

    save_path.mkdir(parents=True, exist_ok=exists_ok)
    return save_path


def save_experiment_metadata(
    save_path: Path,
    start_time: datetime,
    git_info: str,
    config: ExperimentConfig,
    n_regions: int,
    additional_metadata: Optional[Dict] = None,
) -> Path:
    """Save experiment metadata to JSON file."""
    from pydantic import TypeAdapter

    # Use config's built-in serialization
    config_data = TypeAdapter(type(config)).dump_python(config, mode="json")

    metadata = {
        "start_time": start_time.isoformat(),
        "git_info": git_info,
        "config": config_data,
        "n_regions": n_regions,
    }

    if additional_metadata:
        metadata.update(additional_metadata)

    metadata_path = save_path / "experiment_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def split_session_timeseries(session_data, test_size: float):
    """
    Split a single session's timeseries into train and test portions.

    Args:
        session_data: numpy array of shape (n_regions, n_timepoints)
        test_size: fraction of timepoints to use for testing

    Returns:
        tuple of (train_data, test_data) with same shape as input
    """
    assert session_data.ndim == 2, f"Session data must be 2D, got shape {session_data.shape}"
    assert 0 < test_size < 1, f"test_size must be between 0 and 1, got {test_size}"

    n_regions, n_timepoints = session_data.shape
    assert n_regions > 0, f"Number of regions must be positive, got {n_regions}"
    assert n_timepoints > 0, f"Number of timepoints must be positive, got {n_timepoints}"

    train_size = 1.0 - test_size
    split_point = int(n_timepoints * train_size)

    assert 0 < split_point < n_timepoints, f"Invalid split point {split_point} for {n_timepoints} timepoints with test_size {test_size}"

    train_data = session_data[:, :split_point]
    test_data = session_data[:, split_point:]

    return train_data, test_data


def print_completion_summary(start_time: datetime, save_dir: Path):
    """Print training completion summary."""
    end_time = datetime.now()
    total_time = end_time - start_time
    print(f"Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {total_time}")
    print(f"Final model saved to: {save_dir}")
