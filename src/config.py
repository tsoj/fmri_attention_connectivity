#!/usr/bin/env python3
"""
Comprehensive configuration management for fMRI directed connectivity training.

Uses pydantic dataclasses for robust configuration validation and type checking.
"""

import json
from pathlib import Path

from pydantic.dataclasses import dataclass
from pydantic import Field, field_validator, TypeAdapter


@dataclass
class TrainConfig:
    """Training configuration parameters."""

    output_window: int = Field(..., gt=0, description="Number of future time points to predict")
    input_window: int = Field(..., gt=0, description="Number of past time points to use as input")
    hidden_dim: int = Field(..., gt=0, description="Hidden dimension size")
    num_input_layers: int = Field(..., gt=0, description="Number of input processing layers")
    num_prediction_layers: int = Field(..., gt=0, description="Number of prediction layers")
    bottleneck_dim: int = Field(..., gt=0, description="Bottleneck dimension for attention")
    attention_dropout_rate: float = Field(..., ge=0, le=1, description="Dropout rate for attention layers")
    attention_reg_weight: float = Field(..., ge=0, description="Regularization weight for attention")
    num_epochs: int = Field(..., gt=0, description="Number of training epochs")
    batch_size: int = Field(..., gt=0, description="Batch size for training")
    learning_rate: float = Field(..., gt=0, description="Learning rate for optimizer")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Core paths
    atlas_file_path: str = Field(..., description="Path to brain atlas file")
    hcp_data_path: str = Field(..., description="Path to HCP data directory")
    subject_ids_csv_path: str = Field(..., description="Path to CSV file with subject IDs")
    root_save_dir: str = Field(..., description="Root directory for saving results")

    # Training configuration
    train_configs: list[TrainConfig] = Field(..., min_length=1, description="List of training configurations to try")

    # Experiment parameters
    random_seed: int = Field(..., ge=0, description="Random seed for reproducibility")
    test_size: float = Field(..., gt=0, lt=1, description="Fraction of data to use for testing")

    # Optional parameters
    num_cortical_vertices: int = Field(64984, gt=0, description="Number of cortical vertices")

    @field_validator("atlas_file_path", "hcp_data_path", "subject_ids_csv_path")
    @classmethod
    def paths_must_exist(cls, v: str) -> str:
        """Validate that required paths exist."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v

    @field_validator("subject_ids_csv_path")
    @classmethod
    def csv_must_have_correct_extension(cls, v: str) -> str:
        """Validate that subject IDs file is a CSV."""
        if not v.lower().endswith(".csv"):
            raise ValueError(f"Subject IDs file must be a CSV file: {v}")
        return v

    @field_validator("root_save_dir")
    @classmethod
    def root_save_dir_must_be_creatable(cls, v: str) -> str:
        """Validate that root save directory can be created."""
        path = Path(v)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create root save directory {v}: {e}")
        return v

    @classmethod
    def from_json_file(cls, config_path: str) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        text = Path(config_path).read_text()
        return TypeAdapter(cls).validate_json(text)  # runs validators, builds nested dataclasses

    def to_json_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        data = TypeAdapter(type(self)).dump_python(self, mode="json")  # recursive, JSON-friendly
        Path(config_path).write_text(json.dumps(data, indent=2))
