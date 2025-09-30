"""
Refactored training module with clean separation of concerns.
Handles training of individual models and ensembles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm
import numpy as np
from dataclasses import fields
from contextlib import contextmanager
import json

from model import HCPAttentionModel
from utils import get_git_info
from config import TrainConfig


def get_device():
    """Get the best available device (CUDA if available, else CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_data_loader(
    sessions: Dict[str, np.ndarray], input_window: int, output_window: int, batch_size: int, shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader from session data.

    Args:
        sessions: List of numpy arrays with shape (num_regions, time_steps)
        input_window: Number of time steps for input
        output_window: Number of time steps to predict
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader with prepared samples
    """
    assert len(sessions) > 0, "No sessions provided to create data loader"
    assert input_window > 0, f"Input window must be positive, got {input_window}"
    assert output_window > 0, f"Output window must be positive, got {output_window}"
    assert batch_size > 0, f"Batch size must be positive, got {batch_size}"

    all_samples = []
    all_targets = []

    window_size = input_window + output_window

    for key, session in sessions.items():
        assert session.ndim == 2, f"Session {key} must be 2D, got shape {session.shape}"
        num_regions, time_steps = session.shape
        assert time_steps >= window_size, f"Session {key} has {time_steps} time steps, need at least {window_size}"

        # Create sliding windows
        for start_idx in range(time_steps - window_size + 1):
            end_idx = start_idx + window_size
            window = session[:, start_idx:end_idx]

            sample = window[:, :input_window]
            target = window[:, input_window:]

            all_samples.append(sample)
            all_targets.append(target)

    assert len(all_samples) > 0, "No samples created from sessions"

    # Convert to tensors
    X = torch.FloatTensor(np.array(all_samples))
    y = torch.FloatTensor(np.array(all_targets))

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_normalization_params(train_sessions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-region normalization parameters from training sessions.

    Args:
        train_sessions: Dictionary of training sessions

    Returns:
        Tuple of (mean, stddev) arrays with shape (num_regions,)
    """
    assert len(train_sessions) > 0, "No training sessions provided for normalization"

    all_train_data = []
    for key, session in train_sessions.items():
        assert session.ndim == 2, f"Session {key} must be 2D, got shape {session.shape}"
        all_train_data.append(session)  # shape: (num_regions, time_steps)

    # Concatenate all training sessions along time dimension
    train_data_combined = np.concatenate(all_train_data, axis=1)  # (num_regions, total_time_steps)
    assert train_data_combined.shape[1] > 0, "No time points available for normalization"

    # Compute per-region mean and standard deviation
    train_mean = np.mean(train_data_combined, axis=1)  # (num_regions,)
    train_std = np.std(train_data_combined, axis=1)  # (num_regions,)

    # Ensure no zero standard deviations
    assert np.all(train_std > 0), f"Zero standard deviation found in regions: {np.where(train_std == 0)[0]}"

    print(
        f"Computed normalization parameters - Mean range: [{train_mean.min():.4f}, {train_mean.max():.4f}], "
        f"Std range: [{train_std.min():.4f}, {train_std.max():.4f}]"
    )

    return train_mean, train_std


def evaluate_model_metrics(model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate model and compute R² and MSE metrics.

    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to run evaluation on

    Returns:
        Dictionary with 'r2' and 'mse' metrics
    """
    assert len(test_loader) > 0, "Test loader is empty"

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, return_attention=False)
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())

    assert len(all_predictions) > 0, "No predictions generated during evaluation"

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    assert predictions.shape == targets.shape, f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"

    # Calculate R²
    ss_res = ((targets - predictions) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    mse = ((targets - predictions) ** 2).mean().item()

    return {"r2": r2, "mse": mse}


def train_model(
    train_config: TrainConfig,
    train_sessions: Dict[str, np.ndarray],
    test_sessions: Dict[str, np.ndarray],
    n_regions: int,
    save_dir: Path,
    log_interval: int = 10,
) -> Dict[str, float]:
    """
    Train a single model and save it.

    Args:
        train_config: Configuration for the model architecture and training parameters
        train_sessions: Training data sessions
        test_sessions: Test data sessions
        n_regions: Number of parcellation regions
        save_dir: Directory to save the model folder to
        
    Returns:
        Dictionary with final metrics including 'r2' and 'mse'
    """
    assert len(train_sessions) > 0, "No training sessions provided"
    assert len(test_sessions) > 0, "No test sessions provided"
    assert n_regions > 0, f"Number of regions must be positive, got {n_regions}"
    assert train_config.num_epochs > 0, f"Number of epochs must be positive, got {train_config.num_epochs}"

    device = get_device()

    # Compute normalization parameters from training set only
    train_mean, train_std = compute_normalization_params(train_sessions)

    # Create model with normalization parameters
    model = HCPAttentionModel(
        num_regions=n_regions,
        input_window=train_config.input_window,
        output_window=train_config.output_window,
        hidden_dim=train_config.hidden_dim,
        num_input_layers=train_config.num_input_layers,
        num_prediction_layers=train_config.num_prediction_layers,
        bottleneck_dim=train_config.bottleneck_dim,
        attention_dropout_rate=train_config.attention_dropout_rate,
        mean=train_mean,
        stddev=train_std,
    ).to(device)

    # Create data loaders
    train_loader = create_data_loader(
        train_sessions, train_config.input_window, train_config.output_window, train_config.batch_size, shuffle=True
    )

    test_loader = create_data_loader(
        test_sessions, train_config.input_window, train_config.output_window, train_config.batch_size, shuffle=False
    )

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)

    # Initialize training statistics logging
    training_stats = {
        "train_losses": [],
        "train_mse_losses": [],
        "train_attention_losses": [],
        "test_losses": [],
        "epochs": [],
        "batch_losses": [],
        "batch_mse_losses": [],
        "batch_attention_losses": [],
        "batch_numbers": [],
    }

    for epoch in range(train_config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_attention_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_config.num_epochs}", leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Forward pass
            output, attention_weights = model(data, return_attention=True)

            # Compute losses
            mse_loss = F.mse_loss(output, target)
            total_loss = mse_loss
            attention_reg_loss = 0.0

            # Add attention regularization if needed
            if train_config.attention_reg_weight > 0:
                batch_size, num_regions, _ = attention_weights.shape
                identity_target = torch.eye(num_regions, device=device).unsqueeze(0).expand(batch_size, -1, -1)
                attention_reg_loss = F.mse_loss(attention_weights, identity_target)
                total_loss = mse_loss + train_config.attention_reg_weight * attention_reg_loss
                train_attention_loss += attention_reg_loss.item()

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += total_loss.item()
            train_mse_loss += mse_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "mse": f"{mse_loss.item():.4f}"})

            # Log batch-level statistics every log_interval batches
            if (batch_idx + 1) % log_interval == 0:
                training_stats["batch_losses"].append(total_loss.item())
                training_stats["batch_mse_losses"].append(mse_loss.item())
                training_stats["batch_attention_losses"].append(
                    attention_reg_loss.item() if train_config.attention_reg_weight > 0 else 0.0
                )
                training_stats["batch_numbers"].append(epoch * len(train_loader) + batch_idx + 1)

                tqdm.write(
                    f"    Batch {batch_idx + 1}/{len(train_loader)}: "
                    f"Loss={total_loss.item():.4f}, MSE={mse_loss.item():.4f}"
                    f"{f', Attention={attention_reg_loss.item():.6f}' if train_config.attention_reg_weight > 0 else ''}"
                )

        # Compute average training losses
        avg_train_loss = train_loss / num_batches
        avg_train_mse = train_mse_loss / num_batches
        avg_train_attention = train_attention_loss / num_batches if train_config.attention_reg_weight > 0 else 0

        # Evaluation phase
        current_metrics = evaluate_model_metrics(model, test_loader, device)
        avg_test_loss = current_metrics["mse"]  # Use MSE as test loss
        current_r2 = current_metrics["r2"]

        # Log training statistics
        training_stats["epochs"].append(epoch + 1)
        training_stats["train_losses"].append(avg_train_loss)
        training_stats["train_mse_losses"].append(avg_train_mse)
        training_stats["train_attention_losses"].append(avg_train_attention)
        training_stats["test_losses"].append(avg_test_loss)

        print(
            f"  Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Test Loss={avg_test_loss:.4f}, R²={current_r2:.6f}"
        )

    # Calculate final metrics
    final_metrics = evaluate_model_metrics(model, test_loader, device)

    # Save model state dict
    model_path = save_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    # Save training info (without model state dict for JSON compatibility)
    info_path = save_dir / "info.json"
    model_info = {
        "train_config": train_config.__dict__,
        "metrics": final_metrics,
        "normalization_params": {"mean": train_mean.tolist(), "stddev": train_std.tolist()},
        "training_stats": training_stats,
    }

    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"  Final R²: {final_metrics['r2']:.6f}")

    print(f"Saved to {model_path} and {info_path}")
    
    return final_metrics
