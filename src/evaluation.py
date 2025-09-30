"""
Model evaluation module for fMRI directed connectivity neural networks.

Provides functionality to evaluate trained models on dataset sessions,
returning comprehensive metrics and predictions.
"""

import torch
import numpy as np
from typing import Any
from tqdm import tqdm

from model import HCPAttentionModel
from train import create_data_loader, get_device


def evaluate_model(
    model: HCPAttentionModel, session_data: np.ndarray, input_window: int, output_window: int, batch_size: int = 32
) -> dict[str, Any]:
    """
    Evaluate a model on session data and return comprehensive results.

    Args:
        model: Trained HCPAttentionModel
        session_data: Session data with shape (num_regions, time_steps)
        input_window: Number of input time steps
        output_window: Number of output time steps to predict
        batch_size: Batch size for evaluation

    Returns:
        Dictionary containing:
        - mse: Mean squared error
        - r2: R-squared coefficient
        - inputs: List of input arrays (moved to CPU)
        - targets: List of target arrays (moved to CPU)
        - predictions: List of prediction arrays (moved to CPU)
        - attention_weights: List of attention weight arrays (moved to CPU)
        - mean_attention: Mean attention weights across all samples
    """
    assert isinstance(model, HCPAttentionModel), f"Expected HCPAttentionModel, got {type(model)}"
    assert session_data.ndim == 2, f"Session data must be 2D, got shape {session_data.shape}"
    assert input_window > 0, f"Input window must be positive, got {input_window}"
    assert output_window > 0, f"Output window must be positive, got {output_window}"
    assert batch_size > 0, f"Batch size must be positive, got {batch_size}"

    device = get_device()
    _ = model.to(device)
    model.eval()

    # Wrap single session in dict to use shared data loader
    sessions = {"eval_session": session_data}
    data_loader = create_data_loader(sessions, input_window, output_window, batch_size, shuffle=False)

    # Storage for results (all moved to CPU immediately)
    all_inputs = []
    all_targets = []
    all_predictions = []
    all_attention_weights = []

    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Evaluating model", leave=False):
            data, target = data.to(device), target.to(device)

            # Forward pass with attention
            predictions, attention_weights = model(data, return_attention=True)

            # Move everything to CPU immediately to avoid GPU memory buildup
            inputs_cpu = data.cpu().numpy()
            targets_cpu = target.cpu().numpy()
            predictions_cpu = predictions.cpu().numpy()
            attention_weights_cpu = attention_weights.cpu().numpy()

            # Store individual samples
            for i in range(inputs_cpu.shape[0]):
                all_inputs.append(inputs_cpu[i])
                all_targets.append(targets_cpu[i])
                all_predictions.append(predictions_cpu[i])
                all_attention_weights.append(attention_weights_cpu[i])

    assert len(all_inputs) > 0, "No samples were processed during evaluation"
    assert len(all_inputs) == len(all_targets) == len(all_predictions) == len(all_attention_weights), (
        "Mismatch in number of collected samples"
    )

    # Compute metrics using numpy arrays
    predictions_array = np.array(all_predictions)
    targets_array = np.array(all_targets)

    assert predictions_array.shape == targets_array.shape, (
        f"Shape mismatch: predictions {predictions_array.shape} vs targets {targets_array.shape}"
    )

    # Calculate MSE
    mse = np.mean((targets_array - predictions_array) ** 2)

    # Calculate R²
    ss_res = np.sum((targets_array - predictions_array) ** 2)
    ss_tot = np.sum((targets_array - np.mean(targets_array)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Calculate mean attention weights across all samples
    attention_stack = np.stack(all_attention_weights, axis=0)  # (n_samples, n_regions, n_regions)
    mean_attention = np.mean(attention_stack, axis=0)  # (n_regions, n_regions)

    print(f"Evaluation completed: MSE={mse:.6f}, R²={r2:.6f}, {len(all_inputs)} samples processed")

    return {
        "mse": float(mse),
        "r2": float(r2),
        "inputs": all_inputs,
        "targets": all_targets,
        "predictions": all_predictions,
        "attention_weights": all_attention_weights,
        "mean_attention": mean_attention,
    }
