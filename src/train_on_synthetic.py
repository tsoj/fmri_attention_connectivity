#!/usr/bin/env python3
"""
Train HCPAttentionModel on vector-based synthetic data with empirical Jacobian computation.

This script:
1. Generates synthetic vector data using FC networks with known connectivity
2. Computes empirical Jacobian (ground truth influence) from the FC networks
3. Trains an HCPAttentionModel on the synthetic data
4. Extracts attention weights from the trained model
5. Compares ground truth connectivity, empirical Jacobian, and learned attention patterns

Usage:
    python train_on_synthetic.py
"""

import sys
import os
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random
from tqdm import tqdm

# Import from the new framework
from model import HCPAttentionModel
from utils import set_global_seed, check_assert_enabled

warnings.filterwarnings("ignore")

# Check asserts are enabled
check_assert_enabled()


# =============================================================================================
# VECTOR-BASED SYNTHETIC DATA GENERATION
# =============================================================================================
def generate_vector_dataset(
    num_samples=10000,
    num_regions=10,
    input_window=20,
    output_window=5,
    connectivity_density=0.3,
    step_size=1.0,
    seed=None,
):
    """
    Generate dataset with vector inputs/outputs and FC network connections.

    Args:
        num_samples: Number of training examples
        num_regions: Number of brain regions
        input_window: Size of input time window
        output_window: Size of output time window
        connectivity_density: Probability of connection between regions
        step_size: Standard deviation of random walk steps
        seed: Random seed for reproducibility

    Returns:
        inputs: (num_samples, num_regions, input_window)
        targets: (num_samples, num_regions, output_window)
        connectivity_matrix: (num_regions, num_regions)
        fc_networks: Dict of FC networks for each connection
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Create connectivity matrix with random weights
    connectivity_matrix = torch.zeros(num_regions, num_regions)
    total_connections = num_regions * num_regions
    num_connections = int(connectivity_density * total_connections)
    flat_indices = torch.randperm(total_connections)[:num_connections]
    # Assign random weights between 0.5 and 2.0 for connections
    random_weights = torch.rand(num_connections) * 1.5 + 0.5  # Random values in [0.5, 2.0)
    connectivity_matrix.view(-1)[flat_indices] = random_weights

    # Create FC networks for each connection (source -> target)
    fc_networks = {}
    connected_pairs = []
    for source in range(num_regions):
        for target in range(num_regions):
            if connectivity_matrix[target, source] > 0:  # target is influenced by source
                # FC network from source to target
                fc_net = nn.Sequential(
                    nn.Linear(input_window * 2, input_window), nn.ReLU(), nn.Linear(input_window, output_window)
                )
                fc_networks[(source, target)] = fc_net
                connected_pairs.append((source, target))

    print(f"Created {len(fc_networks)} FC networks for connections")

    # Generate all input vectors at once (vectorized random walks)
    # Shape: (num_samples, num_regions, input_window)
    input_vectors = torch.randn(num_samples, num_regions, input_window) * step_size

    # Vectorized normalization for input vectors
    # Normalize each vector individually to mean=0, std=1
    means = input_vectors.mean(dim=2, keepdim=True)
    stds = input_vectors.std(dim=2, keepdim=True)
    input_vectors = (input_vectors - means) / (stds + 1e-8)

    # Shape: (num_samples, num_regions, output_window)
    output_vectors = torch.zeros(num_samples, num_regions, output_window)

    # Add FC network contributions for connected regions (vectorized)
    with torch.no_grad():
        for source, target in connected_pairs:
            fc_net = fc_networks[(source, target)]
            connectivity_weight = connectivity_matrix[target, source]

            # Concatenate source and target vectors for all samples
            # Shape: (num_samples, input_window * 2)
            fc_inputs = torch.cat(
                [
                    input_vectors[:, source, :],  # (num_samples, input_window)
                    input_vectors[:, target, :],  # (num_samples, input_window)
                ],
                dim=1,
            )

            # Apply FC network to all samples at once
            # Shape: (num_samples, output_window)
            fc_outputs = fc_net(fc_inputs)

            # Add scaled contributions to target regions
            output_vectors[:, target, :] += fc_outputs * connectivity_weight

    # Vectorized normalization for output vectors after FC contributions
    # Normalize each output vector individually to mean=0, std=1
    output_means = output_vectors.mean(dim=2, keepdim=True)
    output_stds = output_vectors.std(dim=2, keepdim=True)
    # Only normalize if std > 0 (avoid division by zero)
    valid_stds = output_stds > 1e-8
    output_vectors = torch.where(valid_stds, (output_vectors - output_means) / output_stds, output_vectors)

    return input_vectors, output_vectors, connectivity_matrix, fc_networks


def compute_effective_influence(
    input_vectors: torch.Tensor,
    output_vectors: torch.Tensor,
    fc_networks: Dict,
    connectivity_matrix: torch.Tensor,
    num_samples_for_influence: int = 1000,
) -> np.ndarray:
    """
    Compute effective influence using fc_outputs * connectivity_weight approach.

    The influence on predicting the target is distributed to both source and target inputs,
    since both are used in the FC network's prediction.

    Args:
        input_vectors: Input data (num_samples, num_regions, input_window)
        output_vectors: Output data (num_samples, num_regions, output_window)
        fc_networks: Dictionary of FC networks
        connectivity_matrix: Ground truth connectivity
        num_samples_for_influence: Number of samples to use for influence computation

    Returns:
        influence_matrix: (num_regions, num_regions) average influence matrix
    """
    num_regions = connectivity_matrix.shape[0]
    input_window = input_vectors.shape[2]
    output_window = output_vectors.shape[2]

    # Sample a subset of data for influence computation
    sample_indices = torch.randperm(input_vectors.shape[0])[:num_samples_for_influence]
    sample_inputs = input_vectors[sample_indices]

    # Initialize influence accumulator
    influence_sum = torch.zeros(num_regions, num_regions)
    influence_count = 0

    print("Computing effective influence using fc_outputs * connectivity_weight...")

    with torch.no_grad():
        for sample_idx in range(num_samples_for_influence):
            if sample_idx % 200 == 0:
                print(f"  Processing sample {sample_idx}/{num_samples_for_influence}")

            sample_input = sample_inputs[sample_idx]  # Shape: (num_regions, input_window)

            # For each connected pair, compute fc_output * connectivity_weight
            for source in range(num_regions):
                for target in range(num_regions):
                    if connectivity_matrix[target, source] > 0:
                        fc_net = fc_networks[(source, target)]
                        connectivity_weight = connectivity_matrix[target, source]

                        # Create FC network input
                        fc_input = torch.cat(
                            [
                                sample_input[source, :],  # source vector
                                sample_input[target, :],  # target vector
                            ]
                        )

                        # Forward pass through FC network
                        fc_output = fc_net(fc_input)  # Shape: (output_window,)

                        # Compute effective influence as fc_output * connectivity_weight
                        # Average across output dimensions to get scalar influence
                        influence = (fc_output * connectivity_weight).abs().mean().item()

                        # Distribute influence to both source and target
                        # since both inputs contribute to the prediction
                        influence_sum[target, source] += influence * 0.5  # Source contribution
                        influence_sum[target, target] += influence * 0.5  # Target contribution

            influence_count += 1

    # Average the influence matrix
    influence_matrix = (influence_sum / max(1, influence_count)).numpy()

    print(f"Computed effective influence from {influence_count} samples")
    print(f"Influence matrix range: [{influence_matrix.min():.4f}, {influence_matrix.max():.4f}]")
    return influence_matrix


# =============================================================================================
# DATA PREPARATION FOR TRAINING
# =============================================================================================


def prepare_vector_data_for_training(
    input_vectors: torch.Tensor, output_vectors: torch.Tensor, input_window: int, output_window: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare vector data for custom training that preserves input->output relationships.

    Args:
        input_vectors: (num_samples, num_regions, input_window)
        output_vectors: (num_samples, num_regions, output_window)
        input_window: Size of input window
        output_window: Size of output window

    Returns:
        train_inputs: Training input vectors
        train_outputs: Training output vectors
        test_inputs: Test input vectors
        test_outputs: Test output vectors
    """
    # Split into train/test while preserving sample relationships
    split_idx = int(0.8 * len(input_vectors))

    train_inputs = input_vectors[:split_idx]
    train_outputs = output_vectors[:split_idx]
    test_inputs = input_vectors[split_idx:]
    test_outputs = output_vectors[split_idx:]

    print(f"Training data: {train_inputs.shape} -> {train_outputs.shape}")
    print(f"Test data: {test_inputs.shape} -> {test_outputs.shape}")
    print(f"Model will learn direct {input_window}->{output_window} mappings")

    # Debug: Check data ranges
    print(f"   Train input range: [{train_inputs.min():.3f}, {train_inputs.max():.3f}]")
    print(f"   Train output range: [{train_outputs.min():.3f}, {train_outputs.max():.3f}]")

    return train_inputs, train_outputs, test_inputs, test_outputs


def create_vector_data_loader(inputs: torch.Tensor, outputs: torch.Tensor, batch_size: int, shuffle: bool = True):
    """Create a DataLoader for vector input-output pairs."""
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(inputs, outputs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_normalization_params(train_inputs: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-region normalization parameters from training data.

    Args:
        train_inputs: Training input data (num_samples, num_regions, input_window)

    Returns:
        mean: Per-region mean (num_regions,)
        stddev: Per-region stddev (num_regions,)
    """
    # Compute mean and std across samples and time for each region
    # Shape: (num_samples, num_regions, input_window) -> (num_regions,)
    mean = train_inputs.mean(dim=(0, 2)).numpy()  # Average across samples and time
    stddev = train_inputs.std(dim=(0, 2)).numpy()  # Std across samples and time

    # Ensure no zero standard deviations
    stddev = np.maximum(stddev, 1e-8)

    return mean, stddev


def train_vector_model_custom(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_outputs: torch.Tensor,
    test_inputs: torch.Tensor,
    test_outputs: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """
    Custom training loop for vector synthetic data that preserves input-output relationships.

    Args:
        model: The model to train
        train_inputs: (num_train, num_regions, input_window)
        train_outputs: (num_train, num_regions, output_window)
        test_inputs: (num_test, num_regions, input_window)
        test_outputs: (num_test, num_regions, output_window)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use

    Returns:
        Dictionary with training history
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Create data loaders
    train_loader = create_vector_data_loader(train_inputs, train_outputs, batch_size, shuffle=True)
    test_loader = create_vector_data_loader(test_inputs, test_outputs, batch_size, shuffle=False)

    history = {"train_loss": [], "test_loss": [], "train_r2": [], "test_r2": []}

    print(f"Starting custom vector training on {device}...")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Debug: Check first batch shapes and values
    first_batch_inputs, first_batch_outputs = next(iter(train_loader))
    print(f"   First batch input shape: {first_batch_inputs.shape}")
    print(f"   First batch output shape: {first_batch_outputs.shape}")
    print(f"   Input value range: [{first_batch_inputs.min():.4f}, {first_batch_inputs.max():.4f}]")
    print(f"   Output value range: [{first_batch_outputs.min():.4f}, {first_batch_outputs.max():.4f}]")
    print(f"   Input mean/std: {first_batch_inputs.mean():.4f} / {first_batch_inputs.std():.4f}")
    print(f"   Output mean/std: {first_batch_outputs.mean():.4f} / {first_batch_outputs.std():.4f}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []

        for batch_inputs, batch_outputs in tqdm(train_loader):
            batch_inputs = batch_inputs.to(device)  # (batch_size, num_regions, input_window)
            batch_outputs = batch_outputs.to(device)  # (batch_size, num_regions, output_window)

            optimizer.zero_grad()

            # Forward pass - model expects (batch_size, num_regions, input_window)
            predictions = model(batch_inputs)  # (batch_size, num_regions, output_window)

            loss = criterion(predictions, batch_outputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_predictions.append(predictions.detach().cpu())
            train_targets.append(batch_outputs.cpu())

        # Validation phase
        model.eval()
        test_loss = 0.0
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for batch_inputs, batch_outputs in tqdm(test_loader):
                batch_inputs = batch_inputs.to(device)
                batch_outputs = batch_outputs.to(device)

                # Model expects (batch_size, num_regions, input_window)
                predictions = model(batch_inputs)

                loss = criterion(predictions, batch_outputs)
                test_loss += loss.item()
                test_predictions.append(predictions.cpu())
                test_targets.append(batch_outputs.cpu())

        # Calculate R² scores
        train_preds_flat = torch.cat(train_predictions, dim=0).flatten()
        train_targs_flat = torch.cat(train_targets, dim=0).flatten()
        train_r2 = 1 - torch.sum((train_targs_flat - train_preds_flat) ** 2) / torch.sum(
            (train_targs_flat - train_targs_flat.mean()) ** 2
        )

        test_preds_flat = torch.cat(test_predictions, dim=0).flatten()
        test_targs_flat = torch.cat(test_targets, dim=0).flatten()
        test_r2 = 1 - torch.sum((test_targs_flat - test_preds_flat) ** 2) / torch.sum(
            (test_targs_flat - test_targs_flat.mean()) ** 2
        )

        # Record metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(avg_test_loss)
        history["train_r2"].append(train_r2.item())
        history["test_r2"].append(test_r2.item())

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}, R²: {train_r2:.4f}")
        print(f"  Test Loss:  {avg_test_loss:.6f}, R²: {test_r2:.4f}")

        # Additional debugging for first few epochs
        if epoch < 5:
            print(f"  Sample prediction range: [{test_preds_flat.min():.4f}, {test_preds_flat.max():.4f}]")
            print(f"  Sample target range: [{test_targs_flat.min():.4f}, {test_targs_flat.max():.4f}]")

    return history


def create_vector_comparison_plots(
    connectivity_matrix: np.ndarray, effective_influence: np.ndarray, attention_weights: np.ndarray, save_dir: str
):
    """Create comparison plots for vector synthetic experiment."""

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    n_regions = connectivity_matrix.shape[0]

    # Use effective influence as ground truth for correlations
    inf_off = effective_influence.ravel()
    att_off = attention_weights.ravel()

    corr_inf_att = np.corrcoef(inf_off, att_off)[0, 1] if len(np.unique(inf_off)) > 1 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Effective influence (ground truth for comparison)
    sns.heatmap(effective_influence, ax=axes[0], cmap=sns.cubehelix_palette(as_cmap=True), cbar=True, square=True)
    axes[0].set_title(f"Ground Truth Connectivity Matrix", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Source Region (providing information)")
    axes[0].set_ylabel("Target Region (being predicted)")

    # Learned attention weights
    sns.heatmap(attention_weights, ax=axes[1], cmap=sns.cubehelix_palette(as_cmap=True), cbar=True, square=True)
    axes[1].set_title(f"Learned Attention Weights\nr(GT,Att) = {corr_inf_att:.3f}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Source Region (providing information)")
    axes[1].set_ylabel("Target Region (being predicted)")

    plt.tight_layout()
    plt.savefig(save_path / "synthetic_connectivity_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Print summary statistics
    print(f"\n   CORRELATION SUMMARY:")
    print(f"   Effective Influence ↔ Attention: r = {corr_inf_att:.4f}")

    print(f"\n   MATRIX STATISTICS:")
    print(f"   GT Connectivity:   mean={connectivity_matrix.mean():.4f}, std={connectivity_matrix.std():.4f}")
    print(f"   Effective Influence: mean={effective_influence.mean():.4f}, std={effective_influence.std():.4f}")
    print(f"   Learned Attention:  mean={attention_weights.mean():.4f}, std={attention_weights.std():.4f}")


# =============================================================================================
# MAIN EXPERIMENT FUNCTION
# =============================================================================================


def run_vector_synthetic_experiment():
    """Run the complete synthetic data experiment."""
    print("=" * 80)
    print("SYNTHETIC DATA EXPERIMENT WITH HCPAttentionModel")
    print("=" * 80)

    # Set global seed for reproducibility
    set_global_seed(42)

    # Set up model parameters
    print("\nSetting up model parameters...")
    num_regions = 20
    input_window = 15
    output_window = 1
    connectivity_density = 0.4

    print(f"   Model windows: {input_window} -> {output_window}")
    print(f"   Regions: {num_regions}")

    # Generate vector synthetic data
    print("\nGenerating synthetic data...")
    inputs, outputs, connectivity_matrix, fc_networks = generate_vector_dataset(
        num_samples=1_000_000,
        num_regions=num_regions,
        input_window=input_window,
        output_window=output_window,
        connectivity_density=connectivity_density,
        step_size=0.2,  # Further reduced noise level
        seed=42,
    )

    print(f"   Generated data shapes:")
    print(f"   Inputs: {inputs.shape}")
    print(f"   Outputs: {outputs.shape}")
    print(f"   Connectivity density: {(connectivity_matrix > 0).float().mean():.3f}")
    print(f"   Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
    print(f"   Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    print(
        f"   Connectivity strength range: [{connectivity_matrix[connectivity_matrix > 0].min():.3f}, {connectivity_matrix.max():.3f}]"
    )

    # Compute effective influence
    print("\nComputing effective influence...")
    effective_influence = compute_effective_influence(
        inputs, outputs, fc_networks, connectivity_matrix, num_samples_for_influence=1000
    )

    print(f"   Effective influence shape: {effective_influence.shape}")
    print(f"   Influence range: [{effective_influence.min():.4f}, {effective_influence.max():.4f}]")

    # Prepare data for training (preserving input-output relationships)
    print("\nPreparing data for custom training...")
    train_inputs, train_outputs, test_inputs, test_outputs = prepare_vector_data_for_training(
        inputs, outputs, input_window, output_window
    )

    # Set up model and training
    print("\nSetting up model and training...")

    # Create model directly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")

    # Compute normalization parameters from training data
    train_mean, train_std = compute_normalization_params(train_inputs)

    model = HCPAttentionModel(
        num_regions=num_regions,
        input_window=input_window,
        output_window=output_window,
        hidden_dim=64,
        num_input_layers=2,
        num_prediction_layers=2,
        bottleneck_dim=32,
        attention_dropout_rate=0.0,
        mean=train_mean,
        stddev=train_std,
    )
    model = model.to(device)

    # Update save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results_synthetic_data_training_{timestamp}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"   Model configuration:")
    print(f"   - Regions: {num_regions}")
    print(f"   - Input window: {input_window}")
    print(f"   - Output window: {output_window}")
    print(f"   - Hidden dim: 64")
    print(f"   - Epochs: 50")

    # Debug: Test model with a single batch
    print(f"\n   DEBUG: Testing model with sample input...")
    test_batch = train_inputs[:2]  # Take first 2 samples
    test_targets = train_outputs[:2]
    print(f"   Test batch shape: {test_batch.shape}")
    print(f"   Test targets shape: {test_targets.shape}")

    model.eval()
    with torch.no_grad():
        test_batch = test_batch.to(device)
        test_pred = model(test_batch)
        print(f"   Model output shape: {test_pred.shape}")
        print(f"   Model output range: [{test_pred.min():.6f}, {test_pred.max():.6f}]")

        # Test attention extraction
        test_pred_att, test_attention = model(test_batch, return_attention=True)
        print(f"   Attention weights shape: {test_attention.shape}")
        print(f"   Attention range: [{test_attention.min():.6f}, {test_attention.max():.6f}]")

    print(f"   Shape test PASSED! Starting training...")

    # Train the model with custom training loop
    print("\nTraining HCPAttentionModel with custom training...")
    history = train_vector_model_custom(
        model=model,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        num_epochs=1,
        batch_size=32,
        learning_rate=0.001,
        device=str(device),
    )

    print("   Training completed!")

    # Save the trained model
    model_save_path = Path(save_dir) / "model_0.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_history": history,
            "model_config": {
                "num_regions": num_regions,
                "input_window": input_window,
                "output_window": output_window,
                "hidden_dim": 64,
                "num_input_layers": 2,
                "num_prediction_layers": 2,
                "bottleneck_dim": 32,
                "attention_dropout_rate": 0.0,
                "mean": train_mean.tolist(),
                "stddev": train_std.tolist(),
            },
        },
        model_save_path,
    )

    # Extract attention weights
    print("\nExtracting attention weights...")
    model.eval()

    # Create test data loader for attention extraction
    test_loader = create_vector_data_loader(test_inputs, test_outputs, batch_size=64, shuffle=False)

    all_attention_weights = []

    with torch.no_grad():
        for batch_inputs, _ in test_loader:
            batch_inputs = batch_inputs.to(device)
            # Forward pass to get attention weights using model's built-in functionality
            # Model expects (batch_size, num_regions, input_window)
            predictions, attention_batch = model(batch_inputs, return_attention=True)

            # Store attention weights
            all_attention_weights.append(attention_batch.detach().cpu())

    # Average attention weights across all batches
    attention_weights = torch.cat(all_attention_weights, dim=0).mean(dim=0).numpy()
    print(f"   Extracted attention matrix: {attention_weights.shape}")
    print(f"   Attention range: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")

    # Create comparison visualizations
    print("\nCreating comparison visualizations...")
    create_vector_comparison_plots(
        connectivity_matrix=connectivity_matrix.numpy(),
        effective_influence=effective_influence,
        attention_weights=attention_weights,
        save_dir=save_dir,
    )

    print(f"\n   Results saved to: {save_dir}")
    print("\n" + "=" * 80)
    print("SYNTHETIC EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_vector_synthetic_experiment()
