#!/usr/bin/env python3
"""
Script to predict behavioral data from connectivity matrices using k-fold cross-validation.
Uses k-fold CV where one fold is for testing, one for validation (early stopping),
and the rest for training. Saves model weights for analysis.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

# Import shared utilities
from utils import check_assert_enabled, get_git_info, get_script_name


def parse_age(age_str):
    """Parse age string like '22-25' or '36+' to get the first number."""
    import re
    match = re.match(r'^(\d+)', str(age_str))
    if match:
        return float(match.group(1))
    return None


def find_matrix_files(connectivity_dir: Path, matrix_pattern: str, subject_ids: List[str]) -> Dict[str, Path]:
    """Find matrix files for subjects in their respective directories."""
    matrix_files = {}
    missing_count = 0

    for subject_id in subject_ids:
        subject_dir = connectivity_dir / subject_id
        if not subject_dir.exists():
            missing_count += 1
            continue

        matrix_path = subject_dir / matrix_pattern
        if matrix_path.exists():
            matrix_files[subject_id] = matrix_path
        else:
            missing_count += 1

    print(f"Found matrices for {len(matrix_files)}/{len(subject_ids)} subjects ({missing_count} missing)")
    return matrix_files


def load_behavioral_targets(csv_path: Path, target_columns: List[str], shuffle_targets: bool = False) -> pd.DataFrame:
    """Load behavioral target data from CSV file."""
    df = pd.read_csv(csv_path)

    # Ensure first column is subject ID
    df[df.columns[0]] = df[df.columns[0]].astype(str)
    df = df.set_index(df.columns[0])

    # Shuffle behavioral data across subjects if requested
    if shuffle_targets:
        print("WARNING: Shuffle targets enabled - randomizing subject-behavioral data mapping!")
        # Create a copy of the behavioral data with shuffled subject IDs
        subject_ids = list(df.index)
        shuffled_subject_ids = subject_ids.copy()
        random.shuffle(shuffled_subject_ids)
        
        print("Debug: Subject ID -> Shuffled behavioral data mapping:")
        for orig, shuffled in zip(subject_ids, shuffled_subject_ids):
            print(f"  {orig} -> behavioral data from {shuffled}")
        
        # Create new dataframe with original subject IDs but shuffled behavioral data
        shuffled_df = pd.DataFrame(index=subject_ids, columns=df.columns)
        for i, orig_subject in enumerate(subject_ids):
            shuffled_subject = shuffled_subject_ids[i]
            shuffled_df.loc[orig_subject] = df.loc[shuffled_subject]
        df = shuffled_df

    processed_df = pd.DataFrame(index=df.index)

    for col in target_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in behavioral data")
            continue

        if col == "Gender":
            processed_df[col] = (df[col] == 'M').astype(float)
        elif col == "Age":
            processed_df[col] = df[col].apply(parse_age)
        else:
            processed_df[col] = pd.to_numeric(df[col], errors='coerce')

    return processed_df


def normalize_matrices(train_matrices: Dict[str, np.ndarray],
                      test_matrices: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
    """Z-normalize matrices pixel-wise using train set statistics."""
    train_stack = np.stack(list(train_matrices.values()), axis=0)
    pixel_mean = np.mean(train_stack, axis=0)
    pixel_std = np.std(train_stack, axis=0)
    pixel_std = np.where(pixel_std == 0, 1.0, pixel_std)

    normalized_train = {k: (v - pixel_mean) / pixel_std for k, v in train_matrices.items()}
    normalized_test = {k: (v - pixel_mean) / pixel_std for k, v in test_matrices.items()}

    return normalized_train, normalized_test


def normalize_targets(train_targets: List[float], test_targets: List[float]) -> Tuple[List, List, float, float]:
    """Z-normalize targets using train set statistics."""
    train_array = np.array(train_targets)
    mean = np.mean(train_array)
    std = np.std(train_array)

    if std == 0:
        return train_targets, test_targets, mean, 1.0

    normalized_train = [(t - mean) / std for t in train_targets]
    normalized_test = [(t - mean) / std for t in test_targets]

    return normalized_train, normalized_test, mean, std


class MatrixDataset(Dataset):
    """Dataset for connectivity matrix data."""
    def __init__(self, matrices: Dict[str, np.ndarray], targets: Dict[str, float], subject_ids: List[str]):
        self.matrices = matrices
        self.targets = targets
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        matrix = self.matrices[subject_id].flatten()
        target = self.targets[subject_id]
        return torch.FloatTensor(matrix), torch.FloatTensor([target])


class LinearModel(nn.Module):
    """Linear model for connectivity-based prediction."""
    def __init__(self, input_size=10000):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

    def get_weight_matrix(self, original_shape=(100, 100)):
        """Reshape linear weights back to matrix form."""
        weights = self.linear.weight.data.cpu().numpy().reshape(original_shape)
        return weights


def train_with_early_stopping(model: LinearModel, train_dataset: MatrixDataset, val_dataset: MatrixDataset,
                             num_epochs: int, learning_rate: float, l2_penalty: float, patience: int) -> LinearModel:
    """
    Train model with L2 regularization and early stopping using separate validation set.
    """
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)

    # Prepare training data
    train_matrices = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_targets = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])

    # Prepare validation data
    val_matrices = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
    val_targets = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))])

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        train_outputs = model(train_matrices)
        train_loss = criterion(train_outputs, train_targets)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_matrices)
            val_loss = criterion(val_outputs, val_targets)

        # Check for improvement
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        # Verbose output every 100 epochs
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"    Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item():.6f}, "
                  f"Val Loss: {val_loss.item():.6f}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"    Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.6f}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model: LinearModel, test_dataset: MatrixDataset,
                  target_mean: float = 0, target_std: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on test set and return predictions and targets."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            matrix, target = test_dataset[i]
            pred = model(matrix.unsqueeze(0))

            # Denormalize for regression
            pred_value = pred.item() * target_std + target_mean
            target = target.item() * target_std + target_mean

            predictions.append(pred_value)
            targets.append(target)

    return np.array(predictions), np.array(targets)


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculate performance metrics."""

    mse = float(np.mean((predictions - targets) ** 2))
    mae = float(np.mean(np.abs(predictions - targets)))

    # Calculate correlation
    if len(np.unique(targets)) > 1:
        correlation = float(np.corrcoef(predictions, targets)[0, 1])
    else:
        correlation = 0.0

    # Calculate R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'r_squared': r_squared
    }


def create_result_directory(root_dir: str, git_info: str, start_time: datetime) -> Path:
    """Create timestamped result directory."""
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    script_name = get_script_name()

    result_dir = Path(root_dir) / script_name / git_info / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    return result_dir


def run_kfold_cv(subject_ids: List[str], matrices: Dict[str, np.ndarray],
                targets: List[float], k_folds: int, args: argparse.Namespace) -> Dict:
    """
    Run k-fold cross-validation with special splitting:
    - 1 fold for testing (never seen during training)
    - 1 fold for validation (for early stopping)
    - Remaining folds for training
    """
    assert k_folds >= 3, "Need at least 3 folds for train/val/test split"

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    subject_ids_arr = np.array(subject_ids)

    fold_results = []
    all_predictions = []
    all_targets = []
    weight_matrices = []

    # Convert list of folds to list for easier indexing
    fold_indices = list(kf.split(subject_ids_arr))

    for test_fold_idx in range(k_folds):
        print(f"\n  Fold {test_fold_idx + 1}/{k_folds} (using fold {test_fold_idx + 1} as test set)")

        # Determine validation fold (next fold after test, wrapping around)
        val_fold_idx = (test_fold_idx + 1) % k_folds

        # Get test indices
        _, test_idx = fold_indices[test_fold_idx]
        test_subjects = subject_ids_arr[test_idx]

        # Get validation indices
        _, val_idx = fold_indices[val_fold_idx]
        val_subjects = subject_ids_arr[val_idx]

        # Combine remaining folds for training
        train_idx = []
        for fold_idx in range(k_folds):
            if fold_idx != test_fold_idx and fold_idx != val_fold_idx:
                _ , train_fold_idx = fold_indices[fold_idx]
                train_idx.extend(train_fold_idx)
        train_subjects = subject_ids_arr[train_idx]

        # Assert that train, val, and test subjects are non-overlapping
        train_set = set(train_subjects)
        val_set = set(val_subjects)
        test_set = set(test_subjects)
        assert train_set.isdisjoint(val_set), "Train and validation subjects overlap"
        assert train_set.isdisjoint(test_set), "Train and test subjects overlap"
        assert val_set.isdisjoint(test_set), "Validation and test subjects overlap"

        print(f"    Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)} subjects")

        # Prepare targets
        train_targets_raw = [targets[i] for i in train_idx]
        val_targets_raw = [targets[val_idx[i]] for i in range(len(val_idx))]
        test_targets_raw = [targets[test_idx[i]] for i in range(len(test_idx))]

        # Normalize targets using train+val for normalization statistics
        train_val_targets = train_targets_raw + val_targets_raw
        normalized_train_val, normalized_test, target_mean, target_std = normalize_targets(
            train_val_targets, test_targets_raw
        )
        
        # Split normalized train+val back into train and val
        n_train = len(train_targets_raw)
        train_targets = normalized_train_val[:n_train]
        val_targets = normalized_train_val[n_train:]
        test_targets = normalized_test

        # Create target dictionaries
        train_targets_dict = dict(zip(train_subjects, train_targets))
        val_targets_dict = dict(zip(val_subjects, val_targets))
        test_targets_dict = dict(zip(test_subjects, test_targets))

        # Normalize matrices
        train_val_matrices = {sid: matrices[sid] for sid in np.concatenate([train_subjects, val_subjects])}
        test_matrices = {sid: matrices[sid] for sid in test_subjects}

        normalized_train_val, normalized_test = normalize_matrices(train_val_matrices, test_matrices)

        # Split normalized matrices back into train and val
        normalized_train = {sid: normalized_train_val[sid] for sid in train_subjects}
        normalized_val = {sid: normalized_train_val[sid] for sid in val_subjects}

        # Create datasets
        train_ds = MatrixDataset(normalized_train, train_targets_dict, list(train_subjects))
        val_ds = MatrixDataset(normalized_val, val_targets_dict, list(val_subjects))
        test_ds = MatrixDataset(normalized_test, test_targets_dict, list(test_subjects))

        # Train model
        input_size = matrices[subject_ids[0]].flatten().shape[0]
        model = LinearModel(input_size=input_size)

        model = train_with_early_stopping(
            model, train_ds, val_ds,
            args.epochs, args.learning_rate, args.l2_penalty, args.patience,
        )

        # Evaluate on test set
        predictions, targets_out = evaluate_model(model, test_ds, target_mean, target_std)

        # Store results
        all_predictions.extend(predictions)
        all_targets.extend(targets_out)

        # Store weight matrix (reshape to original dimensions)
        original_shape = matrices[subject_ids[0]].shape
        weight_matrix = model.get_weight_matrix(original_shape)
        weight_matrices.append(weight_matrix)

        # Calculate fold metrics
        fold_metrics = calculate_metrics(predictions, targets_out)
        fold_results.append({
            'fold': test_fold_idx + 1,
            'metrics': fold_metrics,
            'n_train': len(train_subjects),
            'n_val': len(val_subjects),
            'n_test': len(test_subjects)
        })

    # Calculate overall metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    overall_metrics = calculate_metrics(all_predictions, all_targets)

    # Calculate mean weight matrix
    mean_weights = np.mean(weight_matrices, axis=0)

    return {
        'overall_metrics': overall_metrics,
        'fold_results': fold_results,
        'all_predictions': all_predictions.tolist(),
        'all_targets': all_targets.tolist(),
        'weight_matrices': weight_matrices,
        'mean_weights': mean_weights
    }


def main():
    parser = argparse.ArgumentParser(description='Predict behavioral data from connectivity matrices using k-fold CV')
    parser.add_argument('subject_csv', type=str, help='CSV file containing subject IDs')
    parser.add_argument('connectivity_dir', type=str, help='Directory containing subject folders with connectivity matrices')
    parser.add_argument('matrix_pattern', type=str, help='Pattern for matrix files (e.g., "granger_causality.npy")')
    parser.add_argument('behavioral_csv', type=str, help='CSV file with behavioral data')
    parser.add_argument('target_columns_csv', type=str, help='CSV file with target column names to predict')
    parser.add_argument('result_root_dir', type=str, help='Root directory for results')

    # Model parameters
    parser.add_argument('--k-folds', type=int, default=10, help='Number of folds for cross-validation (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate for SGD (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=500, help='Maximum number of epochs (default: 500)')
    parser.add_argument('--l2-penalty', type=float, default=100.0, help='L2 regularization penalty (default: 100.0)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (default: 5)')
    parser.add_argument('--shuffle-targets', action='store_true',
                        help='Shuffle behavioral targets across subjects to test for chance-level performance')

    args = parser.parse_args()

    # Check assertions are enabled
    check_assert_enabled()

    start_time = datetime.now()
    print(f"Behavioral prediction analysis started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get git information
    git_info = get_git_info()
    print(f"Git info: {git_info}")

    # Validate paths
    assert Path(args.subject_csv).exists(), f"Subject CSV not found: {args.subject_csv}"
    assert Path(args.connectivity_dir).exists(), f"Connectivity directory not found: {args.connectivity_dir}"
    assert Path(args.behavioral_csv).exists(), f"Behavioral CSV not found: {args.behavioral_csv}"
    assert Path(args.target_columns_csv).exists(), f"Target columns CSV not found: {args.target_columns_csv}"
    assert args.k_folds >= 3, f"Need at least 3 folds for train/val/test split, got {args.k_folds}"

    # Create result directory
    result_dir = create_result_directory(args.result_root_dir, git_info, start_time)
    print(f"Results will be saved to: {result_dir}")

    # Load subject IDs
    print(f"\nLoading subject IDs from: {args.subject_csv}")
    subject_df = pd.read_csv(args.subject_csv)
    if 'subject_id' in subject_df.columns:
        subject_ids = subject_df['subject_id'].astype(str).tolist()
    else:
        subject_ids = subject_df.iloc[:, 0].astype(str).tolist()
    print(f"Found {len(subject_ids)} subjects")

    # Load target column names
    print(f"Loading target columns from: {args.target_columns_csv}")
    target_columns_df = pd.read_csv(args.target_columns_csv)
    target_columns = target_columns_df.iloc[:, 0].tolist()
    print(f"Will predict {len(target_columns)} targets: {target_columns}")

    # Find matrix files
    print(f"\nSearching for connectivity matrices...")
    matrix_files = find_matrix_files(Path(args.connectivity_dir), args.matrix_pattern, subject_ids)

    assert len(matrix_files) > 0, "No matrix files found!"

    # Load all matrices
    print(f"Loading {len(matrix_files)} connectivity matrices...")
    all_matrices = {}
    matrix_shape = None

    for sid, matrix_path in matrix_files.items():
        matrix = np.load(matrix_path)
        if matrix_shape is None:
            matrix_shape = matrix.shape
            print(f"Matrix shape: {matrix_shape}")
        assert matrix.shape == matrix_shape, f"Inconsistent matrix shape for {sid}: {matrix.shape} != {matrix_shape}"
        all_matrices[sid] = matrix

    # Load behavioral data
    print(f"\nLoading behavioral data from: {args.behavioral_csv}")
    behavioral_df = load_behavioral_targets(Path(args.behavioral_csv), target_columns, args.shuffle_targets)

    # Save configuration
    config = {
        'subject_csv': args.subject_csv,
        'connectivity_dir': args.connectivity_dir,
        'matrix_pattern': args.matrix_pattern,
        'behavioral_csv': args.behavioral_csv,
        'target_columns_csv': args.target_columns_csv,
        'k_folds': args.k_folds,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'l2_penalty': args.l2_penalty,
        'patience': args.patience,
        'shuffle_targets': args.shuffle_targets,
        'n_subjects': len(subject_ids),
        'matrix_shape': list(matrix_shape),
        'git_info': git_info,
        'start_time': start_time.isoformat(),
        'script_name': get_script_name()
    }

    with open(result_dir / 'experiment_metadata.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Process each target column
    all_results = {}

    for target_col in target_columns:
        print(f"\n{'='*70}")
        print(f"Processing target: {target_col}")
        print(f"{'='*70}")

        # Find valid subjects (those with both matrix and target data)
        valid_subject_ids = []
        valid_targets = []

        for sid in subject_ids:
            if sid in matrix_files and sid in behavioral_df.index:
                target_val = behavioral_df.loc[sid, target_col] if target_col in behavioral_df.columns else None
                if target_val is not None and not pd.isna(target_val):
                    valid_subject_ids.append(sid)
                    valid_targets.append(target_val)

        print(f"Valid subjects for {target_col}: {len(valid_subject_ids)}/{len(subject_ids)}")

        # Skip if insufficient data
        min_required = args.k_folds * 2  # At least 2 subjects per fold
        if len(valid_subject_ids) < min_required:
            print(f"Skipping {target_col}: insufficient data ({len(valid_subject_ids)} < {min_required})")
            continue

        # Get matrices for valid subjects
        valid_matrices = {sid: all_matrices[sid] for sid in valid_subject_ids}

        # Run k-fold cross-validation
        cv_results = run_kfold_cv(
            valid_subject_ids, valid_matrices, valid_targets,
            args.k_folds, args
        )

        # Save weight matrices
        weights_dir = result_dir / 'weight_matrices' / target_col
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Save individual fold weights
        for fold_idx, weight_matrix in enumerate(cv_results['weight_matrices']):
            np.save(weights_dir / f'fold_{fold_idx+1}_weights.npy', weight_matrix)

        # Save mean weights
        np.save(weights_dir / f'mean_weights.npy', cv_results['mean_weights'])

        # Prepare results summary
        target_results = {
            'target': target_col,
            'n_valid_subjects': len(valid_subject_ids),
            'overall_metrics': cv_results['overall_metrics'],
            'fold_results': cv_results['fold_results'],
            'predictions': cv_results['all_predictions'],
            'targets': cv_results['all_targets']
        }

        # Save individual target results
        with open(result_dir / f'{target_col}_results.json', 'w') as f:
            json.dump(target_results, f, indent=2)

        # Store in overall results (without predictions for summary)
        summary_results = target_results.copy()
        summary_results.pop('predictions')
        summary_results.pop('targets')
        all_results[target_col] = summary_results

        # Print metrics
        print(f"\nOverall metrics for {target_col}:")
        for metric, value in cv_results['overall_metrics'].items():
            print(f"  {metric}: {value:.4f}")

    # Save overall results summary
    with open(result_dir / 'all_results_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print final summary
    end_time = datetime.now()
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Processed {len(all_results)} targets")
    print(f"Total time: {end_time - start_time}")
    print(f"Results saved to: {result_dir}")


if __name__ == "__main__":
    main()
