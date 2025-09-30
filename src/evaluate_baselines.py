#!/usr/bin/env python3
"""
Baseline evaluation script for fMRI directed connectivity.

This script evaluates three baseline methods on specified subjects:
1. Granger causality (with ridge regression)
2. Pearson correlation
3. Partial correlation (using regression residuals approach)

Results are saved with the directory structure:
result_root_dir/evaluate_baselines/git_info/timestamp/subject_id/
"""

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

from dataset import load_subject_sessions
from training_utils import load_subject_ids
from utils import get_git_info, get_script_name, check_assert_enabled


check_assert_enabled()


def split_timeseries_temporal(data: np.ndarray, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Split timeseries data temporally for train/test.

    Args:
        data: Array of shape (n_regions, n_timepoints)
        test_size: Fraction of data to use for testing

    Returns:
        Tuple of (train_data, test_data)
    """
    assert 0 < test_size < 1, f"test_size must be between 0 and 1, got {test_size}"
    assert data.ndim == 2, f"Data must be 2D, got shape {data.shape}"

    n_regions, n_timepoints = data.shape
    split_point = int(n_timepoints * (1 - test_size))
    assert 0 < split_point < n_timepoints, f"Invalid split point {split_point}"

    return data[:, :split_point], data[:, split_point:]


def prepare_granger_data(data: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for Granger causality using sliding windows.

    Args:
        data: Array of shape (n_regions, n_timepoints)
        window_size: Number of past time steps to use

    Returns:
        Tuple of (X, y) where:
        - X has shape (n_samples, n_regions * window_size)
        - y has shape (n_samples, n_regions)
    """
    assert data.ndim == 2, f"Data must be 2D, got shape {data.shape}"
    assert window_size > 0, f"Window size must be positive, got {window_size}"

    n_regions, n_timepoints = data.shape
    assert n_timepoints > window_size, f"Need at least {window_size + 1} timepoints, got {n_timepoints}"

    X = []
    y = []

    for t in range(window_size, n_timepoints):
        # Input: all regions for past window_size steps
        window = data[:, t - window_size : t]  # (n_regions, window_size)
        X.append(window.T.flatten())  # Flatten to (n_regions * window_size,)

        # Target: all regions at time t
        y.append(data[:, t])

    return np.array(X), np.array(y)


def find_optimal_window_size_group_level(
    subject_sessions: dict, max_window_size: int, test_size: float,
    min_alpha: float, max_alpha: float,
    coarse_steps: int, refine_steps: int
) -> tuple[int, float]:
    """
    Find optimal window size and alpha using data from all subjects.

    Args:
        subject_sessions: Dictionary mapping subject_id to session data
        max_window_size: Maximum window size to try
        test_size: Fraction to use for validation
        min_alpha: Minimum alpha value to try
        max_alpha: Maximum alpha value to try
        coarse_steps: Number of logarithmic steps in coarse search
        refine_steps: Number of refinement steps around best coarse result

    Returns:
        Tuple of (optimal window size, optimal alpha) based on group-level validation
    """
    print(f"Finding optimal window size and alpha at group level (max_window={max_window_size}, alpha_range=[{min_alpha}, {max_alpha}])...")

    def evaluate_alpha_window_combination(alpha_val: float, window_size: int) -> float:
        """Helper function to evaluate a specific alpha-window combination."""
        total_r2 = 0.0
        n_subjects = 0

        for subject_id, session_data in subject_sessions.items():
            # Skip subjects with insufficient data
            if session_data.shape[1] <= window_size * 2:
                continue

            # Split subject data for validation
            train_data, val_data = split_timeseries_temporal(session_data, test_size)

            # Skip if not enough data after splitting
            if train_data.shape[1] <= window_size or val_data.shape[1] <= window_size:
                continue

            # Prepare data
            X_train, y_train = prepare_granger_data(train_data, window_size)
            X_val, y_val = prepare_granger_data(val_data, window_size)

            if len(X_train) == 0 or len(X_val) == 0:
                continue

            # Train model on all regions together
            model = Ridge(alpha=alpha_val, fit_intercept=True)
            model.fit(X_train, y_train)

            # Evaluate on validation set
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)

            total_r2 += r2
            n_subjects += 1

        return total_r2 / n_subjects if n_subjects > 0 else -np.inf

    # Generate coarse alpha values on logarithmic scale
    log_min = np.log10(min_alpha)
    log_max = np.log10(max_alpha)
    coarse_alphas = np.logspace(log_min, log_max, coarse_steps)

    print(f"  Phase 1: Coarse search with {coarse_steps} alpha values")

    best_window_size = 1
    best_alpha = coarse_alphas[0]
    best_r2 = -np.inf
    coarse_results = {}  # Store all coarse results for refinement

    # Coarse search: try all alpha values
    for alpha in coarse_alphas:
        print(f"    Trying alpha = {alpha:.6f}")
        last_r2_values = []

        for window_size in range(1, max_window_size + 1):
            avg_r2 = evaluate_alpha_window_combination(alpha, window_size)

            if avg_r2 == -np.inf:  # No subjects could be evaluated
                break

            print(f"      Window size {window_size}: R² = {avg_r2:.6f}")

            # Update global best if improved
            if avg_r2 > best_r2:
                best_r2 = avg_r2
                best_window_size = window_size
                best_alpha = alpha

            # Store result for potential refinement
            coarse_results[(alpha, window_size)] = avg_r2

            # Track consecutive decreases for stopping criterion
            last_r2_values.append(avg_r2)
            if len(last_r2_values) >= 4:
                # Check if last 3 transitions were all decreases
                decreases = [last_r2_values[i] < last_r2_values[i-1] for i in range(-3, 0)]
                if all(decreases):
                    print(f"      Stopping alpha {alpha:.6f} early due to 3 consecutive decreases")
                    break

    print(f"  Phase 1 best: window_size={best_window_size}, alpha={best_alpha:.6f}, R²={best_r2:.6f}")

    # Phase 2: Fine-grained search around best alpha
    print(f"  Phase 2: Refinement with {refine_steps} steps around best alpha")

    # Find the coarse alpha index for refinement bounds
    best_alpha_idx = np.argmin(np.abs(coarse_alphas - best_alpha))

    # Define refinement range
    if best_alpha_idx == 0:
        # Best is at minimum, refine between best and next
        refine_min = best_alpha
        refine_max = coarse_alphas[1] if len(coarse_alphas) > 1 else best_alpha * 10
    elif best_alpha_idx == len(coarse_alphas) - 1:
        # Best is at maximum, refine between previous and best
        refine_min = coarse_alphas[-2] if len(coarse_alphas) > 1 else best_alpha / 10
        refine_max = best_alpha
    else:
        # Best is in middle, refine between neighbors
        refine_min = coarse_alphas[best_alpha_idx - 1]
        refine_max = coarse_alphas[best_alpha_idx + 1]

    # Generate refinement alpha values
    log_refine_min = np.log10(refine_min)
    log_refine_max = np.log10(refine_max)
    refine_alphas = np.logspace(log_refine_min, log_refine_max, refine_steps + 2)[1:-1]  # Exclude endpoints

    # Refine around best window size (±2 range)
    refine_window_range = range(max(1, best_window_size - 2), min(max_window_size + 1, best_window_size + 3))

    for alpha in refine_alphas:
        print(f"    Refining alpha = {alpha:.6f}")

        for window_size in refine_window_range:
            avg_r2 = evaluate_alpha_window_combination(alpha, window_size)

            if avg_r2 == -np.inf:
                continue

            print(f"      Window size {window_size}: R² = {avg_r2:.6f}")

            # Update best if improved
            if avg_r2 > best_r2:
                best_r2 = avg_r2
                best_window_size = window_size
                best_alpha = alpha

    print(f"Selected group-level optimal: window_size={best_window_size}, alpha={best_alpha:.6f} (R² = {best_r2:.6f})")
    return best_window_size, best_alpha


def compute_granger_causality(
    session_data: np.ndarray, test_size: float, alpha: float, window_size: int
) -> tuple[np.ndarray, dict[str, float | int]]:
    """
    Compute Granger causality connectivity and evaluate predictions.

    Args:
        session_data: Session data of shape (n_regions, n_timepoints)
        test_size: Fraction of data for testing
        alpha: Ridge regression regularization
        window_size: Window size to use (determined at group level)

    Returns:
        Tuple of (connectivity_matrix, metrics_dict)
    """
    n_regions = session_data.shape[0]

    # Split data
    train_data, test_data = split_timeseries_temporal(session_data, test_size)

    # Prepare full train and test sets with group-optimal window
    X_train, y_train = prepare_granger_data(train_data, window_size)
    X_test, y_test = prepare_granger_data(test_data, window_size)

    print("  Training Granger causality models...")

    # Train full model (all regions predicting all regions)
    full_model = Ridge(alpha=alpha, fit_intercept=True)
    full_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = full_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"  Full model - Test MSE: {test_mse:.6f}, R²: {test_r2:.6f}")

    # Compute connectivity matrix using leave-one-out on training set
    print("  Computing connectivity matrix...")
    connectivity = np.zeros((n_regions, n_regions))

    # For each target region
    for target_region in tqdm(range(n_regions), desc="    Target regions", leave=False):
        # Train model with all regions
        full_model_single = Ridge(alpha=alpha, fit_intercept=True)
        full_model_single.fit(X_train, y_train[:, target_region])
        y_pred_full = full_model_single.predict(X_train)
        r2_full = r2_score(y_train[:, target_region], y_pred_full)

        # For each source region, train without it
        for source_region in range(n_regions):
            # Create mask to exclude source region from input
            mask = np.ones(window_size * n_regions, dtype=bool)
            for w in range(window_size):
                mask[w * n_regions + source_region] = False

            # Train model without source region
            X_train_loo = X_train[:, mask]
            loo_model = Ridge(alpha=alpha, fit_intercept=True)
            loo_model.fit(X_train_loo, y_train[:, target_region])
            y_pred_loo = loo_model.predict(X_train_loo)
            r2_loo = r2_score(y_train[:, target_region], y_pred_loo)

            # Connectivity is drop in R² when source is removed
            # Higher value means source is more important for target
            # Matrix layout: rows = targets (being predicted), columns = sources (providing info)
            # This matches the attention matrix where rows = queries (targets), columns = keys/values (sources)
            connectivity[target_region, source_region] = max(0, r2_full - r2_loo)

    metrics = {
        "mse": float(test_mse),
        "r2": float(test_r2),
        "window_size": window_size,
        "alpha": alpha,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
    }

    return connectivity, metrics


def compute_pearson_correlation(session_data: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation matrix.

    Args:
        session_data: Session data of shape (n_regions, n_timepoints)

    Returns:
        Correlation matrix of shape (n_regions, n_regions)
    """
    assert session_data.ndim == 2, f"Data must be 2D, got shape {session_data.shape}"

    # Each row is a region's timeseries
    correlation_matrix = np.corrcoef(session_data)

    return correlation_matrix


def compute_partial_correlation(session_data: np.ndarray) -> np.ndarray:
    """
    Compute partial correlation using precision matrix (inverse covariance) approach.

    Args:
        session_data: Session data of shape (n_regions, n_timepoints)

    Returns:
        Partial correlation matrix of shape (n_regions, n_regions)
    """
    assert session_data.ndim == 2, f"Data must be 2D, got shape {session_data.shape}"

    print("  Computing partial correlations...")

    # Compute covariance matrix (regions x regions)
    cov_matrix = np.cov(session_data)  # Shape: (n_regions, n_regions)

    # Add small regularization for numerical stability
    reg_param = 1e-6 * np.trace(cov_matrix) / cov_matrix.shape[0]
    cov_matrix_reg = cov_matrix + reg_param * np.eye(cov_matrix.shape[0])

    # Compute precision matrix (inverse of covariance)
    try:
        precision_matrix = np.linalg.inv(cov_matrix_reg)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if singular
        precision_matrix = np.linalg.pinv(cov_matrix_reg)

    # Convert precision matrix to partial correlations
    # Partial correlation between i and j: -P_ij / sqrt(P_ii * P_jj)
    diag_sqrt = np.sqrt(np.diag(precision_matrix))
    partial_corr = -precision_matrix / np.outer(diag_sqrt, diag_sqrt)

    return partial_corr


def create_result_directory(result_root_dir: str, git_info: str, start_time: datetime, subject_id: str) -> Path:
    """Create result directory for baseline evaluation."""
    script_name = get_script_name()
    timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    result_dir = Path(result_root_dir) / script_name / git_info / timestamp_str / subject_id
    result_dir.mkdir(parents=True, exist_ok=True)

    return result_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline methods for fMRI connectivity")
    _ = parser.add_argument("subject_ids_csv", type=str, help="Path to CSV file with subject IDs")
    _ = parser.add_argument("data_path", type=str, help="Path to HCP data directory")
    _ = parser.add_argument("atlas_file", type=str, help="Path to brain atlas file")
    _ = parser.add_argument("result_root_dir", type=str, help="Root directory for saving results")
    _ = parser.add_argument("--test-size", type=float, default=0.1, help="Fraction of data for testing")
    _ = parser.add_argument("--min-alpha", type=float, default=0.001, help="Minimum alpha value for ridge regression")
    _ = parser.add_argument("--max-alpha", type=float, default=10000000.0, help="Maximum alpha value for ridge regression")
    _ = parser.add_argument("--coarse-steps", type=int, default=10, help="Number of logarithmic steps in coarse alpha search")
    _ = parser.add_argument("--refine-steps", type=int, default=10, help="Number of refinement steps around best coarse alpha")
    _ = parser.add_argument("--max-window-size", type=int, default=10, help="Maximum window size for Granger causality")

    args = parser.parse_args()

    start_time = datetime.now()
    print(f"Baseline evaluation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get git information
    git_info = get_git_info()
    print(f"Git info: {git_info}")

    # Validate input paths
    assert Path(args.subject_ids_csv).exists(), f"Subject IDs CSV not found: {args.subject_ids_csv}"
    assert Path(args.data_path).exists(), f"Data path not found: {args.data_path}"
    assert Path(args.atlas_file).exists(), f"Atlas file not found: {args.atlas_file}"
    assert 0 < args.test_size < 1, f"test_size must be between 0 and 1, got {args.test_size}"
    assert args.min_alpha > 0, f"min_alpha must be positive, got {args.min_alpha}"
    assert args.max_alpha > args.min_alpha, f"max_alpha must be > min_alpha, got {args.max_alpha} <= {args.min_alpha}"
    assert args.coarse_steps >= 2, f"coarse_steps must be >= 2, got {args.coarse_steps}"
    assert args.refine_steps >= 1, f"refine_steps must be >= 1, got {args.refine_steps}"
    assert args.max_window_size > 0, f"max_window_size must be positive, got {args.max_window_size}"

    # Create result root directory if needed
    Path(args.result_root_dir).mkdir(parents=True, exist_ok=True)

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

    # Find optimal window size and alpha at group level for Granger causality
    print(f"\n{'=' * 60}")
    print("GROUP-LEVEL WINDOW SIZE AND ALPHA OPTIMIZATION")
    print(f"{'=' * 60}")
    optimal_window_size, optimal_alpha = find_optimal_window_size_group_level(
        subject_sessions, args.max_window_size, args.test_size,
        args.min_alpha, args.max_alpha, args.coarse_steps, args.refine_steps
    )

    # Process each subject
    successful_evaluations = []
    granger_r2_scores = []  # Collect R² scores from Granger causality

    for subject_id in subject_sessions.keys():
        print(f"\n{'=' * 60}")
        print(f"Processing subject {subject_id}")
        print(f"{'=' * 60}")

        # Get subject session data
        session_data = subject_sessions[subject_id]
        print(f"Session data shape: {session_data.shape}")

        # Create result directory
        result_dir = create_result_directory(args.result_root_dir, git_info, start_time, subject_id)
        print(f"Saving results to: {result_dir}")

        # 1. Granger Causality
        print(f"\nComputing Granger Causality (using group-optimal window size: {optimal_window_size}, alpha: {optimal_alpha})...")
        granger_connectivity, granger_metrics = compute_granger_causality(
            session_data, args.test_size, optimal_alpha, optimal_window_size
        )

        # Save Granger results
        np.save(result_dir / "granger_causality.npy", granger_connectivity)

        granger_info = {
            "subject_id": subject_id,
            "method": "granger_causality",
            "mse": granger_metrics["mse"],
            "r2": granger_metrics["r2"],
            "window_size": granger_metrics["window_size"],
            "window_size_selection": "group_level",
            "alpha": granger_metrics["alpha"],
            "n_train_samples": granger_metrics["n_train_samples"],
            "n_test_samples": granger_metrics["n_test_samples"],
            "test_size": args.test_size,
            "max_window_size": args.max_window_size,
            "evaluation_timestamp": start_time.isoformat(),
            "note": f"Window size {optimal_window_size} and alpha {optimal_alpha} were selected using all {len(subject_sessions)} subjects",
        }

        with open(result_dir / "granger_metrics.json", "w") as f:
            json.dump(granger_info, f, indent=2)

        print(f"  Granger Causality - MSE: {granger_metrics['mse']:.6f}, R²: {granger_metrics['r2']:.6f}")

        # Collect R² score for summary
        granger_r2_scores.append(granger_metrics['r2'])

        # 2. Pearson Correlation
        print("\nComputing Pearson Correlation...")
        pearson_connectivity = compute_pearson_correlation(session_data)
        np.save(result_dir / "pearson_correlation.npy", pearson_connectivity)
        print("  Pearson correlation matrix saved")

        # 3. Partial Correlation
        print("\nComputing Partial Correlation...")
        partial_connectivity = compute_partial_correlation(session_data)
        np.save(result_dir / "partial_correlation.npy", partial_connectivity)
        print("  Partial correlation matrix saved")

        # Save general info
        general_info = {
            "subject_id": subject_id,
            "n_regions": n_regions,
            "n_timepoints": session_data.shape[1],
            "evaluation_timestamp": start_time.isoformat(),
            "git_info": git_info,
            "methods_computed": ["granger_causality", "pearson_correlation", "partial_correlation"],
        }

        with open(result_dir / "baseline_info.json", "w") as f:
            json.dump(general_info, f, indent=2)

        print(f"Successfully evaluated baselines for subject {subject_id}")
        successful_evaluations.append(subject_id)

    # Print summary
    end_time = datetime.now()
    print(f"\n{'=' * 60}")
    print("BASELINE EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total subjects: {len(subject_ids)}")
    print(f"Successfully evaluated: {len(successful_evaluations)}")
    print(f"Group-level optimal window size: {optimal_window_size}")
    print(f"Group-level optimal alpha: {optimal_alpha}")

    # Compute and print Granger causality R² statistics
    if granger_r2_scores:
        mean_granger_r2 = sum(granger_r2_scores) / len(granger_r2_scores)
        min_granger_r2 = min(granger_r2_scores)
        max_granger_r2 = max(granger_r2_scores)
        print(f"Mean Granger causality R² score: {mean_granger_r2:.6f}")
        print(f"Granger R² range: [{min_granger_r2:.6f}, {max_granger_r2:.6f}]")
        print(f"Individual Granger R² scores: {[f'{r2:.6f}' for r2 in granger_r2_scores]}")

        # Create result directory for summary
        script_name = get_script_name()
        timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        summary_dir = Path(args.result_root_dir) / script_name / git_info / timestamp_str
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Save summary metadata
        summary_metadata = {
            "evaluation_type": "evaluate_baselines_summary",
            "total_subjects_evaluated": len(successful_evaluations),
            "successful_subjects": successful_evaluations,
            "test_size": args.test_size,
            "all_subject_ids": subject_ids,
            "group_optimization": {
                "optimal_window_size": optimal_window_size,
                "optimal_alpha": optimal_alpha,
                "max_window_size": args.max_window_size,
                "min_alpha": args.min_alpha,
                "max_alpha": args.max_alpha,
                "coarse_steps": args.coarse_steps,
                "refine_steps": args.refine_steps
            },
            "granger_r2_statistics": {
                "mean_r2": mean_granger_r2,
                "min_r2": min_granger_r2,
                "max_r2": max_granger_r2,
                "individual_r2_scores": granger_r2_scores,
                "num_subjects_with_r2": len(granger_r2_scores)
            },
            "methods_evaluated": ["granger_causality", "pearson_correlation", "partial_correlation"],
            "evaluation_timestamp": start_time.isoformat(),
            "git_info": git_info
        }

        summary_path = summary_dir / "evaluate_baselines_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_metadata, f, indent=2)

        print(f"Saved evaluation summary to: {summary_path}")

    print(f"Total time: {end_time - start_time}")
    print(f"Results saved to: {args.result_root_dir}")


if __name__ == "__main__":
    main()
