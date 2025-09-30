#!/usr/bin/env python3
"""
Fingerprint individuals based on connectivity matrices across sessions.

This script performs fingerprinting analysis on connectivity matrices by:
1. Computing similarity scores between session pairs
2. Calculating identification accuracy and effect size
3. Analyzing edge importance via ablation
4. Generating visualizations and saving results

The script supports two CSV formats for subject specification:
- Original format: Mixed train/test subjects with group labels
- Test-retest format: Paired subjects (base and _retest versions)

Results are saved with the directory structure:
result_root_dir/fingerprint_connectivity/git_info/timestamp/
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import get_git_info, get_script_name, check_assert_enabled
from scipy import stats
from sklearn.metrics import mutual_info_score


check_assert_enabled()


def load_connectivity_matrices(
    subject_ids: List[str],
    session_test_path: Path,
    session_retest_path: Path,
    matrix_pattern: str,
    shuffle_pairs: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load connectivity matrices for all subjects from both session directories.

    Args:
        subject_ids: List of subject IDs
        session_test_path: Path to test session directory
        session_retest_path: Path to retest session directory
        matrix_pattern: File pattern for matrix files
        shuffle_pairs: If True, shuffle the mapping between test and retest subjects

    Returns:
        Tuple of (session_test_matrices, session_retest_matrices)
    """
    assert len(subject_ids) > 0, "No subject IDs provided"
    assert session_test_path.exists(), f"Session test path does not exist: {session_test_path}"
    assert session_retest_path.exists(), f"Session retest path does not exist: {session_retest_path}"
    assert matrix_pattern, "Matrix pattern cannot be empty"

    session_test_matrices = {}
    session_retest_matrices = {}

    # Create subject ID mapping - shuffle if requested
    retest_subject_ids = subject_ids.copy()
    if shuffle_pairs:
        random.shuffle(retest_subject_ids)
        print(f"Debug: Shuffling pairs enabled. Original->Shuffled mapping:")
        for orig, shuffled in zip(subject_ids, retest_subject_ids):
            print(f"  {orig} -> {shuffled}")

    for i, subject_id in enumerate(subject_ids):
        # Load from session test
        matrix_path_test = session_test_path / subject_id / matrix_pattern
        assert matrix_path_test.exists(), f"Matrix not found: {matrix_path_test}"
        matrix_test = np.load(matrix_path_test)
        assert matrix_test.ndim == 2, f"Matrix must be 2D, got shape {matrix_test.shape} for {subject_id} session test"
        session_test_matrices[subject_id] = matrix_test

        # Load from session retest - subject IDs have "_retest" suffix in retest directory
        # Use the mapped retest subject ID if shuffling is enabled
        retest_subject_for_loading = retest_subject_ids[i]
        subject_id_retest = f"{retest_subject_for_loading}_retest"
        print(f"Debug: Loading {subject_id} -> test: {subject_id}, retest: {subject_id_retest}")
        matrix_path_retest = session_retest_path / subject_id_retest / matrix_pattern
        assert matrix_path_retest.exists(), f"Matrix not found: {matrix_path_retest}"
        matrix_retest = np.load(matrix_path_retest)
        assert matrix_retest.ndim == 2, (
            f"Matrix must be 2D, got shape {matrix_retest.shape} for {subject_id} session retest"
        )
        assert matrix_retest.shape == matrix_test.shape, (
            f"Shape mismatch for {subject_id}: {matrix_test.shape} vs {matrix_retest.shape}"
        )
        # Store with original subject_id (without _retest suffix) for consistent keys
        session_retest_matrices[subject_id] = matrix_retest

    return session_test_matrices, session_retest_matrices


def z_normalize_pixelwise(
    session_test_matrices: Dict[str, np.ndarray], session_retest_matrices: Dict[str, np.ndarray], mode: str = "test_to_retest"
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Z-normalize each pixel within test and retest sessions.

    Args mode controls the normalization strategy:
    - "test_to_retest": Normalize test and retest sessions independently
    - "across_all_sessions": Normalize across all matrices (original behavior)
    
    For each matrix element position (i,j), computes mean and std according to mode,
    then normalizes each matrix element to have mean=0, std=1.

    Args:
        session_test_matrices: Test session matrices
        session_retest_matrices: Retest session matrices
        mode: Normalization mode ("test_to_retest" or "across_all_sessions")

    Returns:
        Tuple of normalized (session_test_matrices, session_retest_matrices)
    """
    assert len(session_test_matrices) > 0, "No session test matrices provided"
    assert len(session_retest_matrices) > 0, "No session retest matrices provided"
    assert set(session_test_matrices.keys()) == set(session_retest_matrices.keys()), (
        "Subject ID mismatch between sessions"
    )

    if mode == "across_all_sessions":
        # Original behavior: normalize across all matrices
        all_matrices = []
        for matrix in session_test_matrices.values():
            all_matrices.append(matrix)
        for matrix in session_retest_matrices.values():
            all_matrices.append(matrix)

        # Validate shapes
        first_shape = all_matrices[0].shape
        for i, matrix in enumerate(all_matrices):
            assert matrix.ndim == 2, f"All matrices must be 2D, matrix {i} has shape {matrix.shape}"
            assert matrix.shape == first_shape, f"All matrices must have same shape, got {matrix.shape} vs {first_shape}"

        stacked = np.stack(all_matrices, axis=0)
        pixel_mean = np.mean(stacked, axis=0)
        pixel_std = np.std(stacked, axis=0)
        pixel_std = np.where(pixel_std == 0, 1, pixel_std)

        # Normalize both sessions using global statistics
        normalized_session_test = {}
        for subject_id, matrix in session_test_matrices.items():
            normalized_session_test[subject_id] = (matrix - pixel_mean) / pixel_std

        normalized_session_retest = {}
        for subject_id, matrix in session_retest_matrices.items():
            normalized_session_retest[subject_id] = (matrix - pixel_mean) / pixel_std

        print(f"Applied pixelwise z-normalization across {len(all_matrices)} matrices")

    elif mode == "test_to_retest":
        # New behavior: normalize test and retest sessions independently
        # Validate shapes for test matrices
        test_matrices = list(session_test_matrices.values())
        first_shape = test_matrices[0].shape
        for i, matrix in enumerate(test_matrices):
            assert matrix.ndim == 2, f"All test matrices must be 2D, matrix {i} has shape {matrix.shape}"
            assert matrix.shape == first_shape, f"All test matrices must have same shape, got {matrix.shape} vs {first_shape}"
        
        # Validate shapes for retest matrices
        retest_matrices = list(session_retest_matrices.values())
        for i, matrix in enumerate(retest_matrices):
            assert matrix.ndim == 2, f"All retest matrices must be 2D, matrix {i} has shape {matrix.shape}"
            assert matrix.shape == first_shape, f"All retest matrices must have same shape, got {matrix.shape} vs {first_shape}"

        # Stack and normalize test matrices independently
        test_stacked = np.stack(test_matrices, axis=0)
        test_pixel_mean = np.mean(test_stacked, axis=0)
        test_pixel_std = np.std(test_stacked, axis=0)
        test_pixel_std = np.where(test_pixel_std == 0, 1, test_pixel_std)

        # Stack and normalize retest matrices independently
        retest_stacked = np.stack(retest_matrices, axis=0)
        retest_pixel_mean = np.mean(retest_stacked, axis=0)
        retest_pixel_std = np.std(retest_stacked, axis=0)
        retest_pixel_std = np.where(retest_pixel_std == 0, 1, retest_pixel_std)

        # Normalize test matrices using test statistics
        normalized_session_test = {}
        for subject_id, matrix in session_test_matrices.items():
            normalized_session_test[subject_id] = (matrix - test_pixel_mean) / test_pixel_std

        # Normalize retest matrices using retest statistics
        normalized_session_retest = {}
        for subject_id, matrix in session_retest_matrices.items():
            normalized_session_retest[subject_id] = (matrix - retest_pixel_mean) / retest_pixel_std

        print(f"Applied pixelwise z-normalization independently: {len(test_matrices)} test matrices, {len(retest_matrices)} retest matrices")
    
    else:
        raise ValueError(f"Unknown normalization mode: {mode}. Must be 'test_to_retest' or 'across_all_sessions'")
    return normalized_session_test, normalized_session_retest


def compute_similarity(
    matrix1: np.ndarray, matrix2: np.ndarray, method: str, n_bins: int, mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute similarity between two connectivity matrices.

    Args:
        matrix1: First connectivity matrix
        matrix2: Second connectivity matrix
        method: Similarity method ('pearson' or 'mutual_info')
        n_bins: Number of bins for mutual information
        mask: Optional mask for edge ablation

    Returns:
        Similarity score
    """
    assert matrix1.shape == matrix2.shape, f"Matrix shape mismatch: {matrix1.shape} vs {matrix2.shape}"
    assert method in ["pearson", "mutual_info"], f"Unknown similarity method: {method}"
    assert n_bins > 0, f"Number of bins must be positive, got {n_bins}"

    # Apply mask if provided (for edge ablation)
    if mask is not None:
        assert mask.shape == matrix1.shape, f"Mask shape mismatch: {mask.shape} vs {matrix1.shape}"
        matrix1 = matrix1.copy()
        matrix2 = matrix2.copy()
        matrix1[mask] = np.nan
        matrix2[mask] = np.nan

    # Flatten matrices and remove NaN values
    flat1 = matrix1.flatten()
    flat2 = matrix2.flatten()

    valid_mask = ~(np.isnan(flat1) | np.isnan(flat2))
    flat1 = flat1[valid_mask]
    flat2 = flat2[valid_mask]

    if len(flat1) == 0:
        return 0.0

    if method == "pearson":
        if np.std(flat1) == 0 or np.std(flat2) == 0:
            return 0.0
        result = stats.pearsonr(flat1, flat2)
        return float(result.statistic)
    elif method == "mutual_info":
        # Bin the data
        bins1 = np.histogram_bin_edges(flat1, bins=n_bins)
        bins2 = np.histogram_bin_edges(flat2, bins=n_bins)

        # Digitize into bins
        digitized1 = np.digitize(flat1, bins1[:-1]) - 1
        digitized2 = np.digitize(flat2, bins2[:-1]) - 1

        assert (0 <= digitized1).all() and (digitized1 < n_bins).all()
        assert (0 <= digitized2).all() and (digitized2 < n_bins).all()

        return float(mutual_info_score(digitized1, digitized2))
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def get_all_matrices_list(
    session_test_matrices: Dict[str, np.ndarray], session_retest_matrices: Dict[str, np.ndarray]
) -> List[Tuple[str, np.ndarray, bool]]:
    """
    Get all matrices as a list of (subject_id, matrix, is_retest) tuples.

    Args:
        session_test_matrices: Test session matrices
        session_retest_matrices: Retest session matrices

    Returns:
        List of (subject_id, matrix, is_retest) tuples
    """
    all_matrices = []
    for subject_id, matrix in session_test_matrices.items():
        all_matrices.append((subject_id, matrix, False))
    for subject_id, matrix in session_retest_matrices.items():
        all_matrices.append((subject_id, matrix, True))
    return all_matrices






def compute_cross_subject_scores_for_subject(
    subject_id: str,
    subject_matrix_test: np.ndarray,
    subject_matrix_retest: np.ndarray,
    session_test_matrices: Dict[str, np.ndarray],
    session_retest_matrices: Dict[str, np.ndarray],
    method: str,
    mode: str = "test_to_retest",
    n_bins: int = 10,
    mask: Optional[np.ndarray] = None,
) -> List[float]:
    """
    Compute cross-subject scores for a specific subject against other subjects.
    
    Args mode controls comparison strategy:
    - "test_to_retest": Only compares test matrices to retest matrices
    - "across_all_sessions": Compares against all other matrices (original behavior)

    Args:
        subject_id: Subject ID
        subject_matrix_test: Subject's matrix from test session
        subject_matrix_retest: Subject's matrix from retest session
        session_test_matrices: All test session matrices
        session_retest_matrices: All retest session matrices
        method: Similarity method
        mode: Comparison mode ("test_to_retest" or "across_all_sessions")
        n_bins: Number of bins for mutual information
        mask: Optional mask for edge ablation

    Returns:
        List of cross-subject similarity scores
    """
    cross_scores = []
    
    if mode == "test_to_retest":
        # Compare subject's test matrix to other subjects' retest matrices
        for other_id, other_retest_matrix in session_retest_matrices.items():
            if other_id != subject_id:
                score = compute_similarity(subject_matrix_test, other_retest_matrix, method, n_bins, mask)
                cross_scores.append(score)
        
        # Compare subject's retest matrix to other subjects' test matrices
        for other_id, other_test_matrix in session_test_matrices.items():
            if other_id != subject_id:
                score = compute_similarity(subject_matrix_retest, other_test_matrix, method, n_bins, mask)
                cross_scores.append(score)
    
    elif mode == "across_all_sessions":
        # Original behavior: compare against all other matrices
        all_matrices = get_all_matrices_list(session_test_matrices, session_retest_matrices)
        for other_id, matrix, _ in all_matrices:
            if other_id != subject_id:
                # Compare both subject matrices with this other matrix
                score1 = compute_similarity(subject_matrix_test, matrix, method, n_bins, mask)
                score2 = compute_similarity(subject_matrix_retest, matrix, method, n_bins, mask)
                cross_scores.extend([score1, score2])
    
    else:
        raise ValueError(f"Unknown comparison mode: {mode}. Must be 'test_to_retest' or 'across_all_sessions'")

    return cross_scores


def compute_all_similarities(
    session_test_matrices: Dict[str, np.ndarray],
    session_retest_matrices: Dict[str, np.ndarray],
    method: str,
    n_bins: int,
    mode: str = "test_to_retest",
    mask: Optional[np.ndarray] = None,
) -> Tuple[List[float], List[float]]:
    """
    Compute all same-subject and cross-subject similarities.

    Args:
        session_test_matrices: Test session matrices
        session_retest_matrices: Retest session matrices
        method: Similarity method
        mode: Comparison mode ("test_to_retest" or "across_all_sessions")
        n_bins: Number of bins for mutual information
        mask: Optional mask for edge ablation

    Returns:
        Tuple of (same_subject_scores, cross_subject_scores)
    """
    assert len(session_test_matrices) > 0, "No session test matrices provided"
    assert len(session_retest_matrices) > 0, "No session retest matrices provided"
    assert set(session_test_matrices.keys()) == set(session_retest_matrices.keys()), (
        "Subject ID mismatch between sessions"
    )

    same_subject_scores = []
    cross_subject_scores = []

    # Same-subject scores
    for subject_id in session_test_matrices.keys():
        score = compute_similarity(
            session_test_matrices[subject_id], session_retest_matrices[subject_id], method, n_bins, mask
        )
        same_subject_scores.append(score)

    # Cross-subject scores
    for subject_id in session_test_matrices.keys():
        cross_subject_scores.extend(
            compute_cross_subject_scores_for_subject(
                subject_id=subject_id,
                subject_matrix_test=session_test_matrices[subject_id],
                subject_matrix_retest=session_retest_matrices[subject_id],
                session_test_matrices=session_test_matrices,
                session_retest_matrices=session_retest_matrices,
                method=method,
                mode=mode,
                n_bins=n_bins,
                mask=mask,
            )
        )

    return same_subject_scores, cross_subject_scores


def calculate_identification_accuracy(
    session_test_matrices: Dict[str, np.ndarray],
    session_retest_matrices: Dict[str, np.ndarray],
    method: str,
    mode: str = "test_to_retest",
    n_bins: int = 10,
) -> float:
    """
    Calculate identification accuracy by comparing same-subject vs all cross-subject scores.

    Args:
        session_test_matrices: Test session matrices
        session_retest_matrices: Retest session matrices
        method: Similarity method
        n_bins: Number of bins for mutual information

    Returns:
        Identification accuracy as fraction of correctly identified subjects
    """
    subject_ids = list(session_test_matrices.keys())
    assert len(subject_ids) > 0, "No subjects provided for identification accuracy calculation"

    correct_identifications = 0

    for subject_id in subject_ids:
        # Compute same-subject score
        # Same subject score
        same_score = compute_similarity(
            session_test_matrices[subject_id], session_retest_matrices[subject_id], method, n_bins
        )  # We assume that the similarity measure is commutative, so we don't compute the flipped same score

        # Cross-subject scores for this subject
        cross_scores = compute_cross_subject_scores_for_subject(
            subject_id,
            session_test_matrices[subject_id],
            session_retest_matrices[subject_id],
            session_test_matrices,
            session_retest_matrices,
            method,
            mode,
            n_bins,
        )

        assert len(cross_scores) > 0

        # Subject is correctly identified if same-subject score beats ALL cross-subject scores
        if same_score > max(cross_scores):
            correct_identifications += 1

    return correct_identifications / len(subject_ids)


def calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        group1: First group of scores
        group2: Second group of scores

    Returns:
        Cohen's d effect size
    """
    assert len(group1) > 1, f"Group 1 must have at least 2 samples, got {len(group1)}"
    assert len(group2) > 1, f"Group 2 must have at least 2 samples, got {len(group2)}"

    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)

    n1 = len(group1)
    n2 = len(group2)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((mean1 - mean2) / pooled_std)


def calculate_glass_delta(subject_score: float, cross_scores: List[float]) -> float:
    """
    Calculate Glass' Δ using only the control (cross-subject) standard deviation.

    Args:
        subject_score: Subject's own similarity score
        cross_scores: List of cross-subject similarity scores

    Returns:
        Glass' Δ effect size
    """
    assert len(cross_scores) > 1, f"Need at least 2 cross scores for Glass' delta, got {len(cross_scores)}"

    mean_cross = np.mean(cross_scores)
    std_cross = np.std(cross_scores, ddof=1)

    if std_cross == 0:
        return 0.0

    return float((subject_score - mean_cross) / std_cross)


def _compute_edge_correlation_contribution(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    Compute edge-wise contribution to Pearson correlation between two matrices.

    Args:
        matrix1: First connectivity matrix
        matrix2: Second connectivity matrix

    Returns:
        Matrix of edge-wise correlation contributions
    """
    # Center the matrices
    flat1 = matrix1.flatten()
    flat2 = matrix2.flatten()
    mean1 = np.mean(flat1)
    mean2 = np.mean(flat2)
    centered1 = matrix1 - mean1
    centered2 = matrix2 - mean2

    # Compute standard deviations
    std1 = np.std(flat1)
    std2 = np.std(flat2)

    # Edge-wise contribution to correlation
    if std1 > 0 and std2 > 0:
        contribution = (centered1 * centered2) / (std1 * std2 * flat1.size)
    else:
        contribution = np.zeros(matrix1.shape)

    return contribution


def compute_edge_importance(
    subject_id: str,
    session_test_matrices: Dict[str, np.ndarray],
    session_retest_matrices: Dict[str, np.ndarray],
    method: str,
    mode: str = "test_to_retest",
    n_bins: int = 10
) -> np.ndarray:
    """
    Compute importance of each edge for identifying a specific subject.
    
    Args mode controls comparison strategy:
    - "test_to_retest": Only considers test-to-retest comparisons
    - "across_all_sessions": Considers comparisons against all matrices (original behavior)

    For Pearson correlation, we compute the edge-wise contribution to the correlation
    directly without ablation, as correlation is linear in edge contributions.

    Args:
        subject_id: Subject ID
        session_test_matrices: All test session matrices
        session_retest_matrices: All retest session matrices
        method: Similarity method (only 'pearson' is optimized)
        mode: Comparison mode ("test_to_retest" or "across_all_sessions")
        n_bins: Number of bins for mutual information (ignored for pearson)

    Returns:
        Edge importance matrix where each element represents how much that edge
        contributes to distinguishing the same-subject pair from cross-subject pairs
    """
    assert subject_id in session_test_matrices, f"Subject {subject_id} not found in session test"
    assert subject_id in session_retest_matrices, f"Subject {subject_id} not found in session retest"
    assert method == "pearson", "Only Pearson correlation is supported for edge importance computation"

    # Get subject matrices
    subject_matrix_test = session_test_matrices[subject_id]
    subject_matrix_retest = session_retest_matrices[subject_id]

    # Compute edge-wise correlation contribution for same-subject pair (test-to-retest)
    same_subject_contribution = _compute_edge_correlation_contribution(subject_matrix_test, subject_matrix_retest)

    # Compute mean edge-wise contribution across cross-subject pairs for this subject
    cross_subject_contributions = []

    if mode == "test_to_retest":
        # Compare subject's test matrix to other subjects' retest matrices
        for other_id, other_retest_matrix in session_retest_matrices.items():
            if other_id != subject_id:
                contrib = _compute_edge_correlation_contribution(subject_matrix_test, other_retest_matrix)
                cross_subject_contributions.append(contrib)
        
        # Compare subject's retest matrix to other subjects' test matrices
        for other_id, other_test_matrix in session_test_matrices.items():
            if other_id != subject_id:
                contrib = _compute_edge_correlation_contribution(subject_matrix_retest, other_test_matrix)
                cross_subject_contributions.append(contrib)
    
    elif mode == "across_all_sessions":
        # Original behavior: compare against all other matrices
        all_matrices = get_all_matrices_list(session_test_matrices, session_retest_matrices)
        for other_id, other_matrix, _ in all_matrices:
            if other_id != subject_id:
                # Pair this subject's test with other's matrix
                contrib_test = _compute_edge_correlation_contribution(subject_matrix_test, other_matrix)
                cross_subject_contributions.append(contrib_test)

                # Pair this subject's retest with other's matrix
                contrib_retest = _compute_edge_correlation_contribution(subject_matrix_retest, other_matrix)
                cross_subject_contributions.append(contrib_retest)
    
    else:
        raise ValueError(f"Unknown comparison mode: {mode}. Must be 'test_to_retest' or 'across_all_sessions'")

    # Compute mean contribution across cross-subject pairs
    if cross_subject_contributions:
        mean_cross_subject_contribution = np.mean(cross_subject_contributions, axis=0)
    else:
        mean_cross_subject_contribution = np.zeros(same_subject_contribution.shape)

    # Edge importance is the difference between same-subject contribution and mean cross-subject contribution
    edge_importance = same_subject_contribution - mean_cross_subject_contribution

    return edge_importance





def create_result_directory(result_root_dir: str, git_info: str, start_time: datetime) -> Path:
    """Create result directory for fingerprinting output."""
    script_name = get_script_name()
    timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    result_dir = Path(result_root_dir) / script_name / git_info / timestamp_str
    result_dir.mkdir(parents=True, exist_ok=True)

    return result_dir


def main():
    parser = argparse.ArgumentParser(description="Fingerprint individuals using connectivity matrices")
    parser.add_argument("subject_csv", type=str, help="CSV file containing subject IDs (supports both original format with mixed subjects and test-retest format with paired subjects)")
    parser.add_argument("matrix_pattern", type=str, help='Pattern for npy matrix files (e.g., "mean_attention.npy")')
    parser.add_argument("session_test_path", type=str, help="Path to test session directory")
    parser.add_argument("session_retest_path", type=str, help="Path to retest session directory")
    parser.add_argument("result_root_dir", type=str, help="Root directory for results")
    parser.add_argument(
        "--method",
        type=str,
        choices=["pearson", "mutual_info"],
        default="pearson",
        help="Similarity measure to use (default: pearson)",
    )
    parser.add_argument("--n-bins", type=int, default=10, help="Number of bins for mutual information (default: 10)")
    parser.add_argument("--z-normalize", action="store_true", help="Apply pixelwise z-normalization across subjects")
    parser.add_argument(
        "--compute-edge-importance",
        action="store_true",
        help="Compute edge importance matrices (computationally intensive)",
    )
    parser.add_argument(
        "--comparison-mode",
        type=str,
        choices=["test_to_retest", "across_all_sessions"],
        default="test_to_retest",
        help="Comparison strategy: 'test_to_retest' only compares test to retest matrices, 'across_all_sessions' compares all matrices (default: test_to_retest)",
    )
    parser.add_argument(
        "--shuffle-pairs",
        action="store_true",
        help="Shuffle test-retest subject pairs to test for chance-level performance",
    )

    args = parser.parse_args()

    start_time = datetime.now()
    print(f"Fingerprinting analysis started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get git information
    git_info = get_git_info()
    print(f"Git info: {git_info}")

    # Validate input paths
    assert Path(args.subject_csv).exists(), f"Subject CSV not found: {args.subject_csv}"
    assert Path(args.session_test_path).exists(), f"Session test path not found: {args.session_test_path}"
    assert Path(args.session_retest_path).exists(), f"Session retest path not found: {args.session_retest_path}"

    # Validate edge importance computation settings
    if args.compute_edge_importance and args.method != "pearson":
        raise ValueError(
            f"Edge importance computation is only supported with 'pearson' method. Got method='{args.method}'"
        )

    # Create result directory
    result_dir = create_result_directory(args.result_root_dir, git_info, start_time)
    print(f"Results will be saved to: {result_dir}")

    # Load subject IDs
    print(f"Loading subject IDs from: {args.subject_csv}")
    subject_df = pd.read_csv(args.subject_csv)
    assert "subject_id" in subject_df.columns or subject_df.shape[1] == 1, (
        "CSV must have 'subject_id' column or single column"
    )

    if "subject_id" in subject_df.columns:
        all_subject_ids = subject_df["subject_id"].astype(str).tolist()
    else:
        all_subject_ids = subject_df.iloc[:, 0].astype(str).tolist()

    print(f"Loaded {len(all_subject_ids)} total subject entries from CSV")

    # Filter to get only base subject IDs (remove _retest suffix subjects)
    # This handles both formats: original subject_groups.csv and test-retest paired CSV
    subject_ids = [sid for sid in all_subject_ids if not sid.endswith("_retest")]
    retest_subjects = [sid for sid in all_subject_ids if sid.endswith("_retest")]

    # Detect and log CSV format
    if retest_subjects:
        print(f"✓ Detected test-retest paired CSV format:")
        print(f"  - {len(subject_ids)} base subjects")
        print(f"  - {len(retest_subjects)} retest subjects")

        # Verify that each base subject has a corresponding retest pair
        expected_retest = [f"{sid}_retest" for sid in subject_ids]
        missing_retest = [sid for sid in expected_retest if sid not in retest_subjects]
        extra_retest = [sid for sid in retest_subjects if sid.replace("_retest", "") not in subject_ids]

        if missing_retest:
            print(f"  ⚠ Warning: Missing retest subjects: {missing_retest}")
        if extra_retest:
            print(f"  ⚠ Warning: Retest subjects without base: {extra_retest}")
        if not missing_retest and not extra_retest:
            print(f"  ✓ All {len(subject_ids)} base subjects have matching retest pairs")
    else:
        print(f"✓ Detected original CSV format with {len(subject_ids)} subjects")
        if len(all_subject_ids) == len(subject_ids):
            print("  - No retest subjects found (standard format)")

    # Show example subjects
    if len(subject_ids) <= 5:
        print(f"  Base subjects: {subject_ids}")
    else:
        print(f"  Example base subjects: {subject_ids[:3]}...{subject_ids[-1]} (+{len(subject_ids)-4} more)")

    assert len(subject_ids) > 1, f"Need at least 2 subjects for fingerprinting, got {len(subject_ids)}"
    print(f"→ Processing {len(subject_ids)} base subjects for fingerprinting analysis")

    # Load connectivity matrices
    print("Loading connectivity matrices...")
    if args.shuffle_pairs:
        print("WARNING: Shuffle pairs enabled - this will randomize test-retest subject mappings!")
    session_test_matrices, session_retest_matrices = load_connectivity_matrices(
        subject_ids,
        Path(args.session_test_path),
        Path(args.session_retest_path),
        args.matrix_pattern,
        args.shuffle_pairs,
    )

    # Get matrix shape for validation
    first_subject = next(iter(session_test_matrices.keys()))
    matrix_shape = session_test_matrices[first_subject].shape
    print(f"Matrix shape: {matrix_shape}")

    # Apply z-normalization if requested
    if args.z_normalize:
        print("Applying pixelwise z-normalization...")
        session_test_matrices, session_retest_matrices = z_normalize_pixelwise(
            session_test_matrices, session_retest_matrices, mode=args.comparison_mode
        )

    # Compute similarity scores
    print("Computing similarities...")
    same_subject_scores, cross_subject_scores = compute_all_similarities(
        session_test_matrices, session_retest_matrices, args.method, args.n_bins, mode=args.comparison_mode
    )

    # Calculate metrics
    print("Calculating metrics...")
    identification_accuracy = calculate_identification_accuracy(
        session_test_matrices, session_retest_matrices, args.method, mode=args.comparison_mode, n_bins=args.n_bins
    )

    effect_size = calculate_cohens_d(same_subject_scores, cross_subject_scores)

    # Compute edge importance if requested
    if args.compute_edge_importance:
        print("Computing edge importance matrices (this may take a while)...")
        importance_matrices = {}

        for subject_id in tqdm(subject_ids, desc="Computing edge importance"):
            importance_matrices[subject_id] = compute_edge_importance(
                subject_id, session_test_matrices, session_retest_matrices, args.method, mode=args.comparison_mode
            )


        for subject_id, matrix in importance_matrices.items():

            # Save individual importance matrices
            importance_dir = result_dir / "importance_matrices" / subject_id
            importance_dir.mkdir(exist_ok=True, parents=True)
            np.save(importance_dir / "importance.npy", matrix)

        # Compute and save group-level importance matrix
        group_importance = np.mean(list(importance_matrices.values()), axis=0)
        np.save(result_dir / "group_importance_matrix.npy", group_importance)

        print(f"Group importance matrix saved to: {result_dir / 'group_importance_matrix.npy'}")

    # Save configuration and metadata
    metadata = {
        "subject_csv": args.subject_csv,
        "matrix_pattern": args.matrix_pattern,
        "session_test_path": args.session_test_path,
        "session_retest_path": args.session_retest_path,
        "method": args.method,
        "comparison_mode": args.comparison_mode,
        "n_bins": args.n_bins,
        "z_normalize": args.z_normalize,
        "compute_edge_importance": args.compute_edge_importance,
        "shuffle_pairs": args.shuffle_pairs,
        "n_subjects": len(subject_ids),
        "subject_ids": subject_ids,
        "matrix_shape": list(matrix_shape),
        "git_info": git_info,
        "start_time": start_time.isoformat(),
        "script_name": get_script_name(),
    }

    with open(result_dir / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save results
    results = {
        "identification_accuracy": float(identification_accuracy),
        "effect_size_cohens_d": float(effect_size),
        "same_subject_scores": [float(x) for x in same_subject_scores],
        "cross_subject_scores": [float(x) for x in cross_subject_scores],
        "same_subject_mean": float(np.mean(same_subject_scores)),
        "same_subject_std": float(np.std(same_subject_scores)),
        "cross_subject_mean": float(np.mean(cross_subject_scores)),
        "cross_subject_std": float(np.std(cross_subject_scores)),
        "n_same_subject_pairs": len(same_subject_scores),
        "n_cross_subject_pairs": len(cross_subject_scores),
    }

    with open(result_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    end_time = datetime.now()
    print(f"\n{'=' * 60}")
    print("FINGERPRINTING RESULTS")
    print(f"{'=' * 60}")
    print(f"Identification Accuracy: {identification_accuracy:.2%}")
    print(f"Effect Size (Cohen's d): {effect_size:.3f}")
    print(f"Same-subject similarity: {np.mean(same_subject_scores):.3f} ± {np.std(same_subject_scores):.3f}")
    print(f"Cross-subject similarity: {np.mean(cross_subject_scores):.3f} ± {np.std(cross_subject_scores):.3f}")
    print(f"Same-subject pairs: {len(same_subject_scores)}")
    print(f"Cross-subject pairs: {len(cross_subject_scores)}")
    print(f"Total time: {end_time - start_time}")
    print(f"Results saved to: {result_dir}")


if __name__ == "__main__":
    main()
