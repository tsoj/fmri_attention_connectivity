"""
Simplified dataset module for HCP fMRI data loading.
Provides basic functionality to load parcellated sessions for specific subjects.
"""

import re
import warnings
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import nibabel as nb
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_schaefer_dlabel(atlas_file: str, num_cortical_vertices: int = 64984):
    """
    Load the Schaefer dlabel atlas and return parcellation information.

    Args:
        atlas_file: Path to the atlas file
        num_cortical_vertices: Number of cortical vertices expected

    Returns:
        Tuple of (labels_cortical, unique_parcels, parcel_index_map, weights_matrix)
    """
    atlas_img = nb.load(atlas_file)
    atlas_labels_float = atlas_img.get_fdata()
    atlas_labels = atlas_labels_float.astype(np.int32).squeeze(0)

    # Restrict to cortex
    assert atlas_labels.shape[0] >= num_cortical_vertices, (
        f"Atlas has fewer grayordinates ({atlas_labels.shape[0]}) than expected cortical vertices ({num_cortical_vertices})"
    )
    labels_cortical = atlas_labels[:num_cortical_vertices]

    unique_parcels = np.unique(labels_cortical)
    unique_parcels = unique_parcels[unique_parcels != 0]  # drop background
    n_parcels = len(unique_parcels)
    assert n_parcels > 0, "No parcels found in atlas after removing background"
    parcel_index_map = {pid: i for i, pid in enumerate(unique_parcels)}

    # Build averaging weight matrix VxP (V=vertices, P=parcels)
    weights = np.zeros((num_cortical_vertices, n_parcels), dtype=np.float32)
    for i, pid in enumerate(unique_parcels):
        mask = labels_cortical == pid
        cnt = int(mask.sum())
        assert cnt > 0, f"Parcel {pid} has no vertices"
        weights[mask, i] = 1.0 / float(cnt)

    return labels_cortical, unique_parcels, parcel_index_map, weights


def find_hcp_rest_dtseries(root_dir: str, subject_ids: List[str]) -> List[Path]:
    """
    Find rfMRI REST dtseries files for specific subjects in the HCP directory structure.

    Args:
        root_dir: Root directory containing HCP data
        subject_ids: List of subject IDs to find files for

    Returns:
        List of paths to dtseries files for the specified subjects
    """
    root = Path(root_dir)
    dtseries_files: List[Path] = []
    subject_ids_set = set(subject_ids)

    # Get all subject directories and sort them by their string representation
    subject_dirs = [p for p in root.iterdir() if p.is_dir() and p.name in subject_ids_set]
    subject_dirs = sorted(subject_dirs, key=lambda p: str(p.absolute()))

    for subj_dir in subject_dirs:
        rest_dir = subj_dir / "MNINonLinear" / "Results" / "rfMRI_REST"
        if rest_dir.is_dir():
            # Get all dtseries files and sort them by their string representation
            rest_files = list(rest_dir.glob("*.dtseries.nii"))
            rest_files = sorted(rest_files, key=lambda p: str(p.absolute()))
            dtseries_files.extend(rest_files)

    return dtseries_files


def subject_id_from_path(path: Path) -> str:
    """Extract subject ID from file path."""
    parts = path.parts
    # Find the segment just before 'MNINonLinear'
    for i, seg in enumerate(parts):
        if seg == "MNINonLinear" and i > 0:
            return parts[i - 1]
    # Fallback to first numeric-looking part
    for seg in parts:
        if re.fullmatch(r"\d+", seg or ""):
            return seg
    return path.parts[0]


def parcellate_dtseries_file(
    dtseries_path: Path,
    weights_matrix: np.ndarray,
    num_cortical_vertices: int = 64984,
) -> np.ndarray:
    """
    Load and parcellate a single dtseries file.

    Args:
        dtseries_path: Path to dtseries file
        weights_matrix: Parcellation weights matrix
        num_cortical_vertices: Number of cortical vertices

    Returns:
        Parcellated time series as np.ndarray with shape (n_parcels, T)
    """
    img = nb.load(str(dtseries_path))
    data = img.get_fdata(dtype=np.float32)  # shape: (T, Ngrayordinates)

    assert data.ndim == 2, f"Expected 2D dtseries data, got {data.ndim}D for {dtseries_path}: {data.shape}"

    T, N = data.shape
    assert T > 0, f"No time points in {dtseries_path}"
    assert N >= num_cortical_vertices, f"{dtseries_path} has {N} grayordinates; expected >= {num_cortical_vertices}"

    # Cortex-only
    cortex = data[:, :num_cortical_vertices]  # (T, V)
    # Parcellate via matrix multiply: (T, V) @ (V, P) -> (T, P)
    parcellated = cortex @ weights_matrix  # (T, n_parcels)
    # Transpose to (P, T) for consistency
    parcel_ts = parcellated.T.copy()  # (n_parcels, T)

    return parcel_ts


def load_subject_sessions(
    root_dir: str,
    subject_ids: List[str],
    atlas_file: str,
    num_cortical_vertices: int = 64984,
) -> tuple[Dict[str, np.ndarray], int]:
    """
    Load parcellated sessions for specified subjects.

    Args:
        root_dir: Root directory containing HCP data
        subject_ids: List of subject IDs to load
        atlas_file: Path to atlas file
        num_cortical_vertices: Number of cortical vertices

    Returns:
        Dictionary mapping subject IDs to their parcellated session arrays
        Each array has shape (n_parcels, T) where T is the number of time points
    """
    # Load atlas & weights
    _, unique_parcels, _, weights = load_schaefer_dlabel(atlas_file, num_cortical_vertices)
    n_parcels = len(unique_parcels)
    print(f"Loaded Schaefer atlas: {n_parcels} parcels (cortical).")

    # Find dtseries files for specified subjects
    dtseries_files = find_hcp_rest_dtseries(root_dir, subject_ids)
    print(f"Found {len(dtseries_files)} rfMRI_REST dtseries files for {len(subject_ids)} subjects.")

    assert len(dtseries_files) > 0, f"No HCP REST dtseries files found for subjects {subject_ids} in {root_dir}"

    # Load and parcellate files
    subject_sessions = {}
    for dtseries_path in tqdm(dtseries_files, desc="Loading and parcellating"):
        parcel_ts = parcellate_dtseries_file(dtseries_path, weights, num_cortical_vertices)
        subj_id = subject_id_from_path(dtseries_path)

        assert subj_id not in subject_sessions, "Currently only one session per subject is supported"
        subject_sessions[subj_id] = parcel_ts

    print(f"Successfully loaded {len(subject_sessions)} sessions.")
    assert len(subject_sessions) > 0, "No sessions were successfully loaded"

    # Check if we found sessions for all requested subjects
    missing_subjects = set(subject_ids) - set(subject_sessions.keys())
    if missing_subjects:
        print(f"Warning: Could not find sessions for subjects: {missing_subjects}")
    assert len(subject_sessions) <= len(subject_ids)

    return subject_sessions, n_parcels
