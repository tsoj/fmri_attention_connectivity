#!/usr/bin/env python3
"""
Connectivity matrix visualization script with multiple visualization types.

This script visualizes connectivity/causality/attention matrices from multiple subjects
using three visualization methods:
- Full connectivity matrix with network/anatomical grouping
- Coarse connectivity matrix (mean between groups)
- Circle connectogram

Results are saved with the directory structure:
result_root_dir/visualize_connectivity/git_info/timestamp/{full_matrix,coarse_matrix,circle}/
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent popup windows
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from datetime import datetime
from pathlib import Path
from typing import Optional
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_destrieux_2009
from nilearn import image
from mne_connectivity.viz import plot_connectivity_circle

from utils import get_git_info, get_script_name, check_assert_enabled


check_assert_enabled()

CMAP = sns.cubehelix_palette(as_cmap=True)

def find_subject_files(base_path: str, file_pattern: str) -> dict[str, Path]:
    """
    Find all files matching the pattern in subject subdirectories.

    Args:
        base_path: Path to directory containing subject subdirectories
        file_pattern: File pattern to match (e.g., "granger_causality.npy")

    Returns:
        Dictionary mapping subject_id -> file_path
    """
    base_path_obj = Path(base_path)
    assert base_path_obj.exists(), f"Base path does not exist: {base_path}"

    subject_files = {}

    for subject_dir in base_path_obj.iterdir():
        if subject_dir.is_dir():
            target_file = subject_dir / file_pattern
            if target_file.exists():
                subject_files[subject_dir.name] = target_file

    assert len(subject_files) > 0, f"No files matching '{file_pattern}' found in subject directories under {base_path}"

    print(f"Found {len(subject_files)} files matching pattern '{file_pattern}'")
    return subject_files


def apply_row_normalization(matrix: np.ndarray) -> np.ndarray:
    """
    Apply row normalization: offset to min=0, then normalize to sum=1.

    Args:
        matrix: Input matrix

    Returns:
        Row-normalized matrix
    """
    normalized = matrix.copy()

    for i in range(matrix.shape[0]):
        row = normalized[i, :]
        # Offset to minimum = 0
        row_min = np.min(row)
        row = row - row_min

        # Normalize to sum = 1 (avoid division by zero)
        row_sum = np.sum(row)
        if row_sum > 0:
            row = row / row_sum

        normalized[i, :] = row

    return normalized


def apply_column_mean_subtraction(matrix: np.ndarray) -> np.ndarray:
    """
    Subtract mean of each column.

    Args:
        matrix: Input matrix

    Returns:
        Column mean-subtracted matrix
    """
    return matrix - np.mean(matrix, axis=0, keepdims=True)


def apply_group_mean_subtraction(matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Subtract group mean across all subjects.

    Args:
        matrices: Dictionary of subject_id -> matrix

    Returns:
        Dictionary of subject_id -> mean-subtracted matrix
    """
    # Compute group mean
    all_matrices = list(matrices.values())
    group_mean = np.mean(all_matrices, axis=0)

    # Subtract from each subject
    result = {}
    for subject_id, matrix in matrices.items():
        result[subject_id] = matrix - group_mean

    return result


def _parse_schaefer_label(lbl: str) -> tuple[str, str]:
    """Parse Schaefer label to extract hemisphere and network."""
    # '7Networks_LH_Vis_1' -> ('LH', 'Vis')
    parts = lbl.split('_')
    hemi = parts[1]
    network = parts[2]
    return hemi, network


def get_schaefer_labels(n_rois: int = 100, yeo_networks: int = 7) -> tuple:
    """Fetch Schaefer atlas and extract labels."""
    atl = fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=yeo_networks, resolution_mm=2)
    labels = atl["labels"][1:]  # skip 'Background'
    label_img = atl["maps"]
    region_to_network = {}
    region_to_hemi = {}
    for roi in labels:
        hemi, net = _parse_schaefer_label(roi)
        region_to_hemi[roi] = hemi
        region_to_network[roi] = net
    return labels, label_img, region_to_network, region_to_hemi


def get_anat_atlas() -> tuple:
    """Fetch Destrieux anatomical atlas."""
    a = fetch_atlas_destrieux_2009()
    return a["maps"], a["labels"]


def majority_overlap_labels(schaefer_img, sch_labels, anat_img, anat_labels) -> dict:
    """Map Schaefer regions to anatomical labels based on majority overlap."""
    anat_resamp = image.resample_to_img(anat_img, schaefer_img, interpolation='nearest',
                                       force_resample=True, copy_header=True)
    sch = image.get_data(schaefer_img).astype(int)
    anat = image.get_data(anat_resamp).astype(int)
    region_to_anat = {}

    for idx, roi in enumerate(sch_labels, start=1):
        mask = (sch == idx)
        if not np.any(mask):
            region_to_anat[roi] = "Unknown"
            continue
        vals, counts = np.unique(anat[mask], return_counts=True)
        nonzero = vals != 0
        vals, counts = vals[nonzero], counts[nonzero]
        if len(vals) == 0:
            region_to_anat[roi] = "Unknown"
        else:
            top = vals[np.argmax(counts)]
            try:
                region_to_anat[roi] = anat_labels[top]
            except Exception:
                region_to_anat[roi] = str(top)
    return region_to_anat


# Anatomical coarse mapping rules
COARSE_RULES = {
    'prefrontal': ['frontal', 'front', 'orbitofrontal', 'pars', 'rectus', 'rostral',
                   'caudal', 'frontomarg', 'frontopol', 'prefrontal'],
    'motor': ['precentral', 'postcentral', 'paracentral', 'central', 'rolandic',
              'supplementary', 'sma'],
    'insular': ['insula', 'opercular'],
    'parietal': ['pariet', 'supramarginal', 'precuneus', 'angular', 'intraparietal',
                 'superiorparietal', 'inferiorparietal'],
    'temporal': ['temporal', 'heschl', 'planum', 'fusiform', 'superiortemporal',
                 'middletemporal', 'inferiortemporal', 'temporo'],
    'occipital': ['occip', 'cuneus', 'lingual', 'calcarine', 'lateraloccipital'],
    'limbic': ['cingul', 'parahippo', 'entorhinal', 'subcallosal', 'paracingul', 'retrosplenial'],
    'cerebellum': ['cerebell'],
    'subcortical': ['thalam', 'caudate', 'putamen', 'pallidum', 'accumbens',
                    'ventraldc', 'striatum'],
    'brainstem': ['brainstem', 'brain stem', 'midbrain', 'pons', 'medulla'],
}

LOBE_CODE = {
    'prefrontal': 'PFC', 'motor': 'MOT', 'insular': 'INS', 'parietal': 'PAR',
    'temporal': 'TEM', 'occipital': 'OCC', 'limbic': 'LIM', 'cerebellum': 'CER',
    'subcortical': 'SUB', 'brainstem': 'BSM'
}

LOBE_ORDER = ['PFC','MOT','INS','PAR','TEM','OCC','LIM','CER','SUB','BSM']


def coarse_from_anat_label(anat_label: str) -> str:
    """Map anatomical label to coarse lobe code."""
    low = anat_label.lower()
    for group, keys in COARSE_RULES.items():
        if any(k in low for k in keys):
            return LOBE_CODE[group]
    # default fallback
    if 'frontal' in low or 'front' in low:
        return 'PFC'
    return 'PFC'


def clean_anat_label(lbl: str) -> str:
    """Clean anatomical label for display."""
    x = lbl
    x = x.replace('ctx-lh-', '').replace('ctx-rh-', '')
    x = x.replace('L ', '').replace('R ', '')
    x = x.replace('G_', 'G ').replace('S_', 'S ')
    x = x.replace('_', ' ')
    return x


def schaefer_coms(sch_img):
    """Calculate center-of-mass (MNI) per Schaefer parcel for A→P ordering."""
    data = image.get_data(sch_img).astype(int)
    affine = image.load_img(sch_img).affine
    coms = {}
    max_lab = int(data.max())
    for idx in range(1, max_lab + 1):
        mask = (data == idx)
        if not np.any(mask):
            coms[idx] = (np.nan, np.nan, np.nan)
            continue
        ijk = np.argwhere(mask)
        ijk_h = np.c_[ijk, np.ones(len(ijk))]
        xyz = ijk_h @ affine.T
        coms[idx] = tuple(xyz[:, :3].mean(axis=0))
    return coms


def build_node_metadata(grouping: str, n_rois: int = 100) -> tuple:
    """
    Build node metadata for the specified grouping.

    Args:
        grouping: 'yeo7', 'yeo17', or 'anat'
        n_rois: Number of ROIs in Schaefer parcellation

    Returns:
        Tuple of (node_meta, GROUP_ORDER, group_colors)
    """
    if grouping == 'yeo7':
        yeo_networks = 7
        GROUP_ORDER = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
        group_colors = {
            'Vis': '#781286',        # Purple
            'SomMot': '#4682B4',     # Steel Blue
            'DorsAttn': '#00760E',   # Green
            'SalVentAttn': '#C43AFA', # Violet
            'Limbic': '#DCDC00',     # Yellow
            'Cont': '#E69422',       # Orange
            'Default': '#CD3E4E'     # Red
        }
    elif grouping == 'yeo17':
        yeo_networks = 17
        # For Yeo17, we'll use a larger color palette
        GROUP_ORDER = ['VisCent', 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB',
                      'SalVentAttnA', 'SalVentAttnB', 'LimbicA', 'LimbicB',
                      'ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB', 'DefaultC', 'TempPar']
        # Use tab20 colormap for 17 networks
        palette = matplotlib.colormaps['tab20']
        group_colors = {net: palette(i/17.0) for i, net in enumerate(GROUP_ORDER)}
    else:  # anat
        yeo_networks = 7  # We'll still fetch yeo7 for the Schaefer parcellation
        GROUP_ORDER = LOBE_ORDER
        palette = matplotlib.colormaps['tab10']
        group_colors = {lo: palette(i/10.0) for i, lo in enumerate(LOBE_ORDER)}

    # Fetch Schaefer labels
    sch_labels, sch_img, region_to_network, region_to_hemi = get_schaefer_labels(n_rois=n_rois, yeo_networks=yeo_networks)

    # Build metadata
    node_meta = []

    if grouping == 'anat':
        # Load anatomical atlas and compute mappings
        anat_img, anat_labels = get_anat_atlas()
        region_to_anat = majority_overlap_labels(sch_img, sch_labels, anat_img, anat_labels)
        coms = schaefer_coms(sch_img)

        for sch_idx0, roi in enumerate(sch_labels):
            hemi = region_to_hemi[roi]
            fine = region_to_anat[roi]
            lobe = coarse_from_anat_label(fine)
            x, y, z = coms[sch_idx0+1]
            node_meta.append({
                "sch_index": sch_idx0,
                "hemi": hemi,
                "group": lobe,
                "fine": fine,
                "fine_clean": clean_anat_label(fine),
                "y_ap": y,
                "roi": roi
            })
    else:
        # Use network grouping (yeo7 or yeo17)
        for sch_idx0, roi in enumerate(sch_labels):
            hemi = region_to_hemi[roi]
            group = region_to_network[roi]
            node_meta.append({
                "sch_index": sch_idx0,
                "hemi": hemi,
                "group": group,
                "roi": roi
            })

    return node_meta, GROUP_ORDER, group_colors


def order_nodes_for_circle(node_meta: list, GROUP_ORDER: list, use_networks: bool) -> tuple:
    """
    Order nodes for circle visualization.

    Returns:
        Tuple of (ordered_nodes, all_groups, order_idx)
    """
    def order_within_hemi(hemi: str):
        ordered = []
        group_info = []

        for group_name in GROUP_ORDER:
            members = [m for m in node_meta if m["hemi"] == hemi and m["group"] == group_name]
            if members:
                if use_networks:
                    # Sort by ROI name for stability
                    members.sort(key=lambda d: d["roi"])
                else:
                    # Sort anterior->posterior using MNI Y
                    members.sort(key=lambda d: (-(d["y_ap"] if d["y_ap"]==d["y_ap"] else -1e9), d["fine_clean"]))

                start_idx = len(ordered)
                ordered.extend(members)
                end_idx = len(ordered) - 1
                group_info.append({
                    "group": group_name,
                    "hemi": hemi,
                    "start": start_idx,
                    "end": end_idx,
                    "count": len(members)
                })

        return ordered, group_info

    lh_nodes, lh_groups = order_within_hemi("LH")
    rh_nodes, rh_groups = order_within_hemi("RH")

    # Adjust RH group indices
    for group in rh_groups:
        group["start"] += len(lh_nodes)
        group["end"] += len(lh_nodes)

    final_nodes = lh_nodes + rh_nodes
    all_groups = lh_groups + rh_groups
    order_idx = [n["sch_index"] for n in final_nodes]

    return final_nodes, all_groups, order_idx


def visualize_full_matrix(
    matrix: np.ndarray,
    subject_id: str,
    save_path: Path,
    node_meta: list,
    GROUP_ORDER: list,
    title_suffix: str = "",
    percentile_clip: float = 5.0,
    grouping: str = "yeo7"
):
    """Create and save full connectivity matrix visualization with grouping."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create region to group mapping for reordering
    region_to_group_idx = {}
    for node in node_meta:
        region_to_group_idx[node["sch_index"]] = GROUP_ORDER.index(node["group"])

    # Sort regions by group
    sorted_indices = sorted(range(len(node_meta)), key=lambda i: (region_to_group_idx[i], i))

    # Reorder matrix
    matrix_plot = matrix[np.ix_(sorted_indices, sorted_indices)]

    # Calculate group boundaries
    boundaries = [0]
    current_group_idx = region_to_group_idx[sorted_indices[0]]
    for i, idx in enumerate(sorted_indices[1:], 1):
        if region_to_group_idx[idx] != current_group_idx:
            boundaries.append(i)
            current_group_idx = region_to_group_idx[idx]
    boundaries.append(len(sorted_indices))

    # Calculate percentile-based color limits
    vmin = np.percentile(matrix_plot, percentile_clip)
    vmax = np.percentile(matrix_plot, 100 - percentile_clip)

    # Create heatmap
    im = ax.imshow(matrix_plot, cmap=CMAP, aspect="equal", vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Connectivity Strength", rotation=270, labelpad=20)
    # Set consistent width for colorbar labels to prevent image shifting
    def format_colorbar(x, pos):
        return f'{x:>7.4f}'  # Right-aligned, 7 characters wide, 4 decimal places
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_colorbar))

    # Set titles and labels
    grouping_name = {'yeo7': 'Yeo 7 Networks', 'yeo17': 'Yeo 17 Networks', 'anat': 'Anatomical Lobes'}[grouping]
    ax.set_title(f"Connectivity Matrix - Subject {subject_id} ({grouping_name}){title_suffix}", fontsize=14, pad=20)
    ax.set_xlabel("Source Regions (providing information)", fontsize=12, labelpad=20)
    ax.set_ylabel("Target Regions (being predicted)", fontsize=12, labelpad=20)

    # Add network boundaries
    for boundary in boundaries[1:-1]:
        ax.axhline(y=boundary - 0.5, color="black", linewidth=2)
        ax.axvline(x=boundary - 0.5, color="black", linewidth=2)

    # Get the groups that actually exist in the data (in order)
    existing_groups = []
    seen_groups = set()
    for idx in sorted_indices:
        group = node_meta[idx]["group"]
        if group not in seen_groups:
            existing_groups.append(group)
            seen_groups.add(group)
    
    # Add group labels
    tick_positions = [(boundaries[i] + boundaries[i + 1]) / 2 - 0.5 for i in range(len(boundaries) - 1)]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(existing_groups, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(existing_groups, fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_coarse_matrix(matrix: np.ndarray, node_meta: list, GROUP_ORDER: list) -> np.ndarray:
    """Compute mean connectivity between coarse regions/groups."""
    n_groups = len(GROUP_ORDER)
    coarse_matrix = np.zeros((n_groups, n_groups))

    for i, group1 in enumerate(GROUP_ORDER):
        for j, group2 in enumerate(GROUP_ORDER):
            # Get indices for each group
            indices1 = [n["sch_index"] for n in node_meta if n["group"] == group1]
            indices2 = [n["sch_index"] for n in node_meta if n["group"] == group2]

            if indices1 and indices2:
                # Extract submatrix and compute mean
                submatrix = matrix[np.ix_(indices1, indices2)]
                if i == j:  # Same group - exclude diagonal
                    mask = ~np.eye(submatrix.shape[0], dtype=bool)
                    if mask.any():
                        coarse_matrix[i, j] = submatrix[mask].mean()
                else:
                    coarse_matrix[i, j] = submatrix.mean()

    return coarse_matrix


def visualize_coarse_matrix(
    coarse_matrix: np.ndarray,
    subject_id: str,
    save_path: Path,
    GROUP_ORDER: list,
    grouping: str = "yeo7",
    title_suffix: str = ""
):
    """Create and save coarse connectivity matrix visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(coarse_matrix, cmap=CMAP, aspect='equal')

    # Set ticks and labels
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_yticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER, rotation=45, ha='right')
    ax.set_yticklabels(GROUP_ORDER)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean Connectivity', rotation=270, labelpad=20)
    # Set consistent width for colorbar labels to prevent image shifting
    def format_colorbar(x, pos):
        return f'{x:>7.4f}'  # Right-aligned, 7 characters wide, 4 decimal places
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_colorbar))

    # Add text annotations
    for i in range(len(GROUP_ORDER)):
        for j in range(len(GROUP_ORDER)):
            text = ax.text(j, i, f'{coarse_matrix[i, j]:.3f}',
                          ha="center", va="center",
                          color="black" if abs(coarse_matrix[i, j]) < 0.3 else "white",
                          fontsize=8)

    # Set title
    grouping_name = {'yeo7': 'Yeo 7 Networks', 'yeo17': 'Yeo 17 Networks', 'anat': 'Anatomical Lobes'}[grouping]
    ax.set_title(f'Mean Connectivity - Subject {subject_id} ({grouping_name}){title_suffix}', fontsize=14, pad=20)
    ax.set_xlabel("Source Regions (providing information)", fontsize=12, labelpad=20)
    ax.set_ylabel("Target Regions (being predicted)", fontsize=12, labelpad=20)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_circle(
    matrix: np.ndarray,
    subject_id: str,
    save_path: Path,
    node_meta: list,
    GROUP_ORDER: list,
    group_colors: dict,
    grouping: str = "yeo7",
    title_suffix: str = ""
):
    """Create and save circle connectogram visualization."""
    use_networks = grouping in ['yeo7', 'yeo17']

    # Order nodes for circle visualization
    final_nodes, all_groups, order_idx = order_nodes_for_circle(node_meta, GROUP_ORDER, use_networks)

    # Calculate angles
    nL = sum(1 for n in final_nodes if n["hemi"] == "LH")
    nR = sum(1 for n in final_nodes if n["hemi"] == "RH")
    angles_L = np.linspace(5, 180, max(nL, 1), endpoint=False) if nL > 0 else []
    angles_R = np.linspace(354, 184, max(nR, 1), endpoint=False) if nR > 0 else []
    node_angles = np.concatenate([angles_L[:nL], angles_R[:nR]]) + 90

    # Create node names (mostly empty, label middle of each group)
    node_names = [''] * len(final_nodes)
    for group in all_groups:
        if group["count"] > 0:
            middle_idx = (group["start"] + group["end"]) // 2
            node_names[middle_idx] = f'{group["hemi"]} {group["group"]}'

    # Get node colors
    node_colors = [group_colors[n["group"]] for n in final_nodes]

    # Reorder connectivity matrix
    C_ord = matrix[np.ix_(order_idx, order_idx)]

    # Set title
    grouping_name = {'yeo7': 'Yeo 7 Networks', 'yeo17': 'Yeo 17 Networks', 'anat': 'Anatomical Lobes'}[grouping]
    title = f'Connectivity - Subject {subject_id} ({grouping_name}){title_suffix}'

    # Create plot
    fig, ax = plot_connectivity_circle(
        C_ord, node_names,
        n_lines=400,
        node_angles=node_angles,
        node_colors=node_colors,
        node_width=4,
        title=title,
        fontsize_names=13
    )

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_result_directories(result_root_dir: str, git_info: str, start_time: datetime) -> dict[str, Path]:
    """Create result directories for all visualization types."""
    script_name = get_script_name()
    timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    base_dir = Path(result_root_dir) / script_name / git_info / timestamp_str

    dirs = {
        'full_matrix': base_dir / 'full_matrix',
        'coarse_matrix': base_dir / 'coarse_matrix',
        'circle': base_dir / 'circle'
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def main():
    parser = argparse.ArgumentParser(description="Visualize connectivity matrices with multiple visualization types")
    parser.add_argument("file_pattern", type=str, help="File pattern to match (e.g., 'granger_causality.npy')")
    parser.add_argument("input_path", type=str, help="Path to directory containing subject subdirectories")
    parser.add_argument("result_root_dir", type=str, help="Root directory for saving results")
    parser.add_argument("--grouping", type=str, default="anat", choices=["yeo7", "yeo17", "anat"],
                       help="Grouping method: yeo7 (7 networks), yeo17 (17 networks), or anat (anatomical lobes)")
    parser.add_argument("--row-normalize", action="store_true", help="Apply row normalization")
    parser.add_argument("--column-mean-subtract", action="store_true", help="Subtract column means")
    parser.add_argument("--group-mean-subtract", action="store_true", help="Subtract group mean across subjects")
    parser.add_argument("--percentile-clip", type=float, default=5.0,
                       help="Percentile for color scale clipping (default: 5.0)")

    args = parser.parse_args()

    start_time = datetime.now()
    print(f"Connectivity visualization started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get git information
    git_info = get_git_info()
    print(f"Git info: {git_info}")

    # Validate paths
    assert Path(args.input_path).exists(), f"Input path not found: {args.input_path}"

    # Create result directories
    result_dirs = create_result_directories(args.result_root_dir, git_info, start_time)
    print(f"Results will be saved to: {result_dirs['full_matrix'].parent}")

    # Find subject files
    subject_files = find_subject_files(args.input_path, args.file_pattern)

    # Build node metadata for the specified grouping
    print(f"Building node metadata for {args.grouping} grouping...")
    node_meta, GROUP_ORDER, group_colors = build_node_metadata(args.grouping)

    # Load all matrices
    print("Loading connectivity matrices...")
    matrices = {}
    for subject_id, file_path in subject_files.items():
        matrix = np.load(file_path).astype(float)
        assert matrix.ndim == 2, f"Matrix must be 2D, got shape {matrix.shape}"
        assert matrix.shape[0] == matrix.shape[1], f"Matrix must be square, got shape {matrix.shape}"

        matrices[subject_id] = matrix
        print(f"  Loaded {subject_id}: {matrix.shape}")

    assert len(matrices) > 0, "No matrices could be loaded"

    # Apply normalizations
    processed_matrices = matrices.copy()
    processing_steps = []

    # Row normalization
    if args.row_normalize:
        print("Applying row normalization...")
        for subject_id in processed_matrices:
            processed_matrices[subject_id] = apply_row_normalization(processed_matrices[subject_id])
        processing_steps.append("row_normalize")

    # Column mean subtraction
    if args.column_mean_subtract:
        print("Applying column mean subtraction...")
        for subject_id in processed_matrices:
            processed_matrices[subject_id] = apply_column_mean_subtraction(processed_matrices[subject_id])
        processing_steps.append("column_mean_subtract")

    # Group mean subtraction
    if args.group_mean_subtract:
        print("Applying group mean subtraction...")
        processed_matrices = apply_group_mean_subtraction(processed_matrices)
        processing_steps.append("group_mean_subtract")

    # Create title suffix based on processing
    title_suffix = ""
    if processing_steps:
        title_suffix = f" ({', '.join(processing_steps)})"

    # Generate all three visualizations for each subject
    print("\nCreating visualizations...")

    for subject_id, matrix in processed_matrices.items():
        print(f"  Processing {subject_id}...")

        # Full matrix visualization
        save_path = result_dirs['full_matrix'] / f"{subject_id}.jpg"
        visualize_full_matrix(
            matrix, subject_id, save_path, node_meta, GROUP_ORDER,
            title_suffix, args.percentile_clip, args.grouping
        )

        # Coarse matrix visualization
        coarse_matrix = compute_coarse_matrix(matrix, node_meta, GROUP_ORDER)
        save_path = result_dirs['coarse_matrix'] / f"{subject_id}.jpg"
        visualize_coarse_matrix(
            coarse_matrix, subject_id, save_path, GROUP_ORDER,
            args.grouping, title_suffix
        )

        # Circle visualization
        save_path = result_dirs['circle'] / f"{subject_id}.svg"
        visualize_circle(
            matrix, subject_id, save_path, node_meta, GROUP_ORDER,
            group_colors, args.grouping, title_suffix
        )

        print(f"    ✓ All visualizations saved")

    # Save metadata
    metadata = {
        "file_pattern": args.file_pattern,
        "input_path": args.input_path,
        "n_subjects": len(processed_matrices),
        "subject_ids": list(processed_matrices.keys()),
        "matrix_shape": list(next(iter(processed_matrices.values())).shape),
        "processing_steps": processing_steps,
        "grouping": args.grouping,
        "group_names": GROUP_ORDER,
        "n_groups": len(GROUP_ORDER),
        "visualization_timestamp": start_time.isoformat(),
        "git_info": git_info,
        "settings": {
            "row_normalize": args.row_normalize,
            "column_mean_subtract": args.column_mean_subtract,
            "group_mean_subtract": args.group_mean_subtract,
            "percentile_clip": args.percentile_clip,
        },
    }

    metadata_file = result_dirs['full_matrix'].parent / "visualization_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    end_time = datetime.now()
    print(f"\n{'=' * 60}")
    print("VISUALIZATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"File pattern: {args.file_pattern}")
    print(f"Subjects processed: {len(processed_matrices)}")
    print(f"Processing applied: {processing_steps if processing_steps else ['none']}")
    print(f"Grouping method: {args.grouping}")
    print(f"Number of groups: {len(GROUP_ORDER)}")
    print(f"Total time: {end_time - start_time}")
    print(f"Results saved to:")
    print(f"  - Full matrices: {result_dirs['full_matrix']}")
    print(f"  - Coarse matrices: {result_dirs['coarse_matrix']}")
    print(f"  - Circle diagrams: {result_dirs['circle']}")


if __name__ == "__main__":
    main()
