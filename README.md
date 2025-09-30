# Setup

```bash
conda create --name fmri python=3.12
conda activate fmri
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4 # use the right torch version for your machine
pip install numpy nibabel nilearn pydantic tqdm matplotlib mne mne-connectivity seaborn
```

# Train

```bash
# Single group level model
python src/group_level_training.py config/group_train_config.json

# One model per individual
python src/subject_level_training.py config/subject_train_config.json
```

# Extract connectivity matrices

```bash
# For baselines (Granger, Pearson, partial)
python src/evaluate_baselines.py \
       csv/subject_groups.csv \
       /path/to/HCP_Young_Adult_2025/data/ \
       atlas/Schaefer2018_100Parcels_17Networks_order.dlabel.nii \
       results/

# For group-level trained models
python src/evaluate_models.py \
       csv/subject_groups.csv \
       /path/to/HCP_Young_Adult_2025/data/ \
       atlas/Schaefer2018_100Parcels_17Networks_order.dlabel.nii \
       /path/to/group/level/model/ \
       results/ \
       --model-type group-level

# For subject-level trained models
python src/evaluate_models.py \
       csv/subject_groups.csv \
       /path/to/HCP_Young_Adult_2025/data/ atlas/Schaefer2018_100Parcels_17Networks_order.dlabel.nii \
       /path/to/subject/level/model/ \
       results/ \
       --model-type subject-level
```

# Run behavioral benchmark

You need to download the csv file behavioral data of the HCP-YA S1200 release with all unrestricted columns. Here it is named `csv/HCP_YA_subjects_2025_09_12_01_57_30.csv`.

```bash
# connectivity_matrix_pattern can be mean_attention, pearson_correlation, partial_correlation, or granger_causality
python src/behavioral_connectivity.py \
csv/subject_groups.csv \
/path/to/connectivity/matrices/ \
connectivity_matrix_pattern.npy \
csv/HCP_YA_subjects_2025_09_12_01_57_30.csv \
csv/behaviour_targets.csv \
results/
```

# Run fingerprinting benchmark

```bash
# connectivity_matrix_pattern can be mean_attention, pearson_correlation, partial_correlation, or granger_causality
# /path/to/test/set/matrices/ and /path/to/retest/set/matrices/ can be the same path
python src/fingerprint_connectivity.py \
csv/subject_ids_test-retest.csv \
connectivity_matrix_pattern.npy \
/path/to/test/set/matrices/ \
/path/to/retest/set/matrices/ \
results/ \
--method pearson --z-normalize --comparison-mode test_to_retest
```

# Visualize

Scripts to visualize the results are:
- `src/compare_behavioral_results.py`
- `src/compare_fingerprint_results.py`
- `src/compare_prediction_r2_distributions.py`
- `src/plot_training_curves.py`
- `src/visualize_connectivity.py`
