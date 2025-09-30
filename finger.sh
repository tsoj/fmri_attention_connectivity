python src/fingerprint_connectivity.py csv/subject_ids_test-retest.csv \
granger_causality.npy \
/mnt/second_disk/results_base/evaluate_baselines/4b9154a/2025-09-27_04-42-20 \
/mnt/second_disk/results_base/evaluate_baselines/4b9154a/2025-09-27_04-42-20 \
results_fake_finger_others_all_sessions/ \
--method pearson --z-normalize --comparison-mode across_all_sessions

python src/fingerprint_connectivity.py csv/subject_ids_test-retest.csv \
granger_causality.npy \
/mnt/second_disk/results_base/evaluate_baselines/4b9154a/2025-09-27_04-42-20 \
/mnt/second_disk/results_base/evaluate_baselines/4b9154a/2025-09-27_04-42-20 \
results_fake_finger_others_test_to_retest/ \
--method pearson --z-normalize --comparison-mode test_to_retest

python src/fingerprint_connectivity.py csv/subject_ids_test-retest.csv \
pearson_correlation.npy \
/mnt/second_disk/results_base/evaluate_baselines/4b9154a/2025-09-27_04-42-20 \
/mnt/second_disk/results_base/evaluate_baselines/4b9154a/2025-09-27_04-42-20 \
results_fake_finger_others_all_sessions/ \
--method pearson --z-normalize --comparison-mode across_all_sessions

python src/fingerprint_connectivity.py csv/subject_ids_test-retest.csv \
pearson_correlation.npy \
/mnt/second_disk/results_base/evaluate_baselines/4b9154a/2025-09-27_04-42-20 \
/mnt/second_disk/results_base/evaluate_baselines/4b9154a/2025-09-27_04-42-20 \
results_fake_finger_others_test_to_retest/ \
--method pearson --z-normalize --comparison-mode test_to_retest

python src/fingerprint_connectivity.py csv/subject_ids_test-retest.csv \
partial_correlation.npy \
/mnt/second_disk/results_base/evaluate_baselines/4b9154a/2025-09-27_04-42-20 \
/mnt/second_disk/results_base/evaluate_baselines/4b9154a/2025-09-27_04-42-20 \
results_fake_finger_others_all_sessions/ \
--method pearson --z-normalize --comparison-mode across_all_sessions

python src/fingerprint_connectivity.py csv/subject_ids_test-retest.csv \
partial_correlation.npy \
/mnt/second_disk/results_base/evaluate_baselines/4b9154a/2025-09-27_04-42-20 \
/mnt/second_disk/results_base/evaluate_baselines/4b9154a/2025-09-27_04-42-20 \
results_fake_finger_others_test_to_retest/ \
--method pearson --z-normalize --comparison-mode test_to_retest

# python src/compare_fingerprint_results.py \
#     --subject-attention results_final/fingerprinting/test_to_retest/results_finger_subject_level_test_to_retest/fingerprint_connectivity/cf910b8_unstaged/2025-09-30_01-59-01 \
#     --group-attention results_final/fingerprinting/test_to_retest/results_finger_group_level_test_to_retest/fingerprint_connectivity/cf910b8_unstaged/2025-09-30_01-58-14 \
#     --group-subject-attention results_final/fingerprinting/test_to_retest/results_finger_group_subject_test_to_retest/fingerprint_connectivity/cf910b8_unstaged/2025-09-30_02-00-57 \
#     --other results_final/fingerprinting/test_to_retest/results_finger_others_test_to_retest/fingerprint_connectivity/cf910b8_unstaged/2025-09-30_02-04-17 \
#     --other results_final/fingerprinting/test_to_retest/results_finger_others_test_to_retest/fingerprint_connectivity/cf910b8_unstaged/2025-09-30_02-04-36 \
#     --other results_final/fingerprinting/test_to_retest/results_finger_others_test_to_retest/fingerprint_connectivity/cf910b8_unstaged/2025-09-30_02-04-55 \
#     --output-dir results_finger_test_to_retest


# python src/compare_fingerprint_results.py \
#     --subject-attention results_final/fingerprinting/all_sessions/results_finger_subject_level_all_sessions/fingerprint_connectivity/cf910b8_unstaged/2025-09-30_01-59-25 \
#     --group-attention results_final/fingerprinting/all_sessions/results_finger_group_level_all_sessions/fingerprint_connectivity/cf910b8_unstaged/2025-09-30_01-57-42 \
#     --group-subject-attention results_final/fingerprinting/all_sessions/results_finger_group_subject_all_sessions/fingerprint_connectivity/cf910b8_unstaged/2025-09-30_02-01-21 \
#     --other results_final/fingerprinting/all_sessions/results_finger_others_all_sessions/fingerprint_connectivity/cf910b8_unstaged/2025-09-30_02-04-07 \
#     --other results_final/fingerprinting/all_sessions/results_finger_others_all_sessions/fingerprint_connectivity/cf910b8_unstaged/2025-09-30_02-04-26 \
#     --other results_final/fingerprinting/all_sessions/results_finger_others_all_sessions/fingerprint_connectivity/cf910b8_unstaged/2025-09-30_02-04-45 \
#     --output-dir results_final/fingerprinting/all_sessions/results_finger_all_sessions
