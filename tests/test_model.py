"""
Unit tests for Model Training, Prediction, and Threshold Tuning Module (Phase 7).
"""

import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.model.prepare_training_data import prepare_training_data
from src.model.train import train_matcher, get_feature_importance, DEFAULT_FEATURE_COLUMNS
from src.model.predict import predict_match_probabilities, assemble_entity_predictions
from src.model.threshold_tuning import sweep_thresholds, find_best_threshold


@pytest.fixture
def synthetic_labeled_features():
    """Builds a small synthetic labeled features dataset for model testing."""
    records = []
    # 5 S1 entities in train, 2 in val
    # S1_01: 1 positive, 2 exact negatives, 5 fuzzy negatives
    records.append({'source1_entity_id': 'S1_01', 'candidate_entity_id': 'S2_01', 'is_true_match': 1, 'is_exact_tier': 1, 'f1': 0.95, 'f2': 0.90})
    records.append({'source1_entity_id': 'S1_01', 'candidate_entity_id': 'S2_02', 'is_true_match': 0, 'is_exact_tier': 1, 'f1': 0.70, 'f2': 0.60})
    records.append({'source1_entity_id': 'S1_01', 'candidate_entity_id': 'S2_03', 'is_true_match': 0, 'is_exact_tier': 1, 'f1': 0.65, 'f2': 0.50})
    for i in range(4, 9):
        records.append({'source1_entity_id': 'S1_01', 'candidate_entity_id': f'S2_{i:02d}', 'is_true_match': 0, 'is_exact_tier': 0, 'f1': 0.20, 'f2': 0.10})

    # S1_02: 1 positive, 1 exact negative, 10 fuzzy negatives
    records.append({'source1_entity_id': 'S1_02', 'candidate_entity_id': 'S2_10', 'is_true_match': 1, 'is_exact_tier': 1, 'f1': 0.90, 'f2': 0.85})
    records.append({'source1_entity_id': 'S1_02', 'candidate_entity_id': 'S2_11', 'is_true_match': 0, 'is_exact_tier': 1, 'f1': 0.55, 'f2': 0.50})
    for i in range(12, 22):
        records.append({'source1_entity_id': 'S1_02', 'candidate_entity_id': f'S2_{i:02d}', 'is_true_match': 0, 'is_exact_tier': 0, 'f1': 0.15, 'f2': 0.05})

    # S1_VAL_01: validation entity
    records.append({'source1_entity_id': 'S1_VAL_01', 'candidate_entity_id': 'S2_VAL_01', 'is_true_match': 1, 'is_exact_tier': 1, 'f1': 0.92, 'f2': 0.88})
    records.append({'source1_entity_id': 'S1_VAL_01', 'candidate_entity_id': 'S2_VAL_02', 'is_true_match': 0, 'is_exact_tier': 0, 'f1': 0.30, 'f2': 0.20})

    # S1_VAL_02: validation singleton (no positive)
    records.append({'source1_entity_id': 'S1_VAL_02', 'candidate_entity_id': 'S2_VAL_03', 'is_true_match': 0, 'is_exact_tier': 0, 'f1': 0.10, 'f2': 0.05})

    return pd.DataFrame(records)


def test_prepare_training_data_tiered_sampling(synthetic_labeled_features):
    train_ids = {'S1_01', 'S1_02'}
    
    # 2 positives total in train (S2_01, S2_10)
    # 3 exact negatives (S2_02, S2_03, S2_11)
    # 15 fuzzy negatives
    sampled_df = prepare_training_data(
        synthetic_labeled_features,
        train_ids=train_ids,
        negative_ratio=5.0,  # Target 2 * 5 = 10 negatives
        random_seed=42
    )

    # 1. Positives check: all 2 positives must be present
    positives = sampled_df[sampled_df['is_true_match'] == 1]
    assert len(positives) == 2
    assert set(positives['candidate_entity_id']) == {'S2_01', 'S2_10'}

    # 2. Exact-tier negatives: all 3 exact negatives must be retained
    exact_negatives = sampled_df[(sampled_df['is_true_match'] == 0) & (sampled_df['is_exact_tier'] == 1)]
    assert len(exact_negatives) == 3
    assert set(exact_negatives['candidate_entity_id']) == {'S2_02', 'S2_03', 'S2_11'}

    # 3. Fuzzy negatives: 10 target - 3 exact = 7 fuzzy negatives
    fuzzy_negatives = sampled_df[(sampled_df['is_true_match'] == 0) & (sampled_df['is_exact_tier'] == 0)]
    assert len(fuzzy_negatives) == 7

    # 4. Total samples = 2 + 3 + 7 = 12
    assert len(sampled_df) == 12

    # 5. Strictly no validation entities present
    assert 'S1_VAL_01' not in sampled_df['source1_entity_id'].values
    assert 'S1_VAL_02' not in sampled_df['source1_entity_id'].values


def test_assemble_entity_predictions_modes():
    scored_df = pd.DataFrame([
        {'source1_entity_id': 'S1_A', 'candidate_entity_id': 'C1', 'match_probability': 0.85},
        {'source1_entity_id': 'S1_A', 'candidate_entity_id': 'C2', 'match_probability': 0.65},
        {'source1_entity_id': 'S1_A', 'candidate_entity_id': 'C3', 'match_probability': 0.20},
        {'source1_entity_id': 'S1_B', 'candidate_entity_id': 'C4', 'match_probability': 0.40},
    ])
    all_s1 = ['S1_A', 'S1_B', 'S1_C']  # S1_C has no candidate pairs

    # 1. Mode 'multi' at threshold 0.50
    preds_multi = assemble_entity_predictions(scored_df, threshold=0.50, mode="multi", all_s1_entities=all_s1)
    assert len(preds_multi) == 3
    pred_map_multi = dict(zip(preds_multi['source1_entity_id'], preds_multi['matched_entity_ids']))
    assert pred_map_multi['S1_A'] == 'C1,C2'
    assert pred_map_multi['S1_B'] == ''
    assert pred_map_multi['S1_C'] == ''

    # 2. Mode 'top1' at threshold 0.50
    preds_top1 = assemble_entity_predictions(scored_df, threshold=0.50, mode="top1", all_s1_entities=all_s1)
    assert len(preds_top1) == 3
    pred_map_top1 = dict(zip(preds_top1['source1_entity_id'], preds_top1['matched_entity_ids']))
    assert pred_map_top1['S1_A'] == 'C1'  # Only highest probability candidate
    assert pred_map_top1['S1_B'] == ''
    assert pred_map_top1['S1_C'] == ''

    # 3. Mode 'multi' at threshold 0.90 (no candidates clear threshold)
    preds_high = assemble_entity_predictions(scored_df, threshold=0.90, mode="multi", all_s1_entities=all_s1)
    pred_map_high = dict(zip(preds_high['source1_entity_id'], preds_high['matched_entity_ids']))
    assert pred_map_high['S1_A'] == ''
    assert pred_map_high['S1_B'] == ''
    assert pred_map_high['S1_C'] == ''


def test_train_matcher_and_predict_probabilities(synthetic_labeled_features, tmp_path):
    train_ids = {'S1_01', 'S1_02'}
    train_df = prepare_training_data(synthetic_labeled_features, train_ids=train_ids, negative_ratio=5.0)

    feature_cols = ['f1', 'f2']
    model_path = str(tmp_path / "test_model.txt")

    # Train model
    model = train_matcher(
        training_df=train_df,
        feature_columns=feature_cols,
        target_col='is_true_match',
        model_params={'n_estimators': 10, 'num_leaves': 7, 'min_child_samples': 1},
        model_output_path=model_path
    )

    # Check model artifact saved
    assert (tmp_path / "test_model.txt").exists()

    # Predict match probabilities
    scored_df = predict_match_probabilities(model, synthetic_labeled_features, feature_columns=feature_cols)
    assert 'match_probability' in scored_df.columns
    assert len(scored_df) == len(synthetic_labeled_features)
    assert scored_df['match_probability'].min() >= 0.0
    assert scored_df['match_probability'].max() <= 1.0

    # Feature importance
    df_imp = get_feature_importance(model, feature_cols)
    assert len(df_imp) == 2
    assert 'importance' in df_imp.columns


def test_sweep_thresholds_end_to_end(synthetic_labeled_features):
    train_ids = {'S1_01', 'S1_02'}
    train_df = prepare_training_data(synthetic_labeled_features, train_ids=train_ids, negative_ratio=5.0)

    feature_cols = ['f1', 'f2']
    model = train_matcher(
        training_df=train_df,
        feature_columns=feature_cols,
        target_col='is_true_match',
        model_params={'n_estimators': 15, 'num_leaves': 7, 'min_child_samples': 1}
    )

    val_features_df = synthetic_labeled_features[synthetic_labeled_features['source1_entity_id'].str.contains('VAL')].copy()
    ground_truth_df = pd.DataFrame([
        {'source1_entity_id': 'S1_VAL_01', 'matched_entity_ids': 'S2_VAL_01'},
        {'source1_entity_id': 'S1_VAL_02', 'matched_entity_ids': ''},
    ])

    sweep_multi = sweep_thresholds(
        model=model,
        val_features_df=val_features_df,
        ground_truth_df=ground_truth_df,
        feature_columns=feature_cols,
        thresholds=[0.3, 0.5, 0.7],
        mode="multi"
    )

    assert len(sweep_multi) == 3
    assert 'macro_f_beta' in sweep_multi.columns
    assert 'macro_precision' in sweep_multi.columns
    assert 'macro_recall' in sweep_multi.columns

    best_config = find_best_threshold(sweep_multi)
    assert 'threshold' in best_config
    assert 'macro_f_beta' in best_config
