import pytest
import pandas as pd
import numpy as np

from src.pipeline.scoring import compute_entity_f_beta, compute_macro_f_beta, parse_id_set


def test_perfect_match():
    true_ids = {'S2-100', 'S3-200'}
    pred_ids = {'S2-100', 'S3-200'}
    res = compute_entity_f_beta(true_ids, pred_ids, beta=0.5)
    assert res['precision'] == 1.0
    assert res['recall'] == 1.0
    assert res['f_beta'] == 1.0


def test_correct_singleton():
    res = compute_entity_f_beta(set(), set(), beta=0.5)
    assert res['precision'] == 1.0
    assert res['recall'] == 1.0
    assert res['f_beta'] == 1.0


def test_false_positive_on_singleton():
    # True singleton, but predicted false match -> explicit penalty: 0.0
    res = compute_entity_f_beta(set(), {'S2-999'}, beta=0.5)
    assert res['precision'] == 0.0
    assert res['recall'] == 0.0
    assert res['f_beta'] == 0.0


def test_false_negative_on_non_singleton():
    # True matches exist, but predicted empty
    res = compute_entity_f_beta({'S2-100'}, set(), beta=0.5)
    assert res['precision'] == 0.0
    assert res['recall'] == 0.0
    assert res['f_beta'] == 0.0


def test_hand_calculated_partial_match():
    true_ids = {'S2-1', 'S2-2', 'S3-1'}
    pred_ids = {'S2-1', 'S2-2', 'S3-99'}
    # TP = 2, Pred = 3 (P = 2/3), True = 3 (R = 2/3)
    # F0.5 = 1.25 * (2/3) * (2/3) / (0.25 * 2/3 + 2/3) = (5/9) / (5/6 * 2/3) = 2/3
    res = compute_entity_f_beta(true_ids, pred_ids, beta=0.5)
    assert pytest.approx(res['precision'], 1e-6) == 2.0 / 3.0
    assert pytest.approx(res['recall'], 1e-6) == 2.0 / 3.0
    assert pytest.approx(res['f_beta'], 1e-6) == 2.0 / 3.0


def test_precision_weighting_at_beta_05():
    # Case 1: High Precision (1.0), Low Recall (0.5) -> F1 = 2/3
    # F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 = 0.833333
    true_high_p = {'S2-1', 'S2-2', 'S2-3', 'S2-4'}
    pred_high_p = {'S2-1', 'S2-2'}
    res_high_p = compute_entity_f_beta(true_high_p, pred_high_p, beta=0.5)
    
    assert pytest.approx(res_high_p['precision'], 1e-6) == 1.0
    assert pytest.approx(res_high_p['recall'], 1e-6) == 0.5
    assert pytest.approx(res_high_p['f_beta'], 1e-6) == 5.0 / 6.0

    # Case 2: Low Precision (0.5), High Recall (1.0) -> F1 = 2/3
    # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 = 0.555556
    true_low_p = {'S2-1', 'S2-2'}
    pred_low_p = {'S2-1', 'S2-2', 'S2-3', 'S2-4'}
    res_low_p = compute_entity_f_beta(true_low_p, pred_low_p, beta=0.5)

    assert pytest.approx(res_low_p['precision'], 1e-6) == 0.5
    assert pytest.approx(res_low_p['recall'], 1e-6) == 1.0
    assert pytest.approx(res_low_p['f_beta'], 1e-6) == 5.0 / 9.0

    # Confirmation: High precision scores significantly higher at beta=0.5 (0.833 vs 0.556)
    assert res_high_p['f_beta'] > res_low_p['f_beta']


def test_macro_f_beta_aggregation():
    # 4 synthetic entities:
    # Entity 1: Perfect match (F0.5 = 1.0)
    # Entity 2: Correct singleton (F0.5 = 1.0)
    # Entity 3: High Precision (F0.5 = 5/6 = 0.833333)
    # Entity 4: False positive on singleton (F0.5 = 0.0)
    # Expected Macro F0.5 = (1.0 + 1.0 + 5/6 + 0.0) / 4 = (17/6) / 4 = 17/24 = 0.708333
    gt_data = {
        'source1_entity_id': ['S1-1', 'S1-2', 'S1-3', 'S1-4'],
        'matched_entity_ids': ['S2-1,S3-1', '', 'S2-10,S2-11,S2-12,S2-13', '']
    }
    pred_data = {
        'source1_entity_id': ['S1-1', 'S1-2', 'S1-3', 'S1-4'],
        'matched_entity_ids': ['S2-1,S3-1', '', 'S2-10,S2-11', 'S2-99']
    }
    df_gt = pd.DataFrame(gt_data)
    df_pred = pd.DataFrame(pred_data)

    res = compute_macro_f_beta(df_pred, df_gt, beta=0.5)
    
    assert res['evaluated_entities_count'] == 4
    assert pytest.approx(res['macro_f_beta'], 1e-5) == 17.0 / 24.0

    # Test filtering subset
    res_subset = compute_macro_f_beta(df_pred, df_gt, beta=0.5, entity_ids_filter=['S1-1', 'S1-2'])
    assert res_subset['evaluated_entities_count'] == 2
    assert res_subset['macro_f_beta'] == 1.0


def test_missing_entities_raise_error():
    df_gt = pd.DataFrame({'source1_entity_id': ['S1-1'], 'matched_entity_ids': ['S2-1']})
    df_pred = pd.DataFrame({'source1_entity_id': ['S1-2'], 'matched_entity_ids': ['S2-1']})
    
    with pytest.raises(KeyError):
        compute_macro_f_beta(df_pred, df_gt, entity_ids_filter=['S1-1'])
