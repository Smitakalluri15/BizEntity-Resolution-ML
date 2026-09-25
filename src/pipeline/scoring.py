"""
Business Entity Resolution — Macro F0.5 Scoring Module
Amazon ML Challenge 2026

══════════════════════════════════════════════════════════════════════════════
MACRO F0.5 METRIC SPECIFICATION & EDGE-CASE SEMANTICS
══════════════════════════════════════════════════════════════════════════════

1. Per-Entity Precision, Recall, and F-Beta:
   For each Source-1 (S1) entity i:
     - True set: T_i = {s in S2 U S3 | s matches S1_i}
     - Predicted set: P_i = {s in S2 U S3 | s predicted to match S1_i}

   - Case A (True Singleton Correctly Predicted Empty):
     T_i = {} and P_i = {}
     Precision = 1.0, Recall = 1.0, F_beta = 1.0
     (Explicit Challenge Rule: Correct empty prediction on a true singleton scores 1.0)

   - Case B (False Positive on True Singleton):
     T_i = {} and P_i != {}
     Precision = 0.0, Recall = 0.0, F_beta = 0.0
     (Explicit Challenge Rule: Any false positive predicted for a singleton incurs a hard 0.0 score)

   - Case C (False Negative / Empty Prediction on True Match):
     T_i != {} and P_i = {}
     Precision = 0.0, Recall = 0.0, F_beta = 0.0
     (Predicting no match when matches exist fails recall completely)

   - Case D (Standard Overlap on Non-Empty Sets):
     T_i != {} and P_i != {}
     Precision = |T_i ∩ P_i| / |P_i|
     Recall    = |T_i ∩ P_i| / |T_i|
     If Precision + Recall == 0 (zero true positives):
       F_beta = 0.0
     Else:
       F_beta = (1 + beta^2) * Precision * Recall / (beta^2 * Precision + Recall)
       For beta = 0.5: F_0.5 = 1.25 * P * R / (0.25 * P + R) = 5 * P * R / (P + 4 * R)

2. Macro Aggregation:
   - Macro Precision = (1 / N) * sum(Precision_i)
   - Macro Recall    = (1 / N) * sum(Recall_i)
   - Macro F_beta    = (1 / N) * sum(F_beta_i)
   where N is the total number of evaluated S1 entities (simple arithmetic mean).
"""

from typing import Set, Dict, Any, Optional, Iterable, Union
import pandas as pd
import numpy as np


def compute_entity_f_beta(
    true_ids: Union[Set[str], Iterable[str]],
    predicted_ids: Union[Set[str], Iterable[str]],
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F-beta score for a single Source-1 entity.
    
    Returns:
        dict: {"precision": float, "recall": float, "f_beta": float}
    """
    t_set = set(true_ids) if true_ids else set()
    p_set = set(predicted_ids) if predicted_ids else set()

    # Clean out empty strings if any
    t_set = {x for x in t_set if str(x).strip()}
    p_set = {x for x in p_set if str(x).strip()}

    # Case A: Correct empty singleton
    if len(t_set) == 0 and len(p_set) == 0:
        return {"precision": 1.0, "recall": 1.0, "f_beta": 1.0}

    # Case B: False positive on singleton
    if len(t_set) == 0 and len(p_set) > 0:
        return {"precision": 0.0, "recall": 0.0, "f_beta": 0.0}

    # Case C: Missed matches (predicted empty when true non-empty)
    if len(t_set) > 0 and len(p_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f_beta": 0.0}

    # Case D: Both non-empty
    correct = len(t_set.intersection(p_set))
    precision = float(correct) / float(len(p_set))
    recall = float(correct) / float(len(t_set))

    if precision == 0.0 or recall == 0.0:
        f_beta = 0.0
    else:
        beta_sq = beta ** 2
        f_beta = (1.0 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

    return {"precision": precision, "recall": recall, "f_beta": f_beta}


def parse_id_set(raw_val: Any) -> Set[str]:
    """
    Safely parses comma-separated string, iterable, or NaN/None into a Set[str].
    """
    if raw_val is None or pd.isna(raw_val):
        return set()
    if isinstance(raw_val, (set, list, tuple)):
        return {str(x).strip() for x in raw_val if str(x).strip()}
    s = str(raw_val).strip()
    if not s or s.lower() in ('nan', 'none', '<null>', 'null'):
        return set()
    return {x.strip() for x in s.split(',') if x.strip()}


def compute_macro_f_beta(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    beta: float = 0.5,
    entity_ids_filter: Optional[Iterable[str]] = None
) -> Dict[str, Any]:
    """
    Computes Macro Precision, Macro Recall, and Macro F_beta across all evaluated S1 entities.
    
    Args:
        predictions_df: DataFrame containing ['source1_entity_id', 'matched_entity_ids']
        ground_truth_df: DataFrame containing ['source1_entity_id', 'matched_entity_ids']
        beta: Weighting factor (default 0.5 for F0.5)
        entity_ids_filter: Optional subset of S1 entity IDs to evaluate (e.g. validation split)
        
    Returns:
        dict: {
            "macro_precision": float,
            "macro_recall": float,
            "macro_f_beta": float,
            "per_entity_scores": pd.DataFrame,
            "evaluated_entities_count": int
        }
    """
    pred_col = 'matched_entity_ids' if 'matched_entity_ids' in predictions_df.columns else predictions_df.columns[1]
    gt_col = 'matched_entity_ids' if 'matched_entity_ids' in ground_truth_df.columns else ground_truth_df.columns[1]

    pred_s1_col = 'source1_entity_id' if 'source1_entity_id' in predictions_df.columns else predictions_df.columns[0]
    gt_s1_col = 'source1_entity_id' if 'source1_entity_id' in ground_truth_df.columns else ground_truth_df.columns[0]

    # Map S1 IDs to sets (fast zip iteration)
    pred_map = {s1: parse_id_set(raw) for s1, raw in zip(predictions_df[pred_s1_col], predictions_df[pred_col])}
    gt_map = {s1: parse_id_set(raw) for s1, raw in zip(ground_truth_df[gt_s1_col], ground_truth_df[gt_col])}

    # Determine evaluation entities
    if entity_ids_filter is not None:
        eval_ids = [str(x).strip() for x in entity_ids_filter if str(x).strip()]
    else:
        eval_ids = list(gt_map.keys())

    if not eval_ids:
        raise ValueError("No S1 entities provided for evaluation.")

    # Validate that all evaluation entities exist in ground truth and predictions
    missing_in_gt = [eid for eid in eval_ids if eid not in gt_map]
    if missing_in_gt:
        raise KeyError(f"Evaluation S1 entities missing from ground truth (first 5): {missing_in_gt[:5]}")

    missing_in_pred = [eid for eid in eval_ids if eid not in pred_map]
    if missing_in_pred:
        raise KeyError(f"Evaluation S1 entities missing from predictions (first 5): {missing_in_pred[:5]}")

    records = []
    for s1_id in eval_ids:
        t_set = gt_map[s1_id]
        p_set = pred_map[s1_id]
        
        scores = compute_entity_f_beta(t_set, p_set, beta=beta)
        correct_count = len(t_set.intersection(p_set))
        
        records.append({
            'source1_entity_id': s1_id,
            'true_count': len(t_set),
            'pred_count': len(p_set),
            'correct_count': correct_count,
            'precision': scores['precision'],
            'recall': scores['recall'],
            'f_beta': scores['f_beta']
        })

    df_per_entity = pd.DataFrame(records)

    macro_precision = float(df_per_entity['precision'].mean())
    macro_recall = float(df_per_entity['recall'].mean())
    macro_f_beta = float(df_per_entity['f_beta'].mean())

    return {
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f_beta": round(macro_f_beta, 6),
        "per_entity_scores": df_per_entity,
        "evaluated_entities_count": len(df_per_entity)
    }
