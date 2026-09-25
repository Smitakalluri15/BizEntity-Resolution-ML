"""
Validation threshold tuning module for Macro F0.5 Optimization in Business Entity Resolution.
"""

from typing import List, Dict, Optional, Iterable, Any, Union
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.pipeline.scoring import compute_macro_f_beta
from src.model.predict import predict_match_probabilities, assemble_entity_predictions


def sweep_thresholds(
    model: Union[lgb.LGBMClassifier, lgb.Booster],
    val_features_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    feature_columns: List[str],
    val_s1_ids: Optional[Iterable[str]] = None,
    thresholds: Optional[Iterable[float]] = None,
    mode: str = "multi",
    beta: float = 0.5
) -> pd.DataFrame:
    """
    Evaluates Macro F-beta across candidate probability thresholds on the validation split.
    
    Args:
        model: Trained classifier.
        val_features_df: DataFrame containing validation candidate pair features.
        ground_truth_df: Ground truth DataFrame ['source1_entity_id', 'matched_entity_ids'].
        feature_columns: List of feature names to use as input.
        val_s1_ids: Optional list/set of validation S1 entity IDs. If None, extracts unique
                    source1_entity_id from val_features_df.
        thresholds: Array or list of thresholds to test (default: np.arange(0.05, 0.96, 0.05)).
        mode: Candidate selection mode ('multi' or 'top1').
        beta: F-beta beta factor (default 0.5).
        
    Returns:
        pd.DataFrame: Sweep results with columns:
          ['threshold', 'mode', 'macro_precision', 'macro_recall', 'macro_f_beta', 'evaluated_entities']
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    # 1. Determine validation entities
    if val_s1_ids is not None:
        eval_s1_set = set(val_s1_ids)
        # Only evaluate S1 entities present in ground truth
        gt_s1_set = set(ground_truth_df['source1_entity_id'].astype(str))
        eval_s1_ids = sorted(list(eval_s1_set.intersection(gt_s1_set)))
    else:
        eval_s1_ids = sorted(val_features_df['source1_entity_id'].astype(str).unique().tolist())

    if not eval_s1_ids:
        raise ValueError("No valid S1 entities found for validation evaluation.")

    # 2. Pre-filter ground truth to eval_s1_ids for fast scoring
    gt_s1_col = 'source1_entity_id' if 'source1_entity_id' in ground_truth_df.columns else ground_truth_df.columns[0]
    gt_filtered = ground_truth_df[ground_truth_df[gt_s1_col].astype(str).isin(set(eval_s1_ids))].copy()

    # 3. Compute probabilities once across all validation pairs
    scored_val_df = predict_match_probabilities(model, val_features_df, feature_columns)

    # 4. Sweep thresholds
    results = []
    for thresh in thresholds:
        t_val = float(thresh)
        # Assemble predictions for all evaluated S1 entities
        pred_df = assemble_entity_predictions(
            scored_df=scored_val_df,
            threshold=t_val,
            mode=mode,
            all_s1_entities=eval_s1_ids
        )

        # Compute Macro F-beta
        score_res = compute_macro_f_beta(
            predictions_df=pred_df,
            ground_truth_df=gt_filtered,
            beta=beta,
            entity_ids_filter=eval_s1_ids
        )

        results.append({
            'threshold': round(t_val, 4),
            'mode': mode,
            'macro_precision': score_res['macro_precision'],
            'macro_recall': score_res['macro_recall'],
            'macro_f_beta': score_res['macro_f_beta'],
            'evaluated_entities': score_res['evaluated_entities_count']
        })

    df_sweep = pd.DataFrame(results)
    return df_sweep


def find_best_threshold(sweep_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies the optimal threshold configuration maximizing macro_f_beta.
    """
    best_idx = sweep_df['macro_f_beta'].idxmax()
    best_row = sweep_df.loc[best_idx].to_dict()
    return best_row
