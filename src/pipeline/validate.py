import os
import sys
import pandas as pd
from typing import Dict, Any, Optional

from src.pipeline.scoring import compute_macro_f_beta


def run_full_validation(
    matching_results_path: str,
    candidate_pairs_path: Optional[str] = None,
    test_dir: str = "dataset/test",
    ground_truth_path: Optional[str] = None,
    check_ids: bool = False,
    entity_ids_filter: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Comprehensive pipeline validation wrapper:
    1. Runs official format and integrity checks from utils/validate_submission.py
    2. If ground_truth_path is provided, evaluates exact Macro F0.5, Macro Precision, and Macro Recall.
    
    Returns:
        dict: {
            "format_valid": bool,
            "errors": List[str],
            "warnings": List[str],
            "scores": Optional[Dict[str, Any]]
        }
    """
    # Import validator dynamically
    try:
        from utils.validate_submission import validate as official_validate
        errors, warnings = official_validate(
            matching_path=matching_results_path,
            candidate_path=candidate_pairs_path,
            test_dir=test_dir,
            check_ids=check_ids
        )
    except Exception as e:
        errors = [f"Format validator failed with exception: {str(e)}"]
        warnings = []

    format_valid = (len(errors) == 0)

    scores_result = None
    if ground_truth_path and os.path.isfile(ground_truth_path):
        try:
            df_pred = pd.read_csv(matching_results_path, sep='\t')
            df_gt = pd.read_csv(ground_truth_path, sep='\t')
            
            f_eval = compute_macro_f_beta(
                predictions_df=df_pred,
                ground_truth_df=df_gt,
                beta=0.5,
                entity_ids_filter=entity_ids_filter
            )
            scores_result = {
                "macro_precision": f_eval["macro_precision"],
                "macro_recall": f_eval["macro_recall"],
                "macro_f_beta": f_eval["macro_f_beta"],
                "evaluated_entities_count": f_eval["evaluated_entities_count"]
            }
        except Exception as e:
            errors.append(f"Ground truth evaluation failed: {str(e)}")
            format_valid = False

    return {
        "format_valid": format_valid,
        "errors": errors,
        "warnings": warnings,
        "scores": scores_result
    }
