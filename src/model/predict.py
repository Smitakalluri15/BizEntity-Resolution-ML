"""
Candidate probability scoring and entity-level prediction assembly for Business Entity Resolution.
"""

from typing import List, Optional, Set, Iterable, Dict, Any, Union
import numpy as np
import pandas as pd
import lightgbm as lgb


def predict_match_probabilities(
    model: Union[lgb.LGBMClassifier, lgb.Booster],
    features_df: pd.DataFrame,
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Computes pairwise match probabilities for all rows in features_df.
    
    Returns:
        pd.DataFrame: Copy of features_df with an added 'match_probability' column [0.0, 1.0].
    """
    out_df = features_df.copy()
    
    # Extract feature matrix
    X = features_df[feature_columns]

    if hasattr(model, 'predict_proba'):
        # LGBMClassifier
        probs = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'predict'):
        # lgb.Booster
        probs = model.predict(X)
    else:
        raise ValueError("Provided model must have a predict_proba or predict method.")

    out_df['match_probability'] = probs.astype(np.float32)
    return out_df


def assemble_entity_predictions(
    scored_df: pd.DataFrame,
    threshold: float = 0.5,
    mode: str = "multi",
    all_s1_entities: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """
    Assembles entity-level predictions formatted as the official matching_results.tsv:
      source1_entity_id | matched_entity_ids (comma-separated string)
      
    Args:
        scored_df: DataFrame containing ['source1_entity_id', 'candidate_entity_id', 'match_probability'].
        threshold: Probability threshold for positive classification.
        mode: Prediction selection strategy:
              - 'multi': Include ALL candidates with match_probability >= threshold.
              - 'top1': Include ONLY the single highest-probability candidate if it clears threshold.
        all_s1_entities: Optional complete list/set of S1 entity IDs. If provided, ensures every
                         entity in all_s1_entities appears exactly once in the output even if it had
                         0 candidates generated or 0 candidates cleared the threshold.
                         
    Returns:
        pd.DataFrame: Entity-level predictions with columns ['source1_entity_id', 'matched_entity_ids'].
    """
    if mode not in ("multi", "top1"):
        raise ValueError(f"Invalid mode '{mode}'. Must be either 'multi' or 'top1'.")

    # 1. Determine universal set of S1 IDs
    if all_s1_entities is not None:
        target_s1_ids = list(all_s1_entities)
    else:
        target_s1_ids = scored_df['source1_entity_id'].drop_duplicates().tolist()

    if scored_df.empty:
        return pd.DataFrame({
            'source1_entity_id': target_s1_ids,
            'matched_entity_ids': [''] * len(target_s1_ids)
        })

    # 2. Filter candidate pairs clearing threshold
    qualifying_mask = scored_df['match_probability'] >= threshold
    qualifying_df = scored_df[qualifying_mask]

    predictions_map: Dict[str, List[str]] = {}

    if not qualifying_df.empty:
        if mode == "top1":
            # Sort descending and drop duplicate S1 IDs
            sorted_qual = qualifying_df.sort_values('match_probability', ascending=False)
            top1_per_s1 = sorted_qual.drop_duplicates(subset=['source1_entity_id'], keep='first')
            for s1_id, cand_id in zip(top1_per_s1['source1_entity_id'], top1_per_s1['candidate_entity_id']):
                predictions_map[s1_id] = [str(cand_id)]
        else:  # multi mode
            # Group by source1_entity_id
            for s1_id, cand_id in zip(qualifying_df['source1_entity_id'], qualifying_df['candidate_entity_id']):
                if s1_id not in predictions_map:
                    predictions_map[s1_id] = []
                predictions_map[s1_id].append(str(cand_id))

    # 3. Format into final DataFrame
    s1_col = []
    matched_col = []
    for s1_id in target_s1_ids:
        cand_list = predictions_map.get(s1_id)
        s1_col.append(str(s1_id))
        if cand_list:
            unique_cands = sorted(list(set(cand_list)))
            matched_col.append(','.join(unique_cands))
        else:
            matched_col.append('')

    return pd.DataFrame({
        'source1_entity_id': s1_col,
        'matched_entity_ids': matched_col
    })
