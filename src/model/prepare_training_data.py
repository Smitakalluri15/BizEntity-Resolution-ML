"""
Training data preparation with tiered negative sampling for Business Entity Resolution.
"""

from typing import Set, Optional, List, Tuple
import numpy as np
import pandas as pd


def prepare_training_data(
    labeled_features_df: pd.DataFrame,
    train_ids: Set[str],
    negative_ratio: float = 15.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Prepares training data using tiered negative sampling:
      1. Strictly filters to records where source1_entity_id is in train_ids (never uses val_ids).
      2. Retains 100% of positive pairs (is_true_match == 1).
      3. Retains 100% of exact-tier negatives (is_true_match == 0 and is_exact_tier == 1) to teach
         the model fine-grained distinctions between confusable near-misses.
      4. Subsamples fuzzy-tier negatives (is_true_match == 0 and is_exact_tier == 0) to achieve
         an overall negative:positive ratio of approximately negative_ratio : 1.
    
    Returns:
        pd.DataFrame: Balanced training dataset containing features, metadata, and is_true_match.
    """
    # 1. Filter strictly to training split
    train_mask = labeled_features_df['source1_entity_id'].astype(str).isin(train_ids)
    train_df = labeled_features_df[train_mask].copy()

    if train_df.empty:
        raise ValueError("No candidate feature rows found for the provided train_ids.")

    if 'is_true_match' not in train_df.columns:
        raise ValueError("labeled_features_df must contain an 'is_true_match' column.")

    pos_mask = train_df['is_true_match'] == 1
    pos_df = train_df[pos_mask].copy()
    n_pos = len(pos_df)

    if n_pos == 0:
        raise ValueError("No positive ground truth pairs found in training split.")

    # 2. Separate exact-tier negatives and fuzzy-tier negatives
    exact_tier_col = 'is_exact_tier' if 'is_exact_tier' in train_df.columns else None
    if exact_tier_col:
        neg_exact_mask = (~pos_mask) & (train_df[exact_tier_col] == 1)
        neg_fuzzy_mask = (~pos_mask) & (train_df[exact_tier_col] == 0)
    else:
        # Fallback if is_exact_tier not present: all negatives treated uniformly
        neg_exact_mask = pd.Series(False, index=train_df.index)
        neg_fuzzy_mask = ~pos_mask

    neg_exact_df = train_df[neg_exact_mask].copy()
    neg_fuzzy_df = train_df[neg_fuzzy_mask].copy()

    n_exact_neg = len(neg_exact_df)
    n_fuzzy_neg = len(neg_fuzzy_df)

    # 3. Target total negatives
    target_total_neg = int(round(n_pos * negative_ratio))
    needed_fuzzy_neg = max(0, target_total_neg - n_exact_neg)

    # 4. Subsample fuzzy negatives
    if needed_fuzzy_neg >= n_fuzzy_neg:
        sampled_fuzzy_df = neg_fuzzy_df
    else:
        sampled_fuzzy_df = neg_fuzzy_df.sample(n=needed_fuzzy_neg, random_state=random_seed)

    # 5. Concatenate and shuffle
    sampled_training_df = pd.concat(
        [pos_df, neg_exact_df, sampled_fuzzy_df],
        ignore_index=True
    ).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    final_pos = (sampled_training_df['is_true_match'] == 1).sum()
    final_neg = (sampled_training_df['is_true_match'] == 0).sum()
    final_ratio = final_neg / max(1, final_pos)

    print(f"[prepare_training_data] Training dataset prepared:")
    print(f"  Positives:           {final_pos:,}")
    print(f"  Exact-tier Negatives:{n_exact_neg:,} (100% retained)")
    print(f"  Fuzzy-tier Negatives:{len(sampled_fuzzy_df):,} (sampled from {n_fuzzy_neg:,})")
    print(f"  Total Samples:       {len(sampled_training_df):,}")
    print(f"  Effective Ratio:     {final_ratio:.2f} : 1 (Neg : Pos)")

    return sampled_training_df
