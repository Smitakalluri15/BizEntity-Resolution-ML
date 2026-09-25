import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Set, Optional


def create_train_val_split(
    train_source1_df: pd.DataFrame,
    train_ground_truth_df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Creates a reproducible, stratified Train/Validation split at the Source-1 (S1) entity level.
    
    CRITICAL:
    - This splits WHICH S1 entities are held out for evaluation vs available for training.
    - S2 and S3 records remain fully accessible for candidate generation and graph resolution.
    - Stratifies across match count buckets (0 matches / singletons, 1 match, 2+ multi-matches)
      to maintain exact population distribution in validation.
    
    Returns:
        (train_s1_ids, val_s1_ids): Tuple of lists containing S1 entity IDs.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0.0 and 1.0, got {val_fraction}")

    # Extract all unique S1 entity IDs
    s1_ids = train_source1_df['entity_id'].dropna().unique()
    
    # Map S1 IDs to ground truth match counts (fast zip iteration)
    gt_map = {}
    for s1_id, raw_m in zip(train_ground_truth_df['source1_entity_id'], train_ground_truth_df['matched_entity_ids']):
        raw_str = str(raw_m) if pd.notna(raw_m) else ""
        if not raw_str.strip():
            count = 0
        else:
            count = len([x for x in raw_str.split(',') if x.strip()])
        gt_map[s1_id] = count

    # Categorize into stratification strata: 0 (singleton), 1 (single match), 2 (multi-match)
    strata = {0: [], 1: [], 2: []}
    for s1_id in s1_ids:
        c = gt_map.get(s1_id, 0)
        if c == 0:
            strata[0].append(s1_id)
        elif c == 1:
            strata[1].append(s1_id)
        else:
            strata[2].append(s1_id)

    rng = np.random.default_rng(random_seed)
    train_s1_ids = []
    val_s1_ids = []

    for bucket, ids in strata.items():
        if not ids:
            continue
        ids_arr = np.array(ids)
        rng.shuffle(ids_arr)
        n_val = int(np.round(len(ids_arr) * val_fraction))
        val_ids = ids_arr[:n_val].tolist()
        train_ids = ids_arr[n_val:].tolist()
        
        val_s1_ids.extend(val_ids)
        train_s1_ids.extend(train_ids)

    # Sort for deterministic output
    train_s1_ids.sort()
    val_s1_ids.sort()

    return train_s1_ids, val_s1_ids


def save_splits(
    train_s1_ids: List[str],
    val_s1_ids: List[str],
    output_dir: str = "data/splits"
) -> Tuple[str, str]:
    """
    Saves train and validation S1 ID lists to text files.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_ids.txt")
    val_path = os.path.join(output_dir, "val_ids.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        for s1_id in train_s1_ids:
            f.write(f"{s1_id}\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s1_id in val_s1_ids:
            f.write(f"{s1_id}\n")

    return train_path, val_path


def load_splits(splits_dir: str = "data/splits") -> Tuple[Set[str], Set[str]]:
    """
    Loads fixed train and validation S1 ID sets from disk.
    """
    train_path = os.path.join(splits_dir, "train_ids.txt")
    val_path = os.path.join(splits_dir, "val_ids.txt")

    if not os.path.isfile(train_path) or not os.path.isfile(val_path):
        raise FileNotFoundError(f"Split files not found in {splits_dir}. Run create_train_val_split() first.")

    with open(train_path, "r", encoding="utf-8") as f:
        train_ids = {line.strip() for line in f if line.strip()}

    with open(val_path, "r", encoding="utf-8") as f:
        val_ids = {line.strip() for line in f if line.strip()}

    return train_ids, val_ids
