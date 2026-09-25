import os
import shutil
import pytest
import pandas as pd
import numpy as np

from src.pipeline.split import create_train_val_split, save_splits, load_splits


def test_split_disjoint_and_complete():
    s1_data = {'entity_id': [f'S1-{i}' for i in range(100)]}
    # Mix of 0 matches, 1 match, multi-match
    gt_data = {
        'source1_entity_id': [f'S1-{i}' for i in range(100)],
        'matched_entity_ids': [
            '' if i < 10 else (f'S2-{i}' if i < 30 else f'S2-{i},S3-{i}') for i in range(100)
        ]
    }
    df_s1 = pd.DataFrame(s1_data)
    df_gt = pd.DataFrame(gt_data)

    train_ids, val_ids = create_train_val_split(df_s1, df_gt, val_fraction=0.2, random_seed=42)

    set_train = set(train_ids)
    set_val = set(val_ids)
    all_s1 = set(df_s1['entity_id'])

    # Disjointness
    assert len(set_train.intersection(set_val)) == 0
    # Completeness
    assert set_train.union(set_val) == all_s1
    # Fraction check
    assert len(val_ids) == 20
    assert len(train_ids) == 80


def test_split_reproducibility():
    s1_data = {'entity_id': [f'S1-{i}' for i in range(50)]}
    gt_data = {
        'source1_entity_id': [f'S1-{i}' for i in range(50)],
        'matched_entity_ids': [f'S2-{i}' for i in range(50)]
    }
    df_s1 = pd.DataFrame(s1_data)
    df_gt = pd.DataFrame(gt_data)

    t1, v1 = create_train_val_split(df_s1, df_gt, val_fraction=0.3, random_seed=123)
    t2, v2 = create_train_val_split(df_s1, df_gt, val_fraction=0.3, random_seed=123)

    assert t1 == t2
    assert v1 == v2


def test_split_serialization(tmp_path):
    train_ids = ['S1-1', 'S1-2', 'S1-3']
    val_ids = ['S1-4', 'S1-5']
    
    out_dir = str(tmp_path / "splits")
    save_splits(train_ids, val_ids, output_dir=out_dir)
    
    loaded_train, loaded_val = load_splits(splits_dir=out_dir)
    assert loaded_train == set(train_ids)
    assert loaded_val == set(val_ids)
