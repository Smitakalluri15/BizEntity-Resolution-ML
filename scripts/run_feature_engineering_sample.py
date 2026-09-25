"""
Feature Engineering Execution, Profiling, and Label Assembly Analysis (Phase 6).
Evaluates feature generation performance, memory usage, class balance, and blocking recall cross-check.
"""

import time
import os
import pandas as pd
import numpy as np

from src.normalize import apply_normalization
from src.normalize.phonetic import apply_phonetic_features
from src.pipeline.split import load_splits
from src.block.name_blocking import build_token_index, build_sorted_neighborhood_index
from src.block.phonetic_blocking import build_phonetic_index
from src.block.address_blocking import build_postal_index, build_city_token_index
from src.block.combine import generate_candidates
from src.features.build_features import build_features, assemble_labels


def main():
    print("=" * 70)
    print("PHASE 6: FEATURE ENGINEERING BENCHMARK & VALIDATION")
    print("=" * 70)

    # 1. Load sample data
    t0 = time.time()
    n_sample_s1 = 5000
    print(f"\n1. Loading and normalizing sample dataset ({n_sample_s1} S1 entities)...")
    s1_full = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=n_sample_s1)
    s2_full = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=50000)
    s3_full = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=50000)
    gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
    train_ids, val_ids = load_splits()

    print("   Normalizing dataframes...")
    s1_norm = apply_phonetic_features(apply_normalization(s1_full))
    s2_norm = apply_phonetic_features(apply_normalization(s2_full))
    s3_norm = apply_phonetic_features(apply_normalization(s3_full))
    cand_norm_combined = pd.concat([s2_norm, s3_norm], ignore_index=True)
    cand_norm_combined = cand_norm_combined[~cand_norm_combined['entity_id'].duplicated(keep='first')]

    print(f"   Normalized in {time.time() - t0:.2f}s")

    # 2. Build blocking indices
    print("\n2. Building Multi-Strategy Blocking Indices...")
    t_idx0 = time.time()
    token_index, stop_tokens = build_token_index(cand_norm_combined, max_token_freq=0.005)
    phonetic_index = build_phonetic_index(cand_norm_combined)
    postal_index = build_postal_index(cand_norm_combined)
    city_token_index, stop_addr_tokens = build_city_token_index(cand_norm_combined, max_token_freq=0.01)
    sn_partitions = build_sorted_neighborhood_index(cand_norm_combined)

    entity_country_map = dict(zip(cand_norm_combined['entity_id'], cand_norm_combined['country_norm']))
    entity_name_map = dict(zip(cand_norm_combined['entity_id'], cand_norm_combined['name_norm']))
    entity_phonetic_map = dict(zip(cand_norm_combined['entity_id'], cand_norm_combined['name_phonetic']))
    print(f"   Blocking indices built in {time.time() - t_idx0:.2f}s")

    # 3. Generate candidate pairs
    print("\n3. Generating Candidate Pairs...")
    t_cand0 = time.time()
    pairs = []
    blocking_meta = {}
    for _, row in s1_norm.iterrows():
        s1_id = row['entity_id']
        cands = generate_candidates(
            s1_row=row,
            token_index=token_index,
            phonetic_index=phonetic_index,
            postal_index=postal_index,
            city_token_index=city_token_index,
            sorted_neighborhood_partitions=sn_partitions,
            stop_tokens=stop_tokens,
            stop_address_tokens=stop_addr_tokens,
            entity_country_map=entity_country_map,
            entity_name_map=entity_name_map,
            entity_phonetic_map=entity_phonetic_map,
            top_k=60
        )
        for cid in cands:
            pairs.append({'source1_entity_id': s1_id, 'candidate_entity_id': cid})

    candidate_pairs_df = pd.DataFrame(pairs)
    t_cand = time.time() - t_cand0
    print(f"   Generated {len(candidate_pairs_df):,} candidate pairs for {len(s1_norm):,} S1 entities in {t_cand:.2f}s")
    print(f"   Average candidates per S1: {len(candidate_pairs_df) / len(s1_norm):.2f}")

    # 4. Profile Feature Engineering
    print("\n4. Profiling Feature Engineering Steps...")
    t_feat0 = time.time()
    
    out_dir = "data/features"
    os.makedirs(out_dir, exist_ok=True)
    out_parquet = os.path.join(out_dir, "sample_candidate_features.parquet")

    features_df = build_features(
        candidate_pairs_df,
        s1_df=s1_norm,
        s2_df=s2_norm,
        s3_df=s3_norm,
        output_path=out_parquet
    )
    t_feat = time.time() - t_feat0

    pairs_per_sec = len(features_df) / t_feat
    print(f"   Feature engineering completed in {t_feat:.2f}s ({pairs_per_sec:,.1f} pairs/sec)")
    print(f"   Shape: {features_df.shape[0]:,} rows x {features_df.shape[1]} columns")
    print(f"   Parquet file size: {os.path.getsize(out_parquet) / (1024 * 1024):.2f} MB")

    # 5. Label Assembly & Class Balance
    print("\n5. Assembling Labels & Analyzing Class Balance...")
    labeled_df = assemble_labels(features_df, gt_df, s1_ids_filter=train_ids)
    
    # Filter to S1 entities in train_ids
    train_labeled = labeled_df[labeled_df['source1_entity_id'].isin(train_ids)]
    pos_count = (train_labeled['is_true_match'] == 1).sum()
    neg_count = (train_labeled['is_true_match'] == 0).sum()
    total_train_pairs = len(train_labeled)
    pos_ratio = (pos_count / total_train_pairs) * 100 if total_train_pairs > 0 else 0

    print(f"   Train split candidate pairs: {total_train_pairs:,}")
    print(f"   Positive pairs (is_true_match=1): {pos_count:,} ({pos_ratio:.2f}%)")
    print(f"   Negative pairs (is_true_match=0): {neg_count:,} ({100 - pos_ratio:.2f}%)")
    print(f"   Imbalance ratio (Negative : Positive): {neg_count / max(1, pos_count):.1f} : 1")

    # 6. Feature List and Stats
    print("\n6. Feature Summary:")
    feature_cols = [c for c in features_df.columns if c not in ('source1_entity_id', 'candidate_entity_id', 'source_flag', 'candidate_source_strategy')]
    print(f"   Total computed numeric features: {len(feature_cols)}")
    for i, col in enumerate(feature_cols, 1):
        mean_val = features_df[col].mean()
        std_val = features_df[col].std()
        print(f"   {i:02d}. {col:<32} (mean: {mean_val:.4f}, std: {std_val:.4f})")

    # 7. Zero NaN check
    nan_sum = features_df[feature_cols].isna().sum().sum()
    print(f"\n7. NaN Validation: {nan_sum} NaNs found in numeric feature columns.")
    assert nan_sum == 0, "Failed NaN check!"

    print("\n" + "=" * 70)
    print("PHASE 6 BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == '__main__':
    main()
