"""
Phase 7 Sample-Scale GBDT Matcher Training, Evaluation, and Threshold Sweep Pipeline.
"""

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.pipeline.split import load_splits
from src.features.build_features import assemble_labels
from src.model.prepare_training_data import prepare_training_data
from src.model.train import train_matcher, get_feature_importance, DEFAULT_FEATURE_COLUMNS
from src.model.predict import predict_match_probabilities, assemble_entity_predictions
from src.model.threshold_tuning import sweep_thresholds, find_best_threshold


def main():
    print("=" * 70)
    print("PHASE 7: GRADIENT BOOSTING MATCHER (SAMPLE-SCALE DRY RUN)")
    print("=" * 70)

    # 1. Load Splits and Ground Truth
    print("\n1. Loading Splits and Ground Truth...")
    train_ids, val_ids = load_splits('data/splits')
    gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
    print(f"   Loaded {len(train_ids):,} train IDs, {len(val_ids):,} val IDs.")

    # 2. Load Sample Features & Assemble Labels
    features_input_path = "data/features/sample_candidate_features.parquet"
    labeled_output_path = "data/features/sample_candidate_features_labeled.parquet"

    print(f"\n2. Loading Sample Features from {features_input_path}...")
    features_df = pd.read_parquet(features_input_path)
    print(f"   Raw sample feature shape: {features_df.shape[0]:,} pairs x {features_df.shape[1]} columns")

    print("   Assembling ground truth labels...")
    labeled_df = assemble_labels(features_df, gt_df)
    
    os.makedirs(os.path.dirname(labeled_output_path), exist_ok=True)
    labeled_df.to_parquet(labeled_output_path, index=False)
    print(f"   Saved labeled dataset to {labeled_output_path}")

    # Analyze S1 split representation
    unique_s1 = set(labeled_df['source1_entity_id'])
    train_s1_in_sample = sorted(list(unique_s1.intersection(train_ids)))
    val_s1_in_sample = sorted(list(unique_s1.intersection(val_ids)))

    print(f"   Total unique S1 entities in sample: {len(unique_s1):,}")
    print(f"   Training S1 entities in sample:     {len(train_s1_in_sample):,}")
    print(f"   Validation S1 entities in sample:   {len(val_s1_in_sample):,}")

    # 3. Prepare Training Data with Tiered Negative Sampling
    print("\n3. Preparing Training Dataset with Tiered Negative Sampling...")
    t0 = time.time()
    train_data = prepare_training_data(
        labeled_features_df=labeled_df,
        train_ids=set(train_s1_in_sample),
        negative_ratio=15.0,
        random_seed=42
    )

    # 4. Train LightGBM Binary Classification Matcher
    print("\n4. Training LightGBM Matcher...")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_artifact_path = os.path.join(model_dir, "phase7_matcher_sample.txt")

    feature_cols = [c for c in DEFAULT_FEATURE_COLUMNS if c in train_data.columns]
    
    # Train model
    model = train_matcher(
        training_df=train_data,
        feature_columns=feature_cols,
        target_col="is_true_match",
        model_params={
            'learning_rate': 0.05,
            'num_leaves': 31,
            'n_estimators': 300,
            'random_state': 42
        },
        model_output_path=model_artifact_path
    )

    # 5. Feature Importance Analysis
    print("\n5. Feature Importance Ranking:")
    feat_imp = get_feature_importance(model, feature_cols)
    for i, row in feat_imp.iterrows():
        print(f"   {i+1:02d}. {row['feature']:<30} | Splits: {int(row['importance']):4d} ({row['importance_pct']:5.2f}%)")

    # 6. Threshold Tuning on Validation Split Sample
    print(f"\n6. Executing Threshold Sweep on Validation Sample ({len(val_s1_in_sample):,} S1 entities)...")
    val_pairs_df = labeled_df[labeled_df['source1_entity_id'].isin(val_s1_in_sample)].copy()
    print(f"   Validation candidate pairs: {len(val_pairs_df):,}")

    thresholds_to_test = np.arange(0.05, 0.96, 0.05)

    # Sweep mode="multi"
    print("\n   [Sweep Mode: 'multi']")
    sweep_multi = sweep_thresholds(
        model=model,
        val_features_df=val_pairs_df,
        ground_truth_df=gt_df,
        feature_columns=feature_cols,
        val_s1_ids=val_s1_in_sample,
        thresholds=thresholds_to_test,
        mode="multi",
        beta=0.5
    )
    best_multi = find_best_threshold(sweep_multi)

    # Sweep mode="top1"
    print("   [Sweep Mode: 'top1']")
    sweep_top1 = sweep_thresholds(
        model=model,
        val_features_df=val_pairs_df,
        ground_truth_df=gt_df,
        feature_columns=feature_cols,
        val_s1_ids=val_s1_in_sample,
        thresholds=thresholds_to_test,
        mode="top1",
        beta=0.5
    )
    best_top1 = find_best_threshold(sweep_top1)

    # 7. Print Threshold Sweep Results Table
    print("\n" + "=" * 75)
    print("VALIDATION THRESHOLD SWEEP RESULTS (SAMPLE SCALE)")
    print("=" * 75)
    print(f"{'Threshold':>10} | {'Mode':>6} | {'Macro Precision':>16} | {'Macro Recall':>14} | {'Macro F0.5':>12}")
    print("-" * 75)
    
    # Print combined table
    for i in range(len(sweep_multi)):
        r_m = sweep_multi.iloc[i]
        r_t = sweep_top1.iloc[i]
        print(f"{r_m['threshold']:10.2f} | {'multi':>6} | {r_m['macro_precision']:16.4f} | {r_m['macro_recall']:14.4f} | {r_m['macro_f_beta']:12.4f}")
        print(f"{r_t['threshold']:10.2f} | {'top1':>6} | {r_t['macro_precision']:16.4f} | {r_t['macro_recall']:14.4f} | {r_t['macro_f_beta']:12.4f}")
        print("-" * 75)

    print("\n" + "=" * 75)
    print("BEST CONFIGURATION SUMMARY")
    print("=" * 75)
    print(f"Best Multi-Match:  Threshold = {best_multi['threshold']:.2f} | Macro F0.5 = {best_multi['macro_f_beta']:.4f} (P: {best_multi['macro_precision']:.4f}, R: {best_multi['macro_recall']:.4f})")
    print(f"Best Top-1 Match:  Threshold = {best_top1['threshold']:.2f} | Macro F0.5 = {best_top1['macro_f_beta']:.4f} (P: {best_top1['macro_precision']:.4f}, R: {best_top1['macro_recall']:.4f})")
    
    overall_best = best_multi if best_multi['macro_f_beta'] >= best_top1['macro_f_beta'] else best_top1
    baseline_f05 = 0.055847
    delta_gain = overall_best['macro_f_beta'] - baseline_f05

    print(f"\nOverall Winning Mode:       {overall_best['mode'].upper()} (Threshold = {overall_best['threshold']:.2f})")
    print(f"Sample Validation Macro F0.5: {overall_best['macro_f_beta']:.6f}")
    print(f"Phase 3 Singleton Baseline:   {baseline_f05:.6f}")
    print(f"Net Score Improvement:        +{delta_gain:.6f} (+{delta_gain / baseline_f05 * 100:.1f}%)")

    print("\n" + "=" * 75)
    print("IMPORTANT DISCLAIMER & NEXT STEPS:")
    print("This phase's results are directionally useful only. Next steps:")
    print("1. Run the full-scale Phase 6 feature build across all 2.2M S1 records.")
    print("2. Re-run this Phase 7 training pipeline end-to-end on the full-scale feature dataset.")
    print("3. Re-tune the decision threshold on the full 441,363-entity validation set before treating any number as final.")
    print("=" * 75)


if __name__ == '__main__':
    main()
