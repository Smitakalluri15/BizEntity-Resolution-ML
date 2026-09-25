import os
import pandas as pd
import numpy as np

from src.pipeline.split import create_train_val_split, save_splits, load_splits
from src.pipeline.scoring import compute_macro_f_beta

print("="*60)
print("PHASE 3: GENERATING OFFICIAL TRAIN/VAL SPLIT & BASELINE CHECK")
print("="*60)

s1_path = 'dataset/train/train_source1.tsv'
gt_path = 'dataset/train/train_ground_truth.tsv'

print(f"Loading {s1_path} and {gt_path}...")
df_s1 = pd.read_csv(s1_path, sep='\t')
df_gt = pd.read_csv(gt_path, sep='\t')
print(f"Loaded: S1 shape={df_s1.shape}, GT shape={df_gt.shape}")

print("\nGenerating 80/20 Stratified Train/Val Split (seed=42)...")
train_ids, val_ids = create_train_val_split(
    train_source1_df=df_s1,
    train_ground_truth_df=df_gt,
    val_fraction=0.2,
    random_seed=42
)

print(f"Total S1 entities: {len(df_s1):,}")
print(f"Train S1 entities: {len(train_ids):,} ({len(train_ids)/len(df_s1)*100:.2f}%)")
print(f"Val S1 entities:   {len(val_ids):,} ({len(val_ids)/len(df_s1)*100:.2f}%)")

# Save to data/splits/
splits_dir = 'data/splits'
train_file, val_file = save_splits(train_ids, val_ids, output_dir=splits_dir)
print(f"\nSaved split files to:\n  - {train_file}\n  - {val_file}")

# Verify stratification in validation set
val_id_set = set(val_ids)
df_gt_val = df_gt[df_gt['source1_entity_id'].isin(val_id_set)].copy()

def get_match_count(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    return len([x for x in str(val).split(',') if x.strip()])

df_gt_val['match_count'] = df_gt_val['matched_entity_ids'].apply(get_match_count)
val_total = len(df_gt_val)
val_singletons = (df_gt_val['match_count'] == 0).sum()
val_one_match = (df_gt_val['match_count'] == 1).sum()
val_multi_match = (df_gt_val['match_count'] >= 2).sum()

print("\nValidation Set Match Distribution:")
print(f"  Singletons (0 matches): {val_singletons:,} ({val_singletons/val_total*100:.2f}%) [Population: ~5.58%]")
print(f"  1 match:                {val_one_match:,} ({val_one_match/val_total*100:.2f}%) [Population: ~5.40%]")
print(f"  2+ matches:             {val_multi_match:,} ({val_multi_match/val_total*100:.2f}%) [Population: ~89.02%]")

# -------------------------------------------------------------
# TASK 6: Trivial Baseline Check (Predict All Empty)
# -------------------------------------------------------------
print("\n" + "="*60)
print("TASK 6: TRIVIAL BASELINE CHECK (PREDICT ALL EMPTY)")
print("="*60)

# Create trivial prediction dataframe: predict empty string for all validation entities
trivial_pred_df = pd.DataFrame({
    'source1_entity_id': val_ids,
    'matched_entity_ids': [''] * len(val_ids)
})

print(f"Scoring trivial 'all empty' baseline across {len(val_ids):,} validation entities...")
res = compute_macro_f_beta(
    predictions_df=trivial_pred_df,
    ground_truth_df=df_gt,
    beta=0.5,
    entity_ids_filter=val_ids
)

print("\nTrivial Baseline Results:")
print(f"  Macro Precision: {res['macro_precision']:.6f} ({res['macro_precision']*100:.2f}%)")
print(f"  Macro Recall:    {res['macro_recall']:.6f} ({res['macro_recall']*100:.2f}%)")
print(f"  Macro F0.5:      {res['macro_f_beta']:.6f} ({res['macro_f_beta']*100:.2f}%)")
print(f"  Evaluated Count: {res['evaluated_entities_count']:,}")

expected_score = val_singletons / val_total
print(f"\nExpected Macro F0.5 based on true singletons: {expected_score:.6f} ({expected_score*100:.2f}%)")
diff = abs(res['macro_f_beta'] - expected_score)
print(f"Difference: {diff:.8f}")

assert diff < 1e-5, f"Trivial baseline score {res['macro_f_beta']} does not match expected {expected_score}"
print("\nSANITY CHECK PASSED: Trivial baseline exactly matches singleton percentage!")
