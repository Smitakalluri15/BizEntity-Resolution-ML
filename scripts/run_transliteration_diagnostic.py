import os
import re
import json
import difflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.normalize.transliteration import detect_script
from src.normalize.name import normalize_name, normalize_name_tokens, get_name_char_ngrams

os.makedirs('notebooks/eda_assets', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

print("="*60)
print("DIAGNOSTIC: CROSS-SCRIPT TRANSLITERATION QUALITY CHECK")
print("="*60)

# -------------------------------------------------------------
# TASK 1: Build a real cross-script ground-truth sample
# -------------------------------------------------------------
print("\n--- TASK 1: Loading Data & Finding Cross-Script GT Pairs ---")

s1_path = 'dataset/train/train_source1.tsv'
s2_path = 'dataset/train/train_source2.tsv'
s3_path = 'dataset/train/train_source3.tsv'
gt_path = 'dataset/train/train_ground_truth.tsv'

df_s1 = pd.read_csv(s1_path, sep='\t')
df_s2 = pd.read_csv(s2_path, sep='\t')
df_s3 = pd.read_csv(s3_path, sep='\t')
df_gt = pd.read_csv(gt_path, sep='\t')

print(f"Loaded: S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)}, GT={len(df_gt)}")

# Map S1 id -> business_name
s1_name_map = dict(zip(df_s1['entity_id'], df_s1['business_name']))

# Map matched_entity_id -> source1_entity_id
print("Building reverse ground truth mapping (matched_id -> s1_id)...")
matched_to_s1 = {}
for idx, row in df_gt.dropna(subset=['matched_entity_ids']).iterrows():
    s1_id = row['source1_entity_id']
    raw_matches = str(row['matched_entity_ids'])
    for m in raw_matches.split(','):
        m = m.strip()
        if m:
            matched_to_s1[m] = s1_id

print(f"Total matched external IDs in Ground Truth: {len(matched_to_s1):,}")

# Filter S2 and S3 for cross-script business_name
print("Scanning external records for non-Latin scripts...")
df_s2_clean = df_s2.dropna(subset=['business_name']).copy()
df_s3_clean = df_s3.dropna(subset=['business_name']).copy()

df_s2_clean['source'] = 'Source 2'
df_s3_clean['source'] = 'Source 3'

df_ext = pd.concat([df_s2_clean, df_s3_clean], ignore_index=True)

# Find unique business names to speed up script detection
unique_ext_names = df_ext['business_name'].unique()
print(f"Unique external business names: {len(unique_ext_names):,}")

name_to_script = {}
for name in unique_ext_names:
    name_to_script[name] = detect_script(name)

df_ext['script'] = df_ext['business_name'].map(name_to_script)

# Filter for non-latin script
df_cross = df_ext[df_ext['script'] != 'latin'].copy()
print(f"Total cross-script external records: {len(df_cross):,}")
print("Script breakdown in external records:")
print(df_cross['script'].value_counts())

# Add S1 ID from ground truth
df_cross['s1_id'] = df_cross['entity_id'].map(matched_to_s1)
df_cross_matched = df_cross.dropna(subset=['s1_id']).copy()

print(f"\nCross-script records with true S1 ground-truth match: {len(df_cross_matched):,}")
print(df_cross_matched['script'].value_counts())

# Sample exactly 500 records with fixed seed 42
SAMPLE_SIZE = 500
if len(df_cross_matched) > SAMPLE_SIZE:
    df_sample = df_cross_matched.sample(n=SAMPLE_SIZE, random_state=42).copy()
else:
    df_sample = df_cross_matched.copy()

print(f"\nFinal sample size: {len(df_sample)}")
print("Sample script distribution:")
print(df_sample['script'].value_counts())
print("\nSample source distribution:")
print(df_sample['source'].value_counts())

# Add S1 raw name
df_sample['s1_name_raw'] = df_sample['s1_id'].map(s1_name_map)
df_sample = df_sample.rename(columns={'entity_id': 'matched_id', 'business_name': 'matched_name_raw', 'script': 'script_detected'})

# -------------------------------------------------------------
# TASK 2: Compute similarity metrics
# -------------------------------------------------------------
print("\n--- TASK 2: Measuring Similarities with Phase 2 Normalization ---")

def calc_jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

def get_char_bigrams(text: str) -> set:
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i+2] for i in range(len(text) - 1)}

results = []
for idx, row in df_sample.iterrows():
    s1_raw = str(row['s1_name_raw']) if pd.notna(row['s1_name_raw']) else ""
    m_raw = str(row['matched_name_raw']) if pd.notna(row['matched_name_raw']) else ""
    
    s1_norm = normalize_name(s1_raw)
    m_norm = normalize_name(m_raw)
    
    # 1. SequenceMatcher ratio
    seq_ratio = difflib.SequenceMatcher(None, s1_norm, m_norm).ratio()
    
    # 2. Bigram Jaccard
    s1_bi = get_char_bigrams(s1_norm)
    m_bi = get_char_bigrams(m_norm)
    bi_jaccard = calc_jaccard(s1_bi, m_bi)
    
    # 3. Trigram Jaccard
    s1_tri = get_name_char_ngrams(s1_norm, n=3)
    m_tri = get_name_char_ngrams(m_norm, n=3)
    tri_jaccard = calc_jaccard(s1_tri, m_tri)
    
    # 4. Token Jaccard
    s1_tok = set(normalize_name_tokens(s1_raw))
    m_tok = set(normalize_name_tokens(m_raw))
    tok_jaccard = calc_jaccard(s1_tok, m_tok)
    
    results.append({
        's1_id': row['s1_id'],
        'matched_id': row['matched_id'],
        'source': row['source'],
        's1_name_raw': s1_raw,
        'matched_name_raw': m_raw,
        's1_name_norm': s1_norm,
        'matched_name_norm': m_norm,
        'script_detected': row['script_detected'],
        'seq_ratio': round(seq_ratio, 4),
        'bigram_jaccard': round(bi_jaccard, 4),
        'trigram_jaccard': round(tri_jaccard, 4),
        'token_jaccard': round(tok_jaccard, 4)
    })

df_res = pd.DataFrame(results)

# Save sample CSV
csv_out = 'notebooks/eda_assets/transliteration_quality_sample.csv'
df_res.to_csv(csv_out, index=False)
print(f"Saved 500-pair quality sample to {csv_out}")

# -------------------------------------------------------------
# TASK 3: Detailed Distribution Analysis
# -------------------------------------------------------------
print("\n--- TASK 3: Distribution Statistics ---")

metrics = ['seq_ratio', 'bigram_jaccard', 'trigram_jaccard', 'token_jaccard']

for m in metrics:
    print(f"\nMetric: {m}")
    s = df_res[m]
    print(f"  Mean:   {s.mean():.4f}")
    print(f"  Median: {s.median():.4f}")
    print(f"  Std:    {s.std():.4f}")
    print(f"  Min:    {s.min():.4f}")
    print(f"  Max:    {s.max():.4f}")
    print(f"  % < 0.30: {(s < 0.30).mean()*100:.1f}%")
    print(f"  % < 0.40: {(s < 0.40).mean()*100:.1f}%")
    print(f"  % < 0.50: {(s < 0.50).mean()*100:.1f}%")
    print(f"  % >= 0.60: {(s >= 0.60).mean()*100:.1f}%")

print("\n--- By Script Breakdown (Mean / Median) ---")
for script, grp in df_res.groupby('script_detected'):
    print(f"\nScript: {script} (N={len(grp)})")
    for m in ['seq_ratio', 'bigram_jaccard', 'trigram_jaccard', 'token_jaccard']:
        print(f"  {m:16s}: Mean={grp[m].mean():.4f}, Median={grp[m].median():.4f}, % < 0.30={(grp[m]<0.30).mean()*100:5.1f}%, % >= 0.60={(grp[m]>=0.60).mean()*100:5.1f}%")

print("\n--- By Source Breakdown (Mean / Median) ---")
for src, grp in df_res.groupby('source'):
    print(f"\nSource: {src} (N={len(grp)})")
    for m in ['seq_ratio', 'bigram_jaccard', 'trigram_jaccard', 'token_jaccard']:
        print(f"  {m:16s}: Mean={grp[m].mean():.4f}, Median={grp[m].median():.4f}")

# Plotting histograms
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = ['#2b5c8f', '#2a9d8f', '#e76f51', '#7b2cbf']
for ax, m, col, title in zip(axes.flatten(), metrics, colors, 
    ['SequenceMatcher Ratio', 'Char Bigram Jaccard', 'Char Trigram Jaccard', 'Token-Set Jaccard']):
    sns.histplot(df_res[m], bins=25, kde=True, ax=ax, color=col, edgecolor='white', alpha=0.7)
    ax.axvline(df_res[m].mean(), color='#b7094c', linestyle='--', linewidth=1.5, label=f'Mean ({df_res[m].mean():.2f})')
    ax.axvline(df_res[m].median(), color='#1d3557', linestyle=':', linewidth=1.5, label=f'Median ({df_res[m].median():.2f})')
    ax.set_title(f'{title} Distribution (N=500)', fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('Similarity Score (0.0 - 1.0)', fontsize=11)
    ax.set_ylabel('Pair Count', fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.legend(loc='upper right', frameon=True)

plt.tight_layout()
hist_png = 'notebooks/eda_assets/transliteration_quality_histograms.png'
plt.savefig(hist_png, dpi=300)
plt.close()
print(f"\nSaved histograms to {hist_png}")

# -------------------------------------------------------------
# TASK 4: Spot-check worst 20 cases
# -------------------------------------------------------------
print("\n--- TASK 4: 20 Lowest-Scoring Pairs by Trigram Jaccard ---")
df_worst = df_res.sort_values(by='trigram_jaccard').head(20)

for rank, (_, row) in enumerate(df_worst.iterrows(), 1):
    print(f"#{rank:02d} | Tri: {row['trigram_jaccard']:.3f} | Bi: {row['bigram_jaccard']:.3f} | Seq: {row['seq_ratio']:.3f} | Script: {row['script_detected']}")
    print(f"     S1 Raw:     {row['s1_name_raw']}")
    print(f"     Match Raw:  {row['matched_name_raw']}")
    print(f"     S1 Norm:    {row['s1_name_norm']}")
    print(f"     Match Norm: {row['matched_name_norm']}")
    print("-" * 50)
