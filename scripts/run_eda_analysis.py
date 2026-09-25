import os
import gc
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('notebooks/eda_assets', exist_ok=True)

print("="*60)
print("PHASE 1 EDA: FULL DATASET ANALYSIS")
print("="*60)

# -------------------------------------------------------------
# TASK 1 & 2: Load S1 & Ground Truth
# -------------------------------------------------------------
print("\n--- TASK 1: Load S1 and Ground Truth ---")
s1_path = 'dataset/train/train_source1.tsv'
s2_path = 'dataset/train/train_source2.tsv'
s3_path = 'dataset/train/train_source3.tsv'
gt_path = 'dataset/train/train_ground_truth.tsv'

df_s1 = pd.read_csv(s1_path, sep='\t')
print(f"S1 Shape: {df_s1.shape}")
df_gt = pd.read_csv(gt_path, sep='\t')
print(f"GT Shape: {df_gt.shape}")

print("\nS1 Nulls:")
print(df_s1.isnull().sum())
print("S1 Duplicate IDs:", df_s1['entity_id'].duplicated().sum())
print("S1 Prefix Valid (starts with 'S1-'):", df_s1['entity_id'].str.startswith('S1-').all())

print("\nGT Nulls:")
print(df_gt.isnull().sum())
print("GT Duplicate S1 IDs:", df_gt['source1_entity_id'].duplicated().sum())
print("GT Prefix Valid:", df_gt['source1_entity_id'].str.startswith('S1-').all())

# -------------------------------------------------------------
# TASK 2: Ground Truth Structure Analysis
# -------------------------------------------------------------
print("\n--- TASK 2: Ground Truth Structure Analysis ---")

def parse_matches(val):
    if pd.isna(val) or str(val).strip() == '':
        return []
    return [x.strip() for x in str(val).split(',') if x.strip()]

df_gt['matches_list'] = df_gt['matched_entity_ids'].apply(parse_matches)
df_gt['match_count'] = df_gt['matches_list'].apply(len)

total_s1 = len(df_gt)
singletons = int((df_gt['match_count'] == 0).sum())
one_match = int((df_gt['match_count'] == 1).sum())
multi_match = int((df_gt['match_count'] >= 2).sum())

print(f"a) Total S1 records in GT: {total_s1:,}")
print(f"b) Singletons (0 matches): {singletons:,} ({singletons/total_s1*100:.2f}%)")
print(f"c) Exactly 1 match: {one_match:,} ({one_match/total_s1*100:.2f}%)")
print(f"d) Two or more matches: {multi_match:,} ({multi_match/total_s1*100:.2f}%)")

match_dist = df_gt['match_count'].value_counts().sort_index()
print("\nMatch count breakdown:")
print(match_dist)

def bucket_match_count(c):
    if c >= 5:
        return '5+'
    return str(c)

df_gt['match_bucket'] = df_gt['match_count'].apply(bucket_match_count)
bucket_order = ['0', '1', '2', '3', '4', '5+']
bucket_counts = df_gt['match_bucket'].value_counts().reindex(bucket_order, fill_value=0)
bucket_pcts = bucket_counts / total_s1 * 100

print("\nMatch Bucket Counts & %:")
for b, c, p in zip(bucket_order, bucket_counts, bucket_pcts):
    print(f"  Bucket {b}: {c:,} ({p:.2f}%)")

# Plot match count distribution chart
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
bars = ax.bar(bucket_order, bucket_counts.values, color=['#4A90E2', '#50E3C2', '#F5A623', '#E94E77', '#BD10E0', '#9013FE'], edgecolor='#333333', linewidth=1)
ax.set_title('Match Count per Source 1 Record Distribution', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Number of Matched External Entities (S2/S3)', fontsize=11, labelpad=10)
ax.set_ylabel('Number of Source 1 Entities', fontsize=11, labelpad=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))

for bar, count, pct in zip(bars, bucket_counts.values, bucket_pcts):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + max(bucket_counts.values)*0.015,
            f"{count:,}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9, fontweight='semibold')

ax.set_ylim(0, max(bucket_counts.values) * 1.15)
plt.tight_layout()
plt.savefig('notebooks/eda_assets/match_count_distribution.png')
plt.close()
print("Saved notebooks/eda_assets/match_count_distribution.png")

# Analyze matched IDs
all_matched_ids = [m for sublist in df_gt['matches_list'] for m in sublist]
total_matched_ids = len(all_matched_ids)
s2_matched_ids = [m for m in all_matched_ids if m.startswith('S2-')]
s3_matched_ids = [m for m in all_matched_ids if m.startswith('S3-')]
other_matched_ids = [m for m in all_matched_ids if not (m.startswith('S2-') or m.startswith('S3-'))]

print(f"\nf) Matched IDs Breakdown:")
print(f"   Total matched references: {total_matched_ids:,}")
print(f"   S2 references: {len(s2_matched_ids):,} ({len(s2_matched_ids)/total_matched_ids*100:.2f}%)")
print(f"   S3 references: {len(s3_matched_ids):,} ({len(s3_matched_ids)/total_matched_ids*100:.2f}%)")
print(f"   Other/Invalid prefix: {len(other_matched_ids):,}")

s1_ids_set = set(df_s1['entity_id'])
gt_s1_ids_set = set(df_gt['source1_entity_id'])

print(f"\ng) S1 in GT vs S1 in train_source1:")
print(f"   S1 in GT not in train_source1: {len(gt_s1_ids_set - s1_ids_set)}")
print(f"   S1 in train_source1 not in GT: {len(s1_ids_set - gt_s1_ids_set)}")

set_s2_matched = set(s2_matched_ids)
set_s3_matched = set(s3_matched_ids)

# Sample paired S1 IDs now across 1-match, 2-match, and 3+ match buckets
np.random.seed(42)
sample_1 = df_gt[df_gt['match_count'] == 1].sample(n=35, random_state=42)
sample_2 = df_gt[df_gt['match_count'] == 2].sample(n=25, random_state=42)
sample_multi = df_gt[df_gt['match_count'] >= 3].sample(n=20, random_state=42)
sampled_gt_df = pd.concat([sample_1, sample_2, sample_multi])

needed_s1_ids = set(sampled_gt_df['source1_entity_id'])
needed_s2_ids = set()
needed_s3_ids = set()
for matches in sampled_gt_df['matches_list']:
    for m in matches:
        if m.startswith('S2-'):
            needed_s2_ids.add(m)
        elif m.startswith('S3-'):
            needed_s3_ids.add(m)

print(f"Sampled GT subset: {len(sampled_gt_df)} S1 entities, requiring {len(needed_s2_ids)} S2 IDs and {len(needed_s3_ids)} S3 IDs.")

# -------------------------------------------------------------
# TASK 6 Helpers (vectorized)
# -------------------------------------------------------------
def get_field_stats(series, source_name, field_name):
    clean_series = series.fillna('').astype(str)
    char_lens = clean_series.str.len()
    # Fast token count: number of spaces + 1 for non-empty, 0 for empty
    tok_counts = np.where(char_lens == 0, 0, clean_series.str.count(' ') + 1)
    
    stats = {
        'Source': source_name,
        'Field': field_name,
        'Char Mean': float(char_lens.mean()),
        'Char Median': float(char_lens.median()),
        'Char Min': int(char_lens.min()),
        'Char Max': int(char_lens.max()),
        'Token Mean': float(tok_counts.mean()),
        'Token Median': float(np.median(tok_counts)),
        'Token Min': int(tok_counts.min()),
        'Token Max': int(tok_counts.max()),
    }
    return stats, char_lens.values, tok_counts

# Compute S1 Stats
s1_name_stats, s1_name_len, s1_name_tok = get_field_stats(df_s1['business_name'], 'Source 1', 'business_name')
s1_addr_stats, s1_addr_len, s1_addr_tok = get_field_stats(df_s1['business_address'], 'Source 1', 'business_address')
s1_country_vc = df_s1['country'].value_counts(dropna=False)

# Extract sampled S1 records
s1_sampled_recs = df_s1[df_s1['entity_id'].isin(needed_s1_ids)].set_index('entity_id').to_dict('index')

# Free S1 large dataframe
del df_s1
gc.collect()

# -------------------------------------------------------------
# TASK 1, 2h, 2i, 3, 6 for Source 2
# -------------------------------------------------------------
print("\n--- Loading S2 and analyzing ---")
df_s2 = pd.read_csv(s2_path, sep='\t')
print(f"S2 Shape: {df_s2.shape}")
print("S2 Nulls:")
print(df_s2.isnull().sum())
print("S2 Duplicate IDs:", df_s2['entity_id'].duplicated().sum())
print("S2 Prefix Valid:", df_s2['entity_id'].str.startswith('S2-').all())

s2_all_ids = set(df_s2['entity_id'])
orphaned_s2 = set_s2_matched - s2_all_ids
print(f"h) Orphaned S2 references in GT: {len(orphaned_s2)}")

s2_referenced_count = len(s2_all_ids.intersection(set_s2_matched))
s2_noise_count = len(df_s2) - s2_referenced_count
print(f"i) S2 records referenced in GT: {s2_referenced_count:,} ({s2_referenced_count/len(df_s2)*100:.2f}%)")
print(f"   S2 noise records (unreferenced): {s2_noise_count:,} ({s2_noise_count/len(df_s2)*100:.2f}%)")

s2_country_vc = df_s2['country'].value_counts(dropna=False)
s2_name_stats, s2_name_len, s2_name_tok = get_field_stats(df_s2['business_name'], 'Source 2', 'business_name')
s2_addr_stats, s2_addr_len, s2_addr_tok = get_field_stats(df_s2['business_address'], 'Source 2', 'business_address')

# Sample unreferenced S2
unref_s2_ids = list(s2_all_ids - set_s2_matched)
np.random.seed(42)
sampled_unref_s2_ids = np.random.choice(unref_s2_ids, size=15, replace=False)

# Extract needed S2 records
s2_sampled_recs = df_s2[df_s2['entity_id'].isin(needed_s2_ids)].set_index('entity_id').to_dict('index')
s2_unref_recs = df_s2[df_s2['entity_id'].isin(sampled_unref_s2_ids)].set_index('entity_id').to_dict('index')

del df_s2, unref_s2_ids
gc.collect()

# -------------------------------------------------------------
# TASK 1, 2h, 2i, 3, 6 for Source 3
# -------------------------------------------------------------
print("\n--- Loading S3 and analyzing ---")
df_s3 = pd.read_csv(s3_path, sep='\t')
print(f"S3 Shape: {df_s3.shape}")
print("S3 Nulls:")
print(df_s3.isnull().sum())
print("S3 Duplicate IDs:", df_s3['entity_id'].duplicated().sum())
print("S3 Prefix Valid:", df_s3['entity_id'].str.startswith('S3-').all())

s3_all_ids = set(df_s3['entity_id'])
orphaned_s3 = set_s3_matched - s3_all_ids
print(f"h) Orphaned S3 references in GT: {len(orphaned_s3)}")

s3_referenced_count = len(s3_all_ids.intersection(set_s3_matched))
s3_noise_count = len(df_s3) - s3_referenced_count
print(f"i) S3 records referenced in GT: {s3_referenced_count:,} ({s3_referenced_count/len(df_s3)*100:.2f}%)")
print(f"   S3 noise records (unreferenced): {s3_noise_count:,} ({s3_noise_count/len(df_s3)*100:.2f}%)")

s3_country_vc = df_s3['country'].value_counts(dropna=False)
s3_name_stats, s3_name_len, s3_name_tok = get_field_stats(df_s3['business_name'], 'Source 3', 'business_name')
s3_addr_stats, s3_addr_len, s3_addr_tok = get_field_stats(df_s3['business_address'], 'Source 3', 'business_address')

# Sample unreferenced S3
unref_s3_ids = list(s3_all_ids - set_s3_matched)
np.random.seed(42)
sampled_unref_s3_ids = np.random.choice(unref_s3_ids, size=15, replace=False)

s3_sampled_recs = df_s3[df_s3['entity_id'].isin(needed_s3_ids)].set_index('entity_id').to_dict('index')
s3_unref_recs = df_s3[df_s3['entity_id'].isin(sampled_unref_s3_ids)].set_index('entity_id').to_dict('index')

del df_s3, unref_s3_ids
gc.collect()

# -------------------------------------------------------------
# TASK 3: Country Visualizations & Tables
# -------------------------------------------------------------
print("\n--- TASK 3: Country Tables and Chart ---")
print("S1 Countries:\n", s1_country_vc)
print("S2 Countries:\n", s2_country_vc)
print("S3 Countries:\n", s3_country_vc)

country_df = pd.DataFrame({
    'Source 1': s1_country_vc,
    'Source 2': s2_country_vc,
    'Source 3': s3_country_vc
}).fillna(0).astype(int)

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
x = np.arange(len(country_df))
width = 0.25

ax.bar(x - width, country_df['Source 1'], width, label='Source 1 (Reference)', color='#4A90E2', edgecolor='#333333')
ax.bar(x, country_df['Source 2'], width, label='Source 2 (External)', color='#50E3C2', edgecolor='#333333')
ax.bar(x + width, country_df['Source 3'], width, label='Source 3 (External)', color='#F5A623', edgecolor='#333333')

ax.set_title('Country Distribution Across Training Sources', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(country_df.index, fontsize=11, fontweight='semibold')
ax.set_ylabel('Record Count', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, loc: f"{int(y):,}"))
ax.legend(frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('notebooks/eda_assets/country_distribution.png')
plt.close()
print("Saved notebooks/eda_assets/country_distribution.png")

# -------------------------------------------------------------
# TASK 6: Length and Token Distribution Plots & Summary Table
# -------------------------------------------------------------
print("\n--- TASK 6: Plotting Length & Token Statistics ---")
stats_table = pd.DataFrame([
    s1_name_stats, s2_name_stats, s3_name_stats,
    s1_addr_stats, s2_addr_stats, s3_addr_stats
])
print(stats_table.to_string(index=False))

# Save summary stats to CSV/JSON
stats_table.to_csv('notebooks/eda_assets/field_statistics.csv', index=False)

# Character Length Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=300)
axes[0, 0].hist(s1_name_len, bins=50, range=(0, 100), color='#4A90E2', alpha=0.85)
axes[0, 0].set_title('S1: Name Character Length', fontweight='bold')
axes[0, 0].set_ylabel('Frequency')
axes[0, 1].hist(s2_name_len, bins=50, range=(0, 100), color='#50E3C2', alpha=0.85)
axes[0, 1].set_title('S2: Name Character Length', fontweight='bold')
axes[0, 2].hist(s3_name_len, bins=50, range=(0, 100), color='#F5A623', alpha=0.85)
axes[0, 2].set_title('S3: Name Character Length', fontweight='bold')

axes[1, 0].hist(s1_addr_len, bins=50, range=(0, 250), color='#4A90E2', alpha=0.85)
axes[1, 0].set_title('S1: Address Character Length', fontweight='bold')
axes[1, 0].set_ylabel('Frequency')
axes[1, 1].hist(s2_addr_len, bins=50, range=(0, 250), color='#50E3C2', alpha=0.85)
axes[1, 1].set_title('S2: Address Character Length', fontweight='bold')
axes[1, 2].hist(s3_addr_len, bins=50, range=(0, 250), color='#F5A623', alpha=0.85)
axes[1, 2].set_title('S3: Address Character Length', fontweight='bold')

for row in axes:
    for ax in row:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, loc: f"{int(y):,}"))
        ax.set_xlabel('Character Count')

plt.suptitle('Business Name & Address Character Length Distributions', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('notebooks/eda_assets/length_distributions.png')
plt.close()
print("Saved notebooks/eda_assets/length_distributions.png")

# Token Count Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=300)
axes[0, 0].hist(s1_name_tok, bins=25, range=(0, 25), color='#4A90E2', alpha=0.85)
axes[0, 0].set_title('S1: Name Word Tokens', fontweight='bold')
axes[0, 0].set_ylabel('Frequency')
axes[0, 1].hist(s2_name_tok, bins=25, range=(0, 25), color='#50E3C2', alpha=0.85)
axes[0, 1].set_title('S2: Name Word Tokens', fontweight='bold')
axes[0, 2].hist(s3_name_tok, bins=25, range=(0, 25), color='#F5A623', alpha=0.85)
axes[0, 2].set_title('S3: Name Word Tokens', fontweight='bold')

axes[1, 0].hist(s1_addr_tok, bins=35, range=(0, 35), color='#4A90E2', alpha=0.85)
axes[1, 0].set_title('S1: Address Word Tokens', fontweight='bold')
axes[1, 0].set_ylabel('Frequency')
axes[1, 1].hist(s2_addr_tok, bins=35, range=(0, 35), color='#50E3C2', alpha=0.85)
axes[1, 1].set_title('S2: Address Word Tokens', fontweight='bold')
axes[1, 2].hist(s3_addr_tok, bins=35, range=(0, 35), color='#F5A623', alpha=0.85)
axes[1, 2].set_title('S3: Address Word Tokens', fontweight='bold')

for row in axes:
    for ax in row:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, loc: f"{int(y):,}"))
        ax.set_xlabel('Token Count')

plt.suptitle('Business Name & Address Word Token Distributions', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('notebooks/eda_assets/token_distributions.png')
plt.close()
print("Saved notebooks/eda_assets/token_distributions.png")

# Free memory of arrays
del s1_name_len, s1_name_tok, s1_addr_len, s1_addr_tok
del s2_name_len, s2_name_tok, s2_addr_len, s2_addr_tok
del s3_name_len, s3_name_tok, s3_addr_len, s3_addr_tok
gc.collect()

# -------------------------------------------------------------
# TASK 4 & 5: Assemble Sampled Pairs & JSON
# -------------------------------------------------------------
print("\n--- TASK 4 & 5: Assembling Sampled Pairs ---")

samples_output = []
for idx, row in sampled_gt_df.iterrows():
    s1_id = row['source1_entity_id']
    s1 = s1_sampled_recs.get(s1_id)
    matches = row['matches_list']
    
    out_item = {
        's1_id': s1_id,
        's1_name': s1['business_name'] if s1 else None,
        's1_address': s1['business_address'] if s1 else None,
        's1_country': s1['country'] if s1 else None,
        'matches': []
    }
    for m_id in matches:
        if m_id.startswith('S2-'):
            m_rec = s2_sampled_recs.get(m_id)
        elif m_id.startswith('S3-'):
            m_rec = s3_sampled_recs.get(m_id)
        else:
            m_rec = None
        
        out_item['matches'].append({
            'matched_id': m_id,
            'matched_name': m_rec['business_name'] if m_rec else None,
            'matched_address': m_rec['business_address'] if m_rec else None,
            'matched_country': m_rec['country'] if m_rec else None,
        })
    samples_output.append(out_item)

with open('notebooks/eda_assets/sampled_pairs.json', 'w') as f:
    json.dump(samples_output, f, indent=2)

print(f"Saved {len(samples_output)} sampled S1 entities with all true matches to notebooks/eda_assets/sampled_pairs.json")

# -------------------------------------------------------------
# TASK 7: Unreferenced Samples JSON
# -------------------------------------------------------------
unref_samples = []
for eid in sampled_unref_s2_ids:
    rec = s2_unref_recs.get(eid)
    unref_samples.append({
        'entity_id': eid,
        'source': 'Source 2',
        'business_name': rec['business_name'] if rec else None,
        'business_address': rec['business_address'] if rec else None,
        'country': rec['country'] if rec else None
    })

for eid in sampled_unref_s3_ids:
    rec = s3_unref_recs.get(eid)
    unref_samples.append({
        'entity_id': eid,
        'source': 'Source 3',
        'business_name': rec['business_name'] if rec else None,
        'business_address': rec['business_address'] if rec else None,
        'country': rec['country'] if rec else None
    })

with open('notebooks/eda_assets/unreferenced_samples.json', 'w') as f:
    json.dump(unref_samples, f, indent=2)

print(f"Saved {len(unref_samples)} unreferenced noise records to notebooks/eda_assets/unreferenced_samples.json")
print("\n--- ALL EDA PROCESSING COMPLETED SUCCESSFULLY ---")
