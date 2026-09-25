"""Full-scale benchmark for apply_normalization across training sources."""

import time
import gc
import pandas as pd
from src.normalize import apply_normalization

print("="*60)
print("PHASE 2: FULL-SCALE NORMALIZATION RUNTIME PROFILING")
print("="*60)

# 1. Profile Source 1 (Reference: 2.2M rows)
print("\n--- Benchmarking Source 1 (2.2M rows) ---")
t0 = time.time()
df_s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
t_load_s1 = time.time() - t0
print(f"Loaded S1 in {t_load_s1:.2f}s | Shape: {df_s1.shape}")

t0 = time.time()
df_s1_norm = apply_normalization(df_s1)
t_norm_s1 = time.time() - t0
print(f"Normalized S1 ({len(df_s1):,} rows) in {t_norm_s1:.2f}s | Throughput: {len(df_s1)/t_norm_s1:,.0f} rows/sec")
print(f"Output columns: {list(df_s1_norm.columns)}")

del df_s1, df_s1_norm
gc.collect()

# 2. Profile Source 2 (External: 5.03M rows)
print("\n--- Benchmarking Source 2 (5.03M rows) ---")
t0 = time.time()
df_s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
t_load_s2 = time.time() - t0
print(f"Loaded S2 in {t_load_s2:.2f}s | Shape: {df_s2.shape}")

t0 = time.time()
df_s2_norm = apply_normalization(df_s2)
t_norm_s2 = time.time() - t0
print(f"Normalized S2 ({len(df_s2):,} rows) in {t_norm_s2:.2f}s | Throughput: {len(df_s2)/t_norm_s2:,.0f} rows/sec")

del df_s2, df_s2_norm
gc.collect()

# 3. Profile Source 3 (External: 5.28M rows)
print("\n--- Benchmarking Source 3 (5.28M rows) ---")
t0 = time.time()
df_s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
t_load_s3 = time.time() - t0
print(f"Loaded S3 in {t_load_s3:.2f}s | Shape: {df_s3.shape}")

t0 = time.time()
df_s3_norm = apply_normalization(df_s3)
t_norm_s3 = time.time() - t0
print(f"Normalized S3 ({len(df_s3):,} rows) in {t_norm_s3:.2f}s | Throughput: {len(df_s3)/t_norm_s3:,.0f} rows/sec")

del df_s3, df_s3_norm
gc.collect()

print("\n" + "="*60)
print(f"SUMMARY OF WALL-CLOCK NORMALIZATION RUNTIMES:")
print(f"  Source 1 (2.2M rows): {t_norm_s1:.2f}s ({len(df_s1) if 'df_s1' in locals() else 2206821 / t_norm_s1:,.0f} rows/sec)")
print(f"  Source 2 (5.03M rows): {t_norm_s2:.2f}s")
print(f"  Source 3 (5.28M rows): {t_norm_s3:.2f}s")
print(f"  Total pipeline time for ~12.5M rows: {t_norm_s1 + t_norm_s2 + t_norm_s3:.2f}s ({(2206821+5034616+5285603)/(t_norm_s1 + t_norm_s2 + t_norm_s3):,.0f} rows/sec)")
print("="*60)
