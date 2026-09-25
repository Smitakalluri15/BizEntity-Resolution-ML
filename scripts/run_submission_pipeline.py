"""
Phase 8: End-to-End Submission Pipeline for Amazon ML Challenge 2026.
Produces:
  1. output/matching_results.tsv
  2. output/candidate_pairs.tsv
  3. Validates format against competition rules
"""

import os
import gc
import time
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.normalize.name import apply_normalization_batch
from src.normalize.phonetic import apply_phonetic_features_dataframe
from src.block.name_blocking import build_token_index, build_sorted_neighborhood_partitions
from src.block.phonetic_blocking import build_phonetic_index
from src.block.address_blocking import build_postal_index, build_city_token_index
from src.block.combine import generate_candidates
from src.features.tfidf_features import fit_tfidf_vectorizers
from src.features.build_features import build_features
from src.model.predict import predict_match_probabilities, assemble_entity_predictions
from src.model.train import DEFAULT_FEATURE_COLUMNS


def parse_args():
    parser = argparse.ArgumentParser(description="Run full-scale or sample submission inference.")
    parser.add_argument("--test_dir", type=str, default="dataset/test", help="Path to test datasets directory")
    parser.add_argument("--model_path", type=str, default="models/phase7_matcher_sample.txt", help="Path to LightGBM model")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save submission TSVs")
    parser.add_argument("--chunk_size", type=int, default=50000, help="Chunk size for S1 streaming")
    parser.add_argument("--threshold", type=float, default=0.60, help="Optimal decision threshold")
    parser.add_argument("--max_s1_records", type=int, default=None, help="Optional limit on S1 records for testing")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K candidate cap per entity")
    return parser.parse_args()


def load_and_normalize_source(file_path: str, source_name: str) -> pd.DataFrame:
    print(f"[{time.strftime('%H:%M:%S')}] Loading {source_name} from {file_path}...")
    t0 = time.time()
    df = pd.read_csv(file_path, sep='\t', dtype=str).fillna('')
    print(f"   Loaded {len(df):,} rows in {time.time()-t0:.2f}s.")
    
    print(f"[{time.strftime('%H:%M:%S')}] Normalizing {source_name}...")
    t0 = time.time()
    df_norm = apply_normalization_batch(df)
    df_norm = apply_phonetic_features_dataframe(df_norm)
    print(f"   Normalized {len(df_norm):,} rows in {time.time()-t0:.2f}s.")
    return df_norm


def build_candidate_indexes(s2_df: pd.DataFrame, s3_df: pd.DataFrame):
    print(f"\n[{time.strftime('%H:%M:%S')}] Building Unified Candidate Inverted Indexes (S2 + S3)...")
    t0 = time.time()
    
    # Combined lookup mapping
    entity_country_map = {}
    entity_name_map = {}
    entity_phonetic_map = {}
    
    for df in [s2_df, s3_df]:
        for _, row in df[['entity_id', 'country_norm', 'name_norm', 'name_phonetic']].iterrows():
            eid = row['entity_id']
            entity_country_map[eid] = row['country_norm']
            entity_name_map[eid] = row['name_norm']
            entity_phonetic_map[eid] = row['name_phonetic']

    token_index, stop_tokens = build_token_index(s2_df, s3_df, max_frequency=10000)
    phonetic_index = build_phonetic_index(s2_df, s3_df)
    postal_index = build_postal_index(s2_df, s3_df)
    city_token_index, stop_addr_tokens = build_city_token_index(s2_df, s3_df, max_frequency=5000)
    sn_partitions = build_sorted_neighborhood_partitions(s2_df, s3_df)
    
    print(f"   Index construction complete in {time.time()-t0:.2f}s.")
    return {
        'token_index': token_index,
        'stop_tokens': stop_tokens,
        'phonetic_index': phonetic_index,
        'postal_index': postal_index,
        'city_token_index': city_token_index,
        'stop_addr_tokens': stop_addr_tokens,
        'sn_partitions': sn_partitions,
        'entity_country_map': entity_country_map,
        'entity_name_map': entity_name_map,
        'entity_phonetic_map': entity_phonetic_map
    }


def main():
    args = parse_args()
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 — SUBMISSION INFERENCE PIPELINE")
    print(f"Test Directory: {args.test_dir}")
    print(f"Model Path:     {args.model_path}")
    print(f"Output Dir:     {args.output_dir}")
    print(f"Threshold:      {args.threshold:.2f}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    matching_out_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_out_path = os.path.join(args.output_dir, "candidate_pairs.tsv")

    # 1. Load Model
    print(f"\n1. Loading LightGBM Model from {args.model_path}...")
    model = lgb.Booster(model_file=args.model_path)
    print("   Model loaded successfully.")

    # 2. Load & Normalize S2 and S3
    s2_file = os.path.join(args.test_dir, "test_source2.tsv")
    s3_file = os.path.join(args.test_dir, "test_source3.tsv")
    
    s2_norm = load_and_normalize_source(s2_file, "Source 2")
    s3_norm = load_and_normalize_source(s3_file, "Source 3")

    # 3. Build Inverted Blocking Indexes
    indexes = build_candidate_indexes(s2_norm, s3_norm)

    # 4. Fit TF-IDF Vectorizers
    print(f"\n[{time.strftime('%H:%M:%S')}] Fitting TF-IDF Vectorizers...")
    t0 = time.time()
    sample_names = pd.concat([s2_norm['name_norm'], s3_norm['name_norm']]).sample(min(100000, len(s2_norm)+len(s3_norm)), random_state=42)
    sample_addrs = pd.concat([s2_norm['address_norm'], s3_norm['address_norm']]).sample(min(100000, len(s2_norm)+len(s3_norm)), random_state=42)
    vectorizers = fit_tfidf_vectorizers(sample_names, sample_addrs)
    print(f"   TF-IDF vectorizers fitted in {time.time()-t0:.2f}s.")

    # 5. Initialize Output Files with Headers
    with open(matching_out_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        
    with open(candidate_out_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

    # 6. Stream S1 and Perform Inference Chunk by Chunk
    s1_file = os.path.join(args.test_dir, "test_source1.tsv")
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting S1 Streaming Inference from {s1_file}...")
    
    s1_reader = pd.read_csv(s1_file, sep='\t', dtype=str, chunksize=args.chunk_size)
    
    total_s1_processed = 0
    total_candidates_generated = 0
    total_matches_predicted = 0
    singletons_predicted = 0
    chunk_idx = 0

    t_start = time.time()

    for s1_chunk_raw in s1_reader:
        chunk_idx += 1
        s1_chunk_raw = s1_chunk_raw.fillna('')
        
        if args.max_s1_records and total_s1_processed >= args.max_s1_records:
            break
            
        if args.max_s1_records and total_s1_processed + len(s1_chunk_raw) > args.max_s1_records:
            s1_chunk_raw = s1_chunk_raw.iloc[:args.max_s1_records - total_s1_processed]

        t_chunk = time.time()
        # Normalize S1 chunk
        s1_chunk = apply_normalization_batch(s1_chunk_raw)
        s1_chunk = apply_phonetic_features_dataframe(s1_chunk)

        # Candidate Generation
        cand_rows = []
        exploded_pairs = []
        
        for _, s1_row in s1_chunk.iterrows():
            s1_id = s1_row['entity_id']
            cands = generate_candidates(
                s1_row=s1_row,
                token_index=indexes['token_index'],
                phonetic_index=indexes['phonetic_index'],
                postal_index=indexes['postal_index'],
                city_token_index=indexes['city_token_index'],
                sorted_neighborhood_partitions=indexes['sn_partitions'],
                stop_tokens=indexes['stop_tokens'],
                stop_address_tokens=indexes['stop_addr_tokens'],
                entity_country_map=indexes['entity_country_map'],
                entity_name_map=indexes['entity_name_map'],
                entity_phonetic_map=indexes['entity_phonetic_map'],
                top_k=args.top_k,
                window_size=10
            )
            c_list = sorted(list(cands))
            cand_rows.append((s1_id, ','.join(c_list)))
            for cid in c_list:
                exploded_pairs.append({'source1_entity_id': s1_id, 'candidate_entity_id': cid})

        total_candidates_generated += len(exploded_pairs)

        # Append candidate pairs to candidate_pairs.tsv
        with open(candidate_out_path, 'a', encoding='utf-8') as f:
            for s1_id, c_str in cand_rows:
                f.write(f"{s1_id}\t{c_str}\n")

        # Feature Extraction & Prediction
        if exploded_pairs:
            pairs_df = pd.DataFrame(exploded_pairs)
            features_df = build_features(
                candidate_pairs_df=pairs_df,
                s1_df=s1_chunk,
                s2_df=s2_norm,
                s3_df=s3_norm,
                vectorizers=vectorizers
            )
            
            # Predict match probabilities
            features_df['match_probability'] = predict_match_probabilities(
                model=model,
                features_df=features_df,
                feature_columns=DEFAULT_FEATURE_COLUMNS
            )

            # Assemble predictions
            all_s1_chunk_ids = s1_chunk['entity_id'].tolist()
            pred_df = assemble_entity_predictions(
                candidate_predictions_df=features_df,
                all_s1_ids=all_s1_chunk_ids,
                threshold=args.threshold,
                mode="multi"
            )
        else:
            # All singletons in this chunk
            pred_df = pd.DataFrame({
                'source1_entity_id': s1_chunk['entity_id'].tolist(),
                'matched_entity_ids': [''] * len(s1_chunk)
            })

        # Append predictions to matching_results.tsv
        with open(matching_out_path, 'a', encoding='utf-8') as f:
            for _, r in pred_df.iterrows():
                m_str = str(r['matched_entity_ids']) if pd.notna(r['matched_entity_ids']) else ''
                f.write(f"{r['source1_entity_id']}\t{m_str}\n")
                if not m_str.strip():
                    singletons_predicted += 1
                else:
                    total_matches_predicted += len([m for m in m_str.split(',') if m.strip()])

        total_s1_processed += len(s1_chunk)
        elapsed = time.time() - t_start
        rate = total_s1_processed / max(1, elapsed)
        print(f"[{time.strftime('%H:%M:%S')}] Chunk {chunk_idx:03d} | Processed: {total_s1_processed:,} S1 entities | Rate: {rate:,.1f} entities/sec | Chunk time: {time.time()-t_chunk:.2f}s")
        gc.collect()

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE — SUMMARY")
    print("=" * 80)
    print(f"Total S1 Entities Processed:    {total_s1_processed:,}")
    print(f"Total Candidates Generated:      {total_candidates_generated:,} ({total_candidates_generated/max(1, total_s1_processed):.2f} / S1)")
    print(f"Total Matches Predicted:         {total_matches_predicted:,}")
    print(f"Singletons (Empty Matches):      {singletons_predicted:,} ({singletons_predicted/max(1, total_s1_processed)*100:.2f}%)")
    print(f"Outputs written to:              {matching_out_path} and {candidate_out_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
