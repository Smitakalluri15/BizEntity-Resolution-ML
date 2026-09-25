"""
End-to-end feature assembly and ground-truth label generation for Business Entity Resolution.
"""

from typing import Optional, Dict, Set, Union, List, Tuple
import os
import time
import numpy as np
import pandas as pd

from src.features.name_features import compute_name_features
from src.features.address_features import compute_address_features
from src.features.tfidf_features import fit_tfidf_vectorizers, compute_tfidf_features
from src.features.cross_features import compute_cross_features


def join_candidate_data(
    candidate_pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins candidate pairs with normalized entity records from S1, S2, and S3.
    
    Accepts candidate_pairs_df in either:
      1. Exploded format: columns ['source1_entity_id', 'candidate_entity_id']
      2. List/comma format: columns ['source1_entity_id', 'candidate_entity_ids']
    
    Prefixes all source 1 columns with 's1_' and all candidate columns with 'cand_'.
    """
    # 1. Standardize candidate pairs to exploded pairs (source1_entity_id, candidate_entity_id)
    if 'candidate_entity_id' not in candidate_pairs_df.columns:
        if 'candidate_entity_ids' in candidate_pairs_df.columns:
            # Explode comma-separated strings or lists
            pairs = []
            for _, row in candidate_pairs_df.iterrows():
                s1_id = row['source1_entity_id']
                raw_cands = row['candidate_entity_ids']
                if isinstance(raw_cands, str) and raw_cands.strip():
                    c_list = [c.strip() for c in raw_cands.split(',') if c.strip()]
                elif isinstance(raw_cands, (list, set, tuple)):
                    c_list = [c for c in raw_cands if c]
                else:
                    c_list = []
                for cid in c_list:
                    pairs.append({'source1_entity_id': s1_id, 'candidate_entity_id': cid})
            pairs_df = pd.DataFrame(pairs)
        else:
            raise ValueError("candidate_pairs_df must contain 'candidate_entity_id' or 'candidate_entity_ids' column.")
    else:
        pairs_df = candidate_pairs_df[['source1_entity_id', 'candidate_entity_id']].copy()

    if pairs_df.empty:
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id'])

    # 2. Prepare normalized lookup tables
    s1_key = 'entity_id' if 'entity_id' in s1_df.columns else 'source1_entity_id'
    s1_lookup = s1_df.set_index(s1_key)
    s1_renamed = s1_lookup.add_prefix('s1_')

    # Combine S2 and S3 for fast single-index candidate lookup
    s2_key = 'entity_id' if 'entity_id' in s2_df.columns else 'candidate_entity_id'
    s3_key = 'entity_id' if 'entity_id' in s3_df.columns else 'candidate_entity_id'
    s2_indexed = s2_df.set_index(s2_key)
    s3_indexed = s3_df.set_index(s3_key)
    
    # Combined candidates lookup
    cand_lookup = pd.concat([s2_indexed, s3_indexed])
    # Deduplicate candidate index in case of any overlaps
    cand_lookup = cand_lookup[~cand_lookup.index.duplicated(keep='first')]
    cand_renamed = cand_lookup.add_prefix('cand_')

    # 3. Join S1 columns
    joined = pairs_df.join(s1_renamed, on='source1_entity_id', how='left')

    # 4. Join Candidate columns
    joined = joined.join(cand_renamed, on='candidate_entity_id', how='left')

    return joined


def build_features(
    candidate_pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    vectorizers: Optional[Dict] = None,
    output_path: Optional[str] = None,
    blocking_metadata: Optional[Dict[tuple, str]] = None
) -> pd.DataFrame:
    """
    Assembles the complete feature set for candidate pairs:
      1. Joins candidate pairs with normalized records
      2. Computes name similarity features (Levenshtein, token Jaccard, char trigrams, phonetic similarity, etc.)
      3. Computes address & postal features (Levenshtein, token Jaccard, postal exact match, landmark flags)
      4. Computes sparse TF-IDF word and character cosine similarities
      5. Computes cross/interaction, group context rankings, and retention tier metadata
      6. Validates 0 unexpected NaNs
      7. Saves to parquet if output_path is specified

    Returns the assembled DataFrame with all identifiers and feature columns.
    """
    t0 = time.time()
    
    # 1. Join Base Data
    joined_df = join_candidate_data(candidate_pairs_df, s1_df, s2_df, s3_df)
    if joined_df.empty:
        return joined_df

    # 2. Name Features
    name_feats = compute_name_features(joined_df)

    # 3. Address Features
    addr_feats = compute_address_features(joined_df)

    # 4. TF-IDF Features
    if vectorizers is None:
        # Fit vectorizers on unique names and addresses from S1, S2, S3
        all_names = pd.concat([
            s1_df['name_norm'] if 'name_norm' in s1_df.columns else pd.Series([]),
            s2_df['name_norm'] if 'name_norm' in s2_df.columns else pd.Series([]),
            s3_df['name_norm'] if 'name_norm' in s3_df.columns else pd.Series([])
        ]).dropna().unique()

        all_addrs = pd.concat([
            s1_df['address_norm'] if 'address_norm' in s1_df.columns else pd.Series([]),
            s2_df['address_norm'] if 'address_norm' in s2_df.columns else pd.Series([]),
            s3_df['address_norm'] if 'address_norm' in s3_df.columns else pd.Series([])
        ]).dropna().unique()

        vectorizers = fit_tfidf_vectorizers(all_names, all_addrs)

    tfidf_feats = compute_tfidf_features(joined_df, vectorizers=vectorizers)

    # 5. Cross & Group Context Features
    cross_feats = compute_cross_features(
        joined_df,
        name_features_df=name_feats,
        address_features_df=addr_feats,
        blocking_metadata=blocking_metadata
    )

    # 6. Assemble All Columns
    feature_blocks = [
        joined_df[['source1_entity_id', 'candidate_entity_id']],
        name_feats,
        addr_feats,
        tfidf_feats,
        cross_feats
    ]

    final_df = pd.concat(feature_blocks, axis=1)

    # 7. Null Checking & Verification
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    nan_counts = final_df[numeric_cols].isna().sum()
    unexplained_nans = nan_counts[nan_counts > 0]
    
    if not unexplained_nans.empty:
        raise ValueError(f"Unexplained NaNs found in numeric feature columns: {unexplained_nans.to_dict()}")

    # 8. Save to parquet if requested
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        final_df.to_parquet(output_path, index=False)

    return final_df


def assemble_labels(
    candidate_features_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    s1_ids_filter: Optional[Set[str]] = None
) -> pd.DataFrame:
    """
    Labels candidate pairs with ground-truth binary targets:
      - is_true_match: 1 if (source1_entity_id, candidate_entity_id) is a true match, else 0
      
    If s1_ids_filter is provided (e.g. train_s1_ids), evaluates labels exclusively on that subset.
    Reports class balance and matches found vs ground truth.
    """
    out_df = candidate_features_df.copy()

    # Parse ground truth into a set of (source1_entity_id, matched_id) pairs
    true_pairs = set()
    for _, row in ground_truth_df.iterrows():
        s1_id = str(row['source1_entity_id'])
        if s1_ids_filter is not None and s1_id not in s1_ids_filter:
            continue
            
        raw_matches = row['matched_entity_ids']
        if isinstance(raw_matches, str) and raw_matches.strip():
            matches = [m.strip() for m in raw_matches.split(',') if m.strip()]
        elif isinstance(raw_matches, (list, set, tuple)):
            matches = [m for m in raw_matches if m]
        else:
            matches = []

        for m_id in matches:
            true_pairs.add((s1_id, str(m_id)))

    # Apply binary label
    s1_list = out_df['source1_entity_id'].astype(str).tolist()
    cand_list = out_df['candidate_entity_id'].astype(str).tolist()

    labels = np.array([
        1 if (s1, cand) in true_pairs else 0
        for s1, cand in zip(s1_list, cand_list)
    ], dtype=np.int32)

    out_df['is_true_match'] = labels

    return out_df
