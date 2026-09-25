"""
Cross, interaction, group context, and tiered retention features for candidate pairs.
"""

from typing import Optional, Dict, Set, Any
import numpy as np
import pandas as pd


def compute_cross_features(
    joined_df: pd.DataFrame,
    name_features_df: pd.DataFrame,
    address_features_df: pd.DataFrame,
    score_col: str = 'name_levenshtein_ratio',
    blocking_metadata: Optional[Dict[tuple, str]] = None
) -> pd.DataFrame:
    """
    Computes cross/interaction, group-context, and retention tier features:
      - country_match: 1 if s1_country_norm == cand_country_norm, else 0
      - name_address_interaction: name_jaccard_tokens * address_jaccard_tokens
      - candidate_rank_in_group: 1-based rank of candidate within S1 entity group by score_col desc
      - candidate_set_size: number of candidates associated with this S1 entity
      - top1_gap: difference between rank 1 and rank 2 score in the S1 candidate group
      - source_flag: 'S2' or 'S3' based on candidate entity ID prefix
      - candidate_source_id: 0 for S2, 1 for S3 (numeric representation)
      - candidate_source_strategy: blocking strategy provenance (e.g. 'token', 'postal', 'phonetic')
      - is_exact_tier: 1 if candidate surfaced via exact token or exact postal match, else 0
    """
    out = pd.DataFrame(index=joined_df.index)

    # 1. Country match
    s1_country = joined_df['s1_country_norm'].fillna('').astype(str).tolist() if 's1_country_norm' in joined_df.columns else [''] * len(joined_df)
    cand_country = joined_df['cand_country_norm'].fillna('').astype(str).tolist() if 'cand_country_norm' in joined_df.columns else [''] * len(joined_df)

    out['country_match'] = np.array([
        1 if (c1 == c2 and c1 != '') or (not c1 or not c2) else 0
        for c1, c2 in zip(s1_country, cand_country)
    ], dtype=np.int32)

    # 2. Name-Address interaction feature
    name_jaccard = name_features_df['name_jaccard_tokens'].to_numpy(dtype=np.float32)
    addr_jaccard = address_features_df['address_jaccard_tokens'].to_numpy(dtype=np.float32)
    out['name_address_interaction'] = (name_jaccard * addr_jaccard).astype(np.float32)

    # 3. Source flags
    cand_ids = joined_df['candidate_entity_id'].astype(str).tolist()
    source_flags = []
    source_ids = []
    for cid in cand_ids:
        c_upper = cid.upper()
        if c_upper.startswith('S3') or 'SOURCE3' in c_upper:
            source_flags.append('S3')
            source_ids.append(1)
        else:
            source_flags.append('S2')
            source_ids.append(0)

    out['source_flag'] = source_flags
    out['candidate_source_id'] = np.array(source_ids, dtype=np.int32)

    # 4. Group context features (candidate_rank_in_group, candidate_set_size, top1_gap)
    # Extract ranking score (e.g. name_levenshtein_ratio or blended score)
    if score_col in name_features_df.columns:
        scores = name_features_df[score_col].to_numpy(dtype=np.float32)
    elif score_col in joined_df.columns:
        scores = joined_df[score_col].to_numpy(dtype=np.float32)
    else:
        # Default blended score
        name_lev = name_features_df['name_levenshtein_ratio'].to_numpy(dtype=np.float32)
        addr_lev = address_features_df['address_levenshtein_ratio'].to_numpy(dtype=np.float32)
        scores = 0.7 * name_lev + 0.3 * addr_lev

    # Build group index mapping for fast vectorized ranking
    s1_ids = joined_df['source1_entity_id'].astype(str).tolist()
    
    # Pre-group indices
    group_indices = {}
    for idx, s1_id in enumerate(s1_ids):
        if s1_id not in group_indices:
            group_indices[s1_id] = []
        group_indices[s1_id].append(idx)

    ranks = np.zeros(len(joined_df), dtype=np.int32)
    sizes = np.zeros(len(joined_df), dtype=np.int32)
    gaps = np.zeros(len(joined_df), dtype=np.float32)

    for s1_id, idx_list in group_indices.items():
        k = len(idx_list)
        for idx in idx_list:
            sizes[idx] = k

        if k == 1:
            ranks[idx_list[0]] = 1
            # For singleton candidate, gap is defined as score itself (or 1.0)
            gaps[idx_list[0]] = float(scores[idx_list[0]])
        else:
            # Sort indices within group by score descending
            sorted_by_score = sorted(idx_list, key=lambda i: scores[i], reverse=True)
            top1_score = scores[sorted_by_score[0]]
            top2_score = scores[sorted_by_score[1]]
            group_gap = float(top1_score - top2_score)

            for rank_0, idx in enumerate(sorted_by_score):
                ranks[idx] = rank_0 + 1
                gaps[idx] = group_gap

    out['candidate_rank_in_group'] = ranks
    out['candidate_set_size'] = sizes
    out['top1_gap'] = gaps

    # 5. Tiered retention strategy provenance (Task 6)
    # If blocking metadata is provided as (s1_id, cand_id) -> strategy string:
    strategies = []
    exact_tiers = []
    
    # We can also infer exact tier from exact token or postal match if metadata not present
    tok_exact = (name_features_df['name_token_overlap_count'] >= 1).to_numpy()
    post_exact = (address_features_df['postal_exact_match'] == 1).to_numpy()

    for i in range(len(joined_df)):
        pair_key = (s1_ids[i], cand_ids[i])
        strat = blocking_metadata.get(pair_key, '') if blocking_metadata else ''
        
        if not strat:
            # Infer strategy heuristically
            strat_parts = []
            if tok_exact[i]:
                strat_parts.append('token')
            if post_exact[i]:
                strat_parts.append('postal')
            if name_features_df['name_phonetic_similarity'].iloc[i] >= 0.5:
                strat_parts.append('phonetic')
            strat = '+'.join(strat_parts) if strat_parts else 'neighborhood_or_fuzzy'
        
        strategies.append(strat)
        
        # Exact tier is True if surfaced via exact token overlap or exact postal code
        is_exact = 1 if ('token' in strat or 'postal' in strat or tok_exact[i] or post_exact[i]) else 0
        exact_tiers.append(is_exact)

    out['candidate_source_strategy'] = strategies
    out['is_exact_tier'] = np.array(exact_tiers, dtype=np.int32)

    return out
