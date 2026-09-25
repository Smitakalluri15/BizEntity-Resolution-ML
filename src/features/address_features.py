"""
Address and postal similarity features for candidate pairs in Business Entity Resolution.
"""

from typing import List, Set, Union
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.normalize.address import get_address_char_ngrams


def _fast_address_token_jaccard(toks1: list, toks2: list) -> float:
    s1 = set(toks1) if isinstance(toks1, (list, tuple, set)) else set()
    s2 = set(toks2) if isinstance(toks2, (list, tuple, set)) else set()
    inter = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return (float(inter) / float(union)) if union > 0 else 0.0


def _fast_address_char_trigram_jaccard(addr1: str, addr2: str) -> float:
    if not addr1 and not addr2:
        return 1.0
    if not addr1 or not addr2:
        return 0.0
    ng1 = get_address_char_ngrams(addr1, n=3)
    ng2 = get_address_char_ngrams(addr2, n=3)
    if not ng1 and not ng2:
        return 1.0
    if not ng1 or not ng2:
        return 0.0
    inter = len(ng1.intersection(ng2))
    union = len(ng1.union(ng2))
    return float(inter) / float(union) if union > 0 else 0.0


def compute_address_features(joined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all address-based similarity features for candidate pairs:
      - address_exact_norm: 1 if normalized addresses are identical, else 0
      - address_levenshtein_ratio: normalized Levenshtein ratio [0.0, 1.0]
      - address_jaccard_tokens: Jaccard similarity of address token sets
      - address_char_trigram_jaccard: Jaccard similarity of character trigrams
      - postal_available: 1 if BOTH records have a non-null/non-empty postal code, else 0
      - postal_exact_match: 1 if postal_available==1 and postal codes match, else 0
      - landmark_flag_either: 1 if either record is a landmark address, else 0

    Operates vectorially and handles missing address / postal fields cleanly.
    """
    out = pd.DataFrame(index=joined_df.index)

    s1_addr = joined_df['s1_address_norm'].fillna('').astype(str).tolist()
    cand_addr = joined_df['cand_address_norm'].fillna('').astype(str).tolist()

    # 1. Exact normalized address match
    out['address_exact_norm'] = np.array([
        1 if a == b and a != '' else 0
        for a, b in zip(s1_addr, cand_addr)
    ], dtype=np.int32)

    # 2. Levenshtein ratio on address_norm
    out['address_levenshtein_ratio'] = np.array([
        (fuzz.ratio(a, b) / 100.0) if (a or b) else 0.0
        for a, b in zip(s1_addr, cand_addr)
    ], dtype=np.float32)

    # 3. Token Jaccard
    s1_toks = joined_df['s1_address_tokens'].tolist() if 's1_address_tokens' in joined_df.columns else [[]] * len(joined_df)
    cand_toks = joined_df['cand_address_tokens'].tolist() if 'cand_address_tokens' in joined_df.columns else [[]] * len(joined_df)

    out['address_jaccard_tokens'] = np.array([
        _fast_address_token_jaccard(t1, t2)
        for t1, t2 in zip(s1_toks, cand_toks)
    ], dtype=np.float32)

    # 4. Character trigram Jaccard
    out['address_char_trigram_jaccard'] = np.array([
        _fast_address_char_trigram_jaccard(a, b)
        for a, b in zip(s1_addr, cand_addr)
    ], dtype=np.float32)

    # 5. Postal code match and availability
    # We must distinguish between:
    #   (a) Both postal codes present and matching -> postal_available=1, postal_exact_match=1
    #   (b) Both postal codes present and differing -> postal_available=1, postal_exact_match=0
    #   (c) One or both missing -> postal_available=0, postal_exact_match=0
    s1_postal = joined_df['s1_postal_code'].fillna('').astype(str).tolist() if 's1_postal_code' in joined_df.columns else [''] * len(joined_df)
    cand_postal = joined_df['cand_postal_code'].fillna('').astype(str).tolist() if 'cand_postal_code' in joined_df.columns else [''] * len(joined_df)

    postal_avail = []
    postal_match = []
    for p1, p2 in zip(s1_postal, cand_postal):
        p1_clean = p1.strip()
        p2_clean = p2.strip()
        if p1_clean and p2_clean and p1_clean.lower() != 'none' and p2_clean.lower() != 'none' and p1_clean.lower() != 'nan' and p2_clean.lower() != 'nan':
            postal_avail.append(1)
            postal_match.append(1 if p1_clean == p2_clean else 0)
        else:
            postal_avail.append(0)
            postal_match.append(0)

    out['postal_available'] = np.array(postal_avail, dtype=np.int32)
    out['postal_exact_match'] = np.array(postal_match, dtype=np.int32)

    # 6. Landmark flag either side
    s1_lm = joined_df['s1_is_landmark_address'].fillna(False).astype(bool).tolist() if 's1_is_landmark_address' in joined_df.columns else [False] * len(joined_df)
    cand_lm = joined_df['cand_is_landmark_address'].fillna(False).astype(bool).tolist() if 'cand_is_landmark_address' in joined_df.columns else [False] * len(joined_df)

    out['landmark_flag_either'] = np.array([
        1 if (l1 or l2) else 0
        for l1, l2 in zip(s1_lm, cand_lm)
    ], dtype=np.int32)

    return out
