"""
Name similarity features for candidate pairs in Business Entity Resolution.
"""

from typing import List, Set, Union
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.normalize.name import get_name_char_ngrams
from src.normalize.phonetic import phonetic_similarity


def _fast_token_jaccard_and_overlap(toks1: list, toks2: list) -> tuple:
    s1 = set(toks1) if isinstance(toks1, (list, tuple, set)) else set()
    s2 = set(toks2) if isinstance(toks2, (list, tuple, set)) else set()
    inter = len(s1.intersection(s2))
    union = len(s1.union(s2))
    jaccard = (float(inter) / float(union)) if union > 0 else 0.0
    return jaccard, inter


def _fast_char_trigram_jaccard(name1: str, name2: str) -> float:
    if not name1 and not name2:
        return 1.0
    if not name1 or not name2:
        return 0.0
    ng1 = get_name_char_ngrams(name1, n=3)
    ng2 = get_name_char_ngrams(name2, n=3)
    if not ng1 and not ng2:
        return 1.0
    if not ng1 or not ng2:
        return 0.0
    inter = len(ng1.intersection(ng2))
    union = len(ng1.union(ng2))
    return float(inter) / float(union) if union > 0 else 0.0


def compute_name_features(joined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all name-based similarity features for candidate pairs:
      - name_exact_raw: 1 if raw names are identical, else 0
      - name_exact_norm: 1 if normalized names are identical, else 0
      - name_levenshtein_ratio: normalized Levenshtein ratio [0.0, 1.0]
      - name_jaccard_tokens: Jaccard similarity of name token sets
      - name_token_overlap_count: integer count of overlapping tokens
      - name_char_trigram_jaccard: Jaccard similarity of character trigrams
      - name_phonetic_similarity: cross-script phonetic similarity from Phase 2.5
      - name_script_match: 1 if both records have the same detected script, else 0

    Operates vectorially and in fast C loops.
    """
    out = pd.DataFrame(index=joined_df.index)

    # 1. Exact raw & norm matches
    if 's1_name_raw' in joined_df.columns and 'cand_name_raw' in joined_df.columns:
        s1_raw = joined_df['s1_name_raw'].fillna('')
        cand_raw = joined_df['cand_name_raw'].fillna('')
        out['name_exact_raw'] = (s1_raw == cand_raw).astype(np.int32)
    else:
        # Fallback to normalized names if raw not in columns
        out['name_exact_raw'] = (joined_df['s1_name_norm'].fillna('') == joined_df['cand_name_norm'].fillna('')).astype(np.int32)

    s1_norm = joined_df['s1_name_norm'].fillna('').astype(str).tolist()
    cand_norm = joined_df['cand_name_norm'].fillna('').astype(str).tolist()

    out['name_exact_norm'] = np.array([1 if a == b and a != '' else 0 for a, b in zip(s1_norm, cand_norm)], dtype=np.int32)

    # 2. Fast Levenshtein ratio via rapidfuzz
    out['name_levenshtein_ratio'] = np.array([
        (fuzz.ratio(a, b) / 100.0) if (a or b) else 0.0
        for a, b in zip(s1_norm, cand_norm)
    ], dtype=np.float32)

    # 3. Token Jaccard and overlap count
    s1_toks = joined_df['s1_name_tokens'].tolist() if 's1_name_tokens' in joined_df.columns else [[]] * len(joined_df)
    cand_toks = joined_df['cand_name_tokens'].tolist() if 'cand_name_tokens' in joined_df.columns else [[]] * len(joined_df)

    jaccards = []
    overlaps = []
    for t1, t2 in zip(s1_toks, cand_toks):
        jac, ovlp = _fast_token_jaccard_and_overlap(t1, t2)
        jaccards.append(jac)
        overlaps.append(ovlp)

    out['name_jaccard_tokens'] = np.array(jaccards, dtype=np.float32)
    out['name_token_overlap_count'] = np.array(overlaps, dtype=np.int32)

    # 4. Character trigram Jaccard
    out['name_char_trigram_jaccard'] = np.array([
        _fast_char_trigram_jaccard(a, b)
        for a, b in zip(s1_norm, cand_norm)
    ], dtype=np.float32)

    # 5. Cross-script phonetic similarity (using Phase 2.5 module)
    out['name_phonetic_similarity'] = np.array([
        phonetic_similarity(a, b)
        for a, b in zip(s1_norm, cand_norm)
    ], dtype=np.float32)

    # 6. Script match flag
    s1_script = joined_df['s1_name_script'].fillna('latin').astype(str).tolist() if 's1_name_script' in joined_df.columns else ['latin'] * len(joined_df)
    cand_script = joined_df['cand_name_script'].fillna('latin').astype(str).tolist() if 'cand_name_script' in joined_df.columns else ['latin'] * len(joined_df)
    out['name_script_match'] = np.array([
        1 if a == b else 0
        for a, b in zip(s1_script, cand_script)
    ], dtype=np.int32)

    return out
