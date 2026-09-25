import re
from collections import defaultdict, Counter
from typing import Dict, Set, List, Optional, Iterable, Tuple, Any
import pandas as pd
import numpy as np


def build_token_index(
    df: pd.DataFrame,
    name_col: str = "name_norm",
    token_col: Optional[str] = "name_tokens",
    id_col: str = "entity_id",
    country_col: Optional[str] = "country_norm",
    max_token_freq: float = 0.005,  # Exclude tokens appearing in >0.5% of records (~50K rows)
    min_token_len: int = 2
) -> Tuple[Dict[str, Dict[str, List[str]]], Set[str]]:
    """
    Builds a country-partitioned inverted token index mapping:
      country -> { token -> [entity_ids] }
      
    Also identifies and excludes overly frequent generic tokens (e.g. 'and', 'the', 'services')
    based on empirical corpus frequency.
    
    Returns:
        (token_index, stop_tokens_set)
    """
    n_records = len(df)
    max_count_threshold = max(500, int(n_records * max_token_freq))
    
    # 1. Count global token frequencies
    token_counter = Counter()
    has_token_col = token_col and token_col in df.columns
    
    if has_token_col:
        for tokens in df[token_col]:
            if isinstance(tokens, (list, set, tuple)):
                token_counter.update(tokens)
            elif isinstance(tokens, str) and tokens:
                # In case it's string representation of list or space-separated
                t_list = [t.strip(" '\"[]") for t in tokens.split(',') if len(t.strip(" '\"[]")) >= min_token_len]
                token_counter.update(t_list)
    else:
        for name in df[name_col].fillna(''):
            if name:
                tokens = [w for w in str(name).split() if len(w) >= min_token_len]
                token_counter.update(tokens)

    # Identify stop tokens exceeding threshold
    stop_tokens = {tok for tok, count in token_counter.items() if count > max_count_threshold or len(tok) < min_token_len}

    # 2. Build inverted index partitioned by country
    # country -> token -> list of entity_ids
    token_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    has_country = country_col and country_col in df.columns
    countries = df[country_col].fillna('').values if has_country else [''] * n_records
    ids = df[id_col].values
    
    if has_token_col:
        for eid, ctry, tokens in zip(ids, countries, df[token_col]):
            ctry_key = str(ctry).strip().lower() if ctry else '__any__'
            if isinstance(tokens, (list, set, tuple)):
                t_set = set(tokens)
            elif isinstance(tokens, str) and tokens:
                t_set = {t.strip(" '\"[]") for t in tokens.split(',') if len(t.strip(" '\"[]")) >= min_token_len}
            else:
                t_set = set()
                
            for tok in t_set:
                if tok and tok not in stop_tokens:
                    token_index[ctry_key][tok].append(eid)
    else:
        for eid, ctry, name in zip(ids, countries, df[name_col].fillna('')):
            ctry_key = str(ctry).strip().lower() if ctry else '__any__'
            if name:
                t_set = {w for w in str(name).split() if len(w) >= min_token_len and w not in stop_tokens}
                for tok in t_set:
                    token_index[ctry_key][tok].append(eid)

    return token_index, stop_tokens


def get_token_candidates(
    s1_name_tokens: Iterable[str],
    s1_country_norm: str,
    token_index: Dict[str, Dict[str, List[str]]],
    stop_tokens: Optional[Set[str]] = None,
    min_shared_tokens: int = 1,
    max_candidates_per_token: int = 2000
) -> Set[str]:
    """
    Retrieves candidate entity IDs sharing at least `min_shared_tokens` with the S1 entity.
    Searches within the S1 country partition and '__any__'.
    """
    if not s1_name_tokens:
        return set()

    c_key = str(s1_country_norm).strip().lower() if s1_country_norm else ''
    country_partitions = [c_key, '__any__'] if c_key else list(token_index.keys())

    candidate_counts: Counter = Counter()
    
    clean_tokens = [t for t in s1_name_tokens if t and (not stop_tokens or t not in stop_tokens)]
    if not clean_tokens:
        # Fallback to all tokens if all were filtered
        clean_tokens = list(s1_name_tokens)

    for ctry in country_partitions:
        if ctry not in token_index:
            continue
        ctry_idx = token_index[ctry]
        for tok in clean_tokens:
            if tok in ctry_idx:
                matched_ids = ctry_idx[tok]
                # Cap extremely wide single-token postings
                if len(matched_ids) <= max_candidates_per_token:
                    candidate_counts.update(matched_ids)

    if min_shared_tokens <= 1:
        return set(candidate_counts.keys())
    return {eid for eid, count in candidate_counts.items() if count >= min_shared_tokens}


def build_sorted_neighborhood_index(
    df: pd.DataFrame,
    sort_key_col: str = "name_norm",
    id_col: str = "entity_id",
    country_col: str = "country_norm"
) -> Tuple[List[str], np.ndarray, Dict[str, Dict[str, int]]]:
    """
    Builds sorted arrays of names per country partition for fast binary search
    and window extraction.
    """
    country_data: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    has_country = country_col in df.columns
    
    for idx, row in df[[id_col, sort_key_col] + ([country_col] if has_country else [])].iterrows():
        name = str(row[sort_key_col]).strip() if pd.notna(row[sort_key_col]) else ""
        if not name:
            continue
        ctry = str(row[country_col]).strip().lower() if has_country and pd.notna(row[country_col]) else "__any__"
        country_data[ctry].append((name, str(row[id_col])))

    # Sort each partition
    sorted_partitions: Dict[str, Tuple[List[str], List[str]]] = {}
    for ctry, records in country_data.items():
        records.sort(key=lambda x: x[0])
        sorted_partitions[ctry] = ([r[0] for r in records], [r[1] for r in records])

    return sorted_partitions


def get_sorted_neighborhood_candidates(
    s1_name_norm: str,
    s1_country_norm: str,
    sorted_partitions: Dict[str, Tuple[List[str], List[str]]],
    window_size: int = 10
) -> Set[str]:
    """
    Extracts +/- window_size records nearest to s1_name_norm in the sorted array.
    """
    if not s1_name_norm or not sorted_partitions:
        return set()

    c_key = str(s1_country_norm).strip().lower() if s1_country_norm else ''
    partitions_to_search = [c_key] if c_key and c_key in sorted_partitions else list(sorted_partitions.keys())

    candidates = set()
    for ctry in partitions_to_search:
        if ctry not in sorted_partitions:
            continue
        names, ids = sorted_partitions[ctry]
        if not names:
            continue
            
        pos = np.searchsorted(names, s1_name_norm)
        start = max(0, pos - window_size)
        end = min(len(ids), pos + window_size + 1)
        
        candidates.update(ids[start:end])

    return candidates
