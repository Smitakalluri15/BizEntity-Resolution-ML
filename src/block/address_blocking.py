from collections import defaultdict, Counter
from typing import Dict, Set, List, Optional, Iterable, Tuple, Any
import pandas as pd

# Generic stop tokens in addresses that should not form standalone blocks
GENERIC_ADDRESS_TOKENS = {
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr',
    'court', 'ct', 'boulevard', 'blvd', 'floor', 'fl', 'suite', 'ste', 'building',
    'bldg', 'near', 'nr', 'behind', 'opp', 'opposite', 'house', 'plot', 'number',
    'no', 'cross', 'main', 'first', 'second', 'third', 'nagar', 'colony', 'block',
    'sector', 'apartment', 'apt', 'west', 'east', 'north', 'south'
}


def build_postal_index(
    df: pd.DataFrame,
    postal_col: str = "postal_code",
    country_col: Optional[str] = "country_norm",
    id_col: str = "entity_id"
) -> Dict[str, Dict[str, List[str]]]:
    """
    Builds country-partitioned postal code index:
      country -> { postal_code -> [entity_ids] }
    """
    postal_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    if postal_col not in df.columns:
        return postal_index
        
    has_country = country_col and country_col in df.columns
    countries = df[country_col].fillna('').values if has_country else [''] * len(df)
    ids = df[id_col].values
    postals = df[postal_col].fillna('').values
    
    for eid, post, ctry in zip(ids, postals, countries):
        p_str = str(post).strip()
        if p_str:
            ctry_key = str(ctry).strip().lower() if ctry else '__any__'
            postal_index[ctry_key][p_str].append(eid)
            
    return postal_index


def build_city_token_index(
    df: pd.DataFrame,
    address_col: str = "address_norm",
    token_col: Optional[str] = "address_tokens",
    country_col: Optional[str] = "country_norm",
    id_col: str = "entity_id",
    max_token_freq: float = 0.01
) -> Tuple[Dict[str, Dict[str, List[str]]], Set[str]]:
    """
    Builds country-partitioned address token index for records with missing postal codes.
    """
    n_records = len(df)
    max_count = max(500, int(n_records * max_token_freq))
    
    token_counter = Counter()
    has_token_col = token_col and token_col in df.columns
    
    if has_token_col:
        for tokens in df[token_col]:
            if isinstance(tokens, (list, set, tuple)):
                token_counter.update([t for t in tokens if len(t) >= 3 and t not in GENERIC_ADDRESS_TOKENS])
            elif isinstance(tokens, str) and tokens:
                t_list = [t.strip(" '\"[]") for t in tokens.split(',') if len(t.strip(" '\"[]")) >= 3 and t.strip(" '\"[]") not in GENERIC_ADDRESS_TOKENS]
                token_counter.update(t_list)
    else:
        for addr in df[address_col].fillna(''):
            if addr:
                tokens = [w for w in str(addr).split() if len(w) >= 3 and w not in GENERIC_ADDRESS_TOKENS]
                token_counter.update(tokens)

    stop_tokens = {tok for tok, cnt in token_counter.items() if cnt > max_count} | GENERIC_ADDRESS_TOKENS

    city_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    has_country = country_col and country_col in df.columns
    countries = df[country_col].fillna('').values if has_country else [''] * len(df)
    ids = df[id_col].values
    
    if has_token_col:
        for eid, ctry, tokens in zip(ids, countries, df[token_col]):
            ctry_key = str(ctry).strip().lower() if ctry else '__any__'
            if isinstance(tokens, (list, set, tuple)):
                t_set = set(tokens)
            elif isinstance(tokens, str) and tokens:
                t_set = {t.strip(" '\"[]") for t in tokens.split(',')}
            else:
                t_set = set()
                
            for tok in t_set:
                if tok and tok not in stop_tokens and len(tok) >= 3:
                    city_index[ctry_key][tok].append(eid)
    else:
        for eid, ctry, addr in zip(ids, countries, df[address_col].fillna('')):
            ctry_key = str(ctry).strip().lower() if ctry else '__any__'
            if addr:
                t_set = {w for w in str(addr).split() if len(w) >= 3 and w not in stop_tokens}
                for tok in t_set:
                    city_index[ctry_key][tok].append(eid)

    return city_index, stop_tokens


def get_address_candidates(
    s1_postal_code: Optional[str],
    s1_address_tokens: Iterable[str],
    s1_country_norm: str,
    postal_index: Dict[str, Dict[str, List[str]]],
    city_token_index: Dict[str, Dict[str, List[str]]],
    stop_address_tokens: Optional[Set[str]] = None,
    max_postal_candidates: int = 1500,
    min_address_tokens: int = 2
) -> Set[str]:
    """
    Retrieves candidates using postal code (primary) and multi-token address overlap (fallback).
    """
    c_key = str(s1_country_norm).strip().lower() if s1_country_norm else ''
    country_partitions = [c_key, '__any__'] if c_key else list(postal_index.keys())

    candidates = set()

    # 1. Postal code exact matching
    p_str = str(s1_postal_code).strip() if s1_postal_code and pd.notna(s1_postal_code) else ""
    if p_str:
        for ctry in country_partitions:
            if ctry in postal_index and p_str in postal_index[ctry]:
                post_matches = postal_index[ctry][p_str]
                if len(post_matches) <= max_postal_candidates:
                    candidates.update(post_matches)

    # 2. City / Address token overlap (for missing postal code or corroboration)
    if s1_address_tokens:
        clean_tokens = [
            t for t in s1_address_tokens
            if t and (not stop_address_tokens or t not in stop_address_tokens) and len(t) >= 3
        ]
        if clean_tokens:
            addr_counts: Counter = Counter()
            for ctry in country_partitions:
                if ctry in city_token_index:
                    c_idx = city_token_index[ctry]
                    for tok in clean_tokens:
                        if tok in c_idx:
                            addr_counts.update(c_idx[tok])
            
            # Require at least min_address_tokens overlap
            shared_matches = {eid for eid, count in addr_counts.items() if count >= min_address_tokens}
            candidates.update(shared_matches)

    return candidates
