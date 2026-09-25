import time
from typing import Dict, Set, List, Optional, Iterable, Tuple, Any
import pandas as pd
import numpy as np

from src.block.name_blocking import get_token_candidates, get_sorted_neighborhood_candidates
from src.block.phonetic_blocking import get_phonetic_candidates
from src.block.address_blocking import get_address_candidates
from src.normalize.phonetic import phonetic_similarity


def filter_by_country(
    candidates: Set[str],
    s1_country_norm: str,
    entity_country_map: Dict[str, str]
) -> Set[str]:
    """
    Applies country-level hard partition:
    - Removes candidates whose country_norm contradicts s1_country_norm.
    - If s1_country_norm is empty/missing, preserves all candidates.
    """
    if not candidates:
        return set()
        
    s1_c = str(s1_country_norm).strip().lower() if s1_country_norm and pd.notna(s1_country_norm) else ""
    if not s1_c:
        return candidates

    filtered = set()
    for cand_id in candidates:
        cand_c = entity_country_map.get(cand_id, "")
        if not cand_c or cand_c == "__any__" or cand_c == s1_c:
            filtered.add(cand_id)
            
    return filtered


def pre_score_candidates(
    s1_name_norm: str,
    s1_name_phonetic: str,
    s1_name_tokens: Set[str],
    candidate_ids: Set[str],
    entity_name_map: Dict[str, str],
    entity_phonetic_map: Optional[Dict[str, str]] = None,
    is_non_latin: bool = False
) -> List[Tuple[str, float]]:
    """
    Computes an ultra-fast, vectorized pre-score for candidate ranking during top_k truncation.
    Takes the maximum of Token Jaccard and Phonetic Bigram Jaccard so cross-script matches
    are never penalized relative to Latin matches.
    """
    s1_target_phon = s1_name_phonetic if s1_name_phonetic else s1_name_norm
    s1_bi = {s1_target_phon[i:i+2] for i in range(len(s1_target_phon)-1)} if len(s1_target_phon) >= 2 else {s1_target_phon}
    
    scored = []
    for cid in candidate_ids:
        c_name = entity_name_map.get(cid, "")
        if not c_name:
            scored.append((cid, 0.0))
            continue
            
        c_tokens = set(c_name.split())
        
        # 1. Token Jaccard
        tok_overlap = len(s1_name_tokens.intersection(c_tokens))
        tok_union = len(s1_name_tokens.union(c_tokens))
        tok_jaccard = (tok_overlap / tok_union) if tok_union > 0 else 0.0
        
        # 2. Phonetic Bigram Jaccard (using pre-stored phonetic representation)
        c_phon = entity_phonetic_map.get(cid, c_name) if entity_phonetic_map else c_name
        c_bi = {c_phon[i:i+2] for i in range(len(c_phon)-1)} if len(c_phon) >= 2 else {c_phon}
        bi_jaccard = len(s1_bi.intersection(c_bi)) / max(1, len(s1_bi.union(c_bi)))
        
        # Combined score is maximum of exact token overlap and phonetic overlap
        combined_score = max(tok_jaccard, bi_jaccard)
        scored.append((cid, combined_score))
        
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def generate_candidates(
    s1_row: Any,
    token_index: Dict[str, Dict[str, List[str]]],
    phonetic_index: Dict[str, Dict[str, List[str]]],
    postal_index: Dict[str, Dict[str, List[str]]],
    city_token_index: Dict[str, Dict[str, List[str]]],
    sorted_neighborhood_partitions: Optional[Dict[str, Tuple[List[str], List[str]]]] = None,
    stop_tokens: Optional[Set[str]] = None,
    stop_address_tokens: Optional[Set[str]] = None,
    entity_country_map: Optional[Dict[str, str]] = None,
    entity_name_map: Optional[Dict[str, str]] = None,
    entity_phonetic_map: Optional[Dict[str, str]] = None,
    top_k: int = 150,
    window_size: int = 10,
    widen_for_tamil: bool = True
) -> Set[str]:
    """
    Executes multi-strategy blocking for a single Source-1 entity:
    - Unions candidates from Token Index, Phonetic Index, Postal/Address Index, Sorted Neighborhood
    - Applies hard country filter
    - Caps candidate set to top_k using fast pre-scoring
    """
    # Extract S1 attributes
    name_norm = str(s1_row.get('name_norm', '') if hasattr(s1_row, 'get') else s1_row['name_norm'])
    country_norm = str(s1_row.get('country_norm', '') if hasattr(s1_row, 'get') else s1_row['country_norm'])
    script = str(s1_row.get('name_script', 'latin') if hasattr(s1_row, 'get') else s1_row['name_script'])
    name_phonetic = str(s1_row.get('name_phonetic', name_norm) if hasattr(s1_row, 'get') else (s1_row['name_phonetic'] if 'name_phonetic' in s1_row else name_norm))
    postal_code = str(s1_row.get('postal_code', '') if hasattr(s1_row, 'get') else s1_row['postal_code']) if pd.notna(s1_row.get('postal_code', '') if hasattr(s1_row, 'get') else s1_row['postal_code']) else ""

    # Parse tokens
    raw_name_tokens = s1_row.get('name_tokens', []) if hasattr(s1_row, 'get') else s1_row['name_tokens']
    if isinstance(raw_name_tokens, (list, set, tuple)):
        name_tokens = set(raw_name_tokens)
    elif isinstance(raw_name_tokens, str) and raw_name_tokens:
        name_tokens = {t.strip(" '\"[]") for t in raw_name_tokens.split(',') if t.strip(" '\"[]")}
    else:
        name_tokens = set(name_norm.split())

    raw_addr_tokens = s1_row.get('address_tokens', []) if hasattr(s1_row, 'get') else s1_row['address_tokens']
    if isinstance(raw_addr_tokens, (list, set, tuple)):
        addr_tokens = set(raw_addr_tokens)
    elif isinstance(raw_addr_tokens, str) and raw_addr_tokens:
        addr_tokens = {t.strip(" '\"[]") for t in raw_addr_tokens.split(',') if t.strip(" '\"[]")}
    else:
        addr_tokens = set()

    # 1. Token Index Candidates
    candidates = set(get_token_candidates(
        name_tokens,
        country_norm,
        token_index,
        stop_tokens=stop_tokens,
        min_shared_tokens=1
    ))

    # 2. Phonetic Index Candidates
    phon_cands = get_phonetic_candidates(
        name_phonetic if name_phonetic else name_norm,
        script,
        country_norm,
        phonetic_index,
        widen_for_tamil=widen_for_tamil
    )
    candidates.update(phon_cands)

    # 3. Address & Postal Candidates
    addr_cands = get_address_candidates(
        postal_code,
        addr_tokens,
        country_norm,
        postal_index,
        city_token_index,
        stop_address_tokens=stop_address_tokens
    )
    candidates.update(addr_cands)

    # 4. Sorted Neighborhood Candidates (if provided)
    if sorted_neighborhood_partitions:
        sn_cands = get_sorted_neighborhood_candidates(
            name_norm,
            country_norm,
            sorted_neighborhood_partitions,
            window_size=window_size
        )
        candidates.update(sn_cands)

    # 5. Country Filter
    if entity_country_map:
        candidates = filter_by_country(candidates, country_norm, entity_country_map)

    # 6. Top-K Capping via Pre-scoring
    if len(candidates) > top_k and entity_name_map:
        is_non_latin = (script != "latin")
        ranked = pre_score_candidates(
            name_norm,
            name_phonetic,
            name_tokens,
            candidates,
            entity_name_map,
            entity_phonetic_map=entity_phonetic_map,
            is_non_latin=is_non_latin
        )
        candidates = {cid for cid, _ in ranked[:top_k]}

    return candidates


def generate_all_candidates_df(
    s1_df: pd.DataFrame,
    token_index: Dict[str, Dict[str, List[str]]],
    phonetic_index: Dict[str, Dict[str, List[str]]],
    postal_index: Dict[str, Dict[str, List[str]]],
    city_token_index: Dict[str, Dict[str, List[str]]],
    sorted_neighborhood_partitions: Optional[Dict[str, Tuple[List[str], List[str]]]] = None,
    stop_tokens: Optional[Set[str]] = None,
    stop_address_tokens: Optional[Set[str]] = None,
    entity_country_map: Optional[Dict[str, str]] = None,
    entity_name_map: Optional[Dict[str, str]] = None,
    entity_phonetic_map: Optional[Dict[str, str]] = None,
    top_k: int = 50,
    window_size: int = 10,
    widen_for_tamil: bool = True
) -> pd.DataFrame:
    """
    Generates candidates for all S1 records and formats as candidate_pairs.tsv DataFrame:
      source1_entity_id | candidate_entity_ids (comma-separated)
    """
    rows = []
    for idx, s1_row in s1_df.iterrows():
        s1_id = s1_row['entity_id'] if 'entity_id' in s1_row else s1_row['source1_entity_id']
        cands = generate_candidates(
            s1_row=s1_row,
            token_index=token_index,
            phonetic_index=phonetic_index,
            postal_index=postal_index,
            city_token_index=city_token_index,
            sorted_neighborhood_partitions=sorted_neighborhood_partitions,
            stop_tokens=stop_tokens,
            stop_address_tokens=stop_address_tokens,
            entity_country_map=entity_country_map,
            entity_name_map=entity_name_map,
            entity_phonetic_map=entity_phonetic_map,
            top_k=top_k,
            window_size=window_size,
            widen_for_tamil=widen_for_tamil
        )
        rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ','.join(sorted(cands))
        })

    return pd.DataFrame(rows)
