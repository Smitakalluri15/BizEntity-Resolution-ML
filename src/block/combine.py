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
    s1_postal_code: Optional[str] = None,
    s1_address_tokens: Optional[Set[str]] = None,
    entity_postal_map: Optional[Dict[str, str]] = None,
    entity_address_tokens_map: Optional[Dict[str, Any]] = None,
    entity_script_map: Optional[Dict[str, str]] = None,
    is_non_latin: bool = False
) -> List[Tuple[str, float]]:
    """
    Computes an ultra-fast, symmetric pre-score for candidate ranking during top_k truncation.
    - Symmetrically maps both S1 and Candidates into the canonical phonetically reduced space.
    - Uses RapidFuzz token_set_ratio for name matching (robust to order and affix variations).
    - Takes max(name_sim, addr_sim) with interaction, postal, and cross-script bonuses so address-driven
      matches (e.g. acronyms/aliases with identical addresses) and cross-script name matches
      are both preserved near the top of the candidate list.
    """
    from rapidfuzz import fuzz
    from src.normalize.phonetic import get_phonetic_reduced

    s1_norm = str(s1_name_norm or "").strip()
    s1_red = get_phonetic_reduced(s1_norm) if s1_norm else ""
    s1_post = str(s1_postal_code or "").strip()
    s1_addr_toks = set(s1_address_tokens) if s1_address_tokens else set()

    scored = []
    for cid in candidate_ids:
        c_name = str(entity_name_map.get(cid, "") or "").strip()
        if not c_name:
            scored.append((cid, 0.0))
            continue

        c_phon = entity_phonetic_map.get(cid, "") if entity_phonetic_map else ""
        c_red = get_phonetic_reduced(c_phon if c_phon else c_name)

        # 1. Phonetic reduced similarity
        name_sim = fuzz.token_set_ratio(s1_red, c_red) / 100.0 if (s1_red and c_red) else 0.0
        
        # 2. Raw name similarity
        raw_sim = fuzz.token_set_ratio(s1_norm, c_name) / 100.0 if (s1_norm and c_name) else 0.0
        best_name = max(name_sim, raw_sim)

        # 3. Address token similarity
        if s1_addr_toks and entity_address_tokens_map:
            c_raw_addr = entity_address_tokens_map.get(cid, [])
            c_addr_toks = set(c_raw_addr) if isinstance(c_raw_addr, (list, set, tuple)) else set(str(c_raw_addr).split())
            addr_union = len(s1_addr_toks.union(c_addr_toks))
            addr_sim = (len(s1_addr_toks.intersection(c_addr_toks)) / addr_union) if addr_union > 0 else 0.0
        else:
            addr_sim = 0.0

        # 4. Postal exact match
        if s1_post and entity_postal_map:
            c_post = str(entity_postal_map.get(cid, "") or "").strip()
            post_match = 1.0 if (s1_post and c_post and s1_post == c_post) else 0.0
        else:
            post_match = 0.0

        # 5. Cross-script bonus for non-Latin candidate to balance against high-frequency Latin matches
        script_boost = 0.08 if (entity_script_map and entity_script_map.get(cid, 'latin') != 'latin') else 0.0

        # Combined pre-score
        final_score = max(best_name, addr_sim) + 0.15 * min(best_name, addr_sim) + 0.10 * post_match + script_boost
        scored.append((cid, final_score))

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
    entity_postal_map: Optional[Dict[str, str]] = None,
    entity_address_tokens_map: Optional[Dict[str, Any]] = None,
    entity_script_map: Optional[Dict[str, str]] = None,
    top_k: int = 400,
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
    if not name_norm and 'business_name' in s1_row:
        import re
        raw_bname = str(s1_row.get('business_name', '') if hasattr(s1_row, 'get') else s1_row['business_name']).lower()
        name_norm = re.sub(r'https?://|www\.|\.(?:com|org|net|in|co|io|biz|info|us|edu|gov)\b|@|[^\w\s]', ' ', raw_bname).strip()

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
            s1_name_norm=name_norm,
            s1_name_phonetic=name_phonetic,
            s1_name_tokens=name_tokens,
            candidate_ids=candidates,
            entity_name_map=entity_name_map,
            entity_phonetic_map=entity_phonetic_map,
            s1_postal_code=postal_code,
            s1_address_tokens=addr_tokens,
            entity_postal_map=entity_postal_map,
            entity_address_tokens_map=entity_address_tokens_map,
            entity_script_map=entity_script_map,
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
    entity_postal_map: Optional[Dict[str, str]] = None,
    entity_address_tokens_map: Optional[Dict[str, Any]] = None,
    top_k: int = 250,
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
            entity_postal_map=entity_postal_map,
            entity_address_tokens_map=entity_address_tokens_map,
            top_k=top_k,
            window_size=window_size,
            widen_for_tamil=widen_for_tamil
        )
        rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ','.join(sorted(cands))
        })

    return pd.DataFrame(rows)
