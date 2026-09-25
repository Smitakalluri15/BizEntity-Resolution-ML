import re
from collections import defaultdict
from typing import Dict, Set, List, Optional, Iterable, Tuple, Any
import pandas as pd
from src.normalize.phonetic import get_phonetic_reduced, should_apply_phonetic_reduction


def compute_soundex(text: str) -> str:
    """
    Computes standard American Soundex for a token.
    """
    if not text:
        return ""
    t = re.sub(r'[^a-zA-Z]', '', text.upper())
    if not t:
        return ""
    
    first_letter = t[0]
    
    table = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    encoded = [first_letter]
    prev = table.get(first_letter, '0')
    
    for char in t[1:]:
        code = table.get(char, '0')
        if code != '0' and code != prev:
            encoded.append(code)
        prev = code
        
    soundex_str = ''.join(encoded).replace('0', '')
    return (soundex_str + '0000')[:4]


def extract_phonetic_blocking_keys(
    name_str: str,
    script: str = "latin",
    widen_for_tamil: bool = False
) -> Set[str]:
    """
    Extracts coarse phonetic blocking keys from a normalized or phonetically reduced name:
    1. Soundex of major tokens (e.g. 'G630' for Great, 'E525' for Engineering)
    2. Phonetic token 3-prefixes (e.g. 'gre', 'inj', 'lim')
    3. If Tamil and widen_for_tamil is True, also adds 2-prefixes and consonant skeletons.
    """
    if not name_str:
        return set()
        
    s = str(name_str).strip()
    if not s:
        return set()
        
    tokens = [w for w in s.split() if len(w) >= 2]
    if not tokens:
        return set()

    keys = set()
    is_tamil = (str(script).lower() == "tamil")

    for tok in tokens:
        # 1. Soundex key
        sx = compute_soundex(tok)
        if sx:
            keys.add(f"sx_{sx}")
            
        # 2. 3-prefix key
        if len(tok) >= 3:
            keys.add(f"p3_{tok[:3]}")
        else:
            keys.add(f"p3_{tok}")
            
        # 3. Widened keys for Tamil / hard non-Latin cases
        if is_tamil and widen_for_tamil:
            if len(tok) >= 2:
                keys.add(f"p2_{tok[:2]}")
            # Consonant skeleton (drop vowels)
            cons = re.sub(r'[aeiou]', '', tok)
            if len(cons) >= 2:
                keys.add(f"cs_{cons[:3]}")

    # First token exact prefix (high precision)
    if tokens:
        first_t = tokens[0]
        keys.add(f"first_{first_t[:4] if len(first_t)>=4 else first_t}")

    return keys


def build_phonetic_index(
    df: pd.DataFrame,
    name_col: str = "name_norm",
    phonetic_col: Optional[str] = "name_phonetic",
    script_col: Optional[str] = "name_script",
    country_col: Optional[str] = "country_norm",
    id_col: str = "entity_id"
) -> Dict[str, Dict[str, List[str]]]:
    """
    Builds country-partitioned phonetic blocking index:
      country -> { phonetic_key -> [entity_ids] }
    """
    phonetic_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    has_phonetic = phonetic_col and phonetic_col in df.columns
    has_script = script_col and script_col in df.columns
    has_country = country_col and country_col in df.columns
    
    ids = df[id_col].values
    names = df[name_col].fillna('').values
    phonetics = df[phonetic_col].fillna('').values if has_phonetic else names
    scripts = df[script_col].fillna('latin').values if has_script else ['latin'] * len(df)
    countries = df[country_col].fillna('').values if has_country else [''] * len(df)
    
    for eid, name, phon, script, ctry in zip(ids, names, phonetics, scripts, countries):
        ctry_key = str(ctry).strip().lower() if ctry else '__any__'
        target_name = phon if phon else name
        if not target_name:
            continue
            
        keys = extract_phonetic_blocking_keys(target_name, script=script, widen_for_tamil=True)
        for k in keys:
            phonetic_index[ctry_key][k].append(eid)
            
    return phonetic_index


def get_phonetic_candidates(
    s1_name_str: str,
    s1_script: str,
    s1_country_norm: str,
    phonetic_index: Dict[str, Dict[str, List[str]]],
    widen_for_tamil: bool = True,
    max_candidates_per_key: int = 2000
) -> Set[str]:
    """
    Retrieves candidates matching phonetic blocking keys of the S1 record.
    """
    if not s1_name_str:
        return set()

    c_key = str(s1_country_norm).strip().lower() if s1_country_norm else ''
    country_partitions = [c_key, '__any__'] if c_key else list(phonetic_index.keys())

    keys = extract_phonetic_blocking_keys(
        s1_name_str,
        script=s1_script,
        widen_for_tamil=widen_for_tamil
    )
    
    candidates = set()
    for ctry in country_partitions:
        if ctry not in phonetic_index:
            continue
        ctry_idx = phonetic_index[ctry]
        for k in keys:
            if k in ctry_idx:
                matched_ids = ctry_idx[k]
                if len(matched_ids) <= max_candidates_per_key:
                    candidates.update(matched_ids)
                    
    return candidates
