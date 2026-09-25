"""Business Entity Resolution Normalization Module.

Exposes end-to-end normalization utilities for names, addresses, countries,
script detection, and full dataframe batch processing.
"""

from typing import Optional
import pandas as pd
import numpy as np

from src.normalize.transliteration import (
    detect_script,
    transliterate_to_latin,
    get_script_flag,
)
from src.normalize.name import (
    normalize_name,
    normalize_name_tokens,
    get_name_char_ngrams,
    extract_url_or_handle,
)
from src.normalize.address import (
    normalize_address,
    get_address_tokens,
    get_address_char_ngrams,
    is_landmark_address,
    extract_postal_code,
)
from src.normalize.country import normalize_country


def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Apply end-to-end normalization to any entity dataframe (Source 1, 2, 3 or Test).

    Preserves all original columns and adds:
      - name_raw, name_norm, name_tokens, name_script, name_embedded_url
      - address_raw, address_norm, address_tokens, address_script, postal_code, is_landmark_address
      - country_raw, country_norm

    Uses single-pass dictionary building across unique values for maximum speed across millions of rows.
    """
    df_out = df.copy()

    # 1. Names - single pass over unique strings
    df_out['name_raw'] = df_out['business_name']
    
    unique_names = df_out['business_name'].dropna().unique()
    norm_name_map = {}
    name_toks_map = {}
    name_script_map = {}
    name_url_map = {}

    for n in unique_names:
        name_script_map[n] = detect_script(n)
        name_url_map[n] = extract_url_or_handle(n)
        norm = normalize_name(n)
        norm_name_map[n] = norm
        name_toks_map[n] = sorted(list(set(norm.split()))) if norm else []

    df_out['name_norm'] = df_out['business_name'].map(norm_name_map).fillna('')
    df_out['name_tokens'] = df_out['business_name'].map(name_toks_map)
    df_out['name_tokens'] = df_out['name_tokens'].apply(lambda x: x if isinstance(x, list) else [])
    df_out['name_script'] = df_out['business_name'].map(name_script_map).fillna('empty')
    df_out['name_embedded_url'] = df_out['business_name'].map(name_url_map)

    # 2. Addresses - single pass over unique strings
    df_out['address_raw'] = df_out['business_address']
    
    unique_addrs = df_out['business_address'].dropna().unique()
    norm_addr_map = {}
    addr_toks_map = {}
    addr_script_map = {}
    postal_map = {}
    landmark_map = {}

    for a in unique_addrs:
        addr_script_map[a] = detect_script(a)
        landmark_map[a] = is_landmark_address(a)
        postal_map[a] = extract_postal_code(a)
        norm = normalize_address(a)
        norm_addr_map[a] = norm
        addr_toks_map[a] = norm.split() if norm else []

    df_out['address_norm'] = df_out['business_address'].map(norm_addr_map).fillna('')
    df_out['address_tokens'] = df_out['business_address'].map(addr_toks_map)
    df_out['address_tokens'] = df_out['address_tokens'].apply(lambda x: x if isinstance(x, list) else [])
    df_out['address_script'] = df_out['business_address'].map(addr_script_map).fillna('empty')
    df_out['postal_code'] = df_out['business_address'].map(postal_map)
    
    mapped_landmark = df_out['business_address'].map(landmark_map)
    df_out['is_landmark_address'] = np.where(mapped_landmark.isna(), False, mapped_landmark.astype(bool))

    # 3. Country - single pass over unique strings
    df_out['country_raw'] = df_out['country']
    
    unique_countries = df_out['country'].dropna().unique()
    country_map = {c: normalize_country(c) for c in unique_countries}
    df_out['country_norm'] = df_out['country'].map(country_map).fillna('')

    return df_out


__all__ = [
    "detect_script",
    "transliterate_to_latin",
    "get_script_flag",
    "normalize_name",
    "normalize_name_tokens",
    "get_name_char_ngrams",
    "extract_url_or_handle",
    "normalize_address",
    "get_address_tokens",
    "get_address_char_ngrams",
    "is_landmark_address",
    "extract_postal_code",
    "normalize_country",
    "apply_normalization",
]
