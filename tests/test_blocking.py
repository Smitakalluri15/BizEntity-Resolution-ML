import pytest
import pandas as pd
import numpy as np

from src.block.name_blocking import build_token_index, get_token_candidates, build_sorted_neighborhood_index, get_sorted_neighborhood_candidates
from src.block.phonetic_blocking import build_phonetic_index, get_phonetic_candidates, compute_soundex
from src.block.address_blocking import build_postal_index, build_city_token_index, get_address_candidates
from src.block.combine import filter_by_country, generate_candidates, generate_all_candidates_df


@pytest.fixture
def synthetic_external_df():
    """
    Synthetic external dataset (S2/S3) with diverse match scenarios.
    """
    data = [
        # 1. Exact token match
        {'entity_id': 'S2-101', 'name_norm': 'acme freight incorporated', 'name_phonetic': 'akme freit', 'name_tokens': ['acme', 'freight'], 'country_norm': 'us', 'postal_code': '94105', 'address_norm': '100 market street san francisco', 'address_tokens': ['market', 'san', 'francisco'], 'name_script': 'latin'},
        # 2. Cross-script phonetic match (Devanagari transliterated)
        {'entity_id': 'S3-202', 'name_norm': 'greta imjiniyarimga', 'name_phonetic': 'gret injiniyaring', 'name_tokens': ['greta', 'imjiniyarimga'], 'country_norm': 'india', 'postal_code': '400001', 'address_norm': 'fort mumbai', 'address_tokens': ['fort', 'mumbai'], 'name_script': 'devanagari'},
        # 3. Postal code match only (completely different name)
        {'entity_id': 'S2-303', 'name_norm': 'zebra enterprises', 'name_phonetic': 'sebra enterprisis', 'name_tokens': ['zebra'], 'country_norm': 'us', 'postal_code': '90210', 'address_norm': 'beverly hills', 'address_tokens': ['beverly', 'hills'], 'name_script': 'latin'},
        # 4. Completely unrelated record (should NOT be retrieved)
        {'entity_id': 'S3-404', 'name_norm': 'unrelated pharmacy', 'name_phonetic': 'unrelated farmasi', 'name_tokens': ['unrelated', 'pharmacy'], 'country_norm': 'france', 'postal_code': '75001', 'address_norm': 'paris rue rivoli', 'address_tokens': ['paris', 'rue', 'rivoli'], 'name_script': 'latin'},
        # 5. Missing postal code but matching city tokens
        {'entity_id': 'S2-505', 'name_norm': 'sharma traders', 'name_phonetic': 'sarma traders', 'name_tokens': ['sharma', 'traders'], 'country_norm': 'india', 'postal_code': '', 'address_norm': 'opp station dadar mumbai', 'address_tokens': ['station', 'dadar', 'mumbai'], 'name_script': 'latin'},
        # 6. Missing country record
        {'entity_id': 'S3-606', 'name_norm': 'global apex corp', 'name_phonetic': 'global apeks', 'name_tokens': ['global', 'apex'], 'country_norm': '', 'postal_code': '99999', 'address_norm': 'unknown place', 'address_tokens': ['unknown'], 'name_script': 'latin'}
    ]
    return pd.DataFrame(data)


def test_token_blocking(synthetic_external_df):
    token_idx, stop_tokens = build_token_index(synthetic_external_df, max_token_freq=0.5)
    cands = get_token_candidates(['acme'], 'us', token_idx, stop_tokens)
    assert 'S2-101' in cands
    assert 'S3-404' not in cands


def test_phonetic_blocking(synthetic_external_df):
    phon_idx = build_phonetic_index(synthetic_external_df)
    # S1: 'great engineering' in India -> phonetic 'gret injiniyaring'
    cands = get_phonetic_candidates('gret injiniyaring', 'devanagari', 'india', phon_idx)
    assert 'S3-202' in cands
    assert 'S3-404' not in cands


def test_address_and_postal_blocking(synthetic_external_df):
    post_idx = build_postal_index(synthetic_external_df)
    city_idx, stop_addr = build_city_token_index(synthetic_external_df, max_token_freq=0.5)

    # Match by postal code 90210
    cands_post = get_address_candidates('90210', [], 'us', post_idx, city_idx)
    assert 'S2-303' in cands_post
    assert 'S3-404' not in cands_post

    # Match by address tokens ('dadar', 'mumbai') when postal code is missing
    cands_addr = get_address_candidates('', ['dadar', 'mumbai'], 'india', post_idx, city_idx, stop_address_tokens=stop_addr)
    assert 'S2-505' in cands_addr


def test_filter_by_country():
    entity_country_map = {
        'S2-101': 'us',
        'S3-202': 'india',
        'S3-404': 'france',
        'S3-606': ''  # missing country
    }
    candidates = {'S2-101', 'S3-202', 'S3-404', 'S3-606'}
    
    # Filter by 'us'
    filtered_us = filter_by_country(candidates, 'us', entity_country_map)
    assert 'S2-101' in filtered_us
    assert 'S3-606' in filtered_us  # missing country is preserved
    assert 'S3-202' not in filtered_us
    assert 'S3-404' not in filtered_us

    # If S1 has no country, all candidates preserved
    filtered_empty = filter_by_country(candidates, '', entity_country_map)
    assert filtered_empty == candidates


def test_generate_candidates_end_to_end(synthetic_external_df):
    token_idx, stop_tok = build_token_index(synthetic_external_df, max_token_freq=0.5)
    phon_idx = build_phonetic_index(synthetic_external_df)
    post_idx = build_postal_index(synthetic_external_df)
    city_idx, stop_addr = build_city_token_index(synthetic_external_df, max_token_freq=0.5)
    sn_partitions = build_sorted_neighborhood_index(synthetic_external_df)

    entity_country_map = dict(zip(synthetic_external_df['entity_id'], synthetic_external_df['country_norm']))
    entity_name_map = dict(zip(synthetic_external_df['entity_id'], synthetic_external_df['name_norm']))
    entity_phon_map = dict(zip(synthetic_external_df['entity_id'], synthetic_external_df['name_phonetic']))

    # Test S1 record 1: 'acme express' in US
    s1_row_1 = {
        'entity_id': 'S1-1',
        'name_norm': 'acme express',
        'name_phonetic': 'akme ekspres',
        'name_tokens': ['acme', 'express'],
        'country_norm': 'us',
        'postal_code': '94105',
        'address_tokens': ['market'],
        'name_script': 'latin'
    }
    cands_1 = generate_candidates(
        s1_row=s1_row_1,
        token_index=token_idx,
        phonetic_index=phon_idx,
        postal_index=post_idx,
        city_token_index=city_idx,
        sorted_neighborhood_partitions=sn_partitions,
        stop_tokens=stop_tok,
        stop_address_tokens=stop_addr,
        entity_country_map=entity_country_map,
        entity_name_map=entity_name_map,
        entity_phonetic_map=entity_phon_map,
        top_k=10
    )
    assert 'S2-101' in cands_1
    assert 'S3-404' not in cands_1  # Unrelated France record not retrieved

    # Test S1 record 2: 'great engineering' in India
    s1_row_2 = {
        'entity_id': 'S1-2',
        'name_norm': 'great engineering',
        'name_phonetic': 'gret injiniyaring',
        'name_tokens': ['great', 'engineering'],
        'country_norm': 'india',
        'postal_code': '400001',
        'address_tokens': ['fort', 'mumbai'],
        'name_script': 'latin'
    }
    cands_2 = generate_candidates(
        s1_row=s1_row_2,
        token_index=token_idx,
        phonetic_index=phon_idx,
        postal_index=post_idx,
        city_token_index=city_idx,
        sorted_neighborhood_partitions=sn_partitions,
        stop_tokens=stop_tok,
        stop_address_tokens=stop_addr,
        entity_country_map=entity_country_map,
        entity_name_map=entity_name_map,
        entity_phonetic_map=entity_phon_map,
        top_k=10
    )
    assert 'S3-202' in cands_2


def test_top_k_capping(synthetic_external_df):
    token_idx, stop_tok = build_token_index(synthetic_external_df, max_token_freq=0.5)
    phon_idx = build_phonetic_index(synthetic_external_df)
    post_idx = build_postal_index(synthetic_external_df)
    city_idx, stop_addr = build_city_token_index(synthetic_external_df, max_token_freq=0.5)

    entity_country_map = dict(zip(synthetic_external_df['entity_id'], synthetic_external_df['country_norm']))
    entity_name_map = dict(zip(synthetic_external_df['entity_id'], synthetic_external_df['name_norm']))
    entity_phon_map = dict(zip(synthetic_external_df['entity_id'], synthetic_external_df['name_phonetic']))

    # Query with top_k = 1
    s1_row = {
        'entity_id': 'S1-1',
        'name_norm': 'acme freight',
        'name_phonetic': 'akme freit',
        'name_tokens': ['acme', 'freight'],
        'country_norm': 'us',
        'postal_code': '94105',
        'address_tokens': ['market'],
        'name_script': 'latin'
    }
    cands = generate_candidates(
        s1_row=s1_row,
        token_index=token_idx,
        phonetic_index=phon_idx,
        postal_index=post_idx,
        city_token_index=city_idx,
        entity_country_map=entity_country_map,
        entity_name_map=entity_name_map,
        entity_phonetic_map=entity_phon_map,
        top_k=1
    )
    assert len(cands) == 1
    assert 'S2-101' in cands
