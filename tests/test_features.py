"""
Unit tests for Feature Engineering Module (Phase 6).
"""

import pytest
import numpy as np
import pandas as pd

from src.features.name_features import compute_name_features
from src.features.address_features import compute_address_features
from src.features.tfidf_features import fit_tfidf_vectorizers, compute_tfidf_features
from src.features.cross_features import compute_cross_features
from src.features.build_features import (
    join_candidate_data,
    build_features,
    assemble_labels,
)


@pytest.fixture
def synthetic_data():
    """Builds synthetic S1, S2, S3 normalized datasets and candidate pairs for testing."""
    s1_df = pd.DataFrame([
        {
            'entity_id': 'S1_001',
            'name_raw': 'Jain Foods',
            'name_norm': 'jain foods',
            'name_tokens': ['jain', 'foods'],
            'name_script': 'latin',
            'address_raw': '123 MG Road, Bangalore 560001',
            'address_norm': '123 mg road bangalore 560001',
            'address_tokens': ['123', 'mg', 'road', 'bangalore', '560001'],
            'postal_code': '560001',
            'is_landmark_address': False,
            'country_norm': 'india'
        },
        {
            'entity_id': 'S1_002',
            'name_raw': 'Prime Trading Co',
            'name_norm': 'prime trading co',
            'name_tokens': ['prime', 'trading', 'co'],
            'name_script': 'latin',
            'address_raw': 'Near City Hospital, Mumbai',
            'address_norm': 'near city hospital mumbai',
            'address_tokens': ['near', 'city', 'hospital', 'mumbai'],
            'postal_code': None,
            'is_landmark_address': True,
            'country_norm': 'india'
        },
        {
            'entity_id': 'S1_003',
            'name_raw': 'Apple Technologies Inc',
            'name_norm': 'apple technologies inc',
            'name_tokens': ['apple', 'technologies', 'inc'],
            'name_script': 'latin',
            'address_raw': '1 Infinite Loop, Cupertino 95014',
            'address_norm': '1 infinite loop cupertino 95014',
            'address_tokens': ['1', 'infinite', 'loop', 'cupertino', '95014'],
            'postal_code': '95014',
            'is_landmark_address': False,
            'country_norm': 'united states'
        }
    ])

    s2_df = pd.DataFrame([
        # Exact match to S1_001
        {
            'entity_id': 'S2_001',
            'name_raw': 'Jain Foods',
            'name_norm': 'jain foods',
            'name_tokens': ['jain', 'foods'],
            'name_script': 'latin',
            'address_raw': '123 MG Road, Bangalore 560001',
            'address_norm': '123 mg road bangalore 560001',
            'address_tokens': ['123', 'mg', 'road', 'bangalore', '560001'],
            'postal_code': '560001',
            'is_landmark_address': False,
            'country_norm': 'india'
        },
        # Cross-script transliteration match to S1_001
        {
            'entity_id': 'S2_002',
            'name_raw': 'जैन फूड्स',
            'name_norm': 'jaina phudsa',
            'name_tokens': ['jaina', 'phudsa'],
            'name_script': 'devanagari',
            'address_raw': '123 MG Road Bangalore',
            'address_norm': '123 mg road bangalore',
            'address_tokens': ['123', 'mg', 'road', 'bangalore'],
            'postal_code': '560001',
            'is_landmark_address': False,
            'country_norm': 'india'
        },
        # Landmark / Missing postal match to S1_002
        {
            'entity_id': 'S2_003',
            'name_raw': 'Prime Trading Enterprise',
            'name_norm': 'prime trading enterprise',
            'name_tokens': ['prime', 'trading', 'enterprise'],
            'name_script': 'latin',
            'address_raw': 'Opposite City Hospital, Mumbai',
            'address_norm': 'opposite city hospital mumbai',
            'address_tokens': ['opposite', 'city', 'hospital', 'mumbai'],
            'postal_code': None,
            'is_landmark_address': True,
            'country_norm': 'india'
        }
    ])

    s3_df = pd.DataFrame([
        # Negative / competitor candidate for S1_001
        {
            'entity_id': 'S3_001',
            'name_raw': 'Global Foods Solutions',
            'name_norm': 'global foods solutions',
            'name_tokens': ['global', 'foods', 'solutions'],
            'name_script': 'latin',
            'address_raw': '789 Brigade Road, Bangalore 560025',
            'address_norm': '789 brigade road bangalore 560025',
            'address_tokens': ['789', 'brigade', 'road', 'bangalore', '560025'],
            'postal_code': '560025',
            'is_landmark_address': False,
            'country_norm': 'india'
        },
        # Exact match to S1_003
        {
            'entity_id': 'S3_002',
            'name_raw': 'Apple Tech Inc',
            'name_norm': 'apple tech inc',
            'name_tokens': ['apple', 'tech', 'inc'],
            'name_script': 'latin',
            'address_raw': '1 Infinite Loop, Cupertino CA 95014',
            'address_norm': '1 infinite loop cupertino ca 95014',
            'address_tokens': ['1', 'infinite', 'loop', 'cupertino', 'ca', '95014'],
            'postal_code': '95014',
            'is_landmark_address': False,
            'country_norm': 'united states'
        }
    ])

    candidate_pairs_df = pd.DataFrame([
        {'source1_entity_id': 'S1_001', 'candidate_entity_id': 'S2_001'},
        {'source1_entity_id': 'S1_001', 'candidate_entity_id': 'S2_002'},
        {'source1_entity_id': 'S1_001', 'candidate_entity_id': 'S3_001'},
        {'source1_entity_id': 'S1_002', 'candidate_entity_id': 'S2_003'},
        {'source1_entity_id': 'S1_003', 'candidate_entity_id': 'S3_002'},
    ])

    ground_truth_df = pd.DataFrame([
        {'source1_entity_id': 'S1_001', 'matched_entity_ids': 'S2_001,S2_002'},
        {'source1_entity_id': 'S1_002', 'matched_entity_ids': 'S2_003'},
        {'source1_entity_id': 'S1_003', 'matched_entity_ids': 'S3_002'},
    ])

    return s1_df, s2_df, s3_df, candidate_pairs_df, ground_truth_df


def test_join_candidate_data(synthetic_data):
    s1_df, s2_df, s3_df, candidate_pairs_df, _ = synthetic_data
    joined = join_candidate_data(candidate_pairs_df, s1_df, s2_df, s3_df)

    assert len(joined) == len(candidate_pairs_df)
    assert 's1_name_norm' in joined.columns
    assert 'cand_name_norm' in joined.columns
    assert 's1_postal_code' in joined.columns
    assert 'cand_postal_code' in joined.columns
    assert joined.loc[0, 's1_name_norm'] == 'jain foods'
    assert joined.loc[0, 'cand_name_norm'] == 'jain foods'


def test_name_features_exact_and_fuzzy(synthetic_data):
    s1_df, s2_df, s3_df, candidate_pairs_df, _ = synthetic_data
    joined = join_candidate_data(candidate_pairs_df, s1_df, s2_df, s3_df)
    name_feats = compute_name_features(joined)

    # Row 0: Exact match
    assert name_feats.loc[0, 'name_exact_norm'] == 1
    assert name_feats.loc[0, 'name_levenshtein_ratio'] == 1.0
    assert name_feats.loc[0, 'name_jaccard_tokens'] == 1.0
    assert name_feats.loc[0, 'name_token_overlap_count'] == 2
    assert name_feats.loc[0, 'name_char_trigram_jaccard'] == 1.0
    assert name_feats.loc[0, 'name_script_match'] == 1

    # Row 1: Cross-script Devanagari match
    assert name_feats.loc[1, 'name_exact_norm'] == 0
    assert name_feats.loc[1, 'name_script_match'] == 0
    # Cross-script phonetic similarity should be high after phonetic reduction
    assert name_feats.loc[1, 'name_phonetic_similarity'] >= 0.60

    # Row 2: Partial overlap (jain foods vs global foods solutions)
    assert name_feats.loc[2, 'name_exact_norm'] == 0
    assert name_feats.loc[2, 'name_token_overlap_count'] == 1  # 'foods'
    assert 0.0 < name_feats.loc[2, 'name_jaccard_tokens'] < 1.0


def test_address_features_and_postal_semantics(synthetic_data):
    s1_df, s2_df, s3_df, candidate_pairs_df, _ = synthetic_data
    joined = join_candidate_data(candidate_pairs_df, s1_df, s2_df, s3_df)
    addr_feats = compute_address_features(joined)

    # Row 0: Exact postal code match (560001 == 560001)
    assert addr_feats.loc[0, 'postal_available'] == 1
    assert addr_feats.loc[0, 'postal_exact_match'] == 1
    assert addr_feats.loc[0, 'address_exact_norm'] == 1
    assert addr_feats.loc[0, 'landmark_flag_either'] == 0

    # Row 2: Mismatched postal code (560001 != 560025)
    assert addr_feats.loc[2, 'postal_available'] == 1
    assert addr_feats.loc[2, 'postal_exact_match'] == 0

    # Row 3: Missing postal code on both sides with landmark address
    assert addr_feats.loc[3, 'postal_available'] == 0
    assert addr_feats.loc[3, 'postal_exact_match'] == 0
    assert addr_feats.loc[3, 'landmark_flag_either'] == 1


def test_tfidf_features(synthetic_data):
    s1_df, s2_df, s3_df, candidate_pairs_df, _ = synthetic_data
    joined = join_candidate_data(candidate_pairs_df, s1_df, s2_df, s3_df)
    tfidf_feats = compute_tfidf_features(joined)

    assert 'name_tfidf_word_cosine' in tfidf_feats.columns
    assert 'name_tfidf_char_cosine' in tfidf_feats.columns
    assert 'address_tfidf_word_cosine' in tfidf_feats.columns

    # Exact match row should have cosine ~ 1.0
    assert tfidf_feats.loc[0, 'name_tfidf_word_cosine'] > 0.99
    assert tfidf_feats.loc[0, 'name_tfidf_char_cosine'] > 0.99
    assert tfidf_feats.loc[0, 'address_tfidf_word_cosine'] > 0.99

    # Negative candidate should have lower cosine
    assert tfidf_feats.loc[2, 'name_tfidf_word_cosine'] < tfidf_feats.loc[0, 'name_tfidf_word_cosine']


def test_cross_features_group_ranking_and_gap(synthetic_data):
    s1_df, s2_df, s3_df, candidate_pairs_df, _ = synthetic_data
    joined = join_candidate_data(candidate_pairs_df, s1_df, s2_df, s3_df)
    name_feats = compute_name_features(joined)
    addr_feats = compute_address_features(joined)
    cross_feats = compute_cross_features(joined, name_feats, addr_feats, score_col='name_levenshtein_ratio')

    # S1_001 has 3 candidates (rows 0, 1, 2)
    assert cross_feats.loc[0, 'candidate_set_size'] == 3
    assert cross_feats.loc[1, 'candidate_set_size'] == 3
    assert cross_feats.loc[2, 'candidate_set_size'] == 3

    # Exact match row 0 should be rank 1
    assert cross_feats.loc[0, 'candidate_rank_in_group'] == 1
    # top1_gap should be positive (score[0] - score[1])
    assert cross_feats.loc[0, 'top1_gap'] >= 0.0

    # Source flags
    assert cross_feats.loc[0, 'source_flag'] == 'S2'
    assert cross_feats.loc[0, 'candidate_source_id'] == 0
    assert cross_feats.loc[2, 'source_flag'] == 'S3'
    assert cross_feats.loc[2, 'candidate_source_id'] == 1

    # Exact tier
    assert cross_feats.loc[0, 'is_exact_tier'] == 1


def test_assemble_labels(synthetic_data):
    s1_df, s2_df, s3_df, candidate_pairs_df, ground_truth_df = synthetic_data
    features_df = build_features(candidate_pairs_df, s1_df, s2_df, s3_df)
    labeled_df = assemble_labels(features_df, ground_truth_df)

    assert 'is_true_match' in labeled_df.columns
    # Row 0 (S1_001, S2_001) -> True match (1)
    assert labeled_df.loc[0, 'is_true_match'] == 1
    # Row 1 (S1_001, S2_002) -> True match (1)
    assert labeled_df.loc[1, 'is_true_match'] == 1
    # Row 2 (S1_001, S3_001) -> False match / negative candidate (0)
    assert labeled_df.loc[2, 'is_true_match'] == 0
    # Row 3 (S1_002, S2_003) -> True match (1)
    assert labeled_df.loc[3, 'is_true_match'] == 1
    # Row 4 (S1_003, S3_002) -> True match (1)
    assert labeled_df.loc[4, 'is_true_match'] == 1


def test_build_features_end_to_end_no_nans(synthetic_data, tmp_path):
    s1_df, s2_df, s3_df, candidate_pairs_df, _ = synthetic_data
    out_parquet = str(tmp_path / "test_features.parquet")
    
    features_df = build_features(
        candidate_pairs_df, s1_df, s2_df, s3_df, output_path=out_parquet
    )

    assert len(features_df) == len(candidate_pairs_df)
    assert 'source1_entity_id' in features_df.columns
    assert 'candidate_entity_id' in features_df.columns

    # Verify 0 NaNs in numeric feature columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    assert features_df[numeric_cols].isna().sum().sum() == 0

    # Verify parquet file was written and is readable
    loaded_df = pd.read_parquet(out_parquet)
    assert len(loaded_df) == len(features_df)
