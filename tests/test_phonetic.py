import pytest
import pandas as pd
import difflib

from src.normalize.phonetic import (
    get_phonetic_reduced,
    get_phonetic_ngrams,
    phonetic_similarity,
    should_apply_phonetic_reduction,
    apply_phonetic_features
)
from src.normalize.name import get_name_char_ngrams


def calc_char_jaccard(a: str, b: str, n: int = 2) -> float:
    ng_a = get_name_char_ngrams(a, n=n)
    ng_b = get_name_char_ngrams(b, n=n)
    if not ng_a and not ng_b:
        return 1.0
    if not ng_a or not ng_b:
        return 0.0
    return len(ng_a.intersection(ng_b)) / len(ng_a.union(ng_b))


def test_should_apply_phonetic_reduction():
    # False cases
    assert should_apply_phonetic_reduction('latin') is False
    assert should_apply_phonetic_reduction('Latin') is False
    assert should_apply_phonetic_reduction('empty') is False
    assert should_apply_phonetic_reduction('unknown') is False
    assert should_apply_phonetic_reduction(None) is False
    assert should_apply_phonetic_reduction('') is False

    # True cases
    assert should_apply_phonetic_reduction('devanagari') is True
    assert should_apply_phonetic_reduction('gujarati') is True
    assert should_apply_phonetic_reduction('tamil') is True
    assert should_apply_phonetic_reduction('telugu') is True
    assert should_apply_phonetic_reduction('kannada') is True
    assert should_apply_phonetic_reduction('bengali') is True
    assert should_apply_phonetic_reduction('malayalam') is True
    assert should_apply_phonetic_reduction('oriya') is True
    assert should_apply_phonetic_reduction('gurmukhi') is True
    assert should_apply_phonetic_reduction('mixed') is True


@pytest.mark.parametrize('s1_norm, matched_norm', [
    ('jain foods', 'jaina phudsa'),
    ('shivam foods', 'shivama phudsa'),
    ('shivam bombay infotech', 'shivama bombe inphoteka'),
    ('classic marketing', 'klasika marketimga'),
    ('ram impex', 'rama impeksa'),
    ('future foods', 'phyuchara phudsa'),
    ('indian power', 'imdiyana pavara'),
    ('vision engineering', 'vijana imjiniyarimga'),
    ('tech care', 'tek ker praivet'),
    ('white media', 'vairr midiya praivarr limirrad'),
    ('bright services', 'vraita sarbhisesa praibheta'),
])
def test_phonetic_similarity_gain_on_real_ground_truth(s1_norm, matched_norm):
    """
    Asserts that applying phonetic reduction increases character-level similarity
    (SequenceMatcher ratio or bigram Jaccard) for known true match pairs.
    """
    raw_seq = difflib.SequenceMatcher(None, s1_norm, matched_norm).ratio()
    
    red_s1 = get_phonetic_reduced(s1_norm)
    red_m = get_phonetic_reduced(matched_norm)
    phonetic_seq = difflib.SequenceMatcher(None, red_s1, red_m).ratio()
    
    # Assert improvement
    assert phonetic_seq >= raw_seq, f"Expected phonetic seq ({phonetic_seq}) >= raw seq ({raw_seq}) for '{s1_norm}' vs '{matched_norm}'"
    
    # Also verify phonetic_similarity helper function
    raw_bi = calc_char_jaccard(s1_norm, matched_norm, n=2)
    phon_bi = phonetic_similarity(s1_norm, matched_norm, n=2)
    assert phon_bi >= raw_bi, f"Expected phonetic bigram ({phon_bi}) >= raw bigram ({raw_bi})"


def test_idempotence_and_edge_cases():
    test_cases = [
        'jaina phudsa',
        'imdiyana pavara',
        'klasika marketimga',
        'vairr midiya praivarr limirrad',
        '7 eleven',
        '3m tech'
    ]
    for text in test_cases:
        first_pass = get_phonetic_reduced(text)
        second_pass = get_phonetic_reduced(first_pass)
        assert first_pass == second_pass, f"Phonetic reduction not idempotent for '{text}': '{first_pass}' != '{second_pass}'"

    # Null / empty handling
    assert get_phonetic_reduced('') == ''
    assert get_phonetic_reduced(None) == ''
    assert get_phonetic_reduced('   ') == ''
    assert get_phonetic_ngrams('') == set()
    assert get_phonetic_ngrams(None) == set()
    assert phonetic_similarity('', '') == 1.0
    assert phonetic_similarity('abc', '') == 0.0


def test_apply_phonetic_features_dataframe():
    data = {
        'entity_id': ['S1-1', 'S2-2', 'S3-3', 'S2-4'],
        'name_norm': ['acme enterprise', 'jaina phudsa', 'global logistics', 'vairr midiya praivarr limirrad'],
        'name_script': ['latin', 'devanagari', 'latin', 'malayalam'],
        'country_norm': ['us', 'india', 'us', 'india']
    }
    df = pd.DataFrame(data)
    
    df_res = apply_phonetic_features(df)
    
    # Check column existence
    assert 'name_phonetic' in df_res.columns
    assert 'name_phonetic_applied' in df_res.columns
    assert len(df_res) == 4
    
    # Check Latin row (index 0)
    assert df_res.loc[0, 'name_phonetic_applied'] == False
    assert df_res.loc[0, 'name_phonetic'] == 'acme enterprise'
    
    # Check Devanagari row (index 1)
    assert df_res.loc[1, 'name_phonetic_applied'] == True
    assert df_res.loc[1, 'name_phonetic'] == 'jen fuds'
    
    # Check Latin row (index 2)
    assert df_res.loc[2, 'name_phonetic_applied'] == False
    assert df_res.loc[2, 'name_phonetic'] == 'global logistics'
    
    # Check Malayalam row (index 3) - regional suffix stripped and phonetically reduced
    assert df_res.loc[3, 'name_phonetic_applied'] == True
    assert 'praivarr' not in df_res.loc[3, 'name_phonetic']
    assert 'limirrad' not in df_res.loc[3, 'name_phonetic']
    assert df_res.loc[3, 'name_phonetic'] == 'wer midiya'
