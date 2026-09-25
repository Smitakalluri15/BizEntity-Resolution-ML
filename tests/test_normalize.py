"""Comprehensive Unit Tests for the Business Entity Resolution Normalization Module.

All test cases are derived directly from empirical patterns cataloged in Phase 1 EDA:
- name_noise_catalog.md
- address_noise_catalog.md
- eda_summary.md
"""

import difflib
import pytest
import pandas as pd
from src.normalize import (
    detect_script,
    transliterate_to_latin,
    normalize_name,
    normalize_name_tokens,
    get_name_char_ngrams,
    extract_url_or_handle,
    normalize_address,
    get_address_tokens,
    get_address_char_ngrams,
    is_landmark_address,
    extract_postal_code,
    normalize_country,
    apply_normalization,
)


# =============================================================================
# 1. Legal Suffix Stripping & Position Transposition Tests
# =============================================================================
def test_legal_suffix_stripping_and_transposition():
    """Verify that legal suffixes are stripped regardless of position (front, middle, end, brackets)."""
    # Suffix at end
    assert normalize_name("Peridos Investment LLC") == "peridos investment"
    # Suffix moved to front
    assert normalize_name("LLC Peridos Investment") == "peridos investment"
    # Suffix inside brackets / parentheses
    assert normalize_name("Sanskruti Technologiesprivate Pvt (Ltd)") == "sanskruti technologiesprivate"
    assert normalize_name("Spears, Philippe, L.C.S.W. [LLC]") == "spears philippe l c s w"
    # Multi-word suffix
    assert normalize_name("Bombay Investment Private Limited") == "bombay investment"
    assert normalize_name("Naidu Balaji Private Limited") == "naidu balaji"
    assert normalize_name("Basalt Corp") == "basalt"
    assert normalize_name("Olanola Allstate Clinic Ltd") == "olanola allstate clinic"


# =============================================================================
# 2. Conjunction Normalization ('&', 'and', '+')
# =============================================================================
def test_conjunction_normalization():
    """Verify '&' and '+' are normalized to ' and ' with clean spacing."""
    s1 = normalize_name("Valencia, Knicely & Prado")
    s2 = normalize_name("Valencia Knicely and Prado")
    assert s1 == s2 == "valencia knicely and prado"

    s3 = normalize_name("Willetts, Curry + Crossman Clinic")
    assert s3 == "willetts curry and crossman clinic"


# =============================================================================
# 3. Word-Order Token Set Equality
# =============================================================================
def test_word_order_token_set_equality():
    """Verify normalize_name_tokens handles word order transpositions."""
    s1_tokens = normalize_name_tokens("Peridos Investment LLC")
    s2_tokens = normalize_name_tokens("LLC Peridos Investment")
    assert set(s1_tokens) == set(s2_tokens) == {"investment", "peridos"}

    s3_tokens = normalize_name_tokens("Novel Technology Center")
    s4_tokens = normalize_name_tokens("Technology Novel Center")
    assert set(s3_tokens) == set(s4_tokens) == {"center", "novel", "technology"}


# =============================================================================
# 4. Address Abbreviation Expansion
# =============================================================================
def test_address_abbreviation_expansion():
    """Verify expansion of address abbreviations based on address_noise_catalog.md."""
    # St -> street, Ave -> avenue, Rd -> road, Dr -> drive, Ct -> court, Ln -> lane
    assert normalize_address("2756 GREELEY ST, SCHENECTADY, NY") == "2756 greeley street schenectady ny"
    assert normalize_address("243 MOUNTAIN AVE, ARLINGTON, MA") == "243 mountain avenue arlington ma"
    assert normalize_address("861 RIVER RD, WINDHAM, ME") == "861 river road windham me"
    assert normalize_address("1129 Cherry Ridge Dr, Sugarcreek, OH") == "1129 cherry ridge drive sugarcreek oh"
    assert normalize_address("2488 Pierce Ct, Simi Valley, CA") == "2488 pierce court simi valley ca"
    assert normalize_address("10866 VERBENA LN, SCOTTSDALE, AZ") == "10866 verbena lane scottsdale az"
    
    # Landmark 'nr' -> 'near'
    norm_nr = normalize_address("Pl 42 New Mangalwar Peth, Nr Ladka T Pump, Pune, Maharashtra")
    assert "near ladka t pump" in norm_nr


# =============================================================================
# 5. Cross-Script Transliteration (Real EDA Examples)
# =============================================================================
@pytest.mark.parametrize("s1_latin, indic_text, script_name", [
    ("Great Engineering Limited", "ग्रेट इंजीनियरिंग लिमिटेड", "devanagari"),
    ("Galaxy Finance Private Limited", "गैलेक्सी फाइनेंस प्राइवेट लिमिटेड", "devanagari"),
    ("Prime Trading", "प्राइम ट्रेडिंग", "devanagari"),
    ("My Power Private Limited", "માય પાવર પ્રાઇવેટ લિમિટેડ", "gujarati"),
    ("Sree Power Pvt Ltd", "ஸ்ரீ பவர் பிரைவேட் லிமிடெட்", "tamil"),
])
def test_cross_script_transliteration_similarity(s1_latin, indic_text, script_name):
    """Test at least 5 real cross-script cases from name_noise_catalog.md.

    Asserts that normalized transliteration brings the text into Latin space with
    meaningful string / character similarity (> 0.35 SequenceMatcher ratio) against S1 Latin.
    """
    detected = detect_script(indic_text)
    assert detected == script_name

    norm_s1 = normalize_name(s1_latin)
    norm_indic = normalize_name(indic_text)

    # SequenceMatcher edit similarity ratio
    sim_ratio = difflib.SequenceMatcher(None, norm_s1, norm_indic).ratio()

    print(f"\nTransliteration evaluation:")
    print(f"  S1:    '{s1_latin}' -> Norm: '{norm_s1}'")
    print(f"  Indic: '{indic_text}' -> Norm: '{norm_indic}'")
    print(f"  Similarity Ratio: {sim_ratio:.4f}")

    # Must be meaningfully non-zero and above 0.35
    assert sim_ratio >= 0.35, f"Expected similarity ratio >= 0.35, got {sim_ratio:.4f}"


# =============================================================================
# 6. URL and Social Handle Extraction & Stripping
# =============================================================================
def test_url_and_handle_extraction():
    """Verify extraction and separation of URLs and handles from business names."""
    # Domain in name
    url_1 = extract_url_or_handle("wavesolora.com")
    assert url_1 == "wavesolora.com"
    assert normalize_name("wavesolora.com") == ""

    # Embedded domain with trade name
    url_2 = extract_url_or_handle("Malad Exports Private Limited | www.maladexpo.com")
    assert "www.maladexpo.com" in url_2
    assert normalize_name("Malad Exports Private Limited | www.maladexpo.com") == "malad exports"

    # Social handle
    handle = extract_url_or_handle("@qualityasset")
    assert handle == "@qualityasset"
    assert normalize_name("@qualityasset") == ""

    # Clean name has no URL/handle
    assert extract_url_or_handle("Quality Asset Solutions, LLC") is None


# =============================================================================
# 7. OCR Digit-for-Letter Substitution Repair & Safeguards
# =============================================================================
def test_ocr_substitution_repair_and_safeguards():
    """Verify OCR substitutions (5->s, 6->g) without corrupting alphanumeric names."""
    # Positive tests: confirmed catalog noise
    assert normalize_name("5olutions Al Spaces Center") == "solutions al spaces center"
    assert normalize_name("RK 6reat Advisors LLC") == "rk great advisors"

    # Negative tests: legitimate numbers MUST NOT be corrupted
    assert normalize_name("7-Eleven, Inc.") == "7 eleven"
    assert normalize_name("3M Company") == "3m"
    assert normalize_name("2598 Center Street Management LLC") == "2598 center street management"
    assert normalize_name("Unit 102") == "unit 102"
    assert normalize_name("1004 7th Street") == "1004 7th street"


# =============================================================================
# 8. Null / Missing / Placeholder Handling
# =============================================================================
@pytest.mark.parametrize("val", [None, "", "nan", "NaN", "NULL", "<NULL>", "N/A", "   "])
def test_null_and_placeholder_handling(val):
    """Verify robust handling of nulls, NaNs, and placeholder tokens across all functions."""
    assert normalize_name(val) == ""
    assert normalize_name_tokens(val) == []
    assert get_name_char_ngrams(val) == set()
    assert extract_url_or_handle(val) is None

    assert normalize_address(val) == ""
    assert get_address_tokens(val) == []
    assert get_address_char_ngrams(val) == set()
    assert is_landmark_address(val) is False
    assert extract_postal_code(val) is None

    assert normalize_country(val) == ""


# =============================================================================
# 9. Landmark Address Detection
# =============================================================================
def test_landmark_address_detection():
    """Verify landmark detection on real examples from address_noise_catalog.md."""
    assert is_landmark_address("Pl 42 New Mangalwar Peth, Nr Ladka T Pump, Pune, Maharashtra") is True
    assert is_landmark_address("Behind Hinduja College, New Charni Road, Mumbai") is True
    assert is_landmark_address("Opposite Post Office, MG Road") is True
    assert is_landmark_address("P. No. 17, Kartarpura Phatak Ke Pass, 22- Godam, Jaipur") is True
    assert is_landmark_address("C/O Nanhak S/O Bansu, Siddharth Nagar") is True

    # Standard street addresses should NOT be flagged as landmarks
    assert is_landmark_address("2488 Pierce Court, Simi Valley, CA") is False
    assert is_landmark_address("1739 Labrador Drive, Costa Mesa, CA") is False


# =============================================================================
# 10. Open-Set Country Normalization
# =============================================================================
def test_country_normalization_open_set():
    """Verify open-set country normalization (handles training and unseen test countries)."""
    assert normalize_country("US") == "us"
    assert normalize_country("India") == "india"
    assert normalize_country("  France  ") == "france"
    assert normalize_country("UNITED-STATES") == "united-states"
    assert normalize_country(None) == ""


# =============================================================================
# 11. Full DataFrame Batch Normalization
# =============================================================================
def test_apply_normalization_batch():
    """Verify apply_normalization() produces all required columns without mutating originals."""
    sample_df = pd.DataFrame({
        "entity_id": ["S1-101", "S2-202", "S3-303"],
        "business_name": [
            "Quality Asset Solutions, LLC",
            "@qualityasset",
            "ग्रेट इंजीनियरिंग लिमिटेड",
        ],
        "business_address": [
            "2488 Pierce Court, Simi Valley, CA 93065",
            None,
            "Nr Ladka T Pump, Pune, Maharashtra 411001",
        ],
        "country": ["US", "US", "India"],
    })

    norm_df = apply_normalization(sample_df)

    # Check all required columns exist
    expected_cols = [
        "entity_id", "business_name", "business_address", "country",
        "name_raw", "name_norm", "name_tokens", "name_script", "name_embedded_url",
        "address_raw", "address_norm", "address_tokens", "address_script",
        "postal_code", "is_landmark_address", "country_raw", "country_norm"
    ]
    for col in expected_cols:
        assert col in norm_df.columns, f"Missing expected column: {col}"

    # Verify values
    assert norm_df.loc[0, "name_norm"] == "quality asset solutions"
    assert norm_df.loc[0, "postal_code"] == "93065"
    assert norm_df.loc[1, "name_norm"] == ""
    assert norm_df.loc[1, "name_embedded_url"] == "@qualityasset"
    assert norm_df.loc[1, "address_norm"] == ""
    assert norm_df.loc[2, "name_script"] == "devanagari"
    assert bool(norm_df.loc[2, "is_landmark_address"]) is True
    assert norm_df.loc[2, "postal_code"] == "411001"
