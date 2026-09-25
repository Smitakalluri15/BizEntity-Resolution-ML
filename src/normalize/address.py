"""Business address normalization, landmark detection, and postal code extraction module.

Applies a sequential address normalization pipeline:
1. Null / empty handling (robust to ~3.35% null address rate)
2. Script detection & transliteration (e.g., Indian states in Indic script)
3. Placeholder value scrubbing ('NULL', '<NULL>', 'nan', 'N/A')
4. Diacritic stripping via Unicode NFKD
5. Lowercasing
6. Address abbreviation expansion (st -> street, rd -> road, ave -> avenue, nr -> near, etc.)
7. Punctuation removal
8. Whitespace collapsing and stripping
"""

import re
import unicodedata
from typing import List, Optional, Set

from src.normalize.transliteration import detect_script, transliterate_to_latin

# Landmark detection keywords
LANDMARK_PATTERN = re.compile(
    r'\b(?:near|behind|nr|opp|opposite|next\s+to|pass\s+ke\s+pass|phatak\s+ke\s+pass|ke\s+pass|in\s+front\s+of|adjacent\s+to|beside|c/o)\b',
    re.IGNORECASE
)

# Postal code patterns (US 5/9-digit and India 6-digit PIN)
US_ZIP_PATTERN = re.compile(r'\b\d{5}(?:-\d{4})?\b')
INDIA_PIN_PATTERN = re.compile(r'\b[1-9]\d{5}\b')

# Placeholders to remove
PLACEHOLDER_PATTERN = re.compile(
    r'\b(?:null|<null>|nan|n/a|none)\b',
    re.IGNORECASE
)

# Address abbreviation dictionary (word-boundary matched)
# Mappings established directly from address_noise_catalog.md
ADDRESS_ABBREVIATIONS = {
    'st': 'street',
    'str': 'street',
    'rd': 'road',
    'ave': 'avenue',
    'av': 'avenue',
    'apt': 'apartment',
    'blvd': 'boulevard',
    'dr': 'drive',
    'ln': 'lane',
    'ct': 'court',
    'fl': 'floor',
    'flr': 'floor',
    'ste': 'suite',
    'bldg': 'building',
    'pkwy': 'parkway',
    'pl': 'place',
    'hwy': 'highway',
    'nr': 'near',
    'opp': 'opposite',
    'hn': 'house number',
}

# Compile abbreviation regex
ABBREV_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in ADDRESS_ABBREVIATIONS.keys()) + r')\b',
    re.IGNORECASE
)


def is_landmark_address(address: Optional[str]) -> bool:
    """Determine whether an address contains landmark or relative location indicators."""
    if address is None:
        return False
    addr_str = str(address).strip()
    if not addr_str or addr_str.lower() in ('nan', 'null', '<null>', 'n/a', 'none'):
        return False
    return bool(LANDMARK_PATTERN.search(addr_str))


def extract_postal_code(address: Optional[str]) -> Optional[str]:
    """Extract US ZIP code or Indian PIN code from address string.

    Returns the postal code string or None if not present.
    """
    if address is None:
        return None
    addr_str = str(address).strip()
    if not addr_str or addr_str.lower() in ('nan', 'null', '<null>', 'n/a', 'none'):
        return None
    
    # Check India PIN first (strict 6 digits)
    india_match = INDIA_PIN_PATTERN.search(addr_str)
    if india_match:
        return india_match.group(0)

    # Check US Zip
    us_match = US_ZIP_PATTERN.search(addr_str)
    if us_match:
        return us_match.group(0)

    return None


def _expand_abbreviations(text: str) -> str:
    """Expand address abbreviations to their canonical full forms."""
    def _replace_match(m):
        tok = m.group(0).lower()
        return ADDRESS_ABBREVIATIONS.get(tok, tok)

    return ABBREV_PATTERN.sub(_replace_match, text)


def _remove_diacritics(text: str) -> str:
    """Remove foreign diacritics and accents."""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_address(address: Optional[str]) -> str:
    """Apply the full address normalization pipeline."""
    if address is None:
        return ""
    
    text = str(address).strip()
    if not text or text.lower() in ('nan', 'null', '<null>', 'n/a', 'none'):
        return ""

    # 1. Script Detection & Transliteration
    script = detect_script(text)
    if script not in ("latin", "empty", "unknown"):
        text = transliterate_to_latin(text, script)

    # 2. Scrub literal placeholder indicators (null, <null>, nan, n/a)
    text = PLACEHOLDER_PATTERN.sub(' ', text)

    # 3. Remove diacritics
    text = _remove_diacritics(text)

    # 4. Lowercase
    text = text.lower()

    # 5. Expand abbreviations
    text = _expand_abbreviations(text)

    # 6. Remove punctuation, replacing with space
    text = re.sub(r'[^\w\s]', ' ', text)

    # 7. Collapse whitespace and strip
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_address_tokens(address: Optional[str]) -> List[str]:
    """Return token list of normalized address."""
    norm = normalize_address(address)
    if not norm:
        return []
    return norm.split()


def get_address_char_ngrams(address: Optional[str], n: int = 3) -> Set[str]:
    """Extract character n-grams from normalized address."""
    norm = normalize_address(address)
    if not norm or len(norm) < n:
        return {norm} if norm else set()
    return {norm[i:i + n] for i in range(len(norm) - n + 1)}
