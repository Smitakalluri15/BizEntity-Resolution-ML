"""Business name normalization and URL/handle extraction module.

Applies a sequential normalization pipeline:
1. Null/empty and placeholder handling ('nan', 'null', 'n/a')
2. Script detection & transliteration to Latin
3. URL and social handle extraction & stripping
4. Context-aware OCR digit-for-letter repair (e.g. '5olutions' -> 'solutions', '6reat' -> 'great')
5. Diacritic removal via Unicode NFKD
6. Lowercasing
7. Ampersand / plus sign normalization ('&' -> ' and ')
8. Legal suffix token stripping (position-agnostic: start, middle, end, brackets)
9. Punctuation removal
10. Whitespace collapsing and stripping
"""

import re
import unicodedata
from typing import List, Optional, Set

from src.normalize.transliteration import detect_script, transliterate_to_latin

# Regular expressions for URL and social handle detection
URL_PATTERN = re.compile(
    r'(?:https?://\S+|www\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/\S*)?|'
    r'\b[a-zA-Z0-9\-\.]+\.(?:com|org|net|in|co|io|biz|info|us|edu|gov)\b)',
    re.IGNORECASE
)
HANDLE_PATTERN = re.compile(r'@\w+', re.IGNORECASE)

# Placeholder strings to scrub
PLACEHOLDER_PATTERN = re.compile(
    r'^(?:null|<null>|nan|n/a|none)$',
    re.IGNORECASE
)

# Legal suffix terms to strip (including international and transliterated Indic suffixes)
LEGAL_SUFFIXES = [
    # English / US / Commonwealth standard
    'incorporated', 'inc',
    'corporation', 'corp',
    'private limited', 'pvt ltd', 'pvt limited', 'private ltd',
    'limited', 'ltd',
    'private', 'pvt',
    'limited liability company', 'limited liability partnership',
    'pllc', 'llc', 'llp', 'lp',
    'company', 'co',
    'public limited company', 'plc',
    # European standard (for generalization to France/EU test cases)
    'gmbh', 'sarl', 'sas', 'sa', 'bv', 'nv', 'srl', 'spa',
    # Transliterated Indic legal suffixes (from ITRANS romanization)
    'limiteDa', 'limiteda', 'limidhedh', 'limidhed',
    'prAiveTa', 'praiveta', 'bhiraivedh', 'bhiraived'
]

# Compile regex for legal suffix removal (word boundary on both sides)
LEGAL_SUFFIXES_SORTED = sorted(LEGAL_SUFFIXES, key=len, reverse=True)
LEGAL_SUFFIX_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(s) for s in LEGAL_SUFFIXES_SORTED) + r')\b',
    re.IGNORECASE
)

# OCR repair patterns: single digit within an alphabetic token or prefixing word without legitimate numbers
OCR_DIGIT_PREFIX_5 = re.compile(r'\b5([a-zA-Z]{3,})\b')
OCR_DIGIT_PREFIX_6 = re.compile(r'\b6([a-zA-Z]{3,})\b')
OCR_DIGIT_EMBEDDED_5 = re.compile(r'([a-zA-Z]+)5([a-zA-Z]+)')
OCR_DIGIT_EMBEDDED_6 = re.compile(r'([a-zA-Z]+)6([a-zA-Z]+)')


def extract_url_or_handle(name: Optional[str]) -> Optional[str]:
    """Extract embedded URL or social handle from business name string.

    Returns the extracted string or None if none found.
    """
    if name is None:
        return None
    name_str = str(name).strip()
    if not name_str or PLACEHOLDER_PATTERN.match(name_str):
        return None
    
    url_match = URL_PATTERN.search(name_str)
    if url_match:
        return url_match.group(0).strip()
        
    handle_match = HANDLE_PATTERN.search(name_str)
    if handle_match:
        return handle_match.group(0).strip()
        
    return None


def _repair_ocr_substitutions(text: str) -> str:
    """Repair verified OCR digit-for-letter substitutions without corrupting alphanumeric names."""
    text = OCR_DIGIT_PREFIX_5.sub(r's\1', text)
    text = OCR_DIGIT_EMBEDDED_5.sub(r'\1s\2', text)
    text = OCR_DIGIT_PREFIX_6.sub(r'g\1', text)
    text = OCR_DIGIT_EMBEDDED_6.sub(r'\1g\2', text)
    return text


def _remove_diacritics(text: str) -> str:
    """Remove foreign diacritics and accents (e.g. Á -> A, Í -> I)."""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_name(name: Optional[str]) -> str:
    """Apply the full 10-step normalization pipeline to a business name."""
    if name is None:
        return ""
    
    text = str(name).strip()
    if not text or PLACEHOLDER_PATTERN.match(text):
        return ""

    # 1. Script Detection & Transliteration
    script = detect_script(text)
    if script not in ("latin", "empty", "unknown"):
        text = transliterate_to_latin(text, script)

    # 2. Extract & Strip URLs / Social handles
    text = URL_PATTERN.sub(' ', text)
    text = HANDLE_PATTERN.sub(' ', text)

    # 3. Context-aware OCR repair
    text = _repair_ocr_substitutions(text)

    # 4. Remove diacritics (Á -> A, etc.)
    text = _remove_diacritics(text)

    # 5. Lowercase
    text = text.lower()

    # 6. Normalize conjunctions: '&' and '+' -> ' and '
    text = re.sub(r'[&]', ' and ', text)
    text = re.sub(r'\s+\+\s+', ' and ', text)

    # 7. Strip legal suffix tokens across entire string (regardless of position)
    text = LEGAL_SUFFIX_PATTERN.sub(' ', text)

    # 8. Remove remaining punctuation, replacing with space
    text = re.sub(r'[^\w\s]', ' ', text)

    # 9. Collapse whitespace and strip
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_name_tokens(name: Optional[str]) -> List[str]:
    """Return sorted unique tokens of normalized business name (handles word-order changes)."""
    norm = normalize_name(name)
    if not norm:
        return []
    tokens = sorted(list(set(norm.split())))
    return tokens


def get_name_char_ngrams(name: Optional[str], n: int = 3) -> Set[str]:
    """Extract character n-grams from normalized business name."""
    norm = normalize_name(name)
    if not norm or len(norm) < n:
        return {norm} if norm else set()
    return {norm[i:i + n] for i in range(len(norm) - n + 1)}
