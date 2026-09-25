import re
import pandas as pd
from typing import Set, Optional

# Non-Latin / Regional transliterated suffix variants identified in EDA and diagnostics
REGIONAL_SUFFIX_PATTERNS = [
    r'\bpraivarr\s+limirrad\b',
    r'\bpraibheta\s+limiteDa\b',
    r'\bpraibheta\s+limiteda\b',
    r'\bpraibheta\b',
    r'\bpraiveta\b',
    r'\bpraivet\b',
    r'\bpraivarr\b',
    r'\blimirrad\b',
    r'\blimidhedh\b',
    r'\blimidhed\b',
    r'\blimiteDa\b',
    r'\blimiteda\b',
    r'\bbhiraivedh\b',
    r'\bbhiraived\b',
]

_COMPILED_REGIONAL_SUFFIXES = [re.compile(p, re.IGNORECASE) for p in REGIONAL_SUFFIX_PATTERNS]


def should_apply_phonetic_reduction(script: Optional[str]) -> bool:
    """
    Determines if phonetic reduction should be applied based on the detected script.
    Applies to all non-Latin scripts (Devanagari, Gujarati, Tamil, Telugu, Kannada,
    Bengali, Malayalam, Oriya, Gurmukhi, Mixed, etc.).
    Returns False for 'latin', 'empty', or None.
    """
    if not script:
        return False
    s = str(script).strip().lower()
    return s not in ('latin', 'empty', 'unknown', 'none', '')


def get_phonetic_reduced(normalized_name: str) -> str:
    """
    Transforms an already-normalized business name into a canonical phonetically-reduced
    representation to reconcile Indic-to-Latin transliteration divergence.
    
    Idempotent and safe for empty/null strings.
    """
    if not normalized_name or not isinstance(normalized_name, str):
        return ""
    
    t = normalized_name.lower().strip()
    if not t:
        return ""

    # 1. Clean regional transliterated legal suffix remnants (Tamil, Malayalam, Bengali, Kannada)
    for pat in _COMPILED_REGIONAL_SUFFIXES:
        t = pat.sub('', t)

    # 2. Indic ITRANS nasal endings: 'mga' -> 'ng', 'imga' -> 'ing', 'nga' -> 'ng'
    t = re.sub(r'imga\b', 'ing', t)
    t = re.sub(r'mga\b', 'ng', t)
    t = re.sub(r'nga\b', 'ng', t)
    t = re.sub(r'mga', 'ng', t)

    # 3. Nasal assimilation: 'm' followed by consonants -> 'n' (e.g. 'imdiyana' -> 'indiyana', 'phainemsa' -> 'phainensa')
    t = re.sub(r'm(?=[b-df-hj-np-tv-z])', 'n', t)

    # 4. Phonetic consonant equivalences & aspirate simplifications
    t = re.sub(r'ph', 'f', t)
    t = re.sub(r'v', 'w', t)
    t = re.sub(r'bh', 'b', t)
    t = re.sub(r'dh', 'd', t)
    t = re.sub(r'th', 't', t)
    t = re.sub(r'kh', 'k', t)
    t = re.sub(r'gh', 'g', t)
    t = re.sub(r'ch', 'c', t)
    t = re.sub(r'sh', 's', t)
    t = re.sub(r'z', 's', t)
    t = re.sub(r'c([eiy])', r's\1', t)
    t = re.sub(r'c', 'k', t)
    t = re.sub(r'q', 'k', t)
    t = re.sub(r'x', 'ks', t)

    # 5. Vowel mergers (long/short vowel neutralization common in romanization)
    t = re.sub(r'ee|ea|ie|ei|ii', 'i', t)
    t = re.sub(r'oo|ou|uu', 'u', t)
    t = re.sub(r'ai|ay', 'e', t)
    t = re.sub(r'au|aw', 'o', t)

    # 6. Collapse consecutive duplicate characters (e.g. 'tt' -> 't', 'pp' -> 'p')
    t = re.sub(r'(.)\1+', r'\1', t)

    # 7. Terminal schwa deletion: drop trailing inherent short 'a' on words after consonants
    # Exclude 'y' to prevent truncating diphthong transcriptions (e.g. 'midiya' -> 'media')
    words = t.split()
    processed_words = []
    for w in words:
        if len(w) >= 3 and w.endswith('a') and w[-2] in 'bcdfghjklmnpqrstvwxz':
            w = w[:-1]
        processed_words.append(w)
    
    t = ' '.join(processed_words)

    # Collapse whitespace and return
    return re.sub(r'\s+', ' ', t).strip()


def get_phonetic_ngrams(phonetic_name: str, n: int = 2) -> Set[str]:
    """
    Extracts character n-grams from a phonetically reduced string.
    Default is bigrams (n=2).
    """
    if not phonetic_name:
        return set()
    s = str(phonetic_name).strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def phonetic_similarity(name_a: str, name_b: str, n: int = 2) -> float:
    """
    Computes character n-gram Jaccard similarity between the phonetically-reduced
    forms of two normalized names.
    """
    red_a = get_phonetic_reduced(name_a)
    red_b = get_phonetic_reduced(name_b)
    
    if not red_a and not red_b:
        return 1.0
    if not red_a or not red_b:
        return 0.0
    
    ng_a = get_phonetic_ngrams(red_a, n=n)
    ng_b = get_phonetic_ngrams(red_b, n=n)
    
    if not ng_a and not ng_b:
        return 1.0
    if not ng_a or not ng_b:
        return 0.0
    
    intersection = len(ng_a.intersection(ng_b))
    union = len(ng_a.union(ng_b))
    return float(intersection) / float(union) if union > 0 else 0.0


def apply_phonetic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selectively generates phonetic features for a dataframe with existing `name_norm`
    and `name_script` columns.
    
    Adds:
      - name_phonetic: string (phonetically reduced where non-Latin, copied from name_norm if Latin/empty)
      - name_phonetic_applied: boolean (True if reduction was applied, False otherwise)
    
    Preserves all existing columns.
    """
    df_out = df.copy()
    
    if 'name_norm' not in df_out.columns or 'name_script' not in df_out.columns:
        raise ValueError("DataFrame must contain 'name_norm' and 'name_script' columns before applying phonetic features.")

    # Determine which rows qualify for reduction
    mask_apply = df_out['name_script'].apply(should_apply_phonetic_reduction)
    df_out['name_phonetic_applied'] = mask_apply
    
    # Initialize name_phonetic with name_norm as default
    df_out['name_phonetic'] = df_out['name_norm'].fillna('')
    
    # Only transform the subset of unique strings where reduction applies
    qualifying_names = df_out.loc[mask_apply, 'name_norm'].dropna().unique()
    
    if len(qualifying_names) > 0:
        reduction_map = {name: get_phonetic_reduced(name) for name in qualifying_names}
        df_out.loc[mask_apply, 'name_phonetic'] = df_out.loc[mask_apply, 'name_norm'].map(reduction_map).fillna('')
        
    return df_out
