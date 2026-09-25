"""
TF-IDF cosine similarity features for candidate pairs in Business Entity Resolution.
"""

from typing import Dict, Optional, Iterable, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp


def fit_tfidf_vectorizers(
    corpus_names: Iterable[str],
    corpus_addrs: Iterable[str],
    max_features_word: int = 50000,
    max_features_char: int = 50000
) -> Dict[str, TfidfVectorizer]:
    """
    Fits three specialized TF-IDF vectorizers across the corpus:
      1. name_word_vec: Word n-grams (1, 2) on business names
      2. name_char_vec: Character n-grams (2, 4) on business names
      3. addr_word_vec: Word n-grams (1, 2) on business addresses

    Uses L2 normalization by default so row-wise dot products equal cosine similarity.
    """
    clean_names = [str(x) if (x and pd.notna(x)) else '' for x in corpus_names]
    clean_addrs = [str(x) if (x and pd.notna(x)) else '' for x in corpus_addrs]

    name_word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features_word,
        norm='l2',
        dtype=np.float32
    )
    name_word_vec.fit(clean_names)

    name_char_vec = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        max_features=max_features_char,
        norm='l2',
        dtype=np.float32
    )
    name_char_vec.fit(clean_names)

    addr_word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features_word,
        norm='l2',
        dtype=np.float32
    )
    addr_word_vec.fit(clean_addrs)

    return {
        'name_word': name_word_vec,
        'name_char': name_char_vec,
        'addr_word': addr_word_vec,
    }


def _sparse_pairwise_cosine_similarity(vec: TfidfVectorizer, texts1: list, texts2: list) -> np.ndarray:
    """
    Computes row-wise cosine similarity between aligned pairs of texts (texts1[i] vs texts2[i]).
    Since TfidfVectorizer with norm='l2' produces unit vectors, cosine similarity is
    the elementwise sparse dot product: (A * B).sum(axis=1).
    """
    if len(texts1) == 0 or len(texts2) == 0:
        return np.array([], dtype=np.float32)

    # Transform both columns
    A = vec.transform(texts1)
    B = vec.transform(texts2)

    # Elementwise sparse multiplication and row sum
    # A.multiply(B) performs sparse elementwise multiplication
    cos_sim = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    
    # Clip numerical precision artifacts to [0.0, 1.0]
    return np.clip(cos_sim, 0.0, 1.0).astype(np.float32)


def compute_tfidf_features(
    joined_df: pd.DataFrame,
    vectorizers: Optional[Dict[str, TfidfVectorizer]] = None
) -> pd.DataFrame:
    """
    Computes sparse TF-IDF cosine similarities for candidate pairs:
      - name_tfidf_word_cosine: Word (1-2 gram) TF-IDF cosine similarity on name_norm
      - name_tfidf_char_cosine: Char (2-4 gram) TF-IDF cosine similarity on name_norm
      - address_tfidf_word_cosine: Word (1-2 gram) TF-IDF cosine similarity on address_norm

    If vectorizers is None, fits new vectorizers on the joined dataframe texts.
    """
    out = pd.DataFrame(index=joined_df.index)

    s1_names = joined_df['s1_name_norm'].fillna('').astype(str).tolist()
    cand_names = joined_df['cand_name_norm'].fillna('').astype(str).tolist()
    s1_addrs = joined_df['s1_address_norm'].fillna('').astype(str).tolist()
    cand_addrs = joined_df['cand_address_norm'].fillna('').astype(str).tolist()

    if vectorizers is None:
        # Fit on available texts
        all_names = s1_names + cand_names
        all_addrs = s1_addrs + cand_addrs
        vectorizers = fit_tfidf_vectorizers(all_names, all_addrs)

    # 1. Name Word TF-IDF Cosine
    out['name_tfidf_word_cosine'] = _sparse_pairwise_cosine_similarity(
        vectorizers['name_word'], s1_names, cand_names
    )

    # 2. Name Char TF-IDF Cosine
    out['name_tfidf_char_cosine'] = _sparse_pairwise_cosine_similarity(
        vectorizers['name_char'], s1_names, cand_names
    )

    # 3. Address Word TF-IDF Cosine
    out['address_tfidf_word_cosine'] = _sparse_pairwise_cosine_similarity(
        vectorizers['addr_word'], s1_addrs, cand_addrs
    )

    return out
