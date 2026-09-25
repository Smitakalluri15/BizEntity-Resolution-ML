"""
Feature engineering module for Business Entity Resolution.
"""

from src.features.name_features import compute_name_features
from src.features.address_features import compute_address_features
from src.features.tfidf_features import fit_tfidf_vectorizers, compute_tfidf_features
from src.features.cross_features import compute_cross_features
from src.features.build_features import (
    join_candidate_data,
    build_features,
    assemble_labels
)

__all__ = [
    "compute_name_features",
    "compute_address_features",
    "fit_tfidf_vectorizers",
    "compute_tfidf_features",
    "compute_cross_features",
    "join_candidate_data",
    "build_features",
    "assemble_labels",
]
