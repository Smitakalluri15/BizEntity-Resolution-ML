"""
Model training, prediction, evaluation, and threshold tuning module for Business Entity Resolution.
"""

from src.model.prepare_training_data import prepare_training_data
from src.model.train import train_matcher, get_feature_importance, DEFAULT_FEATURE_COLUMNS
from src.model.predict import predict_match_probabilities, assemble_entity_predictions
from src.model.threshold_tuning import sweep_thresholds, find_best_threshold

__all__ = [
    "prepare_training_data",
    "train_matcher",
    "get_feature_importance",
    "DEFAULT_FEATURE_COLUMNS",
    "predict_match_probabilities",
    "assemble_entity_predictions",
    "sweep_thresholds",
    "find_best_threshold",
]
