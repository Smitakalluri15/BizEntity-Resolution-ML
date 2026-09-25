"""
Model training module for Gradient Boosted Matcher in Business Entity Resolution.
"""

from typing import List, Dict, Optional, Any, Tuple
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb


# Canonical list of numeric feature columns used for model training
DEFAULT_FEATURE_COLUMNS = [
    # Name features
    'name_exact_raw',
    'name_exact_norm',
    'name_levenshtein_ratio',
    'name_jaccard_tokens',
    'name_token_overlap_count',
    'name_char_trigram_jaccard',
    'name_phonetic_similarity',
    'name_script_match',
    # Address & postal features
    'address_exact_norm',
    'address_levenshtein_ratio',
    'address_jaccard_tokens',
    'address_char_trigram_jaccard',
    'postal_available',
    'postal_exact_match',
    'landmark_flag_either',
    # TF-IDF cosine features
    'name_tfidf_word_cosine',
    'name_tfidf_char_cosine',
    'address_tfidf_word_cosine',
    # Cross, interaction, and group features
    'country_match',
    'name_address_interaction',
    'candidate_source_id',
    'candidate_rank_in_group',
    'candidate_set_size',
    'top1_gap',
    'is_exact_tier',
]


def train_matcher(
    training_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    target_col: str = "is_true_match",
    model_params: Optional[Dict[str, Any]] = None,
    val_df: Optional[pd.DataFrame] = None,
    early_stopping_rounds: int = 30,
    model_output_path: Optional[str] = None
) -> lgb.LGBMClassifier:
    """
    Trains a LightGBM Binary Classifier on candidate pairwise features.
    
    Args:
        training_df: DataFrame containing features and target_col.
        feature_columns: List of feature names to use as input.
        target_col: Column name containing binary 0/1 match labels.
        model_params: Dict of LightGBM hyperparameters.
        val_df: Optional validation DataFrame for early stopping.
        early_stopping_rounds: Early stopping patience rounds.
        model_output_path: Optional file path to save the trained model artifact.
        
    Returns:
        lgb.LGBMClassifier: Trained model instance.
    """
    if feature_columns is None:
        feature_columns = [col for col in DEFAULT_FEATURE_COLUMNS if col in training_df.columns]

    # Validate feature columns exist
    missing_cols = [c for c in feature_columns if c not in training_df.columns]
    if missing_cols:
        raise ValueError(f"Feature columns missing from training_df: {missing_cols}")

    X_train = training_df[feature_columns]
    y_train = training_df[target_col].values.astype(int)

    # Default LightGBM hyperparameters optimized for fast convergence and high precision
    default_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 300,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    if model_params:
        default_params.update(model_params)

    model = lgb.LGBMClassifier(**default_params)

    t0 = time.time()
    if val_df is not None and len(val_df) > 0 and target_col in val_df.columns:
        X_val = val_df[feature_columns].values
        y_val = val_df[target_col].values.astype(int)
        callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )
    else:
        model.fit(X_train, y_train)

    train_time = time.time() - t0
    print(f"[train_matcher] LightGBM model trained in {train_time:.2f}s on {len(X_train):,} samples with {len(feature_columns)} features.")

    # Save model artifact if output path specified
    if model_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(model_output_path)), exist_ok=True)
        model.booster_.save_model(model_output_path)
        print(f"[train_matcher] Saved trained model to {model_output_path}")

    return model


def get_feature_importance(model: lgb.LGBMClassifier, feature_columns: List[str]) -> pd.DataFrame:
    """
    Returns feature importance ranking as a DataFrame sorted descending by importance.
    """
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    df_imp['importance_pct'] = (df_imp['importance'] / df_imp['importance'].sum()) * 100.0
    return df_imp
