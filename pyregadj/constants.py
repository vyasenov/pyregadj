"""
Constants and configuration for the PyRegAdj package.

This module contains all configuration constants used throughout the package,
including valid adjustment methods, ML model configurations, and default values.
"""

# Valid adjustment methods
VALID_ADJUSTMENTS = ['none', 'pooled', 'groupwise', 'ml']

# Valid ML methods
VALID_ML_METHODS = ['rf', 'gbm', 'lasso', 'ridge', 'elastic-net']

# Cross-fitting configuration
MIN_FOLDS = 2
MAX_FOLDS = 10
DEFAULT_FOLDS = 5

# Default values
DEFAULT_CROSS_FIT = False
DEFAULT_CENTERED = False
DEFAULT_ADJUSTMENT = 'none'

# Random seed for reproducibility
RANDOM_SEED = 1988

# ML model configurations
ML_MODEL_CONFIGS = {
    'rf': {
        'n_estimators': 100,
        'random_state': RANDOM_SEED
    },
    'gbm': {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 2,
        'random_state': RANDOM_SEED
    },
    'lasso': {
        'alpha': 0.1,
        'random_state': RANDOM_SEED
    },
    'ridge': {
        'alpha': 1.0,
        'random_state': RANDOM_SEED
    },
    'elastic-net': {
        'alpha': 0.1,
        'l1_ratio': 0.5,
        'random_state': RANDOM_SEED
    }
}

# Confidence interval level
CONFIDENCE_LEVEL = 0.95 