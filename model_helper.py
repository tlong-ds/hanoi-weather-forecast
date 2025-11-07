"""
Helper module for model training and evaluation.
Centralizes configuration, data loading, and utility functions.
"""

import os
import pandas as pd
import numpy as np
import torch


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = 'processed_data'
MODELS_DIR = 'trained_models'

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Data files
X_TRAIN_FILE = os.path.join(DATA_DIR, 'X_train_transformed.csv')
y_TRAIN_FILE = os.path.join(DATA_DIR, 'y_train.csv')
X_DEV_FILE = os.path.join(DATA_DIR, 'X_dev_transformed.csv')
y_DEV_FILE = os.path.join(DATA_DIR, 'y_dev.csv')
X_TEST_FILE = os.path.join(DATA_DIR, 'X_test_transformed.csv')
y_TEST_FILE = os.path.join(DATA_DIR, 'y_test.csv')

# ============================================================================
# TARGET CONFIGURATION (Synced with preprocessing.py)
# ============================================================================

N_STEPS_AHEAD = 10  # Must match preprocessing.py N_STEPS_AHEAD
TARGET_COLUMN = 'temp'  # Must match preprocessing.py TARGET_COLUMN

# Dynamically generate target columns based on N_STEPS_AHEAD
TARGET_COLUMNS = [f'target_{TARGET_COLUMN}_t+{i}' for i in range(1, N_STEPS_AHEAD + 1)]

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Random state for reproducibility
RANDOM_STATE = 42

# Model configurations
MODEL_CONFIGS = {
    # 'Random Forest': {
    #     'params': {
    #         'n_estimators': 200,
    #         'max_depth': 12,
    #         'min_samples_split': 4,
    #         'random_state': RANDOM_STATE,
    #         'n_jobs': -1,
    #         'verbose': 0
    #     },
    #     'enabled': True,
    #     'description': 'Ensemble tree-based model'
    # },
    # 'XGBoost': {
    #     'params': {
    #         'n_estimators': 300,
    #         'max_depth': 6,
    #         'learning_rate': 0.05,
    #         'subsample': 0.8,
    #         'colsample_bytree': 0.8,
    #         'random_state': RANDOM_STATE,
    #         'n_jobs': -1,
    #         'verbosity': 0
    #     },
    #     'enabled': True,
    #     'description': 'Gradient boosting model with multi-output wrapper'
    # },
    "CatBoost": {
        'params': {
            'iterations': 400, 
            'depth': 6, 
            'learning_rate': 0.016517575668639525, 
            'l2_leaf_reg': 1.8809861076799286, 
            'subsample': 0.711841663235127,
            "random_state": RANDOM_STATE,
            "verbose": 0,
            "task_type": "GPU" if str(DEVICE) == "cuda" else "CPU",
        },
        'enabled': True,
        'description': 'CatBoost Regressor with multi-output wrapper'
    }
        
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training options
SAVE_MODELS = True  # Save trained models to disk
LOAD_SAVED_MODELS = False  # Load pre-trained models if available
MODEL_FORMAT = 'onnx'  # 'onnx', 'joblib', or 'pickle'
CONVERT_TO_ONNX = True  # Convert sklearn/XGBoost models to ONNX format

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Dev set metrics (no R² to avoid overfitting bias)
DEV_METRICS = ['MAE', 'RMSE', 'MAPE']

# Test set metrics (includes R² for final evaluation)
TEST_METRICS = ['MAE', 'RMSE', 'MAPE', 'R2']

# Evaluation output
SAVE_RESULTS = True
RESULTS_FILENAME = 'model_evaluation_results.csv'
RESULTS_DIR = DATA_DIR

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Plot settings
PLOT_ALL_HORIZONS = True  # Show all forecast horizons instead of just first/last
PLOT_DPI = 100
PLOT_FIGSIZE_SCATTER = (12, 7)
PLOT_FIGSIZE_TIMESERIES = (16, 8)

# Horizons to plot (None = all, or specify list like [1, 5, 10])
HORIZONS_TO_PLOT = None  # None means plot all horizons

# Plot saving configuration
SAVE_PLOTS = True  # Save plots to disk
PLOTS_DIR = 'plots'  # Directory to save plots
PLOT_FORMAT = 'png'  # Format: 'png', 'jpg', 'pdf', 'svg'

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Training logging
LOG_TRAINING_TIME = True
VERBOSE_TRAINING = True

# Evaluation logging
VERBOSE_EVALUATION = True
PRINT_METRICS_PER_HORIZON = True

# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================

def load_data(data_type='dev'):
    """
    Load processed data for training, development, or testing.
    
    Unified function that handles loading X and y data for any split,
    with consistent error handling and logging.

    Args:
        data_type (str): 'train', 'dev', or 'test'

    Returns:
        tuple: (X_data, y_data) DataFrames, or (None, None) on error
    """
    if data_type not in ['train', 'dev', 'test']:
        print(f"✗ Error: data_type must be 'train', 'dev', or 'test', not '{data_type}'")
        return None, None
    
    # Get file paths based on data_type
    file_paths = {
        'train': (X_TRAIN_FILE, y_TRAIN_FILE),
        'dev': (X_DEV_FILE, y_DEV_FILE),
        'test': (X_TEST_FILE, y_TEST_FILE)
    }
    
    x_path, y_path = file_paths[data_type]

    try:
        X_data = pd.read_csv(x_path, index_col='datetime', parse_dates=True)
        y_data = pd.read_csv(y_path, index_col='datetime', parse_dates=True)[TARGET_COLUMNS]

        if VERBOSE_TRAINING or VERBOSE_EVALUATION:
            print(f"✓ Loaded {data_type} set successfully!")
            print(f"  X_{data_type} shape: {X_data.shape}")
            print(f"  y_{data_type} shape: {y_data.shape}")
        return X_data, y_data
    
    except FileNotFoundError as e:
        print(f"✗ Error: {data_type} set file not found.")
        print(f"  Expected: {x_path}, {y_path}")
        print(e)
        return None, None
    except KeyError as e:
        print(f"✗ Error: Target columns not found in y_{data_type}.csv")
        print(e)
        return None, None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_name(model_key):
    """Get display name for model configuration key."""
    return model_key


def get_enabled_models():
    """Get dictionary of only enabled models."""
    return {k: v for k, v in MODEL_CONFIGS.items() if v['enabled']}


def get_all_target_columns():
    """Get list of all target columns."""
    return TARGET_COLUMNS


def get_data_paths():
    """Get dictionary of all data file paths."""
    return {
        'X_train': X_TRAIN_FILE,
        'y_train': y_TRAIN_FILE,
        'X_dev': X_DEV_FILE,
        'y_dev': y_DEV_FILE,
        'X_test': X_TEST_FILE,
        'y_test': y_TEST_FILE,
    }
