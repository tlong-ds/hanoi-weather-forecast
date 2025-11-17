import os
import pandas as pd
import numpy as np
import torch
import joblib
import json


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# ============================================================================
# DATA CONFIGURATION - HOURLY MODEL
# ============================================================================

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Hourly-specific paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data_processing_hourly')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'src', 'hourly_forecast_model', 'final_model')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'src', 'hourly_forecast_model', 'evaluate_results', 'plots')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Data files for hourly model
X_TRAIN_FILE = os.path.join(DATA_DIR, 'X_train_transformed_hourly.csv')
y_TRAIN_FILE = os.path.join(DATA_DIR, 'y_train_hourly.csv')
X_DEV_FILE = os.path.join(DATA_DIR, 'X_dev_transformed_hourly.csv')
y_DEV_FILE = os.path.join(DATA_DIR, 'y_dev_hourly.csv')
X_TEST_FILE = os.path.join(DATA_DIR, 'X_test_transformed_hourly.csv')
y_TEST_FILE = os.path.join(DATA_DIR, 'y_test_hourly.csv')

# ============================================================================
# TARGET CONFIGURATION - HOURLY FORECASTING
# ============================================================================

N_STEPS_AHEAD = 24  # Predict 24 hours ahead (t+1h to t+24h)
TARGET_COLUMN = 'temp'  # Temperature forecasting

# Hourly target columns: target_temp_t+1h, target_temp_t+2h, ..., target_temp_t+24h
TARGET_COLUMNS = [f'target_{TARGET_COLUMN}_t+{i}h' for i in range(1, N_STEPS_AHEAD + 1)]

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Random state for reproducibility
RANDOM_STATE = 42

# Model configurations
def load_model_config_from_results():
    """
    Load model configuration from tuning results for HOURLY model.
    
    Priority:
    1. src/hourly_forecast_model/tuning_results/best_params_per_target.json (per-target optimized params)
    2. src/hourly_forecast_model/tuning_results/architecture_selection.json (best architecture)
    3. Default configuration
    
    Returns:
        tuple: (model_name, params_dict, per_target_params or None)
    """
    # Try new two-stage tuning results first
    try:
        results_path = os.path.join(PROJECT_ROOT, 'src', 'hourly_forecast_model', 'tuning_results', 'best_params_per_target.json')
        with open(results_path, 'r') as f:
            best_params_per_target = json.load(f)
            
            # Get model name from first target (all targets use same model architecture)
            first_target = list(best_params_per_target.values())[0]
            model_name = first_target['model']
            
            # Use representative target (middle of 24 hours)
            representative_target = 't+12h' if 't+12h' in best_params_per_target else list(best_params_per_target.keys())[0]
            params = best_params_per_target[representative_target]['params']
            
            print(f"✓ Loaded per-target optimized configurations from src/hourly_forecast_model/tuning_results/best_params_per_target.json")
            print(f"  Model architecture: {model_name}")
            print(f"  Per-target params available for: {', '.join(best_params_per_target.keys())}")
            
            return model_name, params, best_params_per_target
            
    except FileNotFoundError:
        pass
    
    # Try stage 1 results (architecture only)
    try:
        results_path = os.path.join(PROJECT_ROOT, 'src', 'hourly_forecast_model', 'tuning_results', 'architecture_selection.json')
        with open(results_path, 'r') as f:
            arch_results = json.load(f)
            model_name = arch_results['best_architecture']
            params = arch_results.get('best_params', {})
            
            print(f"✓ Loaded architecture selection from src/hourly_forecast_model/tuning_results/architecture_selection.json")
            print(f"  Best architecture: {model_name}")
            
            return model_name, params, None
            
    except FileNotFoundError:
        pass
    
    # Default configuration
    print("⚠ No tuning results found, using default RandomForest configuration")
    return "RandomForest", {
        'n_estimators': 200,
        'max_depth': 12,
        'min_samples_split': 4,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }, None


# Try to load configuration from tuning results
try:
    MODEL_NAME, DEFAULT_PARAMS, PER_TARGET_PARAMS = load_model_config_from_results()
except Exception as e:
    print(f"⚠ Error loading model config: {e}")
    print("  Using default RandomForest configuration")
    MODEL_NAME = "RandomForest"
    DEFAULT_PARAMS = {
        'n_estimators': 200,
        'max_depth': 12,
        'min_samples_split': 4,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    PER_TARGET_PARAMS = None


# Model registry
MODEL_CONFIGS = {
    'RandomForest': DEFAULT_PARAMS,
    'XGBoost': DEFAULT_PARAMS if MODEL_NAME == 'XGBoost' else {},
    'LightGBM': DEFAULT_PARAMS if MODEL_NAME == 'LightGBM' else {},
    'CatBoost': DEFAULT_PARAMS if MODEL_NAME == 'CatBoost' else {}
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

VERBOSE_TRAINING = True
LOG_TRAINING_TIME = True
USE_MULTIOUTPUT = False  # Train separate model per hour vs one multi-output model

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

VERBOSE_EVALUATION = True
SAVE_PLOTS = True
PLOT_FORMAT = 'png'
PLOT_DPI = 150

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_data():
    """
    Load all preprocessed data (train, dev, test) with all target columns.
    
    Returns:
        tuple: (X_train, y_train, X_dev, y_dev, X_test, y_test)
               All y DataFrames contain ALL 24 target columns
    """
    try:
        X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train_transformed.csv'), index_col=0)
        y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'), index_col=0)
        
        X_dev = pd.read_csv(os.path.join(DATA_DIR, 'X_dev_transformed.csv'), index_col=0)
        y_dev = pd.read_csv(os.path.join(DATA_DIR, 'y_dev.csv'), index_col=0)
        
        X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test_transformed.csv'), index_col=0)
        y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'), index_col=0)
        
        return X_train, y_train, X_dev, y_dev, X_test, y_test
    
    except FileNotFoundError as e:
        print(f"✗ Error: Data files not found in '{DATA_DIR}'")
        print(f"  Error: {e}")
        print(f"  Run process.py first to generate preprocessed data")
        return None, None, None, None, None, None


def load_data_for_hour(hour_step):
    """
    Load preprocessed data for specific hour.
    
    Args:
        hour_step (int): Hour ahead (1-24)
    
    Returns:
        tuple: (X_train, y_train, X_dev, y_dev, X_test, y_test)
               y values are Series for the specific target hour
    """
    target_col = f"target_temp_t+{hour_step}h"
    
    try:
        # Load all data
        X_train, y_train_all, X_dev, y_dev_all, X_test, y_test_all = load_data()
        
        if X_train is None:
            return None, None, None, None, None, None
        
        # Extract specific target column
        y_train = y_train_all[target_col]
        y_dev = y_dev_all[target_col]
        y_test = y_test_all[target_col]
        
        return X_train, y_train, X_dev, y_dev, X_test, y_test
    
    except KeyError as e:
        print(f"  ✗ ERROR: Target column '{target_col}' not found")
        print(f"    Error: {e}")
        return None, None, None, None, None, None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_name(model_key):
    """Get display name for model."""
    names = {
        'RandomForest': 'Random Forest',
        'XGBoost': 'XGBoost',
        'LightGBM': 'LightGBM',
        'CatBoost': 'CatBoost'
    }
    return names.get(model_key, model_key)


def get_data_paths():
    """Get dictionary of all data file paths."""
    return {
        'X_train': X_TRAIN_FILE,
        'y_train': y_TRAIN_FILE,
        'X_dev': X_DEV_FILE,
        'y_dev': y_DEV_FILE,
        'X_test': X_TEST_FILE,
        'y_test': y_TEST_FILE,
        'data_dir': DATA_DIR,
        'models_dir': MODELS_DIR,
        'plots_dir': PLOTS_DIR
    }


def print_config():
    """Print current configuration."""
    print("=" * 70)
    print("HOURLY FORECAST MODEL CONFIGURATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Plots Directory: {PLOTS_DIR}")
    print(f"\nTarget Configuration:")
    print(f"  Forecast Horizon: {N_STEPS_AHEAD} hours (t+1h to t+{N_STEPS_AHEAD}h)")
    print(f"  Target Variable: {TARGET_COLUMN}")
    print(f"\nModel Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Per-Target Params: {'Yes' if PER_TARGET_PARAMS else 'No'}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
    
    # Verify data files exist
    print("\nData Files Status:")
    paths = get_data_paths()
    for name, path in paths.items():
        if name.endswith('_dir'):
            exists = os.path.exists(path)
            print(f"  {name}: {'✓' if exists else '✗'} {path}")
        else:
            exists = os.path.isfile(path)
            print(f"  {name}: {'✓' if exists else '✗'} {path}")

