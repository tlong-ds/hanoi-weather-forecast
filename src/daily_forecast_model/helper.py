import os
import pandas as pd
import numpy as np
import torch
import joblib


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Paths relative to project root
DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'trained_models')

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

N_STEPS_AHEAD = 5  # Must match preprocessing.py N_STEPS_AHEAD
TARGET_COLUMN = 'temp'  # Must match preprocessing.py TARGET_COLUMN

# Dynamically generate target columns based on N_STEPS_AHEAD
TARGET_COLUMNS = [f'target_{TARGET_COLUMN}_t+{i}' for i in range(1, N_STEPS_AHEAD + 1)]

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Random state for reproducibility
RANDOM_STATE = 42

# Model configurations
import json

# Load tuning results from results/ directory (new structure from two-stage tuning)
def load_model_config_from_results():
    """
    Load model configuration from tuning results.
    
    Priority:
    1. results/best_params_per_target.json (per-target optimized params)
    2. results/architecture_selection.json (best architecture)
    3. best_params.json (legacy single-file format)
    
    Returns:
        tuple: (model_name, params_dict, per_target_params or None)
    
    Raises:
        FileNotFoundError: If no tuning results files are found
    """
    # Try new two-stage tuning results first
    try:
        results_path = os.path.join(PROJECT_ROOT, 'src', 'daily_forecast_model', 'final', 'best_params_per_target.json')
        with open(results_path, 'r') as f:
            best_params_per_target = json.load(f)
            
            # Get model name from first target (all targets use same model architecture)
            first_target = list(best_params_per_target.values())[0]
            model_name = first_target['model']
            
            # Return per-target params directly (each target has optimized params)
            # Note: This is for backward compatibility with MODEL_CONFIGS
            # Actual training uses PER_TARGET_PARAMS directly
            representative_target = 't+3' if 't+3' in best_params_per_target else list(best_params_per_target.keys())[0]
            params = best_params_per_target[representative_target]['params']
            
            print(f"✓ Loaded per-target optimized configurations from src/daily_forecast_model/final/best_params_per_target.json")
            print(f"  Model architecture: {model_name}")
            print(f"  Per-target params available for: {', '.join(best_params_per_target.keys())}")
            
            return model_name, params, best_params_per_target
            
    except FileNotFoundError:
        pass
    
    # Try stage 1 results (architecture only, no deep tuning)
    try:
        results_path = os.path.join(PROJECT_ROOT, 'src', 'daily_forecast_model', 'final', 'architecture_selection.json')
        with open(results_path, 'r') as f:
            architecture_selection = json.load(f)
            model_name = architecture_selection['best_architecture']
            params = {k: v for k, v in architecture_selection['best_params'].items() if k != 'model_name'}
            
            print(f"✓ Loaded {model_name} architecture from src/daily_forecast_model/final/architecture_selection.json")
            print(f"  Note: Using stage 1 params (not deeply optimized). Run stage 2 for better results.")
            
            return model_name, params, None
            
    except FileNotFoundError:
        pass
    
    # Try legacy best_params.json
    try:
        results_path = os.path.join(PROJECT_ROOT, 'best_params.json')
        with open(results_path, 'r') as f:
            best_params = json.load(f)
            model_name = best_params.pop('model_name')
            params = {k: v for k, v in best_params.items() if k != 'model_name'}
            
            print(f"✓ Loaded {model_name} configuration from best_params.json (legacy format)")
            
            return model_name, params, None
            
    except FileNotFoundError:
        pass
    
    # No tuning results found - raise error
    raise FileNotFoundError(
        "No tuning results found. Please run tuning first:\n"
        "  1. Execute run_tuning.ipynb in Google Colab, or\n"
        "  2. Run: python src/daily_forecast_model/tune.py\n"
        "This will generate src/daily_forecast_model/final/best_params_per_target.json and src/daily_forecast_model/final/architecture_selection.json"
    )

# Load configuration
model_name, model_params, per_target_params = load_model_config_from_results()

MODEL_CONFIGS = {
    model_name: {
        "params": {**model_params},
        "enabled": True,
        "description": f"{model_name} with multi-output wrapper (optimized via two-stage tuning)"
    }
}

# Store per-target params for future use (if available)
PER_TARGET_PARAMS = per_target_params

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
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')  # Directory to save plots
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


def get_per_target_params(target_name):
    """
    Get optimized parameters for a specific target horizon.
    
    Args:
        target_name (str): Target name like 't+1', 't+2', etc.
    
    Returns:
        dict: Parameters for that target, or None if not available
    """
    if PER_TARGET_PARAMS is not None and target_name in PER_TARGET_PARAMS:
        return PER_TARGET_PARAMS[target_name]['params']
    return None


def get_tuning_summary():
    """
    Get summary of tuning results.
    
    Returns:
        dict: Summary with model name, RMSE per target, feature types, etc.
    """
    if PER_TARGET_PARAMS is None:
        return None
    
    summary = {
        'model': model_name,
        'n_targets': len(PER_TARGET_PARAMS),
        'targets': {}
    }
    
    for target_name, target_info in PER_TARGET_PARAMS.items():
        summary['targets'][target_name] = {
            'rmse': target_info.get('best_rmse', None),
            'feature_type': target_info.get('feature_type', None),
            'n_features': target_info.get('n_features', None)
        }
    
    return summary


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


# ============================================================================
# MODEL SAVING FUNCTIONS
# ============================================================================

def save_models(trained_models, model_format=None, convert_to_onnx=None):
    """
    Save trained models to disk in specified format.
    
    Centralized model saving function used by train.py and other modules.
    Supports multiple formats: ONNX (with backup), joblib, pickle.
    
    Args:
        trained_models (dict): Dictionary of trained model objects {model_name: model}
        model_format (str, optional): 'onnx', 'joblib', or 'pickle'. Uses global config if None
        convert_to_onnx (bool, optional): Whether to convert to ONNX. Uses global config if None
    """
    if model_format is None:
        model_format = MODEL_FORMAT
    if convert_to_onnx is None:
        convert_to_onnx = CONVERT_TO_ONNX
    
    # Check if ONNX is available
    onnx_available = False
    if model_format == 'onnx' and convert_to_onnx:
        try:
            import skl2onnx
            import onnx
            onnx_available = True
        except ImportError:
            print("⚠️  Warning: skl2onnx not installed. ONNX export will be skipped.")
            onnx_available = False
    
    print(f"\nSaving trained models to '{MODELS_DIR}/'...")
    
    for model_name, model in trained_models.items():
        try:
            model_filename = f"{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}"
            
            if model_format == 'onnx' and convert_to_onnx:
                if not onnx_available:
                    print(f"  ⚠️  {model_name}: ONNX not available, falling back to joblib")
                    save_model_joblib(model, model_filename)
                else:
                    save_model_onnx(model, model_name, model_filename)
            elif model_format == 'joblib':
                save_model_joblib(model, model_filename)
            else:  # pickle
                save_model_pickle(model, model_filename)
            
            print(f"  ✓ {model_name} saved successfully")
        except Exception as e:
            print(f"  ✗ Failed to save {model_name}: {e}")


def save_model_onnx(model, model_name, filename):
    """
    Convert and save model to ONNX format with joblib backup.
    
    Supports: Linear Regression, Random Forest, XGBoost
    Unsupported: CatBoost, LightGBM (falls back to joblib)
    
    Args:
        model: Trained sklearn/XGBoost model (possibly wrapped in MultiOutputRegressor)
        model_name (str): Name of the model for display
        filename (str): Base filename without extension
    """
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx import convert_sklearn
    import onnx
    
    try:
        # Get the actual model from MultiOutputRegressor wrapper if needed
        actual_model = model.estimator if hasattr(model, 'estimator') else model
        
        # Check if model type is supported by skl2onnx
        actual_model_type = type(actual_model).__name__
        unsupported_types = ['CatBoostRegressor', 'LGBMRegressor', 'LGBMRFRegressor']
        
        if actual_model_type in unsupported_types:
            print(f"    ℹ️  {actual_model_type} not natively supported by ONNX, using joblib instead")
            save_model_joblib(model, filename)
            return
        
        # Load sample data to determine number of features
        X_sample, _ = load_data('train')
        n_features = X_sample.shape[1]
        
        # Define ONNX input type
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(
            actual_model,
            initial_types=initial_type,
            target_opset=12
        )
        
        # Save ONNX model
        filepath = os.path.join(MODELS_DIR, f"{filename}.onnx")
        with open(filepath, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        # Also save the original model as backup for loading
        backup_filepath = os.path.join(MODELS_DIR, f"{filename}_backup.joblib")
        joblib.dump(model, backup_filepath)
        
        print(f"    📦 Saved as ONNX: {filepath}")
        print(f"    💾 Backup joblib: {backup_filepath}")
        
    except Exception as e:
        print(f"    ⚠️  ONNX conversion failed: {e}")
        print(f"    Falling back to joblib...")
        save_model_joblib(model, filename)


def save_model_joblib(model, filename):
    """
    Save model using joblib format.
    
    Args:
        model: Trained model object
        filename (str): Base filename without extension
    """
    filepath = os.path.join(MODELS_DIR, f"{filename}.joblib")
    joblib.dump(model, filepath)


def save_model_pickle(model, filename):
    """
    Save model using pickle format.
    
    Args:
        model: Trained model object
        filename (str): Base filename without extension
    """
    import pickle
    filepath = os.path.join(MODELS_DIR, f"{filename}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_trained_models(verbose=True):
    """
    Load previously trained models from disk.
    
    Centralized model loading function used by evaluate.py, inference.py, and other modules.
    Supports multiple formats with automatic fallback priority:
    1. ONNX backup (joblib) - for ONNX-converted models
    2. Regular joblib
    3. Pickle
    4. ONNX (requires ONNX Runtime, not recommended)
    
    Args:
        verbose (bool): Whether to print loading status messages
    
    Returns:
        dict: Dictionary of loaded models {model_name: model_object}
              Empty dict if no models found or loading fails
    """
    loaded_models = {}
    
    if not os.path.exists(MODELS_DIR):
        if verbose:
            print(f"✗ Models directory not found: {MODELS_DIR}")
        return loaded_models
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Loading trained models from '{MODELS_DIR}/'...")
        print(f"{'='*70}\n")
    
    try:
        # Try to load models for each enabled model type
        for model_name in get_enabled_models().keys():
            model_filename = f"{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}"
            
            # Try loading in order of preference: backup joblib -> joblib -> pickle -> ONNX
            backup_path = os.path.join(MODELS_DIR, f"{model_filename}_backup.joblib")
            joblib_path = os.path.join(MODELS_DIR, f"{model_filename}.joblib")
            pickle_path = os.path.join(MODELS_DIR, f"{model_filename}.pkl")
            onnx_path = os.path.join(MODELS_DIR, f"{model_filename}.onnx")
            
            model = None
            loaded_from = None
            
            # Priority 1: Backup joblib (for ONNX-converted models)
            if os.path.exists(backup_path):
                try:
                    model = joblib.load(backup_path)
                    loaded_from = "backup joblib (ONNX model)"
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Failed to load {model_name} from backup: {e}")
            
            # Priority 2: Regular joblib
            if model is None and os.path.exists(joblib_path):
                try:
                    model = joblib.load(joblib_path)
                    loaded_from = "joblib"
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Failed to load {model_name} from joblib: {e}")
            
            # Priority 3: Pickle
            if model is None and os.path.exists(pickle_path):
                try:
                    import pickle
                    with open(pickle_path, 'rb') as f:
                        model = pickle.load(f)
                    loaded_from = "pickle"
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Failed to load {model_name} from pickle: {e}")
            
            # Priority 4: ONNX (warning: requires ONNX runtime, returns reference only)
            if model is None and os.path.exists(onnx_path):
                if verbose:
                    print(f"  ℹ️  {model_name}: ONNX file exists but requires ONNX Runtime for inference")
                    print(f"      Use backup joblib instead (recommended)")
                continue
            
            # Report result
            if model is not None:
                loaded_models[model_name] = model
                if verbose:
                    print(f"  ✓ {model_name:<20} loaded from {loaded_from}")
            else:
                if verbose:
                    print(f"  ✗ {model_name:<20} not found (no saved model file)")
    
    except Exception as e:
        if verbose:
            print(f"\n✗ Error loading models: {e}")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Loaded {len(loaded_models)} model(s) successfully")
        print(f"{'='*70}\n")
        
        if not loaded_models:
            print("⚠️  No trained models found. Please train models first using train.py")
    
    return loaded_models


def load_single_model(model_path):
    """
    Load a single model from a specific path.
    
    Utility function for loading models by direct path (used in inference.py).
    
    Args:
        model_path (str): Full path to model file (.joblib, .pkl, or .onnx)
    
    Returns:
        model: Loaded model object
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If unsupported model format
        RuntimeError: If loading fails
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        if model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        elif model_path.endswith('.pkl'):
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported model format: {model_path}. Use .joblib or .pkl")
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def get_best_model_path(models_dir=None):
    """
    Auto-detect the best available model in the models directory.
    
    Priority:
    1. ONNX backup files (most recent)
    2. Regular joblib files (most recent)
    3. Pickle files (most recent)
    
    Args:
        models_dir (str, optional): Path to models directory. Uses global MODELS_DIR if None
    
    Returns:
        str: Path to best available model file, or None if no models found
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    
    if not os.path.exists(models_dir):
        return None
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith(('.joblib', '.pkl'))]
    
    if not model_files:
        return None
    
    # Prioritize backup joblib files (ONNX models), then regular joblib
    backup_files = [f for f in model_files if 'backup' in f and f.endswith('.joblib')]
    if backup_files:
        return os.path.join(models_dir, backup_files[0])
    
    joblib_files = [f for f in model_files if f.endswith('.joblib')]
    if joblib_files:
        return os.path.join(models_dir, joblib_files[0])
    
    # Fall back to pickle
    return os.path.join(models_dir, model_files[0])

