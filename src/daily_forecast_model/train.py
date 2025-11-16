import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# Import configuration and helper functions
from daily_forecast_model.helper import (
    TARGET_COLUMNS, RANDOM_STATE, MODEL_CONFIGS, 
    SAVE_MODELS, LOAD_SAVED_MODELS, MODEL_FORMAT, MODELS_DIR, CONVERT_TO_ONNX,
    LOG_TRAINING_TIME, VERBOSE_TRAINING, get_enabled_models, load_data
)

# Optional ONNX support
try:
    import skl2onnx
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  Warning: skl2onnx not installed. ONNX export will be skipped.")

def train_models(X_train, y_train):
    """
    Initialize and train models based on configuration.
    
    Args:
        X_train: Feature training DataFrame
        y_train: Target training DataFrame
    
    Returns:
        dict: Dictionary of trained models with metadata
    """
    trained_models = {}
    training_times = {}
    
    enabled_models = get_enabled_models()
    
    if not enabled_models:
        print("✗ Error: No models enabled in configuration!")
        return trained_models
    
    print(f"\n{'='*70}")
    print(f"Training {len(enabled_models)} models...")
    print(f"{'='*70}")
    
    for model_name, model_config in enabled_models.items():
        print(f"\n[{model_name}]")
        print(f"  Description: {model_config['description']}")
        print(f"  Status: Training...", end="", flush=True)
        
        # Create model with configuration parameters
        start_time = time.time()
        
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest":
            model = RandomForestRegressor(**model_config['params'])
        elif model_name == "XGBoost":
            # XGBoost requires MultiOutputRegressor for multi-step forecasting
            xgb_model = XGBRegressor(**model_config['params'])
            model = MultiOutputRegressor(xgb_model)
        elif model_name == "LightGBM":
            from lightgbm import LGBMRegressor
            lgbm_model = LGBMRegressor(**model_config['params'])
            model = MultiOutputRegressor(lgbm_model)
        elif model_name == "CatBoost":
            from catboost import CatBoostRegressor
            catboost_model = CatBoostRegressor(**model_config['params'])
            model = MultiOutputRegressor(catboost_model)
        else:
            print(f"  ✗ Unknown model type: {model_name}")
            continue
        
        # Train model
        try:
            model.fit(X_train, y_train)
            elapsed_time = time.time() - start_time
            
            training_times[model_name] = elapsed_time
            trained_models[model_name] = model
            
            if LOG_TRAINING_TIME:
                print(f" ✓ Complete ({elapsed_time:.2f}s)")
            else:
                print(f" ✓ Complete")
        
        except Exception as e:
            print(f" ✗ Failed")
            print(f"  Error: {e}")
            continue
    
    # Print training time summary
    if LOG_TRAINING_TIME and training_times:
        print(f"\n{'-'*70}")
        print("Training Time Summary:")
        for model_name, elapsed_time in sorted(training_times.items(), key=lambda x: x[1]):
            print(f"  {model_name}: {elapsed_time:.2f}s")
        print(f"{'-'*70}")
    
    # Save models if configured
    if SAVE_MODELS and trained_models:
        save_models(trained_models)
    
    return trained_models

def save_models(trained_models):
    """
    Save trained models to disk in ONNX, joblib, or pickle format.
    
    Args:
        trained_models (dict): Dictionary of trained model objects
    """
    print(f"\nSaving trained models to '{MODELS_DIR}/'...")
    
    for model_name, model in trained_models.items():
        try:
            model_filename = f"{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}"
            
            if MODEL_FORMAT == 'onnx' and CONVERT_TO_ONNX:
                if not ONNX_AVAILABLE:
                    print(f"  ⚠️  {model_name}: ONNX not available, falling back to joblib")
                    save_model_joblib(model, model_filename)
                else:
                    save_model_onnx(model, model_name, model_filename)
            elif MODEL_FORMAT == 'joblib':
                save_model_joblib(model, model_filename)
            else:  # pickle
                save_model_pickle(model, model_filename)
            
            print(f"  ✓ {model_name} saved successfully")
        except Exception as e:
            print(f"  ✗ Failed to save {model_name}: {e}")


def save_model_onnx(model, model_name, filename):
    """
    Convert and save model to ONNX format.
    Supports: Linear Regression, Random Forest, XGBoost
    Unsupported: CatBoost, LightGBM (fall back to joblib)
    
    Args:
        model: Trained sklearn/XGBoost model (possibly wrapped in MultiOutputRegressor)
        model_name: Name of the model
        filename: Base filename without extension
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
        
        # Convert to ONNX using correct function
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
    """Save model using joblib format."""
    filepath = os.path.join(MODELS_DIR, f"{filename}.joblib")
    joblib.dump(model, filepath)


def save_model_pickle(model, filename):
    """Save model using pickle format."""
    import pickle
    filepath = os.path.join(MODELS_DIR, f"{filename}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_saved_models():
    """
    Load previously trained models from disk (ONNX, joblib, or pickle).
    
    Returns:
        dict: Dictionary of loaded models, or empty dict if none found
    """
    loaded_models = {}
    
    if not LOAD_SAVED_MODELS:
        return loaded_models
    
    print(f"\nLoading saved models from '{MODELS_DIR}/'...")
    
    try:
        for model_name in get_enabled_models().keys():
            model_filename = f"{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}"
            
            # Try loading in order: ONNX -> backup joblib -> other formats
            onnx_path = os.path.join(MODELS_DIR, f"{model_filename}.onnx")
            backup_path = os.path.join(MODELS_DIR, f"{model_filename}_backup.joblib")
            joblib_path = os.path.join(MODELS_DIR, f"{model_filename}.joblib")
            pickle_path = os.path.join(MODELS_DIR, f"{model_filename}.pkl")
            
            model = None
            
            # Try ONNX first (with backup joblib)
            if os.path.exists(onnx_path) and os.path.exists(backup_path):
                try:
                    model = joblib.load(backup_path)
                    print(f"  ✓ Loaded {model_name} (ONNX + backup joblib)")
                except Exception as e:
                    print(f"  ⚠️  Failed to load {model_name} backup: {e}")
            
            # Try joblib
            elif os.path.exists(joblib_path):
                try:
                    model = joblib.load(joblib_path)
                    print(f"  ✓ Loaded {model_name} (joblib)")
                except Exception as e:
                    print(f"  ⚠️  Failed to load {model_name}: {e}")
            
            # Try pickle
            elif os.path.exists(pickle_path):
                try:
                    import pickle
                    with open(pickle_path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"  ✓ Loaded {model_name} (pickle)")
                except Exception as e:
                    print(f"  ⚠️  Failed to load {model_name}: {e}")
            
            if model is not None:
                loaded_models[model_name] = model
    
    except Exception as e:
        print(f"✗ Error loading models: {e}")
    
    return loaded_models

if __name__ == "__main__":
    # Example usage
    X_train, y_train = load_data('train')
    X_dev, y_dev = load_data('dev')

    final_X_train = pd.concat([X_train, X_dev])
    final_y_train = pd.concat([y_train, y_dev])
    
    trained_models = train_models(final_X_train, final_y_train)