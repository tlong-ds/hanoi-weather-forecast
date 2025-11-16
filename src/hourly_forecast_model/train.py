import os
import sys
import time
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hourly_forecast_model.helper import (
    PROJECT_ROOT, N_STEPS_AHEAD, MODELS_DIR, load_data
)

# Training configuration
LOG_TRAINING_TIME = True


def load_best_config():
    """Load best model configuration from tuning results."""
    config_file = os.path.join(PROJECT_ROOT, 'src', 'hourly_forecast_model', 
                               'final', 'best_model_config.json')
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded best model config from: {config_file}")
        print(f"  Model: {config['model']}")
        print(f"  Best RMSE: {config['best_rmse']:.4f}°C")
        return config
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_file}")
        print("  Run tune.py first to generate model configuration.")
        return None


def create_model(model_name, params):
    """
    Create a model instance with the given hyperparameters from tuning.
    
    Args:
        model_name (str): Name of the model ('RandomForest', 'XGBoost', 'LightGBM', 'CatBoost')
        params (dict): Model hyperparameters from tuning results
    
    Returns:
        model: Instantiated model
    """
    if model_name == 'RandomForest':
        return RandomForestRegressor(**params)
    
    elif model_name == 'XGBoost':
        return xgb.XGBRegressor(**params)
    
    elif model_name == 'LightGBM':
        return lgb.LGBMRegressor(**params)
    
    elif model_name == 'CatBoost':
        return CatBoostRegressor(**params)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_multioutput_model():
    """
    Train multi-output model for all 24 hours.
    Uses combined train+dev data for final model training.
    
    Returns:
        model: Trained multi-output model or None on error
    """
    print(f"\n{'='*70}")
    print(f"TRAINING MULTI-OUTPUT MODEL")
    print(f"{'='*70}")
    
    # Load best configuration
    config = load_best_config()
    if config is None:
        return None
    
    model_name = config['model']
    params = config['params']
    
    # Load data
    print("\nLoading data...")
    X_train, y_train, X_dev, y_dev, _, _ = load_data()
    
    if X_train is None:
        print("✗ Failed to load data")
        return None
    
    # Combine train and dev sets for final training
    X_combined = pd.concat([X_train, X_dev], axis=0)
    y_combined = pd.concat([y_train, y_dev], axis=0)
    
    print(f"✓ Data loaded:")
    print(f"  Combined train+dev: {X_combined.shape[0]} samples")
    print(f"  Features: {X_combined.shape[1]}")
    print(f"  Targets: {y_combined.shape[1]} hours (t+1h to t+{N_STEPS_AHEAD}h)")
    
    # Create base model
    print(f"\nCreating {model_name} model...")
    base_model = create_model(model_name, params)
    
    # Wrap in MultiOutputRegressor
    print(f"Wrapping in MultiOutputRegressor...")
    model = MultiOutputRegressor(base_model, n_jobs=-1)
    
    # Train model
    print(f"\nTraining multi-output model...")
    print(f"  This will train {N_STEPS_AHEAD} estimators in parallel...")
    
    start_time = time.time()
    model.fit(X_combined, y_combined)
    elapsed_time = time.time() - start_time
    
    if LOG_TRAINING_TIME:
        print(f"✓ Training complete in {elapsed_time:.2f}s")
    else:
        print(f"✓ Training complete")
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'model_multioutput_24h.joblib')
    
    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save training metadata
    metadata = {
        'model_type': model_name,
        'n_targets': N_STEPS_AHEAD,
        'n_features': X_combined.shape[1],
        'n_samples': X_combined.shape[0],
        'training_time': elapsed_time,
        'tuned_rmse': config['best_rmse'],
        'params': params
    }
    
    metadata_path = os.path.join(MODELS_DIR, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Training metadata saved to: {metadata_path}")
    
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Targets: {N_STEPS_AHEAD} hours")
    print(f"Training samples: {X_combined.shape[0]}")
    print(f"Features: {X_combined.shape[1]}")
    if LOG_TRAINING_TIME:
        print(f"Training time: {elapsed_time:.2f}s")
    print(f"Tuned RMSE: {config['best_rmse']:.4f}°C")
    print(f"{'='*70}\n")
    
    return model


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HOURLY FORECAST MULTI-OUTPUT MODEL TRAINING")
    print("="*70 + "\n")
    
    # Train multi-output model using combined train+dev data
    model = train_multioutput_model()
    
    if model:
        print("✅ Training complete! Model saved to:", MODELS_DIR)
    else:
        print("❌ Training failed.")
