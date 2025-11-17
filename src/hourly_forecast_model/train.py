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


class PerHorizonWrapper:
    """Wrapper class that mimics MultiOutputRegressor interface but stores per-horizon models."""
    
    def __init__(self, models_dict):
        """
        Args:
            models_dict: Dictionary mapping horizon names (e.g., 't+1h') to trained models
        """
        self.models_dict = models_dict
        self.horizons = sorted(models_dict.keys(), key=lambda x: int(x.replace('t+', '').replace('h', '')))
    
    def predict(self, X):
        """Predict using all per-horizon models."""
        predictions = []
        for horizon in self.horizons:
            model = self.models_dict[horizon]
            pred = model.predict(X)
            predictions.append(pred.reshape(-1, 1))
        
        # Stack predictions horizontally to match MultiOutputRegressor output shape
        return pd.np.hstack(predictions) if hasattr(pd, 'np') else __import__('numpy').hstack(predictions)
    
    def get_model_for_horizon(self, horizon):
        """Get the model for a specific horizon."""
        return self.models_dict.get(horizon)


def load_best_config():
    """Load best model configuration from tuning results."""
    # Try to load per-horizon parameters first (from two-stage tuning)
    per_horizon_file = os.path.join(PROJECT_ROOT, 'src', 'hourly_forecast_model', 
                                    'tuning_results', 'best_params_per_horizon.json')
    
    if os.path.exists(per_horizon_file):
        try:
            with open(per_horizon_file, 'r') as f:
                per_horizon_config = json.load(f)
            print(f"✓ Loaded per-horizon model configs from: {per_horizon_file}")
            print(f"  Found configurations for {len(per_horizon_config)} horizons")
            return {'type': 'per_horizon', 'configs': per_horizon_config}
        except Exception as e:
            print(f"⚠ Error loading per-horizon config: {e}")
    
    # FUTURE: Uncomment to use single best architecture for all horizons
    # This uses the architecture from Stage 1 of two-stage tuning
    # architecture_file = os.path.join(PROJECT_ROOT, 'src', 'hourly_forecast_model',
    #                                  'tuning_results', 'architecture_selection.json')
    # if os.path.exists(architecture_file):
    #     try:
    #         with open(architecture_file, 'r') as f:
    #             arch_config = json.load(f)
    #         best_architecture = arch_config['best_architecture']
    #         best_params = arch_config['best_params']
    #         print(f"✓ Using best architecture: {best_architecture}")
    #         return {'type': 'single', 'config': {
    #             'model': best_architecture,
    #             'params': best_params,
    #             'best_rmse': arch_config['best_rmse']
    #         }}
    #     except Exception as e:
    #         print(f"⚠ Error loading architecture config: {e}")
    
    # Fallback to single best model config (from multi-output tuning)
    config_file = os.path.join(PROJECT_ROOT, 'src', 'hourly_forecast_model', 
                               'tuning_results', 'best_model_config.json')
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded best model config from: {config_file}")
        print(f"  Model: {config['model']}")
        print(f"  Best RMSE: {config['best_rmse']:.4f}°C")
        return {'type': 'single', 'config': config}
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_file}")
        print("  Run tune.py or tune_per_horizon.py first to generate model configuration.")
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
    # Remove 'model_name' from params if present (it's metadata, not a parameter)
    clean_params = {k: v for k, v in params.items() if k != 'model_name'}
    
    if model_name == 'RandomForest':
        return RandomForestRegressor(**clean_params)
    
    elif model_name == 'XGBoost':
        return xgb.XGBRegressor(**clean_params)
    
    elif model_name == 'LightGBM':
        return lgb.LGBMRegressor(**clean_params)
    
    elif model_name == 'CatBoost':
        return CatBoostRegressor(**clean_params)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_multioutput_model():
    """
    Train multi-output model for all 24 hours.
    Uses combined train+dev data for final model training.
    
    Supports two modes:
    1. Per-horizon models: Different model/params for each hour (from two-stage tuning)
    2. Single model: Same architecture for all hours (from multi-output tuning)
    
    Returns:
        model: Trained multi-output model or None on error
    """
    print(f"\n{'='*70}")
    print(f"TRAINING MULTI-OUTPUT MODEL")
    print(f"{'='*70}")
    
    # Load best configuration
    config_data = load_best_config()
    if config_data is None:
        return None
    
    config_type = config_data['type']
    
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
    
    # Train based on configuration type
    if config_type == 'per_horizon':
        print(f"\n🎯 Using PER-HORIZON tuning results")
        print(f"   Each hour gets its own optimized model/parameters")
        model = train_per_horizon_models(X_combined, y_combined, config_data['configs'])
        
    else:  # config_type == 'single'
        print(f"\n🎯 Using SINGLE ARCHITECTURE for all horizons")
        config = config_data['config']
        model_name = config['model']
        params = config['params']
        
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
    if config_type == 'per_horizon':
        avg_rmse = sum(h['best_rmse'] for h in config_data['configs'].values()) / len(config_data['configs'])
        metadata = {
            'training_type': 'per_horizon',
            'n_targets': N_STEPS_AHEAD,
            'n_features': X_combined.shape[1],
            'n_samples': X_combined.shape[0],
            'average_tuned_rmse': avg_rmse,
            'models_per_horizon': {h: cfg['model'] for h, cfg in config_data['configs'].items()}
        }
    else:
        metadata = {
            'training_type': 'single_architecture',
            'model_type': config['model'],
            'n_targets': N_STEPS_AHEAD,
            'n_features': X_combined.shape[1],
            'n_samples': X_combined.shape[0],
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
    if config_type == 'per_horizon':
        print(f"Training Type: Per-Horizon (each hour optimized separately)")
        print(f"Targets: {N_STEPS_AHEAD} hours")
        print(f"Average Tuned RMSE: {avg_rmse:.4f}°C")
    else:
        print(f"Training Type: Single Architecture")
        print(f"Model: {config['model']}")
        print(f"Targets: {N_STEPS_AHEAD} hours")
        print(f"Tuned RMSE: {config['best_rmse']:.4f}°C")
    print(f"Training samples: {X_combined.shape[0]}")
    print(f"Features: {X_combined.shape[1]}")
    print(f"{'='*70}\n")
    
    return model


def train_per_horizon_models(X_combined, y_combined, per_horizon_configs):
    """
    Train individual models for each horizon using per-horizon optimized parameters.
    
    Args:
        X_combined: Combined training features
        y_combined: Combined training targets (all 24 hours)
        per_horizon_configs: Dict with configs for each horizon
    
    Returns:
        PerHorizonWrapper object containing all trained models
    """
    print(f"\nTraining {N_STEPS_AHEAD} individual models...")
    
    models_dict = {}
    start_time = time.time()
    
    for h in range(1, N_STEPS_AHEAD + 1):
        horizon_str = f"t+{h}h"
        
        if horizon_str not in per_horizon_configs:
            print(f"  ⚠ No config for {horizon_str}, skipping")
            continue
        
        config = per_horizon_configs[horizon_str]
        model_name = config['model']
        params = config['params']
        
        # Get target for this hour (column index h-1)
        y_h = y_combined.iloc[:, h-1]
        
        print(f"  Training {horizon_str}: {model_name}...", end=' ')
        
        # Create and train model for this horizon
        model_h = create_model(model_name, params)
        model_h.fit(X_combined, y_h)
        
        models_dict[horizon_str] = model_h
        print(f"✓")
    
    elapsed_time = time.time() - start_time
    
    if LOG_TRAINING_TIME:
        print(f"\n✓ All {len(models_dict)} models trained in {elapsed_time:.2f}s")
    else:
        print(f"\n✓ All {len(models_dict)} models trained")
    
    return PerHorizonWrapper(models_dict)


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
