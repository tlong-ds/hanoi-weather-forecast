import pandas as pd
import numpy as np
import time
import os
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Import configuration
from src.daily_forecast_model.helper import (
    PROJECT_ROOT, N_STEPS_AHEAD, PER_TARGET_PARAMS,
    MODELS_DIR, VERBOSE_TRAINING, LOG_TRAINING_TIME, DEVICE
)


def load_data_for_target(day_step, use_combined_train_dev=True):
    """
    Load preprocessed data for specific target day (t+1 through t+5).
    
    Args:
        day_step (int): Target day (1-5)
        use_combined_train_dev (bool): If True, combine train+dev for final training
    
    Returns:
        tuple: (X_train, y_train, X_dev, y_dev) or (X_combined, y_combined, None, None)
    """
    day_str = f"t_{day_step}"
    data_dir = os.path.join(PROJECT_ROOT, 'processed_data', f'target_{day_str}')
    
    if VERBOSE_TRAINING:
        print(f"  Loading data from: {data_dir}")
    
    try:
        X_train = pd.read_csv(os.path.join(data_dir, f'X_train_t{day_step}.csv'), index_col=0)
        y_train = pd.read_csv(os.path.join(data_dir, f'y_train_t{day_step}.csv'), index_col=0)
        X_dev = pd.read_csv(os.path.join(data_dir, f'X_dev_t{day_step}.csv'), index_col=0)
        y_dev = pd.read_csv(os.path.join(data_dir, f'y_dev_t{day_step}.csv'), index_col=0)
        
        # Convert y to 1D array
        y_train = y_train.values.ravel()
        y_dev = y_dev.values.ravel()
        
        if use_combined_train_dev:
            # Combine train and dev for final training
            X_combined = pd.concat([X_train, X_dev], axis=0)
            y_combined = np.concatenate([y_train, y_dev])
            
            if VERBOSE_TRAINING:
                print(f"  ✓ Combined train+dev: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
            
            return X_combined, y_combined, None, None
        else:
            if VERBOSE_TRAINING:
                print(f"  ✓ Train: {X_train.shape[0]} samples, Dev: {X_dev.shape[0]} samples, Features: {X_train.shape[1]}")
            
            return X_train, y_train, X_dev, y_dev
    
    except FileNotFoundError as e:
        print(f"  ✗ ERROR: Files not found in '{data_dir}'.")
        print(f"    Make sure you ran preprocessing to generate per-target data.")
        print(f"    Error: {e}")
        return None, None, None, None


def create_model(model_name, params):
    """
    Create a model instance with the given parameters from tuning.
    
    Args:
        model_name (str): Model architecture name
        params (dict): Model hyperparameters from tuning results
    
    Returns:
        model: Instantiated model
    """
    if model_name == "RandomForest":
        return RandomForestRegressor(**params)
    
    elif model_name == "XGBoost":
        return XGBRegressor(**params)
    
    elif model_name == "LightGBM":
        return LGBMRegressor(**params)
    
    elif model_name == "CatBoost":
        return CatBoostRegressor(**params)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_per_target_models(use_combined_train_dev=True):
    """
    Train individual models for each forecast horizon using optimized parameters.
    
    Args:
        use_combined_train_dev (bool): If True, train on combined train+dev data
    
    Returns:
        dict: Dictionary of trained models {target_name: model}
    """
    if PER_TARGET_PARAMS is None:
        raise FileNotFoundError(
            "No per-target tuning results found. Please run tuning first:\n"
            "  1. Execute run_tuning.ipynb in Google Colab, or\n"
            "  2. Run: python src/daily_forecast_model/tune.py"
        )
    
    trained_models = {}
    training_times = {}
    training_metadata = {}
    
    print(f"\n{'='*70}")
    print(f"TRAINING PER-TARGET MODELS")
    print(f"{'='*70}")
    print(f"Targets: {len(PER_TARGET_PARAMS)}")
    print(f"Data: {'Combined train+dev' if use_combined_train_dev else 'Separate train/dev'}")
    print(f"{'='*70}\n")
    
    for day_step in range(1, N_STEPS_AHEAD + 1):
        target_name = f"t+{day_step}"
        
        print(f"{'='*70}")
        print(f"[{target_name}]")
        print(f"{'='*70}")
        
        # Check if tuning results exist for this target
        if target_name not in PER_TARGET_PARAMS:
            print(f"  ✗ No tuning results found for {target_name}, skipping...")
            continue
        
        # Get model configuration
        target_config = PER_TARGET_PARAMS[target_name]
        model_name = target_config['model']
        params = target_config['params']
        feature_type = target_config['feature_type']
        
        print(f"  Model: {model_name}")
        print(f"  Features: {feature_type} ({target_config['n_features']} features)")
        print(f"  Tuned RMSE: {target_config['best_rmse']:.4f}°C")
        
        # Load target-specific data
        X_train, y_train, X_dev, y_dev = load_data_for_target(day_step, use_combined_train_dev)
        
        if X_train is None:
            print(f"  ✗ Failed to load data for {target_name}, skipping...")
            continue
        
        # Create and train model
        print(f"  Training...", end="", flush=True)
        start_time = time.time()
        
        try:
            model = create_model(model_name, params)
            model.fit(X_train, y_train)
            
            elapsed_time = time.time() - start_time
            training_times[target_name] = elapsed_time
            trained_models[target_name] = model
            
            # Store metadata
            training_metadata[target_name] = {
                'model': model_name,
                'feature_type': feature_type,
                'n_features': target_config['n_features'],
                'n_samples': len(y_train),
                'tuned_rmse': target_config['best_rmse'],
                'training_time': elapsed_time
            }
            
            if LOG_TRAINING_TIME:
                print(f" ✓ Complete ({elapsed_time:.2f}s)")
            else:
                print(f" ✓ Complete")
        
        except Exception as e:
            print(f" ✗ Failed")
            print(f"  Error: {e}")
            continue
        
        print()
    
    # Print training summary
    if trained_models:
        print(f"{'='*70}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Successfully trained: {len(trained_models)}/{N_STEPS_AHEAD} models")
        
        if LOG_TRAINING_TIME and training_times:
            total_time = sum(training_times.values())
            print(f"\nTraining Times:")
            for target_name, elapsed_time in training_times.items():
                print(f"  {target_name}: {elapsed_time:.2f}s")
            print(f"  Total: {total_time:.2f}s")
        
        print(f"{'='*70}\n")
        
        # Save metadata
        save_training_metadata(training_metadata)
    
    return trained_models


def save_training_metadata(metadata):
    """Save training metadata to JSON file."""
    metadata_path = os.path.join(MODELS_DIR, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Training metadata saved to: {metadata_path}")


def save_models(trained_models):
    """
    Save trained models to disk.
    
    Args:
        trained_models (dict): Dictionary of trained models {target_name: model}
    """
    if not trained_models:
        print("No models to save.")
        return
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SAVING MODELS")
    print(f"{'='*70}")
    
    for target_name, model in trained_models.items():
        model_filename = f"model_{target_name}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        try:
            joblib.dump(model, model_path)
            print(f"  ✓ {target_name}: {model_path}")
        except Exception as e:
            print(f"  ✗ {target_name}: Failed - {e}")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PER-TARGET MODEL TRAINING")
    print("="*70 + "\n")
    
    # Train models using combined train+dev data (final training)
    trained_models = train_per_target_models(use_combined_train_dev=True)
    
    # Save trained models
    if trained_models:
        save_models(trained_models)
        print("✅ Training complete! Models saved to:", MODELS_DIR)
    else:
        print("❌ No models were trained successfully.")
