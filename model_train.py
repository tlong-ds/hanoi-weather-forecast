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
from model_helper import (
    TARGET_COLUMNS, RANDOM_STATE, MODEL_CONFIGS, 
    SAVE_MODELS, LOAD_SAVED_MODELS, MODEL_FORMAT, MODELS_DIR,
    LOG_TRAINING_TIME, VERBOSE_TRAINING, get_enabled_models, load_data
)

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
    Save trained models to disk using joblib or pickle.
    
    Args:
        trained_models (dict): Dictionary of trained model objects
    """
    print(f"\nSaving trained models to '{MODELS_DIR}/'...")
    
    for model_name, model in trained_models.items():
        try:
            filename = f"{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.{MODEL_FORMAT}"
            filepath = f"{MODELS_DIR}/{filename}"
            
            if MODEL_FORMAT == 'joblib':
                joblib.dump(model, filepath)
            else:
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            
            print(f"  ✓ {model_name} -> {filepath}")
        except Exception as e:
            print(f"  ✗ Failed to save {model_name}: {e}")


def load_saved_models():
    """
    Load previously trained models from disk.
    
    Returns:
        dict: Dictionary of loaded models, or empty dict if none found
    """
    loaded_models = {}
    
    if not LOAD_SAVED_MODELS:
        return loaded_models
    
    print(f"\nLoading saved models from '{MODELS_DIR}/'...")
    
    try:
        for model_name in get_enabled_models().keys():
            filename = f"{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.{MODEL_FORMAT}"
            filepath = f"{MODELS_DIR}/{filename}"
            
            if not os.path.exists(filepath):
                continue
            
            try:
                if MODEL_FORMAT == 'joblib':
                    model = joblib.load(filepath)
                else:
                    import pickle
                    with open(filepath, 'rb') as f:
                        model = pickle.load(f)
                
                loaded_models[model_name] = model
                print(f"  ✓ Loaded {model_name}")
            except Exception as e:
                print(f"  ✗ Failed to load {model_name}: {e}")
    
    except Exception as e:
        print(f"✗ Error loading models: {e}")
    
    return loaded_models

