import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
import optuna
from clearml import Task, Logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hourly_forecast_model.helper import (
    PROJECT_ROOT, N_STEPS_AHEAD, DEVICE, load_data
)

# Tuning configuration
N_TRIALS = 100
RANDOM_STATE = 42
LOG_EVERY_N_TRIALS = 5  # Reduce logging frequency
USE_PRUNING = True  # Enable Optuna pruning for faster convergence

# =============== INITIALIZE CLEARML TASK ===============
task = Task.init(
    project_name="HanoiTemp_Forecast",
    task_name="Hourly_Forecast_MultiOutput_Tuning"
)
logger = Logger.current_logger()


def objective_random_forest(trial, X_train, y_train, X_dev, y_dev):
    """Objective function for Random Forest multi-output model."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),  # Reduced max
        'max_depth': trial.suggest_int('max_depth', 8, 25),  # Narrower range
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),  # Removed None
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    model = RandomForestRegressor(**params)
    
    # Wrap in MultiOutputRegressor for multiple targets
    multi_model = MultiOutputRegressor(model, n_jobs=-1)
    multi_model.fit(X_train, y_train)
    
    # Predict and calculate average RMSE across all hours
    y_pred = multi_model.predict(X_dev)
    rmse_per_hour = [np.sqrt(mean_squared_error(y_dev.iloc[:, i], y_pred[:, i])) 
                     for i in range(y_dev.shape[1])]
    avg_rmse = np.mean(rmse_per_hour)
    
    # Reduced logging frequency - only log every N trials
    iteration = trial.number
    if iteration % LOG_EVERY_N_TRIALS == 0:
        logger.report_scalar(title="RandomForest_RMSE", series="avg", value=avg_rmse, iteration=iteration)
        # Log only key hours to reduce overhead
        for i in [1, 6, 12, 18, 24]:
            if i <= len(rmse_per_hour):
                logger.report_scalar(title="RandomForest_RMSE", series=f"t+{i}h", value=rmse_per_hour[i-1], iteration=iteration)
    
    return avg_rmse


def objective_xgboost(trial, X_train, y_train, X_dev, y_dev):
    """Objective function for XGBoost multi-output model."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=100),  # Reduced max
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 3),  # Reduced max
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'tree_method': 'hist',
        'device': 'cuda' if str(DEVICE) == 'cuda' else 'cpu',  # GPU acceleration
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': 50  # Early stopping
    }
    
    # Extract early stopping param for fit
    early_stopping = params.pop('early_stopping_rounds')
    
    model = xgb.XGBRegressor(**params)
    multi_model = MultiOutputRegressor(model, n_jobs=-1)
    
    # Fit with early stopping using eval_set
    multi_model.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],
        verbose=False
    )
    
    y_pred = multi_model.predict(X_dev)
    rmse_per_hour = [np.sqrt(mean_squared_error(y_dev.iloc[:, i], y_pred[:, i])) 
                     for i in range(y_dev.shape[1])]
    avg_rmse = np.mean(rmse_per_hour)
    
    # Reduced logging frequency
    iteration = trial.number
    if iteration % LOG_EVERY_N_TRIALS == 0:
        logger.report_scalar(title="XGBoost_RMSE", series="avg", value=avg_rmse, iteration=iteration)
        for i in [1, 6, 12, 18, 24]:
            if i <= len(rmse_per_hour):
                logger.report_scalar(title="XGBoost_RMSE", series=f"t+{i}h", value=rmse_per_hour[i-1], iteration=iteration)
    
    return avg_rmse


def objective_lightgbm(trial, X_train, y_train, X_dev, y_dev):
    """Objective function for LightGBM multi-output model."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=100),  # Reduced max
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),  # Reduced max
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 40),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),  # Reduced max
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),  # Reduced max
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'n_jobs': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    multi_model = MultiOutputRegressor(model, n_jobs=-1)
    multi_model.fit(X_train, y_train)
    
    y_pred = multi_model.predict(X_dev)
    rmse_per_hour = [np.sqrt(mean_squared_error(y_dev.iloc[:, i], y_pred[:, i])) 
                     for i in range(y_dev.shape[1])]
    avg_rmse = np.mean(rmse_per_hour)
    
    # Reduced logging frequency
    iteration = trial.number
    if iteration % LOG_EVERY_N_TRIALS == 0:
        logger.report_scalar(title="LightGBM_RMSE", series="avg", value=avg_rmse, iteration=iteration)
        for i in [1, 6, 12, 18, 24]:
            if i <= len(rmse_per_hour):
                logger.report_scalar(title="LightGBM_RMSE", series=f"t+{i}h", value=rmse_per_hour[i-1], iteration=iteration)
    
    return avg_rmse


def objective_catboost(trial, X_train, y_train, X_dev, y_dev):
    """Objective function for CatBoost multi-output model."""
    params = {
        'iterations': trial.suggest_int('iterations', 200, 600, step=100),  # Reduced max
        'depth': trial.suggest_int('depth', 3, 8),  # Reduced max
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 7),  # Reduced max
        'border_count': trial.suggest_int('border_count', 32, 128),  # Reduced max
        'task_type': 'GPU' if str(DEVICE) == 'cuda' else 'CPU',  # GPU acceleration
        'random_state': RANDOM_STATE,
        'verbose': False,
        'early_stopping_rounds': 50  # Early stopping
    }
    
    model = CatBoostRegressor(**params)
    multi_model = MultiOutputRegressor(model, n_jobs=-1)
    multi_model.fit(
        X_train, y_train,
        eval_set=(X_dev, y_dev),
        verbose=False
    )
    
    y_pred = multi_model.predict(X_dev)
    rmse_per_hour = [np.sqrt(mean_squared_error(y_dev.iloc[:, i], y_pred[:, i])) 
                     for i in range(y_dev.shape[1])]
    avg_rmse = np.mean(rmse_per_hour)
    
    # Reduced logging frequency
    iteration = trial.number
    if iteration % LOG_EVERY_N_TRIALS == 0:
        logger.report_scalar(title="CatBoost_RMSE", series="avg", value=avg_rmse, iteration=iteration)
        for i in [1, 6, 12, 18, 24]:
            if i <= len(rmse_per_hour):
                logger.report_scalar(title="CatBoost_RMSE", series=f"t+{i}h", value=rmse_per_hour[i-1], iteration=iteration)
    
    return avg_rmse


def tune_model(model_name, X_train, y_train, X_dev, y_dev, n_trials=N_TRIALS):
    """
    Tune hyperparameters for a specific model.
    
    Args:
        model_name (str): Name of model to tune
        X_train, y_train: Training data
        X_dev, y_dev: Development data for validation
        n_trials (int): Number of Optuna trials
    
    Returns:
        dict: Best parameters and metrics
    """
    print(f"\n{'='*70}")
    print(f"Tuning {model_name} Multi-Output Model")
    print(f"{'='*70}")
    print(f"Trials: {n_trials}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Targets: {y_train.shape[1]} hours")
    print(f"{'='*70}\n")
    
    # Select objective function
    objective_funcs = {
        'RandomForest': objective_random_forest,
        'XGBoost': objective_xgboost,
        'LightGBM': objective_lightgbm,
        'CatBoost': objective_catboost
    }
    
    objective_func = objective_funcs[model_name]
    
    # Create study with pruning if enabled
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5) if USE_PRUNING else optuna.pruners.NopPruner()
    
    study = optuna.create_study(
        direction='minimize',
        study_name=f'{model_name}_hourly_multioutput',
        pruner=pruner
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective_func(trial, X_train, y_train, X_dev, y_dev),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Sequential execution for stability with MultiOutputRegressor
    )
    
    # Get best results
    best_params = study.best_params
    best_rmse = study.best_value
    
    print(f"\n{'='*70}")
    print(f"Best {model_name} Results")
    print(f"{'='*70}")
    print(f"Best Average RMSE: {best_rmse:.4f}°C")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"{'='*70}\n")
    
    # Log best result to ClearML
    logger.report_scalar(title="Best_Model_RMSE", series=model_name, value=best_rmse, iteration=0)
    logger.report_text(f"{model_name} Best RMSE: {best_rmse:.4f}°C")
    
    return {
        'model': model_name,
        'best_rmse': float(best_rmse),
        'params': best_params,
        'n_trials': n_trials
    }


def main():
    """Run hyperparameter tuning for all models."""
    print("\n" + "="*70)
    print("HOURLY FORECAST MULTI-OUTPUT MODEL TUNING")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    X_train, y_train, X_dev, y_dev, _, _ = load_data()
    
    if X_train is None:
        print("✗ Failed to load data. Run process.py first.")
        return
    
    print(f"✓ Data loaded:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Dev: {X_dev.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Targets: {y_train.shape[1]} hours (t+1h to t+{N_STEPS_AHEAD}h)")
    
    # Tune all models
    models_to_tune = ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost']
    results = {}
    
    for model_name in models_to_tune:
        result = tune_model(model_name, X_train, y_train, X_dev, y_dev, n_trials=N_TRIALS)
        results[model_name] = result
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['best_rmse'])
    
    print(f"\n{'='*70}")
    print("TUNING SUMMARY")
    print(f"{'='*70}")
    print("\nAll Models Performance:")
    for model_name, result in results.items():
        print(f"  {model_name}: {result['best_rmse']:.4f}°C")
    
    print(f"\n🏆 Best Model: {best_model[0]} (RMSE: {best_model[1]['best_rmse']:.4f}°C)")
    print(f"{'='*70}\n")
    
    # Log final summary to ClearML
    logger.report_text(f"Best Overall Model: {best_model[0]} (RMSE: {best_model[1]['best_rmse']:.4f}°C)")
    logger.report_scalar(title="Final_Best_RMSE", series="Winner", value=best_model[1]['best_rmse'], iteration=0)
    
    # Save results
    results_dir = os.path.join(PROJECT_ROOT, 'src', 'hourly_forecast_model', 'final')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'tuning_results_multioutput.json')
    with open(results_file, 'w') as f:
        json.dump({
            'best_model': best_model[0],
            'best_rmse': best_model[1]['best_rmse'],
            'all_results': results
        }, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    # Save best model config
    best_config_file = os.path.join(results_dir, 'best_model_config.json')
    with open(best_config_file, 'w') as f:
        json.dump({
            'model': best_model[0],
            'params': best_model[1]['params'],
            'best_rmse': best_model[1]['best_rmse'],
            'n_targets': N_STEPS_AHEAD
        }, f, indent=2)
    
    print(f"✓ Best model config saved to: {best_config_file}")
    print("\n✅ Tuning complete!")
    print("🎉 All results logged to ClearML!")
    print(f"\n⚡ Performance optimizations enabled:")
    print(f"  - Reduced logging (every {LOG_EVERY_N_TRIALS} trials)")
    print(f"  - Pruning: {'Enabled' if USE_PRUNING else 'Disabled'}")
    print(f"  - GPU acceleration: {DEVICE.upper()}")
    print(f"  - Early stopping: Enabled for XGBoost & CatBoost")


if __name__ == "__main__":
    main()
