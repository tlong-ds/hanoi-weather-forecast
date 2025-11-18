import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
import optuna
from clearml import Task, Logger

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hourly_forecast_model.helper import (
    PROJECT_ROOT, N_STEPS_AHEAD, DEVICE, DATA_DIR
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# =============== CONFIGURATION ===============
TRIALS_PER_HORIZON = 100  # Trials per horizon (test all models)
RANDOM_STATE = 42

# =============== INITIALIZE CLEARML TASK ===============
task = Task.init(
    project_name="HanoiTemp_Forecast",
    task_name="Hourly_Per_Horizon_Tuning (Best Model Per Horizon)"
)
logger = Logger.current_logger()


# =============== DATA LOADING ===============
def load_data_for_horizon(h_step):
    """
    Load data for specific horizon t+{h_step}h.
    Returns X_train, y_train (single target), X_dev, y_dev (single target)
    """
    X_train_file = os.path.join(DATA_DIR, 'X_train_transformed.csv')
    y_train_file = os.path.join(DATA_DIR, 'y_train.csv')
    X_dev_file = os.path.join(DATA_DIR, 'X_dev_transformed.csv')
    y_dev_file = os.path.join(DATA_DIR, 'y_dev.csv')
    
    target_col = f"target_temp_t+{h_step}h"
    
    print(f"  Loading data for: {target_col}")
    
    try:
        X_train = pd.read_csv(X_train_file, index_col=0)
        y_train_df = pd.read_csv(y_train_file, index_col=0)
        X_dev = pd.read_csv(X_dev_file, index_col=0)
        y_dev_df = pd.read_csv(y_dev_file, index_col=0)
        
        if target_col not in y_train_df.columns:
            raise KeyError(f"Column {target_col} not found in y_train.csv")
        
        # Extract single target column
        y_train = y_train_df[target_col].values.ravel()
        y_dev = y_dev_df[target_col].values.ravel()
        
        return X_train, y_train, X_dev, y_dev
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: Data files not found: {e}")
        print("   Make sure to run process.py first.")
        return None, None, None, None
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return None, None, None, None


# =============== SINGLE-STAGE OBJECTIVE: FIND BEST MODEL PER HORIZON ===============
def objective_per_horizon(trial, X_train, y_train, X_dev, y_dev):
    """
    Single-stage optimization: Test all model architectures for each horizon.
    Returns best model + hyperparameters for this specific horizon.
    """
    model_name = trial.suggest_categorical(
        "model_name",
        ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
    )
    
    if model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
        model = RandomForestRegressor(**params)
    
    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "tree_method": "hist",
            "device": "cuda" if str(DEVICE) == "cuda" else "cpu",
            "random_state": RANDOM_STATE
        }
        model = xgb.XGBRegressor(**params)
    
    elif model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        }
        model = lgb.LGBMRegressor(**params)
    
    elif model_name == "CatBoost":
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "depth": trial.suggest_int("depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "task_type": "GPU" if str(DEVICE) == "cuda" else "CPU",
            "verbose": 0,
            "random_state": RANDOM_STATE
        }
        model = CatBoostRegressor(**params)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_dev)
    
    rmse = np.sqrt(mean_squared_error(y_dev, y_pred))
    mae = mean_absolute_error(y_dev, y_pred)
    
    # Log to ClearML
    logger.report_scalar(title="RMSE_by_Model", series=model_name, value=rmse, iteration=trial.number)
    logger.report_scalar(title="MAE_by_Model", series=model_name, value=mae, iteration=trial.number)
    
    print(f"  Trial {trial.number:3d} | {model_name:12s}: RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    return rmse  # Minimize RMSE


# =============== MAIN: PER-HORIZON TUNING PIPELINE ===============
def main():
    """Run hyperparameter tuning for all 24 horizons (find best model per horizon)."""
    
    print("="*80)
    print("🚀 PER-HORIZON HYPERPARAMETER TUNING FOR HOURLY FORECAST")
    print("="*80)
    print(f"Strategy: Find the best model (RF/XGB/LGB/CatBoost) for each horizon")
    print(f"Method: {TRIALS_PER_HORIZON} trials per horizon × {N_STEPS_AHEAD} horizons = {TRIALS_PER_HORIZON * N_STEPS_AHEAD} total trials")
    print("="*80)
    
    best_params_all_horizons = {}
    results_dir = os.path.join(PROJECT_ROOT, 'src', 'hourly_forecast_model', 'tuning_results')
    os.makedirs(results_dir, exist_ok=True)
    
    for h_step in range(1, N_STEPS_AHEAD + 1):
        horizon_str = f"t+{h_step}h"
        
        print(f"\n{'='*80}")
        print(f"🎯 TUNING FOR HORIZON: {horizon_str} ({h_step}/{N_STEPS_AHEAD})")
        print(f"{'='*80}")
        
        # Load horizon-specific data
        X_train, y_train, X_dev, y_dev = load_data_for_horizon(h_step)
        
        if X_train is None:
            print(f"❌ Skipping {horizon_str} - data not found")
            continue
        
        print(f"✅ Data loaded: {X_train.shape[0]} train samples, {X_train.shape[1]} features")
        print(f"✅ Target: {horizon_str}")
        print(f"✅ Running {TRIALS_PER_HORIZON} optimization trials (testing all models)...")
        
        # Run optimization for this horizon
        study = optuna.create_study(
            direction="minimize",
            study_name=f"PerHorizon_{horizon_str}"
        )
        study.optimize(
            lambda trial: objective_per_horizon(trial, X_train, y_train, X_dev, y_dev),
            n_trials=TRIALS_PER_HORIZON,
            show_progress_bar=True
        )
        
        # Get best results for this horizon
        best_params = study.best_params
        best_rmse = study.best_value
        best_model = best_params["model_name"]
        
        print(f"\n{'='*80}")
        print(f"RESULTS FOR {horizon_str}")
        print(f"{'='*80}")
        print(f"🏆 Best Model: {best_model}")
        print(f"✅ Best RMSE: {best_rmse:.4f}°C")
        print(f"Best Hyperparameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"{'='*80}")
        
        # Log to ClearML
        logger.report_scalar(title="Best_RMSE_Per_Horizon", series=horizon_str, value=best_rmse, iteration=h_step)
        logger.report_text(f"{horizon_str}: {best_model} - RMSE={best_rmse:.4f}°C")
        
        # Store results
        best_params_all_horizons[horizon_str] = {
            "model": best_model,
            "n_features": X_train.shape[1],
            "params": best_params,
            "best_rmse": best_rmse
        }
    
    # Save results
    with open(os.path.join(results_dir, 'best_params_per_horizon.json'), 'w') as f:
        json.dump(best_params_all_horizons, f, indent=2)
    
    print("\n" + "="*80)
    print("✅✅✅ PER-HORIZON TUNING COMPLETE ✅✅✅")
    print("="*80)
    print(f"\nOptimized hyperparameters for {len(best_params_all_horizons)} horizons")
    print("\nPer-Horizon Results:")
    
    # Analyze model distribution
    model_counts = {}
    for horizon, info in best_params_all_horizons.items():
        model = info['model']
        model_counts[model] = model_counts.get(model, 0) + 1
        print(f"  {horizon}: {model:12s} - RMSE={info['best_rmse']:.4f}°C ({info['n_features']} features)")
    
    # Calculate statistics
    avg_rmse = np.mean([info['best_rmse'] for info in best_params_all_horizons.values()])
    
    print(f"\n📊 Model Distribution:")
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(best_params_all_horizons)
        print(f"  {model:12s}: {count:2d} horizons ({pct:.1f}%)")
    
    print(f"\n📊 Average RMSE across all horizons: {avg_rmse:.4f}°C")
    
    logger.report_scalar(title="Final_Summary", series="Average_RMSE", value=avg_rmse, iteration=0)
    
    print("\n📁 Results saved:")
    print("  - best_params_per_horizon.json")
    print("="*80)
    print("🎉 All results logged to ClearML!")
    print("="*80)


if __name__ == "__main__":
    main()
