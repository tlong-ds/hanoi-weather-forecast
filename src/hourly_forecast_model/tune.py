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
STAGE1_TRIALS = 40  # Architecture selection trials
STAGE2_TRIALS = 100  # Deep tuning trials per horizon
RANDOM_STATE = 42

# =============== INITIALIZE CLEARML TASK ===============
task = Task.init(
    project_name="HanoiTemp_Forecast",
    task_name="Hourly_Two_Stage_Tuning (Architecture + Per_Horizon)"
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


# =============== STAGE 1: ARCHITECTURE SELECTION ===============
def objective_stage1(trial, X_train, y_train, X_dev, y_dev):
    """
    Stage 1: Quick architecture comparison using categorical hyperparameters.
    Goal: Find which model architecture works best overall.
    """
    model_name = trial.suggest_categorical(
        "model_name",
        ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
    )
    
    # Use CATEGORICAL hyperparameters for faster exploration
    if model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_categorical("rf_n_estimators", [100, 200, 300]),
            "max_depth": trial.suggest_categorical("rf_max_depth", [8, 12, 16]),
            "min_samples_split": trial.suggest_categorical("rf_min_samples_split", [2, 5, 10]),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = RandomForestRegressor(**params)
    
    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_categorical("xgb_n_estimators", [300, 500, 700]),
            "max_depth": trial.suggest_categorical("xgb_max_depth", [4, 6, 8]),
            "learning_rate": trial.suggest_categorical("xgb_learning_rate", [0.01, 0.05, 0.1]),
            "subsample": trial.suggest_categorical("xgb_subsample", [0.7, 0.85, 1.0]),
            "tree_method": "hist",
            "device": "cuda" if str(DEVICE) == "cuda" else "cpu",
            "random_state": RANDOM_STATE,
        }
        model = xgb.XGBRegressor(**params)
    
    elif model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_categorical("lgbm_n_estimators", [300, 500, 700]),
            "num_leaves": trial.suggest_categorical("lgbm_num_leaves", [31, 50, 70]),
            "max_depth": trial.suggest_categorical("lgbm_max_depth", [6, 9, 12]),
            "learning_rate": trial.suggest_categorical("lgbm_learning_rate", [0.01, 0.05, 0.1]),
            "subsample": trial.suggest_categorical("lgbm_subsample", [0.7, 0.85, 1.0]),
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        }
        model = lgb.LGBMRegressor(**params)
    
    elif model_name == "CatBoost":
        params = {
            "iterations": trial.suggest_categorical("cat_iterations", [300, 500, 700]),
            "depth": trial.suggest_categorical("cat_depth", [4, 6, 8]),
            "learning_rate": trial.suggest_categorical("cat_learning_rate", [0.01, 0.05, 0.1]),
            "l2_leaf_reg": trial.suggest_categorical("cat_l2_leaf_reg", [1.0, 3.0, 5.0]),
            "task_type": "CPU",
            "random_state": RANDOM_STATE,
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
    logger.report_scalar(title="Stage1_RMSE", series=model_name, value=rmse, iteration=trial.number)
    logger.report_scalar(title="Stage1_MAE", series=model_name, value=mae, iteration=trial.number)
    
    print(f"  Trial {trial.number:3d} | {model_name:12s}: RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    return rmse  # Minimize RMSE


# =============== STAGE 2: DEEP HYPERPARAMETER TUNING ===============
def objective_stage2(trial, model_name, X_train, y_train, X_dev, y_dev):
    """
    Stage 2: Deep hyperparameter tuning for the winning architecture.
    Uses CONTINUOUS ranges for fine-grained optimization.
    """
    
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
    
    return rmse


# =============== MAIN: TWO-STAGE TUNING PIPELINE ===============
def main():
    """Run two-stage hyperparameter tuning for all 24 horizons."""
    
    print("="*80)
    print("🚀 TWO-STAGE HYPERPARAMETER TUNING PIPELINE FOR HOURLY FORECAST")
    print("="*80)
    
    # ========== STAGE 1: ARCHITECTURE SELECTION ==========
    print("\n" + "="*80)
    print("STAGE 1: ARCHITECTURE SELECTION")
    print("="*80)
    print(f"Goal: Find the best model architecture")
    print(f"Method: Test 4 models with {STAGE1_TRIALS} trials")
    print(f"Strategy: Use representative horizon (t+12h - middle of range)")
    print("="*80)
    
    # Load data for representative horizon (middle: t+12h)
    representative_horizon = 12
    print(f"\nUsing t+{representative_horizon}h as representative horizon...")
    X_train_rep, y_train_rep, X_dev_rep, y_dev_rep = load_data_for_horizon(representative_horizon)
    
    if X_train_rep is None:
        print("❌ Failed to load representative data. Exiting.")
        return
    
    print(f"✅ Data loaded: {X_train_rep.shape[0]} train samples, {X_train_rep.shape[1]} features")
    
    # Run Stage 1 optimization
    print(f"\nRunning {STAGE1_TRIALS} trials for architecture selection...")
    study1 = optuna.create_study(
        direction="minimize",
        study_name="Stage1_Architecture_Selection_Hourly"
    )
    study1.optimize(
        lambda trial: objective_stage1(trial, X_train_rep, y_train_rep, X_dev_rep, y_dev_rep),
        n_trials=STAGE1_TRIALS,
        show_progress_bar=True
    )
    
    # Get best architecture
    best_architecture = study1.best_params["model_name"]
    best_stage1_rmse = study1.best_value
    
    print("\n" + "="*80)
    print("STAGE 1 RESULTS")
    print("="*80)
    print(f"🏆 Best Architecture: {best_architecture}")
    print(f"✅ Best RMSE (on t+{representative_horizon}h): {best_stage1_rmse:.4f}°C")
    print(f"Best Parameters:")
    for k, v in study1.best_params.items():
        print(f"  {k}: {v}")
    print("="*80)
    
    # Log Stage 1 results
    logger.report_text(f"Stage 1 Winner: {best_architecture} (RMSE={best_stage1_rmse:.4f}°C)")
    logger.report_scalar(title="Stage1_Best_RMSE", series="Final", value=best_stage1_rmse, iteration=0)
    
    # Save Stage 1 results
    stage1_results = {
        "best_architecture": best_architecture,
        "best_rmse": best_stage1_rmse,
        "best_params": study1.best_params,
        "representative_horizon": f"t+{representative_horizon}h"
    }
    
    results_dir = os.path.join(PROJECT_ROOT, 'src', 'hourly_forecast_model', 'final')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'architecture_selection.json'), 'w') as f:
        json.dump(stage1_results, f, indent=2)
    print(f"\n✅ Stage 1 results saved to 'architecture_selection.json'")
    
    
    # ========== STAGE 2: DEEP PER-HORIZON TUNING ==========
    print("\n" + "="*80)
    print("STAGE 2: DEEP PER-HORIZON HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Goal: Optimize {best_architecture} for each forecast horizon")
    print(f"Method: {STAGE2_TRIALS} trials per horizon × {N_STEPS_AHEAD} horizons = {STAGE2_TRIALS * N_STEPS_AHEAD} total trials")
    print("="*80)
    
    best_params_all_horizons = {}
    
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
        print(f"✅ Model: {best_architecture}")
        print(f"✅ Running {STAGE2_TRIALS} optimization trials...")
        
        # Run Stage 2 optimization for this horizon
        study2 = optuna.create_study(
            direction="minimize",
            study_name=f"Stage2_{best_architecture}_{horizon_str}"
        )
        study2.optimize(
            lambda trial: objective_stage2(trial, best_architecture, X_train, y_train, X_dev, y_dev),
            n_trials=STAGE2_TRIALS,
            show_progress_bar=True
        )
        
        # Get best results for this horizon
        best_params = study2.best_params
        best_rmse = study2.best_value
        
        print(f"\n{'='*80}")
        print(f"RESULTS FOR {horizon_str}")
        print(f"{'='*80}")
        print(f"✅ Best RMSE: {best_rmse:.4f}°C")
        print(f"Best Hyperparameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"{'='*80}")
        
        # Log to ClearML
        logger.report_scalar(title="Stage2_Best_RMSE", series=horizon_str, value=best_rmse, iteration=h_step)
        logger.report_text(f"{horizon_str} Best RMSE: {best_rmse:.4f}°C")
        
        # Store results
        best_params_all_horizons[horizon_str] = {
            "model": best_architecture,
            "n_features": X_train.shape[1],
            "params": best_params,
            "best_rmse": best_rmse
        }
    
    # Save Stage 2 results
    with open(os.path.join(results_dir, 'best_params_per_horizon.json'), 'w') as f:
        json.dump(best_params_all_horizons, f, indent=2)
    
    print("\n" + "="*80)
    print("✅✅✅ TWO-STAGE TUNING COMPLETE ✅✅✅")
    print("="*80)
    print("\nFinal Summary:")
    print(f"  Stage 1: Best architecture = {best_architecture}")
    print(f"  Stage 2: Optimized hyperparameters for {len(best_params_all_horizons)} horizons")
    print("\nPer-Horizon Results:")
    for horizon, info in best_params_all_horizons.items():
        print(f"  {horizon}: RMSE={info['best_rmse']:.4f}°C ({info['n_features']} features)")
    
    # Calculate average performance
    avg_rmse = np.mean([info['best_rmse'] for info in best_params_all_horizons.values()])
    print(f"\n📊 Average RMSE across all horizons: {avg_rmse:.4f}°C")
    
    logger.report_scalar(title="Final_Summary", series="Average_RMSE", value=avg_rmse, iteration=0)
    
    print("\n📁 Results saved:")
    print("  - architecture_selection.json")
    print("  - best_params_per_horizon.json")
    print("="*80)
    print("🎉 All results logged to ClearML!")
    print("="*80)


if __name__ == "__main__":
    main()
