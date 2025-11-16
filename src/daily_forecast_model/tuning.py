# ================== run_tuning.py ==================
# Two-Stage Hyperparameter Tuning:
#   Stage 1: Find best architecture (40 trials)
#   Stage 2: Deep tune winner per target (100 trials × 5)
# ====================================================

import warnings
warnings.filterwarnings("ignore")

import optuna
import numpy as np
import pandas as pd
import os
import json
from clearml import Task, Logger
from daily_forecast_model.helper import DEVICE 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# =============== CONFIGURATION ===============
N_STEPS_AHEAD = 5
STAGE1_TRIALS = 40   # Architecture selection
STAGE2_TRIALS = 100  # Deep tuning per target

# =============== INITIALIZE CLEARML TASK ===============
task = Task.init(
    project_name="HanoiTemp_Forecast",
    task_name="Two_Stage_Tuning (Architecture + Per_Target)"
)
logger = Logger.current_logger()

# =============== DATA LOADING FUNCTIONS ===============
def load_data_for_day(day_step):
    """
    Load preprocessed data for specific target day (t+1 through t+5).
    These datasets already have target-specific features selected.
    """
    day_str = f"t_{day_step}"
    data_dir = f'processed_data/target_{day_str}'
    
    print(f"Loading data from: {data_dir}")
    
    try:
        X_train_file = f'X_train_t{day_step}.csv'
        y_train_file = f'y_train_t{day_step}.csv'
        X_dev_file = f'X_dev_t{day_step}.csv'
        y_dev_file = f'y_dev_t{day_step}.csv'

        X_train = pd.read_csv(os.path.join(data_dir, X_train_file), index_col=0)
        y_train = pd.read_csv(os.path.join(data_dir, y_train_file), index_col=0)
        X_dev = pd.read_csv(os.path.join(data_dir, X_dev_file), index_col=0)
        y_dev = pd.read_csv(os.path.join(data_dir, y_dev_file), index_col=0)
        
        # Convert y to 1D array
        return X_train, y_train.values.ravel(), X_dev, y_dev.values.ravel()
    
    except FileNotFoundError as e:
        print(f"❌ ERROR: Files not found in '{data_dir}'.")
        print(f"  Make sure you ran 'preprocessing.py' first.")
        print(f"  Error: {e}")
        return None, None, None, None


def load_combined_features_data():
    """
    Load combined features for Stage 1 (architecture selection).
    Uses the first target's data as representative sample.
    """
    print("\n" + "="*70)
    print("STAGE 1: Loading COMBINED features for architecture selection")
    print("="*70)
    
    # Use t+3 (middle target) as representative
    X_train, y_train, X_dev, y_dev = load_data_for_day(3)
    
    if X_train is not None:
        print(f"✅ Loaded combined features: {X_train.shape[1]} features")
        print(f"✅ Train samples: {X_train.shape[0]}, Dev samples: {X_dev.shape[0]}")
    
    return X_train, y_train, X_dev, y_dev

# =============== STAGE 1: ARCHITECTURE SELECTION ===============
def objective_stage1(trial, X_train, y_train, X_dev, y_dev):
    """
    Stage 1: Quick architecture comparison using categorical hyperparameters.
    Goal: Find which model architecture works best.
    """
    model_name = trial.suggest_categorical(
        "model_name",
        ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
    )

    # Use CATEGORICAL hyperparameters for faster exploration
    if model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]),
            "max_depth": trial.suggest_categorical("max_depth", [8, 12, 16]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10]),
            "random_state": 42,
            "n_jobs": -1,
        }
        model = RandomForestRegressor(**params)

    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [300, 500, 700]),
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
            "subsample": trial.suggest_categorical("subsample", [0.7, 0.85, 1.0]),
            "tree_method": "hist",
            "device": "cuda" if str(DEVICE) == "cuda" else "cpu",
            "random_state": 42,
        }
        model = XGBRegressor(**params)

    elif model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [300, 500, 700]),
            "num_leaves": trial.suggest_categorical("num_leaves", [31, 50, 70]),
            "max_depth": trial.suggest_categorical("max_depth", [6, 9, 12]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
            "subsample": trial.suggest_categorical("subsample", [0.7, 0.85, 1.0]),
            "device_type": "cpu",
            "n_jobs": -1,
            "random_state": 42,
        }
        model = LGBMRegressor(**params)

    elif model_name == "CatBoost":
        params = {
            "iterations": trial.suggest_categorical("iterations", [300, 500, 700]),
            "depth": trial.suggest_categorical("depth", [4, 6, 8]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
            "l2_leaf_reg": trial.suggest_categorical("l2_leaf_reg", [1.0, 3.0, 5.0]),
            "task_type": "GPU" if str(DEVICE) == "cuda" else "CPU",
            "verbose": 0,
            "random_state": 42,
        }
        model = CatBoostRegressor(**params)

    else:
        raise ValueError(f"❌ Model {model_name} not available.")

    # Train and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_dev)

    rmse = np.sqrt(mean_squared_error(y_dev, y_pred))
    mae = mean_absolute_error(y_dev, y_pred)
    
    # Log to ClearML
    logger.report_scalar("Stage1_RMSE", model_name, rmse, trial.number)
    logger.report_scalar("Stage1_MAE", model_name, mae, trial.number)
    
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
            "random_state": 42,
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
            "random_state": 42
        }
        model = XGBRegressor(**params)
    
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
            "device_type": "cpu",
            "n_jobs": -1,
            "random_state": 42
        }
        model = LGBMRegressor(**params)
    
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
            "random_state": 42
        }
        model = CatBoostRegressor(**params)
    
    else:
        raise ValueError(f"❌ Unknown model: {model_name}")
    
    # Train and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_dev)
    
    rmse = np.sqrt(mean_squared_error(y_dev, y_pred))
    
    return rmse
    return rmse


# =============== MAIN: TWO-STAGE TUNING PIPELINE ===============
if __name__ == "__main__":
    
    print("="*70)
    print("🚀 TWO-STAGE HYPERPARAMETER TUNING PIPELINE")
    print("="*70)
    
    # ========== STAGE 1: ARCHITECTURE SELECTION ==========
    print("\n" + "="*70)
    print("STAGE 1: ARCHITECTURE SELECTION")
    print("="*70)
    print(f"Goal: Find the best model architecture")
    print(f"Method: Test 4 models with {STAGE1_TRIALS} trials")
    print(f"Features: Using combined features (representative)")
    print("="*70)
    
    # Load combined features data
    X_train_comb, y_train_comb, X_dev_comb, y_dev_comb = load_combined_features_data()
    
    if X_train_comb is None:
        print("❌ Failed to load data for Stage 1. Exiting.")
        exit(1)
    
    # Run Stage 1 optimization
    print(f"\nRunning {STAGE1_TRIALS} trials for architecture selection...")
    study1 = optuna.create_study(
        direction="minimize",
        study_name="Stage1_Architecture_Selection"
    )
    study1.optimize(
        lambda trial: objective_stage1(trial, X_train_comb, y_train_comb, X_dev_comb, y_dev_comb),
        n_trials=STAGE1_TRIALS,
        show_progress_bar=True
    )
    
    # Get best architecture
    best_architecture = study1.best_params["model_name"]
    best_stage1_rmse = study1.best_value
    
    print("\n" + "="*70)
    print("STAGE 1 RESULTS")
    print("="*70)
    print(f"🏆 Best Architecture: {best_architecture}")
    print(f"✅ Best RMSE: {best_stage1_rmse:.4f}")
    print(f"Best Parameters:")
    for k, v in study1.best_params.items():
        print(f"  {k}: {v}")
    print("="*70)
    
    # Log Stage 1 results
    logger.report_text(f"Stage 1 Winner: {best_architecture} (RMSE={best_stage1_rmse:.4f})", level="INFO")
    logger.report_scalar("Stage1_Best_RMSE", "Final", best_stage1_rmse, 0)
    
    # Save Stage 1 results
    stage1_results = {
        "best_architecture": best_architecture,
        "best_rmse": best_stage1_rmse,
        "best_params": study1.best_params
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/stage1_architecture_selection.json', 'w') as f:
        json.dump(stage1_results, f, indent=2)
    print(f"\n✅ Stage 1 results saved to 'results/stage1_architecture_selection.json'")
    
    
    # ========== STAGE 2: DEEP PER-TARGET TUNING ==========
    print("\n" + "="*70)
    print("STAGE 2: DEEP PER-TARGET HYPERPARAMETER TUNING")
    print("="*70)
    print(f"Goal: Optimize {best_architecture} for each forecast horizon")
    print(f"Method: {STAGE2_TRIALS} trials per target × {N_STEPS_AHEAD} targets = {STAGE2_TRIALS * N_STEPS_AHEAD} trials")
    print(f"Features: Target-specific (short-term for t+1, long-term for t+5)")
    print("="*70)
    
    best_params_all_targets = {}
    
    for day_step in range(1, N_STEPS_AHEAD + 1):
        target_name = f"t+{day_step}"
        
        print(f"\n{'='*70}")
        print(f"🎯 TUNING FOR TARGET: {target_name}")
        print(f"{'='*70}")
        
        # Load target-specific data
        X_train, y_train, X_dev, y_dev = load_data_for_day(day_step)
        
        if X_train is None:
            print(f"❌ Skipping {target_name} - data not found")
            continue
        
        # Determine feature type
        if day_step == 1:
            feature_type = "SHORT-TERM"
        elif day_step == N_STEPS_AHEAD:
            feature_type = "LONG-TERM"
        else:
            feature_type = "COMBINED"
        
        print(f"✅ Data loaded: {X_train.shape[0]} train samples, {X_train.shape[1]} features ({feature_type})")
        print(f"✅ Target: {target_name}")
        print(f"✅ Model: {best_architecture}")
        print(f"✅ Running {STAGE2_TRIALS} optimization trials...")
        
        # Run Stage 2 optimization for this target
        study2 = optuna.create_study(
            direction="minimize",
            study_name=f"Stage2_{best_architecture}_{target_name}"
        )
        study2.optimize(
            lambda trial: objective_stage2(trial, best_architecture, X_train, y_train, X_dev, y_dev),
            n_trials=STAGE2_TRIALS,
            show_progress_bar=True
        )
        
        # Get best results for this target
        best_params = study2.best_params
        best_rmse = study2.best_value
        
        print(f"\n{'='*70}")
        print(f"RESULTS FOR {target_name}")
        print(f"{'='*70}")
        print(f"✅ Best RMSE: {best_rmse:.4f}")
        print(f"Best Hyperparameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"{'='*70}")
        
        # Log to ClearML
        logger.report_scalar("Stage2_Best_RMSE", target_name, best_rmse, day_step)
        logger.report_text(f"{target_name} Best RMSE: {best_rmse:.4f}", level="INFO")
        
        # Store results
        best_params_all_targets[target_name] = {
            "model": best_architecture,
            "feature_type": feature_type,
            "n_features": X_train.shape[1],
            "params": best_params,
            "best_rmse": best_rmse
        }
    
    # Save Stage 2 results
    with open('results/stage2_best_params_per_target.json', 'w') as f:
        json.dump(best_params_all_targets, f, indent=2)
    
    print("\n" + "="*70)
    print("✅✅✅ TWO-STAGE TUNING COMPLETE ✅✅✅")
    print("="*70)
    print("\nFinal Summary:")
    print(f"  Stage 1: Best architecture = {best_architecture}")
    print(f"  Stage 2: Optimized hyperparameters for {len(best_params_all_targets)} targets")
    print("\nPer-Target Results:")
    for target, info in best_params_all_targets.items():
        print(f"  {target}: RMSE={info['best_rmse']:.4f} ({info['feature_type']}, {info['n_features']} features)")
    
    print("\n📁 Results saved:")
    print("  - results/stage1_architecture_selection.json")
    print("  - results/stage2_best_params_per_target.json")
    print("="*70)
    print("🎉 All results logged to ClearML!")
    print("="*70)