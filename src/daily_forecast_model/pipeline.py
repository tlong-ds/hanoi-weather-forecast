from clearml import PipelineDecorator, Task
from pathlib import Path
import joblib
import pandas as pd
from typing import Dict, List, Tuple, Any
import json

# Project configuration
PROJECT_NAME = "Hanoi Weather Forecast"
PIPELINE_NAME = "Daily Forecast - Decorator Pipeline"


# ============================================================================
# PIPELINE COMPONENTS (Decorated Functions)
# ============================================================================

@PipelineDecorator.component(
    return_values=['success'],
    cache=True,
    execution_queue="default"
)
def preprocess_data(data_path: str) -> bool:
    """
    Step 1: Preprocess raw weather data using process.py module
    
    This leverages the complete DataProcessor class which includes:
    - Feature engineering (rolling windows, lags, cyclical encoding)
    - Data cleaning and outlier removal
    - Train/dev/test splitting
    - Per-target feature selection
    
    Returns:
        bool: Success status
    """
    from src.daily_forecast_model.process import main as process_main
    
    print(f"📊 Running full preprocessing pipeline from process.py...")
    
    # Run the complete preprocessing pipeline
    process_main()
    
    print(f"✅ Preprocessing complete - all per-target datasets created")
    
    return True

@PipelineDecorator.component(
    return_values=['tuning_complete'],
    cache=True,
    execution_queue="default"
)
def hyperparameter_tuning(n_trials: int = 100) -> bool:
    """
    Step 2: Hyperparameter tuning using tune.py module
    
    This leverages the two-stage tuning approach:
    - Stage 1: Architecture selection (RandomForest, XGBoost, LightGBM, CatBoost)
    - Stage 2: Per-target deep hyperparameter optimization
    
    Returns:
        bool: Success status
    """
    from src.daily_forecast_model.tune import run_two_stage_tuning
    
    print(f"🎯 Running two-stage hyperparameter tuning...")
    print(f"  Stage 1: Architecture selection ({n_trials//2} trials)")
    print(f"  Stage 2: Per-target tuning ({n_trials} trials per target)")
    
    # Run the complete tuning pipeline from tune.py
    best_architecture, per_target_params = run_two_stage_tuning(
        stage1_trials=n_trials//2,
        stage2_trials=n_trials
    )
    
    print(f"✅ Tuning complete. Best architecture: {best_architecture}")
    
    return True


@PipelineDecorator.component(
    return_values=['training_complete'],
    cache=True,
    execution_queue="default"
)
def train_all_models() -> bool:
    """
    Step 3: Train all models using train.py module
    
    This leverages the complete training pipeline which:
    - Loads per-target preprocessed data
    - Uses optimized hyperparameters from tuning
    - Trains models with best architecture (XGBoost/LightGBM/CatBoost)
    - Combines train+dev for final training
    - Saves models to trained_models/
    
    Returns:
        bool: Success status
    """
    from src.daily_forecast_model.train import train_all_targets
    
    print(f"🚂 Training all 5 target models...")
    
    # Run the complete training pipeline from train.py
    train_all_targets(use_combined_train_dev=True)
    
    print(f"✅ All models trained successfully")
    
    return True


@PipelineDecorator.component(
    return_values=['evaluation_complete'],
    cache=True,
    execution_queue="default"
)
def evaluate_models() -> bool:
    """
    Step 4: Evaluate all models using evaluate.py module
    
    This leverages the comprehensive evaluation pipeline which:
    - Loads per-target test data
    - Calculates metrics (MAE, RMSE, MAPE, R²)
    - Generates visualizations (scatter plots, time series)
    - Creates evaluation report
    - Saves results to evaluate_results/
    
    Returns:
        bool: Success status
    """
    from src.daily_forecast_model.evaluate import evaluate_all_targets
    
    print(f"📈 Evaluating all models on test set...")
    
    # Run the complete evaluation pipeline from evaluate.py
    evaluate_all_targets()
    
    print(f"✅ Evaluation complete with visualizations")
    
    return True


@PipelineDecorator.component(
    return_values=['deployment_complete'],
    cache=False,  # Don't cache deployment
    execution_queue="default"
)
def deploy_models(deployment_stage: str = "staging") -> bool:
    """
    Step 5: Deploy models using deploy.py module
    
    This leverages the ClearML deployment pipeline which:
    - Registers all 5 models to ClearML Model Registry
    - Uploads model files and metadata
    - Tags models with deployment stage
    - Tracks performance metrics
    - Enables model versioning and promotion
    
    Returns:
        bool: Success status
    """
    from src.daily_forecast_model.deploy import ClearMLModelDeployer
    
    print(f"🚀 Deploying models to ClearML ({deployment_stage})...")
    
    # Initialize deployer
    deployer = ClearMLModelDeployer(
        project_name="Hanoi Weather Forecast",
        task_name=f"Model Deployment - {deployment_stage}"
    )
    
    # Initialize ClearML task
    deployer.initialize_task()
    
    # Deploy all models
    deployer.deploy_all_models(stage=deployment_stage)
    
    # Deploy preprocessors
    deployer.deploy_preprocessors(stage=deployment_stage)
    
    print(f"✅ All models deployed to {deployment_stage}")
    
    return True


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

@PipelineDecorator.pipeline(
    name=PIPELINE_NAME,
    project=PROJECT_NAME,
    version="1.0.0"
)
def run_weather_forecast_pipeline(
    data_path: str = "dataset/hn_daily.csv",
    tuning_trials: int = 100,
    deployment_stage: str = "staging",
    skip_tuning: bool = False
):
    """
    Main pipeline execution function leveraging existing modules
    
    Args:
        data_path: Path to input CSV data
        tuning_trials: Number of Optuna trials for stage 2 tuning
        deployment_stage: 'staging' or 'production'
        skip_tuning: Skip hyperparameter tuning
    """
    
    print("=" * 70)
    print(f"🚀 Starting Weather Forecast Pipeline")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Tuning Trials: {tuning_trials}")
    print(f"Deployment: {deployment_stage}")
    print(f"Skip Tuning: {skip_tuning}")
    print("=" * 70 + "\n")
    
    # Step 1: Preprocess data using process.py
    # This creates all per-target datasets with feature engineering
    preprocessing_success = preprocess_data(data_path=data_path)
    
    if not preprocessing_success:
        print("❌ Preprocessing failed!")
        return False
    
    # Step 2: Hyperparameter tuning using tune.py (optional)
    if not skip_tuning:
        tuning_success = hyperparameter_tuning(n_trials=tuning_trials)
        
        if not tuning_success:
            print("⚠️  Tuning failed, continuing with default parameters...")
    else:
        print("⏭️  Skipping hyperparameter tuning")
    
    # Step 3: Train all models using train.py
    training_success = train_all_models()
    
    if not training_success:
        print("❌ Training failed!")
        return False
    
    # Step 4: Evaluate models using evaluate.py
    evaluation_success = evaluate_models()
    
    if not evaluation_success:
        print("❌ Evaluation failed!")
        return False
    
    # Step 5: Deploy to ClearML using deploy.py
    deployment_success = deploy_models(deployment_stage=deployment_stage)
    
    if not deployment_success:
        print("❌ Deployment failed!")
        return False
    
    print("\n" + "=" * 70)
    print("✅ Pipeline Complete!")
    print("=" * 70)
    print(f"Deployment Stage: {deployment_stage}")
    print(f"Results: src/daily_forecast_model/evaluate_results/")
    print(f"Models: trained_models/")
    print(f"View ClearML: https://app.clear.ml")
    print("=" * 70)
    
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run the pipeline using PipelineDecorator
    
    Usage:
        # Local execution (for testing)
        python src/daily_forecast_model/pipeline_decorator.py
        
        # Remote execution on ClearML
        python src/daily_forecast_model/pipeline_decorator.py --remote
        
        # Production deployment
        python src/daily_forecast_model/pipeline_decorator.py --remote --stage production --trials 100
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Weather Forecast Pipeline (Decorator)')
    parser.add_argument(
        '--data-path',
        type=str,
        default='dataset/hn_daily.csv',
        help='Path to input data'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Number of Optuna trials for tuning'
    )
    parser.add_argument(
        '--stage',
        type=str,
        default='staging',
        choices=['staging', 'production'],
        help='Deployment stage'
    )
    parser.add_argument(
        '--skip-tuning',
        action='store_true',
        help='Skip hyperparameter tuning'
    )
    parser.add_argument(
        '--remote',
        action='store_true',
        help='Execute pipeline remotely on ClearML agents'
    )
    
    args = parser.parse_args()
    
    # Set execution mode
    if args.remote:
        print("🌐 Running pipeline remotely on ClearML agents...")
        PipelineDecorator.set_default_execution_queue("default")
    else:
        print("💻 Running pipeline locally...")
        PipelineDecorator.run_locally()
    
    # Execute pipeline
    run_weather_forecast_pipeline(
        data_path=args.data_path,
        tuning_trials=args.trials,
        deployment_stage=args.stage,
        skip_tuning=args.skip_tuning
    )
    
    print("\n✅ Pipeline execution completed!")
    print("View results at: https://app.clear.ml")
