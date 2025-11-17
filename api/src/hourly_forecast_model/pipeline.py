from clearml import PipelineDecorator, Task
from pathlib import Path
import joblib
import pandas as pd
from typing import Dict, Any
import json

# Project configuration
PROJECT_NAME = "Hanoi Weather Forecast - Hourly"
PIPELINE_NAME = "Hourly Forecast - Multi-Output Pipeline"


# ============================================================================
# PIPELINE COMPONENTS (Decorated Functions)
# ============================================================================

@PipelineDecorator.component(
    return_values=['success'],
    cache=True,
    execution_queue="default"
)
def preprocess_data() -> bool:
    """
    Step 1: Preprocess raw hourly weather data
    
    This includes:
    - Feature engineering (hourly lags, rolling windows, cyclical encoding)
    - Data cleaning and outlier removal
    - Train/dev/test splitting
    - Feature selection
    - Transformation pipeline fitting
    
    Returns:
        bool: Success status
    """
    from src.hourly_forecast_model.process import main as process_main
    
    print(f"📊 Running hourly data preprocessing pipeline...")
    
    # Run the complete preprocessing pipeline
    process_main()
    
    print(f"✅ Preprocessing complete - data saved to data_processing_hourly/")
    
    return True


@PipelineDecorator.component(
    return_values=['tuning_complete'],
    cache=True,
    execution_queue="default"
)
def hyperparameter_tuning(n_trials: int = 100) -> bool:
    """
    Step 2: Hyperparameter tuning for multi-output model
    
    Tunes all model architectures:
    - RandomForest
    - XGBoost
    - LightGBM
    - CatBoost
    
    Selects best model based on average RMSE across all 24 hours.
    
    Args:
        n_trials: Number of Optuna trials per model
    
    Returns:
        bool: Success status
    """
    from src.hourly_forecast_model.tune import main as tune_main
    
    print(f"🎯 Running hyperparameter tuning for multi-output model...")
    print(f"  Testing 4 architectures with {n_trials} trials each")
    
    # Run the complete tuning pipeline
    tune_main()
    
    print(f"✅ Tuning complete - best model config saved")
    
    return True


@PipelineDecorator.component(
    return_values=['training_complete'],
    cache=True,
    execution_queue="default"
)
def train_multioutput_model() -> bool:
    """
    Step 3: Train multi-output model for 24-hour forecast
    
    This:
    - Loads best hyperparameters from tuning
    - Trains single multi-output model (24 estimators in parallel)
    - Uses combined train+dev data
    - Saves model to trained_models_hourly/
    
    Returns:
        bool: Success status
    """
    from src.hourly_forecast_model.train import train_multioutput_model
    
    print(f"🚂 Training multi-output model for 24-hour forecast...")
    
    # Run the training pipeline
    model = train_multioutput_model()
    
    if model is None:
        print("❌ Training failed")
        return False
    
    print(f"✅ Multi-output model trained successfully")
    
    return True


@PipelineDecorator.component(
    return_values=['evaluation_complete'],
    cache=True,
    execution_queue="default"
)
def evaluate_multioutput_model() -> bool:
    """
    Step 4: Evaluate multi-output model on test set
    
    This:
    - Loads trained multi-output model
    - Evaluates on test set for all 24 hours
    - Calculates per-hour metrics (MAE, RMSE, MAPE, R²)
    - Generates visualizations
    - Saves comprehensive evaluation report
    
    Returns:
        bool: Success status
    """
    from src.hourly_forecast_model.evaluate import main as evaluate_main
    
    print(f"📈 Evaluating multi-output model on test set...")
    
    # Run the evaluation pipeline
    evaluate_main()
    
    print(f"✅ Evaluation complete with visualizations")
    
    return True


@PipelineDecorator.component(
    return_values=['deployment_complete'],
    cache=False,  # Don't cache deployment
    execution_queue="default"
)
def deploy_model(deployment_stage: str = "staging") -> bool:
    """
    Step 5: Deploy multi-output model to ClearML
    
    This:
    - Registers model to ClearML Model Registry
    - Uploads model file and metadata
    - Tags model with deployment stage
    - Tracks performance metrics
    - Enables model versioning
    
    Args:
        deployment_stage: 'staging' or 'production'
    
    Returns:
        bool: Success status
    """
    from src.hourly_forecast_model.deploy import ClearMLHourlyModelDeployer
    
    print(f"🚀 Deploying multi-output model to ClearML ({deployment_stage})...")
    
    # Initialize deployer
    deployer = ClearMLHourlyModelDeployer(
        project_name=PROJECT_NAME,
        task_name=f"Model Deployment - {deployment_stage}"
    )
    
    # Initialize ClearML task
    deployer.initialize_task()
    
    # Deploy model
    model = deployer.deploy_model(stage=deployment_stage)
    
    if model is None:
        print("❌ Deployment failed")
        return False
    
    print(f"✅ Multi-output model deployed to {deployment_stage}")
    
    return True


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

@PipelineDecorator.pipeline(
    name=PIPELINE_NAME,
    project=PROJECT_NAME,
    version="1.0"
)
def run_hourly_forecast_pipeline(
    skip_preprocessing: bool = False,
    skip_tuning: bool = False,
    n_tuning_trials: int = 100,
    deployment_stage: str = "staging"
):
    """
    Complete hourly weather forecast pipeline
    
    Pipeline steps:
    1. Preprocess hourly data (feature engineering, splitting)
    2. Tune multi-output model (find best architecture + hyperparameters)
    3. Train final multi-output model (on train+dev)
    4. Evaluate on test set (metrics + visualizations)
    5. Deploy to ClearML Model Registry
    
    Args:
        skip_preprocessing: Skip data preprocessing (use existing data)
        skip_tuning: Skip hyperparameter tuning (use existing config)
        n_tuning_trials: Number of Optuna trials per model architecture
        deployment_stage: 'staging' or 'production'
    """
    
    print(f"\n{'='*70}")
    print(f"HOURLY WEATHER FORECAST PIPELINE")
    print(f"{'='*70}")
    print(f"Project: {PROJECT_NAME}")
    print(f"Pipeline: {PIPELINE_NAME}")
    print(f"{'='*70}\n")
    
    # Step 1: Data Preprocessing
    if not skip_preprocessing:
        print("\n[STEP 1/5] Data Preprocessing")
        print("-" * 70)
        preprocessing_success = preprocess_data()
        
        if not preprocessing_success:
            print("❌ Pipeline failed at preprocessing step")
            return
    else:
        print("\n[STEP 1/5] Skipping preprocessing (using existing data)")
    
    # Step 2: Hyperparameter Tuning
    if not skip_tuning:
        print("\n[STEP 2/5] Hyperparameter Tuning")
        print("-" * 70)
        tuning_success = hyperparameter_tuning(n_trials=n_tuning_trials)
        
        if not tuning_success:
            print("❌ Pipeline failed at tuning step")
            return
    else:
        print("\n[STEP 2/5] Skipping tuning (using existing config)")
    
    # Step 3: Model Training
    print("\n[STEP 3/5] Model Training")
    print("-" * 70)
    training_success = train_multioutput_model()
    
    if not training_success:
        print("❌ Pipeline failed at training step")
        return
    
    # Step 4: Model Evaluation
    print("\n[STEP 4/5] Model Evaluation")
    print("-" * 70)
    evaluation_success = evaluate_multioutput_model()
    
    if not evaluation_success:
        print("❌ Pipeline failed at evaluation step")
        return
    
    # Step 5: Model Deployment
    print("\n[STEP 5/5] Model Deployment")
    print("-" * 70)
    deployment_success = deploy_model(deployment_stage=deployment_stage)
    
    if not deployment_success:
        print("❌ Pipeline failed at deployment step")
        return
    
    # Pipeline complete
    print(f"\n{'='*70}")
    print("✅ PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print("\nPipeline Summary:")
    print(f"  ✓ Data preprocessed")
    print(f"  ✓ Multi-output model tuned")
    print(f"  ✓ Model trained on train+dev")
    print(f"  ✓ Model evaluated on test set")
    print(f"  ✓ Model deployed to {deployment_stage}")
    print(f"\nNext Steps:")
    print(f"  1. Review evaluation metrics in ClearML")
    print(f"  2. Test deployed model in {deployment_stage}")
    if deployment_stage == "staging":
        print(f"  3. Promote to production when ready")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run hourly weather forecast pipeline")
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip data preprocessing step (use existing data)'
    )
    parser.add_argument(
        '--skip-tuning',
        action='store_true',
        help='Skip hyperparameter tuning (use existing config)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of Optuna trials per model architecture (default: 100)'
    )
    parser.add_argument(
        '--stage',
        choices=['staging', 'production'],
        default='staging',
        help='Deployment stage (default: staging)'
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='Run pipeline locally instead of on ClearML agents'
    )
    
    args = parser.parse_args()
    
    # Configure pipeline for local or remote execution
    if args.local:
        print("\n🏠 Running pipeline LOCALLY")
        print("   All steps will execute in this Python process\n")
        
        # Start the pipeline locally
        PipelineDecorator.run_locally()
    else:
        print("\n☁️  Running pipeline on CLEARML AGENTS")
        print("   Steps will be distributed to available agents\n")
    
    # Execute the pipeline
    run_hourly_forecast_pipeline(
        skip_preprocessing=args.skip_preprocessing,
        skip_tuning=args.skip_tuning,
        n_tuning_trials=args.n_trials,
        deployment_stage=args.stage
    )
    
    print("\n✅ Pipeline submitted successfully!")
    print("   Monitor progress in ClearML Web UI")
