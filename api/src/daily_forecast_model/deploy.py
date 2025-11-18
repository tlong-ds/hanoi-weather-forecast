import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import joblib

try:
    from clearml import Task, Model, Logger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("⚠️  ClearML not installed. Install with: pip install clearml")

from src.daily_forecast_model.helper import PROJECT_ROOT, MODELS_DIR


class ClearMLModelDeployer:
    """
    Handles deployment of weather forecasting models to ClearML
    
    Features:
    - Model registration with metadata
    - Performance tracking
    - Version management
    - Staging and production deployment
    """
    
    def __init__(
        self,
        project_name: str = "Hanoi Weather Forecast",
        task_name: str = "Daily Temperature Forecast - Deployment"
    ):
        """
        Initialize ClearML deployer
        
        Args:
            project_name: ClearML project name
            task_name: Task name for this deployment
        """
        if not CLEARML_AVAILABLE:
            raise ImportError("ClearML is not installed. Install with: pip install clearml")
        
        self.project_name = project_name
        self.task_name = task_name
        self.task = None
        self.logger = None
        
    def initialize_task(self, task_type: str = "inference"):
        """
        Initialize ClearML task
        
        Args:
            task_type: Type of task (training, inference, etc.)
        """
        self.task = Task.init(
            project_name=self.project_name,
            task_name=self.task_name,
            task_type=task_type
        )
        self.logger = self.task.get_logger()
        print(f"✅ ClearML task initialized: {self.task.name}")
        
    def register_model(
        self,
        model_path: str,
        model_name: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        comment: Optional[str] = None
    ) -> Model:
        """
        Register a trained model to ClearML Model Registry
        
        Args:
            model_path: Path to the .joblib model file
            model_name: Name for the model in registry
            tags: List of tags (e.g., ['t+1', 'xgboost', 'production'])
            metadata: Additional metadata dictionary
            comment: Description of the model
            
        Returns:
            ClearML Model object
        """
        if not self.task:
            self.initialize_task()
        
        # Create OutputModel from task
        from clearml import OutputModel
        
        output_model = OutputModel(
            task=self.task,
            name=model_name,
            tags=tags or [],
            comment=comment or f"Weather forecast model: {model_name}"
        )
        
        # Upload the model weights
        output_model.update_weights(
            weights_filename=model_path,
            auto_delete_file=False
        )
        
        # Add metadata as configuration
        if metadata:
            output_model.update_design(config_dict=metadata)
        
        print(f"✅ Model registered: {model_name}")
        print(f"   Model ID: {output_model.id}")
        print(f"   Tags: {tags}")
        
        return output_model
    
    def deploy_all_models(
        self,
        evaluation_results_path: Optional[str] = None,
        stage: str = "staging"
    ):
        """
        Deploy all 5 trained models (t+1 to t+5) to ClearML
        
        Args:
            evaluation_results_path: Path to evaluation_results.json
            stage: Deployment stage ('staging' or 'production')
        """
        if not self.task:
            self.initialize_task()
        
        # Load evaluation metrics
        if evaluation_results_path is None:
            evaluation_results_path = os.path.join(
                PROJECT_ROOT, 
                'src/daily_forecast_model/evaluate_results/evaluation_results.json'
            )
        
        with open(evaluation_results_path, 'r') as f:
            eval_metrics = json.load(f)
        
        deployed_models = []
        
        for target in ['t+1', 't+2', 't+3', 't+4', 't+5']:
            print(f"\n{'='*60}")
            print(f"Deploying model: {target}")
            print(f"{'='*60}")
            
            # Model path
            model_path = os.path.join(MODELS_DIR, f"model_{target}.joblib")
            
            if not os.path.exists(model_path):
                print(f"⚠️  Model file not found: {model_path}")
                continue
            
            # Load model to get type
            model_obj = joblib.load(model_path)
            model_type = type(model_obj).__name__
            
            # Get metrics for this target
            metrics = eval_metrics[target]['metrics']
            n_samples = eval_metrics[target]['n_samples']
            
            # Prepare metadata
            metadata = {
                'target': target,
                'model_type': model_type,
                'framework': 'scikit-learn',
                'performance': {
                    'test_rmse': round(metrics['RMSE'], 3),
                    'test_mae': round(metrics['MAE'], 3),
                    'test_r2': round(metrics['R2'], 3),
                    'test_mape': round(metrics['MAPE'], 3)
                },
                'test_samples': n_samples,
                'forecast_horizon': target,
                'deployment_stage': stage,
                'features': 'Per-target feature selection',
                'preprocessing': 'StandardScaler + feature engineering'
            }
            
            # Register model
            model_name = f"hanoi_weather_{target}"
            tags = [target, model_type.lower(), stage, 'temperature_forecast']
            
            model = self.register_model(
                model_path=model_path,
                model_name=model_name,
                tags=tags,
                metadata=metadata,
                comment=f"Temperature forecast model for {target} (RMSE: {metrics['RMSE']:.2f}°C)"
            )
            
            # Log metrics to ClearML
            if self.logger:
                self.logger.report_single_value(f"{target}_RMSE", metrics['RMSE'])
                self.logger.report_single_value(f"{target}_MAE", metrics['MAE'])
                self.logger.report_single_value(f"{target}_R2", metrics['R2'])
                self.logger.report_single_value(f"{target}_MAPE", metrics['MAPE'])
            
            deployed_models.append({
                'target': target,
                'model_id': model.id,
                'model_name': model_name,
                'rmse': metrics['RMSE']
            })
        
        print(f"\n{'='*60}")
        print("✅ Deployment Summary")
        print(f"{'='*60}")
        print(f"Deployed {len(deployed_models)} models to ClearML")
        print(f"Stage: {stage}")
        print(f"Project: {self.project_name}")
        
        for m in deployed_models:
            print(f"  • {m['target']}: {m['model_name']} (RMSE: {m['rmse']:.2f}°C)")
        
        return deployed_models
    
    def deploy_preprocessors(
        self,
        preprocessors_dir: Optional[str] = None
    ):
        """
        Deploy preprocessing pipelines to ClearML
        
        Args:
            preprocessors_dir: Directory containing preprocessor files
        """
        if not self.task:
            self.initialize_task()
        
        if preprocessors_dir is None:
            preprocessors_dir = os.path.join(
                PROJECT_ROOT,
                'processed_data/pipelines'
            )
        
        deployed_preprocessors = []
        
        for target in ['t+1', 't+2', 't+3', 't+4', 't+5']:
            target_name = f"t_{target.split('+')[1]}"
            preprocessor_path = os.path.join(preprocessors_dir, f"preprocessor_{target_name}.joblib")
            
            if not os.path.exists(preprocessor_path):
                print(f"⚠️  Preprocessor not found: {preprocessor_path}")
                continue
            
            model_name = f"preprocessor_{target}"
            metadata = {
                'type': 'preprocessing_pipeline',
                'target': target,
                'components': [
                    'temporal_features',
                    'day_length',
                    'cyclical_wind_direction',
                    'interaction_features',
                    'lag_features',
                    'rolling_windows',
                    'standard_scaler'
                ]
            }
            
            model = self.register_model(
                model_path=preprocessor_path,
                model_name=model_name,
                tags=[target, 'preprocessor', 'pipeline'],
                metadata=metadata,
                comment=f"Preprocessing pipeline for {target}"
            )
            
            deployed_preprocessors.append({
                'target': target,
                'model_id': model.id,
                'model_name': model_name
            })
        
        print(f"\n✅ Deployed {len(deployed_preprocessors)} preprocessing pipelines")
        return deployed_preprocessors
    
    def promote_to_production(self, model_id: str):
        """
        Promote a model from staging to production
        
        Args:
            model_id: ClearML model ID
        """
        model = Model(model_id=model_id)
        
        # Update tags
        current_tags = model.labels.get('tags', [])
        if 'staging' in current_tags:
            current_tags.remove('staging')
        if 'production' not in current_tags:
            current_tags.append('production')
        
        model.labels['tags'] = current_tags
        model.update_labels(model.labels)
        
        print(f"✅ Model {model_id} promoted to production")
        
    def get_production_models(self) -> List[Model]:
        """
        Get all models tagged as 'production'
        
        Returns:
            List of production models
        """
        models = Model.query_models(
            project_name=self.project_name,
            tags=['production']
        )
        return models


def deploy_models_to_clearml(stage: str = "staging"):
    """
    Convenience function to deploy all models to ClearML
    
    Args:
        stage: Deployment stage ('staging' or 'production')
    
    Usage:
        from src.daily_forecast_model.deploy_clearml import deploy_models_to_clearml
        deploy_models_to_clearml(stage='staging')
    """
    deployer = ClearMLModelDeployer()
    
    # Deploy models
    models = deployer.deploy_all_models(stage=stage)
    
    # Deploy preprocessors
    preprocessors = deployer.deploy_preprocessors()
    
    print("\n" + "="*60)
    print("🎉 Deployment Complete!")
    print("="*60)
    print(f"Models deployed: {len(models)}")
    print(f"Preprocessors deployed: {len(preprocessors)}")
    print(f"Stage: {stage}")
    print("\nView in ClearML dashboard:")
    print("https://app.clear.ml")
    
    return deployer, models, preprocessors


if __name__ == "__main__":
    """
    Run this script to deploy all models to ClearML
    
    Usage:
        python src/daily_forecast_model/deploy_clearml.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy weather forecast models to ClearML')
    parser.add_argument(
        '--stage',
        type=str,
        default='staging',
        choices=['staging', 'production'],
        help='Deployment stage (default: staging)'
    )
    parser.add_argument(
        '--models-only',
        action='store_true',
        help='Deploy only models, skip preprocessors'
    )
    parser.add_argument(
        '--preprocessors-only',
        action='store_true',
        help='Deploy only preprocessors, skip models'
    )
    
    args = parser.parse_args()
    
    deployer = ClearMLModelDeployer()
    
    if not args.preprocessors_only:
        print("Deploying models...")
        models = deployer.deploy_all_models(stage=args.stage)
    
    if not args.models_only:
        print("\nDeploying preprocessors...")
        preprocessors = deployer.deploy_preprocessors()
    
    print("\n✅ All done! Check ClearML dashboard for deployed artifacts.")
