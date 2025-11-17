import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import joblib

try:
    from clearml import Task, Model, Logger, OutputModel
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("⚠️  ClearML not installed. Install with: pip install clearml")

from src.hourly_forecast_model.helper import PROJECT_ROOT, MODELS_DIR, N_STEPS_AHEAD


class ClearMLHourlyModelDeployer:
    """
    Handles deployment of hourly weather forecasting model to ClearML
    
    Features:
    - Multi-output model registration with metadata
    - Performance tracking across all 24 hours
    - Version management
    - Staging and production deployment
    """
    
    def __init__(
        self,
        project_name: str = "Hanoi Weather Forecast - Hourly",
        task_name: str = "Hourly Temperature Forecast - Deployment"
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
    ) -> OutputModel:
        """
        Register the multi-output model to ClearML Model Registry
        
        Args:
            model_path: Path to the .joblib model file
            model_name: Name for the model in registry
            tags: List of tags (e.g., ['multi-output', 'xgboost', 'production'])
            metadata: Additional metadata dictionary
            comment: Description of the model
            
        Returns:
            ClearML OutputModel object
        """
        if not self.task:
            self.initialize_task()
        
        # Create OutputModel from task
        output_model = OutputModel(
            task=self.task,
            name=model_name,
            tags=tags or [],
            comment=comment or f"Hourly weather forecast multi-output model: {model_name}"
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
    
    def deploy_model(
        self,
        evaluation_results_path: Optional[str] = None,
        stage: str = "staging"
    ):
        """
        Deploy multi-output model to ClearML
        
        Args:
            evaluation_results_path: Path to evaluation_results.json
            stage: Deployment stage ('staging' or 'production')
        """
        if not self.task:
            self.initialize_task()
        
        print(f"\n{'='*70}")
        print(f"Deploying Hourly Forecast Multi-Output Model")
        print(f"{'='*70}\n")
        
        # Load evaluation metrics
        if evaluation_results_path is None:
            evaluation_results_path = os.path.join(
                PROJECT_ROOT, 
                'src/hourly_forecast_model/evaluate_results/evaluation_results.json'
            )
        
        try:
            with open(evaluation_results_path, 'r') as f:
                eval_metrics = json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Evaluation results not found: {evaluation_results_path}")
            print("   Proceeding without evaluation metrics")
            eval_metrics = {}
        
        # Load training metadata
        metadata_path = os.path.join(MODELS_DIR, 'training_metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                training_metadata = json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Training metadata not found: {metadata_path}")
            training_metadata = {}
        
        # Model path
        model_path = os.path.join(MODELS_DIR, 'model_multioutput_24h.joblib')
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("   Please train the model first:")
            print("   python -m src.hourly_forecast_model.train")
            return None
        
        # Prepare metadata
        model_metadata = {
            'model_type': training_metadata.get('model_type', 'Unknown'),
            'n_targets': N_STEPS_AHEAD,
            'n_features': training_metadata.get('n_features', 'Unknown'),
            'n_samples': training_metadata.get('n_samples', 'Unknown'),
            'training_time': training_metadata.get('training_time', 'Unknown'),
            'tuned_rmse': training_metadata.get('tuned_rmse', 'Unknown'),
            'average_metrics': eval_metrics.get('average_metrics', {}),
            'deployment_stage': stage,
            'framework': 'scikit-learn MultiOutputRegressor'
        }
        
        # Prepare tags
        tags = [
            f"multi-output-24h",
            f"model_{training_metadata.get('model_type', 'unknown').lower()}",
            stage,
            "temperature_forecast",
            "hourly"
        ]
        
        # Create comment with performance summary
        avg_metrics = eval_metrics.get('average_metrics', {})
        comment = (
            f"Multi-output hourly temperature forecast model (24 hours)\n"
            f"Model: {training_metadata.get('model_type', 'Unknown')}\n"
            f"Average RMSE: {avg_metrics.get('RMSE', 'N/A')}°C\n"
            f"Average MAE: {avg_metrics.get('MAE', 'N/A')}°C\n"
            f"Average R²: {avg_metrics.get('R2', 'N/A')}\n"
            f"Deployment stage: {stage}"
        )
        
        # Register model
        model_name = f"hourly_forecast_multioutput_{stage}"
        output_model = self.register_model(
            model_path=model_path,
            model_name=model_name,
            tags=tags,
            metadata=model_metadata,
            comment=comment
        )
        
        # Log performance metrics to ClearML
        if self.logger and avg_metrics:
            print("\nLogging metrics to ClearML...")
            self.logger.report_single_value("Average RMSE", avg_metrics.get('RMSE', 0))
            self.logger.report_single_value("Average MAE", avg_metrics.get('MAE', 0))
            self.logger.report_single_value("Average MAPE", avg_metrics.get('MAPE', 0))
            self.logger.report_single_value("Average R2", avg_metrics.get('R2', 0))
            
            # Log per-hour metrics if available
            per_hour_metrics = eval_metrics.get('per_hour_metrics', {})
            if per_hour_metrics:
                hours = []
                rmse_values = []
                mae_values = []
                
                for hour_name, metrics in per_hour_metrics.items():
                    hours.append(metrics['hour'])
                    rmse_values.append(metrics['metrics']['RMSE'])
                    mae_values.append(metrics['metrics']['MAE'])
                
                # Create series plots
                for hour, rmse, mae in zip(hours, rmse_values, mae_values):
                    self.logger.report_scalar(
                        title="Per-Hour RMSE",
                        series="RMSE",
                        value=rmse,
                        iteration=hour
                    )
                    self.logger.report_scalar(
                        title="Per-Hour MAE",
                        series="MAE",
                        value=mae,
                        iteration=hour
                    )
            
            print("✅ Metrics logged to ClearML")
        
        print(f"\n{'='*70}")
        print("✅ DEPLOYMENT COMPLETE")
        print(f"{'='*70}")
        print(f"Model Name: {model_name}")
        print(f"Model ID: {output_model.id}")
        print(f"Stage: {stage}")
        print(f"Registry: ClearML Model Registry")
        print(f"{'='*70}\n")
        
        return output_model
    
    def promote_to_production(self, model_id: str):
        """
        Promote a staging model to production
        
        Args:
            model_id: ClearML model ID to promote
        """
        if not self.task:
            self.initialize_task()
        
        print(f"\n{'='*70}")
        print(f"Promoting model to production: {model_id}")
        print(f"{'='*70}\n")
        
        # Load model from registry
        model = Model(model_id=model_id)
        
        # Update tags
        current_tags = model.tags or []
        if 'staging' in current_tags:
            current_tags.remove('staging')
        if 'production' not in current_tags:
            current_tags.append('production')
        
        model.tags = current_tags
        
        print(f"✅ Model promoted to production")
        print(f"   Model ID: {model_id}")
        print(f"   Tags: {current_tags}")
        print(f"{'='*70}\n")


def main():
    """Main deployment function."""
    print("\n" + "="*70)
    print("HOURLY FORECAST MODEL DEPLOYMENT")
    print("="*70 + "\n")
    
    if not CLEARML_AVAILABLE:
        print("❌ ClearML is not installed.")
        print("   Install with: pip install clearml")
        return
    
    # Initialize deployer
    deployer = ClearMLHourlyModelDeployer()
    
    # Deploy to staging
    model = deployer.deploy_model(stage='staging')
    
    if model:
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("1. Test the model in staging environment")
        print("2. If satisfied, promote to production:")
        print(f"   deployer.promote_to_production('{model.id}')")
        print("3. Monitor model performance in ClearML")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
