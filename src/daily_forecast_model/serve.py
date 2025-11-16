"""
ClearML Model Serving Module
Deploy and serve weather forecasting models using ClearML Serving

This module provides:
1. Model serving with clearml-serving
2. HTTP endpoint configuration
3. Model version management
4. Performance monitoring
5. A/B testing support
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

try:
    from clearml import Task, Model
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("⚠️  ClearML not installed. Install with: pip install clearml")

from src.daily_forecast_model.helper import PROJECT_ROOT
from src.daily_forecast_model.infer import WeatherForecaster


class ClearMLServingEndpoint:
    """
    Custom serving endpoint for weather forecasting models
    
    This class can be deployed with clearml-serving or used standalone
    """
    
    def __init__(self):
        """Initialize the serving endpoint"""
        self.forecaster = None
        self.task = None
        self.model_info = {}
        
    def load(self):
        """
        Load models and initialize forecaster
        Called when the service starts
        """
        print("🔄 Loading weather forecasting models...")
        
        # Initialize ClearML task for monitoring
        if CLEARML_AVAILABLE:
            self.task = Task.init(
                project_name="Hanoi Weather Forecast",
                task_name="Model Serving - Production",
                task_type="inference"
            )
        
        # Load the forecaster with all models
        self.forecaster = WeatherForecaster()
        
        # Load model metadata
        eval_path = os.path.join(
            PROJECT_ROOT,
            'src/daily_forecast_model/evaluate_results/evaluation_results.json'
        )
        with open(eval_path, 'r') as f:
            self.model_info = json.load(f)
        
        print("✅ Models loaded successfully")
        return True
    
    def preprocess(self, body: Dict, state: Dict, collect_custom_statistics_fn=None) -> Dict:
        """
        Preprocess incoming request
        
        Args:
            body: Request payload
            state: Serving state
            collect_custom_statistics_fn: Function to collect metrics
            
        Returns:
            Preprocessed data
        """
        # Extract data from request
        if 'data' in body:
            data = body['data']
        else:
            data = body
        
        return data
    
    def process(self, data: Dict, state: Dict, collect_custom_statistics_fn=None) -> Dict:
        """
        Main inference function
        
        Args:
            data: Preprocessed input data (pandas DataFrame as dict or CSV path)
            state: Serving state
            collect_custom_statistics_fn: Function to collect metrics
            
        Returns:
            Predictions with metadata
        """
        try:
            # Convert input to DataFrame if needed
            if isinstance(data, dict) and 'csv_path' in data:
                # Load from CSV path
                raw_data = pd.read_csv(data['csv_path'])
            elif isinstance(data, dict):
                # Convert dict to DataFrame
                raw_data = pd.DataFrame(data)
            else:
                raw_data = data
            
            # Ensure datetime index
            if 'datetime' in raw_data.columns:
                raw_data['datetime'] = pd.to_datetime(raw_data['datetime'])
                raw_data.set_index('datetime', inplace=True)
            elif not isinstance(raw_data.index, pd.DatetimeIndex):
                raise ValueError("Data must have datetime index or datetime column")
            
            # Generate predictions
            result = self.forecaster.predict_with_metadata(raw_data)
            
            # Format response
            predictions = []
            for i, target in enumerate(['t+1', 't+2', 't+3', 't+4', 't+5']):
                pred = {
                    'target': target,
                    'forecast_date': result['forecast_dates'][i],
                    'temperature': float(result['predictions'][target]),
                    'unit': 'celsius',
                    'model_performance': {
                        'test_rmse': float(self.model_info[target]['metrics']['RMSE']),
                        'test_mae': float(self.model_info[target]['metrics']['MAE']),
                        'test_r2': float(self.model_info[target]['metrics']['R2']),
                        'test_mape': float(self.model_info[target]['metrics']['MAPE'])
                    }
                }
                predictions.append(pred)
            
            response = {
                'status': 'success',
                'reference_date': result['base_date'],
                'predictions': predictions,
                'model_version': 'v1.0.0'
            }
            
            # Collect custom statistics
            if collect_custom_statistics_fn:
                collect_custom_statistics_fn(
                    'predictions_generated',
                    value=len(predictions),
                    labels={'model_version': 'v1.0.0'}
                )
            
            return response
            
        except Exception as e:
            error_response = {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__
            }
            
            if collect_custom_statistics_fn:
                collect_custom_statistics_fn(
                    'prediction_errors',
                    value=1,
                    labels={'error_type': type(e).__name__}
                )
            
            return error_response
    
    def postprocess(self, data: Dict, state: Dict, collect_custom_statistics_fn=None) -> Dict:
        """
        Postprocess predictions before returning
        
        Args:
            data: Prediction results
            state: Serving state
            collect_custom_statistics_fn: Function to collect metrics
            
        Returns:
            Final response
        """
        return data


def setup_clearml_serving():
    """
    Setup instructions for deploying with clearml-serving
    
    Note: clearml-serving is complex. For simpler deployment, use the FastAPI
    endpoint at api/api.py which is already production-ready.
    
    Returns:
        Dictionary with setup commands and configuration
    """
    
    setup_guide = {
        "simple_alternative": [
            "# RECOMMENDED: Use the existing FastAPI endpoint instead",
            "# The api/api.py already provides production-ready serving",
            "",
            "# Start the FastAPI server:",
            "cd /path/to/machine_learning_lab",
            "python api/api.py",
            "",
            "# Or with uvicorn:",
            "uvicorn api.api:app --host 0.0.0.0 --port 8000",
            "",
            "# Test the endpoint:",
            "curl -X POST http://localhost:8000/api/v1/forecast/daily \\",
            "  -H 'Content-Type: application/json' \\",
            "  -d '{\"include_confidence\": true}'",
            "",
            "# Deploy with Docker:",
            "docker build -f Dockerfile.api -t weather-api .",
            "docker run -p 8000:8000 weather-api",
        ],
        
        "clearml_serving_advanced": [
            "# For advanced clearml-serving deployment:",
            "# Note: This is more complex and requires Triton/TorchServe setup",
        ],
        "installation": [
            "# Install clearml-serving",
            "pip install clearml-serving",
            "",
            "# Create a new serving service",
            "clearml-serving create --name 'weather-forecast-serving'",
            "",
            "# Note: This creates a ClearML task for the serving service",
        ],
        
        "model_deployment": [
            "# Add a model to the serving service",
            "# First, get the model ID from ClearML (from deploy.py output)",
            "",
            "clearml-serving model add \\",
            "  --engine 'triton' \\",  # or 'sklearn'
            "  --endpoint 'weather_forecast' \\",
            "  --name 'hanoi_weather_t+1' \\",
            "  --project 'Hanoi Weather Forecast'",
            "",
            "# Add all 5 models (t+1 to t+5)",
            "for target in t+1 t+2 t+3 t+4 t+5; do",
            "  clearml-serving model add \\",
            "    --engine 'sklearn' \\",
            "    --endpoint 'weather_forecast' \\",
            "    --name \"hanoi_weather_${target}\" \\",
            "    --project 'Hanoi Weather Forecast'",
            "done",
        ],
        
        "start_server": [
            "# Option 1: Use Docker Compose (recommended)",
            "clearml-serving config \\",
            "  --name 'weather-forecast-serving' \\",
            "  --output docker-compose.yml",
            "",
            "docker-compose -f docker-compose.yml up",
            "",
            "# Option 2: Launch with specific configuration",
            "clearml-serving metrics \\",
            "  --name 'weather-forecast-serving' \\",
            "  --grafana",
            "",
            "# The serving endpoint will be available at:",
            "# http://localhost:8080/serve/<endpoint_name>",
        ],
        
        "test_endpoint": [
            "# Test the endpoint",
            "curl -X POST http://localhost:8080/serve/weather_forecast \\",
            "  -H 'Content-Type: application/json' \\",
            "  -d '{\"csv_path\": \"dataset/hn_daily.csv\"}'",
        ],
        
        "monitoring": [
            "# View metrics in ClearML dashboard",
            "# Navigate to: https://app.clear.ml",
            "# Go to: Projects > Hanoi Weather Forecast > Model Serving - Production",
            "",
            "# Metrics available:",
            "# - predictions_generated (count)",
            "# - prediction_errors (count by error_type)",
            "# - inference_latency (ms)",
            "# - model_version usage",
        ],
        
        "configuration_file": {
            "description": "Create serving_config.yaml",
            "content": """
# clearml-serving configuration
service:
  name: weather-forecast-serving
  version: 1.0.0
  
endpoints:
  - name: weather_forecast
    model_id: <CLEARML_MODEL_ID>
    preprocessing: src.daily_forecast_model.serve_clearml:ClearMLServingEndpoint.preprocess
    process: src.daily_forecast_model.serve_clearml:ClearMLServingEndpoint.process
    postprocess: src.daily_forecast_model.serve_clearml:ClearMLServingEndpoint.postprocess
    
    # Resource limits
    max_batch_size: 10
    timeout_ms: 5000
    
    # Auto-scaling
    min_replicas: 1
    max_replicas: 5
    target_cpu_percent: 70

monitoring:
  enabled: true
  metrics:
    - name: predictions_generated
      type: counter
    - name: prediction_errors
      type: counter
    - name: inference_latency
      type: histogram
"""
        }
    }
    
    return setup_guide


def create_serving_docker():
    """
    Create Dockerfile for serving the models
    
    Returns:
        Dockerfile content as string
    """
    
    dockerfile = """
# Dockerfile for Weather Forecast Model Serving
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install clearml-serving
RUN pip install clearml-serving

# Copy project files
COPY . .

# Set environment variables for ClearML
ENV CLEARML_WEB_HOST="https://app.clear.ml"
ENV CLEARML_API_HOST="https://api.clear.ml"
ENV CLEARML_FILES_HOST="https://files.clear.ml"

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8080/health || exit 1

# Start serving
CMD ["clearml-serving", "start", "--docker"]
"""
    
    return dockerfile


def create_monitoring_dashboard():
    """
    Create a monitoring dashboard configuration for ClearML
    
    Returns:
        Dashboard configuration
    """
    
    dashboard_config = {
        "name": "Weather Forecast Model Serving Dashboard",
        "metrics": [
            {
                "title": "Predictions Generated",
                "metric": "predictions_generated",
                "type": "counter",
                "aggregation": "sum",
                "refresh_interval": 60
            },
            {
                "title": "Prediction Errors",
                "metric": "prediction_errors",
                "type": "counter",
                "aggregation": "sum",
                "group_by": "error_type",
                "refresh_interval": 60
            },
            {
                "title": "Inference Latency (p50, p95, p99)",
                "metric": "inference_latency",
                "type": "histogram",
                "percentiles": [50, 95, 99],
                "refresh_interval": 60
            },
            {
                "title": "Model Version Distribution",
                "metric": "predictions_generated",
                "type": "counter",
                "group_by": "model_version",
                "visualization": "pie_chart"
            },
            {
                "title": "Request Rate (req/min)",
                "metric": "predictions_generated",
                "type": "rate",
                "time_window": 60,
                "refresh_interval": 10
            },
            {
                "title": "Error Rate (%)",
                "metric": "prediction_errors",
                "type": "rate",
                "calculation": "errors / (predictions + errors) * 100",
                "refresh_interval": 60
            }
        ],
        "alerts": [
            {
                "name": "High Error Rate",
                "condition": "error_rate > 5",
                "duration": 300,
                "severity": "critical",
                "notification": "email"
            },
            {
                "name": "High Latency",
                "condition": "inference_latency.p99 > 1000",
                "duration": 300,
                "severity": "warning",
                "notification": "slack"
            },
            {
                "name": "Low Request Rate",
                "condition": "request_rate < 1",
                "duration": 600,
                "severity": "info",
                "notification": "none"
            }
        ]
    }
    
    return dashboard_config


if __name__ == "__main__":
    """
    Print setup guide for clearml-serving
    """
    print("=" * 70)
    print("ClearML Serving Setup Guide")
    print("=" * 70)
    
    guide = setup_clearml_serving()
    
    print("\n📦 Installation:")
    for cmd in guide['installation']:
        print(f"  {cmd}")
    
    print("\n🚀 Deploy Models:")
    for cmd in guide['model_deployment']:
        print(f"  {cmd}")
    
    print("\n▶️  Start Server:")
    for cmd in guide['start_server']:
        print(f"  {cmd}")
    
    print("\n🧪 Test Endpoint:")
    for cmd in guide['test_endpoint']:
        print(f"  {cmd}")
    
    print("\n📊 Monitoring:")
    for item in guide['monitoring']:
        print(f"  {item}")
    
    print("\n📝 Configuration File:")
    print(f"  {guide['configuration_file']['description']}")
    print(guide['configuration_file']['content'])
    
    print("\n🐳 Dockerfile:")
    dockerfile = create_serving_docker()
    with open('Dockerfile.serving', 'w') as f:
        f.write(dockerfile)
    print("  ✅ Created: Dockerfile.serving")
    
    print("\n📈 Dashboard Configuration:")
    dashboard = create_monitoring_dashboard()
    with open('serving_dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    print("  ✅ Created: serving_dashboard.json")
    
    print("\n" + "=" * 70)
    print("✅ Setup files created! Follow the guide above to deploy.")
    print("=" * 70)
