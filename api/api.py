# Suppress TensorFlow warnings BEFORE any imports
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import json
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
# Import Gemini helper (generative enrichment)
from src.generative.gemini_weather import (
    get_weather_details_via_gemini,
    get_hourly_metadata_via_gemini,
)
# Import data feeder
from src.data_feeder import update_weather_data

project_root = Path(__file__).parent
if project_root.name == '':  # Handle case where parent is root
    project_root = Path('/app')
sys.path.insert(0, str(project_root))

# Import ClearML for production monitoring
try:
    from clearml import Task, Logger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("⚠️  ClearML not available - monitoring disabled")
    print("   Install with: pip install clearml")

# Import ALL custom transformers and functions BEFORE loading models (needed for unpickling)
# This ensures they're in the module namespace when joblib unpickles the preprocessing pipelines
from src.daily_forecast_model.process import (
    OutlierClipper,
    create_lag_features,
    create_rolling_features,
    create_cyclical_wind_direction,
    create_temporal_features,
    create_day_length_feature,
    create_interaction_features
)

# Import hourly model wrapper
from src.hourly_forecast_model.train import PerHorizonWrapper

# CRITICAL FIX: Make custom classes available in __main__ namespace
# The preprocessors were pickled with __main__ as the module reference
import __main__
__main__.OutlierClipper = OutlierClipper
__main__.PerHorizonWrapper = PerHorizonWrapper

# Import forecasters
from src.daily_forecast_model.infer import WeatherForecaster
from src.hourly_forecast_model.infer import HourlyWeatherForecaster

# Initialize FastAPI app
app = FastAPI(
    title="Weather Forecast API",
    description="ML-based temperature forecasting for Hanoi, Vietnam",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
daily_forecaster = None
hourly_forecaster = None
evaluation_metrics = None

# ClearML monitoring
clearml_task = None
clearml_logger = None

# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models and evaluation metrics at startup"""
    global daily_forecaster, hourly_forecaster, evaluation_metrics, clearml_task, clearml_logger
    
    try:
        # Initialize ClearML monitoring for production
        if CLEARML_AVAILABLE:
            print("🔄 Initializing ClearML production monitoring...")
            clearml_task = Task.init(
                project_name="Hanoi Weather Forecast",
                task_name="Production API - Inference Monitoring",
                task_type="inference",
                reuse_last_task_id=True,  # Reuse same task for continuous monitoring
                auto_resource_monitoring=False  # Disable GPU/resource monitoring for CPU-only deployment
            )
            clearml_logger = clearml_task.get_logger()
            print("✅ ClearML monitoring enabled")
            
            # Log API configuration
            clearml_task.connect_configuration({
                "api_version": "1.0.0",
                "host": "0.0.0.0",
                "port": 8000,
                "model_path": str(project_root / "trained_models"),
                "data_path": str(project_root / "dataset/hn_daily.csv")
            })
        
        # Load daily forecaster
        print("🔄 Loading daily forecaster...")
        daily_forecaster = WeatherForecaster()
        print(f"✅ Daily forecaster loaded successfully with {len(daily_forecaster.models)} models")
        
        # Load hourly forecaster
        print("🔄 Loading hourly forecaster...")
        try:
            hourly_forecaster = HourlyWeatherForecaster()
            print(f"✅ Hourly forecaster loaded successfully")
        except Exception as e:
            print(f"⚠️  Hourly forecaster not available: {e}")
            hourly_forecaster = None
        
        # Load evaluation metrics
        eval_path = project_root / "src/daily_forecast_model/evaluate_results/evaluation_results.json"
        with open(eval_path, 'r') as f:
            evaluation_metrics = json.load(f)
        print(f"✅ Evaluation metrics loaded successfully")
        
        # Log model info to ClearML
        if clearml_logger:
            for target in ['t+1', 't+2', 't+3', 't+4', 't+5']:
                metrics = evaluation_metrics[target]['metrics']
                clearml_logger.report_single_value(f"baseline_{target}_RMSE", metrics['RMSE'])
                clearml_logger.report_single_value(f"baseline_{target}_R2", metrics['R2'])
        
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down API...")
    
    # Close ClearML task
    if clearml_task:
        print("📊 Finalizing ClearML monitoring...")
        clearml_task.close()

# ============================================================================
# REQUEST/RESPONSE MODELS (Pydantic)
# ============================================================================

class DailyForecastRequest(BaseModel):
    """Request schema for daily temperature forecast"""
    location: Optional[str] = Field(default="Hanoi, Vietnam", description="Location name")
    include_confidence: Optional[bool] = Field(default=False, description="Include 95% confidence intervals")


class HourlyForecastRequest(BaseModel):
    """Request schema for hourly temperature forecast"""
    location: Optional[str] = Field(default="Hanoi, Vietnam", description="Location name")
    reference_datetime: Optional[str] = Field(default=None, description="Reference datetime to predict from (ISO 8601)")
    include_confidence: Optional[bool] = Field(default=False, description="Include confidence intervals")
    
    @field_validator('reference_datetime')
    @classmethod
    def validate_datetime(cls, v):
        if v is not None:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError('Reference datetime must be in ISO 8601 format')
        return v


class ConfidenceInterval(BaseModel):
    """Confidence interval for temperature prediction"""
    lower: float = Field(..., description="Lower bound (celsius)")
    upper: float = Field(..., description="Upper bound (celsius)")
    confidence_level: float = Field(default=0.95, description="Confidence level (0.95 = 95%)")


class ModelPerformance(BaseModel):
    """Model performance metrics from test set"""
    test_rmse: float = Field(..., description="Root Mean Squared Error (celsius)")
    test_mae: float = Field(..., description="Mean Absolute Error (celsius)")
    test_r2: float = Field(..., description="R-squared score")
    test_mape: float = Field(..., description="Mean Absolute Percentage Error (%)")


class DailyPrediction(BaseModel):
    """Single day temperature prediction"""
    target: str = Field(..., description="Forecast horizon (t+1, t+2, etc.)")
    forecast_date: str = Field(..., description="Date of forecast (YYYY-MM-DD)")
    temperature: float = Field(..., description="Predicted temperature (celsius)")
    unit: str = Field(default="celsius", description="Temperature unit")
    confidence_interval: Optional[ConfidenceInterval] = None
    model_performance: Optional[ModelPerformance] = None


class HourlyPrediction(BaseModel):
    """Single hour temperature prediction"""
    target: str = Field(..., description="Forecast horizon (h+1, h+2, etc.)")
    forecast_datetime: str = Field(..., description="Datetime of forecast (ISO 8601)")
    temperature: float = Field(..., description="Predicted temperature (celsius)")
    unit: str = Field(default="celsius", description="Temperature unit")
    confidence_interval: Optional[ConfidenceInterval] = None


class ForecastMetadata(BaseModel):
    """Metadata about the forecast"""
    total_predictions: int
    average_confidence: Optional[float] = None
    data_requirements: Dict[str, Any]


class DailyForecastResponse(BaseModel):
    """Response schema for daily forecast"""
    status: str = Field(..., description="Response status (success/error)")
    forecast_type: str = Field(default="daily", description="Type of forecast")
    location: str
    reference_date: str
    generated_at: str = Field(..., description="Timestamp when forecast was generated")
    model_version: str = Field(default="v1.0.0", description="Model version")
    predictions: List[DailyPrediction]
    metadata: Optional[ForecastMetadata] = None


class HourlyForecastResponse(BaseModel):
    """Response schema for hourly forecast"""
    status: str = Field(..., description="Response status (success/error)")
    forecast_type: str = Field(default="hourly", description="Type of forecast")
    location: str
    reference_datetime: str
    generated_at: str
    model_version: str = Field(default="v1.0.0-hourly", description="Model version")
    predictions: List[HourlyPrediction]
    metadata: Optional[ForecastMetadata] = None


class ErrorResponse(BaseModel):
    """Error response schema"""
    status: str = Field(default="error", description="Response status")
    error: Dict[str, Any] = Field(..., description="Error details")
    timestamp: str = Field(..., description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    models: Dict[str, Any]
    data: Optional[Dict[str, Any]] = None
    system: Optional[Dict[str, Any]] = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_confidence_interval(temperature: float, rmse: float, confidence_level: float = 0.95) -> ConfidenceInterval:
    """
    Calculate confidence interval using RMSE
    
    Formula: temperature ± z * RMSE
    where z = 1.96 for 95% confidence level
    
    Note: This is an approximation based on test set performance,
    not true predictive uncertainty from the model.
    """
    z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
    margin = z_score * rmse
    
    return ConfidenceInterval(
        lower=round(temperature - margin, 1),
        upper=round(temperature + margin, 1),
        confidence_level=confidence_level
    )


def get_model_performance(target: str) -> ModelPerformance:
    """Get performance metrics for a specific target"""
    if not evaluation_metrics or target not in evaluation_metrics:
        raise ValueError(f"Metrics not found for target: {target}")
    
    metrics = evaluation_metrics[target]['metrics']
    return ModelPerformance(
        test_rmse=round(metrics['RMSE'], 2),
        test_mae=round(metrics['MAE'], 2),
        test_r2=round(metrics['R2'], 3),
        test_mape=round(metrics['MAPE'], 2)
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs"""
    return {
        "message": "Weather Forecast API",
        "version": "1.0.0",
        "docs": "/api/v1/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/data/historical", tags=["Data"])
async def get_historical_data(days: Optional[int] = None):
    """
    Get historical weather data
    
    Args:
        days: Number of days of historical data to return (max: 365). If omitted or null,
              the endpoint will return the entire historical dataset.
    
    Returns:
        Historical weather data in JSON format
    """
    try:
        # If `days` is provided, enforce reasonable limits (max 365).
        # If `days` is None, return the full dataset.
        if days is not None:
            # ensure non-negative and cap to 365
            days = max(0, days)
            days = min(days, 365)
        
        # Load historical data
        daily_data_path = project_root / "dataset/hn_daily.csv"
        df = pd.read_csv(daily_data_path, parse_dates=['datetime'])
        df = df.sort_values('datetime', ascending=False)
        
        # If days was provided, return the last N days; otherwise return all records
        if days is not None and days > 0:
            df = df.head(days).sort_values('datetime', ascending=True)
        else:
            df = df.sort_values('datetime', ascending=True)
        
        # Replace NaN and Inf values with None for JSON serialization
        df = df.replace([np.nan, np.inf, -np.inf], None)
        
        # Convert to records
        records = df.to_dict('records')
        
        # Convert datetime to string and clean up values
        for record in records:
            if isinstance(record.get('datetime'), pd.Timestamp):
                record['datetime'] = record['datetime'].strftime('%Y-%m-%d')
            
            # Ensure all numeric values are JSON-compliant
            for key, value in record.items():
                if value is None:
                    continue
                if isinstance(value, (np.integer, np.floating)):
                    if np.isnan(value) or np.isinf(value):
                        record[key] = None
                    else:
                        record[key] = float(value) if isinstance(value, np.floating) else int(value)
        
        return {
            "status": "success",
            "data_type": "historical",
            "total_records": len(records),
            "date_range": {
                "start": records[0]['datetime'] if records else None,
                "end": records[-1]['datetime'] if records else None
            },
            "records": records
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Historical data file not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading historical data: {str(e)}"
        )


@app.get("/api/v1/data/historical/hourly", tags=["Data"])
async def get_historical_hourly_data(hours: int = 24):
    """
    Get historical hourly weather data
    
    Args:
        hours: Number of hours of historical data to return (default: 24, max: 168)
    
    Returns:
        Historical hourly weather data in JSON format
    """
    try:
        # Limit hours to prevent excessive data transfer
        hours = min(hours, 168)
        
        # Load historical hourly data
        hourly_data_path = project_root / "dataset/hn_hourly.csv"
        df = pd.read_csv(hourly_data_path, parse_dates=['datetime'])
        df = df.sort_values('datetime', ascending=False)
        
        # Get last N hours
        df = df.head(hours).sort_values('datetime', ascending=True)
        
        # Replace NaN and Inf values with None for JSON serialization
        df = df.replace([np.nan, np.inf, -np.inf], None)
        
        # Convert to records
        records = df.to_dict('records')
        
        # Convert datetime to string and clean up values
        for record in records:
            if isinstance(record.get('datetime'), pd.Timestamp):
                record['datetime'] = record['datetime'].isoformat()
            
            # Ensure all numeric values are JSON-compliant
            for key, value in record.items():
                if value is None:
                    continue
                if isinstance(value, (np.integer, np.floating)):
                    if np.isnan(value) or np.isinf(value):
                        record[key] = None
                    else:
                        record[key] = float(value) if isinstance(value, np.floating) else int(value)
        
        return {
            "status": "success",
            "data_type": "historical_hourly",
            "total_records": len(records),
            "datetime_range": {
                "start": records[0]['datetime'] if records else None,
                "end": records[-1]['datetime'] if records else None
            },
            "records": records
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Historical hourly data file not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading historical hourly data: {str(e)}"
        )


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API and model health status
    
    Returns:
    - API status (healthy/unhealthy)
    - Model loading status
    - System information
    """
    models_status = {
        "daily": {
            "status": "loaded" if daily_forecaster else "not_loaded",
            "count": len(daily_forecaster.models) if daily_forecaster else 0,
            "models": list(daily_forecaster.models.keys()) if daily_forecaster else []
        },
        "hourly": {
            "status": "loaded" if hourly_forecaster else "not_loaded",
            "count": 24 if hourly_forecaster else 0,
            "models": [f"t+{i}h" for i in range(1, 25)] if hourly_forecaster else []
        }
    }
    
    # Check if critical components are loaded
    status = "healthy" if daily_forecaster and evaluation_metrics else "unhealthy"
    
    # Optional: Add data statistics
    data_info = None
    try:
        import pandas as pd
        daily_data = pd.read_csv(project_root / "dataset/hn_daily.csv")
        hourly_data = pd.read_csv(project_root / "dataset/hn_hourly.csv")
        
        data_info = {
            "daily_records": len(daily_data),
            "hourly_records": len(hourly_data),
            "latest_date": daily_data['datetime'].max() if 'datetime' in daily_data.columns else None
        }
    except Exception as e:
        print(f"Warning: Could not load data info: {e}")
    
    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        models=models_status,
        data=data_info
    )


@app.post("/api/v1/forecast/daily", response_model=DailyForecastResponse, tags=["Forecasting"])
async def forecast_daily(request: DailyForecastRequest):
    """
    Get 5-day temperature forecast (t+1 to t+5)
    
    **Requirements:**
    - Minimum 84 days of historical data
    - Daily weather dataset at dataset/hn_daily.csv
    
    **Returns:**
    - Temperature predictions for next 5 days after the last date in dataset
    - Model performance metrics from test set (always included)
    - Optional: 95% confidence intervals (±1.96 × RMSE)
    
    **Example Request:**
    ```json
    {
        "location": "Hanoi, Vietnam",
        "include_confidence": true
    }
    ```
    """
    # Check if models are loaded
    if not daily_forecaster:
        raise HTTPException(
            status_code=503,
            detail="Daily forecaster not loaded. Service unavailable."
        )
    
    if not evaluation_metrics:
        raise HTTPException(
            status_code=503,
            detail="Evaluation metrics not loaded. Service unavailable."
        )
    
    try:
        # Track request start time
        start_time = time.time()
        
        # Load raw data - use ALL available data in dataset
        import pandas as pd
        data_path = project_root / "dataset/hn_daily.csv"
        raw_data = pd.read_csv(data_path)
        
        # Set datetime as index (column is named 'datetime' in the CSV)
        raw_data['datetime'] = pd.to_datetime(raw_data['datetime'])
        raw_data.set_index('datetime', inplace=True)
        
        # Check minimum data requirement
        if len(raw_data) < 84:
            raise ValueError(f"Insufficient data: need at least 84 days, got {len(raw_data)} days")
        
        # Get the last date in dataset as reference date
        reference_date = raw_data.index.max().strftime('%Y-%m-%d')
        
        # Get predictions from forecaster (predicts 5 days after the last date in dataset)
        result = daily_forecaster.predict_with_metadata(raw_data)
        
        # Build predictions list
        predictions = []
        forecast_dates = result['forecast_dates']
        
        for i, target in enumerate(['t+1', 't+2', 't+3', 't+4', 't+5']):
            temp = round(result['predictions'][target], 1)
            
            pred = DailyPrediction(
                target=target,
                forecast_date=forecast_dates[i],
                temperature=temp,
                unit="celsius"
            )
            
            # Add confidence interval if requested
            if request.include_confidence:
                rmse = evaluation_metrics[target]['metrics']['RMSE']
                pred.confidence_interval = calculate_confidence_interval(temp, rmse)
            
            # Always include performance metrics
            pred.model_performance = get_model_performance(target)
            
            predictions.append(pred)
        
        # Always build metadata
        # Calculate average R² as confidence measure
        avg_r2 = sum(
            evaluation_metrics[t]['metrics']['R2'] 
            for t in ['t+1', 't+2', 't+3', 't+4', 't+5']
        ) / 5
        
        metadata = ForecastMetadata(
            total_predictions=5,
            average_confidence=round(avg_r2, 3),
            data_requirements={
                "minimum_historical_days": 84,
                "features_used": len(daily_forecaster.feature_names.get('t+1', [])),
                "latest_data_date": reference_date
            }
        )
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Log to ClearML
        if clearml_logger:
            # Log inference metrics
            clearml_logger.report_scalar(
                title="Production Metrics",
                series="inference_latency_ms",
                value=inference_time,
                iteration=clearml_task.get_last_iteration()
            )
            
            # Log request count
            clearml_logger.report_scalar(
                title="Production Metrics",
                series="prediction_requests",
                value=1,
                iteration=clearml_task.get_last_iteration()
            )
            
            # Log predictions for each target
            for i, target in enumerate(['t+1', 't+2', 't+3', 't+4', 't+5']):
                clearml_logger.report_scalar(
                    title="Predictions",
                    series=f"{target}_temperature",
                    value=predictions[i].temperature,
                    iteration=clearml_task.get_last_iteration()
                )
            
            # Log confidence interval usage
            if request.include_confidence:
                clearml_logger.report_scalar(
                    title="Production Metrics",
                    series="confidence_interval_requests",
                    value=1,
                    iteration=clearml_task.get_last_iteration()
                )
        
        return DailyForecastResponse(
            status="success",
            forecast_type="daily",
            location=request.location,
            reference_date=reference_date,
            generated_at=datetime.now().isoformat(),
            model_version="v1.0.0",
            predictions=predictions,
            metadata=metadata
        )
        
    except FileNotFoundError as e:
        # Log error to ClearML
        if clearml_logger:
            clearml_logger.report_scalar(
                title="Production Metrics",
                series="data_load_errors",
                value=1,
                iteration=clearml_task.get_last_iteration()
            )
        
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DATA_LOAD_ERROR",
                "message": f"Data file not found: {str(e)}",
                "details": {"file": "dataset/hn_daily.csv"}
            }
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DATA_LOAD_ERROR",
                "message": f"Data file not found: {str(e)}",
                "details": {"file": "dataset/hn_daily.csv"}
            }
        )
    
    except ValueError as e:
        error_msg = str(e)
        if "not enough" in error_msg.lower() or "insufficient" in error_msg.lower():
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "INSUFFICIENT_DATA",
                    "message": "Not enough historical data for prediction",
                    "details": {
                        "required_days": 84,
                        "error": error_msg
                    }
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_INPUT",
                    "message": error_msg
                }
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "PREDICTION_ERROR",
                "message": f"Failed to generate prediction: {str(e)}",
                "details": {"error_type": type(e).__name__}
            }
        )


@app.post("/api/v1/forecast/hourly", response_model=HourlyForecastResponse, tags=["Forecasting"])
async def forecast_hourly(request: HourlyForecastRequest):
    """
    Get 24-hour temperature forecast (h+1 to h+24)
    
    **Requirements:**
    - Uses internal hourly dataset (dataset/hn_hourly.csv)
    - Predicts 24 hours after the last datetime in dataset
    
    **Returns:**
    - Temperature predictions for next 24 hours
    - Optional: Confidence intervals (±1.96 × RMSE)
    
    **Example Request:**
    ```json
    {
        "location": "Hanoi, Vietnam",
        "include_confidence": true,
        "hours_ahead": 24
    }
    ```
    """
    # Check if hourly forecaster is available
    if not hourly_forecaster:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "SERVICE_UNAVAILABLE",
                "message": "Hourly forecasting model not loaded",
                "details": {
                    "available_endpoint": "/api/v1/forecast/daily",
                    "status": "Hourly model not initialized at startup"
                }
            }
        )
    
    try:
        start_time = time.time()
        
        # Load hourly historical data from internal dataset
        hourly_data_path = project_root / "dataset/hn_hourly.csv"
        if not hourly_data_path.exists():
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "DATA_NOT_FOUND",
                    "message": "Internal hourly dataset not found",
                    "details": {"expected_path": str(hourly_data_path)}
                }
            )
        
        # Load all available data
        df_hourly = pd.read_csv(hourly_data_path)
        df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
        df_hourly = df_hourly.sort_values('datetime').reset_index(drop=True)
        
        # Check minimum data requirement (168 hours = 7 days)
        if len(df_hourly) < 168:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "INSUFFICIENT_DATA",
                    "message": f"Internal dataset has insufficient data: need at least 168 hours, got {len(df_hourly)}",
                    "details": {"min_required": 168, "available": len(df_hourly)}
                }
            )
        
        # Get reference datetime - use from request or default to last datetime in dataset
        if request.reference_datetime:
            reference_datetime = pd.to_datetime(request.reference_datetime)
            # Filter data up to reference datetime
            df_hourly = df_hourly[df_hourly['datetime'] <= reference_datetime]
            if len(df_hourly) < 168:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "INSUFFICIENT_DATA",
                        "message": f"Not enough data before reference datetime: need 168 hours, got {len(df_hourly)}",
                        "details": {"reference_datetime": request.reference_datetime}
                    }
                )
        else:
            reference_datetime = df_hourly['datetime'].iloc[-1]
        
        # Get predictions from forecaster
        results_df = hourly_forecaster.predict(df_hourly)
        
        # Format predictions
        predictions = []
        for _, row in results_df.iterrows():
            pred = HourlyPrediction(
                target=row['hour_name'],
                forecast_datetime=row['timestamp'].isoformat(),
                temperature=round(row['predicted_temp'], 1),
                unit="celsius"
            )
            
            # Add confidence intervals if requested
            if request.include_confidence:
                # Use approximate RMSE from hourly model evaluation
                # Average RMSE across 24-hour horizons
                approx_rmse = 1.6
                pred.confidence_interval = calculate_confidence_interval(
                    pred.temperature, 
                    approx_rmse
                )
            
            predictions.append(pred)
        
        # Always return all 24 hours of predictions
        
        # Calculate metadata
        metadata = ForecastMetadata(
            total_predictions=len(predictions),
            average_confidence=None,
            data_requirements={
                "minimum_historical_hours": 168,
                "data_available": len(df_hourly),
                "latest_data_datetime": reference_datetime.isoformat()
            }
        )
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Log to ClearML
        if clearml_logger:
            clearml_logger.report_scalar(
                title="Production Metrics",
                series="hourly_inference_latency_ms",
                value=inference_time,
                iteration=clearml_task.get_last_iteration()
            )
            
            clearml_logger.report_scalar(
                title="Production Metrics",
                series="hourly_prediction_requests",
                value=1,
                iteration=clearml_task.get_last_iteration()
            )
        
        return HourlyForecastResponse(
            status="success",
            forecast_type="hourly",
            location=request.location,
            reference_datetime=reference_datetime.isoformat(),
            generated_at=datetime.now().isoformat(),
            model_version="v1.0.0-hourly",
            predictions=predictions,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log error to ClearML
        if clearml_logger:
            clearml_logger.report_scalar(
                title="Production Metrics",
                series="hourly_prediction_errors",
                value=1,
                iteration=clearml_task.get_last_iteration()
            )
        
        raise HTTPException(
            status_code=500,
            detail={
                "code": "PREDICTION_ERROR",
                "message": f"Failed to generate hourly forecast: {str(e)}",
                "details": {"error_type": type(e).__name__}
            }
        )


@app.get("/api/v1/gemini/weather_details", tags=["Generative"])
async def gemini_weather_details(date: str):
    """Return non-temperature weather details for the specified date.

    This endpoint loads the internal daily dataset and extracts the row for the
    provided `date` (YYYY-MM-DD). It then calls the Gemini helper to format or
    enrich those details. Temperature fields are explicitly excluded from the
    generative output.
    """
    try:
        # Validate and load dataset
        daily_data_path = project_root / "dataset/hn_daily.csv"
        if not daily_data_path.exists():
            raise HTTPException(status_code=404, detail="Daily dataset not found")

        df = pd.read_csv(daily_data_path, parse_dates=["datetime"])
        # Normalize date string
        try:
            query_date = pd.to_datetime(date).date()
        except Exception:
            raise HTTPException(status_code=400, detail="`date` must be in YYYY-MM-DD format")

        # Find the row with matching date
        df['date_only'] = pd.to_datetime(df['datetime']).dt.date
        row = df[df['date_only'] == query_date]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"No record found for date {date}")

        # Use the first match
        record = row.iloc[0].to_dict()

        # Remove temperature-related fields before sending to Gemini helper
        for tkey in ('temp', 'tempmin', 'tempmax'):
            record.pop(tkey, None)

        details = get_weather_details_via_gemini(date, record)

        return {
            "status": "success",
            "date": date,
            "details": details
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "GEMINI_ERROR", "message": str(e)})


@app.get("/api/v1/gemini/hourly_metadata", tags=["Generative"])
async def gemini_hourly_metadata(datetime: str):
    """Return non-temperature hourly metadata for the specified datetime (ISO 8601).

    Loads `dataset/hn_hourly.csv`, finds the matching row for the provided
    `datetime`, strips temperature fields and delegates to the Gemini helper.
    """
    try:
        hourly_data_path = project_root / "dataset/hn_hourly.csv"
        if not hourly_data_path.exists():
            raise HTTPException(status_code=404, detail="Hourly dataset not found")

        df = pd.read_csv(hourly_data_path, parse_dates=["datetime"])

        # Parse query datetime
        try:
            query_dt = pd.to_datetime(datetime)
        except Exception:
            raise HTTPException(status_code=400, detail="`datetime` must be ISO 8601")

        # Find the row matching the provided datetime (allow exact match)
        row = df[df['datetime'] == query_dt]
        if row.empty:
            # If exact match not found, try matching by truncating seconds
            row = df[df['datetime'].dt.floor('min') == query_dt.floor('min')]

        if row.empty:
            raise HTTPException(status_code=404, detail=f"No hourly record found for datetime {datetime}")

        record = row.iloc[0].to_dict()

        # Remove temp fields before calling Gemini
        for tkey in ("temp", "temperature", "tempmin", "tempmax"):
            record.pop(tkey, None)

        details = get_hourly_metadata_via_gemini(datetime, record)

        return {
            "status": "success",
            "datetime": datetime,
            "details": details
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "GEMINI_HOURLY_ERROR", "message": str(e)})


@app.get("/api/v1/models/metadata", tags=["Models"])
async def get_models_metadata(model_type: Optional[str] = "all"):
    """
    Get detailed model metadata and performance metrics
    
    **Parameters:**
    - model_type: Filter by type ("daily", "hourly", or "all")
    
    **Returns:**
    - Model architecture details
    - Training information
    - Test set performance metrics
    - Feature counts and preprocessing info
    """
    if not daily_forecaster:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Service unavailable."
        )
    
    if not evaluation_metrics:
        raise HTTPException(
            status_code=503,
            detail="Evaluation metrics not loaded."
        )
    
    try:
        # Build daily models metadata
        daily_models = []
        for target in ['t+1', 't+2', 't+3', 't+4', 't+5']:
            metrics = evaluation_metrics[target]['metrics']
            n_samples = evaluation_metrics[target]['n_samples']
            
            # Get actual model info
            model = daily_forecaster.models.get(target)
            model_type_name = type(model).__name__ if model else "Unknown"
            
            # Get model-specific info
            model_params = {}
            if hasattr(model, 'get_params'):
                params = model.get_params()
                # Extract key hyperparameters based on model type
                if 'XGB' in model_type_name:
                    model_params = {
                        "n_estimators": params.get('n_estimators', 'N/A'),
                        "max_depth": params.get('max_depth', 'N/A'),
                        "learning_rate": params.get('learning_rate', 'N/A'),
                        "subsample": params.get('subsample', 'N/A')
                    }
                elif 'RandomForest' in model_type_name:
                    model_params = {
                        "n_estimators": params.get('n_estimators', 'N/A'),
                        "max_depth": params.get('max_depth', 'N/A'),
                        "min_samples_split": params.get('min_samples_split', 'N/A')
                    }
                elif 'LightGBM' in model_type_name or 'LGBM' in model_type_name:
                    model_params = {
                        "n_estimators": params.get('n_estimators', 'N/A'),
                        "max_depth": params.get('max_depth', 'N/A'),
                        "learning_rate": params.get('learning_rate', 'N/A'),
                        "num_leaves": params.get('num_leaves', 'N/A')
                    }
                elif 'CatBoost' in model_type_name:
                    model_params = {
                        "iterations": params.get('iterations', 'N/A'),
                        "depth": params.get('depth', 'N/A'),
                        "learning_rate": params.get('learning_rate', 'N/A')
                    }
            
            # Get model file size
            model_path = project_root / "trained_models" / f"model_{target}.joblib"
            model_size_mb = round(model_path.stat().st_size / (1024 * 1024), 2) if model_path.exists() else None
            
            # Get training date from file modification time
            training_date = None
            if model_path.exists():
                import time
                training_date = time.strftime('%Y-%m-%d', time.localtime(model_path.stat().st_mtime))
            
            model_info = {
                "target": target,
                "model_type": model_type_name,
                "features": len(daily_forecaster.feature_names.get(target, [])),
                "hyperparameters": model_params,
                "performance": {
                    "test_rmse": round(metrics['RMSE'], 3),
                    "test_mae": round(metrics['MAE'], 3),
                    "test_r2": round(metrics['R2'], 3),
                    "test_mape": round(metrics['MAPE'], 3)
                },
                "test_samples": n_samples,
                "model_path": f"trained_models/model_{target}.joblib",
                "model_size_mb": model_size_mb,
                "training_date": training_date
            }
            daily_models.append(model_info)
        
        # Build response
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        # Add daily models if requested
        if model_type in ["all", "daily"]:
            # Get overall training date (use the first model's date)
            overall_training_date = daily_models[0].get('training_date', 'Unknown') if daily_models else 'Unknown'
            
            response["models"]["daily"] = {
                "model_count": 5,
                "architecture": "Per-target ensemble (5 independent models)",
                "training_date": overall_training_date,
                "models": daily_models
            }
        
        # Add hourly models placeholder
        if model_type in ["all", "hourly"]:
            response["models"]["hourly"] = {
                "status": "not_implemented",
                "message": "Hourly models not yet trained"
            }
        
        # Add preprocessing info
        response["preprocessing"] = {
            "daily_pipeline": {
                "steps": [
                    "temporal_features",
                    "day_length",
                    "cyclical_wind_direction",
                    "interaction_features",
                    "lag_features (1, 3, 7 days)",
                    "rolling_windows (7, 14, 30, 84 days)",
                    "standard_scaler"
                ],
                "pipeline_path": "processed_data/pipelines/"
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "METADATA_ERROR",
                "message": f"Error loading model metadata: {str(e)}"
            }
        )


# ============================================================================
# PERFORMANCE ENDPOINTS
# ============================================================================

@app.get("/api/v1/performance/daily", tags=["Performance"])
async def get_daily_performance():
    """Get evaluation metrics for the daily forecast model."""
    try:
        metrics_path = project_root / "src/daily_forecast_model/evaluate_results/evaluation_metrics.csv"
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail="Daily performance metrics not found.")
        
        df = pd.read_csv(metrics_path)
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/performance/hourly", tags=["Performance"])
async def get_hourly_performance():
    """Get evaluation metrics for the hourly forecast model."""
    try:
        results_path = project_root / "src/hourly_forecast_model/evaluate_results/evaluation_results.json"
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Hourly performance results not found.")
        
        with open(results_path, 'r') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/performance/daily/plots/{filename}", tags=["Performance"])
async def get_daily_plot(filename: str):
    """Get a specific evaluation plot for the daily model."""
    try:
        plot_path = project_root / f"src/daily_forecast_model/evaluate_results/plots/{filename}"
        if not plot_path.exists():
            raise HTTPException(status_code=404, detail=f"Plot '{filename}' not found.")
        return FileResponse(plot_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/performance/hourly/plots/{filename}", tags=["Performance"])
async def get_hourly_plot(filename: str):
    """Get a specific evaluation plot for the hourly model."""
    try:
        plot_path = project_root / f"src/hourly_forecast_model/evaluate_results/plots/{filename}"
        if not plot_path.exists():
            raise HTTPException(status_code=404, detail=f"Plot '{filename}' not found.")
        return FileResponse(plot_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ADMIN ENDPOINTS (for cronjobs)
# ============================================================================

class AdminTaskResponse(BaseModel):
    """Response for admin tasks"""
    status: str
    task: str
    timestamp: str
    message: str
    details: Optional[Dict[str, Any]] = None

@app.post("/api/v1/admin/trigger-data-update", response_model=AdminTaskResponse, tags=["Admin"])
async def trigger_data_update():
    """
    **[Admin]** Trigger a data update process.
    
    This endpoint should be called periodically (e.g., daily) by a cronjob
    to fetch the latest weather data and append it to the internal datasets.
    
    **Cronjob Example (daily at 1 AM):**
    ```bash
    0 1 * * * curl -X POST http://localhost:8000/api/v1/admin/trigger-data-update
    ```
    """
    # In a real application, this would trigger a background task
    # For now, we'll just return a success message
    return AdminTaskResponse(
        status="success",
        task="data_update",
        timestamp=datetime.now().isoformat(),
        message="Data update process triggered successfully.",
        details={
            "info": "This is a placeholder. Implement the actual data fetching logic here."
        }
    )

@app.post("/api/v1/admin/trigger-retraining", response_model=AdminTaskResponse, tags=["Admin"])
async def trigger_retraining():
    """
    **[Admin]** Trigger the model retraining pipeline.
    
    This endpoint should be called periodically (e.g., weekly or monthly) 
    by a cronjob to retrain the models with the latest data.
    
    **Cronjob Example (every Sunday at 2 AM):**
    ```bash
    0 2 * * 0 curl -X POST http://localhost:8000/api/v1/admin/trigger-retraining
    ```
    """
    # In a real application, this would trigger a long-running background task
    # (e.g., using Celery or FastAPI's BackgroundTasks)
    return AdminTaskResponse(
        status="success",
        task="model_retraining",
        timestamp=datetime.now().isoformat(),
        message="Model retraining process triggered successfully.",
        details={
            "info": "This is a placeholder. Implement the actual retraining logic here."
        }
    )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": exc.detail if isinstance(exc.detail, dict) else {
                "code": "HTTP_ERROR",
                "message": str(exc.detail)
            },
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for uncaught errors"""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🌤️  Weather Forecast API")
    print("=" * 60)
    print("📚 Documentation: http://localhost:8000/docs")
    print("🏥 Health Check:  http://localhost:8000/api/v1/health")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
