"""
Weather Forecasting API
FastAPI implementation for daily and hourly temperature predictions

Author: ML Team
Version: 1.0.0
Date: 2025-11-16
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

# CRITICAL FIX: Make OutlierClipper available in __main__ namespace
# The preprocessors were pickled with __main__ as the module reference
import __main__
__main__.OutlierClipper = OutlierClipper

# Import forecaster
from src.daily_forecast_model.infer import WeatherForecaster

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
evaluation_metrics = None

# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models and evaluation metrics at startup"""
    global daily_forecaster, evaluation_metrics
    
    try:
        # Load daily forecaster
        print("🔄 Loading daily forecaster...")
        daily_forecaster = WeatherForecaster()
        print(f"✅ Daily forecaster loaded successfully with {len(daily_forecaster.models)} models")
        
        # Load evaluation metrics
        eval_path = project_root / "src/daily_forecast_model/evaluate_results/evaluation_results.json"
        with open(eval_path, 'r') as f:
            evaluation_metrics = json.load(f)
        print(f"✅ Evaluation metrics loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down API...")

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
    datetime: Optional[str] = Field(default=None, description="Reference datetime (ISO 8601)")
    include_confidence: Optional[bool] = Field(default=False, description="Include confidence intervals")
    hours_ahead: Optional[int] = Field(default=24, ge=1, le=24, description="Number of hours to forecast (1-24)")
    
    @field_validator('datetime')
    @classmethod
    def validate_datetime(cls, v):
        if v is not None:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError('Datetime must be in ISO 8601 format')
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
            "status": "not_implemented",
            "count": 0,
            "models": []
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
    - Minimum 168 hours (7 days) of historical data
    - Hourly weather dataset at dataset/hn_hourly.csv
    
    **Returns:**
    - Temperature predictions for next 24 hours
    - Optional: Confidence intervals
    
    **Status:** Not yet implemented
    
    **Next Steps:**
    1. Implement HourlyForecaster class in src/hourly_forecast_model/infer_hourly.py
    2. Train hourly models (h+1 to h+24) using preprocessed_hourly.py
    3. Add evaluation metrics for hourly models
    """
    raise HTTPException(
        status_code=501,
        detail={
            "code": "NOT_IMPLEMENTED",
            "message": "Hourly forecasting not yet implemented",
            "details": {
                "available_endpoint": "/api/v1/forecast/daily",
                "status": "Use daily forecast endpoint for now",
                "roadmap": "Hourly forecasting scheduled for implementation"
            }
        }
    )


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
