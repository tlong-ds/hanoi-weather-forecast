# Weather Forecasting API Design

**Version:** 1.0  
**Last Updated:** 2025-11-16  
**Project:** Hanoi Weather Forecast ML System

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [API Endpoints](#api-endpoints)
   - [Daily Temperature Forecast](#1-daily-temperature-forecast)
   - [Hourly Temperature Forecast](#2-hourly-temperature-forecast)
   - [Model Health Check](#3-model-health-check)
   - [Model Metadata](#4-model-metadata)
4. [Data Models](#data-models)
5. [Error Handling](#error-handling)
6. [Implementation Guide](#implementation-guide)
7. [Security & Performance](#security--performance)

---

## Overview

This API provides **machine learning-based temperature forecasting** for Hanoi, Vietnam using trained models:

- **Daily Forecast:** 5-day ahead predictions (t+1 to t+5)
- **Hourly Forecast:** 24-hour ahead predictions (t+1 to t+24)

### Key Features

✅ **Per-target trained models** (separate model for each forecast horizon)  
✅ **Feature engineering pipeline** (lag features, rolling windows, temporal features)  
✅ **Confidence intervals** and prediction metadata  
✅ **Performance metrics** from test set evaluation  
✅ **Model versioning** and health monitoring  

### Technology Stack

- **ML Models:** RandomForest, XGBoost, LightGBM, CatBoost (optimized via Optuna)
- **Feature Engineering:** scikit-learn pipelines with custom transformers
- **Inference Module:** `src/daily_forecast_model/infer.py`
- **Preprocessing:** `src/daily_forecast_model/process.py`
- **Framework:** FastAPI (recommended) or Flask

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
├─────────────────────────────────────────────────────────────┤
│  /api/v1/forecast/daily    │  /api/v1/forecast/hourly       │
│  /api/v1/health            │  /api/v1/models/metadata       │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│              Inference Layer (Python)                        │
├─────────────────────────────────────────────────────────────┤
│  WeatherForecaster class (src/daily_forecast_model/infer.py)│
│  - predict_single()      (daily: t+1 to t+5)                │
│  - predict_with_metadata() (returns dates + confidence)     │
│                                                              │
│  HourlyForecaster class (to be implemented)                 │
│  - predict_hourly()      (hourly: t+1 to t+24)              │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│          Feature Engineering Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  src/daily_forecast_model/process.py                        │
│  - Temporal features (month, day, hour, cyclical encoding)  │
│  - Day length calculation                                   │
│  - Cyclical wind direction                                  │
│  - Interaction features (temp × humidity, etc.)             │
│  - Lag features (1, 3, 7 days/hours)                        │
│  - Rolling windows (7, 14, 30, 84 days/hours)               │
│  - StandardScaler preprocessing                             │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│              Trained Models & Pipelines                      │
├─────────────────────────────────────────────────────────────┤
│  Daily Models:                                               │
│  - trained_models/model_t+1.joblib                          │
│  - trained_models/model_t+2.joblib                          │
│  - trained_models/model_t+3.joblib                          │
│  - trained_models/model_t+4.joblib                          │
│  - trained_models/model_t+5.joblib                          │
│                                                              │
│  Preprocessors:                                              │
│  - processed_data/pipelines/preprocessor_t_1.joblib         │
│  - processed_data/pipelines/preprocessor_t_2.joblib         │
│  - ... (per-target pipelines)                               │
│                                                              │
│  Hourly Models (to be trained):                             │
│  - trained_models_hourly/model_h+1.joblib                   │
│  - trained_models_hourly/model_h+2.joblib                   │
│  - ... (h+1 to h+24)                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## API Endpoints

### Base URL

```
Production:  https://api.weather-forecast.com/api/v1
Development: http://localhost:8000/api/v1
```

---

### 1. Daily Temperature Forecast

**Endpoint:** `POST /forecast/daily`

**Description:** Returns 5-day temperature forecast (t+1 to t+5) based on historical weather data.

#### Request

**Headers:**
```http
Content-Type: application/json
Authorization: Bearer {API_KEY}  # Optional: for production
```

**Body:**
```json
{
  "location": "Hanoi, Vietnam",
  "date": "2024-01-15",
  "include_metadata": true,
  "include_confidence": true
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `location` | string | No | Location name (default: "Hanoi, Vietnam") |
| `date` | string | No | Reference date (ISO 8601). Default: today |
| `include_metadata` | boolean | No | Include model performance metrics (default: false) |
| `include_confidence` | boolean | No | Include prediction intervals (default: false) |

#### Response

**Success (200 OK):**
```json
{
  "status": "success",
  "forecast_type": "daily",
  "location": "Hanoi, Vietnam",
  "reference_date": "2024-01-15",
  "generated_at": "2024-01-15T08:30:00Z",
  "model_version": "v1.0.0",
  "predictions": [
    {
      "target": "t+1",
      "forecast_date": "2024-01-16",
      "temperature": 23.5,
      "unit": "celsius",
      "confidence_interval": {
        "lower": 22.0,
        "upper": 25.0,
        "confidence_level": 0.95
      },
      "model_performance": {
        "test_rmse": 1.52,
        "test_mae": 1.17,
        "test_r2": 0.902,
        "test_mape": 4.94
      }
    },
    {
      "target": "t+2",
      "forecast_date": "2024-01-17",
      "temperature": 24.1,
      "unit": "celsius",
      "confidence_interval": {
        "lower": 22.1,
        "upper": 26.1,
        "confidence_level": 0.95
      },
      "model_performance": {
        "test_rmse": 1.98,
        "test_mae": 1.55,
        "test_r2": 0.833,
        "test_mape": 6.66
      }
    },
    {
      "target": "t+3",
      "forecast_date": "2024-01-18",
      "temperature": 22.8,
      "unit": "celsius",
      "confidence_interval": {
        "lower": 20.7,
        "upper": 24.9,
        "confidence_level": 0.95
      },
      "model_performance": {
        "test_rmse": 2.14,
        "test_mae": 1.65,
        "test_r2": 0.806,
        "test_mape": 7.14
      }
    },
    {
      "target": "t+4",
      "forecast_date": "2024-01-19",
      "temperature": 21.9,
      "unit": "celsius",
      "confidence_interval": {
        "lower": 19.7,
        "upper": 24.1,
        "confidence_level": 0.95
      },
      "model_performance": {
        "test_rmse": 2.21,
        "test_mae": 1.73,
        "test_r2": 0.793,
        "test_mape": 7.55
      }
    },
    {
      "target": "t+5",
      "forecast_date": "2024-01-20",
      "temperature": 20.3,
      "unit": "celsius",
      "confidence_interval": {
        "lower": 18.1,
        "upper": 22.5,
        "confidence_level": 0.95
      },
      "model_performance": {
        "test_rmse": 2.21,
        "test_mae": 1.75,
        "test_r2": 0.789,
        "test_mape": 7.58
      }
    }
  ],
  "metadata": {
    "total_predictions": 5,
    "average_confidence": 0.852,
    "data_requirements": {
      "minimum_historical_days": 84,
      "features_used": 156,
      "latest_data_date": "2024-01-15"
    }
  }
}
```

#### Implementation Module

**File:** `src/daily_forecast_model/infer.py`

**Core Functions:**
```python
from src.daily_forecast_model.infer import WeatherForecaster

# Initialize forecaster (loads all 5 models)
forecaster = WeatherForecaster()

# Get predictions with metadata
result = forecaster.predict_with_metadata(
    data_path="dataset/hn_daily.csv",
    reference_date="2024-01-15"
)

# Returns:
# {
#     'predictions': {'t+1': 23.5, 't+2': 24.1, ...},
#     'dates': {'t+1': '2024-01-16', 't+2': '2024-01-17', ...},
#     'reference_date': '2024-01-15'
# }
```

---

### 2. Hourly Temperature Forecast

**Endpoint:** `POST /forecast/hourly`

**Description:** Returns 24-hour temperature forecast (t+1 to t+24 hours) for intra-day predictions.

#### Request

**Headers:**
```http
Content-Type: application/json
Authorization: Bearer {API_KEY}
```

**Body:**
```json
{
  "location": "Hanoi, Vietnam",
  "datetime": "2024-01-15T14:00:00Z",
  "include_metadata": true,
  "include_confidence": true,
  "hours_ahead": 24
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `location` | string | No | Location name (default: "Hanoi, Vietnam") |
| `datetime` | string | No | Reference datetime (ISO 8601). Default: now |
| `include_metadata` | boolean | No | Include model performance metrics |
| `include_confidence` | boolean | No | Include prediction intervals |
| `hours_ahead` | integer | No | Number of hours to forecast (1-24, default: 24) |

#### Response

**Success (200 OK):**
```json
{
  "status": "success",
  "forecast_type": "hourly",
  "location": "Hanoi, Vietnam",
  "reference_datetime": "2024-01-15T14:00:00Z",
  "generated_at": "2024-01-15T14:05:00Z",
  "model_version": "v1.0.0-hourly",
  "predictions": [
    {
      "target": "h+1",
      "forecast_datetime": "2024-01-15T15:00:00Z",
      "temperature": 25.2,
      "unit": "celsius",
      "confidence_interval": {
        "lower": 24.5,
        "upper": 25.9,
        "confidence_level": 0.95
      }
    },
    {
      "target": "h+2",
      "forecast_datetime": "2024-01-15T16:00:00Z",
      "temperature": 25.8,
      "unit": "celsius",
      "confidence_interval": {
        "lower": 25.0,
        "upper": 26.6,
        "confidence_level": 0.95
      }
    },
    // ... h+3 to h+24
  ],
  "metadata": {
    "total_predictions": 24,
    "average_confidence": 0.89,
    "data_requirements": {
      "minimum_historical_hours": 168,
      "features_used": 180,
      "latest_data_datetime": "2024-01-15T14:00:00Z"
    }
  }
}
```

#### Implementation Module

**File:** `src/hourly_forecast_model/infer_hourly.py` (to be created)

**Preprocessing:** `preprocessed_hourly.py` (existing)

**Core Functions:**
```python
from src.hourly_forecast_model.infer_hourly import HourlyForecaster

# Initialize forecaster (loads 24 hourly models)
forecaster = HourlyForecaster()

# Get predictions
result = forecaster.predict_hourly(
    data_path="dataset/hn_hourly.csv",
    reference_datetime="2024-01-15T14:00:00",
    hours_ahead=24
)

# Returns hourly predictions for next 24 hours
```

**Note:** This module follows the same architecture as daily forecasting but uses:
- `data_processing_hourly/` for processed features
- `trained_models_hourly/` for model storage
- 24 separate models (h+1 to h+24)
- Hourly lag features (1, 3, 6, 12, 24 hours)
- Hourly rolling windows (6h, 12h, 24h, 168h)

---

### 3. Model Health Check

**Endpoint:** `GET /health`

**Description:** Check API and model health status.

#### Request

```http
GET /api/v1/health
```

#### Response

**Success (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T08:30:00Z",
  "version": "1.0.0",
  "models": {
    "daily": {
      "status": "loaded",
      "count": 5,
      "models": ["t+1", "t+2", "t+3", "t+4", "t+5"],
      "last_trained": "2024-01-10"
    },
    "hourly": {
      "status": "loaded",
      "count": 24,
      "models": ["h+1", "h+2", ..., "h+24"],
      "last_trained": "2024-01-10"
    }
  },
  "data": {
    "daily_records": 5840,
    "hourly_records": 140160,
    "latest_date": "2024-01-15"
  },
  "system": {
    "cpu_usage": 15.2,
    "memory_usage": 45.8,
    "disk_usage": 22.1
  }
}
```

**Unhealthy (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "timestamp": "2024-01-15T08:30:00Z",
  "errors": [
    "Daily model t+3 not found",
    "Preprocessing pipeline missing for t_5",
    "Insufficient historical data (only 50 days, need 84)"
  ]
}
```

---

### 4. Model Metadata

**Endpoint:** `GET /models/metadata`

**Description:** Get detailed information about trained models and their performance.

#### Request

```http
GET /api/v1/models/metadata?model_type=daily
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_type` | string | No | Filter by type: "daily", "hourly", or "all" (default) |

#### Response

**Success (200 OK):**
```json
{
  "status": "success",
  "timestamp": "2024-01-15T08:30:00Z",
  "models": {
    "daily": {
      "model_count": 5,
      "architecture": "Per-target ensemble",
      "training_date": "2024-01-10",
      "training_samples": 4095,
      "test_samples": 585,
      "models": [
        {
          "target": "t+1",
          "model_type": "XGBoost",
          "features": 156,
          "hyperparameters": {
            "n_estimators": 800,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8
          },
          "performance": {
            "test_rmse": 1.518,
            "test_mae": 1.167,
            "test_r2": 0.902,
            "test_mape": 4.944
          },
          "feature_selection": "SHORT-TERM (mutual_info + lasso + random_forest + xgboost ensemble)",
          "model_path": "trained_models/model_t+1.joblib",
          "model_size_mb": 12.4
        },
        {
          "target": "t+2",
          "model_type": "XGBoost",
          "features": 145,
          "hyperparameters": {
            "n_estimators": 750,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85
          },
          "performance": {
            "test_rmse": 1.982,
            "test_mae": 1.546,
            "test_r2": 0.833,
            "test_mape": 6.658
          },
          "feature_selection": "COMBINED (short + long term features)",
          "model_path": "trained_models/model_t+2.joblib",
          "model_size_mb": 11.8
        },
        // ... t+3, t+4, t+5
      ]
    },
    "hourly": {
      "model_count": 24,
      "architecture": "Per-target multi-output",
      "training_date": "2024-01-10",
      "training_samples": 98112,
      "test_samples": 14016,
      "models": [
        {
          "target": "h+1",
          "model_type": "RandomForest",
          "features": 180,
          "performance": {
            "test_rmse": 0.85,
            "test_mae": 0.62,
            "test_r2": 0.94
          },
          "model_path": "trained_models_hourly/model_h+1.joblib",
          "model_size_mb": 8.2
        }
        // ... h+2 to h+24
      ]
    }
  },
  "preprocessing": {
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
      "total_features_generated": 200,
      "pipeline_path": "processed_data/pipelines/"
    },
    "hourly_pipeline": {
      "steps": [
        "temporal_features",
        "cyclical_encoding",
        "lag_features (1, 3, 6, 12, 24 hours)",
        "rolling_windows (6h, 12h, 24h, 168h)",
        "standard_scaler"
      ],
      "total_features_generated": 220,
      "pipeline_path": "data_processing_hourly/"
    }
  },
  "tuning": {
    "method": "Two-stage Optuna",
    "stage1_trials": 40,
    "stage2_trials_per_target": 100,
    "total_trials": 540,
    "optimization_metric": "RMSE",
    "tuning_date": "2024-01-08"
  }
}
```

---

## Data Models

### Temperature Forecast (Daily)

```typescript
interface DailyForecastRequest {
  location?: string;           // Default: "Hanoi, Vietnam"
  date?: string;               // ISO 8601 date (YYYY-MM-DD)
  include_metadata?: boolean;  // Default: false
  include_confidence?: boolean; // Default: false
}

interface DailyForecastResponse {
  status: "success" | "error";
  forecast_type: "daily";
  location: string;
  reference_date: string;
  generated_at: string;        // ISO 8601 datetime
  model_version: string;
  predictions: DailyPrediction[];
  metadata?: ForecastMetadata;
  error?: ErrorDetail;
}

interface DailyPrediction {
  target: "t+1" | "t+2" | "t+3" | "t+4" | "t+5";
  forecast_date: string;       // ISO 8601 date
  temperature: number;         // Celsius
  unit: "celsius";
  confidence_interval?: {
    lower: number;
    upper: number;
    confidence_level: number;  // 0.95 for 95%
  };
  model_performance?: {
    test_rmse: number;
    test_mae: number;
    test_r2: number;
    test_mape: number;
  };
}
```

### Temperature Forecast (Hourly)

```typescript
interface HourlyForecastRequest {
  location?: string;
  datetime?: string;           // ISO 8601 datetime
  include_metadata?: boolean;
  include_confidence?: boolean;
  hours_ahead?: number;        // 1-24
}

interface HourlyForecastResponse {
  status: "success" | "error";
  forecast_type: "hourly";
  location: string;
  reference_datetime: string;
  generated_at: string;
  model_version: string;
  predictions: HourlyPrediction[];
  metadata?: ForecastMetadata;
  error?: ErrorDetail;
}

interface HourlyPrediction {
  target: string;              // "h+1", "h+2", ..., "h+24"
  forecast_datetime: string;   // ISO 8601 datetime
  temperature: number;
  unit: "celsius";
  confidence_interval?: {
    lower: number;
    upper: number;
    confidence_level: number;
  };
}
```

### Common Types

```typescript
interface ForecastMetadata {
  total_predictions: number;
  average_confidence: number;
  data_requirements: {
    minimum_historical_days?: number;
    minimum_historical_hours?: number;
    features_used: number;
    latest_data_date: string;
  };
}

interface ErrorDetail {
  code: string;
  message: string;
  details?: any;
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid input parameters |
| 401 | Unauthorized | Missing or invalid API key |
| 404 | Not Found | Endpoint not found |
| 422 | Unprocessable Entity | Valid JSON but invalid business logic |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Models not loaded or unhealthy |

### Error Response Format

```json
{
  "status": "error",
  "error": {
    "code": "INSUFFICIENT_DATA",
    "message": "Not enough historical data for prediction",
    "details": {
      "required_days": 84,
      "available_days": 50,
      "missing_days": 34
    }
  },
  "timestamp": "2024-01-15T08:30:00Z"
}
```

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_DATE` | Date format invalid or out of range | 400 |
| `INVALID_LOCATION` | Location not supported | 400 |
| `INSUFFICIENT_DATA` | Not enough historical data (< 84 days) | 422 |
| `MODEL_NOT_FOUND` | Required model file missing | 503 |
| `PIPELINE_ERROR` | Feature engineering pipeline failed | 500 |
| `PREDICTION_ERROR` | Model prediction failed | 500 |
| `DATA_LOAD_ERROR` | Failed to load weather data | 500 |
| `UNAUTHORIZED` | Invalid or missing API key | 401 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |

---

## Implementation Guide

### FastAPI Implementation

**File:** `api/main.py`

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date
import json

from src.daily_forecast_model.infer import WeatherForecaster
# from src.hourly_forecast_model.infer_hourly import HourlyForecaster

app = FastAPI(
    title="Weather Forecast API",
    description="ML-based temperature forecasting for Hanoi",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
daily_forecaster = None
# hourly_forecaster = None

@app.on_event("startup")
async def startup_event():
    """Load models when API starts"""
    global daily_forecaster
    try:
        daily_forecaster = WeatherForecaster()
        print("✅ Daily forecaster loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load daily forecaster: {e}")
    
    # TODO: Load hourly forecaster
    # global hourly_forecaster
    # hourly_forecaster = HourlyForecaster()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class DailyForecastRequest(BaseModel):
    location: Optional[str] = "Hanoi, Vietnam"
    date: Optional[str] = None  # ISO 8601
    include_metadata: Optional[bool] = False
    include_confidence: Optional[bool] = False

class HourlyForecastRequest(BaseModel):
    location: Optional[str] = "Hanoi, Vietnam"
    datetime: Optional[str] = None  # ISO 8601
    include_metadata: Optional[bool] = False
    include_confidence: Optional[bool] = False
    hours_ahead: Optional[int] = Field(24, ge=1, le=24)

class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    confidence_level: float = 0.95

class ModelPerformance(BaseModel):
    test_rmse: float
    test_mae: float
    test_r2: float
    test_mape: float

class DailyPrediction(BaseModel):
    target: str
    forecast_date: str
    temperature: float
    unit: str = "celsius"
    confidence_interval: Optional[ConfidenceInterval] = None
    model_performance: Optional[ModelPerformance] = None

class HourlyPrediction(BaseModel):
    target: str
    forecast_datetime: str
    temperature: float
    unit: str = "celsius"
    confidence_interval: Optional[ConfidenceInterval] = None

class ForecastMetadata(BaseModel):
    total_predictions: int
    average_confidence: Optional[float] = None
    data_requirements: dict

class DailyForecastResponse(BaseModel):
    status: str
    forecast_type: str = "daily"
    location: str
    reference_date: str
    generated_at: str
    model_version: str
    predictions: List[DailyPrediction]
    metadata: Optional[ForecastMetadata] = None

class HourlyForecastResponse(BaseModel):
    status: str
    forecast_type: str = "hourly"
    location: str
    reference_datetime: str
    generated_at: str
    model_version: str
    predictions: List[HourlyPrediction]
    metadata: Optional[ForecastMetadata] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/api/v1/health")
async def health_check():
    """Check API and model health"""
    models_status = {
        "daily": {
            "status": "loaded" if daily_forecaster else "not_loaded",
            "count": len(daily_forecaster.models) if daily_forecaster else 0
        }
    }
    
    status = "healthy" if daily_forecaster else "unhealthy"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "models": models_status
    }


@app.post("/api/v1/forecast/daily", response_model=DailyForecastResponse)
async def forecast_daily(request: DailyForecastRequest):
    """
    Get 5-day temperature forecast (t+1 to t+5)
    
    Requires:
    - Minimum 84 days of historical data
    - Daily weather dataset at dataset/hn_daily.csv
    
    Returns:
    - Predictions for next 5 days
    - Optional: confidence intervals and model performance
    """
    if not daily_forecaster:
        raise HTTPException(status_code=503, detail="Daily forecaster not loaded")
    
    try:
        # Use provided date or today
        reference_date = request.date if request.date else str(date.today())
        
        # Get predictions from forecaster
        result = daily_forecaster.predict_with_metadata(
            data_path="dataset/hn_daily.csv",
            reference_date=reference_date
        )
        
        # Load evaluation metrics for performance data
        with open('src/daily_forecast_model/evaluate_results/evaluation_results.json', 'r') as f:
            eval_metrics = json.load(f)
        
        # Build predictions list
        predictions = []
        for target in ['t+1', 't+2', 't+3', 't+4', 't+5']:
            pred = DailyPrediction(
                target=target,
                forecast_date=result['dates'][target],
                temperature=round(result['predictions'][target], 1),
                unit="celsius"
            )
            
            # Add confidence interval if requested
            if request.include_confidence:
                # Use RMSE to estimate 95% CI (±1.96 * RMSE)
                rmse = eval_metrics[target]['metrics']['RMSE']
                temp = pred.temperature
                pred.confidence_interval = ConfidenceInterval(
                    lower=round(temp - 1.96 * rmse, 1),
                    upper=round(temp + 1.96 * rmse, 1),
                    confidence_level=0.95
                )
            
            # Add performance metrics if requested
            if request.include_metadata:
                metrics = eval_metrics[target]['metrics']
                pred.model_performance = ModelPerformance(
                    test_rmse=round(metrics['RMSE'], 2),
                    test_mae=round(metrics['MAE'], 2),
                    test_r2=round(metrics['R2'], 3),
                    test_mape=round(metrics['MAPE'], 2)
                )
            
            predictions.append(pred)
        
        # Build metadata if requested
        metadata = None
        if request.include_metadata:
            avg_r2 = sum(eval_metrics[t]['metrics']['R2'] for t in ['t+1', 't+2', 't+3', 't+4', 't+5']) / 5
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
        raise HTTPException(status_code=500, detail=f"Data file not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Insufficient data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/v1/forecast/hourly", response_model=HourlyForecastResponse)
async def forecast_hourly(request: HourlyForecastRequest):
    """
    Get 24-hour temperature forecast (h+1 to h+24)
    
    Requires:
    - Minimum 168 hours (7 days) of historical data
    - Hourly weather dataset at dataset/hn_hourly.csv
    
    Returns:
    - Predictions for next 24 hours
    - Optional: confidence intervals
    """
    # TODO: Implement hourly forecasting
    raise HTTPException(
        status_code=501,
        detail="Hourly forecasting not yet implemented. Use /api/v1/forecast/daily for now."
    )
    
    # Implementation will be similar to daily, but using HourlyForecaster


@app.get("/api/v1/models/metadata")
async def get_models_metadata(model_type: Optional[str] = "all"):
    """Get detailed model metadata and performance"""
    
    if not daily_forecaster:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Load evaluation results
        with open('src/daily_forecast_model/evaluate_results/evaluation_results.json', 'r') as f:
            eval_metrics = json.load(f)
        
        # Build metadata
        daily_models = []
        for target in ['t+1', 't+2', 't+3', 't+4', 't+5']:
            metrics = eval_metrics[target]['metrics']
            model_info = {
                "target": target,
                "model_type": "XGBoost",  # Could be read from model
                "features": len(daily_forecaster.feature_names.get(target, [])),
                "performance": {
                    "test_rmse": round(metrics['RMSE'], 3),
                    "test_mae": round(metrics['MAE'], 3),
                    "test_r2": round(metrics['R2'], 3),
                    "test_mape": round(metrics['MAPE'], 3)
                },
                "test_samples": eval_metrics[target]['n_samples']
            }
            daily_models.append(model_info)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "models": {
                "daily": {
                    "model_count": 5,
                    "architecture": "Per-target ensemble",
                    "models": daily_models
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metadata: {str(e)}")


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Running the API

```bash
# Install dependencies
pip install fastapi uvicorn pydantic

# Run development server
python api/main.py

# Or using uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Daily forecast (basic)
curl -X POST http://localhost:8000/api/v1/forecast/daily \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Hanoi, Vietnam",
    "date": "2024-01-15"
  }'

# Daily forecast (with metadata and confidence)
curl -X POST http://localhost:8000/api/v1/forecast/daily \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Hanoi, Vietnam",
    "date": "2024-01-15",
    "include_metadata": true,
    "include_confidence": true
  }'

# Model metadata
curl http://localhost:8000/api/v1/models/metadata?model_type=daily
```

---

## Security & Performance

### Security Recommendations

1. **API Key Authentication**
   ```python
   from fastapi import Security, HTTPException
   from fastapi.security.api_key import APIKeyHeader
   
   API_KEY_NAME = "X-API-Key"
   api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
   
   async def get_api_key(api_key: str = Security(api_key_header)):
       if api_key != os.getenv("API_KEY"):
           raise HTTPException(status_code=401, detail="Invalid API Key")
       return api_key
   
   @app.post("/api/v1/forecast/daily", dependencies=[Depends(get_api_key)])
   async def forecast_daily(request: DailyForecastRequest):
       # ...
   ```

2. **Rate Limiting**
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
   
   @app.post("/api/v1/forecast/daily")
   @limiter.limit("100/hour")
   async def forecast_daily(request: Request, data: DailyForecastRequest):
       # ...
   ```

3. **HTTPS Only**
   - Use reverse proxy (nginx/Caddy) with SSL certificates
   - Enforce HTTPS in production

4. **Input Validation**
   - Pydantic models handle basic validation
   - Add custom validators for dates, locations

### Performance Optimizations

1. **Model Caching**
   - Load models once at startup (✅ already implemented)
   - Use global variables to avoid reloading

2. **Response Caching**
   ```python
   from fastapi_cache import FastAPICache
   from fastapi_cache.backends.redis import RedisBackend
   from fastapi_cache.decorator import cache
   
   @app.post("/api/v1/forecast/daily")
   @cache(expire=3600)  # Cache for 1 hour
   async def forecast_daily(request: DailyForecastRequest):
       # ...
   ```

3. **Async Data Loading**
   ```python
   import aiofiles
   import pandas as pd
   
   async def load_data_async(path: str):
       # Use async file operations for large datasets
       pass
   ```

4. **Batch Predictions**
   - Add batch endpoint for multiple dates/locations
   ```python
   @app.post("/api/v1/forecast/daily/batch")
   async def forecast_daily_batch(dates: List[str]):
       # Return predictions for multiple dates
       pass
   ```

5. **Monitoring**
   - Add Prometheus metrics
   - Track prediction latency, error rates
   ```python
   from prometheus_fastapi_instrumentator import Instrumentator
   
   Instrumentator().instrument(app).expose(app)
   ```

---

## Next Steps

### Immediate (Week 1)

1. ✅ Create `api/main.py` with daily forecast endpoint
2. ✅ Test basic functionality with existing models
3. ✅ Add error handling and validation
4. ✅ Deploy locally and test with curl/Postman

### Short-term (Week 2-3)

1. 🔨 Implement hourly forecasting module (`src/hourly_forecast_model/infer_hourly.py`)
2. 🔨 Train hourly models (h+1 to h+24) using `preprocessed_hourly.py`
3. 🔨 Add hourly forecast endpoint
4. 🔨 Add API authentication and rate limiting

### Medium-term (Month 1-2)

1. 📝 Integrate with Streamlit interface (`interface/main.py`)
2. 📝 Add model monitoring and drift detection endpoints
3. 📝 Implement automated retraining triggers
4. 📝 Add batch prediction endpoints
5. 📝 Create API documentation (Swagger/OpenAPI)

### Long-term (Month 3+)

1. 🚀 Deploy to production (Docker + Kubernetes)
2. 🚀 Add multi-location support
3. 🚀 Implement model A/B testing
4. 🚀 Add weather alerts and anomaly detection
5. 🚀 Create mobile-friendly endpoints

---

## Appendix

### Directory Structure

```
machine_learning_lab/
├── api/
│   ├── main.py                    # FastAPI application
│   ├── requirements.txt           # API dependencies
│   └── Dockerfile                 # Docker configuration
├── src/
│   ├── daily_forecast_model/
│   │   ├── infer.py              # Daily inference ✅
│   │   ├── process.py            # Daily preprocessing ✅
│   │   ├── train.py              # Daily training ✅
│   │   ├── evaluate.py           # Daily evaluation ✅
│   │   └── helper.py             # Shared utilities ✅
│   └── hourly_forecast_model/    # To be created
│       ├── infer_hourly.py       # Hourly inference 🔨
│       ├── train_hourly.py       # Hourly training 🔨
│       └── evaluate_hourly.py    # Hourly evaluation 🔨
├── trained_models/               # Daily models ✅
│   ├── model_t+1.joblib
│   ├── model_t+2.joblib
│   └── ...
├── trained_models_hourly/        # Hourly models 🔨
├── dataset/
│   ├── hn_daily.csv              # Daily data ✅
│   └── hn_hourly.csv             # Hourly data ✅
├── preprocessed_hourly.py        # Hourly preprocessing ✅
└── interface/
    └── main.py                    # Streamlit UI ✅
```

### References

- **Daily Forecasting:** `src/daily_forecast_model/infer.py`
- **Hourly Preprocessing:** `preprocessed_hourly.py`
- **Model Evaluation:** `src/daily_forecast_model/evaluate_results/evaluation_results.json`
- **Streamlit Interface:** `interface/main.py`

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-16  
**Author:** ML Team  
**Status:** Ready for Implementation
