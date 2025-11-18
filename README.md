# 🌦️ Hanoi Weather Forecasting System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%20|%203.13-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Advanced Machine Learning System for Accurate Temperature Predictions**

[🚀 Live Demo](https://huggingface.co/spaces/your-space) • [📖 Documentation](./API_DESIGN.md) • [📊 Analysis](./hanoi_weather_forecast.ipynb)

</div>

---

## 📋 Project Information

```yaml
Group: 1
Class: DSEB 65B
Instructor: Trinh Tuan Phong
Members:
  - Thieu Dieu Thuy (Leader)
  - Nguyen Tran Tuan Kiet
  - Ly Thanh Long
  - Nguyen Thanh Mo
  - Ngo Hoang Phuc
```

---

## 🎯 Project Overview

A production-ready machine learning system that provides **highly accurate temperature forecasting** for Hanoi, Vietnam using state-of-the-art ensemble methods and advanced feature engineering.

### 🌟 Key Features

- **🔮 Dual Forecasting Models**
  - **Daily Forecasts:** 5-day ahead predictions (t+1 to t+5)
  - **Hourly Forecasts:** 24-hour ahead predictions (t+1 to t+24)

- **🤖 Advanced ML Pipeline**
  - Per-horizon model optimization (29 specialized models)
  - Ensemble of CatBoost, LightGBM, XGBoost
  - Automated hyperparameter tuning with Optuna (2,400+ trials)

- **📊 Impressive Performance Metrics**
  - **Hourly Model:** RMSE 1.61°C, MAE 1.19°C, R² 0.907
  - **Daily Model:** High accuracy across all forecast horizons
  - Real-time confidence intervals

- **🚀 Production-Ready Infrastructure**
  - RESTful API with FastAPI
  - Interactive Streamlit dashboard
  - Docker containerization
  - Hugging Face Spaces deployment
  - ClearML monitoring integration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend Layer                               │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  Streamlit UI    │────────▶│   FastAPI Docs   │             │
│  │  (Port 8501)     │         │   (/docs)        │             │
│  └──────────────────┘         └──────────────────┘             │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                      API Layer                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FastAPI Application (api/api.py)                        │  │
│  │  • POST /api/v1/forecast/daily                           │  │
│  │  • POST /api/v1/forecast/hourly                          │  │
│  │  • GET  /api/v1/health                                   │  │
│  │  • GET  /api/v1/models/metadata                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   Inference Layer                                │
│  ┌──────────────────────┐    ┌──────────────────────┐          │
│  │ WeatherForecaster    │    │ HourlyForecaster     │          │
│  │ (5 daily models)     │    │ (24 hourly models)   │          │
│  └──────────────────────┘    └──────────────────────┘          │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│              Feature Engineering Pipeline                        │
│  • Temporal Features (cyclic encoding)                          │
│  • Lag Features (1, 3, 7 days/hours)                            │
│  • Rolling Windows (7, 14, 30, 84 periods)                      │
│  • Interaction Features (temp × humidity, etc.)                 │
│  • StandardScaler Normalization                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  Trained Models (29 total)                       │
│  Daily: CatBoost/LightGBM/XGBoost (5 horizons)                  │
│  Hourly: CatBoost 17, LightGBM 5, XGBoost 2 (24 horizons)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

### Hourly Forecasting (24 hours ahead)

| Metric | Value | Description |
|--------|-------|-------------|
| **RMSE** | 1.61°C | Root Mean Squared Error |
| **MAE** | 1.19°C | Mean Absolute Error |
| **MAPE** | 4.85% | Mean Absolute Percentage Error |
| **R²** | 0.907 | Coefficient of Determination |

**Model Distribution:**
- CatBoost: 17/24 horizons (70.8%)
- LightGBM: 5/24 horizons (20.8%)
- XGBoost: 2/24 horizons (8.3%)

### Performance by Forecast Horizon

- **t+1h:** MAE 0.55°C, R² 0.978 (Excellent)
- **t+6h:** MAE 1.00°C, R² 0.934 (Very Good)
- **t+12h:** MAE 1.29°C, R² 0.898 (Good)
- **t+24h:** MAE 1.60°C, R² 0.851 (Good)

---

## 🚀 Quick Start

### Prerequisites

- **Python:** 3.11+ (Python 3.11.13 recommended for deployment)
- **Git:** For version control
- **Conda/Miniconda:** For environment management
- **Docker (Optional):** For containerized deployment

### 1️⃣ Clone Repository

```bash
git clone https://github.com/tlong-ds/machine_learning_lab.git
cd machine_learning_lab
```

### 2️⃣ Create Python Environment

```bash
# Create conda environment
conda create -n weather-forecast python=3.11.13 -y
conda activate weather-forecast

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Run API Server

```bash
# Start FastAPI backend
uvicorn api.api:app --reload --host 0.0.0.0 --port 8000

# API will be available at:
# - http://localhost:8000
# - Docs: http://localhost:8000/docs
# - Redoc: http://localhost:8000/redoc
```

### 4️⃣ Launch Streamlit Interface

```bash
# In a new terminal
streamlit run interface/main.py

# Dashboard will open at http://localhost:8501
```

### 5️⃣ Docker Deployment (Production)

```bash
# Build Docker image
docker build -t hanoi-weather-forecast .

# Run container
docker run -p 7860:7860 hanoi-weather-forecast

# Access at http://localhost:7860
```

---

## 📁 Project Structure

```
machine_learning_lab/
├── api/
│   └── api.py                          # FastAPI application
├── interface/
│   └── main.py                         # Streamlit dashboard
├── src/
│   ├── daily_forecast_model/
│   │   ├── infer.py                   # Daily forecaster class
│   │   ├── process.py                 # Feature engineering
│   │   └── tune.py                    # Hyperparameter optimization
│   └── hourly_forecast_model/
│       ├── infer.py                   # Hourly forecaster class
│       ├── process.py                 # Hourly feature engineering
│       ├── tune.py                    # Per-horizon optimization
│       └── evaluate_results/
│           └── evaluation_results.json # Performance metrics
├── trained_models/                     # Daily models (5 files)
├── trained_models_hourly/             # Hourly models (24 files)
├── data_processing_hourly/
│   ├── pipelines/                     # Preprocessing pipelines
│   └── preprocessing_pipeline.joblib  # Main pipeline
├── dataset/
│   ├── hn_daily.csv                   # Daily weather data
│   └── hn_hourly.csv                  # Hourly weather data
├── hanoi_weather_forecast.ipynb       # Analysis & documentation
├── API_DESIGN.md                      # API specifications
├── Dockerfile                         # Container configuration
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🔧 Technology Stack

### Machine Learning
- **Models:** CatBoost, LightGBM, XGBoost, RandomForest
- **Optimization:** Optuna (100 trials per horizon)
- **Feature Engineering:** scikit-learn pipelines
- **Serialization:** joblib, ONNX Runtime

### Backend & API
- **Framework:** FastAPI 1.0.0
- **Server:** Uvicorn (ASGI)
- **Data Processing:** pandas, numpy
- **Validation:** Pydantic models

### Frontend & Visualization
- **Dashboard:** Streamlit
- **Plotting:** matplotlib, seaborn, plotly

### Deployment
- **Containerization:** Docker
- **Cloud Platform:** Hugging Face Spaces
- **Monitoring:** ClearML
- **CI/CD:** GitHub Actions (planned)

### Development Tools
- **Environment:** Conda/venv
- **Notebooks:** Jupyter Lab
- **Version Control:** Git

---

## 🌐 API Usage

### Daily Forecast Example

```bash
curl -X POST "http://localhost:8000/api/v1/forecast/daily" \
  -H "Content-Type: application/json" \
  -d '{
    "hours_ahead": 5,
    "include_confidence_intervals": true
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "target_day": 1,
      "date": "2025-11-18",
      "temperature": 24.5,
      "confidence_interval": {
        "lower": 23.2,
        "upper": 25.8
      }
    }
  ],
  "metadata": {
    "model_version": "1.0",
    "generated_at": "2025-11-17T10:30:00Z"
  }
}
```

### Hourly Forecast Example

```bash
curl -X POST "http://localhost:8000/api/v1/forecast/hourly" \
  -H "Content-Type: application/json" \
  -d '{
    "hours_ahead": 24,
    "include_confidence_intervals": true
  }'
```

See [API_DESIGN.md](./API_DESIGN.md) for complete documentation.

---

## 📚 Documentation

- **[API Design Document](./API_DESIGN.md)** - Complete API specifications
- **[Jupyter Notebook Analysis](./hanoi_weather_forecast.ipynb)** - Model development & evaluation
- **[Tutorial: Create Environment](./tutorials/create_environment.md)** - Setup guide
- **[Tutorial: Git Workflow](./tutorials/clone_repo.md)** - Version control basics

---

## 🔬 Model Development

### Feature Engineering Pipeline

Our models use 60+ engineered features:

1. **Temporal Features**
   - Cyclic encoding (month, day, hour)
   - Day length calculation
   - Cyclical wind direction

2. **Lag Features**
   - Temperature lags: 1h, 3h, 7h
   - Humidity lags: 1h, 3h, 7h
   - Pressure lags: 1h, 3h

3. **Rolling Window Statistics**
   - 7-hour rolling mean/std
   - 14-hour rolling mean/std
   - 30-hour rolling mean/std

4. **Interaction Features**
   - Temperature × Humidity
   - Temperature × Wind Speed
   - Pressure × Humidity

### Hyperparameter Optimization

- **Strategy:** Per-horizon optimization (single-stage)
- **Trials:** 100 trials × 29 horizons = 2,900 total
- **Search Space:**
  - CatBoost: depth, learning_rate, l2_leaf_reg, iterations
  - LightGBM: num_leaves, learning_rate, min_child_samples
  - XGBoost: max_depth, learning_rate, min_child_weight

### Training Process

```bash
# Train daily models
python src/daily_forecast_model/tune.py

# Train hourly models
python src/hourly_forecast_model/tune.py

# Evaluate models
python train_and_evaluate_model_hourly.py
```

---

## 🐳 Deployment

### Hugging Face Spaces

1. Push to Hugging Face repository
2. Configure Dockerfile (port 7860)
3. Set environment variables
4. Deploy automatically

### Local Docker

```bash
# Build
docker build -t weather-api:latest .

# Run
docker run -d -p 7860:7860 \
  --name weather-forecast \
  weather-api:latest

# Logs
docker logs -f weather-forecast
```

---

## 📈 Performance Monitoring

The system includes ClearML integration for:
- Model performance tracking
- Inference latency monitoring
- Prediction drift detection
- API usage analytics

---

## 🤝 Contributing

### Workflow

1. **Pull latest changes:** `git pull origin dev`
2. **Create feature branch:** `git checkout -b feature/your-feature`
3. **Make changes and test**
4. **Commit:** `git commit -m "feat: description"`
5. **Push:** `git push origin feature/your-feature`
6. **Create Pull Request**

### Commit Convention
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Testing
- `chore:` Maintenance

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Visual Crossing Weather API** for historical data
- **Optuna Team** for hyperparameter optimization framework
- **CatBoost, LightGBM, XGBoost** communities
- **FastAPI & Streamlit** teams

---

## 📞 Contact

**Project Maintainer:** Thieu Dieu Thuy (Team Leader)

For questions or support:
- 📧 Email: [your-email]
- 🐙 GitHub Issues: [Create an issue](https://github.com/tlong-ds/machine_learning_lab/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/tlong-ds/machine_learning_lab/discussions)

---

<div align="center">

**⭐ If you find this project useful, please star it! ⭐**

Made with ❤️ by DSEB 65B Group 1

</div>



