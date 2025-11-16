"""
Production inference module for weather forecasting.

This module provides a simple interface for making 5-day temperature predictions
using trained per-target models. Designed for web deployment and real-time inference.
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta

# Import configuration
from src.daily_forecast_model.helper import (
    PROJECT_ROOT, N_STEPS_AHEAD, MODELS_DIR
)


class WeatherForecaster:
    """
    Production forecaster for 5-day temperature predictions.
    
    This class encapsulates the entire inference pipeline:
    1. Load trained per-target models
    2. Load per-target preprocessing pipelines
    3. Apply target-specific feature engineering
    4. Generate predictions for each horizon
    5. Return formatted results
    """
    
    def __init__(self):
        """Initialize the forecaster by loading all trained models and pipelines."""
        self.models = {}
        self.preprocessors = {}
        self.feature_names = {}
        
        self._load_models()
        self._load_preprocessors()
        
        print(f"\n{'='*70}")
        print("✓ WeatherForecaster initialized successfully")
        print(f"  Loaded {len(self.models)} models for {N_STEPS_AHEAD}-day forecast")
        print(f"  Models directory: {MODELS_DIR}")
        print(f"{'='*70}\n")
    
    def _load_models(self):
        """Load all trained per-target models."""
        print("Loading trained models...")
        
        for day_step in range(1, N_STEPS_AHEAD + 1):
            target_name = f"t+{day_step}"
            model_path = os.path.join(MODELS_DIR, f"model_{target_name}.joblib")
            
            try:
                self.models[target_name] = joblib.load(model_path)
                print(f"  ✓ {target_name}: {model_path}")
            except FileNotFoundError:
                print(f"  ✗ {target_name}: Model not found at {model_path}")
                raise FileNotFoundError(
                    f"Model for {target_name} not found. Please train models first:\n"
                    f"  python src/daily_forecast_model/train.py"
                )
        
        print(f"✓ All {len(self.models)} models loaded successfully\n")
    
    def _load_preprocessors(self):
        """Load all per-target preprocessing pipelines."""
        print("Loading preprocessing pipelines...")
        
        for day_step in range(1, N_STEPS_AHEAD + 1):
            target_name = f"t+{day_step}"
            day_str = f"t_{day_step}"
            data_dir = os.path.join(PROJECT_ROOT, 'processed_data', f'target_{day_str}')
            preprocessor_path = os.path.join(data_dir, f'preprocessor_target_temp_t+{day_step}.joblib')
            
            try:
                self.preprocessors[target_name] = joblib.load(preprocessor_path)
                print(f"  ✓ {target_name}: {preprocessor_path}")
                
                # Try to load feature names
                feature_names_path = os.path.join(data_dir, f'X_train_t{day_step}.csv')
                if os.path.exists(feature_names_path):
                    feature_df = pd.read_csv(feature_names_path, index_col=0, nrows=0)
                    self.feature_names[target_name] = feature_df.columns.tolist()
                
            except FileNotFoundError:
                print(f"  ✗ {target_name}: Preprocessor not found at {preprocessor_path}")
                raise FileNotFoundError(
                    f"Preprocessor for {target_name} not found. Please run preprocessing first:\n"
                    f"  python src/daily_forecast_model/process.py"
                )
        
        print(f"✓ All {len(self.preprocessors)} preprocessors loaded successfully\n")
    
    def prepare_features(self, raw_data, target_name):
        """
        Apply feature engineering for a specific target.
        
        Args:
            raw_data (pd.DataFrame): Raw weather data with required columns
            target_name (str): Target horizon (e.g., 't+1', 't+2')
        
        Returns:
            pd.DataFrame: Transformed features ready for prediction
        """
        # Apply preprocessing pipeline
        preprocessor = self.preprocessors[target_name]
        transformed_data = preprocessor.transform(raw_data)
        
        # Convert to DataFrame with correct feature names
        if target_name in self.feature_names:
            transformed_df = pd.DataFrame(
                transformed_data,
                columns=self.feature_names[target_name],
                index=raw_data.index
            )
        else:
            transformed_df = pd.DataFrame(transformed_data, index=raw_data.index)
        
        return transformed_df
    
    def predict(self, raw_data):
        """
        Generate 5-day temperature forecast.
        
        Args:
            raw_data (pd.DataFrame): Raw weather data with datetime index
                Required columns depend on feature engineering pipeline
        
        Returns:
            pd.DataFrame: Predictions for all horizons with columns [t+1, t+2, ..., t+5]
        """
        predictions = {}
        
        for day_step in range(1, N_STEPS_AHEAD + 1):
            target_name = f"t+{day_step}"
            
            # Prepare features for this target
            X = self.prepare_features(raw_data, target_name)
            
            # Make prediction
            model = self.models[target_name]
            y_pred = model.predict(X)
            
            predictions[target_name] = y_pred
        
        # Combine into DataFrame
        predictions_df = pd.DataFrame(predictions, index=raw_data.index)
        
        return predictions_df
    
    def predict_single(self, raw_data):
        """
        Generate forecast for a single time point (most recent in dataset).
        
        Args:
            raw_data (pd.DataFrame): Raw weather data (uses last row)
        
        Returns:
            dict: Single forecast {target: temperature}
        """
        # Use only the last row
        last_data = raw_data.iloc[[-1]]
        
        # Get predictions
        predictions_df = self.predict(last_data)
        
        # Return as dictionary
        forecast = predictions_df.iloc[0].to_dict()
        
        return forecast
    
    def predict_with_metadata(self, raw_data):
        """
        Generate forecast with additional metadata.
        
        Args:
            raw_data (pd.DataFrame): Raw weather data
        
        Returns:
            dict: {
                'predictions': DataFrame of predictions,
                'forecast_dates': list of forecast dates,
                'base_date': base date for forecast,
                'model_info': model information
            }
        """
        # Get predictions
        predictions_df = self.predict(raw_data)
        
        # Get base date (last date in input)
        base_date = raw_data.index[-1]
        
        # Calculate forecast dates
        forecast_dates = [base_date + timedelta(days=i) for i in range(1, N_STEPS_AHEAD + 1)]
        
        # Get model info
        model_info = {
            'n_models': len(self.models),
            'horizons': list(self.models.keys()),
            'models_dir': MODELS_DIR
        }
        
        return {
            'predictions': predictions_df,
            'forecast_dates': forecast_dates,
            'base_date': base_date,
            'model_info': model_info
        }


def demo_prediction():
    """Demonstration of how to use the WeatherForecaster."""
    print("\n" + "="*70)
    print("WEATHER FORECASTER DEMO")
    print("="*70 + "\n")
    
    # Initialize forecaster
    forecaster = WeatherForecaster()
    
    # Load sample data (using test data as example)
    print("Loading sample data...")
    sample_data_path = os.path.join(PROJECT_ROOT, 'processed_data', 'target_t_1', 'X_test_t1.csv')
    
    if not os.path.exists(sample_data_path):
        print(f"✗ Sample data not found at: {sample_data_path}")
        print("  Please run preprocessing first to generate test data.")
        return
    
    sample_data = pd.read_csv(sample_data_path, index_col=0, nrows=5)
    print(f"✓ Loaded {len(sample_data)} sample records\n")
    
    # Make predictions
    print("Generating predictions...")
    predictions = forecaster.predict(sample_data)
    
    print("\nPredictions:")
    print(predictions)
    
    # Single prediction example
    print("\n" + "="*70)
    print("Single Record Prediction Example:")
    print("="*70)
    
    single_forecast = forecaster.predict_single(sample_data)
    print("\nForecast:")
    for target, temp in single_forecast.items():
        print(f"  {target}: {temp:.2f}°C")
    
    # Prediction with metadata
    print("\n" + "="*70)
    print("Prediction with Metadata Example:")
    print("="*70)
    
    result = forecaster.predict_with_metadata(sample_data)
    print(f"\nBase date: {result['base_date']}")
    print(f"Forecast dates: {result['forecast_dates'][0]} to {result['forecast_dates'][-1]}")
    print(f"Models loaded: {result['model_info']['n_models']}")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    # Run demonstration
    demo_prediction()
