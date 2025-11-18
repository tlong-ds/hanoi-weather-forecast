import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta

# Import configuration
from src.daily_forecast_model.helper import (
    PROJECT_ROOT, MODELS_DIR, N_STEPS_AHEAD
)

# Import feature engineering functions and OutlierClipper for unpickling
from src.daily_forecast_model.process import (
    OutlierClipper,
    create_lag_features,
    create_rolling_features,
    create_cyclical_wind_direction,
    create_temporal_features,
    create_day_length_feature,
    create_interaction_features,
    LAG_PERIODS,
    ROLLING_WINDOWS
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
            
            # Preprocessors are stored in processed_data/pipelines/
            preprocessor_path = os.path.join(PROJECT_ROOT, 'processed_data', 'pipelines', f'preprocessor_{day_str}.joblib')
            
            try:
                self.preprocessors[target_name] = joblib.load(preprocessor_path)
                print(f"  ✓ {target_name}: {preprocessor_path}")
                
                # Try to load feature names from training data
                data_dir = os.path.join(PROJECT_ROOT, 'processed_data', f'target_{day_str}')
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
        
        This replicates the preprocessing pipeline:
        1. Create temporal features (day, week, month, sin/cos encodings)
        2. Create day length feature
        3. Create cyclical wind direction (sin/cos)
        4. Create interaction features (daylength_uv, windspeed_sq, etc.)
        5. Create lag features (temp_lag1, temp_lag3, temp_lag7, etc.)
        6. Create rolling window features
        7. Apply scaling/normalization via preprocessor
        
        Args:
            raw_data (pd.DataFrame): Raw weather data with datetime index
            target_name (str): Target horizon (e.g., 't+1', 't+2')
        
        Returns:
            pd.DataFrame: Transformed features ready for prediction
        """
        # Make a copy to avoid modifying original
        df = raw_data.copy()
        
        # Step 1: Create temporal features (includes sin/cos cyclical encodings)
        df = create_temporal_features(df)
        
        # Step 2: Create day length feature
        df = create_day_length_feature(df)
        
        # Step 3: Create cyclical wind direction
        df = create_cyclical_wind_direction(df)
        
        # Step 4: Create interaction features (daylength_uv, etc.)
        df = create_interaction_features(df)
        
        # Step 5: Create lag features
        df = create_lag_features(df, lag_config=LAG_PERIODS)
        
        # Step 6: Create rolling window features
        df = create_rolling_features(df, windows=ROLLING_WINDOWS)
        
        # Drop rows with NaN (due to lag/rolling window creation)
        n_input_rows = len(raw_data)
        df = df.dropna()
        
        if len(df) == 0:
            # Calculate required minimum data
            max_lag = max(max(lags) for lags in LAG_PERIODS.values())
            max_window = max(ROLLING_WINDOWS) if ROLLING_WINDOWS else 0
            min_required = max(max_lag, max_window)
            
            raise ValueError(
                f"No valid data after feature engineering. "
                f"Need at least {min_required} historical records "
                f"(max lag: {max_lag} days, max rolling window: {max_window} days). "
                f"Provided data has {n_input_rows} rows, "
                f"but all became NaN after lag/rolling features. "
                f"Please provide at least {min_required + 5} rows for reliable predictions."
            )
        
        # Step 7: Apply the fitted preprocessor (scaling/normalization)
        preprocessor = self.preprocessors[target_name]
        transformed_data = preprocessor.transform(df)
        
        # Convert to DataFrame with correct feature names and index from df (after dropna)
        if target_name in self.feature_names:
            transformed_df = pd.DataFrame(
                transformed_data,
                columns=self.feature_names[target_name],
                index=df.index  # Use df.index (after dropna), not raw_data.index
            )
        else:
            transformed_df = pd.DataFrame(
                transformed_data,
                index=df.index  # Use df.index (after dropna), not raw_data.index
            )
        
        return transformed_df
    
    def predict(self, raw_data):
        """
        Generate 5-day temperature forecast.
        
        Args:
            raw_data (pd.DataFrame): Raw weather data with datetime index
                Required columns depend on feature engineering pipeline
        
        Returns:
            pd.DataFrame: Predictions for all horizons with columns [t+1, t+2, ..., t+5]
                Note: Index may be shorter than input due to NaN removal from lag features
        """
        predictions = {}
        result_index = None
        
        for day_step in range(1, N_STEPS_AHEAD + 1):
            target_name = f"t+{day_step}"
            
            # Prepare features for this target
            X = self.prepare_features(raw_data, target_name)
            
            # Store the index from first target (all targets should have same index after feature engineering)
            if result_index is None:
                result_index = X.index
            
            # Make prediction
            model = self.models[target_name]
            y_pred = model.predict(X)
            
            predictions[target_name] = y_pred
        
        # Combine into DataFrame using the actual result index (after dropna)
        predictions_df = pd.DataFrame(predictions, index=result_index)
        
        return predictions_df
    
    def predict_single(self, raw_data):
        """
        Generate forecast for a single time point (most recent in dataset).
        Uses all historical data to compute lag/rolling features, then predicts for the last date.
        
        Args:
            raw_data (pd.DataFrame): Full historical raw weather data
        
        Returns:
            dict: Single forecast {target: temperature}
        """
        # Use all historical data for feature engineering, get predictions for all valid dates
        predictions_df = self.predict(raw_data)
        
        if predictions_df is None or len(predictions_df) == 0:
            print("✗ No valid predictions generated")
            return None
        
        # Return the most recent prediction as dictionary
        forecast = predictions_df.iloc[-1].to_dict()
        
        return forecast
    
    def predict_with_metadata(self, raw_data):
        """
        Generate forecast with additional metadata for the most recent date.
        Uses all historical data to compute lag/rolling features.
        
        Args:
            raw_data (pd.DataFrame): Full historical raw weather data
        
        Returns:
            dict: {
                'predictions': dict of {target: temperature},
                'forecast_dates': list of forecast date strings,
                'base_date': base date for forecast,
                'model_info': model information
            }
        """
        # Get single prediction using all historical data
        predictions = self.predict_single(raw_data)
        
        if predictions is None:
            return None
        
        # Get base date (last date in input)
        base_date = raw_data.index[-1]
        
        # Calculate forecast dates
        forecast_dates = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                         for i in range(1, N_STEPS_AHEAD + 1)]
        
        # Get model info
        model_info = {
            'n_models': len(self.models),
            'horizons': list(self.models.keys()),
            'models_dir': str(MODELS_DIR)
        }
        
        return {
            'predictions': predictions,
            'forecast_dates': forecast_dates,
            'base_date': base_date.strftime('%Y-%m-%d'),
            'model_info': model_info
        }


def demo_prediction():
    """Demonstration of how to use the WeatherForecaster."""
    print("\n" + "="*70)
    print("WEATHER FORECASTER DEMO")
    print("="*70 + "\n")
    
    # Initialize forecaster
    forecaster = WeatherForecaster()
    
    # Load raw weather data (before preprocessing)
    print("Loading raw data...")
    raw_data_path = os.path.join(PROJECT_ROOT, 'dataset', 'hn_daily.csv')
    
    if not os.path.exists(raw_data_path):
        print(f"✗ Raw data not found at: {raw_data_path}")
        print("  Please ensure dataset/hn_daily.csv exists.")
        return
    
    # Load raw data - need sufficient historical records for lag/rolling features
    raw_data = pd.read_csv(raw_data_path, parse_dates=['datetime'])
    raw_data.set_index('datetime', inplace=True)
    
    print(f"✓ Loaded {len(raw_data)} total records")
    print(f"  Most recent date: {raw_data.index[-1]}\n")
    
    # Make single prediction from most recent data
    print("Generating 5-day forecast from most recent data...")
    result = forecaster.predict_single(raw_data)
    
    if result is not None:
        base_date = raw_data.index[-1]
        forecast_dates = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                         for i in range(1, 6)]
        
        print("\n" + "="*70)
        print("5-Day Temperature Forecast")
        print("="*70)
        print(f"Base date: {base_date.strftime('%Y-%m-%d')}")
        print(f"\nPredictions:")
        for target, temp in result.items():
            print(f"  {target}: {temp:.2f}°C")
        print(f"\nForecast dates: {', '.join(forecast_dates)}")
        print("="*70)
        print("\n✅ Demo complete!")


if __name__ == "__main__":
    # Run demonstration
    demo_prediction()
