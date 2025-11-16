import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta

# Import configuration
from src.hourly_forecast_model.helper import (
    PROJECT_ROOT, MODELS_DIR, N_STEPS_AHEAD, TARGET_COLUMNS
)

# Import feature engineering functions for unpickling
from src.hourly_forecast_model.process import (
    OutlierClipper,
    create_temporal_features,
    create_lag_features,
    create_rolling_features,
    create_interaction_features,
    create_seasonal_indicators,
    create_day_length_feature,
    create_cyclical_wind_direction,
    LAG_PERIODS_HOURS,
    ROLLING_WINDOWS_HOURS
)


class HourlyWeatherForecaster:
    """
    Production forecaster for 24-hour temperature predictions.
    
    This class encapsulates the entire inference pipeline:
    1. Load trained multi-output model
    2. Load preprocessing pipeline
    3. Apply feature engineering
    4. Generate predictions for all 24 hours
    5. Return formatted results
    """
    
    def __init__(self):
        """Initialize the forecaster by loading the trained model and pipeline."""
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
        self._load_model()
        self._load_preprocessor()
        
        print(f"\n{'='*70}")
        print("✓ HourlyWeatherForecaster initialized successfully")
        print(f"  Loaded multi-output model for {N_STEPS_AHEAD}-hour forecast")
        print(f"  Models directory: {MODELS_DIR}")
        print(f"{'='*70}\n")
    
    def _load_model(self):
        """Load trained multi-output model."""
        print("Loading trained multi-output model...")
        
        model_path = os.path.join(MODELS_DIR, 'model_multioutput_24h.joblib')
        
        try:
            self.model = joblib.load(model_path)
            print(f"  ✓ Model loaded: {model_path}")
        except FileNotFoundError:
            print(f"  ✗ Model not found at {model_path}")
            raise FileNotFoundError(
                f"Multi-output model not found. Please train the model first:\n"
                f"  python -m src.hourly_forecast_model.train"
            )
        
        print(f"✓ Model loaded successfully\n")
    
    def _load_preprocessor(self):
        """Load preprocessing pipeline."""
        print("Loading preprocessing pipeline...")
        
        # Preprocessor is stored in data_processing_hourly/
        preprocessor_path = os.path.join(PROJECT_ROOT, 'data_processing_hourly', 'preprocessing_pipeline.joblib')
        
        try:
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"  ✓ Preprocessor loaded: {preprocessor_path}")
            
            # Load feature names from training data
            feature_file = os.path.join(PROJECT_ROOT, 'data_processing_hourly', 'X_train_transformed.csv')
            if os.path.exists(feature_file):
                feature_df = pd.read_csv(feature_file, index_col=0, nrows=0)
                self.feature_names = feature_df.columns.tolist()
                print(f"  ✓ Loaded {len(self.feature_names)} feature names")
            
        except FileNotFoundError:
            print(f"  ✗ Preprocessor not found at {preprocessor_path}")
            raise FileNotFoundError(
                f"Preprocessor not found. Please run preprocessing first:\n"
                f"  python -m src.hourly_forecast_model.process"
            )
        
        print(f"✓ Preprocessor loaded successfully\n")
    
    def prepare_features(self, raw_data):
        """
        Apply feature engineering to raw data.
        
        This replicates the preprocessing pipeline:
        1. Create temporal features (hour, day, sin/cos encodings)
        2. Create day length feature
        3. Create cyclical wind direction
        4. Create seasonal indicators
        5. Create interaction features
        6. Create lag features (hourly lags: 1h, 3h, 6h, 12h, 24h)
        7. Create rolling window features (6h, 12h, 24h)
        8. Apply scaling/normalization via preprocessor
        
        Args:
            raw_data (pd.DataFrame): Raw weather data with datetime index
        
        Returns:
            pd.DataFrame: Transformed features ready for prediction
        """
        # Make a copy to avoid modifying original
        df = raw_data.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
            else:
                raise ValueError("Data must have datetime index or 'datetime' column")
        
        # Step 1: Create temporal features
        df = create_temporal_features(df)
        
        # Step 2: Create day length feature
        df = create_day_length_feature(df)
        
        # Step 3: Create cyclical wind direction
        df = create_cyclical_wind_direction(df)
        
        # Step 4: Create seasonal indicators
        df = create_seasonal_indicators(df)
        
        # Step 5: Create interaction features
        df = create_interaction_features(df)
        
        # Step 6: Create lag features (hourly: 1h, 3h, 6h, 12h, 24h)
        df = create_lag_features(df, lag_config=LAG_PERIODS_HOURS)
        
        # Step 7: Create rolling window features
        df = create_rolling_features(df, windows=ROLLING_WINDOWS_HOURS)
        
        # Drop NaN rows (created by lag/rolling features)
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("No valid data after feature engineering. Need sufficient historical data.")
        
        return df
    
    def predict(self, raw_data):
        """
        Make 24-hour temperature predictions.
        
        Args:
            raw_data (pd.DataFrame): Raw weather data with historical observations
                                    Must have datetime index and required weather variables
        
        Returns:
            pd.DataFrame: Predictions for all 24 hours with metadata
        """
        print(f"\n{'='*70}")
        print("Making 24-hour temperature predictions")
        print(f"{'='*70}\n")
        
        # Step 1: Prepare features
        print("Step 1: Feature engineering...")
        features_df = self.prepare_features(raw_data)
        print(f"  ✓ Features prepared: {features_df.shape}")
        
        # Step 2: Select only the features used in training
        if self.feature_names:
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(features_df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            features_df = features_df[self.feature_names]
        
        # Step 3: Apply preprocessing (scaling, etc.)
        print("\nStep 2: Applying preprocessing...")
        features_transformed = self.preprocessor.transform(features_df)
        print(f"  ✓ Transformation complete")
        
        # Step 4: Make predictions
        print("\nStep 3: Generating predictions...")
        predictions = self.model.predict(features_transformed)
        print(f"  ✓ Predictions shape: {predictions.shape}")
        
        # Step 5: Format results
        print("\nStep 4: Formatting results...")
        
        # Get the latest timestamp from input data
        forecast_time = features_df.index[-1]
        
        # Create forecast timestamps (next 24 hours)
        forecast_timestamps = [forecast_time + timedelta(hours=i+1) for i in range(N_STEPS_AHEAD)]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'forecast_hour': list(range(1, N_STEPS_AHEAD + 1)),
            'timestamp': forecast_timestamps,
            'predicted_temp': predictions[-1, :]  # Use last row of predictions
        })
        
        # Add hour name
        results['hour_name'] = [f't+{h}h' for h in results['forecast_hour']]
        
        print(f"  ✓ Results formatted")
        print(f"\n{'='*70}")
        print("✅ Prediction complete!")
        print(f"{'='*70}\n")
        
        return results
    
    def predict_single_timestamp(self, raw_data):
        """
        Convenience method to get predictions for the latest timestamp only.
        
        Args:
            raw_data (pd.DataFrame): Raw weather data
        
        Returns:
            dict: Predictions with timestamps
        """
        results_df = self.predict(raw_data)
        
        return {
            'forecast_time': results_df['timestamp'].iloc[0] - timedelta(hours=1),
            'predictions': results_df[['timestamp', 'hour_name', 'predicted_temp']].to_dict('records')
        }
    
    def get_model_info(self):
        """Get information about the loaded model."""
        metadata_path = os.path.join(MODELS_DIR, 'training_metadata.json')
        
        try:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except FileNotFoundError:
            return {
                'error': 'Metadata not found',
                'model_path': os.path.join(MODELS_DIR, 'model_multioutput_24h.joblib')
            }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_forecaster():
    """Load and return a ready-to-use HourlyWeatherForecaster."""
    return HourlyWeatherForecaster()


def quick_predict(data_path):
    """
    Quick prediction from CSV file.
    
    Args:
        data_path (str): Path to CSV file with historical weather data
    
    Returns:
        pd.DataFrame: 24-hour temperature predictions
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize forecaster
    forecaster = HourlyWeatherForecaster()
    
    # Make predictions
    return forecaster.predict(df)


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*70)
    print("HOURLY WEATHER FORECASTER - EXAMPLE USAGE")
    print("="*70 + "\n")
    
    # Initialize forecaster
    forecaster = HourlyWeatherForecaster()
    
    # Show model info
    print("Model Information:")
    info = forecaster.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("To use for predictions:")
    print("  forecaster = HourlyWeatherForecaster()")
    print("  predictions = forecaster.predict(your_data)")
    print("="*70 + "\n")
