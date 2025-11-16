
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from src.daily_forecast_model.helper import (
    TARGET_COLUMNS, MODELS_DIR, N_STEPS_AHEAD, TARGET_COLUMN,
    load_single_model, get_best_model_path
)
from src.daily_forecast_model.process import (
    remove_leakage_columns, create_day_length_feature,
    create_temporal_features, create_cyclical_features,
    create_rolling_features, create_lag_features,
    create_season_features, N_STEPS_AHEAD as PROCESS_N_STEPS
)


class WeatherForecaster:
    """
    Production forecaster for 5-day temperature predictions.
    
    This class encapsulates the entire inference pipeline:
    1. Load trained model and preprocessing pipeline
    2. Apply feature engineering to new data
    3. Transform features using fitted pipeline
    4. Generate predictions
    5. Return formatted results
    """
    
    def __init__(self, model_path=None, pipeline_path=None):
        """
        Initialize the forecaster by loading the trained model and pipeline.
        
        Args:
            model_path (str, optional): Path to saved model. If None, uses best model from MODELS_DIR
            pipeline_path (str, optional): Path to preprocessing pipeline. If None, uses default
        """
        self.model = None
        self.preprocessor = None
        self.model_name = None
        self.feature_names = None
        
        # Load model
        self._load_model(model_path)
        
        # Load preprocessing pipeline
        self._load_preprocessor(pipeline_path)
        
        print(f"\n{'='*70}")
        print("✓ WeatherForecaster initialized successfully")
        print(f"  Model: {self.model_name}")
        print(f"  Target horizons: {N_STEPS_AHEAD} days (t+1 to t+{N_STEPS_AHEAD})")
        print(f"{'='*70}\n")
    
    def _load_model(self, model_path=None):
        """Load the trained model from disk using centralized helper function."""
        if model_path is None:
            # Auto-detect best model using helper function
            model_path = get_best_model_path()
            
            if model_path is None:
                raise FileNotFoundError(
                    f"No trained models found in '{MODELS_DIR}/'. "
                    "Please train a model first using train.py"
                )
        
        print(f"Loading model from: {model_path}")
        
        try:
            # Use centralized model loading function from helper
            self.model = load_single_model(model_path)
            self.model_name = os.path.basename(model_path).replace('_backup', '').replace('.joblib', '').replace('.pkl', '')
            print(f"✓ Model loaded: {self.model_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _load_preprocessor(self, pipeline_path=None):
        """Load the preprocessing pipeline from disk."""
        if pipeline_path is None:
            # Use default pipeline for combined features (t+2 to t+4 use combined features)
            pipeline_path = 'processed_data/preprocessor_target_temp_t+3.joblib'
            
            # If combined pipeline doesn't exist, try any available pipeline
            if not os.path.exists(pipeline_path):
                # Look for any preprocessor file
                data_dir = 'processed_data'
                pipeline_files = []
                
                # Check all target directories
                for i in range(1, N_STEPS_AHEAD + 1):
                    target_dir = os.path.join(data_dir, f'target_t_{i}')
                    potential_pipeline = os.path.join(target_dir, f'preprocessor_target_temp_t+{i}.joblib')
                    if os.path.exists(potential_pipeline):
                        pipeline_files.append(potential_pipeline)
                
                if not pipeline_files:
                    raise FileNotFoundError(
                        f"No preprocessing pipeline found. "
                        "Please run process.py first to create the pipeline."
                    )
                
                pipeline_path = pipeline_files[0]  # Use first available pipeline
        
        print(f"Loading preprocessing pipeline from: {pipeline_path}")
        
        try:
            self.preprocessor = joblib.load(pipeline_path)
            
            # Extract feature names from the pipeline
            if hasattr(self.preprocessor, 'named_transformers_'):
                numeric_features = self.preprocessor.named_transformers_['num'].feature_names_in_.tolist()
                print(f"✓ Pipeline loaded with {len(numeric_features)} features")
                self.feature_names = numeric_features
            
        except Exception as e:
            raise RuntimeError(f"Failed to load preprocessing pipeline from {pipeline_path}: {e}")
    
    def prepare_features(self, raw_data):
        """
        Apply feature engineering to raw input data.
        
        This replicates the same transformations used during training:
        - Remove leakage columns
        - Create day length feature
        - Create temporal features
        - Create cyclical features
        - Create rolling statistics
        - Create lag features
        - Create season features
        
        Args:
            raw_data (pd.DataFrame): Raw weather data with datetime index
        
        Returns:
            pd.DataFrame: Feature-engineered data ready for preprocessing
        """
        print(f"\nApplying feature engineering...")
        
        # Make a copy to avoid modifying original data
        df = raw_data.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            else:
                raise ValueError("Data must have a datetime index or 'datetime' column")
        
        # Sort by date
        df = df.sort_index(ascending=True)
        
        # Apply feature engineering steps (same as preprocessing)
        df = remove_leakage_columns(df)
        df = create_day_length_feature(df)
        df = create_temporal_features(df)
        df = create_cyclical_features(df)
        df = create_rolling_features(df)
        df = create_lag_features(df)
        df = create_season_features(df)
        
        # Drop rows with NaN (from lag/rolling features)
        original_len = len(df)
        df = df.dropna()
        dropped = original_len - len(df)
        
        if dropped > 0:
            print(f"  ⚠️  Dropped {dropped} rows due to NaN values from lag/rolling features")
        
        print(f"✓ Feature engineering complete: {df.shape[0]} samples, {df.shape[1]} features")
        
        return df
    
    def predict(self, raw_data, return_dataframe=True):
        """
        Generate 5-day temperature forecast for new data.
        
        Args:
            raw_data (pd.DataFrame): Raw weather data with datetime index
            return_dataframe (bool): If True, return DataFrame; if False, return dict
        
        Returns:
            pd.DataFrame or dict: Predictions for each forecast horizon (t+1 to t+5)
        """
        print(f"\n{'='*70}")
        print("GENERATING FORECAST")
        print(f"{'='*70}")
        
        # Step 1: Feature engineering
        featured_data = self.prepare_features(raw_data)
        
        # Step 2: Select only the features used by the preprocessor
        if self.feature_names is not None:
            # Get intersection of available features and required features
            available_features = [f for f in self.feature_names if f in featured_data.columns]
            missing_features = [f for f in self.feature_names if f not in featured_data.columns]
            
            if missing_features:
                print(f"\n⚠️  Warning: {len(missing_features)} required features are missing:")
                for feat in missing_features[:5]:  # Show first 5
                    print(f"    • {feat}")
                if len(missing_features) > 5:
                    print(f"    ... and {len(missing_features) - 5} more")
                
                # Create missing features with zeros (fallback)
                for feat in missing_features:
                    featured_data[feat] = 0
            
            # Reorder columns to match pipeline
            X = featured_data[self.feature_names]
        else:
            X = featured_data
        
        print(f"\n[1] Input data shape: {X.shape}")
        
        # Step 3: Apply preprocessing transformations
        print(f"[2] Applying preprocessing transformations...")
        X_transformed = self.preprocessor.transform(X)
        print(f"    ✓ Transformed shape: {X_transformed.shape}")
        
        # Step 4: Generate predictions
        print(f"[3] Generating predictions...")
        predictions = self.model.predict(X_transformed)
        print(f"    ✓ Predictions generated: {predictions.shape}")
        
        # Step 5: Format output
        predictions_df = pd.DataFrame(
            predictions,
            index=X.index,
            columns=TARGET_COLUMNS
        )
        
        print(f"\n{'='*70}")
        print("FORECAST COMPLETE")
        print(f"{'='*70}")
        print(f"  Forecast period: {predictions_df.index[0]} to {predictions_df.index[-1]}")
        print(f"  Number of predictions: {len(predictions_df)}")
        print(f"  Horizons: {', '.join(TARGET_COLUMNS)}")
        print(f"{'='*70}\n")
        
        if return_dataframe:
            return predictions_df
        else:
            # Return as dictionary
            return predictions_df.to_dict(orient='index')
    
    def predict_single_day(self, date_str=None):
        """
        Generate forecast for a single day (returns 5-day ahead predictions for that day).
        
        Args:
            date_str (str, optional): Date in 'YYYY-MM-DD' format. If None, uses latest available
        
        Returns:
            dict: Predictions for t+1 through t+5
        """
        # This would require loading recent historical data
        # For now, raise NotImplementedError
        raise NotImplementedError(
            "Single day prediction requires access to historical data context. "
            "Please use predict() method with a DataFrame containing recent historical data."
        )
    
    def get_model_info(self):
        """Return information about the loaded model and pipeline."""
        info = {
            'model_name': self.model_name,
            'n_steps_ahead': N_STEPS_AHEAD,
            'target_column': TARGET_COLUMN,
            'target_columns': TARGET_COLUMNS,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'model_type': type(self.model).__name__
        }
        return info


# ============================================================================
# STANDALONE INFERENCE FUNCTION (for simple use cases)
# ============================================================================

def predict_temperature(raw_data, model_path=None, pipeline_path=None):
    """
    Convenience function for one-off predictions.
    
    Args:
        raw_data (pd.DataFrame): Raw weather data with datetime index
        model_path (str, optional): Path to trained model
        pipeline_path (str, optional): Path to preprocessing pipeline
    
    Returns:
        pd.DataFrame: Temperature predictions for t+1 through t+5
    
    Example:
        >>> import pandas as pd
        >>> from daily_forecast_model.inference import predict_temperature
        >>> 
        >>> # Load your new data
        >>> new_data = pd.read_csv('recent_weather.csv', index_col='datetime', parse_dates=True)
        >>> 
        >>> # Get predictions
        >>> predictions = predict_temperature(new_data)
        >>> print(predictions)
    """
    forecaster = WeatherForecaster(model_path=model_path, pipeline_path=pipeline_path)
    return forecaster.predict(raw_data)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("WEATHER FORECASTING INFERENCE MODULE")
    print(f"{'='*70}\n")
    
    # Example: Load some test data for demonstration
    try:
        # Try to load test set as example
        from daily_forecast_model.helper import X_TEST_FILE
        
        if os.path.exists(X_TEST_FILE):
            print(f"Loading example data from: {X_TEST_FILE}")
            X_test = pd.read_csv(X_TEST_FILE, index_col='datetime', parse_dates=True)
            
            # Take a small sample
            sample_data = X_test.head(10)
            
            print(f"\nExample: Forecasting for {len(sample_data)} days")
            print(f"Date range: {sample_data.index[0]} to {sample_data.index[-1]}")
            
            # Initialize forecaster
            forecaster = WeatherForecaster()
            
            # Note: This won't work directly because X_test is already transformed
            # In production, you would load raw data and let the forecaster handle feature engineering
            print("\n⚠️  Note: For this example to work, you need raw (untransformed) weather data")
            print("   The test set is already preprocessed. In production:")
            print("   1. Load raw weather data (e.g., from API or database)")
            print("   2. Pass it to forecaster.predict()")
            print("   3. Get 5-day forecast predictions")
            
        else:
            print("Test data not found. Please run process.py first.")
            
    except Exception as e:
        print(f"\n⚠️  Example failed: {e}")
        print("\nUsage example:")
        print("="*70)
        print("from daily_forecast_model.inference import WeatherForecaster")
        print("")
        print("# Load your raw weather data")
        print("raw_data = pd.read_csv('weather_data.csv', index_col='datetime', parse_dates=True)")
        print("")
        print("# Initialize forecaster")
        print("forecaster = WeatherForecaster()")
        print("")
        print("# Generate predictions")
        print("predictions = forecaster.predict(raw_data)")
        print("")
        print("# Display results")
        print("print(predictions)")
        print("="*70)
