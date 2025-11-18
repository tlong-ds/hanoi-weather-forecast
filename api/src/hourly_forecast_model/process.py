import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_regression
import joblib

from src.hourly_forecast_model.helper import (
    PROJECT_ROOT, DATA_DIR, N_STEPS_AHEAD, TARGET_COLUMN, TARGET_COLUMNS
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "hn_hourly.csv")
ROLLING_WINDOWS_HOURS = [6, 12, 24]  # Hours for rolling statistics
LAG_PERIODS_HOURS = {
    'temp': [1, 3, 6, 12, 24],
    'humidity': [1, 6, 24],
    'sealevelpressure': [6, 24],
    'windspeed': [1, 3, 6, 24],
}

# Data splitting ratios
TRAIN_RATIO = 0.7
DEV_RATIO = 0.15
TEST_RATIO = 0.15

# Feature selection
FEATURE_SELECTION_TOP_N = 30
RANDOM_STATE = 42


# ============================================================================
# CUSTOM TRANSFORMERS
# ============================================================================

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outliers using IQR method."""
    
    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier
        self.lower_ = None
        self.upper_ = None
    
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.multiplier * iqr
        self.upper_ = q3 + self.multiplier * iqr
        return self
    
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.minimum(np.maximum(X, self.lower_), self.upper_)


# ============================================================================
# DATA LOADING AND CLEANING
# ============================================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load hourly weather data from CSV."""
    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    df = df[~df.index.duplicated(keep="first")]
    print(f"✓ Loaded: {df.shape} from {file_path}")
    return df


def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that would cause data leakage or are non-numeric."""
    cols_to_drop = [
        "tempmax", "tempmin", "feelslikemax", "feelslikemin", "feelslike",
        "name", "stations", "source", "season", "conditions", "description",
        "preciptype", "snow", "snowdepth", "severerisk", "icon"
    ]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_day_length_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate day length from sunrise/sunset."""
    df_new = df.copy()
    if "sunrise" in df_new.columns and "sunset" in df_new.columns:
        sr = pd.to_datetime(df_new["sunrise"], errors="coerce")
        ss = pd.to_datetime(df_new["sunset"], errors="coerce")
        df_new["day_length_h"] = (ss - sr).dt.total_seconds() / 3600.0
        df_new = df_new.drop(columns=["sunrise", "sunset"])
    return df_new


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from datetime index."""
    df_new = df.copy()
    
    # Basic temporal features
    df_new["year"] = df_new.index.year
    df_new["month"] = df_new.index.month
    df_new["day_of_year"] = df_new.index.dayofyear
    df_new["week_of_year"] = df_new.index.isocalendar().week.astype(int)
    df_new["hour"] = df_new.index.hour
    df_new["is_night"] = df_new["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    
    # Cyclical encoding
    df_new["month_sin"] = np.sin(2 * np.pi * df_new["month"] / 12)
    df_new["month_cos"] = np.cos(2 * np.pi * df_new["month"] / 12)
    df_new["day_sin"] = np.sin(2 * np.pi * df_new["day_of_year"] / 365)
    df_new["day_cos"] = np.cos(2 * np.pi * df_new["day_of_year"] / 365)
    df_new["week_sin"] = np.sin(2 * np.pi * df_new["week_of_year"] / 52)
    df_new["week_cos"] = np.cos(2 * np.pi * df_new["week_of_year"] / 52)
    df_new["hour_sin"] = np.sin(2 * np.pi * df_new["hour"] / 24)
    df_new["hour_cos"] = np.cos(2 * np.pi * df_new["hour"] / 24)
    
    # Seasonal flags
    df_new["is_hot_month"] = df_new["month"].isin([5, 6, 7, 8]).astype(int)
    df_new["is_cool_month"] = df_new["month"].isin([12, 1, 2]).astype(int)
    df_new["is_monsoon_season"] = df_new["month"].isin([5, 6, 7, 8, 9]).astype(int)
    df_new["is_dry_season"] = df_new["month"].isin([11, 12, 1, 2, 3]).astype(int)
    
    # Drop redundant columns
    df_new = df_new.drop(columns=["month", "day_of_year", "week_of_year"], errors="ignore")
    
    return df_new


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features for weather variables."""
    df_new = df.copy()
    
    for base_col, lags in LAG_PERIODS_HOURS.items():
        if base_col in df_new.columns:
            for lag in lags:
                df_new[f"{base_col}_lag{lag}h"] = df_new[base_col].shift(lag)
    
    return df_new


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling window statistics."""
    df_new = df.copy()
    
    if "precip" in df_new.columns:
        df_new["precip_roll_mean_6h"] = df_new["precip"].shift(1).rolling(6).mean()
        df_new["precip_roll_mean_24h"] = df_new["precip"].shift(1).rolling(24).mean()
    
    if "humidity" in df_new.columns:
        df_new["humidity_roll_mean_6h"] = df_new["humidity"].shift(1).rolling(6).mean()
        df_new["humidity_roll_mean_24h"] = df_new["humidity"].shift(1).rolling(24).mean()
    
    if "windspeed" in df_new.columns:
        df_new["windspeed_roll_max_3h"] = df_new["windspeed"].shift(1).rolling(3).max()
        df_new["windspeed_roll_max_24h"] = df_new["windspeed"].shift(1).rolling(24).max()
    
    return df_new


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features."""
    df_new = df.copy()
    
    if "windspeed" in df_new.columns:
        df_new["windspeed_sq"] = df_new["windspeed"] ** 2
    
    if {"sealevelpressure", "humidity"}.issubset(df_new.columns):
        df_new["pressure_humidity"] = df_new["sealevelpressure"] * df_new["humidity"]
    
    if "day_length_h" in df_new.columns and "uvindex" in df_new.columns:
        df_new["daylength_uv"] = df_new["day_length_h"] * df_new["uvindex"]
    
    return df_new


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    print("\n📊 Feature Engineering:")
    df = remove_leakage_columns(df)
    print("  ✓ Removed leakage columns")
    
    df = create_day_length_feature(df)
    print("  ✓ Created day length features")
    
    df = create_temporal_features(df)
    print("  ✓ Created temporal features")
    
    df = create_lag_features(df)
    print("  ✓ Created lag features")
    
    df = create_rolling_features(df)
    print("  ✓ Created rolling window features")
    
    df = create_interaction_features(df)
    print("  ✓ Created interaction features")
    
    return df


# ============================================================================
# TARGET CREATION
# ============================================================================

def create_multistep_hourly_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create multi-step hourly forecast targets (t+1h to t+24h)."""
    df_new = df.copy()
    target_cols = []
    
    for i in range(1, N_STEPS_AHEAD + 1):
        name = f"target_temp_t+{i}h"
        df_new[name] = df_new[TARGET_COLUMN].shift(-i)
        target_cols.append(name)
    
    print(f"\n🎯 Created {len(target_cols)} hourly targets: t+1h to t+{N_STEPS_AHEAD}h")
    
    return df_new, target_cols


# ============================================================================
# TRAIN/DEV/TEST SPLIT
# ============================================================================

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train/dev/test."""
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_dev = int(n * DEV_RATIO)
    
    train = df.iloc[:n_train]
    dev = df.iloc[n_train:n_train + n_dev]
    test = df.iloc[n_train + n_dev:]
    
    print(f"\n✂️  Data Split:")
    print(f"  Train: {len(train):,} samples ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Dev:   {len(dev):,} samples ({DEV_RATIO*100:.0f}%)")
    print(f"  Test:  {len(test):,} samples ({TEST_RATIO*100:.0f}%)")
    
    return train, dev, test


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features(X: pd.DataFrame, y: pd.Series, top_n: int = 30) -> List[str]:
    """Select top features using ensemble of methods."""
    print(f"\n🔍 Feature Selection (top {top_n}):")
    
    # Correlation
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    
    # Mutual Information
    mi = mutual_info_regression(X.fillna(0), y, random_state=RANDOM_STATE)
    mi_s = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    
    # Lasso
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1).fit(X.fillna(0), y)
    coef_abs = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(X.fillna(0), y)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    # Combine rankings
    rank_df = pd.DataFrame({
        "corr_rank": corrs.rank(ascending=False),
        "mi_rank": mi_s.rank(ascending=False),
        "lasso_rank": coef_abs.rank(ascending=False),
        "rf_rank": rf_imp.rank(ascending=False),
    }).fillna(1e6)
    
    rank_df["mean_rank"] = rank_df.mean(axis=1)
    selected = rank_df.sort_values("mean_rank").head(top_n).index.tolist()
    
    print(f"  ✓ Selected {len(selected)} features from {len(X.columns)} total")
    
    return selected


# ============================================================================
# TRANSFORMATION PIPELINE
# ============================================================================

def create_preprocessing_pipeline() -> Pipeline:
    """Create sklearn preprocessing pipeline."""
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('outlier_clipper', OutlierClipper(multiplier=1.5)),
        ('scaler', StandardScaler())
    ])
    
    return pipeline


# ============================================================================
# SAVE FUNCTIONS
# ============================================================================

def save_data(data: dict, folder_path: str) -> None:
    """Save preprocessed data to CSV files."""
    os.makedirs(folder_path, exist_ok=True)
    
    for name, df in data.items():
        file_path = os.path.join(folder_path, f'{name}.csv')
        df.to_csv(file_path)
        print(f"    ✓ Saved: {file_path}")


# ============================================================================
# MAIN PREPROCESSING
# ============================================================================

def main(data_path: str = None):
    """
    Run complete preprocessing pipeline for hourly forecasting.
    
    Args:
        data_path: Path to raw hourly data CSV
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    
    print("=" * 70)
    print("HOURLY WEATHER FORECAST - DATA PREPROCESSING")
    print("=" * 70)
    print(f"Data path: {data_path}")
    print(f"Output dir: {DATA_DIR}")
    print("=" * 70)
    
    # Load data
    print("\n1️⃣  Loading data...")
    df = load_data(data_path)
    
    # Feature engineering
    print("\n2️⃣  Feature engineering...")
    df = apply_feature_engineering(df)
    
    # Create targets
    print("\n3️⃣  Creating targets...")
    df, target_cols = create_multistep_hourly_target(df)
    
    # Remove rows with NaN in targets (due to shift)
    df = df.dropna(subset=target_cols)
    print(f"  ✓ Removed {N_STEPS_AHEAD} rows with NaN targets")
    
    # Split data
    print("\n4️⃣  Splitting data...")
    train_df, dev_df, test_df = split_data(df)
    
    # Separate features and targets
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    X_train_full = train_df[feature_cols]
    y_train_full = train_df[target_cols]
    X_dev_full = dev_df[feature_cols]
    y_dev_full = dev_df[target_cols]
    X_test_full = test_df[feature_cols]
    y_test_full = test_df[target_cols]
    
    print(f"  ✓ Train: {X_train_full.shape}, Dev: {X_dev_full.shape}, Test: {X_test_full.shape}")
    
    # Feature selection (using first target as representative)
    print("\n5️⃣  Feature selection...")
    print(f"  Selecting top {FEATURE_SELECTION_TOP_N} features...")
    selected_features = select_features(X_train_full, y_train_full.iloc[:, 0], top_n=FEATURE_SELECTION_TOP_N)
    
    print(f"  ✓ Selected {len(selected_features)} features")
    
    # Save feature selection info
    feature_selection_dir = os.path.join(DATA_DIR, 'feature_selection')
    os.makedirs(feature_selection_dir, exist_ok=True)
    
    pd.DataFrame({
        'feature': selected_features,
        'rank': range(1, len(selected_features) + 1)
    }).to_csv(os.path.join(feature_selection_dir, 'selected_features.csv'), index=False)
    
    print(f"  ✓ Feature selection info saved to '{feature_selection_dir}/'")
    
    # Select and transform features once (same for all hours)
    print(f"\n{'='*70}")
    print("PREPROCESSING FEATURES")
    print(f"{'='*70}")
    
    X_train_selected = X_train_full[selected_features]
    X_dev_selected = X_dev_full[selected_features]
    X_test_selected = X_test_full[selected_features]
    
    # Create and fit preprocessing pipeline
    pipeline_path = os.path.join(DATA_DIR, 'preprocessing_pipeline.joblib')
    
    if os.path.exists(pipeline_path):
        print(f"  Loading existing pipeline...")
        pipeline = joblib.load(pipeline_path)
    else:
        print(f"  Creating and fitting pipeline...")
        pipeline = create_preprocessing_pipeline()
        pipeline.fit(X_train_selected)
        joblib.dump(pipeline, pipeline_path)
        print(f"  ✓ Pipeline saved to '{pipeline_path}'")
    
    # Transform data
    print(f"  Transforming data...")
    X_train_transformed = pipeline.transform(X_train_selected)
    X_dev_transformed = pipeline.transform(X_dev_selected)
    X_test_transformed = pipeline.transform(X_test_selected)
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_transformed, columns=selected_features, index=X_train_full.index)
    X_dev_df = pd.DataFrame(X_dev_transformed, columns=selected_features, index=X_dev_full.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=selected_features, index=X_test_full.index)
    
    print(f"  ✓ Transformation complete: {X_train_df.shape[1]} features")
    
    # Save transformed features and all targets
    print(f"\n{'='*70}")
    print("SAVING PROCESSED DATA")
    print(f"{'='*70}")
    
    save_data({
        'X_train_transformed': X_train_df,
        'X_dev_transformed': X_dev_df,
        'X_test_transformed': X_test_df,
        'y_train': y_train_full,
        'y_dev': y_dev_full,
        'y_test': y_test_full
    }, DATA_DIR)
    
    print(f"  ✓ All data saved to '{DATA_DIR}/'")
    
    print(f"\n{'='*70}")
    print("✓✓✓ PREPROCESSING COMPLETE ✓✓✓")
    print(f"{'='*70}")
    print("\nSummary:")
    print(f"  • Features: {len(selected_features)}")
    print(f"  • Targets: {N_STEPS_AHEAD} hours (t+1h to t+{N_STEPS_AHEAD}h)")
    print(f"  • Train samples: {X_train_df.shape[0]}")
    print(f"  • Dev samples: {X_dev_df.shape[0]}")
    print(f"  • Test samples: {X_test_df.shape[0]}")
    print(f"  • Data directory: {DATA_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
