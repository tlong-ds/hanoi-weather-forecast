import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import os
import joblib

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Feature Engineering Configuration
ROLLING_WINDOWS = [7, 14, 21, 28, 56, 84]  # Days for rolling statistics
LAG_PERIODS = {
    'temp': [1, 3, 7],
    'dew': [1, 3, 7],
    'humidity': [1, 3, 7],
    'sealevelpressure': [1, 3, 7],
    'windspeed': [1, 3, 7],
    'precip': [1, 3, 7],
    'cloudcover': [1, 3, 7],
    'winddir_sin': [1, 3, 7],
    'winddir_cos': [1, 3, 7],
}
CAT_LAG_PERIODS = [1, 3]  # Lags for categorical (encoded) features

# Data Splitting Configuration
TRAIN_RATIO = 0.7
DEV_RATIO = 0.15
TEST_RATIO = 0.15

# Target Creation Configuration
N_STEPS_AHEAD = 5 # Predict 5 days ahead
TARGET_COLUMN = 'temp'

# Feature Selection Configuration
FEATURE_SELECTION_TOP_N = 30
LASSO_CV_FOLDS = 5
RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_STATE = 42

# Data Quality Configuration
OUTLIER_IQR_MULTIPLIER = 1.5
CATEGORICAL_FEATURES = ['icon']

# Seasonal Months Configuration (for Hanoi's climate)
SEASON_CONFIG = {
    'summer': [4, 5, 6],  
    'autumn': [7, 8, 9],  
    'winter': [10, 11, 12],  
    'spring': [1, 2, 3]  
}

# Columns to Drop (Data Leakage Prevention)
LEAKAGE_COLUMNS = [
    'tempmax', 'tempmin', 'feelslikemax', 'feelslikemin', 'feelslike',
    'name', 'stations', 'source', 'season',
    'conditions', 'description',
    'preciptype', 'snow', 'snowdepth', 'severerisk'
]

# ============================================================================

def load_data(file_path):
    """Loads weather data from a CSV file."""
    daily_data = pd.read_csv(file_path)
    daily_data['datetime'] = pd.to_datetime(daily_data['datetime'])
    daily_data.set_index('datetime', inplace=True)
    daily_data = daily_data.sort_index(ascending=True)
    return daily_data

def split_data(daily_data, train_ratio=TRAIN_RATIO, dev_ratio=DEV_RATIO):
    """Splits the data into training, development, and test sets."""
    train_size = int(len(daily_data) * train_ratio)
    dev_size = int(len(daily_data) * dev_ratio)
    
    train_data = daily_data.iloc[:train_size]
    dev_data = daily_data.iloc[train_size:train_size + dev_size]
    test_data = daily_data.iloc[train_size + dev_size:]
    
    return train_data, dev_data, test_data

def remove_leakage_columns(df, cols_to_drop=LEAKAGE_COLUMNS):
    """
    Remove columns that cause data leakage or are non-informative.
    (Based on the provided data definition)
    """
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    return df_clean

def create_day_length_feature(df):
    """
    Transform sunrise/sunset times into day_length feature (in hours).
    """
    df_new = df.copy()
    if 'sunrise' in df_new.columns and 'sunset' in df_new.columns:
        sr = pd.to_datetime(df_new['sunrise'], errors='coerce')
        ss = pd.to_datetime(df_new['sunset'], errors='coerce')
        # Compute difference in hours
        df_new['day_length_h'] = (ss - sr).dt.total_seconds() / 3600
        # Drop original columns
        df_new = df_new.drop(columns=['sunrise', 'sunset'])
    # If columns don't exist or were already dropped, do nothing
    return df_new

def create_cyclical_wind_direction(df):
    """
    Transform wind direction (degrees: 0-360) into cyclical sine and cosine components.
    This prevents treating 359° and 1° as far apart (they're only 2° apart).
    
    Wind direction is measured in degrees: 0°=North, 90°=East, 180°=South, 270°=West
    """
    df_new = df.copy()
    if 'winddir' in df_new.columns:
        # Convert degrees to radians, then to sin/cos for cyclical encoding
        winddir_rad = np.radians(df_new['winddir'])
        df_new['winddir_sin'] = np.sin(winddir_rad)
        df_new['winddir_cos'] = np.cos(winddir_rad)
        # Optionally drop original winddir column if you only want the cyclical components
        # df_new = df_new.drop(columns=['winddir'])
    return df_new

def create_temporal_features(df, season_config=SEASON_CONFIG):
    """
    Create comprehensive time-based features based on EDA insights.
    Captures seasonal patterns, monsoon cycles, and temporal interactions
    while minimizing redundancy.
    
    Args:
        df: Input DataFrame with datetime index
        season_config: Dictionary with seasonal month configurations
    """
    df_new = df.copy()
    
    # 1. Long-Term Trend
    df_new['year'] = df_new.index.year
    
    # 2. Weekly Pattern (Simple)
    df_new['day_of_week'] = df_new.index.weekday
    df_new['is_weekend'] = df_new['day_of_week'].isin([5, 6]).astype(int)

    # 3. Best-in-Class Cyclical Encodings
    # We drop the raw 'month', 'day_of_year' etc. later, as these
    # _sin/_cos features are the superior representation.
    df_new['month'] = df_new.index.month
    df_new['day_of_year'] = df_new.index.dayofyear
    df_new['week_of_year'] = df_new.index.isocalendar().week.astype(int)

    df_new['month_sin'] = np.sin(2 * np.pi * df_new['month'] / 12)
    df_new['month_cos'] = np.cos(2 * np.pi * df_new['month'] / 12)
    df_new['day_sin'] = np.sin(2 * np.pi * df_new['day_of_year'] / 365)
    df_new['day_cos'] = np.cos(2 * np.pi * df_new['day_of_year'] / 365)
    df_new['week_sin'] = np.sin(2 * np.pi * df_new['week_of_year'] / 52)
    df_new['week_cos'] = np.cos(2 * np.pi * df_new['week_of_year'] / 52)
    
    # 4. Domain-Specific Indicators (Based on seasonal configuration)
    
    # Create season indicators from config
    for season_name, months in season_config.items():
        df_new[f'is_{season_name}'] = df_new['month'].isin(months).astype(int)
    
    # 5. Drop the Redundant Original Columns
    # The _sin/_cos features have replaced them.
    # 'year' and 'is_weekend' are kept as they are not cyclical.
    cols_to_drop = ['month', 'day_of_year', 'week_of_year', 'day_of_week']
    df_new = df_new.drop(columns=cols_to_drop)
    
    return df_new

def create_lag_features(df, lag_config=LAG_PERIODS):
    """
    Create lag features to capture previous days' conditions.
    """
    df_new = df.copy()

    for feature, lags in lag_config.items():
        if feature in df_new.columns:
            for lag in lags:
                df_new[f'{feature}_lag{lag}'] = df_new[feature].shift(lag)
    
    return df_new

def create_rolling_features(df, features=None, windows=ROLLING_WINDOWS):
    """
    Create rolling window features to capture trends.
    
    Args:
        df: Input DataFrame
        features: List of features to create rolling stats for. Defaults to 
                 ['precip', 'humidity', 'windspeed', 'sealevelpressure', 'cloudcover']
        windows: List of window sizes for rolling statistics
    """
    if features is None:
        features = ['precip', 'humidity', 'windspeed', 'sealevelpressure', 'cloudcover']
    
    df_new = df.copy()

    for feature in features:
        if feature in df_new.columns:
            for window in windows:
                df_new[f'{feature}_roll_mean_{window}'] = df_new[feature].shift(1).rolling(window).mean()
                df_new[f'{feature}_roll_std_{window}'] = df_new[feature].shift(1).rolling(window).std()

    return df_new

def create_interaction_features(df):
    """
    Create physically meaningful interaction features.
    """
    df_new = df.copy()
    
    # Wind intensity (non-linear effect)
    if 'windspeed' in df_new.columns:
        df_new['windspeed_sq'] = df_new['windspeed'] ** 2
    
    # Atmospheric moisture interaction
    if {'sealevelpressure', 'humidity'}.issubset(df_new.columns):
        df_new['pressure_humidity'] = df_new['sealevelpressure'] * df_new['humidity']
    
    # Solar exposure (day length × UV index) - only if day_length_h exists
    if 'day_length_h' in df_new.columns and 'uvindex' in df_new.columns:
        df_new['daylength_uv'] = df_new['day_length_h'] * df_new['uvindex']
    
    return df_new

def encode_categorical_features(df, train_df, cat_cols=None):
    """
    One-hot encodes categorical columns using OneHotEncoder.
    Fits encoder ONLY on training data to prevent leakage.
    Fills 'none' for missing values.
    
    Args:
        df: DataFrame to transform
        train_df: Training DataFrame (used to fit encoder)
        cat_cols: List of categorical columns. Defaults to ['icon']
    
    Returns:
        df_new: DataFrame with encoded columns
        encoded_cols: Array of new encoded column names
    """
    if cat_cols is None:
        cat_cols = CATEGORICAL_FEATURES
    
    df_new = df.copy()
    
    # Check if any categorical columns exist
    if not any(col in df_new.columns for col in cat_cols):
        return df_new, []

    # Use OneHotEncoder to handle unseen values gracefully
    # This is more robust than pd.get_dummies
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Fit ONLY on the training data (prevents leakage)
    encoder.fit(train_df[cat_cols].fillna('none').astype(str))
    
    # Transform the current dataset (df)
    encoded_arr = encoder.transform(df_new[cat_cols].fillna('none').astype(str))

    encoded_cols = encoder.get_feature_names_out(cat_cols)

    # Create encoded dataframe
    df_enc = pd.DataFrame(encoded_arr, columns=encoded_cols, index=df_new.index)
    
    # Concatenate and drop original categorical columns
    df_new = pd.concat([df_new.drop(columns=cat_cols), df_enc], axis=1)
    
    # Return the new dataframe and the names of the new columns
    return df_new, encoded_cols

def create_lagged_cat_features(df, encoded_cols, lags=CAT_LAG_PERIODS):
    """
    Creates lagged features for the encoded categorical columns.
    
    Args:
        df: Input DataFrame with encoded categorical columns
        encoded_cols: List of encoded column names to lag
        lags: List of lag periods to create
    
    Returns:
        df_new: DataFrame with lagged categorical features (original dropped)
    """
    df_new = df.copy()
    for col in encoded_cols:
        for lag in lags:
            df_new[f'{col}_lag{lag}'] = df_new[col].shift(lag)
    
    # Drop the original (non-lagged) encoded columns to prevent data leakage
    df_new = df_new.drop(columns=encoded_cols)
    return df_new

def create_multiday_target(df, target_col=TARGET_COLUMN, n_steps_out=N_STEPS_AHEAD):
    """
    Creates a target DataFrame for multi-step forecasting.
    Predicts t+1 through t+n_steps_out.
    
    Args:
        df: Input DataFrame with target column
        target_col: Name of the column to create targets from
        n_steps_out: Number of steps ahead to predict
    
    Returns:
        df_new: DataFrame with target columns added
        target_cols: List of created target column names
    """
    df_new = df.copy()
    target_cols = []
    
    # Create targets for steps 1 to n_steps_out into the future
    for i in range(1, n_steps_out + 1):
        col_name = f'target_{target_col}_t+{i}'
        df_new[col_name] = df_new[target_col].shift(-i)
        target_cols.append(col_name)
        
    return df_new, target_cols

# Custom transformer: IQR-based outlier clipping
class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips outliers using the IQR method.
    Fit on training data to calculate bounds, then apply to all sets.
    """
    def __init__(self, multiplier=OUTLIER_IQR_MULTIPLIER):
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
        X_clipped = np.minimum(np.maximum(X, self.lower_), self.upper_)
        return X_clipped
    
    def get_feature_names_out(self, input_features=None):
        return input_features

def apply_feature_engineering(df):
    """Apply all feature engineering steps in sequence."""
    df = remove_leakage_columns(df)
    df = create_day_length_feature(df)
    df = create_cyclical_wind_direction(df)
    df = create_temporal_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_interaction_features(df)
    return df

def select_features_for_target(X, y, top_n, lasso_cv=LASSO_CV_FOLDS, 
                              rf_n_est=RANDOM_FOREST_N_ESTIMATORS, random_state=RANDOM_STATE,
                              target_name=""):
    """
    Selects the best features for a single target using a combination of methods.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        top_n: Number of features to select
        lasso_cv: Cross-validation folds for LassoCV
        rf_n_est: Number of estimators for RandomForestRegressor
        random_state: Random seed for reproducibility
        target_name: Name of the target (for logging)
    
    Returns:
        selected_features: List of selected feature names
        scores_df: DataFrame with feature importance scores and ranks
    """
    print(f"  Selecting features for {target_name}...")
    
    # Pearson correlation
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    
    # Mutual Information
    mi = mutual_info_regression(X.fillna(0), y, random_state=random_state)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    
    # LassoCV
    lasso = LassoCV(cv=lasso_cv, random_state=random_state, n_jobs=-1)
    lasso.fit(X.fillna(0), y)
    coef_abs = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
    
    # RandomForest importance
    rf = RandomForestRegressor(n_estimators=rf_n_est, random_state=random_state, n_jobs=-1)
    rf.fit(X.fillna(0), y)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    # Aggregate ranks
    rank_df = pd.DataFrame({
        'corr_rank': corrs.rank(ascending=False),
        'mi_rank': mi_series.rank(ascending=False),
        'lasso_rank': coef_abs.rank(ascending=False),
        'rf_rank': rf_imp.rank(ascending=False)
    }).fillna(1e6)
    
    rank_df['mean_rank'] = rank_df.mean(axis=1)
    
    # Store scores for analysis
    scores_df = pd.DataFrame({
        'correlation': corrs,
        'mutual_info': mi_series,
        'lasso_coef': coef_abs,
        'rf_importance': rf_imp,
        'mean_rank': rank_df['mean_rank']
    }).sort_values('mean_rank')
    
    selected_features = scores_df.head(top_n).index.tolist()
    print(f"    ✓ Selected {len(selected_features)} features for {target_name}")
    
    return selected_features, scores_df


# def select_features_combined(X, y_df, target_cols, top_n=FEATURE_SELECTION_TOP_N, 
#                              lasso_cv=LASSO_CV_FOLDS, rf_n_est=RANDOM_FOREST_N_ESTIMATORS, 
#                              random_state=RANDOM_STATE):
#     """
#     Combined feature selection for both short-term (t+1) and long-term (t+N) forecasting.
    
#     This robust method runs feature selection twice:
#     1. Finds the best features for t+1 (capturing volatility/immediate changes)
#     2. Finds the best features for t+N (capturing long-term trends)
#     3. Combines them into one unified list
    
#     This guarantees your model receives the best features for both tasks.
    
#     Args:
#         X: Feature DataFrame
#         y_df: Target DataFrame with all target columns
#         target_cols: List of target column names (e.g., ['target_temp_t+1', ..., 'target_temp_t+5'])
#         top_n: Number of features to select per target
#         lasso_cv: Cross-validation folds for LassoCV
#         rf_n_est: Number of estimators for RandomForestRegressor
#         random_state: Random seed for reproducibility
    
#     Returns:
#         combined_features: List of selected feature names (union of short-term and long-term)
#         feature_info: Dict with detailed selection information
#     """
#     print(f"\n{'='*70}")
#     print("COMBINED FEATURE SELECTION: Short-Term + Long-Term")
#     print(f"{'='*70}")
    
#     # Short-term target: t+1 (volatility, immediate changes)
#     short_term_target = target_cols[0]  # First target (t+1)
#     y_short = y_df[short_term_target]
    
#     print(f"\n[1] SHORT-TERM FEATURES (for {short_term_target})")
#     print(f"    Purpose: Capture immediate volatility and rapid changes")
#     short_features, short_scores = select_features_for_target(
#         X, y_short, top_n, lasso_cv, rf_n_est, random_state, target_name=short_term_target
#     )
    
#     # Long-term target: t+N (trend, seasonal patterns)
#     long_term_target = target_cols[-1]  # Last target (t+N)
#     y_long = y_df[long_term_target]
    
#     print(f"\n[2] LONG-TERM FEATURES (for {long_term_target})")
#     print(f"    Purpose: Capture long-term trends and seasonal patterns")
#     long_features, long_scores = select_features_for_target(
#         X, y_long, top_n, lasso_cv, rf_n_est, random_state, target_name=long_term_target
#     )
    
#     # Combine features (union to avoid duplicates)
#     combined_features = list(set(short_features + long_features))
    
#     # Analyze feature overlap
#     short_only = set(short_features) - set(long_features)
#     long_only = set(long_features) - set(short_features)
#     shared = set(short_features) & set(long_features)
    
#     print(f"\n{'='*70}")
#     print("FEATURE SELECTION SUMMARY")
#     print(f"{'='*70}")
#     print(f"  Short-term features selected: {len(short_features)}")
#     print(f"  Long-term features selected:  {len(long_features)}")
#     print(f"  Shared features:              {len(shared)}")
#     print(f"  Short-term only:              {len(short_only)}")
#     print(f"  Long-term only:               {len(long_only)}")
#     print(f"  TOTAL COMBINED features:      {len(combined_features)}")
#     print(f"{'='*70}")
    
#     # Additional insights
#     if shared:
#         print(f"\n✓ {len(shared)} features are important for BOTH short and long-term forecasting:")
#         for feat in sorted(list(shared))[:10]:  # Show top 10
#             print(f"    • {feat}")
#         if len(shared) > 10:
#             print(f"    ... and {len(shared) - 10} more")
    
#     if short_only:
#         print(f"\n✓ {len(short_only)} features are UNIQUE to short-term forecasting (volatility):")
#         for feat in sorted(list(short_only))[:5]:  # Show top 5
#             print(f"    • {feat}")
#         if len(short_only) > 5:
#             print(f"    ... and {len(short_only) - 5} more")
    
#     if long_only:
#         print(f"\n✓ {len(long_only)} features are UNIQUE to long-term forecasting (trends):")
#         for feat in sorted(list(long_only))[:5]:  # Show top 5
#             print(f"    • {feat}")
#         if len(long_only) > 5:
#             print(f"    ... and {len(long_only) - 5} more")
    
#     # Store detailed information
#     feature_info = {
#         'short_term_features': short_features,
#         'long_term_features': long_features,
#         'combined_features': combined_features,
#         'short_only': list(short_only),
#         'long_only': list(long_only),
#         'shared': list(shared),
#         'short_term_scores': short_scores,
#         'long_term_scores': long_scores
#     }
    
#     return combined_features, feature_info

def save_data(data, folder_path):
    """Saves the preprocessed data to CSV files."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    for name, df in data.items():
        df.to_csv(os.path.join(folder_path, f'{name}.csv'))

    
# Main function to run preprocessing
if __name__ == '__main__':
    """
    Main function to run the entire preprocessing pipeline.
    1. Load data
    2. Feature Engineer (full dataset)
    3. Split data 
    4. Create Target 
    5. Drop NaNs
    6. Select Features
    7. Fit Preprocessing Pipeline
    8. Transform & Save
    """
    
    # 1. Load data
    print("Loading data...")
    daily_data = load_data('dataset/hn_daily.csv')
    print(f"✓ Data loaded: {daily_data.shape[0]} rows, {daily_data.shape[1]} columns")
    
    # 2. Apply feature engineering (on full dataset)
    print("Applying feature engineering to full dataset...")
    featured_data = apply_feature_engineering(daily_data)

    # 3. Split data FIRST (before creating targets to avoid leakage!)
    print("Splitting data into train, dev, and test sets...")
    train_fe, dev_fe, test_fe = split_data(featured_data)

    # 4. Create targets AFTER split (prevents data leakage)
    print(f"Creating {N_STEPS_AHEAD}-day ahead targets...")
    train_fe, target_cols = create_multiday_target(train_fe, target_col=TARGET_COLUMN, n_steps_out=N_STEPS_AHEAD)
    dev_fe, _ = create_multiday_target(dev_fe, target_col=TARGET_COLUMN, n_steps_out=N_STEPS_AHEAD)
    test_fe, _ = create_multiday_target(test_fe, target_col=TARGET_COLUMN, n_steps_out=N_STEPS_AHEAD)
    print(f"Created multi-step target columns: {target_cols}")

    # 5. Drop rows with NaN in critical columns (separately for each set)
    print("Dropping NaN rows from lags/targets...")
    # Drop columns that are entirely NaN
    cols_to_drop_nan = [col for col in train_fe.columns if train_fe[col].isnull().all()]
    if cols_to_drop_nan:
        train_fe = train_fe.drop(columns=cols_to_drop_nan)
        dev_fe = dev_fe.drop(columns=cols_to_drop_nan)
        test_fe = test_fe.drop(columns=cols_to_drop_nan)

    # Drop rows with NaN in critical columns
    lag_roll_cols = [col for col in train_fe.columns if 'lag' in col or 'roll' in col]
    critical_cols = lag_roll_cols + target_cols # Drop row if any lag/roll OR any target is NaN
    
    train_fe_clean = train_fe.dropna(subset=critical_cols)
    dev_fe_clean = dev_fe.dropna(subset=critical_cols)
    test_fe_clean = test_fe.dropna(subset=critical_cols)

    print(f"✓ Data cleaning complete:")
    print(f"  Train: {train_fe_clean.shape[0]} rows, {train_fe_clean.shape[1]} columns")
    print(f"  Dev:   {dev_fe_clean.shape[0]} rows, {dev_fe_clean.shape[1]} columns")
    print(f"  Test:  {test_fe_clean.shape[0]} rows, {test_fe_clean.shape[1]} columns")

    # 6. Separate X and y for feature selection
    all_target_drop_cols = target_cols + [TARGET_COLUMN] 
    
    X_train_full = train_fe_clean.drop(columns=all_target_drop_cols, errors='ignore')
    y_train_full = train_fe_clean[target_cols]  # All targets

    X_dev_full = dev_fe_clean.drop(columns=all_target_drop_cols, errors='ignore')
    y_dev_full = dev_fe_clean[target_cols]
    
    X_test_full = test_fe_clean.drop(columns=all_target_drop_cols, errors='ignore')
    y_test_full = test_fe_clean[target_cols]

    # Lấy các cột số để chọn feature
    numeric_fs_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
    # Lấy các cột hạng mục
    categorical_features_to_keep = [col for col in CATEGORICAL_FEATURES if col in X_train_full.columns]

    # 7. Select features using COMBINED approach
    print(f"\n{'='*70}")
    print("STARTING PER-TARGET FEATURE SELECTION & PREPROCESSING")
    print(f"{'='*70}")

    for target_col_name in target_cols:
        target_day_str = target_col_name.split('+')[-1] # Lấy '1', '2', ...
        
        print(f"\nProcessing target: {target_col_name} (t+{target_day_str})")
        print(f"{'-'*70}")

        # 7.1. Lấy y cho target cụ thể này
        y_train_target = y_train_full[target_col_name]
        
        # 7.2. Chạy Feature Selection cho target này
        print(f"  [1] Running feature selection for {target_col_name}...")
        selected_numeric_features, scores_df = select_features_for_target(
            X_train_full[numeric_fs_cols], 
            y_train_target, 
            top_n=FEATURE_SELECTION_TOP_N,
            lasso_cv=LASSO_CV_FOLDS,
            rf_n_est=RANDOM_FOREST_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            target_name=target_col_name
        )
        
        # 7.3. Lưu kết quả feature selection cho target này
        feature_selection_dir = f'processed_data/feature_selection/t_{target_day_str}'
        if not os.path.exists(feature_selection_dir):
            os.makedirs(feature_selection_dir)
            
        pd.DataFrame({'feature': selected_numeric_features}).to_csv(
            os.path.join(feature_selection_dir, f'selected_features_t{target_day_str}.csv'), index=False
        )
        scores_df.to_csv(
            os.path.join(feature_selection_dir, f'scores_t{target_day_str}.csv')
        )
        print(f"    ✓ Feature selection details saved to '{feature_selection_dir}/'")

        # 7.4. Tạo pipeline preprocessing DÀNH RIÊNG cho target này
        
        # Pipeline cho numeric features (chỉ dùng các feature đã chọn)
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('outlier_clipper', OutlierClipper()),
            ('scaler', StandardScaler())
        ])
        
        # Pipeline cho categorical features (dùng các feature mặc định)
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='none')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, selected_numeric_features),
                ('cat', categorical_pipeline, categorical_features_to_keep)
            ],
            remainder='drop' # Chỉ giữ lại các cột đã chọn
        )

        # 7.5. Fit pipeline CHỈ trên X_train
        print(f"  [2] Fitting preprocessor for {target_col_name}...")
        
        # Lấy X_train chỉ với các cột đã chọn (cả số và hạng mục)
        # Hàm preprocessor tự động chọn cột dựa trên định nghĩa ở trên
        # nên chúng ta có thể truyền X_train_full
        preprocessor.fit(X_train_full)
        
        # 7.6. Lưu pipeline đã fit
        pipeline_dir = f'processed_data/pipelines'
        if not os.path.exists(pipeline_dir):
            os.makedirs(pipeline_dir)
        pipeline_filename = os.path.join(pipeline_dir, f'preprocessor_t_{target_day_str}.joblib')
        joblib.dump(preprocessor, pipeline_filename)
        print(f"    ✓ Preprocessor saved to '{pipeline_filename}'")

        # 7.7. Transform X_train, X_dev, X_test
        # preprocessor sẽ tự động lọc các cột trong `final_features_to_keep`
        print(f"  [3] Transforming data for {target_col_name}...")
        
        X_train_trans = preprocessor.transform(X_train_full)
        X_dev_trans = preprocessor.transform(X_dev_full)
        X_test_trans = preprocessor.transform(X_test_full)
        
        # 7.8. Lấy tên cột sau khi transform
        # Lấy tên cột hạng mục đã được one-hot
        try:
            cat_feature_names = preprocessor.named_transformers_['cat'] \
                                            .named_steps['onehot'] \
                                            .get_feature_names_out(categorical_features_to_keep).tolist()
        except AttributeError:
             # Xử lý trường hợp không có đặc trưng hạng mục nào được giữ lại
             cat_feature_names = []

        feature_names = selected_numeric_features + cat_feature_names
        
        print(f"    ✓ Transformation complete: {len(feature_names)} features")

        # 7.9. Convert về DataFrame
        X_train_df = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train_full.index)
        X_dev_df = pd.DataFrame(X_dev_trans, columns=feature_names, index=X_dev_full.index)
        X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test_full.index)

        # 7.10. Lưu dữ liệu đã transform
        data_dir = f'processed_data/target_t_{target_day_str}'
        save_data({
            f'X_train_t{target_day_str}': X_train_df,
            f'X_dev_t{target_day_str}': X_dev_df,
            f'X_test_t{target_day_str}': X_test_df,
            f'y_train_t{target_day_str}': y_train_target.to_frame(), # Chỉ lưu y của target này
            f'y_dev_t{target_day_str}': y_dev_full[[target_col_name]],
            f'y_test_t{target_day_str}': y_test_full[[target_col_name]]
        }, data_dir)
        print(f"    ✓ Transformed data saved to '{data_dir}/'")

    # Lưu lại data gốc đã feature engineering (chưa scale) để tham khảo
    save_data({
        'train_features_cleaned': train_fe_clean,
        'dev_features_cleaned': dev_fe_clean,
        'test_features_cleaned': test_fe_clean
    }, 'processed_data')

    print(f"\n{'='*70}")
    print("✓✓✓ All preprocessing complete for 5 targets! ✓✓✓")
    print(f"{'='*70}")