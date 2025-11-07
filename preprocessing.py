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
N_STEPS_AHEAD = 10 # Predict 10 days ahead
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

def select_features(X, y, top_n=FEATURE_SELECTION_TOP_N, lasso_cv=LASSO_CV_FOLDS, 
                   rf_n_est=RANDOM_FOREST_N_ESTIMATORS, random_state=RANDOM_STATE):
    """
    Selects the best features using a combination of methods.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        top_n: Number of features to select
        lasso_cv: Cross-validation folds for LassoCV
        rf_n_est: Number of estimators for RandomForestRegressor
        random_state: Random seed for reproducibility
    """
    # Pearson correlation
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    
    # Mutual Information
    mi = mutual_info_regression(X.fillna(0), y)
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
    ranked = rank_df.sort_values('mean_rank')
    
    selected_features = ranked.head(top_n).index.tolist()
    return selected_features

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
    fs_target_col = f'target_{TARGET_COLUMN}_t+{N_STEPS_AHEAD}'
    all_target_drop_cols = target_cols + [TARGET_COLUMN] 
    
    X_train_fs = train_fe_clean.drop(columns=all_target_drop_cols, errors='ignore')
    y_train_fs = train_fe_clean[fs_target_col]

    # 7. Select features
    # Separate numeric and categorical for selection
    numeric_fs_cols = X_train_fs.select_dtypes(include=[np.number]).columns.tolist()
    
    # **Run selection ONLY on numeric features**
    print(f"Running feature selection on numeric features (selecting top {FEATURE_SELECTION_TOP_N})...")
    selected_numeric_features = select_features(X_train_fs[numeric_fs_cols], y_train_fs)
    print(f"✓ Selected {len(selected_numeric_features)} numeric features.")
    
    # Validate that we got features
    if len(selected_numeric_features) == 0:
        print("⚠️ WARNING: No numeric features selected! Check your data.")
        selected_numeric_features = numeric_fs_cols[:FEATURE_SELECTION_TOP_N]  # Fallback
    
    # **Define categorical features to keep**
    categorical_features = CATEGORICAL_FEATURES  # Defaults to ['icon']

    # 8. Filter datasets with selected features
    final_features_to_keep = selected_numeric_features + categorical_features
    
    X_train = train_fe_clean[final_features_to_keep]
    y_train = train_fe_clean[target_cols]

    X_dev = dev_fe_clean[final_features_to_keep]
    y_dev = dev_fe_clean[target_cols]

    X_test = test_fe_clean[final_features_to_keep]
    y_test = test_fe_clean[target_cols]

    # 9. Define preprocessing pipelines (NOW with OneHotEncoder)
    
    # Pipeline for numeric features
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_clipper', OutlierClipper()),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for categorical features
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='none')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine pipelines in ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, selected_numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    # 10. Fit and transform data
    print("Fitting pipeline on X_train...")
    # Fit the preprocessor pipeline ONLY on X_train
    preprocessor.fit(X_train)
    
    print("✓ Transforming X_train, X_dev, and X_test...")
    X_train_trans = preprocessor.transform(X_train)
    X_dev_trans = preprocessor.transform(X_dev)
    X_test_trans = preprocessor.transform(X_test)
    print(f"✓ Transformation complete: {X_train_trans.shape[1]} features")

    # 11. Get feature names back from the pipeline
    # This is crucial as OneHotEncoder creates new columns
    feature_names = (
        selected_numeric_features + 
        preprocessor.named_transformers_['cat']
                    .named_steps['onehot']
                    .get_feature_names_out(categorical_features).tolist()
    )

    # 12. Convert to DataFrames, preserving the original index
    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    X_dev_df = pd.DataFrame(X_dev_trans, columns=feature_names, index=X_dev.index)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    # 13. Save transformed data
    print("Saving processed data to 'processed_data' folder...")
    save_data({
        'X_train_transformed': X_train_df,
        'X_dev_transformed': X_dev_df,
        'X_test_transformed': X_test_df,
        'y_train': y_train,
        'y_dev': y_dev,
        'y_test': y_test
    }, 'processed_data')

    # Save feature-engineered data (before scaling/encoding)
    save_data({
        'train_features': train_fe_clean,
        'dev_features': dev_fe_clean,
        'test_features': test_fe_clean
    }, 'processed_data')

    print("✓ Preprocessing complete!")