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

def load_data(file_path):
    """Loads weather data from a CSV file."""
    daily_data = pd.read_csv(file_path)
    daily_data['datetime'] = pd.to_datetime(daily_data['datetime'])
    daily_data.set_index('datetime', inplace=True)
    daily_data = daily_data.sort_index(ascending=True)
    return daily_data

def split_data(daily_data):
    """Splits the data into training, development, and test sets."""
    train_size = int(len(daily_data) * 0.7)
    dev_size = int(len(daily_data) * 0.15)
    
    train_data = daily_data.iloc[:train_size]
    dev_data = daily_data.iloc[train_size:train_size + dev_size]
    test_data = daily_data.iloc[train_size + dev_size:]
    
    return train_data, dev_data, test_data

def remove_leakage_columns(df):
    """
    Remove columns that cause data leakage or are non-informative.
    Keeps descriptive columns for lagged feature creation.
    """
    cols_to_drop = ['tempmax', 'tempmin', 'name', 'stations', 'source', 'season', 'snow', 'snowdepth']
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    return df_clean

def create_day_length_feature(df):
    """
    Transform sunrise/sunset times into day_length feature (in hours).
    """
    df_new = df.copy()
    if 'sunrise' in df_new.columns and 'sunset' in df_new.columns:
        # Parse as time strings (HH:MM:SS format)
        sr = pd.to_datetime(df_new['sunrise'], format='%H:%M:%S', errors='coerce')
        ss = pd.to_datetime(df_new['sunset'], format='%H:%M:%S', errors='coerce')
        # Compute difference in hours
        df_new['day_length_h'] = (ss - sr).dt.total_seconds() / 3600
        # Drop original columns
        df_new = df_new.drop(columns=['sunrise', 'sunset'])
    # If columns don't exist or were already dropped, do nothing
    return df_new

def create_temporal_features(df):
    """
    Create comprehensive time-based features based on EDA insights.
    Captures seasonal patterns, monsoon cycles, and temporal interactions.
    """
    df_new = df.copy()
    
    # Basic datetime components
    df_new['year'] = df_new.index.year
    df_new['quarter'] = df_new.index.quarter
    df_new['month'] = df_new.index.month
    df_new['day_of_year'] = df_new.index.dayofyear
    df_new['day_of_week'] = df_new.index.weekday
    df_new['week_of_year'] = df_new.index.isocalendar().week.astype(int)
    df_new['is_weekend'] = df_new['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encodings for smooth seasonality
    df_new['month_sin'] = np.sin(2 * np.pi * df_new['month'] / 12)
    df_new['month_cos'] = np.cos(2 * np.pi * df_new['month'] / 12)
    df_new['day_sin'] = np.sin(2 * np.pi * df_new['day_of_year'] / 365)
    df_new['day_cos'] = np.cos(2 * np.pi * df_new['day_of_year'] / 365)
    df_new['week_sin'] = np.sin(2 * np.pi * df_new['week_of_year'] / 52)
    df_new['week_cos'] = np.cos(2 * np.pi * df_new['week_of_year'] / 52)
    
    # Season indicators (based on Hanoi's climate)
    df_new['season_winter'] = df_new['month'].isin([12, 1, 2]).astype(int)
    df_new['season_spring'] = df_new['month'].isin([3, 4, 5]).astype(int)
    df_new['season_summer'] = df_new['month'].isin([6, 7, 8]).astype(int)
    df_new['season_autumn'] = df_new['month'].isin([9, 10, 11]).astype(int)
    
    # Monsoon season indicators (from EDA: peak July-August)
    df_new['is_monsoon_peak'] = df_new['month'].isin([7, 8]).astype(int)
    df_new['is_monsoon_season'] = df_new['month'].isin([5, 6, 7, 8, 9]).astype(int)
    df_new['is_dry_season'] = df_new['month'].isin([11, 12, 1, 2, 3]).astype(int)
    
    # Transition periods (rapid weather changes)
    df_new['is_spring_transition'] = df_new['month'].isin([3, 4]).astype(int)
    df_new['is_autumn_transition'] = df_new['month'].isin([9, 10]).astype(int)
    
    # Days since winter solstice (Dec 21, day ~355)
    winter_solstice_day = 355
    df_new['days_since_winter_solstice'] = (df_new['day_of_year'] - winter_solstice_day) % 365
    
    # Days since summer solstice (Jun 21, day ~172)
    summer_solstice_day = 172
    df_new['days_since_summer_solstice'] = (df_new['day_of_year'] - summer_solstice_day) % 365
    
    # Days until monsoon peak (assume Aug 1, day ~213)
    monsoon_peak_day = 213
    df_new['days_to_monsoon_peak'] = np.abs(df_new['day_of_year'] - monsoon_peak_day)
    df_new['days_to_monsoon_peak'] = df_new['days_to_monsoon_peak'].apply(
        lambda x: min(x, 365 - x)  # Circular distance
    )
    
    # Solar angle proxy (simplified - higher in summer, lower in winter)
    # Using day_of_year with offset to summer solstice
    df_new['solar_angle_proxy'] = np.cos(2 * np.pi * (df_new['day_of_year'] - 172) / 365)
    
    # Month groups based on temperature patterns (from EDA)
    # Hot months: May-August (temp > 28°C)
    df_new['is_hot_month'] = df_new['month'].isin([5, 6, 7, 8]).astype(int)
    # Cool months: Dec-Feb (temp < 20°C)
    df_new['is_cool_month'] = df_new['month'].isin([12, 1, 2]).astype(int)
    # Moderate months: Mar-Apr, Sep-Nov
    df_new['is_moderate_month'] = (~df_new['is_hot_month'].astype(bool) & 
                                    ~df_new['is_cool_month'].astype(bool)).astype(int)
    
    # Rainfall season indicators (from EDA precipitation patterns)
    # High rainfall: June-September
    df_new['is_high_rainfall_month'] = df_new['month'].isin([6, 7, 8, 9]).astype(int)
    # Low rainfall: November-March
    df_new['is_low_rainfall_month'] = df_new['month'].isin([11, 12, 1, 2, 3]).astype(int)
    
    return df_new

def feature_engineer_description(df):
    """
    Creates binary features from the 'description' text column.
    """
    df_new = df.copy()
    if 'description' in df_new.columns:
        # Ensure description is a string and handle NaNs
        description_series = df_new['description'].fillna('').astype(str)
        df_new['has_chance_of_rain'] = description_series.str.contains('chance of rain', case=False).astype(int)
        df_new['is_morning_event'] = description_series.str.contains('morning', case=False).astype(int)
        df_new['is_afternoon_event'] = description_series.str.contains('afternoon', case=False).astype(int)
        df_new['is_clearing_later'] = description_series.str.contains('clearing later', case=False).astype(int)
        df_new = df_new.drop(columns=['description'])
    return df_new

def create_lag_features(df):
    """
    Create lag features to capture previous days' conditions.
    """
    df_new = df.copy()
    
    # Define lag configuration
    USE_LAGS = {
        'temp': [1, 2, 3, 7],           # Yesterday, 2-3 days ago, last week
        'feelslike': [1, 3],             # Recent feels-like temperature
        'humidity': [1], 
        'sealevelpressure': [3],         # 3 days ago pressure
        'windspeed': [1, 2]              # Recent wind patterns
    }
    
    for base_col, lags in USE_LAGS.items():
        if base_col in df_new.columns:
            for lag in lags:
                df_new[f'{base_col}_lag{lag}'] = df_new[base_col].shift(lag)
    
    return df_new

def create_rolling_features(df):
    """
    Create rolling window features to capture trends.
    """
    df_new = df.copy()
    
    if 'precip' in df_new.columns:
        df_new['precip_roll_mean_7'] = df_new['precip'].shift(1).rolling(7).mean()
    
    if 'humidity' in df_new.columns:
        df_new['humidity_roll_mean_7'] = df_new['humidity'].shift(1).rolling(7).mean()
    
    if 'windspeed' in df_new.columns:
        df_new['windspeed_roll_max_3'] = df_new['windspeed'].shift(1).rolling(3).max()
    
    return df_new

def create_interaction_features(df):
    """
    Create physically meaningful interaction features.
    """
    df_new = df.copy()
    
    # Heat index approximation: feels-like temperature × humidity
    if {'feelslikemax', 'humidity'}.issubset(df_new.columns):
        df_new['feelslike_humidity_int'] = df_new['feelslikemax'] * df_new['humidity']
    
    # Wind chill effect: feels-like min temperature × wind speed
    if {'feelslikemin', 'windspeed'}.issubset(df_new.columns):
        df_new['windchill_int'] = df_new['feelslikemin'] * df_new['windspeed']
    
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

def encode_categorical_features(df, train_df):
    """
    One-hot encode categorical variables using categories from the training set.
    """
    df_new = df.copy()
    
    cat_cols = [col for col in ['preciptype', 'conditions', 'icon'] if col in df_new.columns]
    
    if not cat_cols:
        return df_new, []

    # Use OrdinalEncoder to handle high cardinality and unseen values gracefully
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Fit on the training data's categories
    encoder.fit(train_df[cat_cols].fillna('none').astype(str))
    
    # Transform the current dataset
    encoded_arr = encoder.transform(df_new[cat_cols].fillna('none').astype(str))
    
    # Create new column names
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    
    # Create encoded dataframe
    df_enc = pd.DataFrame(encoded_arr, columns=encoded_cols, index=df_new.index)
    
    # Concatenate and drop original categorical columns
    df_new = pd.concat([df_new.drop(columns=cat_cols), df_enc], axis=1)
    
    return df_new, encoded_cols

def create_lagged_cat_features(df, encoded_cols, lags=[1, 3]):
    """
    Creates lagged features for the encoded categorical columns.
    """
    df_new = df.copy()
    for col in encoded_cols:
        for lag in lags:
            df_new[f'{col}_lag{lag}'] = df_new[col].shift(lag)
    
    # Drop the original (non-lagged) encoded columns to prevent data leakage
    df_new = df_new.drop(columns=encoded_cols)
    return df_new

def create_multiday_target(df, target_col='temp', days_ahead=5):
    """
    Create prediction target for multiple days ahead.
    """
    df_new = df.copy()
    df_new[f'target_temp_{days_ahead}d'] = df_new[target_col].shift(-days_ahead)
    return df_new

# Custom transformer: IQR-based outlier clipping
class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips outliers using the IQR method.
    Fit on training data to calculate bounds, then apply to all sets.
    """
    def __init__(self, multiplier=1.5):
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

def apply_feature_engineering(df, train_df_for_encoding):
    """Apply all feature engineering steps in sequence."""
    df = remove_leakage_columns(df)
    df = create_day_length_feature(df)
    df = create_temporal_features(df)
    
    # Engineer description and get new column names
    original_cols = set(df.columns)
    df = feature_engineer_description(df)
    desc_cols = list(set(df.columns) - original_cols)

    # One-hot encode and get new column names
    df, ohe_cols = encode_categorical_features(df, train_df_for_encoding)
    
    # Lag all categorical-derived features
    all_cat_derived_cols = list(ohe_cols) + desc_cols
    df = create_lagged_cat_features(df, all_cat_derived_cols)
    df = df.drop(columns=all_cat_derived_cols, errors='ignore') # Drop non-lagged versions

    # Lag numerical features
    numeric_lags = {'temp': [1, 2, 3, 7], 'feelslike': [1, 3], 'humidity': [1], 'sealevelpressure': [3], 'windspeed': [1, 2]}
    df = create_lag_features(df)
    
    df = create_rolling_features(df)
    df = create_interaction_features(df)
    return df

def select_features(X, y, top_n=25):
    """
    Selects the best features using a combination of methods.
    """
    # Pearson correlation
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    
    # Mutual Information
    mi = mutual_info_regression(X.fillna(0), y)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    
    # LassoCV
    lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
    lasso.fit(X.fillna(0), y)
    coef_abs = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
    
    # RandomForest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
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

def preprocess_data():
    """
    Main function to run the entire preprocessing pipeline.
    """
    # Load data
    daily_data = load_data('dataset/hn_daily.csv')
    
    # Split data
    train_data, dev_data, test_data = split_data(daily_data)
    
    # Apply feature engineering
    train_fe = apply_feature_engineering(train_data, train_data)
    dev_fe = apply_feature_engineering(dev_data, train_data)
    test_fe = apply_feature_engineering(test_data, train_data)

    # Create 5-day ahead target
    train_fe = create_multiday_target(train_fe, days_ahead=5)
    dev_fe = create_multiday_target(dev_fe, days_ahead=5)
    test_fe = create_multiday_target(test_fe, days_ahead=5)

    # Drop columns that are entirely NaN
    cols_to_drop_nan = [col for col in train_fe.columns if train_fe[col].isnull().all()]
    if cols_to_drop_nan:
        train_fe = train_fe.drop(columns=cols_to_drop_nan)
        dev_fe = dev_fe.drop(columns=cols_to_drop_nan)
        test_fe = test_fe.drop(columns=cols_to_drop_nan)

    # Drop rows with NaN in critical columns
    critical_cols = [col for col in train_fe.columns if 'lag' in col or 'roll' in col or 'target' in col]
    train_fe_clean = train_fe.dropna(subset=critical_cols)
    dev_fe_clean = dev_fe.dropna(subset=critical_cols)
    test_fe_clean = test_fe.dropna(subset=critical_cols)

    # Separate X and y for feature selection
    X_train_fs = train_fe_clean.drop(columns=['target_temp_5d', 'temp'], errors='ignore')
    y_train_fs = train_fe_clean['target_temp_5d']

    # Select features
    selected_features = select_features(X_train_fs, y_train_fs, top_n=30)
    print(f"Selected {len(selected_features)} features: {selected_features}")

    # Filter datasets with selected features
    X_train = train_fe_clean[selected_features]
    y_train = train_fe_clean['target_temp_5d']

    X_dev = dev_fe_clean[selected_features]
    y_dev = dev_fe_clean['target_temp_5d']

    X_test = test_fe_clean[selected_features]
    y_test = test_fe_clean['target_temp_5d']

    # Identify feature types
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Define preprocessing pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_clipper', OutlierClipper()),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        [('num', numeric_pipeline, numeric_features)],
        remainder='drop'
    )

    # Final preprocessing pipeline
    preprocessing_pipeline = Pipeline([('preprocessor', preprocessor)])

    # Fit and transform data
    preprocessing_pipeline.fit(X_train)
    X_train_trans = preprocessing_pipeline.transform(X_train)
    X_dev_trans = preprocessing_pipeline.transform(X_dev)
    X_test_trans = preprocessing_pipeline.transform(X_test)

    # Get feature names
    feature_names = selected_features

    # Convert to DataFrames, preserving the original index
    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    X_dev_df = pd.DataFrame(X_dev_trans, columns=feature_names, index=X_dev.index)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    # Save transformed data
    save_data({
        'X_train_transformed': X_train_df,
        'X_dev_transformed': X_dev_df,
        'X_test_transformed': X_test_df,
        'y_train': y_train.to_frame(),
        'y_dev': y_dev.to_frame(),
        'y_test': y_test.to_frame()
    }, 'processed_data')

    # Save feature-engineered data
    save_data({
        'train_features': train_fe_clean,
        'dev_features': dev_fe_clean,
        'test_features': test_fe_clean
    }, 'processed_data')

    print("Preprocessing complete. Processed data saved to 'processed_data' folder.")

if __name__ == '__main__':
    preprocess_data()