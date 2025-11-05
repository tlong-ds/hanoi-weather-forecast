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
    (Based on the provided data definition)
    """
    cols_to_drop = [
        'tempmax', 'tempmin', 'feelslikemax', 'feelslikemin', 'feelslike', 
        
        # Non-numeric or descriptive
        'name', 'stations', 'source', 'season',
        'conditions', 'description',
        
        # Other potential drops (can be tested later)
        'preciptype', 'snow', 'snowdepth', 'severerisk'
    ]
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

def create_temporal_features(df):
    """
    Create comprehensive time-based features based on EDA insights.
    Captures seasonal patterns, monsoon cycles, and temporal interactions
    while minimizing redundancy.
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
    
    # 4. Domain-Specific Indicators (Your excellent EDA features)
    
    # Temperature-based seasons
    df_new['is_hot_month'] = df_new['month'].isin([5, 6, 7, 8]).astype(int)
    df_new['is_cool_month'] = df_new['month'].isin([12, 1, 2]).astype(int)
    
    # Rainfall/Monsoon-based seasons
    df_new['is_monsoon_season'] = df_new['month'].isin([5, 6, 7, 8, 9]).astype(int)
    df_new['is_dry_season'] = df_new['month'].isin([11, 12, 1, 2, 3]).astype(int)
    
    # 5. Drop the Redundant Original Columns
    # The _sin/_cos features have replaced them.
    # 'year' and 'is_weekend' are kept as they are not cyclical.
    cols_to_drop = ['month', 'day_of_year', 'week_of_year', 'day_of_week']
    df_new = df_new.drop(columns=cols_to_drop)
    
    return df_new

def create_lag_features(df):
    """
    Create lag features to capture previous days' conditions.
    """
    df_new = df.copy()

    # Define lag configuration
    USE_LAGS = {
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
        df_new['precip_roll_std_7'] = df_new['precip'].shift(1).rolling(7).std()
        df_new['precip_roll_mean_14'] = df_new['precip'].shift(1).rolling(14).mean()
        df_new['precip_roll_std_14'] = df_new['precip'].shift(1).rolling(14).std()
        df_new['precip_roll_mean_21'] = df_new['precip'].shift(1).rolling(21).mean()
        df_new['precip_roll_std_21'] = df_new['precip'].shift(1).rolling(21).std()
        df_new['precip_roll_mean_28'] = df_new['precip'].shift(1).rolling(28).mean()
        df_new['precip_roll_std_28'] = df_new['precip'].shift(1).rolling(28).std()
        df_new['precip_roll_mean_56'] = df_new['precip'].shift(1).rolling(56).mean()
        df_new['precip_roll_std_56'] = df_new['precip'].shift(1).rolling(56).std()
        df_new['precip_roll_mean_84'] = df_new['precip'].shift(1).rolling(84).mean()
        df_new['precip_roll_std_84'] = df_new['precip'].shift(1).rolling(84).std()
    
    if 'humidity' in df_new.columns:
        df_new['humidity_roll_mean_7'] = df_new['humidity'].shift(1).rolling(7).mean()
        df_new['humidity_roll_std_7'] = df_new['humidity'].shift(1).rolling(7).std()
        df_new['humidity_roll_mean_14'] = df_new['humidity'].shift(1).rolling(14).mean()
        df_new['humidity_roll_std_14'] = df_new['humidity'].shift(1).rolling(14).std()
        df_new['humidity_roll_mean_21'] = df_new['humidity'].shift(1).rolling(21).mean()
        df_new['humidity_roll_std_21'] = df_new['humidity'].shift(1).rolling(21).std()
        df_new['humidity_roll_mean_28'] = df_new['humidity'].shift(1).rolling(28).mean()
        df_new['humidity_roll_std_28'] = df_new['humidity'].shift(1).rolling(28).std()
        df_new['humidity_roll_mean_56'] = df_new['humidity'].shift(1).rolling(56).mean()
        df_new['humidity_roll_std_56'] = df_new['humidity'].shift(1).rolling(56).std()
        df_new['humidity_roll_mean_84'] = df_new['humidity'].shift(1).rolling(84).mean()
        df_new['humidity_roll_std_84'] = df_new['humidity'].shift(1).rolling(84).std()
    
    if 'windspeed' in df_new.columns:
        df_new['windspeed_roll_mean_7'] = df_new['windspeed'].shift(1).rolling(7).mean()
        df_new['windspeed_roll_std_7'] = df_new['windspeed'].shift(1).rolling(7).std()
        df_new['windspeed_roll_mean_14'] = df_new['windspeed'].shift(1).rolling(14).mean()
        df_new['windspeed_roll_std_14'] = df_new['windspeed'].shift(1).rolling(14).std()
        df_new['windspeed_roll_mean_21'] = df_new['windspeed'].shift(1).rolling(21).mean()
        df_new['windspeed_roll_std_21'] = df_new['windspeed'].shift(1).rolling(21).std()
        df_new['windspeed_roll_mean_28'] = df_new['windspeed'].shift(1).rolling(28).mean()
        df_new['windspeed_roll_std_28'] = df_new['windspeed'].shift(1).rolling(28).std()
        df_new['windspeed_roll_mean_56'] = df_new['windspeed'].shift(1).rolling(56).mean()
        df_new['windspeed_roll_std_56'] = df_new['windspeed'].shift(1).rolling(56).std()
        df_new['windspeed_roll_mean_84'] = df_new['windspeed'].shift(1).rolling(84).mean()
        df_new['windspeed_roll_std_84'] = df_new['windspeed'].shift(1).rolling(84).std()

    if 'sealevelpressure' in df_new.columns:
        df_new['sealevelpressure_roll_mean_7'] = df_new['sealevelpressure'].shift(1).rolling(7).mean()
        df_new['sealevelpressure_roll_std_7'] = df_new['sealevelpressure'].shift(1).rolling(7).std()
        df_new['sealevelpressure_roll_mean_14'] = df_new['sealevelpressure'].shift(1).rolling(14).mean()
        df_new['sealevelpressure_roll_std_14'] = df_new['sealevelpressure'].shift(1).rolling(14).std()
        df_new['sealevelpressure_roll_mean_21'] = df_new['sealevelpressure'].shift(1).rolling(21).mean()
        df_new['sealevelpressure_roll_std_21'] = df_new['sealevelpressure'].shift(1).rolling(21).std()
        df_new['sealevelpressure_roll_mean_28'] = df_new['sealevelpressure'].shift(1).rolling(28).mean()
        df_new['sealevelpressure_roll_std_28'] = df_new['sealevelpressure'].shift(1).rolling(28).std()
        df_new['sealevelpressure_roll_mean_56'] = df_new['sealevelpressure'].shift(1).rolling(56).mean()
        df_new['sealevelpressure_roll_std_56'] = df_new['sealevelpressure'].shift(1).rolling(56).std()
        df_new['sealevelpressure_roll_mean_84'] = df_new['sealevelpressure'].shift(1).rolling(84).mean()
        df_new['sealevelpressure_roll_std_84'] = df_new['sealevelpressure'].shift(1).rolling(84).std()
    
    if 'cloudcover' in df_new.columns:
        df_new['cloudcover_roll_mean_7'] = df_new['cloudcover'].shift(1).rolling(7).mean()
        df_new['cloudcover_roll_std_7'] = df_new['cloudcover'].shift(1).rolling(7).std()
        df_new['cloudcover_roll_mean_14'] = df_new['cloudcover'].shift(1).rolling(14).mean()
        df_new['cloudcover_roll_std_14'] = df_new['cloudcover'].shift(1).rolling(14).std()
        df_new['cloudcover_roll_mean_21'] = df_new['cloudcover'].shift(1).rolling(21).mean()
        df_new['cloudcover_roll_std_21'] = df_new['cloudcover'].shift(1).rolling(21).std()
        df_new['cloudcover_roll_mean_28'] = df_new['cloudcover'].shift(1).rolling(28).mean()
        df_new['cloudcover_roll_std_28'] = df_new['cloudcover'].shift(1).rolling(28).std()
        df_new['cloudcover_roll_mean_56'] = df_new['cloudcover'].shift(1).rolling(56).mean()
        df_new['cloudcover_roll_std_56'] = df_new['cloudcover'].shift(1).rolling(56).std()
        df_new['cloudcover_roll_mean_84'] = df_new['cloudcover'].shift(1).rolling(84).mean()
        df_new['cloudcover_roll_std_84'] = df_new['cloudcover'].shift(1).rolling(84).std()
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

def encode_categorical_features(df, train_df):
    """
    One-hot encodes the 'icon' column using OneHotEncoder.
    Fills 'none' for missing values.
    """
    df_new = df.copy()
    cat_cols = ['icon']
    
    if 'icon' not in df_new.columns:
        return df_new, []

    # Use OneHotEncoder to handle unseen values gracefully
    # This is more robust than pd.get_dummies
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Fill NaNs with 'none' and fit/transform
    encoded_arr = encoder.fit(df_new[cat_cols].fillna('none').astype(str))
    
    # Fit ONLY on the training data
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

def create_multiday_target(df, target_col='temp', n_steps_out=5):
    """
    Creates a target DataFrame for multi-step forecasting.
    Predicts t+1 through t+n_steps_out.
    """
    df_new = df.copy()
    target_cols = []
    
    # We want to predict t+1 through t+5 (steps 1 to 5 into the future)
    for i in range(1, n_steps_out + 1):
        col_name = f'target_temp_t+{i}'
        df_new[col_name] = df_new[target_col].shift(-i)
        target_cols.append(col_name)
        
    return df_new, target_cols

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

def apply_feature_engineering(df):
    """Apply all feature engineering steps in sequence."""
    df = remove_leakage_columns(df)
    df = create_day_length_feature(df)
    df = create_temporal_features(df)
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

    
# Main function to run preprocessing
if __name__ == '__main__':
    """
    Main function to run the entire preprocessing pipeline.
    **REWRITTEN**: Follows correct order of operations.
    1. Load
    2. Feature Engineer (full dataset)
    3. Create Target (full dataset)
    4. Split
    5. Drop NaNs
    6. Select Features
    7. Fit Pipeline
    8. Transform & Save
    """
    
    # 1. Load data
    print("Loading data...")
    daily_data = load_data('dataset/hn_daily.csv')
    
    # 2. Apply feature engineering (on full dataset)
    print("Applying feature engineering to full dataset...")
    featured_data = apply_feature_engineering(daily_data)
    
    # 3. Create 5-day ahead multi-step target (on full dataset)
    print("Creating multi-step targets...")
    featured_data, target_cols = create_multiday_target(featured_data, target_col='temp', n_steps_out=5)
    print(f"Target columns created: {target_cols}")

    # 4. Split data (NOW we split)
    print("Splitting data into train, dev, and test sets...")
    train_fe, dev_fe, test_fe = split_data(featured_data)

    train_fe, target_cols = create_multiday_target(train_fe, target_col='temp', n_steps_out=5)
    dev_fe, _ = create_multiday_target(dev_fe, target_col='temp', n_steps_out=5)
    test_fe, _ = create_multiday_target(test_fe, target_col='temp', n_steps_out=5)
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

    print(f"Train shape after NaN drop: {train_fe_clean.shape}")

    # 6. Separate X and y for feature selection
    fs_target_col = 'target_temp_t+5'
    all_target_drop_cols = target_cols + ['temp'] 
    
    X_train_fs = train_fe_clean.drop(columns=all_target_drop_cols, errors='ignore')
    y_train_fs = train_fe_clean[fs_target_col]

    # 7. Select features
    # Separate numeric and categorical for selection
    numeric_fs_cols = X_train_fs.select_dtypes(include=[np.number]).columns.tolist()
    
    # **Run selection ONLY on numeric features**
    print("Running feature selection on numeric features...")
    selected_numeric_features = select_features(X_train_fs[numeric_fs_cols], y_train_fs, top_n=30)
    print(f"Selected {len(selected_numeric_features)} numeric features.")
    
    # **Define categorical features to keep**
    categorical_features = ['icon'] # Add any other categorical cols here

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
    
    print("Transforming X_train, X_dev, and X_test...")
    X_train_trans = preprocessor.transform(X_train)
    X_dev_trans = preprocessor.transform(X_dev)
    X_test_trans = preprocessor.transform(X_test)

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

    print("Preprocessing complete.")