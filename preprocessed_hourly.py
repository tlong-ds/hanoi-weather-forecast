# preprocessing_hourly.py
# hourly preprocessing (24-step ahead)
# ALWAYS saves directly into:  <repo>/data_processing_hourly/
# all output CSV filenames include "_hourly"

import os, sys, traceback, argparse
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_regression

# ==== CONFIG ====
DEFAULT_DATA_PATH = r"C:\Users\lenovo\Downloads\python exercises\ml 2\hanoi-weather-forecast\dataset\hn_hourly.csv"
DEFAULT_HORIZON   = 24  # t+1..t+24 hours

# ==== SINGLE OUTPUT FOLDER (FLAT) ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "data_processing_hourly")  # << only this folder
os.makedirs(SAVE_DIR, exist_ok=True)
print(">> RUNNING FILE:", os.path.abspath(__file__))
print(">> NOTEBOOK CWD:", os.getcwd())
print(">> Will save EVERYTHING to (flat):", SAVE_DIR)

def save_data_hourly(data: dict, folder: str) -> None:
    """Write DataFrames to CSV under 'folder'; filenames include '_hourly'."""
    os.makedirs(folder, exist_ok=True)
    for name, df in data.items():
        out_path = os.path.join(folder, f"{name}_hourly.csv")
        df.to_csv(out_path)
        print("   - saved:", os.path.abspath(out_path))

# ==== IO ====
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    df = df[~df.index.duplicated(keep="first")]
    print(">> Loaded:", df.shape, "from", file_path)
    return df

# ==== SPLIT ====
def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_tr = int(n * 0.7)
    n_dev = int(n * 0.15)
    train = df.iloc[:n_tr]
    dev   = df.iloc[n_tr:n_tr + n_dev]
    test  = df.iloc[n_tr + n_dev:]
    return train, dev, test

# ==== CLEAN / LEAKAGE ====
def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        "tempmax","tempmin","feelslikemax","feelslikemin","feelslike",
        "name","stations","source","season","conditions","description",
        "preciptype","snow","snowdepth","severerisk"
    ]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

# ==== FEATURES ====
def create_day_length_feature(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    if "sunrise" in df_new.columns and "sunset" in df_new.columns:
        sr = pd.to_datetime(df_new["sunrise"], errors="coerce")
        ss = pd.to_datetime(df_new["sunset"], errors="coerce")
        df_new["day_length_h"] = (ss - sr).dt.total_seconds() / 3600.0
        df_new = df_new.drop(columns=["sunrise","sunset"])
    return df_new

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    df_new["year"] = df_new.index.year
    df_new["month"] = df_new.index.month
    df_new["day_of_year"] = df_new.index.dayofyear
    df_new["week_of_year"] = df_new.index.isocalendar().week.astype(int)
    df_new["hour"] = df_new.index.hour
    df_new["is_night"] = df_new["hour"].isin([0,1,2,3,4,5]).astype(int)

    df_new["month_sin"] = np.sin(2*np.pi*df_new["month"]/12)
    df_new["month_cos"] = np.cos(2*np.pi*df_new["month"]/12)
    df_new["day_sin"]   = np.sin(2*np.pi*df_new["day_of_year"]/365)
    df_new["day_cos"]   = np.cos(2*np.pi*df_new["day_of_year"]/365)
    df_new["week_sin"]  = np.sin(2*np.pi*df_new["week_of_year"]/52)
    df_new["week_cos"]  = np.cos(2*np.pi*df_new["week_of_year"]/52)
    df_new["hour_sin"]  = np.sin(2*np.pi*df_new["hour"]/24)
    df_new["hour_cos"]  = np.cos(2*np.pi*df_new["hour"]/24)

    df_new["is_hot_month"]   = df_new["month"].isin([5,6,7,8]).astype(int)
    df_new["is_cool_month"]  = df_new["month"].isin([12,1,2]).astype(int)
    df_new["is_monsoon_season"] = df_new["month"].isin([5,6,7,8,9]).astype(int)
    df_new["is_dry_season"]     = df_new["month"].isin([11,12,1,2,3]).astype(int)

    df_new = df_new.drop(columns=["month","day_of_year","week_of_year"], errors="ignore")
    return df_new

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    lags = {
        "temp": [1,3,6,12,24],
        "humidity": [1,6,24],
        "sealevelpressure": [6,24],
        "windspeed": [1,3,6,24],
    }
    for base_col, ls in lags.items():
        if base_col in df_new.columns:
            for lag in ls:
                df_new[f"{base_col}_lag{lag}"] = df_new[base_col].shift(lag)
    return df_new

def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    if "precip" in df_new.columns:
        df_new["precip_roll_mean_6h"]  = df_new["precip"].shift(1).rolling(6).mean()
        df_new["precip_roll_mean_24h"] = df_new["precip"].shift(1).rolling(24).mean()
    if "humidity" in df_new.columns:
        df_new["humidity_roll_mean_6h"]  = df_new["humidity"].shift(1).rolling(6).mean()
        df_new["humidity_roll_mean_24h"] = df_new["humidity"].shift(1).rolling(24).mean()
    if "windspeed" in df_new.columns:
        df_new["windspeed_roll_max_3h"]  = df_new["windspeed"].shift(1).rolling(3).max()
        df_new["windspeed_roll_max_24h"] = df_new["windspeed"].shift(1).rolling(24).max()
    return df_new

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    if "windspeed" in df_new.columns:
        df_new["windspeed_sq"] = df_new["windspeed"] ** 2
    if {"sealevelpressure","humidity"}.issubset(df_new.columns):
        df_new["pressure_humidity"] = df_new["sealevelpressure"] * df_new["humidity"]
    if "day_length_h" in df_new.columns and "uvindex" in df_new.columns:
        df_new["daylength_uv"] = df_new["day_length_h"] * df_new["uvindex"]
    return df_new

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_leakage_columns(df)
    df = create_day_length_feature(df)
    df = create_temporal_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_interaction_features(df)
    return df

# ==== TARGETS ====
def create_multistep_hourly_target(df: pd.DataFrame, target_col: str, n_steps_out: int) -> Tuple[pd.DataFrame, List[str]]:
    df_new = df.copy()
    target_cols = []
    for i in range(1, n_steps_out + 1):
        name = f"target_temp_t+{i}h"
        df_new[name] = df_new[target_col].shift(-i)
        target_cols.append(name)
    return df_new, target_cols

# ==== TRANSFORMERS ====
class OutlierClipper(BaseEstimator, TransformerMixin):
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

# ==== FEATURE SELECTION ====
def select_features(X: pd.DataFrame, y: pd.Series, top_n: int = 30) -> List[str]:
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    mi = mutual_info_regression(X.fillna(0), y)
    mi_s = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    lasso = LassoCV(cv=5, random_state=42, n_jobs=-1).fit(X.fillna(0), y)
    coef_abs = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X.fillna(0), y)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    rank_df = pd.DataFrame({
        "corr_rank": corrs.rank(ascending=False),
        "mi_rank": mi_s.rank(ascending=False),
        "lasso_rank": coef_abs.rank(ascending=False),
        "rf_rank": rf_imp.rank(ascending=False),
    }).fillna(1e6)
    rank_df["mean_rank"] = rank_df.mean(axis=1)
    return rank_df.sort_values("mean_rank").head(top_n).index.tolist()

# ==== MAIN ====
def main(args):
    print(">> Loading hourly data …")
    df = load_data(args.data)

    print(">> Feature engineering (hourly) …")
    featured = apply_feature_engineering(df)

    print(f">> Creating {args.horizon}-step hourly targets …")
    featured, target_cols = create_multistep_hourly_target(featured, "temp", args.horizon)
    print(f">> Target columns (sample): {target_cols[:5]} … total={len(target_cols)}")

    print(">> Split into train/dev/test …")
    train_fe, dev_fe, test_fe = split_data(featured)

    # re-create targets per split
    train_fe, target_cols = create_multistep_hourly_target(train_fe, "temp", args.horizon)
    dev_fe,   _          = create_multistep_hourly_target(dev_fe,   "temp", args.horizon)
    test_fe,  _          = create_multistep_hourly_target(test_fe,  "temp", args.horizon)

    print(">> Dropping NaNs from critical columns …")
    lag_roll_cols = [c for c in train_fe.columns if ("lag" in c or "roll" in c)]
    critical_cols = lag_roll_cols + target_cols

    fully_nan_cols = [c for c in train_fe.columns if train_fe[c].isna().all()]
    if fully_nan_cols:
        train_fe = train_fe.drop(columns=fully_nan_cols, errors="ignore")
        dev_fe   = dev_fe.drop(columns=fully_nan_cols, errors="ignore")
        test_fe  = test_fe.drop(columns=fully_nan_cols, errors="ignore")

    train_fe_clean = train_fe.dropna(subset=critical_cols)
    dev_fe_clean   = dev_fe.dropna(subset=critical_cols)
    test_fe_clean  = test_fe.dropna(subset=critical_cols)
    print(f">> Train after drop: {train_fe_clean.shape}")

    # feature selection target (farthest horizon)
    fs_target_col = target_cols[-1]
    all_target_drop = target_cols + ["temp"]
    X_train_fs = train_fe_clean.drop(columns=all_target_drop, errors="ignore")
    y_train_fs = train_fe_clean[fs_target_col]

    print(">> Selecting numeric features …")
    numeric_cols = X_train_fs.select_dtypes(include=[np.number]).columns.tolist()
    selected_numeric = select_features(X_train_fs[numeric_cols], y_train_fs, top_n=30)
    print(f">> Selected numeric: {len(selected_numeric)}")

    categorical_features = ["icon"] if "icon" in train_fe_clean.columns else []
    final_keep = selected_numeric + categorical_features

    X_train = train_fe_clean[final_keep]
    y_train = train_fe_clean[target_cols]
    X_dev   = dev_fe_clean[final_keep]
    y_dev   = dev_fe_clean[target_cols]
    X_test  = test_fe_clean[final_keep]
    y_test  = test_fe_clean[target_cols]

    print(">> Fitting preprocessing …")
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clip", OutlierClipper()),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="none")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, selected_numeric),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
    )
    pre.fit(X_train)

    X_train_trans = pre.transform(X_train)
    X_dev_trans   = pre.transform(X_dev)
    X_test_trans  = pre.transform(X_test)

    feat_names = list(selected_numeric)
    if categorical_features:
        feat_names += pre.named_transformers_["cat"].named_steps["onehot"] \
                         .get_feature_names_out(categorical_features).tolist()

    X_train_df = pd.DataFrame(X_train_trans, columns=feat_names, index=X_train.index)
    X_dev_df   = pd.DataFrame(X_dev_trans,   columns=feat_names, index=X_dev.index)
    X_test_df  = pd.DataFrame(X_test_trans,  columns=feat_names, index=X_test.index)

    print(f">> Saving artifacts to '{SAVE_DIR}' …")
    save_data_hourly({
        "X_train_transformed": X_train_df,
        "X_dev_transformed":   X_dev_df,
        "X_test_transformed":  X_test_df,
        "y_train": y_train,
        "y_dev":   y_dev,
        "y_test":  y_test,
    }, SAVE_DIR)

    save_data_hourly({
        "train_features": train_fe_clean,
        "dev_features":   dev_fe_clean,
        "test_features":  test_fe_clean,
    }, SAVE_DIR)

    print(">> Listing saved files in:", SAVE_DIR)
    for fn in os.listdir(SAVE_DIR):
        print("   *", fn)
    print(">> Done. Saved folder:", SAVE_DIR)

# ==== ENTRY ====
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Preprocessing (hourly) for multi-step forecasting")
        parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="absolute CSV path (hourly)")
        parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="forecast horizon steps (default 24)")
        args = parser.parse_args()
        main(args)
    except Exception as e:
        print("!! ERROR during run:", e)
        traceback.print_exc()
        sys.exit(1)
