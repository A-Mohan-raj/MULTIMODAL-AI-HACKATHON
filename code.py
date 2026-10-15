# ================================================================
# AGRISENSE AI — ONE-CELL GOOGLE COLAB MVP
# Crop Yield Prediction + Mandi Price Forecasting
# Run this entire cell in Google Colab.
#
# IMPORTANT:
# - Demo mode uses SYNTHETIC data and is only for testing the workflow.
# - Upload real datasets to evaluate real-world performance.
# - Synthetic-data metrics must never be presented as real model accuracy.
# ================================================================

import os, sys, json, shutil, warnings, subprocess, importlib.util
warnings.filterwarnings("ignore")

# Install any missing packages
required = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
}
for module, package in required.items():
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from google.colab import files

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PROJECT_DIR = "/content/agrisense_ai"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 70)
print("🌾 AGRISENSE AI — CROP YIELD + MANDI PRICE FORECASTING")
print("=" * 70)
print("Choose data mode:")
print("  1 = DEMO MODE (synthetic data; runs immediately)")
print("  2 = REAL DATA MODE (upload two CSV files)")
mode = input("Enter 1 or 2 [default 1]: ").strip() or "1"
if mode not in ("1", "2"):
    mode = "1"

# ----------------------------
# Helpers
# ----------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_r2(y_true, y_pred):
    try:
        return float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    except Exception:
        return float("nan")

def normalize_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        for c in df.columns
    ]
    return df

def find_col(df, candidates, required=True, label="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"Could not identify {label}. Expected one of {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None

def save_metrics(path, metrics):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=True)

# ----------------------------
# 1. DATA: DEMO OR UPLOAD
# ----------------------------
if mode == "1":
    print("\n⚠️ DEMO MODE: generating SYNTHETIC data. Do not present these as real observations.")
    # Synthetic district-season agricultural data for a turmeric yield demonstration.
    n_yield = 650
    rng = np.random.default_rng(RANDOM_STATE)
    rainfall = rng.normal(850, 180, n_yield).clip(300, 1500)
    avg_temp = rng.normal(27.5, 2.4, n_yield).clip(21, 35)
    ndvi = rng.uniform(0.25, 0.88, n_yield)
    soil_ph = rng.normal(6.4, 0.7, n_yield).clip(4.5, 8.5)
    irrigation = rng.choice(["Rainfed", "Partial", "Full"], n_yield, p=[0.25, 0.40, 0.35])
    season = rng.choice(["Monsoon", "Post-Monsoon", "Winter"], n_yield)
    # Synthetic formula with noise; NOT a scientific or calibrated yield equation.
    irrigation_effect = pd.Series(irrigation).map({"Rainfed": -1.2, "Partial": 0.2, "Full": 1.1}).to_numpy()
    season_effect = pd.Series(season).map({"Monsoon": 0.7, "Post-Monsoon": 0.3, "Winter": -0.2}).to_numpy()
    yield_t_ha = (
        2.5 + 0.006 * rainfall - 0.10 * np.abs(avg_temp - 27)
        + 5.0 * ndvi - 0.30 * np.abs(soil_ph - 6.5)
        + irrigation_effect + season_effect + rng.normal(0, 0.8, n_yield)
    ).clip(2, 20)
    yield_df = pd.DataFrame({
        "rainfall_mm": rainfall,
        "avg_temp_c": avg_temp,
        "ndvi": ndvi,
        "soil_ph": soil_ph,
        "irrigation": irrigation,
        "season": season,
        "yield_tonnes_per_hectare": yield_t_ha
    })

    # Synthetic daily mandi series. Trend + seasonality + noise.
    n_days = 900
    dates = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=n_days, freq="D")
    t = np.arange(n_days)
    price = 8200 + 0.9*t + 650*np.sin(2*np.pi*t/365.25) + 220*np.sin(2*np.pi*t/30) + rng.normal(0, 260, n_days)
    price = np.maximum(price, 3000)
    arrivals = np.maximum(1, 40 + 12*np.sin(2*np.pi*t/180) + rng.normal(0, 8, n_days))
    price_df = pd.DataFrame({
        "date": dates,
        "market": "DEMO_MARKET_NOT_REAL_ATTUR_DATA",
        "commodity": "Turmeric",
        "modal_price": price.round(2),
        "arrivals": arrivals.round(2)
    })
    yield_df.to_csv(os.path.join(DATA_DIR, "yield_demo_synthetic.csv"), index=False)
    price_df.to_csv(os.path.join(DATA_DIR, "mandi_demo_synthetic.csv"), index=False)
else:
    print("\nUpload TWO CSV files when prompted:")
    print("1) Yield CSV: target column should be yield_tonnes_per_hectare or yield")
    print("2) Mandi CSV: needs date and modal_price (or equivalent price column)")
    print("You can select multiple files in the upload dialog.")
    uploaded = files.upload()
    if len(uploaded) < 2:
        raise ValueError("Please upload at least two CSV files: one yield dataset and one mandi price dataset.")
    csv_names = [name for name in uploaded if name.lower().endswith(".csv")]
    if len(csv_names) < 2:
        raise ValueError("Uploaded files must include at least two .csv files.")
    # Guess the files by column names.
    dfs = {}
    for name in csv_names:
        try:
            dfs[name] = normalize_columns(pd.read_csv(name))
        except Exception as e:
            print(f"Could not read {name}: {e}")
    yield_df = None
    price_df = None
    for name, df in dfs.items():
        has_date = any(c in df.columns for c in ["date", "arrival_date", "arrival_date_", "reported_date"])
        has_price = any(c in df.columns for c in ["modal_price", "modal_price_rs_quintal", "price", "price_rs_quintal", "max_price"])
        has_yield = any(c in df.columns for c in ["yield_tonnes_per_hectare", "yield", "yield_t_ha", "production_yield"])
        if has_date and has_price:
            price_df = df.copy()
        elif has_yield:
            yield_df = df.copy()
    if yield_df is None or price_df is None:
        print("\nCould not automatically identify both datasets.")
        print("Available uploaded files and columns:")
        for name, df in dfs.items():
            print(name, "=>", list(df.columns))
        raise ValueError(
            "Rename the yield target to 'yield_tonnes_per_hectare' and the mandi date/price columns to "
            "'date' and 'modal_price', then rerun and upload again."
        )

# Normalize columns
yield_df = normalize_columns(yield_df)
price_df = normalize_columns(price_df)

# ----------------------------
# 2. YIELD MODEL
# ----------------------------
print("\n" + "="*70)
print("🌱 CROP YIELD MODEL")
print("="*70)

target_candidates = [
    "yield_tonnes_per_hectare", "yield_t_ha", "yield", "yield_per_hectare",
    "crop_yield", "production_yield", "yield_tons_per_hectare"
]
yield_target = find_col(yield_df, target_candidates, label="yield target")
yield_df[yield_target] = pd.to_numeric(yield_df[yield_target], errors="coerce")
yield_df = yield_df.dropna(subset=[yield_target]).copy()

if len(yield_df) < 20:
    raise ValueError("Yield dataset needs at least 20 valid target rows for a basic train/test evaluation.")

# Drop obvious identifiers/free-text fields and target leakage columns.
drop_candidates = [yield_target, "id", "record_id", "farmer_name", "address", "notes", "production_tonnes"]
feature_cols = [c for c in yield_df.columns if c not in drop_candidates]
feature_cols = [c for c in feature_cols if yield_df[c].nunique(dropna=True) > 1]
if not feature_cols:
    raise ValueError("No usable yield features found. Include relevant numeric or categorical predictor columns.")

X = yield_df[feature_cols].copy()
y = yield_df[yield_target].astype(float)

# Infer types. Treat low-cardinality text as categorical; discard near-unique text.
cat_cols, num_cols = [], []
for c in X.columns:
    if pd.api.types.is_numeric_dtype(X[c]):
        num_cols.append(c)
    else:
        unique_ratio = X[c].nunique(dropna=True) / max(1, X[c].notna().sum())
        if X[c].nunique(dropna=True) <= 30 and unique_ratio < 0.5:
            cat_cols.append(c)
        else:
            X = X.drop(columns=[c])

feature_cols = list(X.columns)
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

transformers = []
if num_cols:
    transformers.append(("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols))
preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

# Random split for yield is acceptable only if rows are independent; for grouped/time-dependent data, adapt split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
yield_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    ))
])
yield_pipeline.fit(X_train, y_train)
yield_pred = yield_pipeline.predict(X_test)
yield_metrics = {
    "model": "RandomForestRegressor",
    "target": yield_target,
    "feature_columns": feature_cols,
    "n_rows": int(len(yield_df)),
    "test_rows": int(len(y_test)),
    "MAE": float(mean_absolute_error(y_test, yield_pred)),
    "RMSE": rmse(y_test, yield_pred),
    "R2": safe_r2(y_test, yield_pred),
    "data_mode": "synthetic_demo" if mode == "1" else "uploaded_data"
}
joblib.dump(yield_pipeline, os.path.join(MODEL_DIR, "yield_model.joblib"))
with open(os.path.join(MODEL_DIR, "yield_metadata.json"), "w", encoding="utf-8") as f:
    json.dump({
        "target": yield_target,
        "feature_columns": feature_cols,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "data_mode": yield_metrics["data_mode"],
        "target_unit": "as provided by uploaded dataset; demo uses tonnes/hectare"
    }, f, indent=2)

print("Rows:", len(yield_df), "| Features:", feature_cols)
print("MAE:", round(yield_metrics["MAE"], 4))
print("RMSE:", round(yield_metrics["RMSE"], 4))
print("R²:", round(yield_metrics["R2"], 4))
print("Saved:", os.path.join(MODEL_DIR, "yield_model.joblib"))

plt.figure(figsize=(7, 5))
plt.scatter(y_test, yield_pred, alpha=0.7)
lo = min(float(np.min(y_test)), float(np.min(yield_pred)))
hi = max(float(np.max(y_test)), float(np.max(yield_pred)))
plt.plot([lo, hi], [lo, hi], linestyle="--")
plt.xlabel("Actual yield")
plt.ylabel("Predicted yield")
plt.title("Crop Yield: Actual vs Predicted" + (" (SYNTHETIC DEMO)" if mode == "1" else ""))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "yield_actual_vs_predicted.png"), dpi=160)
plt.show()

# ----------------------------
# 3. PRICE FORECASTING
# ----------------------------
print("\n" + "="*70)
print("📈 MANDI PRICE FORECASTING")
print("="*70)

date_col = find_col(price_df, ["date", "arrival_date", "reported_date", "market_date"], label="date")
price_col = find_col(price_df, ["modal_price", "modal_price_rs_quintal", "price", "price_rs_quintal", "max_price"], label="price")
price_df[date_col] = pd.to_datetime(price_df[date_col], errors="coerce")
price_df[price_col] = pd.to_numeric(price_df[price_col].astype(str).str.replace(",", ""), errors="coerce")
price_df = price_df.dropna(subset=[date_col, price_col]).copy()
price_df = price_df.sort_values(date_col)
# If several rows exist per date, average prices by date for this simple first version.
series = price_df.groupby(date_col)[price_col].mean().sort_index()
series = series[~series.index.duplicated(keep="last")]

if len(series) < 40:
    raise ValueError("Price dataset needs at least 40 valid dates for a basic time-ordered demonstration.")

# Resample to daily calendar frequency and interpolate short gaps. For real data, interpret this carefully.
daily = series.asfreq("D")
daily = daily.interpolate(method="time", limit=7)
daily = daily.dropna()
if len(daily) < 40:
    raise ValueError("Not enough contiguous price data after date cleaning. Provide more historical records.")

# Supervised lag features: past prices only.
price_frame = pd.DataFrame({"price": daily})
for lag in [1, 2, 3, 7, 14, 30]:
    price_frame[f"lag_{lag}"] = price_frame["price"].shift(lag)
price_frame["rolling_mean_7"] = price_frame["price"].shift(1).rolling(7).mean()
price_frame["rolling_mean_30"] = price_frame["price"].shift(1).rolling(30).mean()
price_frame["day_of_year"] = price_frame.index.dayofyear
price_frame["month"] = price_frame.index.month
price_frame["day_of_week"] = price_frame.index.dayofweek
price_frame = price_frame.dropna()

if len(price_frame) < 30:
    raise ValueError("Not enough usable rows after creating lag features. Supply a longer price history.")

X_price = price_frame.drop(columns=["price"])
y_price = price_frame["price"]
split_idx = int(len(price_frame) * 0.8)
Xtr, Xte = X_price.iloc[:split_idx], X_price.iloc[split_idx:]
ytr, yte = y_price.iloc[:split_idx], y_price.iloc[split_idx:]

price_model = RandomForestRegressor(
    n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
)
price_model.fit(Xtr, ytr)
pred_rf = price_model.predict(Xte)
# Baseline: last observed value carried forward for each test date.
pred_naive = Xte["lag_1"].to_numpy()

price_metrics = {
    "model": "RandomForestRegressor with lag features",
    "target": price_col,
    "n_observations": int(len(daily)),
    "train_rows": int(len(Xtr)),
    "test_rows": int(len(Xte)),
    "MAE": float(mean_absolute_error(yte, pred_rf)),
    "RMSE": rmse(yte, pred_rf),
    "R2": safe_r2(yte, pred_rf),
    "naive_baseline_MAE": float(mean_absolute_error(yte, pred_naive)),
    "naive_baseline_RMSE": rmse(yte, pred_naive),
    "last_observed_date": str(daily.index[-1].date()),
    "data_mode": "synthetic_demo" if mode == "1" else "uploaded_data"
}
joblib.dump(price_model, os.path.join(MODEL_DIR, "price_model.joblib"))
with open(os.path.join(MODEL_DIR, "price_metadata.json"), "w", encoding="utf-8") as f:
    json.dump({
        "feature_columns": list(X_price.columns),
        "price_column": price_col,
        "date_column": date_col,
        "data_mode": price_metrics["data_mode"],
        "price_unit": "verify uploaded dataset unit; commonly INR/quintal in Indian mandi datasets",
        "last_observed_date": price_metrics["last_observed_date"]
    }, f, indent=2)

print("Price observations:", len(daily))
print("Chronological train/test rows:", len(Xtr), "/", len(Xte))
print("Model MAE:", round(price_metrics["MAE"], 2))
print("Model RMSE:", round(price_metrics["RMSE"], 2))
print("Naive baseline MAE:", round(price_metrics["naive_baseline_MAE"], 2))
print("Naive baseline RMSE:", round(price_metrics["naive_baseline_RMSE"], 2))
print("Saved:", os.path.join(MODEL_DIR, "price_model.joblib"))

plt.figure(figsize=(11, 5))
plt.plot(yte.index, yte.values, label="Observed test prices")
plt.plot(yte.index, pred_rf, label="Model predictions")
plt.plot(yte.index, pred_naive, label="Naive baseline", alpha=0.8)
plt.xlabel("Date")
plt.ylabel(f"Price ({price_col}; verify unit)")
plt.title("Mandi Price: Time-Ordered Test Evaluation" + (" (SYNTHETIC DEMO)" if mode == "1" else ""))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "price_test_forecast.png"), dpi=160)
plt.show()

# ----------------------------
# 4. RECURSIVE FUTURE FORECAST
# ----------------------------
forecast_days = 7
last_history = daily.copy()
future_dates = pd.date_range(last_history.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="D")
future_predictions = []

for dt in future_dates:
    vals = {}
    for lag in [1, 2, 3, 7, 14, 30]:
        vals[f"lag_{lag}"] = float(last_history.iloc[-lag])
    vals["rolling_mean_7"] = float(last_history.iloc[-7:].mean())
    vals["rolling_mean_30"] = float(last_history.iloc[-30:].mean())
    vals["day_of_year"] = int(dt.dayofyear)
    vals["month"] = int(dt.month)
    vals["day_of_week"] = int(dt.dayofweek)
    xrow = pd.DataFrame([vals], columns=X_price.columns)
    forecast_value = float(price_model.predict(xrow)[0])
    future_predictions.append(forecast_value)
    last_history.loc[dt] = forecast_value

forecast_df = pd.DataFrame({
    "forecast_date": future_dates,
    "predicted_price": np.round(future_predictions, 2),
    "data_mode": "synthetic_demo" if mode == "1" else "uploaded_data"
})
forecast_df.to_csv(os.path.join(OUTPUT_DIR, "7_day_price_forecast.csv"), index=False)
print("\nNext 7 days forecast:")
display(forecast_df)

plt.figure(figsize=(10, 4))
recent = daily.tail(90)
plt.plot(recent.index, recent.values, label="Observed history")
plt.plot(future_dates, future_predictions, marker="o", linestyle="--", label="7-day model forecast")
plt.xlabel("Date")
plt.ylabel(f"Price ({price_col}; verify unit)")
plt.title("7-Day Mandi Price Forecast" + (" (SYNTHETIC DEMO)" if mode == "1" else ""))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "price_7_day_forecast.png"), dpi=160)
plt.show()

# ----------------------------
# 5. SAVE METRICS + SAMPLE INPUT
# ----------------------------
all_metrics = {
    "project": "AgriSense AI",
    "warning": (
        "SYNTHETIC DEMONSTRATION ONLY — not real-world evidence."
        if mode == "1" else
        "Metrics are based on uploaded data. Verify source, units, coverage, and data quality before use."
    ),
    "yield_model": yield_metrics,
    "price_model": price_metrics
}
save_metrics(os.path.join(OUTPUT_DIR, "model_metrics.json"), all_metrics)

# Save clean datasets used for reproducibility
yield_df.to_csv(os.path.join(DATA_DIR, "yield_data_used.csv"), index=False)
price_df.to_csv(os.path.join(DATA_DIR, "mandi_price_data_used.csv"), index=False)

# Make a sample yield input row for the UI, preserving original feature schema.
sample_input = X.iloc[[0]].copy()
sample_input.to_csv(os.path.join(OUTPUT_DIR, "yield_sample_input.csv"), index=False)

# Add README for app integration
readme = f"""# AgriSense AI model artifacts

## Important data status
- Mode: {"SYNTHETIC DEMO DATA — NOT REAL AGRICULTURAL OR MARKET OBSERVATIONS" if mode == "1" else "USER-UPLOADED DATA; VERIFY SOURCE AND COVERAGE"}
- Do not represent synthetic-demo predictions or metrics as validated performance.

## Artifacts
- `models/yield_model.joblib`: fitted yield prediction pipeline
- `models/yield_metadata.json`: yield feature schema
- `models/price_model.joblib`: fitted price lag-feature model
- `models/price_metadata.json`: price model metadata
- `outputs/model_metrics.json`: evaluation metrics
- `outputs/7_day_price_forecast.csv`: seven-day demonstration forecast
- `outputs/yield_sample_input.csv`: example yield input row

## Yield prediction integration
Load `yield_model.joblib` with joblib.load, read the feature columns from `yield_metadata.json`,
construct a one-row pandas DataFrame with those exact feature names, and call model.predict(input_df).

## Price forecast integration
The price model expects the lag and calendar columns listed in `price_metadata.json`.
To forecast recursively, build each next day's feature row using only observed history and earlier
predictions, then append each prediction to the history before creating the next row.

## Run the UI
Use the separately generated Streamlit app with these artifacts placed in its `models/` and `outputs/` folders.
"""
with open(os.path.join(PROJECT_DIR, "README_MODEL_ARTIFACTS.md"), "w", encoding="utf-8") as f:
    f.write(readme)

# Package everything for download / upload into Antigravity
zip_path = shutil.make_archive("/content/agrisense_ai_artifacts", "zip", PROJECT_DIR)

print("\n" + "="*70)
print("✅ PIPELINE FINISHED")
print("="*70)
print("Project folder:", PROJECT_DIR)
print("Artifact ZIP:", zip_path)
print("\nDATA STATUS:", all_metrics["warning"])
print("\nFiles to integrate into your Antigravity project:")
print("  models/yield_model.joblib")
print("  models/yield_metadata.json")
print("  models/price_model.joblib")
print("  models/price_metadata.json")
print("  outputs/model_metrics.json")
print("  outputs/7_day_price_forecast.csv")
print("\nDownload the complete artifact package now.")
files.download(zip_path)
