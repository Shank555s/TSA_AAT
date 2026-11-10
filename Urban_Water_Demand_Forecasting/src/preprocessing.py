"""Clean the raw data, build features and split chronologically 70/15/15."""

import numpy as np
import pandas as pd

from utils import (
    CALENDAR_FEATURES, DATA_FILE, HISTORY_FEATURES, LAGS, PROCESSED_DIR,
    WEATHER_FEATURES, WINDOWS,
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_FILE, sep=";")
df.columns = df.columns.str.strip()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["v-0"] = pd.to_numeric(df["v-0"], errors="coerce")

if df["Date"].isna().any():
    raise ValueError("Invalid dates found.")

if df["Date"].duplicated().any():
    raise ValueError("Duplicate dates found.")

df = df.sort_values("Date").set_index("Date")

# 2. Check timeline continuity
expected_dates = pd.date_range(df.index.min(), df.index.max(), freq="D")
missing_dates = expected_dates.difference(df.index)

print("Missing calendar dates:", len(missing_dates))

if len(missing_dates):
    raise ValueError("Missing dates detected. Inspect before continuing.")

if df["v-0"].isna().any():
    raise ValueError("Missing target values found.")

df["demand"] = df["v-0"]

# 3. Historical demand features (all shifted, so no information leaks
#    from the day being predicted)
for lag in LAGS:
    df[f"lag_{lag}"] = df["demand"].shift(lag)

shifted = df["demand"].shift(1)
for window in WINDOWS:
    df[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
    df[f"rolling_std_{window}"] = shifted.rolling(window).std()

# 4. Calendar features
df["day_of_week"] = df.index.day_of_week
df["month"] = df.index.month
df["is_weekend"] = (df.index.day_of_week >= 5).astype(int)
df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

# 5. Weather features
for col in WEATHER_FEATURES:
    if col not in df.columns:
        raise ValueError(f"Missing weather feature: {col}")
    df[col] = pd.to_numeric(df[col], errors="coerce")

all_features = HISTORY_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES

# 6. Remove rows without required historical information
model_df = df[["demand"] + all_features].dropna().copy()

print("Original rows:", len(df))
print("Usable rows:", len(model_df))
print("Removed rows:", len(df) - len(model_df))

# 7. Chronological split (no shuffling)
n = len(model_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

splits = {
    "train": model_df.iloc[:train_end],
    "validation": model_df.iloc[train_end:val_end],
    "test": model_df.iloc[val_end:],
}

for name, part in splits.items():
    part.to_csv(PROCESSED_DIR / f"{name}.csv")

print("\nPREPROCESSING COMPLETE")
print("-" * 45)

for name, part in splits.items():
    print(
        f"{name.capitalize():<11}: {len(part)} rows | "
        f"{part.index.min().date()} to {part.index.max().date()}"
    )

print("\nHistorical features:", len(HISTORY_FEATURES))
print("Calendar features:", len(CALENDAR_FEATURES))
print("Weather features:", len(WEATHER_FEATURES))
print("Total features:", len(all_features))
print("\nSaved to:", PROCESSED_DIR)
