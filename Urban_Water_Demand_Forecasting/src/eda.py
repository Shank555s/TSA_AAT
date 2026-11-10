"""Exploratory data analysis: summary statistics and six overview figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import ACCENT, DATA_FILE, RESULTS_DIR, UNIT_LABEL, format_dates, save

OUTPUT = RESULTS_DIR / "eda"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Load the dataset
df = pd.read_csv(DATA_FILE, sep=";")
df.columns = df.columns.str.strip()

if "Date" not in df.columns or "v-0" not in df.columns:
    raise ValueError(f"Expected Date and v-0 columns, found: {df.columns.tolist()}")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["v-0"] = pd.to_numeric(df["v-0"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Dataset summary
print("Dataset shape:", df.shape)
print("Date range:", df["Date"].min(), "to", df["Date"].max())
print("\nMissing values:\n", df.isna().sum())
print("\nSummary statistics:\n", df["v-0"].describe())

with open(OUTPUT / "eda_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Date range: {df['Date'].min()} to {df['Date'].max()}\n\n")
    f.write("Missing values:\n")
    f.write(df.isna().sum().to_string())
    f.write("\n\nDemand statistics:\n")
    f.write(df["v-0"].describe().to_string())

# Figure 1: Daily water demand
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(df["Date"], df["v-0"], color=ACCENT, linewidth=1.1)
ax.set_title("Daily Urban Water Demand")
ax.set_ylabel(UNIT_LABEL)
format_dates(ax)
save(OUTPUT / "01_demand_trend.png")

# Figure 2: Demand distribution
fig, ax = plt.subplots(figsize=(9, 4.8))
sns.histplot(df["v-0"], bins=30, kde=True, color=ACCENT, edgecolor="white", ax=ax)
ax.set_title("Water Demand Distribution")
ax.set_xlabel(UNIT_LABEL)
ax.grid(axis="x", visible=False)
save(OUTPUT / "02_demand_distribution.png")

# Figure 3: Weekday vs weekend
df["DayType"] = np.where(df["Date"].dt.day_of_week >= 5, "Weekend", "Weekday")

fig, ax = plt.subplots(figsize=(7, 4.8))
sns.boxplot(data=df, x="DayType", y="v-0", order=["Weekday", "Weekend"],
            color=ACCENT, width=0.5, linewidth=1.2, ax=ax)
ax.set_title("Weekday vs Weekend Water Demand")
ax.set_xlabel("")
ax.set_ylabel(UNIT_LABEL)
ax.grid(axis="x", visible=False)
save(OUTPUT / "03_weekday_weekend.png")

# Figure 4: Monthly demand patterns
df["Month"] = df["Date"].dt.month

fig, ax = plt.subplots(figsize=(11, 4.8))
sns.boxplot(data=df, x="Month", y="v-0", color=ACCENT, linewidth=1.1, ax=ax)
ax.set_title("Monthly Water Demand Patterns")
ax.set_ylabel(UNIT_LABEL)
ax.grid(axis="x", visible=False)
save(OUTPUT / "04_monthly_patterns.png")

# Figure 5: Autocorrelation (approximate 95% band under white noise)
demand = df["v-0"].dropna().to_numpy()
max_lag = min(30, len(demand) - 1)
centred = demand - demand.mean()
acf = np.array([
    np.sum(centred[: len(centred) - k] * centred[k:]) / np.sum(centred**2)
    for k in range(max_lag + 1)
])
band = 1.96 / np.sqrt(len(demand))

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.axhspan(-band, band, color=ACCENT, alpha=0.12, linewidth=0)
ax.vlines(range(max_lag + 1), 0, acf, color=ACCENT, linewidth=1.6)
ax.plot(range(max_lag + 1), acf, "o", color=ACCENT, markersize=4)
ax.axhline(0, color="#9CA3AF", linewidth=0.8)
ax.set_title("Water Demand Autocorrelation")
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Autocorrelation")
ax.grid(axis="x", visible=False)
save(OUTPUT / "05_autocorrelation.png")

# Figure 6: Correlation heatmap
columns = [c for c in ["v-0", "v-1", "temp-1", "rain-1", "humidity-1", "wind-1"]
           if c in df.columns]

if len(columns) >= 2:
    corr = df[columns].apply(pd.to_numeric, errors="coerce").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=1, linecolor="white",
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation")
    ax.grid(False)
    save(OUTPUT / "06_correlation.png")

print("\nEDA completed successfully!")
print("Results saved to:", OUTPUT)
