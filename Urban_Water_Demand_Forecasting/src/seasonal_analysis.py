"""Compare demand across train / validation / test (seasonal coverage)."""

import matplotlib.pyplot as plt
import pandas as pd

from utils import (
    ACCENT, ALERT, MUTED, RESULTS_DIR, UNIT_LABEL, format_dates,
    load_split, peak_threshold, save,
)

OUTPUT = RESULTS_DIR / "seasonal_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

datasets = {
    "Training": load_split("train"),
    "Validation": load_split("validation"),
    "Testing": load_split("test"),
}
threshold = peak_threshold(datasets["Training"])
split_colors = {"Training": ACCENT, "Validation": "#F59E0B", "Testing": "#10B981"}

summary = []

for name, df in datasets.items():
    peak_count = (df["demand"] >= threshold).sum()

    summary.append({
        "Dataset": name,
        "Start": df.index.min().date(),
        "End": df.index.max().date(),
        "Days": len(df),
        "Mean": df["demand"].mean(),
        "Minimum": df["demand"].min(),
        "Maximum": df["demand"].max(),
        "Peak days": int(peak_count),
        "Peak percentage": 100 * peak_count / len(df),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUTPUT / "seasonal_summary.csv", index=False)

print("\nSEASONAL DATASET ANALYSIS")
print("=" * 75)
print(f"Peak threshold: {threshold:.4f}")
print(summary_df.to_string(index=False))

# Demand distribution by split
fig, ax = plt.subplots(figsize=(8, 4.8))
box = ax.boxplot(
    [df["demand"].values for df in datasets.values()],
    tick_labels=list(datasets.keys()),
    patch_artist=True,
    widths=0.5,
    medianprops={"color": "#111827", "linewidth": 1.5},
)
for patch, name in zip(box["boxes"], datasets):
    patch.set(facecolor=split_colors[name], alpha=0.75, edgecolor="#374151")
ax.axhline(threshold, color=ALERT, linestyle="--", linewidth=1,
           label=f"Peak threshold ({threshold:.2f})")
ax.set_ylabel(UNIT_LABEL)
ax.set_title("Demand Distribution Across Splits")
ax.grid(axis="x", visible=False)
ax.legend()
save(OUTPUT / "demand_distribution.png")

# Demand over time, coloured by split
fig, ax = plt.subplots(figsize=(12, 4.8))
for name, df in datasets.items():
    ax.plot(df.index, df["demand"], color=split_colors[name], linewidth=1.2,
            label=name)
ax.axhline(threshold, color=ALERT, linestyle="--", linewidth=1,
           label=f"Peak threshold ({threshold:.2f})")
ax.set_ylabel(UNIT_LABEL)
ax.set_title("Water Demand Across Training, Validation and Test")
format_dates(ax)
ax.legend(ncol=4, loc="upper right")
save(OUTPUT / "seasonal_demand_timeline.png")

# Monthly demand averages across all splits
all_data = pd.concat(datasets.values())
monthly = (
    all_data.groupby(all_data.index.month)["demand"]
    .agg(["mean", "count"])
    .rename_axis("month")
    .reset_index()
)
monthly.to_csv(OUTPUT / "monthly_demand.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4.8))
peak_month = monthly.loc[monthly["mean"].idxmax(), "month"]
colors = [ACCENT if m == peak_month else MUTED for m in monthly["month"]]
ax.bar(monthly["month"], monthly["mean"], color=colors, width=0.7)
ax.set_xticks(monthly["month"])
ax.set_xlabel("Calendar month")
ax.set_ylabel("Average water demand")
ax.set_title("Average Demand by Calendar Month")
ax.grid(axis="x", visible=False)
save(OUTPUT / "monthly_demand.png")

print("\nMonthly averages:")
print(monthly.to_string(index=False))
print("\nResults saved to:", OUTPUT)
