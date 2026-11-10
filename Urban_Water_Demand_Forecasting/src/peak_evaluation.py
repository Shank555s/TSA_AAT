"""Validation error on high-demand (peak) days vs. all days."""

import numpy as np
import pandas as pd

from utils import (
    LABELS, RESULTS_DIR, load_split, peak_threshold, plot_bars,
)

OUTPUT = RESULTS_DIR / "peak_evaluation"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_predictions(folder):
    return pd.read_csv(
        RESULTS_DIR / folder / "validation_predictions.csv",
        parse_dates=["Date"],
    ).set_index("Date")


sarima = load_predictions("sarima")
xgb = load_predictions("xgboost")

# Match the exact same validation dates
combined = sarima.join(xgb.drop(columns=["actual"]), how="inner",
                       validate="one_to_one")

if len(combined) != len(sarima) or len(combined) != len(xgb):
    raise ValueError("Validation prediction dates do not match.")

if not np.allclose(sarima.loc[combined.index, "actual"],
                   xgb.loc[combined.index, "actual"]):
    raise ValueError("Actual demand values do not match.")

# Threshold is calculated using training data only
threshold = peak_threshold(load_split("train"))
peak = combined[combined["actual"] >= threshold]

print(f"Training 90th percentile: {threshold:.4f}")
print(f"Validation observations: {len(combined)}")
print(f"High-demand observations: {len(peak)}")

models = {
    "Naive": "naive_prediction",
    "SARIMA": "sarima_prediction",
    **{LABELS[k]: k for k in LABELS},
}

results = []

for name, column in models.items():
    results.append({
        "Model": name,
        "Overall MAE": np.mean(np.abs(combined["actual"] - combined[column])),
        "Peak MAE": (
            np.mean(np.abs(peak["actual"] - peak[column]))
            if len(peak) > 0 else np.nan
        ),
    })

comparison = pd.DataFrame(results)
comparison.to_csv(OUTPUT / "peak_comparison.csv", index=False)

print("\nPEAK-DEMAND EVALUATION")
print(comparison.to_string(index=False))

if len(peak) > 0:
    plot_bars(
        comparison["Model"],
        comparison["Peak MAE"],
        f"Peak-Demand Forecasting Comparison ({len(peak)} high-demand days)",
        OUTPUT / "peak_mae_comparison.png",
        "MAE on high-demand days (lower is better)",
    )

print("\nResults saved to:", OUTPUT)
