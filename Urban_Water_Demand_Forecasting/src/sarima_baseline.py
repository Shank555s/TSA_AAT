"""Naive and SARIMA baselines, evaluated one day ahead on the validation set."""

import numpy as np
import pandas as pd

from utils import (
    RESULTS_DIR, fit_sarima, load_split, metrics, plot_forecasts,
    rolling_one_step,
)

OUTPUT = RESULTS_DIR / "sarima"
OUTPUT.mkdir(parents=True, exist_ok=True)

# 1. Load chronologically split datasets
y_train = load_split("train")["demand"].astype(float).asfreq("D")
y_val = load_split("validation")["demand"].astype(float).asfreq("D")

if y_train.isna().any() or y_val.isna().any():
    raise ValueError("Missing demand values or dates.")

# 2. Naive one-day-ahead baseline: tomorrow = today
previous_actual = pd.concat([y_train, y_val]).shift(1)
naive_predictions = previous_actual.loc[y_val.index].to_numpy()
naive_scores = metrics(y_val, naive_predictions)

# 3. Try a small number of SARIMA configurations (weekly seasonality = 7)
configurations = [
    ((1, 0, 0), (1, 0, 0, 7)),
    ((1, 1, 0), (1, 0, 0, 7)),
    ((1, 0, 1), (1, 0, 0, 7)),
    ((1, 1, 1), (1, 0, 0, 7)),
]

results = []
best_predictions = None
best_mae = float("inf")
best_config = None

for order, seasonal_order in configurations:
    print(f"\nEvaluating SARIMA{order}x{seasonal_order}...", flush=True)

    try:
        fitted = fit_sarima(y_train, order, seasonal_order)
        predictions = rolling_one_step(fitted, y_val)
        scores = metrics(y_val, predictions)

        results.append({
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            **scores,
        })

        print(
            f"MAE={scores['MAE']:.4f} | RMSE={scores['RMSE']:.4f} | "
            f"sMAPE={scores['sMAPE (%)']:.2f}%",
            flush=True,
        )

        if scores["MAE"] < best_mae:
            best_mae = scores["MAE"]
            best_predictions = predictions
            best_config = (order, seasonal_order)

    except Exception as error:
        print(f"Configuration failed: {error}", flush=True)

if best_predictions is None:
    raise RuntimeError("All SARIMA configurations failed.")

# 4. Save results
pd.DataFrame(results).to_csv(OUTPUT / "sarima_model_selection.csv", index=False)

pd.DataFrame({
    "actual": y_val.to_numpy(),
    "naive_prediction": naive_predictions,
    "sarima_prediction": best_predictions,
}, index=y_val.index).to_csv(OUTPUT / "validation_predictions.csv",
                             index_label="Date")

plot_forecasts(
    y_val,
    {"SARIMA": best_predictions},
    "SARIMA: One-Day-Ahead Validation Forecast",
    OUTPUT / "sarima_validation.png",
)

# 5. Print summary
print("\n" + "=" * 50)
print("BASELINE MODEL RESULTS")
print("=" * 50)

print("\nNaive baseline:")
for name, value in naive_scores.items():
    print(f"  {name}: {value:.4f}")

print("\nBest SARIMA configuration:", best_config)
best_scores = metrics(y_val, best_predictions)
for name, value in best_scores.items():
    print(f"  {name}: {value:.4f}")

improvement = (naive_scores["MAE"] - best_scores["MAE"]) / naive_scores["MAE"] * 100
print(f"\nMAE improvement over naive: {improvement:.2f}%")
print("\nResults saved to:", OUTPUT)
