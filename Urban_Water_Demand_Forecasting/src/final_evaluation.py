"""Final held-out test: refit on train + validation, forecast one day ahead."""

import numpy as np
import pandas as pd

from utils import (
    FEATURE_SETS, LABELS, RESULTS_DIR, fit_sarima, load_split, make_xgb,
    metrics, peak_threshold, plot_bars, plot_forecasts, rolling_one_step,
)

OUTPUT = RESULTS_DIR / "final_evaluation"
OUTPUT.mkdir(parents=True, exist_ok=True)

# 1. Load datasets
train = load_split("train")
validation = load_split("validation")
test = load_split("test")
train_val = pd.concat([train, validation])

if not (train.index.max() < validation.index.min() < test.index.min()):
    raise ValueError("Dataset dates overlap or are out of order.")

if not train_val.index.is_unique or not test.index.is_unique:
    raise ValueError("Duplicate dates found.")

# Fixed peak threshold from the original training data
threshold = peak_threshold(train)

predictions = pd.DataFrame({"actual": test["demand"]})
results = []


def save_scores(name, predicted):
    scores = metrics(test["demand"], predicted, threshold)
    results.append({"Model": name, **scores})
    predictions[name] = predicted

    print(
        f"{name}: MAE={scores['MAE']:.4f} | RMSE={scores['RMSE']:.4f} | "
        f"sMAPE={scores['sMAPE (%)']:.2f}% | Peak MAE={scores['Peak MAE']:.4f}",
        flush=True,
    )


# 2. Naive baseline
previous_demand = pd.concat([train_val["demand"], test["demand"]]).shift(1)
save_scores("Naive", previous_demand.loc[test.index].to_numpy())

# 3. SARIMA (configuration selected on validation MAE)
print("\nTraining final SARIMA...", flush=True)

y_train_val = train_val["demand"].astype(float).asfreq("D")
y_test = test["demand"].astype(float).asfreq("D")

if y_train_val.isna().any() or y_test.isna().any():
    raise ValueError("Missing dates or target values.")

sarima = fit_sarima(y_train_val, (1, 0, 0), (1, 0, 0, 7))
save_scores("SARIMA", rolling_one_step(sarima, y_test))

# 4. XGBoost feature-ablation models
for key, features in FEATURE_SETS.items():
    name = LABELS[key]
    print(f"\nTraining {name}...", flush=True)

    model = make_xgb()
    model.fit(train_val[features], train_val["demand"])
    save_scores(name, model.predict(test[features]))

# 5. Save results
comparison = pd.DataFrame(results).sort_values("MAE")
comparison.to_csv(OUTPUT / "final_model_comparison.csv", index=False)
predictions.to_csv(OUTPUT / "final_test_predictions.csv", index_label="Date")

# 6. Figures
plot_forecasts(
    test["demand"],
    {name: predictions[name] for name in ["SARIMA", "XGBoost B", "XGBoost C"]},
    "Final Test: One-Day-Ahead Water Demand Forecast",
    OUTPUT / "final_predictions.png",
)

plot_bars(
    comparison["Model"],
    comparison["MAE"],
    "Final Test: Model MAE Comparison",
    OUTPUT / "final_mae_comparison.png",
    "MAE (lower is better)",
)

# The test period has no high-demand days (winter), so Peak MAE is empty.
if (test["demand"] >= threshold).any():
    plot_bars(
        comparison["Model"],
        comparison["Peak MAE"],
        "Final Test: Peak-Demand MAE",
        OUTPUT / "final_peak_mae.png",
        "Peak MAE (lower is better)",
    )

# 7. Print summary
print("\n" + "=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)
print("Test observations:", len(test))
print("Peak threshold:", round(threshold, 4))
print("Peak-demand days:", int((test["demand"] >= threshold).sum()))
print("\n")
print(comparison.to_string(index=False))
print("\nBest overall MAE:", comparison.iloc[0]["Model"])
print("\nResults saved to:", OUTPUT)
