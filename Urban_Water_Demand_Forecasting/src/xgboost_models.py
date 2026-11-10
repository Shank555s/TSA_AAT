"""XGBoost feature ablation (history -> + calendar -> + weather), validation."""

import pandas as pd

from utils import (
    FEATURE_SETS, LABELS, RESULTS_DIR, load_split, make_xgb, metrics,
    plot_bars, plot_forecasts,
)

OUTPUT = RESULTS_DIR / "xgboost"
OUTPUT.mkdir(parents=True, exist_ok=True)

train = load_split("train")
validation = load_split("validation")

assert train.index.max() < validation.index.min(), (
    "Training and validation dates overlap."
)

results = []
predictions_df = pd.DataFrame({"actual": validation["demand"]})

for name, features in FEATURE_SETS.items():
    print(f"\nTraining {name}...", flush=True)

    model = make_xgb()
    model.fit(train[features], train["demand"])
    predictions = model.predict(validation[features])
    predictions_df[name] = predictions

    scores = metrics(validation["demand"], predictions)
    results.append({"Model": name, "Features": len(features), **scores})

    print(
        f"MAE={scores['MAE']:.4f} | RMSE={scores['RMSE']:.4f} | "
        f"sMAPE={scores['sMAPE (%)']:.2f}%",
        flush=True,
    )

# Save results
comparison = pd.DataFrame(results).sort_values("MAE")
comparison.to_csv(OUTPUT / "validation_comparison.csv", index=False)
predictions_df.to_csv(OUTPUT / "validation_predictions.csv", index_label="Date")

# Plots
plot_forecasts(
    validation["demand"],
    {LABELS[name]: predictions_df[name] for name in FEATURE_SETS},
    "XGBoost Feature Ablation: Validation Forecasts",
    OUTPUT / "validation_forecasts.png",
)

plot_bars(
    [LABELS[m] for m in comparison["Model"]],
    comparison["MAE"],
    "XGBoost Feature Ablation: Validation MAE",
    OUTPUT / "validation_mae_comparison.png",
    "MAE (lower is better)",
)

print("\n" + "=" * 60)
print("XGBOOST FEATURE ABLATION RESULTS")
print("=" * 60)
print(comparison.to_string(index=False))

# Compare with the best SARIMA from sarima_baseline.py
sarima_file = RESULTS_DIR / "sarima" / "sarima_model_selection.csv"
best = comparison.iloc[0]

print("\nBest XGBoost model:", best["Model"])
if sarima_file.exists():
    sarima_mae = pd.read_csv(sarima_file)["MAE"].min()
    change = (sarima_mae - best["MAE"]) / sarima_mae * 100
    print(f"MAE vs best SARIMA ({sarima_mae:.4f}): {change:+.2f}% "
          "(positive = XGBoost better)")
else:
    print("Run sarima_baseline.py first to compare against SARIMA.")
print("\nResults saved to:", OUTPUT)
