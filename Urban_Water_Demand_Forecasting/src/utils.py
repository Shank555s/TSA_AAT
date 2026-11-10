"""Shared paths, features, metrics, model builders and plot styling."""

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "daily_features_DMA_A.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# ------------------------------------------------------------- features
LAGS = [1, 2, 3, 7, 14]
WINDOWS = [3, 7, 14]

HISTORY_FEATURES = [f"lag_{k}" for k in LAGS] + [
    name for w in WINDOWS for name in (f"rolling_mean_{w}", f"rolling_std_{w}")
]
CALENDAR_FEATURES = ["day_of_week", "month", "is_weekend", "month_sin", "month_cos"]
WEATHER_FEATURES = ["temp-1", "rain-1", "humidity-1", "wind-1"]

# Feature ablation: history only -> + calendar -> + weather
FEATURE_SETS = {
    "A_history": HISTORY_FEATURES,
    "B_history_calendar": HISTORY_FEATURES + CALENDAR_FEATURES,
    "C_history_calendar_weather": (
        HISTORY_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES
    ),
}
LABELS = {
    "A_history": "XGBoost A",
    "B_history_calendar": "XGBoost B",
    "C_history_calendar_weather": "XGBoost C",
}

PEAK_QUANTILE = 0.90  # "high-demand" = top 10% of training days
UNIT_LABEL = "Water demand (dataset units)"


# ------------------------------------------------------------- data/metrics
def load_split(name):
    """Load train / validation / test as a Date-indexed DataFrame."""
    return pd.read_csv(
        PROCESSED_DIR / f"{name}.csv", parse_dates=["Date"], index_col="Date"
    ).sort_index()


def peak_threshold(train):
    """High-demand threshold, computed from training data only."""
    return train["demand"].quantile(PEAK_QUANTILE)


def metrics(actual, predicted, threshold=None):
    """MAE, RMSE, sMAPE and (optionally) MAE on high-demand days."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = actual - predicted
    denominator = np.abs(actual) + np.abs(predicted)
    smape = 100 * np.mean(
        np.divide(
            2 * np.abs(error),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0,
        )
    )

    scores = {
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error**2)),
        "sMAPE (%)": smape,
    }

    if threshold is not None:
        mask = actual >= threshold
        scores["Peak MAE"] = (
            np.mean(np.abs(error[mask])) if mask.any() else np.nan
        )
    return scores


# ------------------------------------------------------------------ models
def make_xgb():
    """The XGBoost configuration used in every experiment."""
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )


def fit_sarima(y, order, seasonal_order):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=150)


def rolling_one_step(fitted, y):
    """One-day-ahead forecasts; each true value is appended after use
    (no refitting), matching how the XGBoost models see lagged demand."""
    predictions = []
    model = fitted

    for date, actual in y.items():
        predictions.append(float(model.forecast(steps=1).iloc[0]))
        observation = pd.Series(
            [actual], index=pd.DatetimeIndex([date], freq="D"), name="demand"
        )
        model = model.append(observation, refit=False)

    return np.array(predictions)


# ---------------------------------------------------------------- plotting
PALETTE = {
    "Actual": "#111827",
    "Naive": "#9CA3AF",
    "SARIMA": "#F59E0B",
    "XGBoost A": "#93C5FD",
    "XGBoost B": "#3B82F6",
    "XGBoost C": "#1D4ED8",
}
ACCENT = "#2563EB"
MUTED = "#CBD5E1"
ALERT = "#DC2626"


def apply_style():
    plt.rcParams.update(
        {
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10.5,
            "axes.labelcolor": "#374151",
            "axes.edgecolor": "#9CA3AF",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "legend.frameon": False,
        }
    )


apply_style()


def save(path):
    """Save the current figure and close it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def format_dates(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(axis="x", visible=False)


def plot_forecasts(actual, predictions, title, path, threshold=None):
    """Actual demand vs. one or more forecasts (dict: label -> values)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    if threshold is not None:
        ax.axhline(
            threshold, color=ALERT, linestyle="--", linewidth=1,
            label=f"Peak threshold ({threshold:.2f})",
        )

    ax.plot(actual.index, actual, color=PALETTE["Actual"], linewidth=2.2,
            label="Actual")
    for label, values in predictions.items():
        ax.plot(actual.index, values, color=PALETTE.get(label), linewidth=1.4,
                alpha=0.9, label=label)

    ax.set_title(title)
    ax.set_ylabel(UNIT_LABEL)
    format_dates(ax)
    ax.legend(ncol=len(predictions) + 1, loc="upper right")
    save(path)


def plot_bars(labels, values, title, path, ylabel):
    """Bar chart with the best (lowest) bar highlighted and values labelled."""
    values = np.asarray(values, dtype=float)
    best = int(np.nanargmin(values))
    colors = [ACCENT if i == best else MUTED for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10, color="#111827")
    ax.set_ylim(0, np.nanmax(values) * 1.15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    save(path)
