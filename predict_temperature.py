"""
Machine learning project: predict tomorrow's high temperature.

Uses the daily weather data collected in the weather project (weather_data.csv).
For each day, the models see today's weather and the last few days, and try to
predict tomorrow's high temperature. The data is split by TIME (train on the
past, test on the most recent days) so the test is a fair "predict the future" test.

Two simple baselines are included, so you can tell whether machine learning
actually helps:
  - Persistence: "tomorrow will be the same as today"
  - Climatology: "tomorrow will be the average for that city and month"

Install:  pip install -r requirements.txt
Run:      python predict_temperature.py
Output:   model_results.csv, ml_predictions.png, ml_feature_importance.png
"""
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")  # save charts to files instead of opening a window
import matplotlib.pyplot as plt  # noqa: E402

DATA_FILE = "weather_data.csv"
TEST_FRACTION = 0.2            # the most recent 20% of days are the test set
TARGET = "target_high_tomorrow"
COLORS = ["#e8743b", "#2a6fdb", "#2e9e5b", "#8e44ad"]

NUMBER_FEATURES = ["high_today", "high_yesterday", "high_2days_ago", "high_7day_avg",
                   "low_today", "rain_today", "rain_3day_total", "wind_today",
                   "season_sin", "season_cos"]


def load_data():
    """Read the CSV collected by weather_project.py."""
    try:
        df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    except FileNotFoundError:
        raise SystemExit(f"Could not find {DATA_FILE}. Copy it into this folder first.")
    if "city" not in df.columns:
        df["city"] = "City"
    return df


def build_features(df):
    """Turn raw daily weather into one row per day: features today, answer tomorrow."""
    frames = []
    for city, g in df.groupby("city", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        if (g["date"].diff().dt.days.dropna() != 1).any():
            print(f"Warning: {city} has missing days, so some rows may be off.")

        f = pd.DataFrame({"date": g["date"], "city": city})
        f["high_today"] = g["temp_high_f"]
        f["high_yesterday"] = g["temp_high_f"].shift(1)
        f["high_2days_ago"] = g["temp_high_f"].shift(2)
        f["high_7day_avg"] = g["temp_high_f"].rolling(7).mean()
        f["low_today"] = g["temp_low_f"]
        f["rain_today"] = g["precip_in"]
        f["rain_3day_total"] = g["precip_in"].rolling(3).sum()
        f["wind_today"] = g["wind_max_mph"]

        # Time of year (known in advance). Sine/cosine let the model see that
        # December 31 is close to January 1.
        tomorrow = g["date"] + pd.Timedelta(days=1)
        day_of_year = tomorrow.dt.day_of_year
        f["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        f["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
        f["tomorrow_month"] = tomorrow.dt.month       # used by the climatology baseline only

        f[TARGET] = g["temp_high_f"].shift(-1)         # what we want to predict
        frames.append(f)

    data = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    city_columns = pd.get_dummies(data["city"], prefix="city", dtype=int)
    return pd.concat([data, city_columns], axis=1), list(city_columns.columns)


def split_by_time(data):
    """Train on the past, test on the most recent days (never shuffle time series)."""
    dates = sorted(data["date"].unique())
    cutoff = dates[int(len(dates) * (1 - TEST_FRACTION))]
    return data[data["date"] < cutoff], data[data["date"] >= cutoff]


def score(y_true, y_pred):
    return {"MAE (F)": mean_absolute_error(y_true, y_pred),
            "RMSE (F)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred)}


def climatology_predictions(train, test):
    """Predict the training-set average high for that city and month."""
    averages = train.groupby(["city", "tomorrow_month"])[TARGET].mean()
    overall = train[TARGET].mean()
    return np.array([averages.get((c, m), overall)
                     for c, m in zip(test["city"], test["tomorrow_month"])])


def main():
    data, city_columns = build_features(load_data())
    features = NUMBER_FEATURES + city_columns
    train, test = split_by_time(data)
    print(f"Training rows: {len(train)} (up to {train['date'].max():%b %d, %Y})")
    print(f"Testing rows:  {len(test)} ({test['date'].min():%b %d, %Y} onward)")

    predictions = {
        "Baseline: same as today": test["high_today"].to_numpy(),
        "Baseline: monthly average": climatology_predictions(train, test),
    }
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               random_state=42, n_jobs=-1),
    }
    for name, model in models.items():
        model.fit(train[features], train[TARGET])
        predictions[name] = model.predict(test[features])

    # ---- results table -------------------------------------------------
    rows = {}
    for name, predicted in predictions.items():
        row = score(test[TARGET], predicted)
        for city in test["city"].unique():
            mask = (test["city"] == city).to_numpy()
            row[f"MAE {city} (F)"] = mean_absolute_error(test[TARGET][mask], predicted[mask])
        rows[name] = row
    results = pd.DataFrame(rows).T.round(2).sort_values("MAE (F)")
    results.to_csv("model_results.csv")
    print("\nHow far off were the predictions? (lower MAE is better)\n")
    print(results.to_string())

    best_model = min(models, key=lambda n: results.loc[n, "MAE (F)"])
    baseline_mae = results.loc["Baseline: same as today", "MAE (F)"]
    best_mae = results.loc[best_model, "MAE (F)"]
    change = (baseline_mae - best_mae) / baseline_mae * 100
    if best_mae < baseline_mae:
        print(f"\nBest model: {best_model}, off by {best_mae:.2f} F on average, "
              f"{change:.0f}% better than 'same as today'.")
    else:
        print(f"\nBest model: {best_model} ({best_mae:.2f} F), but it did not beat "
              f"'same as today' ({baseline_mae:.2f} F).")

    # ---- chart 1: predictions vs. what really happened ------------------
    cities = list(test["city"].unique())
    fig, axes = plt.subplots(len(cities), 1, figsize=(10, 4 * len(cities)), squeeze=False)
    for ax, city in zip(axes[:, 0], cities):
        mask = (test["city"] == city).to_numpy()
        days = test["date"][mask] + pd.Timedelta(days=1)
        ax.plot(days, test[TARGET][mask], color="black", linewidth=1.8, label="Actual high")
        ax.plot(days, predictions[best_model][mask], color="#e8743b", linewidth=1.8,
                label=f"Predicted ({best_model})")
        ax.set_title(f"{city}: predicted vs. actual daily high", fontweight="bold")
        ax.set_ylabel("Temperature (F)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig("ml_predictions.png", dpi=150)
    plt.close(fig)

    # ---- chart 2: what the Random Forest paid attention to --------------
    importance = pd.Series(models["Random Forest"].feature_importances_,
                           index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance.index, importance.values, color=COLORS[1])
    ax.set_title("Which inputs mattered most? (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig("ml_feature_importance.png", dpi=150)
    plt.close(fig)

    print("\nSaved model_results.csv, ml_predictions.png, ml_feature_importance.png")


if __name__ == "__main__":
    main()
