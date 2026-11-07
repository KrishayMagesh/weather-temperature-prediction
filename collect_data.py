from datetime import date, timedelta
 
import matplotlib
import pandas as pd
import requests
 
matplotlib.use("Agg")  # save charts to files instead of opening a window
import matplotlib.pyplot as plt  # noqa: E402
 
# ---- Settings you can change -------------------------------------------
# Add or swap cities here. Format: "Name": (latitude, longitude)
# For example: "Phoenix": (33.4484, -112.0740)
CITIES = {
    "San Jose": (37.3382, -121.8863),
    "Seattle": (47.6062, -122.3321),
}
DAYS_OF_HISTORY = 1825         # about five years
# -------------------------------------------------------------------------
 
URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARIABLES = ["temperature_2m_max", "temperature_2m_min",
                   "precipitation_sum", "wind_speed_10m_max"]
COLUMN_NAMES = {"time": "date", "temperature_2m_max": "temp_high_f",
                "temperature_2m_min": "temp_low_f",
                "precipitation_sum": "precip_in",
                "wind_speed_10m_max": "wind_max_mph"}
COLORS = ["#e8743b", "#2a6fdb", "#2e9e5b", "#8e44ad"]
 
 
def fetch_weather(latitude, longitude, start, end):
    """Ask the API for daily weather and return it as a DataFrame."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",
    }
    try:
        response = requests.get(URL, params=params, timeout=30)
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise SystemExit(f"Could not get data from the weather API: {exc}")
 
    if response.status_code != 200 or "daily" not in data:
        raise SystemExit(f"The weather API returned an error: "
                         f"{data.get('reason', response.status_code)}")
 
    df = pd.DataFrame(data["daily"]).rename(columns=COLUMN_NAMES)
    df["date"] = pd.to_datetime(df["date"])
    # The newest day or two can still be empty, so drop rows with no temperature
    return df.dropna(subset=["temp_high_f", "temp_low_f"]).reset_index(drop=True)
 
 
def fetch_all_cities(start, end):
    """Fetch every city and stack the results into one DataFrame."""
    frames = []
    for city, (latitude, longitude) in CITIES.items():
        print(f"Fetching {city}...")
        city_df = fetch_weather(latitude, longitude, start, end)
        city_df.insert(1, "city", city)
        frames.append(city_df)
    return pd.concat(frames, ignore_index=True)
 
 
def compare_cities(df):
    """Return a table with one column per city."""
    columns = {}
    for city, group in df.groupby("city", sort=False):
        hottest = group["temp_high_f"].max()
        coldest = group["temp_low_f"].min()
        columns[city] = {
            "Days of data": len(group),
            "Average high (F)": group["temp_high_f"].mean(),
            "Average low (F)": group["temp_low_f"].mean(),
            "Hottest day (F)": hottest,
            "Coldest night (F)": coldest,
            "Rainy days (%)": (group["precip_in"] >= 0.01).mean() * 100,
            "Total rain (in)": group["precip_in"].sum(),
            "Average max wind (mph)": group["wind_max_mph"].mean(),
        }
    return pd.DataFrame(columns).round(1)
 
 
def print_findings(table):
    """Print the comparison table and a few plain-English takeaways."""
    print("\n" + table.to_string())
 
    first, second = list(table.columns)[:2]
    warmer, cooler = (first, second) if (table.loc["Average high (F)", first]
                                         >= table.loc["Average high (F)", second]) \
        else (second, first)
    gap = table.loc["Average high (F)", warmer] - table.loc["Average high (F)", cooler]
    print(f"\n- {warmer} is warmer: its average high is {gap:.1f} F above {cooler}'s.")
 
    rain_first = table.loc["Total rain (in)", first]
    rain_second = table.loc["Total rain (in)", second]
    wetter, drier = (first, second) if rain_first >= rain_second else (second, first)
    low = min(rain_first, rain_second)
    if low > 0:
        ratio = max(rain_first, rain_second) / low
        print(f"- {wetter} is wetter: about {ratio:.1f}x the rainfall of {drier}.")
 
 
def make_charts(df):
    """Save a temperature comparison chart and a monthly rainfall chart."""
    # Chart 1: 30-day average high (solid) and low (dashed) for each city
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (city, group) in enumerate(df.groupby("city", sort=False)):
        color = COLORS[i % len(COLORS)]
        ax.plot(group["date"], group["temp_high_f"].rolling(30, min_periods=10).mean(),
                color=color, linewidth=2.5, label=f"{city} high")
        ax.plot(group["date"], group["temp_low_f"].rolling(30, min_periods=10).mean(),
                color=color, linewidth=2, linestyle="--", label=f"{city} low")
    ax.set_title("Temperature Comparison (30-day averages)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (F)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("weather_temperature.png", dpi=150)
    plt.close(fig)
 
    # Chart 2: total rainfall per month, side-by-side bars for each city
    monthly = (df.groupby([df["date"].dt.to_period("M"), "city"])["precip_in"]
               .sum().unstack("city").fillna(0)[list(CITIES)])
    labels = monthly.index.astype(str)
    width = 0.8 / len(monthly.columns)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, city in enumerate(monthly.columns):
        positions = [j - 0.4 + width * (i + 0.5) for j in range(len(monthly))]
        ax.bar(positions, monthly[city], width, label=city, color=COLORS[i % len(COLORS)])
    ax.set_title("Monthly Rainfall Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rain (inches)")
    ax.set_xticks(range(0, len(labels), 2))
    ax.set_xticklabels(labels[::2], rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("weather_rainfall.png", dpi=150)
    plt.close(fig)
 
 
def main():
    # The archive runs about a week behind, so stop a week before today
    end = date.today() - timedelta(days=7)
    start = end - timedelta(days=DAYS_OF_HISTORY)
 
    df = fetch_all_cities(start, end)
    df.to_csv("weather_data.csv", index=False)
    print(f"Saved {len(df)} rows of data to weather_data.csv")
 
    table = compare_cities(df)
    table.to_csv("city_comparison.csv")
    print_findings(table)
 
    make_charts(df)
    print("\nSaved weather_temperature.png and weather_rainfall.png")
 
 
if __name__ == "__main__":
    main()