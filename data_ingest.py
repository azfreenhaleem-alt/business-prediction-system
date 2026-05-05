
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

FRED_KEY = os.getenv("FRED_API_KEY")
RAW_DIR  = "data/raw"

# Pakistan-relevant FRED series
# Note: FRED has limited direct Pakistan data; we use global proxies
# that directly affect Pakistani businesses + IMF/World Bank available series.
FRED_SERIES = {
    "USD_index":         "DTWEXBGS",   # US Dollar broad index (drives PKR pressure)
    "oil_price":         "DCOILWTICO", # Crude oil price (Pakistan is oil-importer)
    "us_cpi":            "CPIAUCSL",   # US CPI (global inflation signal)
    "us_consumer_conf":  "UMCSENT",    # Consumer sentiment (global demand proxy)
    "fed_rate":          "FEDFUNDS",   # Fed funds rate (capital flow impact on PKR)
    "gold_price":        "pau7012",    # Gold (major import/remittance hedge)
}


def fetch_fred_series(series_id: str, series_name: str, start: str = "2020-01-01") -> pd.DataFrame | None:
    """Fetch a single FRED series and return a dated dataframe."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":           series_id,
        "api_key":             FRED_KEY,
        "file_type":           "json",
        "observation_start":   start,
        "sort_order":          "asc",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        if "observations" not in data:
            print(f"  [!] No observations returned for {series_id}")
            return None

        df = pd.DataFrame(data["observations"])[["date", "value"]]
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df.rename(columns={"value": series_name})
        print(f"  [ok] {series_name}: {len(df)} rows ({df['date'].min().date()} to {df['date'].max().date()})")
        return df

    except requests.exceptions.RequestException as e:
        print(f"  [error] Failed to fetch {series_id}: {e}")
        return None


def fetch_all_fred() -> None:
    """Fetch all FRED series and save individually to data/raw/."""
    print("\n--- Fetching FRED (Pakistan-relevant macro indicators) ---")

    if not FRED_KEY or FRED_KEY == "FRED_API_KEY":
        print("  [!] FRED_API_KEY not set. Generating synthetic macro data instead.")
        _generate_synthetic_macro()
        return

    for series_name, series_id in FRED_SERIES.items():
        df = fetch_fred_series(series_id, series_name)
        if df is not None:
            path = os.path.join(RAW_DIR, f"fred_{series_name}.csv")
            df.to_csv(path, index=False)
            print(f"      Saved → {path}")


def _generate_synthetic_macro() -> None:
    """
    Fallback: generate plausible synthetic macro data if no FRED key.
    Uses realistic ranges for Pakistan-affecting indicators (2020–present).
    """
    print("  Generating synthetic macro data...")
    np.random.seed(99)

    weeks = pd.date_range(start="2020-01-01", end=datetime.today(), freq="W")
    n = len(weeks)

    def trend_noise(start, end, noise_std, seed=0):
        np.random.seed(seed)
        trend = np.linspace(start, end, n)
        noise = np.random.normal(0, noise_std, n)
        return np.clip(trend + noise.cumsum() * 0.05, 0, None)

    synthetic = {
        "USD_index":        trend_noise(120, 106, 1.0, seed=1),
        "oil_price":        trend_noise(50,  80,  3.0, seed=2),
        "us_cpi":           trend_noise(258, 315, 1.5, seed=3),
        "us_consumer_conf": trend_noise(90,  68,  4.0, seed=4),
        "fed_rate":         np.clip(trend_noise(0.1, 5.3, 0.3, seed=5), 0, 6),
        "gold_price":       trend_noise(1580, 2050, 20,  seed=6),
    }

    for name, values in synthetic.items():
        df = pd.DataFrame({"date": weeks, name: np.round(values, 3)})
        path = os.path.join(RAW_DIR, f"fred_{name}.csv")
        df.to_csv(path, index=False)
    print("  [ok] Synthetic macro data saved.")


def generate_internal_data() -> None:
    """
    Generate 3 years of realistic synthetic internal business data.
    Simulates a Pakistani SME with seasonal patterns, growth trend,
    exchange-rate sensitivity, and realistic cost structures.
    """
    print("\n--- Generating internal business data ---")
    np.random.seed(42)

    weeks = pd.date_range(start="2021-01-01", end=datetime.today(), freq="W")
    n = len(weeks)

    # --- Revenue ---
    # Base trend: moderate growth (~15% annual) + weekly seasonality + noise
    base_revenue = 800_000   # PKR weekly (small business scale)
    growth_rate  = 1.15 ** (1/52)  # weekly compounding of 15% annual
    trend        = base_revenue * (growth_rate ** np.arange(n))

    # Seasonal: higher mid-year (Eid/summer) + year-end dip
    week_of_year = np.array([w.week for w in weeks])
    seasonality  = 1.0 + 0.18 * np.sin(2 * np.pi * week_of_year / 52 - 1.2)

    # Shock: COVID dip early 2021, recovery
    shock = np.ones(n)
    shock[:20] = np.linspace(0.6, 1.0, 20)

    noise   = np.random.normal(1.0, 0.06, n)
    revenue = trend * seasonality * shock * noise

    # --- Units sold (correlated with revenue, inverse price sensitivity) ---
    avg_price = 450 + np.random.normal(0, 20, n)  # PKR per unit, slowly drifts up
    units_sold = (revenue / avg_price).astype(int)
    units_sold = np.clip(units_sold, 100, None)

    # --- Cost of goods (COGS) ---
    # ~55% of revenue base + inflation pressure over time
    cogs_ratio = np.linspace(0.54, 0.62, n) + np.random.normal(0, 0.02, n)
    cogs = revenue * cogs_ratio

    # --- Operating expenses (rent, salaries, utilities) ---
    opex = 200_000 + np.linspace(0, 80_000, n) + np.random.normal(0, 15_000, n)

    # --- Inventory on hand (units) ---
    safety_stock = 500
    reorder_point = 300
    inventory = safety_stock + np.random.randint(-reorder_point, reorder_point, n)
    inventory = np.clip(inventory, 50, 1200).astype(int)

    # --- Customer satisfaction score (1–5) ---
    satisfaction = np.clip(
        3.6 + np.random.normal(0, 0.3, n) + np.linspace(0, 0.4, n),
        1.0, 5.0
    )

    df = pd.DataFrame({
        "date":          weeks,
        "revenue_pkr":   np.round(revenue, 0).astype(int),
        "units_sold":    units_sold,
        "avg_price_pkr": np.round(avg_price, 2),
        "cogs_pkr":      np.round(cogs, 0).astype(int),
        "opex_pkr":      np.round(opex, 0).astype(int),
        "gross_profit":  np.round(revenue - cogs, 0).astype(int),
        "inventory_units": inventory,
        "satisfaction":  np.round(satisfaction, 2),
    })

    path = os.path.join(RAW_DIR, "internal_data.csv")
    df.to_csv(path, index=False)
    print(f"  [ok] Internal data: {len(df)} weeks saved → {path}")
    print(f"       Revenue range: PKR {df['revenue_pkr'].min():,} – {df['revenue_pkr'].max():,} / week")
    print(f"       Units range:   {df['units_sold'].min()} – {df['units_sold'].max()} / week")


def generate_reviews() -> None:
    """Generate synthetic customer reviews for sentiment context."""
    print("\n--- Generating customer reviews ---")
    np.random.seed(7)

    positive = [
        "Delivery was faster than expected, very happy with the quality.",
        "Great product, exactly as described. Will order again.",
        "Customer service resolved my issue quickly. Impressed.",
        "Price is fair and product is durable. Recommended.",
        "Packaging was secure and the item arrived in perfect condition.",
        "Very satisfied. The product exceeded my expectations.",
        "Quick response from the team. Quality is consistent.",
    ]
    neutral = [
        "Product is okay but shipping took longer than expected.",
        "Average quality for the price. Nothing special.",
        "Decent product but the website could be improved.",
        "Order arrived correctly but packaging could be better.",
    ]
    negative = [
        "Product quality has declined compared to my last purchase.",
        "Took three weeks to arrive. No updates during that time.",
        "Item was slightly damaged on arrival. Waiting for replacement.",
        "Price has increased but quality feels the same.",
    ]

    reviews = []
    dates = pd.date_range(end=datetime.today(), periods=40, freq="10D")
    for date in dates:
        r = np.random.random()
        if r < 0.65:
            text = np.random.choice(positive)
            rating = np.random.randint(4, 6)
        elif r < 0.85:
            text = np.random.choice(neutral)
            rating = np.random.randint(3, 5)
        else:
            text = np.random.choice(negative)
            rating = np.random.randint(1, 4)
        reviews.append({"date": date.date(), "review": text, "rating": rating})

    df = pd.DataFrame(reviews)
    path = os.path.join(RAW_DIR, "reviews.csv")
    df.to_csv(path, index=False)
    print(f"  [ok] {len(df)} reviews saved → {path}")


def run_ingestion() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    print("=" * 55)
    print("  DATA INGESTION")
    print("=" * 55)
    fetch_all_fred()
    generate_internal_data()
    generate_reviews()
    print("\n[done] All raw data ready in data/raw/\n")


if __name__ == "__main__":
    run_ingestion()
