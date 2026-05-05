

import pandas as pd
import numpy as np
from datetime import datetime



# FORECASTING ENGINE


def _try_prophet(series: pd.Series, dates: pd.Series, periods: int) -> pd.DataFrame | None:
    """Try to use Prophet. Returns None if not installed."""
    try:
        from prophet import Prophet
        prophet_df = pd.DataFrame({"ds": dates, "y": series.values})
        prophet_df = prophet_df.dropna()

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,   # controls how flexible the trend is
            seasonality_prior_scale=10,
        )
        m.fit(prophet_df)
        future   = m.make_future_dataframe(periods=periods, freq="W")
        forecast = m.predict(future)
        result   = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
        result   = result.rename(columns={"ds": "date", "yhat": "forecast",
                                          "yhat_lower": "lower", "yhat_upper": "upper"})
        return result.reset_index(drop=True)

    except ImportError:
        return None


def _manual_forecast(series: pd.Series, dates: pd.Series, periods: int) -> pd.DataFrame:
    """
    Fallback forecasting without Prophet.
    Uses: linear trend + additive annual seasonality (52-week rolling average).
    Good enough for business planning purposes.
    """
    n = len(series)
    x = np.arange(n)
    y = series.values.astype(float)

    # Remove NaNs
    valid = ~np.isnan(y)
    x_v, y_v = x[valid], y[valid]

    # Linear trend via least squares
    coeffs    = np.polyfit(x_v, y_v, deg=1)
    slope, intercept = coeffs

    # Seasonal component: average residual by week-of-year
    trend     = slope * x_v + intercept
    residuals = y_v - trend
    week_nums = pd.to_datetime(dates.values[valid]).isocalendar().week.values
    seasonal  = {}
    for w in range(1, 54):
        mask = week_nums == w
        seasonal[w] = residuals[mask].mean() if mask.sum() > 0 else 0.0

    # Forecast future points
    future_dates = pd.date_range(
        start=dates.iloc[-1] + pd.Timedelta(weeks=1),
        periods=periods,
        freq="W"
    )
    future_weeks   = future_dates.isocalendar().week.values
    future_x       = np.arange(n, n + periods)
    future_trend   = slope * future_x + intercept
    future_seas    = np.array([seasonal.get(w, 0) for w in future_weeks])
    future_vals    = future_trend + future_seas

    # Confidence interval: ±1 std dev of recent residuals
    recent_std = np.std(residuals[-26:]) if len(residuals) >= 26 else np.std(residuals)

    df = pd.DataFrame({
        "date":     future_dates,
        "forecast": np.round(future_vals, 2),
        "lower":    np.round(future_vals - 1.5 * recent_std, 2),
        "upper":    np.round(future_vals + 1.5 * recent_std, 2),
    })
    return df


def forecast_series(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 12) -> pd.DataFrame:
    """
    Forecast a single column 'periods' weeks into the future.
    Tries Prophet first, falls back to manual if not installed.
    """
    series = df[value_col].copy()
    dates  = pd.to_datetime(df[date_col])

    result = _try_prophet(series, dates, periods)
    if result is not None:
        print(f"  [prophet] Forecast for {value_col} complete.")
    else:
        result = _manual_forecast(series, dates, periods)
        print(f"  [manual]  Forecast for {value_col} complete (install prophet for better accuracy).")

    result["series"] = value_col
    return result



# INVENTORY RECOMMENDATION


def recommend_inventory(demand_forecast: pd.DataFrame, current_inventory: int) -> dict:
    """
    Computes recommended inventory level based on demand forecast.

    Logic:
    - Project monthly demand (next 4 weeks average)
    - Compute demand trend (rising/flat/falling)
    - Apply safety factor accordingly
    - Compare against current inventory
    """
    fc_vals = demand_forecast["forecast"].values

    # Trend: compare last 4 forecast weeks vs previous 4
    next4 = fc_vals[:4].mean()
    if len(fc_vals) >= 8:
        prev4 = fc_vals[4:8].mean()
        trend_pct = (next4 - prev4) / prev4 * 100 if prev4 != 0 else 0
    else:
        trend_pct = 0.0

    # Safety factor based on trend
    if trend_pct > 5:
        safety     = 1.20
        trend_label = "rising"
    elif trend_pct < -5:
        safety     = 0.90
        trend_label = "falling"
    else:
        safety     = 1.05
        trend_label = "stable"

    projected_monthly = round(next4 * 4)
    recommended       = round(projected_monthly * safety)
    gap               = recommended - current_inventory
    action            = "reorder" if gap > 0 else "hold" if gap > -100 else "reduce"

    return {
        "projected_monthly_demand_units": projected_monthly,
        "demand_trend":                   trend_label,
        "trend_pct":                      round(trend_pct, 1),
        "safety_factor":                  safety,
        "recommended_inventory_units":    recommended,
        "current_inventory_units":        current_inventory,
        "gap_units":                      gap,
        "action":                         action,
        "12_week_forecast_avg":           round(fc_vals.mean(), 1),
        "12_week_forecast_peak":          round(fc_vals.max(), 1),
    }



# SUMMARY FOR PROMPT BUILDING


def build_forecast_summary(df: pd.DataFrame) -> dict:
    """
    Run all forecasts and return a structured summary dict
    ready to be injected into the LLM prompt.
    """
    print("\n--- Running forecasts ---")

    summaries = {}

    # Revenue forecast
    if "revenue_pkr" in df.columns:
        rev_fc = forecast_series(df, "date", "revenue_pkr", periods=12)
        summaries["revenue"] = {
            "unit": "PKR",
            "forecast_12w": rev_fc[["date","forecast","lower","upper"]].to_dict("records"),
            "avg_forecast": round(rev_fc["forecast"].mean(), 0),
            "peak_forecast": round(rev_fc["forecast"].max(), 0),
        }

    # Demand (units) forecast + inventory recommendation
    if "units_sold" in df.columns:
        dem_fc = forecast_series(df, "date", "units_sold", periods=12)
        current_inv = int(df["inventory_units"].iloc[-1]) if "inventory_units" in df.columns else 500
        inv_rec     = recommend_inventory(dem_fc, current_inv)
        summaries["demand"] = {
            "unit": "units",
            "forecast_12w": dem_fc[["date","forecast","lower","upper"]].to_dict("records"),
            "inventory_recommendation": inv_rec,
        }

    # Cost forecast
    if "cogs_pkr" in df.columns:
        cost_fc = forecast_series(df, "date", "cogs_pkr", periods=12)
        summaries["costs"] = {
            "unit": "PKR",
            "avg_forecast": round(cost_fc["forecast"].mean(), 0),
            "trend_direction": "up" if cost_fc["forecast"].iloc[-1] > cost_fc["forecast"].iloc[0] else "down",
        }

    print(f"  [ok] All forecasts complete.")
    return summaries


def format_forecast_for_prompt(summaries: dict) -> str:
    """Convert forecast dict into a clean string for the LLM prompt."""
    lines = []

    if "revenue" in summaries:
        r = summaries["revenue"]
        lines.append(f"Revenue forecast (next 12 weeks):")
        lines.append(f"  Average: PKR {r['avg_forecast']:,.0f}/week")
        lines.append(f"  Peak:    PKR {r['peak_forecast']:,.0f}/week")

    if "demand" in summaries:
        d   = summaries["demand"]
        inv = d["inventory_recommendation"]
        lines.append(f"\nDemand forecast:")
        lines.append(f"  Projected monthly demand: {inv['projected_monthly_demand_units']:,} units")
        lines.append(f"  Demand trend: {inv['demand_trend']} ({inv['trend_pct']:+.1f}%)")
        lines.append(f"\nInventory recommendation:")
        lines.append(f"  Current stock:     {inv['current_inventory_units']:,} units")
        lines.append(f"  Recommended level: {inv['recommended_inventory_units']:,} units")
        lines.append(f"  Gap:               {inv['gap_units']:+,} units")
        lines.append(f"  Action:            {inv['action'].upper()}")

    if "costs" in summaries:
        c = summaries["costs"]
        lines.append(f"\nCost forecast:")
        lines.append(f"  Avg projected COGS: PKR {c['avg_forecast']:,.0f}/week")
        lines.append(f"  Trend direction:    {c['trend_direction']}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    from pipeline import run_pipeline
    from data_ingest import run_ingestion
    run_ingestion()
    df = run_pipeline()
    summaries = build_forecast_summary(df)
    print("\n" + format_forecast_for_prompt(summaries))
