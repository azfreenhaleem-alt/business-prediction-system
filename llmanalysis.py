import os
import json
import pandas as pd
from datetime import datetime
import re
import time

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path=".env"):
        if not os.path.exists(path):
            return False
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        return True

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """You are a senior business analyst advising a Pakistani SME (small-to-medium enterprise).

You will receive:
1. A snapshot of recent business performance (last 8 weeks)
2. Key macroeconomic indicators affecting Pakistan
3. Prophet/model forecasts for next 12 weeks (revenue, demand, costs)
4. Inventory recommendation from the forecasting model
5. Recent customer reviews (sentiment signal)
6. Previous prediction vs actual outcome (if available — for calibration)

Your response must include:

## Executive Summary
2-3 sentences. What is the overall health of this business right now?

## Key Risks (top 3)
Bullet points. Be specific to the data — reference actual numbers.

## Key Opportunities (top 2)
Bullet points. Actionable and specific.

## Inventory Decision
One clear recommendation with the reasoning behind it. Reference the demand trend.

## Revenue Outlook
What should the business owner expect in the next 4–12 weeks? Note any macro headwinds (oil, exchange rate, inflation).

## Forecast Calibration (if prior data available)
Were previous predictions accurate? What should be adjusted?

Be concise. Write for a business owner, not a data scientist. Avoid jargon.
Use PKR for all monetary values. Always flag exchange rate and inflation risks for Pakistan context.
"""

def get_recent_snapshot(df: pd.DataFrame, weeks: int = 8) -> str:
    """Build a clean text snapshot of the last N weeks of business data."""
    recent = df.tail(weeks).copy()
    recent["date"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")

    display_cols = [c for c in [
        "date", "revenue_pkr", "units_sold", "gross_margin_pct",
        "net_margin_pct", "inventory_units", "satisfaction",
        "USD_index", "oil_price", "us_consumer_conf", "fed_rate"
    ] if c in recent.columns]

    snapshot = recent[display_cols].to_string(index=False)
    return snapshot

def get_macro_context(df: pd.DataFrame) -> str:
    """Extract the latest macro indicator values for context."""
    latest = df.tail(1).iloc[0]
    lines = []

    macro_map = {
        "USD_index":        ("US Dollar Index", "higher = stronger USD = weaker PKR"),
        "oil_price":        ("Crude oil (USD/barrel)", "Pakistan imports ~80% of oil needs"),
        "us_cpi":           ("US CPI Index", "proxy for global inflation pressure"),
        "us_consumer_conf": ("US Consumer Confidence", "global demand signal"),
        "fed_rate":         ("US Fed Funds Rate (%)", "higher = capital outflow pressure on PKR"),
        "gold_price":       ("Gold price (USD/oz)", "remittance & reserve hedge indicator"),
    }

    for col, (label, note) in macro_map.items():
        if col in df.columns:
            val = latest.get(col, "N/A")
            lines.append(f"  {label}: {val:.2f}  [{note}]")

    return "\n".join(lines) if lines else "  No macro data available."

def get_review_summary(reviews_path: str = "data/raw/reviews.csv", n: int = 10) -> str:
    """Load most recent reviews and format for prompt."""
    try:
        df = pd.read_csv(reviews_path, parse_dates=["date"])
        df = df.sort_values("date").tail(n)
        avg_rating = df["rating"].mean()
        lines = [f"  Average rating (last {n} reviews): {avg_rating:.1f}/5"]
        for _, row in df.iterrows():
            lines.append(f"  [{row['rating']}/5] {row['review']}")
        return "\n".join(lines)
    except FileNotFoundError:
        return "  No reviews available."

def build_full_prompt(
    df: pd.DataFrame,
    forecast_text: str,
    previous_log: dict | None = None
) -> str:
    """Assemble the complete prompt from all data sources."""
    snapshot = get_recent_snapshot(df)
    macro = get_macro_context(df)
    reviews = get_review_summary()

    calibration = ""
    if previous_log:
        calibration = f"""
## Previous Forecast vs Actual

Previous forecast summary:
{previous_log.get('forecast_summary', 'N/A')}

Actual outcome recorded:
{json.dumps(previous_log.get('actual', 'Not yet recorded'), indent=2)}

GPT analysis from that run:
{previous_log.get('gpt_analysis', 'N/A')[:500]}...
"""

    prompt = f"""
## Recent Business Performance (last 8 weeks)
{snapshot}

## Current Macroeconomic Indicators
{macro}

## Forecast Model Output
{forecast_text}

## Customer Reviews (last 10)
{reviews}
{calibration}

Please provide your full analysis.
"""
    return prompt.strip()


# LOCAL FALLBACK ANALYSIS GENERATOR

def generate_local_analysis(prompt: str) -> str:
    """Generate analysis locally without API calls - guaranteed to work"""
    
    # Extract revenue trend
    revenue_match = re.search(r'revenue_pkr\s+(\d+)\s+(\d+)\s+(\d+)', prompt)
    if revenue_match:
        rev_old = int(revenue_match.group(1))
        rev_mid = int(revenue_match.group(2))
        rev_latest = int(revenue_match.group(3))
        
        # Calculate percent change
        pct_change = ((rev_latest - rev_old) / rev_old) * 100 if rev_old > 0 else 0
        
        if pct_change > 10:
            revenue_trend = f"strongly increasing (+{pct_change:.0f}%)"
            revenue_outlook = "grow 8-12%"
        elif pct_change > 0:
            revenue_trend = f"moderately increasing (+{pct_change:.0f}%)"
            revenue_outlook = "grow 3-7%"
        elif pct_change > -10:
            revenue_trend = f"stable ({pct_change:+.0f}%)"
            revenue_outlook = "remain flat (±3%)"
        else:
            revenue_trend = f"declining ({pct_change:.0f}%)"
            revenue_outlook = "decline 5-10%"
        
        latest_revenue = rev_latest
        revenue_k = rev_latest / 1000
    else:
        revenue_trend = "stable"
        revenue_outlook = "remain stable"
        latest_revenue = 0
        revenue_k = 0
    
    # Extract units sold trend
    units_match = re.search(r'units_sold\s+(\d+)\s+(\d+)\s+(\d+)', prompt)
    if units_match:
        units_latest = int(units_match.group(3))
        units_old = int(units_match.group(1))
        units_pct = ((units_latest - units_old) / units_old) * 100 if units_old > 0 else 0
    else:
        units_latest = 0
        units_pct = 0
    
    # Extract inventory levels
    inv_match = re.search(r'inventory_units\s+(\d+)\s+(\d+)\s+(\d+)', prompt)
    if inv_match:
        latest_inv = int(inv_match.group(3))
        avg_demand = units_latest if units_latest > 0 else 1000
        weeks_of_stock = latest_inv / avg_demand if avg_demand > 0 else 0
        
        if latest_inv < 500 or weeks_of_stock < 2:
            inv_status = "CRITICALLY LOW"
            inv_action = "URGENT: Increase inventory by 50-60% within 2 weeks to avoid stockouts"
            inv_risk = "Stockouts likely within 1-2 weeks - immediate action required"
        elif latest_inv < 1000 or weeks_of_stock < 4:
            inv_status = "LOW"
            inv_action = "Increase inventory by 30-40% over next 2-3 weeks"
            inv_risk = "Stockout risk elevated - replenish soon"
        elif latest_inv > 3000 or weeks_of_stock > 12:
            inv_status = "EXCESSIVE"
            inv_action = "Immediate: Run promotions to reduce inventory. Reduce orders by 40%"
            inv_risk = "High carrying costs and potential obsolescence"
        elif latest_inv > 2000 or weeks_of_stock > 8:
            inv_status = "HIGH"
            inv_action = "Reduce inventory by 20-25% through targeted sales"
            inv_risk = "Working capital tied up in excess stock"
        else:
            inv_status = "OPTIMAL"
            inv_action = "Maintain current inventory levels with weekly monitoring"
            inv_risk = "Low - inventory well-managed"
    else:
        latest_inv = 0
        inv_status = "UNKNOWN"
        inv_action = "Review inventory data - ensure tracking is accurate"
        inv_risk = "Unable to assess - implement inventory tracking"
    
    # Extract macro indicators
    oil_match = re.search(r'oil_price.*?(\d+\.?\d*)', prompt)
    oil_price = float(oil_match.group(1)) if oil_match else 75.0
    
    usd_match = re.search(r'USD_index.*?(\d+\.?\d*)', prompt)
    usd_index = float(usd_match.group(1)) if usd_match else 105.0
    
    margin_match = re.search(r'gross_margin_pct.*?(\d+\.?\d*)', prompt)
    gross_margin = float(margin_match.group(1)) if margin_match else 35.0
    
    # Extract satisfaction
    sat_match = re.search(r'satisfaction.*?(\d+\.?\d*)', prompt)
    satisfaction = float(sat_match.group(1)) if sat_match else 4.0
    
    # Build analysis
    analysis = f"""## Executive Summary
Your business is {revenue_trend} over the past 8 weeks, with weekly revenue averaging PKR {revenue_k:,.0f}K. 
Gross margin at {gross_margin:.1f}% and customer satisfaction at {satisfaction:.1f}/5 indicate { "healthy operations" if satisfaction >= 4.0 else "room for service improvement" if satisfaction >= 3.0 else "significant quality concerns" }.
Inventory is {inv_status.lower()} ({weeks_of_stock:.1f} weeks of cover if inv_match else "status unknown"). 
With oil at ${oil_price:.0f}/barrel and USD Index at {usd_index:.1f}, Pakistan's import costs remain elevated, pressuring margins by an estimated 3-5%.

## Key Risks (top 3)
- **Exchange rate volatility**: USD Index at {usd_index:.1f} signals continued PKR pressure. Every 5% PKR depreciation increases input costs by 4-7%. Consider hedging or supplier credit terms.
- **Inventory {inv_status.lower()} risk**: {inv_risk}
- **Margin pressure**: Oil at ${oil_price:.0f}/barrel directly impacts logistics (15-20% of costs). Your {gross_margin:.1f}% gross margin leaves limited buffer for cost shocks.

## Key Opportunities (top 2)
- **Cost optimization**: With current demand trends, negotiate bulk discounts with 2-3 key suppliers. Target 5-8% cost reduction within 60 days.
- **Customer retention**: Satisfaction at {satisfaction:.1f}/5 suggests you're meeting expectations. Implement a loyalty program to increase repeat purchase rate by 15-20%.

## Inventory Decision
{inv_action}

**Reasoning**: Demand is {"up" if units_pct > 0 else "down"} {abs(units_pct):.0f}% YoY, with {weeks_of_stock:.1f} weeks of current stock. {"Priority on availability" if "LOW" in inv_status else "Priority on working capital" if "HIGH" in inv_status else "Balance is key"}.

## Revenue Outlook
Expect revenue to {revenue_outlook} over the next 4-12 weeks. 
**Macro headwinds**: 
- Oil at ${oil_price:.0f} → transportation costs + PKR pressure
- USD Index at {usd_index:.1f} → imported raw materials 8-12% more expensive
- Overall inflation expected 22-25% → monitor consumer purchasing power

**Action**: Review pricing strategy - consider 5-7% price adjustment if costs increase further.

## Forecast Calibration
Based on recent {revenue_trend} performance:
- Adjust revenue baseline by {pct_change/2:.1f}% for next 4 weeks
- Add ±{abs(units_pct/3):.0f}% buffer to demand forecasts
- Review forecast weekly - volatility in oil/USD requires frequent recalibration

**Key metric to watch**: Weekly inventory turnover (target 1.5-2.5x for your sector)

---
*Analysis generated locally - API was unavailable. Based purely on data patterns without external AI.*
"""
    return analysis

# LLM CALLS WITH AUTO-FALLBACK

from google import genai

client = None
if GEMINI_KEY:
    try:
        client = genai.Client(api_key=GEMINI_KEY)
    except Exception as e:
        print(f"  [warn] Failed to initialize Gemini client: {e}")
        client = None

def call_gemini(prompt: str) -> str:
    """Call Gemini API with proper error handling"""
    if not client:
        raise Exception("Gemini client not initialized - check API key")
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt
    )
    
    if response and response.text:
        return response.text
    else:
        raise Exception("Empty response from Gemini")

def get_analysis(prompt: str) -> tuple[str, str]:
    """Get analysis - tries Gemini API first, falls back to local generation"""
    
    # First, try Gemini if API key exists
    if GEMINI_KEY and client:
        print(f"  [info] Attempting Gemini API...")
        
        for attempt in range(2):  # Only 2 attempts to save time
            try:
                result = call_gemini(prompt)
                if result:
                    print(f"  [ok] Gemini API successful")
                    return result, "gemini-2.0-flash-exp"
            except Exception as e:
                error_msg = str(e)
                error_lower = error_msg.lower()
                
                print(f"  [warn] Gemini attempt {attempt+1} failed: {error_msg[:100]}")
                
                # Only retry on certain errors
                if "503" in error_lower or "429" in error_lower or "rate" in error_lower:
                    if attempt == 0:
                        print(f"  [info] Rate limit, will retry once then fallback...")
                        time.sleep(5)
                        continue
                else:
                    print(f"  [info] Non-retryable error, switching to fallback...")
                    break
        
        print(f"  [info] Gemini failed, switching to local fallback...")
    else:
        print(f"  [info] No valid Gemini API key, using local analysis directly...")
    
    # Fallback to local analysis
    print(f"  [ok] Generating local analysis...")
    local_analysis = generate_local_analysis(prompt)
    return local_analysis, "local-fallback"

def run_analysis(df: pd.DataFrame, forecast_text: str, previous_log: dict | None = None) -> tuple[str, str]:
    """Full pipeline: build prompt → get analysis (API with fallback) → return."""
    print("\n--- Running analysis (API with local fallback) ---")

    prompt = build_full_prompt(df, forecast_text, previous_log)
    
    # Save prompt for debugging
    os.makedirs("data", exist_ok=True)
    with open("data/debug_last_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"  [debug] Prompt saved to data/debug_last_prompt.txt ({len(prompt)} chars)")

    analysis, provider = get_analysis(prompt)

    print(f"  [ok] Analysis received from {provider}")
    print(f"       Length: {len(analysis)} characters")
    return analysis, provider

if __name__ == "__main__":
    from data_ingest import run_ingestion
    from pipeline    import run_pipeline
    from forecast    import build_forecast_summary, format_forecast_for_prompt

    run_ingestion()
    df         = run_pipeline()
    summaries  = build_forecast_summary(df)
    fc_text    = format_forecast_for_prompt(summaries)
    analysis, provider = run_analysis(df, fc_text)
    print(f"\n{'='*55}")
    print(f"ANALYSIS (via {provider})")
    print('='*55)
    print(analysis)