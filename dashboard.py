import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="Business Prediction System",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Business Prediction System")

# Create demo data function
def create_demo_data():
    """Create realistic demo data for cloud deployment"""
    dates = pd.date_range(start='2024-01-01', periods=52, freq='W')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'date': dates,
        'revenue_pkr': 800000 + np.arange(52) * 3000 + np.random.normal(0, 50000, 52),
        'units_sold': 1800 + np.arange(52) * 8 + np.random.normal(0, 100, 52),
        'gross_margin_pct': 35 + np.random.normal(0, 1.5, 52),
        'inventory_units': 1500 - np.arange(52) * 5 + np.random.normal(0, 80, 52),
        'satisfaction': 4.2 + np.random.normal(0, 0.15, 52),
        'USD_index': 105 + np.random.normal(0, 1.5, 52),
        'oil_price': 75 + np.random.normal(0, 3, 52),
    })
    return df

def create_demo_analysis():
    """Create demo analysis text"""
    return """## Executive Summary
Your business is showing steady growth with revenue up 12% over the past 8 weeks. 
Inventory levels are healthy at 4.2 weeks of coverage. Customer satisfaction remains strong at 4.2/5.

## Key Risks (top 3)
- **Exchange rate volatility**: USD Index at 105 signals potential PKR pressure of 5-8%
- **Oil price sensitivity**: Current $75/barrel impacts logistics costs by 15-20%
- **Margin compression**: Gross margins down 2% due to input cost inflation

## Key Opportunities (top 2)
- **Bulk purchasing**: Lock in supplier prices before potential PKR depreciation
- **Inventory optimization**: Current turnover suggests room for 15% reduction in safety stock

## Inventory Decision
Maintain current inventory levels but increase reorder frequency from weekly to bi-weekly to reduce working capital by 20%.

## Revenue Outlook
Expect 5-8% revenue growth over next 12 weeks driven by seasonal demand. Monitor USD/PKR for import cost impacts.

## Forecast Calibration
Previous forecast accuracy at 92%. Adjust baseline forecasts by +5% for upcoming seasonal peak."""

# Check if we have real data
data_path = "data/processed/merged.csv"
log_path = "data/prediction_log.json"

if os.path.exists(data_path):
    df = pd.read_csv(data_path, parse_dates=["date"])
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
    else:
        log = []
    st.success("✅ Using your real business data")
else:
    df = create_demo_data()
    log = [{
        "run_id": 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provider": "demo-mode",
        "gpt_analysis": create_demo_analysis(),
        "inventory_recommendation": {
            "current_inventory_units": 1450,
            "recommended_inventory_units": 1300,
            "gap_units": -150,
            "action": "reduce"
        }
    }]
    st.info("💡 **Demo Mode** - Showing sample data. Your real data will appear when you run the pipeline locally.")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Sidebar
with st.sidebar:
    st.header("📊 Dashboard Info")
    st.write("**Business Prediction System**")
    st.write("Powered by Prophet + Gemini AI")
    st.divider()
    st.caption("Run locally with:")
    st.code("streamlit run dashboard.py")

# KPI Cards
latest = df.tail(1).iloc[0]
prev_4 = df.tail(5).head(4).mean()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    rev = latest['revenue_pkr']
    rev_change = ((rev - prev_4['revenue_pkr']) / prev_4['revenue_pkr']) * 100
    st.metric("Weekly Revenue", f"PKR {rev:,.0f}", f"{rev_change:+.1f}%")

with col2:
    units = latest['units_sold']
    units_change = ((units - prev_4['units_sold']) / prev_4['units_sold']) * 100
    st.metric("Units Sold", f"{int(units):,}", f"{units_change:+.1f}%")

with col3:
    st.metric("Gross Margin", f"{latest['gross_margin_pct']:.1f}%")

with col4:
    st.metric("Inventory", f"{int(latest['inventory_units']):,} units")

with col5:
    st.metric("Satisfaction", f"{latest['satisfaction']:.2f}/5")

st.divider()

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'].tail(26),
        y=df['revenue_pkr'].tail(26),
        mode='lines+markers',
        name='Revenue',
        line=dict(color='#2c3e50', width=2)
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue (PKR)",
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Units Sold Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'].tail(26),
        y=df['units_sold'].tail(26),
        mode='lines+markers',
        name='Units Sold',
        line=dict(color='#27ae60', width=2)
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Units Sold",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Inventory Chart
st.subheader("Inventory Levels")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['date'].tail(26),
    y=df['inventory_units'].tail(26),
    mode='lines+markers',
    name='Inventory',
    line=dict(color='#e74c3c', width=2),
    fill='tozeroy'
))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Units",
    height=350
)
st.plotly_chart(fig, use_container_width=True)

# AI Analysis
st.divider()
st.subheader("🤖 AI Business Analysis")

if log:
    latest_run = log[-1]
    analysis = latest_run.get('gpt_analysis', 'No analysis available')
    
    # Parse markdown sections
    lines = analysis.split('\n')
    for line in lines:
        if line.startswith('##'):
            st.subheader(line.replace('#', '').strip())
        elif line.strip():
            st.write(line)
    
    # Show inventory recommendation
    inv_rec = latest_run.get('inventory_recommendation', {})
    if inv_rec:
        st.divider()
        st.subheader("📦 Inventory Recommendation")
        a, b, c, d = st.columns(4)
        a.metric("Current", f"{inv_rec.get('current_inventory_units', 0):,}")
        b.metric("Recommended", f"{inv_rec.get('recommended_inventory_units', 0):,}")
        c.metric("Gap", f"{inv_rec.get('gap_units', 0):+,}")
        d.metric("Action", inv_rec.get('action', '—').upper())

st.divider()
st.caption("Built with Prophet · Gemini AI · Streamlit · FRED API")
