import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="PakBiz",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to remove Streamlit's default styling
st.markdown("""
<style>
    /* Remove default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 1rem;
        color: white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        padding: 1rem;
        border-radius: 1rem;
        color: white;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1rem;
        border-radius: 1rem;
        color: white;
    }
    
    /* Custom recommendation banner */
    .rec-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        text-align: center;
        color: white;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Custom divider */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title with custom styling
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h1 style="font-size: 2.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        PakBiz
    </h1>
    <p style="color: #666;">Probabilistic forecasting for SME decision support</p>
</div>
""", unsafe_allow_html=True)

# Demo data functions
def create_demo_data():
    """Create realistic demo data"""
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
    """Launching first run summary"""
    return """## Executive Summary
Your business is showing steady growth with revenue up 12% over the past 8 weeks. 
Inventory levels are healthy at 4.2 weeks of coverage.

## Key Risks (top 3)
- **Exchange rate volatility**: USD Index at 105 signals potential PKR pressure
- **Oil price sensitivity**: Current $75/barrel impacts logistics costs
- **Margin compression**: Gross margins down 2% due to input cost inflation

## Key Opportunities (top 2)
- **Bulk purchasing**: Lock in supplier prices before potential PKR depreciation
- **Inventory optimization**: Current turnover suggests room for 15% reduction

## Inventory Decision
Maintain current inventory levels but increase reorder frequency to bi-weekly.

## Revenue Outlook
Expect 5-8% revenue growth over next 12 weeks. Monitor USD/PKR for import impacts."""

# Check for real data
data_path = "data/processed/merged.csv"
log_path = "data/prediction_log.json"
real_data_exists = os.path.exists(data_path)

if real_data_exists:
    df = pd.read_csv(data_path, parse_dates=["date"])
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
    else:
        log = []
else:
    df = create_demo_data()
    log = [{
        "run_id": 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provider": "demo",
        "gpt_analysis": create_demo_analysis(),
        "inventory_recommendation": {
            "current_inventory_units": 1450,
            "recommended_inventory_units": 1300,
            "gap_units": -150,
            "action": "REDUCE"
        }
    }]

# Sidebar - Custom styling
with st.sidebar:
    st.markdown("###  Controls")
    
    # Data source indicator
    if real_data_exists:
        st.success(" **Live Mode**\nYour real data")
    else:
        st.info(" **Demo Mode**\nShowing first run data")
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("###  Quick Stats")
    latest = df.tail(1).iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Revenue", f"PKR {latest['revenue_pkr']/1000:.0f}K")
    with col2:
        st.metric("Units", f"{int(latest['units_sold']):,}")
    
    st.markdown("---")
    
    # Run button
    run_pipeline = st.button(" Run Full Pipeline", use_container_width=True, type="primary")
    
    st.markdown("---")
    st.caption("Built with Prophet · Gemini AI")
    st.caption(f"v1.0 | {datetime.now().strftime('%b %Y')}")

# Main content
if run_pipeline:
    with st.spinner("Running analysis..."):
        import subprocess, sys
        result = subprocess.run([sys.executable, "main.py", "--skip-ingest"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            st.success(" Pipeline complete! Refresh page to see updates.")
            st.balloons()
        else:
            st.error("Pipeline failed. Check logs.")

# Custom divider
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# KPI Cards - Custom styled metrics
st.markdown("###  Current Performance")
col1, col2, col3, col4 = st.columns(4)

latest = df.tail(1).iloc[0]
prev = df.tail(5).head(4).mean() if len(df) >= 5 else latest

with col1:
    rev_change = ((latest['revenue_pkr'] - prev['revenue_pkr']) / prev['revenue_pkr']) * 100
    arrow = "▲" if rev_change > 0 else "▼"
    color = "#19c741" if rev_change > 0 else "#e70e24"
    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <p style="color: #666; margin:0;">Weekly Revenue</p>
        <h2 style="margin:0; color:#333;">PKR {latest['revenue_pkr']/1000:.0f}K</h2>
        <p style="margin:0; color:{color};">{arrow} {abs(rev_change):.1f}% vs last week</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    units_change = ((latest['units_sold'] - prev['units_sold']) / prev['units_sold']) * 100
    arrow = "▲" if units_change > 0 else "▼"
    color = "#10cf3d" if units_change > 0 else "#d11225"
    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <p style="color: #666; margin:0;">Units Sold</p>
        <h2 style="margin:0; color:#333;">{int(latest['units_sold']):,}</h2>
        <p style="margin:0; color:{color};">{arrow} {abs(units_change):.1f}% vs last week</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    margin_color = "#088926" if latest['gross_margin_pct'] > 30 else "#c49509"
    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <p style="color: #666; margin:0;">Gross Margin</p>
        <h2 style="margin:0; color:{margin_color};">{latest['gross_margin_pct']:.1f}%</h2>
        <p style="margin:0; color:#666;">Industry avg: 32%</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    inventory = int(latest['inventory_units'])
    inv_color = "#28a745" if 800 < inventory < 1500 else "#cb9d14" if inventory < 2000 else "#b91828"
    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <p style="color: #666; margin:0;">Inventory</p>
        <h2 style="margin:0; color:{inv_color};">{inventory:,}</h2>
        <p style="margin:0; color:#666;">units in stock</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Charts section
st.markdown("###  Trend Analysis")
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'].tail(26),
        y=df['revenue_pkr'].tail(26),
        mode='lines+markers',
        name='Revenue',
        line=dict(color="#1136da", width=3),
        marker=dict(size=8, color="#410979")
    ))
    fig.update_layout(
        title="Revenue Trend (Last 26 Weeks)",
        xaxis_title="Date",
        yaxis_title="Revenue (PKR)",
        height=350,
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'].tail(26),
        y=df['units_sold'].tail(26),
        mode='lines+markers',
        name='Units Sold',
        line=dict(color="#024a44", width=3),
        marker=dict(size=8, color="#055825")
    ))
    fig.update_layout(
        title="Units Sold Trend (Last 26 Weeks)",
        xaxis_title="Date",
        yaxis_title="Units",
        height=350,
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# AI Analysis Section
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("###  AI Business Analysis")

if log:
    latest_run = log[-1]
    
    # Risk indicator based on inventory recommendation
    inv_rec = latest_run.get('inventory_recommendation', {})
    action = inv_rec.get('action', 'MONITOR')
    
    if action == 'REDUCE':
        risk_class = "risk-high"
        risk_text = "HIGH - Immediate Action Required"
    elif action == 'INCREASE':
        risk_class = "risk-medium"
        risk_text = "MEDIUM - Monitor Closely"
    else:
        risk_class = "risk-low"
        risk_text = "LOW - On Track"
    
    st.markdown(f"""
    <div class="{risk_class}" style="margin-bottom: 1rem;">
        <p style="margin:0; font-size: 0.9rem;">Risk Assessment</p>
        <h3 style="margin:0;">{risk_text}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendation banner
    st.markdown("""
    <div class="rec-banner">
        <p style="margin:0; font-size: 0.9rem;"> Recommendation</p>
        <h3 style="margin:0;">""" + inv_rec.get('action', 'MAINTAIN CURRENT LEVELS') + """</h3>
        <p style="margin:0.5rem 0 0 0; opacity:0.9;">Based on demand forecast & inventory position</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis text
    analysis = latest_run.get('gpt_analysis', 'No analysis available')
    
    # Parse and display
    lines = analysis.split('\n')
    for line in lines:
        if line.startswith('##'):
            st.markdown(f"#### {line.replace('#', '').strip()}")
        elif line.strip() and not line.startswith('-'):
            st.write(line)
        elif line.strip():
            st.markdown(line)
    
    # Inventory metrics
    if inv_rec:
        st.markdown("---")
        st.markdown("###  Inventory Details")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Current Stock", f"{inv_rec.get('current_inventory_units', 0):,} units")
        with m2:
            st.metric("Recommended", f"{inv_rec.get('recommended_inventory_units', 0):,} units")
        with m3:
            gap = inv_rec.get('gap_units', 0)
            st.metric("Gap", f"{gap:+,} units", delta_color="inverse" if gap < 0 else "normal")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p style="color: #666; font-size: 0.8rem;">
         PAKBIZ | Powered by Prophet & Gemini AI
    </p>
</div>
""", unsafe_allow_html=True)

