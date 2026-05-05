

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="Business Prediction System",
    page_icon="",
    layout="wide"
)

st.title("Business Prediction System")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# Sidebar 
with st.sidebar:
    st.header("Controls")

    run_pipeline = st.button(" Run Full Pipeline", type="primary", use_container_width=True)
    skip_ingest  = st.checkbox("Skip data ingestion (use cached)", value=True)

    st.divider()
    st.subheader("Record Actual Outcomes")
    run_id_input   = st.number_input("Run ID", min_value=0, value=0, step=1)
    actual_rev     = st.number_input("Actual Revenue (PKR)", value=900000, step=10000)
    actual_units   = st.number_input("Actual Units Sold", value=2000, step=50)
    record_btn     = st.button("Record Actuals", use_container_width=True)

    if record_btn:
        from feedback import record_actual
        record_actual(int(run_id_input), float(actual_rev), int(actual_units))
        st.success(f"Recorded for run #{run_id_input}")

    st.divider()
    st.caption("Free tier: Gemini Flash\nFallback: Ollama (local)")


# Run pipeline if requested 
if run_pipeline:
    with st.spinner("Running full pipeline..."):
        import subprocess, sys
        flag = "--skip-ingest" if skip_ingest else ""
        cmd  = [sys.executable, "main.py"] + ([flag] if flag else [])
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            st.success("Pipeline complete!")
        else:
            st.error(f"Pipeline failed:\n{result.stderr[-1000:]}")


# Load data 
@st.cache_data(ttl=300)
def load_data():
    path = "data/processed/merged.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"])

@st.cache_data(ttl=300)
def load_log():
    path = "data/prediction_log.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)

df  = load_data()
log = load_log()


# KPI cards 
if df is not None and not df.empty:
    latest   = df.tail(1).iloc[0]
    prev4avg = df.tail(5).head(4)

    st.subheader("Current Performance")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        rev = latest.get("revenue_pkr", 0)
        rev_prev = prev4avg["revenue_pkr"].mean() if "revenue_pkr" in df.columns else rev
        delta = f"{(rev - rev_prev)/rev_prev*100:+.1f}%" if rev_prev else ""
        st.metric("Weekly Revenue", f"PKR {rev:,.0f}", delta)

    with c2:
        units = latest.get("units_sold", 0)
        units_prev = prev4avg["units_sold"].mean() if "units_sold" in df.columns else units
        delta = f"{(units - units_prev)/units_prev*100:+.1f}%" if units_prev else ""
        st.metric("Units Sold", f"{int(units):,}", delta)

    with c3:
        gm = latest.get("gross_margin_pct", 0)
        st.metric("Gross Margin", f"{gm:.1f}%")

    with c4:
        inv = latest.get("inventory_units", 0)
        st.metric("Inventory", f"{int(inv):,} units")

    with c5:
        sat = latest.get("satisfaction", 0)
        st.metric("Satisfaction", f"{sat:.2f} / 5")

    st.divider()

    # Revenue chart 
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue Trend")
        rev_df = df[["date","revenue_pkr"]].tail(52).copy()
        rev_df = rev_df.rename(columns={"revenue_pkr": "Revenue (PKR)"})
        st.line_chart(rev_df.set_index("date"))

    with col2:
        st.subheader("Units Sold Trend")
        if "units_sold" in df.columns:
            units_df = df[["date","units_sold"]].tail(52).copy()
            units_df = units_df.rename(columns={"units_sold": "Units Sold"})
            st.line_chart(units_df.set_index("date"))

    # Macro indicators 
    st.subheader("Macro Indicators (last 52 weeks)")
    macro_cols = [c for c in ["USD_index","oil_price","us_consumer_conf","fed_rate"] if c in df.columns]
    if macro_cols:
        macro_df = df[["date"] + macro_cols].tail(52).set_index("date")
        st.line_chart(macro_df)

else:
    st.info("No processed data found. Click **Run Full Pipeline** to generate data.")


# Latest LLM analysis 
st.subheader("Latest AI Analysis")

if log:
    latest_run = log[-1]
    st.caption(f"Run #{latest_run['run_id']} — {latest_run['timestamp'][:16]} — via {latest_run['provider']}")

    with st.expander("Forecast data used", expanded=False):
        st.text(latest_run.get("forecast_summary",""))

    analysis_text = latest_run.get("gpt_analysis","No analysis available.")
    st.markdown(analysis_text)

    # Inventory recommendation
    inv = latest_run.get("inventory_recommendation", {})
    if inv:
        st.divider()
        st.subheader("Inventory Recommendation")
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Current Stock",      f"{inv.get('current_inventory_units', 0):,} units")
        ic2.metric("Recommended Level",  f"{inv.get('recommended_inventory_units', 0):,} units")
        ic3.metric("Gap",                f"{inv.get('gap_units', 0):+,} units")
        ic4.metric("Action",             inv.get("action","").upper())
else:
    st.info("No analysis yet. Run the pipeline first.")


#  Prediction log table 
st.divider()
st.subheader("Prediction History")

if log:
    log_df = pd.DataFrame([{
        "Run":      e["run_id"],
        "Date":     e["timestamp"][:16],
        "Provider": e["provider"],
        "Actuals":  "✓" if e.get("actual") else "○",
        "Action":   e.get("inventory_recommendation", {}).get("action","—"),
    } for e in log])
    st.dataframe(log_df, use_container_width=True, hide_index=True)
else:
    st.caption("No runs logged yet.")


# Footer 
st.divider()
st.caption("Built with Prophet · Gemini Flash · Streamlit · FRED API")
