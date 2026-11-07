# AI-Powered Business Prediction & Calibration System

An end-to-end automated business intelligence system that ingests financial data, runs localized pipeline forecasting, leverages Large Language Models (LLMs) for strategic decision analysis, and features a time-lagged calibration feedback loop to continuously optimize predictive accuracy.

---

## Architecture & Data Pipeline

The core execution path flows sequentially across 6 distinct modules to move from raw data to structural business insights:


1. **`data_ingest`**: Collects and structures raw transactional and business metrics.
2. **`pipeline`**: Cleans data, handles features, and structures time-series frames.
3. **`forecast`**: Extracts demand variations, calculates statistical forecast trends, and structures an optimized string payload for LLM parsing.
4. **`feedback`**: Loads historical prediction logs alongside their recorded ground-truth "actuals" to identify past model drifts and provide self-correcting calibration context.
5. **`llm_analysis`**: Couples forecast payloads with historical feedback matrices to generate qualitative operational strategies using advanced generative reasoning.
6. **`feedback (logging)`**: Saves structural parameters, logs unique tracking IDs, and serializes raw analytical text to dynamic local reports.

---

## System Capabilities

* **Closed-Loop Performance Tracking**: Keeps an immutable ledger of past forecasts. Using subsequent true data (e.g., actual revenue or units sold during peak seasonal events like Eid), the system reviews its own errors before generating fresh forecasts.
* **Inventory Recommendations**: Autonomously produces operational metrics to minimize supply bottlenecks based on forecasted demand volatility.
* **Automated Job Scheduling**: Includes internal structural support for persistent scheduling to run analytics autonomously at critical operational intervals.
* **Dynamic File Compilation**: Generates timestamped reports localized to your business targets (e.g., tracking indicators such as Revenue in `PKR`).

---

## Installation & Setup

### Prerequisites
* Python 3.8+
* Active API keys configured for your underlying LLM service provider.

### Dependencies
Install the required packages using your terminal:
```bash
pip install numpy pandas scipy apscheduler
```
*(Note: Ensure your custom modular dependencies `data_ingest`, `pipeline`, `forecast`, `feedback`, and `llm_analysis` are located within the root directory or your system path.)*

---

## Execution & Usage Guide

The application supports direct command-line arguments to toggle running modes:

### 1. Default Standard Execution
Runs the complete script step-by-step (Data Ingestion ➔ Forecasting Pipeline ➔ LLM Calibration Analysis ➔ Local Report Export):
```bash
python main.py
```

### 2. Fast-Track Pipeline (Skip Ingestion)
Skips the extraction step and instantly recalculates projections from your existing raw data files:
```bash
python main.py --skip-ingest
```

### 3. Review Historical Log Summaries
Prints a performance summary ledger from past inference cycles directly inside your terminal:
```bash
python main.py --log
```

### 4. Enable Production Weekly Automation
Deploys a persistent background daemon via `APScheduler` configured to autonomously process your forecasts **every Monday morning at 08:00 AM**:
```bash
python main.py --schedule
```

---

## Calibration & Recording True Outcomes

To ground the AI and allow the model to actively correct historical error drifts, actual outcomes should be tracked and logged retroactively once a period concludes (e.g., updating metrics manually a week later). 

You can log these true metrics by running an explicit script or calling the module method directly:

```python
from feedback import record_actual

# Record ground truth outcomes for calibration
record_actual(
    run_id=14,  # Use the unique identifier generated from your target run
    actual_revenue_pkr=920000,
    actual_units_sold=2100,
    notes="Eid week, higher consumer traffic than expected."
)
```

## Output Reports
Every execution writes a comprehensive text artifact to the local environment matching this file convention: 
`data/reports/report_YYYYMMDD_HHMM.txt
