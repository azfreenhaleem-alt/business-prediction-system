
import sys
import os
from datetime import datetime


def run(skip_ingest: bool = False) -> None:
    print("\n" + "=" * 55)
    print("  BUSINESS PREDICTION SYSTEM")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)

    # Step 1: Ingest 
    if not skip_ingest:
        from data_ingest import run_ingestion
        run_ingestion()
    else:
        print("\n[skip] Data ingestion skipped (using existing raw data)")

    # Step 2: Pipeline 
    from pipeline import run_pipeline
    df = run_pipeline()

    # Step 3: Forecast 
    from forecast import build_forecast_summary, format_forecast_for_prompt
    summaries   = build_forecast_summary(df)
    fc_text     = format_forecast_for_prompt(summaries)

    # Step 4: Load feedback from previous run
    from feedback import get_latest_with_actual, get_latest_prediction
    previous_log = get_latest_with_actual()
    if previous_log:
        print(f"\n  [feedback] Using calibration data from run #{previous_log['run_id']}")
    else:
        prev = get_latest_prediction()
        if prev:
            print(f"\n  [feedback] Previous run #{prev['run_id']} has no actuals recorded yet.")
            print("             Tip: call feedback.record_actual() after a week to enable calibration.")
        else:
            print("\n  [feedback] First run — no prior predictions to compare.")

    # Step 5: LLM analysis 
    from llm_analysis import run_analysis
    analysis, provider = run_analysis(df, fc_text, previous_log)

    #  Step 6: Log this run 
    from feedback import log_prediction
    inv_rec = summaries.get("demand", {}).get("inventory_recommendation", {})
    run_id  = log_prediction(fc_text, inv_rec, analysis, provider)

    # Output 
    print("\n" + "=" * 55)
    print(f"  ANALYSIS (via {provider})")
    print("=" * 55)
    print(analysis)

    # Save output to file
    os.makedirs("data/reports", exist_ok=True)
    report_path = f"data/reports/report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    with open(report_path, "w") as f:
        f.write(f"Business Prediction Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Provider: {provider}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write("=" * 55 + "\n\n")
        f.write("FORECAST DATA\n")
        f.write(fc_text + "\n\n")
        f.write("LLM ANALYSIS\n")
        f.write(analysis + "\n")

    print(f"\n[saved] Report → {report_path}")
    print(f"\nTip: To record actual outcomes after this period:\n")
    print(f"     from feedback import record_actual")
    print(f"     record_actual(run_id={run_id}, actual_revenue_pkr=..., actual_units_sold=...)")
    print()


def schedule_weekly() -> None:
    """Set up APScheduler to run every Monday at 8am."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        scheduler = BlockingScheduler()
        scheduler.add_job(
            lambda: run(skip_ingest=False),
            "cron",
            day_of_week="mon",
            hour=8,
            minute=0,
        )
        print("\n[scheduler] Weekly run scheduled for every Monday at 08:00")
        print("            Press Ctrl+C to stop\n")
        scheduler.start()
    except ImportError:
        print("[!] APScheduler not installed. Run: pip install apscheduler")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--log" in args:
        from feedback import print_log_summary
        print_log_summary()
    elif "--schedule" in args:
        schedule_weekly()
    elif "--skip-ingest" in args:
        run(skip_ingest=True)
    else:
        run(skip_ingest=False)
        
from feedback import record_actual

record_actual(
    run_id=0,                    # from the log
    actual_revenue_pkr=920000,
    actual_units_sold=2100,
    notes="Eid week, higher than expected"
)
