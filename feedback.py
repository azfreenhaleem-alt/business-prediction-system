

import os
import json
from datetime import datetime


LOG_PATH = "data/prediction_log.json"


def load_log() -> list:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_log(log: list) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)


def log_prediction(
    forecast_summary: str,
    inventory_recommendation: dict,
    gpt_analysis: str,
    provider: str,
) -> int:
    """
    Save this run's predictions to the log.
    Returns the index of the new entry.
    """
    log = load_log()

    entry = {
        "run_id":                  len(log),
        "timestamp":               datetime.now().isoformat(),
        "provider":                provider,
        "forecast_summary":        forecast_summary,
        "inventory_recommendation": inventory_recommendation,
        "gpt_analysis":            gpt_analysis,
        "actual":                  None,   # filled in later via record_actual()
    }

    log.append(entry)
    save_log(log)
    print(f"  [ok] Prediction logged (run #{entry['run_id']}) → {LOG_PATH}")
    return entry["run_id"]


def record_actual(
    run_id: int,
    actual_revenue_pkr: float,
    actual_units_sold: int,
    notes: str = "",
) -> None:
    """
    After a week/period passes, call this to record what actually happened.
    This gets picked up in the next run's prompt for calibration.

    Usage:
        from feedback import record_actual
        record_actual(run_id=0, actual_revenue_pkr=920000, actual_units_sold=2100)
    """
    log = load_log()

    if run_id >= len(log):
        print(f"  [!] No entry with run_id={run_id}")
        return

    log[run_id]["actual"] = {
        "revenue_pkr":  actual_revenue_pkr,
        "units_sold":   actual_units_sold,
        "notes":        notes,
        "recorded_at":  datetime.now().isoformat(),
    }

    save_log(log)
    print(f"  [ok] Actual data recorded for run #{run_id}")


def get_latest_with_actual() -> dict | None:
    """
    Returns the most recent log entry that has actual outcomes recorded.
    Used to build the calibration section of the next prompt.
    """
    log = load_log()
    for entry in reversed(log):
        if entry.get("actual") is not None:
            return entry
    return None


def get_latest_prediction() -> dict | None:
    """Returns the most recent log entry regardless of actual status."""
    log = load_log()
    return log[-1] if log else None


def print_log_summary() -> None:
    """Print a human-readable summary of all logged runs."""
    log = load_log()
    if not log:
        print("  No predictions logged yet.")
        return

    print(f"\n{'='*55}")
    print(f"  PREDICTION LOG ({len(log)} runs)")
    print(f"{'='*55}")
    for entry in log:
        has_actual = "actual recorded" if entry.get("actual") else "○ awaiting actual"
        print(f"  Run #{entry['run_id']:02d}  {entry['timestamp'][:16]}  [{has_actual}]  via {entry['provider']}")
    print()


if __name__ == "__main__":
    print_log_summary()
