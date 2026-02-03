# rainfall_functions.py


def build_rainfall_windows(runoff_dates, buffer_hours=2):
    """Expand runoff events to rainfall windows."""
    rainfall_dates = runoff_dates.copy()

    rainfall_dates["start_time"] -= timedelta(hours=buffer_hours)
    rainfall_dates["end_time"] += timedelta(hours=buffer_hours)

    return rainfall_dates[["event_label", "start_time", "end_time"]]
