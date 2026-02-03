"""
rainfall_functions.py

Helper functions to separate rainfall events based on time windows of runoff events and rainfall feature extraction.

"""
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract constants
RAINFALL_BUFFER_HOURS = config["rain_event_padding_hours"]
time_step = config["rainfall_time_step"]
downsample_factor = config["downsample_timestep"]


def load_rainfall_csvs(base_directory, rainfall_subdir="rainfall", pattern="rainfall_*.csv"):
    
    path_pattern = os.path.join(base_directory, rainfall_subdir, pattern)
    path_pattern = os.path.normpath(path_pattern)

    csv_files = glob.glob(path_pattern)

    if not csv_files:
        raise FileNotFoundError(f"No rainfall CSVs found at {path_pattern}")

    dfs = [pd.read_csv(file, sep=",", comment="#", skiprows=8, low_memory=False,) for file in csv_files]

    return dfs


def prepare_rainfall_dataframe(dfs_rainfall):
    """
    Merge and clean rainfall data into a standardized dataframe.
    """
    df = pd.concat(dfs_rainfall, ignore_index=True)

    # Build timestamps
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Start_Time"])
    df["Real_Time"] = df["Timestamp"] + pd.to_timedelta(
        df["Elapsed_Time (min.)"], unit="m"
    )

    # Remove internal zero rainfall artifacts
    df = df.drop(df[(df["Rainfall_Rate (mm/hr)"] == 0) & (df["Elapsed_Time (min.)"] != 0)].index)

    return df[["Gage", "Rainfall_Rate (mm/hr)", "Depth (mm)", "Real_Time"]]


def downsample_rainfall_events(ds_event, timestep):
    """Downsample xarray Dataset along all dimensions."""
    return xr.Dataset(
        {var: ds_event[var].isel({dim: slice(None, None, timestep) for dim in ds_event.dims})
         for var in ds_event.data_vars},
        coords={dim: ds_event.coords[dim][::timestep] for dim in ds_event.dims}
    )

def select_time_window_size(df_rainfall_event):
    """Select appropriate time window based on event duration."""
    time_windows_dict = {15: (0, 90), 30: (90, 270), 60: (271, 360), 120: (360, 5000)}
    duration = df_rainfall_event.shape[0]
    for window, (min_dur, max_dur) in time_windows_dict.items():
        if min_dur <= duration <= max_dur:
            return window

def max_average_intensity_within_time_window(df_rainfall_event, time_window):
    """Compute maximum average intensity across rolling windows of given size."""
    duration = df_rainfall_event.shape[0]
    n_windows = int(duration / time_window) - 1

    if n_windows < 1:
        # Event too short for rolling window
        return 0
        
    window_intensities = [
        df_rainfall_event.iloc[i*time_window:(i+1)*time_window].mean().mean()
        for i in range(int(duration/time_window)-1)
    ]
    return round(max(window_intensities), 2).item()

def find_keys_by_value(dictionary, target_value):
    """Return dictionary keys whose values contain target_value."""
    return [key for key, values in dictionary.items() if target_value in values]


def build_rainfall_windows(runoff_dates, buffer_hours):
    """Expand runoff events to rainfall windows."""
    rainfall_dates = runoff_dates.copy()

    rainfall_dates["start_time"] -= timedelta(hours=buffer_hours)
    rainfall_dates["end_time"] += timedelta(hours=buffer_hours)

    return rainfall_dates[["event_label", "start_time", "end_time"]]

def extract_flumes_from_event(xr_event):
    flume_strs = list(xr_event.data_vars)
    return np.unique([
        int("".join(filter(str.isdigit, s)))
        for s in flume_strs if any(char.isdigit() for char in s)
    ])


def load_flume_raingauge_mapping(path):
    df = pd.read_csv(path)
    df["Rain_Gauge_Num"] = [
        int("".join(filter(str.isdigit, s)))
        for s in df["Rain_gauge_name"].values
    ]
    return df


def get_event_gauges(flumes, mapping_df):
    return np.unique(
        mapping_df[mapping_df["Flume"].isin(flumes)].Rain_Gauge_Num.values
    )

def extract_event_rainfall(df_rainfall, event_row, gauges):
    df = df_rainfall[(df_rainfall.Real_Time >= event_row.start_time) & 
         (df_rainfall.Real_Time <= event_row.end_time) &
        (df_rainfall.Gage.isin(gauges))]

    if df.empty:
        return None

    time_index = pd.date_range(
        df.Real_Time.min(),
        df.Real_Time.max(),
        freq=time_step
    )

    rainfall = {f"gauge_{int(g)}": (["time"], df[df.Gage == g].set_index("Real_Time").reindex(time_index)["Rainfall_Rate (mm/hr)"].fillna(0).values)
                for g in gauges}

    return xr.Dataset(rainfall, coords={"time": time_index})

def trim_zero_rainfall(ds, threshold=1):
    df = ds.to_dataframe().apply(pd.to_numeric, errors="coerce")

    non_zero_mask = ~(df <= threshold).all(axis=1)
    if not non_zero_mask.any():
        return None

    trimmed = df.loc[non_zero_mask.idxmax():non_zero_mask[::-1].idxmax()]

    return xr.Dataset({v: (["time"], trimmed[v].values) for v in trimmed.columns}, coords={"time": trimmed.index})


def build_rainfall_events(runoff_trees, rainfall_df, runoff_dates, flume_raingauges):
    rainfall_events = {}

    runoff_dates = runoff_dates.set_index("event_label")

    for tree in runoff_trees:
        for node in tree.descendants:

            event_label = node.name
            if event_label not in runoff_dates.index:
                continue

            flumes = extract_flumes_from_event(tree[event_label])
            gauges = get_event_gauges(flumes, flume_raingauges)

            ds = extract_event_rainfall(rainfall_df, runoff_dates.loc[event_label], gauges)

            if ds is None:
                continue

            ds = trim_zero_rainfall(ds)
            if ds is None:
                continue

            ds = downsample_rainfall_events(ds, timestep=downsample_factor)
            rainfall_events[event_label] = ds

    return rainfall_events
