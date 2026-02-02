"""
rainfall_features_extraction.py

Extracts rainfall event features:
- number of gauges per event
- duration
- average intensity
- maximum rolling window intensity

Saves the resulting DataFrame to CSV.
"""

import numpy as np
import pandas as pd
import xarray as xr
import glob
import os

# -------------------------------
# Configurable paths
# -------------------------------
RAINFALL_FOLDER = "data/rainfall_files"
FLUME_RAINGAUGES_FILE = "data/flume_raingauges.csv"
FLUME_WATERSHEDS_FILE = "data/flume_watersheds.csv"
OUTPUT_FEATURES_FILE = "data/df_rainfall_features.csv"

# -------------------------------
# Helper functions
# -------------------------------

def downsample_rainfall_events(ds_event, timestep=4):
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
    window_intensities = [
        df_rainfall_event.iloc[i*time_window:(i+1)*time_window].mean().mean()
        for i in range(int(duration/time_window)-1)
    ]
    return round(max(window_intensities), 2).item()

def find_keys_by_value(dictionary, target_value):
    """Return dictionary keys whose values contain target_value."""
    return [key for key, values in dictionary.items() if target_value in values]

# -------------------------------
# Load data
# -------------------------------

flume_raingauges_data = pd.read_csv(FLUME_RAINGAUGES_FILE)
df_flume_watersheds = pd.read_csv(FLUME_WATERSHEDS_FILE)

flume_raingauges_data['Rain_Gauge_Num'] = [int(''.join(filter(str.isdigit, s))) for s in flume_raingauges_data['Rain_gauge_name'].values if any(char.isdigit() for char in s)]

# Map flumes to gauges
flume_raingauges_dict = {
    flume: flume_raingauges_data.loc[flume_raingauges_data['Flume'] == flume, 'Rain_Gauge_Num'].tolist()
    for flume in df_flume_watersheds['Flume'].unique()
}

# -------------------------------
# Feature extraction
# -------------------------------

df_rainfall_features = pd.DataFrame(columns=[
    'number of gauges', 'duration', 'average intensity (mm/hr)', 'max window intensity (mm/hr)'
])

number_of_gauges = []
event_durations = []
average_intensities = []
max_average_window_intensities = []

for key in tree_keys:  # `tree_keys` and `ds_rainfall_events` should already be defined
    ds_event = ds_rainfall_events[key]
    df_event = ds_event.to_dataframe()
    
    event_gauge_list = df_event.columns
    event_gauge_nums = [int(''.join(filter(str.isdigit, s))) for s in event_gauge_list if any(c.isdigit() for c in s)]

    avg_intensity = df_event.mean().mean()
    time_window = select_time_window_size(df_event)
    max_window_intensity = max_average_intensity_within_time_window(df_event, time_window)

    event_durations.append(len(ds_event.time))
    average_intensities.append(round(avg_intensity, 2))
    max_average_window_intensities.append(max_window_intensity)

    # Number of gauges per event
    event_flumes = np.unique([flume for gauge_num in event_gauge_nums
                              for flume in find_keys_by_value(flume_raingauges_dict, gauge_num)])
    number_of_gauges.append(len(event_flumes))

# Assign features to DataFrame
df_rainfall_features['duration'] = event_durations
df_rainfall_features['average intensity (mm/hr)'] = average_intensities
df_rainfall_features['max window intensity (mm/hr)'] = max_average_window_intensities
df_rainfall_features['number of gauges'] = number_of_gauges

# Save to CSV
df_rainfall_features.to_csv(OUTPUT_FEATURES_FILE, index=False)
print(f"Rainfall features saved to {OUTPUT_FEATURES_FILE}")

