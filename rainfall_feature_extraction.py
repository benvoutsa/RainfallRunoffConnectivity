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

from rainfall_functions import *
from runoff_functions import load_runoff_trees, load_runoff_dates, get_all_event_labels

import yaml

# -------------------------------
# Configurable paths
# -------------------------------
RAINFALL_FOLDER = "data/rainfall_files"
FLUME_RAINGAUGES_FILE = "data/flume_raingauges.csv"
FLUME_WATERSHEDS_FILE = "data/flume_watersheds.csv"
OUTPUT_FEATURES_FILE = "data/df_rainfall_features.csv"

RUNOFF_EVENT_FILES = ["data/runoff_events_2000_2006.nc", "data/runoff_events_2007_2013.nc", "data/runoff_events_2014_2024.nc"]
RUNOFF_DATES_FILES = ["data/dates_of_runoff_events_2000_2006.csv", "data/dates_of_runoff_events_2007_2013.csv", "data/dates_of_runoff_events_2014_2024.csv"]

# -------------------------------
#  config variables
#--------------------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract constants
RAINFALL_BUFFER_HOURS = config["rain_event_padding_hours"]
time_step = config["rainfall_time_step"]
downsample_factor = config["downsample_timestep"]

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

runoff_trees = load_runoff_trees(RUNOFF_EVENT_FILES)
runoff_dates = load_runoff_dates(RUNOFF_DATES_FILES)
rainfall_dates = build_rainfall_windows(runoff_dates, RAINFALL_BUFFER_HOURS)

runoff_trees = load_runoff_trees(RUNOFF_EVENT_FILES)
tree_keys = get_all_event_labels(runoff_trees)

dfs_rainfall = load_rainfall_csvs(RAINFALL_FOLDER)

# Clean & merge
df_rainfall = prepare_rainfall_dataframe(dfs_rainfall)

ds_rainfall_events = build_rainfall_events(runoff_trees, df_rainfall, rainfall_dates, flume_raingauges_data)

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

for key in tree_keys: 
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






