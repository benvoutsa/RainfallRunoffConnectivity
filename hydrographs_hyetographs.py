import numpy as np
import pandas as pd
import os
import glob
import re
import xarray as xr
import datatree as dtree
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------
# IMPORT EXISTING MODULES
# ---------------------------------------------------
from rainfall_functions import *
from runoff_functions import *

import yaml
from pathlib import Path

# ---------------------------------------------------
# SETTINGS
# ---------------------------------------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_DIR = Path(config["DATA_DIR"])
RESULTS_DIR = Path(config["RESULTS_DIR"])

figure_names = [
    "hydrograph_events_8_9.pdf",
    "hydrograph_events_14_119.pdf",
    "hydrograph_events_45_201.pdf"
]

# ---------------------------------------------------
# LOAD RUNOFF DATA
# ---------------------------------------------------
tree_00_06 = dtree.open_datatree(os.path.join(DATA_DIR, "runoff", "runoff_events_2000_2006.nc"))
tree_07_13 = dtree.open_datatree(os.path.join(DATA_DIR, "runoff", "runoff_events_2007_2013.nc"))
tree_14_24 = dtree.open_datatree(os.path.join(DATA_DIR, "runoff", "runoff_events_2014_2024.nc"))

tree_keys = {
    "00_06": list(tree_00_06.descendants),
    "07_13": list(tree_07_13.descendants),
    "14_24": list(tree_14_24.descendants)
}

# ---------------------------------------------------
# RAINFALL DATA
# ---------------------------------------------------
rainfall_files = glob.glob(os.path.join(DATA_DIR, "rainfall", "rainfall_*.csv"))

dfs_rainfall = [
    pd.read_csv(f, sep=",", comment="#", skiprows=8, low_memory=False)
    for f in rainfall_files
]

df_rainfall = pd.concat(dfs_rainfall, ignore_index=True)

df_rainfall["Timestamp"] = pd.to_datetime(df_rainfall["Date"] + " " + df_rainfall["Start_Time"])
df_rainfall["Real_Time"] = df_rainfall["Timestamp"] + pd.to_timedelta(df_rainfall["Elapsed_Time (min.)"], unit="m")

df_rainfall = df_rainfall[
    (df_rainfall["Rainfall_Rate (mm/hr)"] != 0) | (df_rainfall["Elapsed_Time (min.)"] == 0)
]

# ---------------------------------------------------
# EVENT DATES
# ---------------------------------------------------
runoff_dates = pd.concat([
    pd.read_csv(os.path.join(DATA_DIR, f))
    for f in [
        "dates_of_runoff_events_2000_2006.csv",
        "dates_of_runoff_events_2007_2013.csv",
        "dates_of_runoff_events_2014_2024.csv"
    ]
], axis=0).reset_index(drop=True)

runoff_dates["start_time"] = pd.to_datetime(runoff_dates["start_time"])
runoff_dates["end_time"] = pd.to_datetime(runoff_dates["end_time"])

rainfall_dates = runoff_dates.copy()
rainfall_dates["start_time"] -= pd.Timedelta(hours=2)
rainfall_dates["end_time"] += pd.Timedelta(hours=2)

# ---------------------------------------------------
# BUILD EVENT DICTIONARY
# ---------------------------------------------------
rainfall_dfs = {}

for i in range(len(rainfall_dates)):
    label = rainfall_dates["event_label"].iloc[i]
    start = rainfall_dates["start_time"].iloc[i]
    end = rainfall_dates["end_time"].iloc[i]

    rainfall_dfs[label] = df_rainfall[
        (df_rainfall["Real_Time"] >= start) &
        (df_rainfall["Real_Time"] <= end)
    ]

# ---------------------------------------------------
# FLUME / COLOR SETUP
# ---------------------------------------------------
flume_data = pd.read_csv(os.path.join(DATA_DIR, "flume_raingauges.csv"))
flume_data["Rain_Gauge_Num"] = flume_data["Rain_gauge_name"].str.extract("(/d+)").astype(int)

float_values = flume_data["Contributing_area_km2"]
integers = flume_data["Flume"]

norm = plt.Normalize(float_values.min(), float_values.max())
colormap = plt.cm.plasma
colors = colormap(norm(float_values))

integer_to_color = dict(zip(integers, colors))

# ---------------------------------------------------
# HELPER: GET RUNOFF DATASET
# ---------------------------------------------------
def get_runoff_ds(key):
    if key in tree_00_06:
        return tree_00_06[key].to_dataset()
    elif key in tree_07_13:
        return tree_07_13[key].to_dataset()
    else:
        return tree_14_24[key].to_dataset()

# ---------------------------------------------------
# PLOTTING FUNCTION
# ---------------------------------------------------
def plot_event(ax, key):

    # rainfall
    df_rain = rainfall_dfs[key]
    ds_rain = df_rain.set_index("Real_Time").to_xarray()

    ax_rain = ax.twinx()

    for _, v in ds_rain.data_vars.items():
        v.plot(ax=ax_rain, color="dodgerblue", alpha=0.5)

    ax_rain.invert_yaxis()
    ax_rain.set_ylabel("rainfall (mm/hr)", color="blue")
    ax_rain.tick_params(axis="y", colors="blue")

    # runoff
    ds_runoff = get_runoff_ds(key)

    for name, v in ds_runoff.data_vars.items():
        num = int(re.findall(r"\d+", name)[0])

        if num == 126:
            num = 113

        v = cfs_to_m3(v).where(lambda x: x > 0)
        v.plot(ax=ax, color=integer_to_color.get(num, "black"))

    ax.set_title(key)
    ax.set_ylabel("runoff (m³/s)")
    ax.set_xlabel("time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

# ---------------------------------------------------
# MAIN LOOP + PDF EXPORT
# ---------------------------------------------------
for group, fig_name in zip(event_groups, figure_names):

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for ax, event in zip(axes, group):
        plot_event(ax, event)

    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, fig_name),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {fig_name}")
