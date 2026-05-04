import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import datatree as dtree
import xarray as xr

from rainfall_functions import (
    load_rainfall_csvs,
    prepare_rainfall_dataframe,
    build_rainfall_windows,
    extract_event_rainfall,
    trim_zero_rainfall,
    downsample_rainfall_events,
)

from runoff_functions import (
    load_runoff_trees,
    load_runoff_dates,
)


BASE_DIR = "data"

RUNOFF_FILES = [
    os.path.join(BASE_DIR, "runoff", "runoff_events_2000_2006.nc"),
    os.path.join(BASE_DIR, "runoff", "runoff_events_2007_2013.nc"),
    os.path.join(BASE_DIR, "runoff", "runoff_events_2014_2024.nc"),
]

RUNOFF_DATE_FILES = [
    os.path.join(BASE_DIR, "dates_of_runoff_events_2000_2006.csv"),
    os.path.join(BASE_DIR, "dates_of_runoff_events_2007_2013.csv"),
    os.path.join(BASE_DIR, "dates_of_runoff_events_2014_2024.csv"),
]

FLUME_MAPPING_PATH = os.path.join(BASE_DIR, "flume_raingauges.csv")


def cfs_to_m3(cfs):
    return cfs * 0.0283168


def get_runoff_event(tree_list, event_label):
    for tree in tree_list:
        if event_label in tree:
            return tree[event_label].to_dataset()
    return None



def build_color_map(flume_mapping_df):

    values = flume_mapping_df["Contributing_area_km2"].values
    flumes = flume_mapping_df["Flume"].values

    norm = plt.Normalize(values.min(), values.max())
    cmap = plt.cm.plasma

    colors = cmap(norm(values))

    return dict(zip(flumes, colors)), cmap, norm


def build_rainfall_events(runoff_dates, df_rainfall):

    rainfall_windows = build_rainfall_windows(runoff_dates, buffer_hours=2)

    rainfall_events = {}

    for _, row in rainfall_windows.iterrows():

        ds = extract_event_rainfall(
            df_rainfall=df_rainfall,
            event_row=row,
            gauges=df_rainfall["Gage"].unique()
        )

        if ds is None:
            continue

        ds = trim_zero_rainfall(ds)
        if ds is None:
            continue

        ds = downsample_rainfall_events(ds, timestep=4)

        rainfall_events[row["event_label"]] = ds

    return rainfall_events


def plot_event(ax, ax_rain, event_label, rainfall_events, runoff_tree, color_map):

    # ---------- rainfall ----------
    if event_label in rainfall_events:
        ds_rain = rainfall_events[event_label]

        for _, data in ds_rain.data_vars.items():
            data.plot(ax=ax_rain, color="dodgerblue", alpha=0.5)

        ax_rain.invert_yaxis()
        ax_rain.set_ylabel("Rainfall (mm/hr)", color="blue")
        ax_rain.tick_params(axis="y", colors="blue")

    # ---------- runoff ----------
    if runoff_tree is None:
        return

    for flume_name, flume_data in runoff_tree.data_vars.items():

        nums = re.findall(r"\d+", flume_name)
        if not nums:
            continue

        flume_id = int(nums[0])

        runoff = cfs_to_m3(flume_data)
        runoff = runoff.where(runoff > 0)

        ax.plot(
            runoff.time,
            runoff.values,
            color=color_map.get(flume_id, "black"),
            linewidth=1
        )

    ax.set_ylabel("Runoff (m³/s)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

runoff_trees = load_runoff_trees(RUNOFF_FILES)
runoff_dates = load_runoff_dates(RUNOFF_DATE_FILES)

dfs_rainfall = load_rainfall_csvs(BASE_DIR)
df_rainfall = prepare_rainfall_dataframe(dfs_rainfall)

flume_mapping = pd.read_csv(FLUME_MAPPING_PATH)

color_map, cmap, norm = build_color_map(flume_mapping)

rainfall_events = build_rainfall_events(runoff_dates, df_rainfall)


event_pairs = [
    ("event_8", "event_9"),
    ("event_14", "event_119"),
    ("event_45", "event_201"),
]


output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

for i, (e1, e2) in enumerate(event_pairs):

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    ax1, ax2 = axes

    ax1_rain = ax1.twinx()
    ax2_rain = ax2.twinx()

    # ---- event 1 ----
    runoff1 = get_runoff_event(runoff_trees, e1)
    plot_event(ax1, ax1_rain, e1, rainfall_events, runoff1, color_map)
    ax1.set_title(e1)

    # ---- event 2 ----
    runoff2 = get_runoff_event(runoff_trees, e2)
    plot_event(ax2, ax2_rain, e2, rainfall_events, runoff2, color_map)
    ax2.set_title(e2)

    # formatting
    for ax in axes:
        ax.set_xlabel("Time")
        ax.tick_params(axis="y")

    plt.tight_layout()

    outpath = os.path.join(output_dir, f"event_pair_{i+1}.pdf")
    plt.savefig(outpath, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {outpath}")
