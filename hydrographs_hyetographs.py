"""
hydrographs_hyetographs.py

Plot runoff hydrographs together with rainfall hyetographs
for selected runoff events.

Refactored to reuse helper functions from:
- rainfall_functions.py
- runoff_functions.py
"""

# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datatree as dtree
import yaml

from rainfall_functions import (
    load_rainfall_csvs,
    prepare_rainfall_dataframe,
    build_rainfall_windows,
    load_flume_raingauge_mapping,
)

from runoff_functions import (
    load_runoff_dates,
)

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_DIR = Path(config["DATA_DIR"])
RESULTS_DIR = Path(config["RESULTS_DIR"])

RAINFALL_BUFFER_HOURS = config["rain_event_padding_hours"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------

event_groups = [
    ["event_8", "event_9"],
    ["event_14", "event_119"],
    ["event_45", "event_201"],
]

figure_names = [
    "hydrograph_events_8_9.pdf",
    "hydrograph_events_14_119.pdf",
    "hydrograph_events_45_201.pdf",
]

# ---------------------------------------------------
# LOAD RUNOFF EVENT TREES
# ---------------------------------------------------
runoff_tree_files = [
    DATA_DIR / "runoff" / "runoff_2000_2006.nc",
    DATA_DIR / "runoff" / "runoff_2007_2013.nc",
    DATA_DIR / "runoff" / "runoff_2014_2024.nc",
]

runoff_trees = [
    dtree.open_datatree(file, format="NETCDF4")
    for file in runoff_tree_files
]

# ---------------------------------------------------
# LOAD RAINFALL DATA
# ---------------------------------------------------
dfs_rainfall = load_rainfall_csvs(DATA_DIR)
df_rainfall = prepare_rainfall_dataframe(dfs_rainfall)

# ---------------------------------------------------
# LOAD EVENT DATES
# ---------------------------------------------------
runoff_date_files = [
    DATA_DIR / "dates_of_runoff_events_2000_2006.csv",
    DATA_DIR / "dates_of_runoff_events_2007_2013.csv",
    DATA_DIR / "dates_of_runoff_events_2014_2024.csv",
]

runoff_dates = load_runoff_dates(runoff_date_files)

# ---------------------------------------------------
# BUILD RAINFALL WINDOWS
# ---------------------------------------------------
rainfall_dates = build_rainfall_windows(
    runoff_dates,
    buffer_hours=RAINFALL_BUFFER_HOURS,
)

# ---------------------------------------------------
# BUILD EVENT -> RAINFALL DATAFRAME DICTIONARY
# ---------------------------------------------------
rainfall_event_data = {}

for _, row in rainfall_dates.iterrows():

    rainfall_event_data[row.event_label] = df_rainfall[
        (df_rainfall.Real_Time >= row.start_time)
        & (df_rainfall.Real_Time <= row.end_time)
    ]

# ---------------------------------------------------
# LOAD FLUME / RAINGAUGE MAPPING
# ---------------------------------------------------
flume_data = load_flume_raingauge_mapping(
    DATA_DIR / "flume_raingauges.csv"
)

# ---------------------------------------------------
# CREATE FLUME COLOR MAPPING
# ---------------------------------------------------
norm = plt.Normalize(
    flume_data["Contributing_area_km2"].min(),
    flume_data["Contributing_area_km2"].max(),
)

colormap = plt.cm.plasma

colors = colormap(
    norm(flume_data["Contributing_area_km2"])
)

flume_to_color = dict(
    zip(flume_data["Flume"], colors)
)

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------
def cfs_to_m3(flow_cfs):
    """Convert cubic feet/sec to cubic meters/sec."""
    return flow_cfs * 0.0283168


def get_runoff_dataset(event_label, runoff_trees):
    """
    Search all runoff trees for an event label
    and return its dataset.
    """

    for tree in runoff_trees:

        if event_label in tree:
            return tree[event_label].to_dataset()

    raise KeyError(f"Event '{event_label}' not found.")


# ---------------------------------------------------
# PLOTTING
# ---------------------------------------------------
def plot_event(ax, event_label):
    """
    Plot rainfall + runoff for a single event.
    """

    # ------------------------------------------------
    # RAINFALL
    # ------------------------------------------------
    df_rain = rainfall_event_data[event_label]

    ax_rain = ax.twinx()

    for gauge in np.unique(df_rain["Gage"]):

        gauge_df = df_rain[df_rain["Gage"] == gauge]

        ax_rain.plot(
            gauge_df["Real_Time"],
            gauge_df["Rainfall_Rate (mm/hr)"],
            color="dodgerblue",
            alpha=0.4,
            linewidth=1,
        )

    ax_rain.invert_yaxis()

    ax_rain.set_ylabel(
        "rainfall (mm/hr)",
        color="dodgerblue",
    )

    ax_rain.tick_params(
        axis="y",
        colors="dodgerblue",
    )

    # ------------------------------------------------
    # RUNOFF
    # ------------------------------------------------
    ds_runoff = get_runoff_dataset(
        event_label,
        runoff_trees,
    )

    for variable_name, values in ds_runoff.data_vars.items():

        flume_num = int(
            re.findall(r"\d+", variable_name)[0]
        )

        # special correction
        if flume_num == 126:
            flume_num = 113

        runoff = cfs_to_m3(values)
        runoff = runoff.where(runoff > 0)

        ax.plot(
            ds_runoff.time.values,
            runoff.values,
            color=flume_to_color.get(flume_num, "black"),
            linewidth=1.5,
        )

    # ------------------------------------------------
    # AXIS FORMATTING
    # ------------------------------------------------
    ax.set_title(event_label)

    ax.set_ylabel("runoff (m³/s)")
    ax.set_xlabel("time")

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M")
    )


# ---------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------
for group, figure_name in zip(
    event_groups,
    figure_names,
):

    fig, axes = plt.subplots(
        len(group),
        1,
        figsize=(11, 7),
        sharex=True,
    )

    # handle single subplot edge case
    if len(group) == 1:
        axes = [axes]

    for ax, event_label in zip(axes, group):

        plot_event(
            ax=ax,
            event_label=event_label,
        )

    plt.tight_layout()

    output_path = RESULTS_DIR / figure_name

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")
