import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

import datatree as dtree
import xarray as xr

from rainfall_functions import load_rainfall_csvs, prepare_rainfall_dataframe, build_rainfall_windows, extract_event_rainfall,\
    build_rainfall_events, trim_zero_rainfall, downsample_rainfall_events, load_flume_raingauge_mapping


from runoff_functions import load_runoff_trees, load_runoff_dates

from sc_fc_functions import load_flume_coordinates, load_contributing_areas, load_edge_list, compute_sc_sim, compute_fc_seq_for_event

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

flume_order = config["runoff"]["flume_order"]

print(flume_order)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = "data"

RUNOFF_FILES = [os.path.join(BASE_DIR, "runoff_events_2000_2006.nc"),
    os.path.join(BASE_DIR, "runoff_events_2007_2013.nc"),
    os.path.join(BASE_DIR, "runoff_events_2014_2024.nc")]

RUNOFF_DATE_FILES = [os.path.join(BASE_DIR, "dates_of_runoff_events_2000_2006.csv"),
    os.path.join(BASE_DIR, "dates_of_runoff_events_2007_2013.csv"),
    os.path.join(BASE_DIR, "dates_of_runoff_events_2014_2024.csv")]

FLUME_RAINGAUGES_PATH = os.path.join(BASE_DIR, "flume_raingauges.csv")
FLUME_WATERSHEDS_PATH = os.path.join(BASE_DIR, "flume_watersheds.csv")

df_coords = load_flume_coordinates()
df_areas = load_contributing_areas()
df_edges_seq = load_edge_list()

# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------
def cfs_to_m3(cfs):
    return cfs * 0.0283168


def get_runoff_event(tree_list, event_label):
    for tree in tree_list:
        if event_label in tree:
            return tree[event_label].to_dataset()
    return None


# ------------------------------------------------------------
# FIXED COLOR MAP (uses contributing area)
# ------------------------------------------------------------
def build_color_map(flume_master_df):

    df = flume_master_df[["Flume", "Contributing_area_km2"]].drop_duplicates()

    df["Contributing_area_km2"] = pd.to_numeric(
        df["Contributing_area_km2"],
        errors="coerce"
    )

    values = df["Contributing_area_km2"].values
    flumes = df["Flume"].values

    norm = plt.Normalize(np.nanmin(values), np.nanmax(values))
    cmap = plt.cm.plasma

    colors = cmap(norm(values))

    return dict(zip(flumes, colors)), cmap, norm


# ------------------------------------------------------------
# load and merge flume data
# ------------------------------------------------------------
def load_flume_master():

    flume_raingauges = pd.read_csv(
        FLUME_RAINGAUGES_PATH,
        sep=r"\s+|\t+|,",
        engine="python"
    )

    flume_watersheds = pd.read_csv(
        FLUME_WATERSHEDS_PATH,
        sep=r"\s+|\t+|,",
        engine="python"
    )

    flume_raingauges.columns = flume_raingauges.columns.str.strip()
    flume_watersheds.columns = flume_watersheds.columns.str.strip()

    flume_master = flume_raingauges.merge(
        flume_watersheds,
        on="Flume",
        how="left"
    )

    if "Contributing_area_km2" not in flume_master.columns:
        raise ValueError("Missing Contributing_area_km2 after merge")

    return flume_master


# ------------------------------------------------------------
# plotting
# ------------------------------------------------------------
def plot_event(ax, ax_rain, event_label, rainfall_events, runoff_tree, color_map):

    # ---------- rainfall ----------
    if event_label in rainfall_events:
        ds_rain = rainfall_events[event_label]
    
        for _, data in ds_rain.data_vars.items():
            data.plot(ax=ax_rain, color="dodgerblue", alpha=0.5)
    
        ax_rain.invert_yaxis()
    
        rain_max = max(
            float(data.max().values)
            for data in ds_rain.data_vars.values()
        )
    
        padding = rain_max * 0.1
    
        ax_rain.set_ylim(rain_max + padding, 0)
        ax_rain.margins(y=0)
    
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


# ------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------
runoff_trees = load_runoff_trees(RUNOFF_FILES)
runoff_dates = load_runoff_dates(RUNOFF_DATE_FILES)

dfs_rainfall = load_rainfall_csvs(BASE_DIR)
df_rainfall = prepare_rainfall_dataframe(dfs_rainfall)
df_rainfall = df_rainfall.dropna(subset=["Gage"])

flume_master = load_flume_master()
color_map, cmap, norm = build_color_map(flume_master)

# map raingauges to corresponding flumes
flume_raingauges = load_flume_raingauge_mapping(FLUME_RAINGAUGES_PATH)

rainfall_events = build_rainfall_events(runoff_trees, df_rainfall, runoff_dates, flume_raingauges)

adj_sim, flume_labels = compute_sc_sim(df_coords, df_areas)

event_pairs = [("event_8", "event_9"),
    ("event_14", "event_119"),
    ("event_45", "event_201")]


output_dir = "results"; os.makedirs(output_dir, exist_ok=True)

for i, (e1, e2) in enumerate(event_pairs):

    if int(e1.split("_")[1]) > 178: e1 = f"event_{int(e1.split('_')[1]) + 1}"
    if int(e2.split("_")[1]) > 178: e2 = f"event_{int(e2.split('_')[1]) + 1}"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col_idx, event_label in enumerate([e1, e2]):

        ax_hydro = axes[0, col_idx]; ax_fc = axes[1, col_idx]; ax_rain = ax_hydro.twinx()

        runoff_ds = get_runoff_event(runoff_trees, event_label)

        if runoff_ds is None: continue

        plot_event(ax_hydro, ax_rain, event_label, rainfall_events, runoff_ds, color_map)

        ax_hydro.set_title(event_label, fontsize=13)
        runoff_max = max([float(cfs_to_m3(data).max().values) for data in runoff_ds.data_vars.values()])
        ax_hydro.set_ylim(0, runoff_max * 1.2)

        if event_label in rainfall_events:
            rain_max = max([float(data.max().values) for data in rainfall_events[event_label].data_vars.values()])
            ax_rain.set_ylim(rain_max * 1.4, 0)

        ax_hydro.set_xlabel("Time"); ax_hydro.set_ylabel("Runoff (m³/s)"); ax_rain.set_ylabel("Rainfall (mm/hr)", color="blue")

        df_event = runoff_ds.to_dataframe()#.reindex(columns=flume_labels)
        print(df_event.head())
        flume_labels = [f"flume_{i}" for i in flume_order]
        df_complete = pd.DataFrame(columns=flume_labels)
        common_columns = df_complete.columns.intersection(df_event.columns)
        print(df_complete)
        print(common_columns)
        # Map the common columns to the "patent" DataFrame
        df_complete[common_columns] = df_event[common_columns]
        print(df_complete)
        fc_sim = df_complete.corr().fillna(0).to_numpy()
        print(fc_sim)
        fc_seq = compute_fc_seq_for_event(df_event, flume_labels, df_edges_seq)
        
        im = ax_fc.matshow(fc_sim, vmin=-1, vmax=1, cmap="coolwarm_r")

        ax_fc.set_title(f"{event_label} - FC_seq", fontsize=12)

        ax_fc.set_xticks(np.arange(len(flume_labels))); ax_fc.set_yticks(np.arange(len(flume_labels)))
        ax_fc.set_xticklabels(flume_labels, rotation=90, fontsize=7); ax_fc.set_yticklabels(flume_labels, fontsize=7)

        ax_fc.xaxis.set_ticks_position("bottom"); ax_fc.invert_yaxis()

        divider = make_axes_locatable(ax_fc); cax = divider.append_axes("right", size="5%", pad=0.15)

        cbar = fig.colorbar(im, cax=cax); cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    plt.tight_layout()

    outpath = os.path.join(output_dir, f"event_pair_{i+1}.pdf")
    plt.show()
    plt.savefig(outpath, bbox_inches="tight"); plt.close(fig)

    print(f"Saved: {outpath}")


# output_dir = "results"
# os.makedirs(output_dir, exist_ok=True)

# for i, (e1, e2) in enumerate(event_pairs):

     
#     # Extract the numeric part of the string
#     if int(e1.split("_")[1]) > 178:
#         e1 = f"event_{int(e1.split("_")[1]) + 1}"
#     if int(e2.split("_")[1]) > 178:
#         e2 = f"event_{int(e2.split("_")[1]) + 1}"
        
#     # ============================================================
#     # figure: 1 row, 2 columns
#     # ============================================================
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

#     ax1, ax2 = axes

#     ax1_rain = ax1.twinx()
#     ax2_rain = ax2.twinx()

#     # ============================================================
#     # EVENT 1
#     # ============================================================
#     runoff1 = get_runoff_event(runoff_trees, e1)

#     plot_event(ax1,ax1_rain,e1, rainfall_events, runoff1,color_map)

#     ax1.set_title(e1, fontsize=13)

#     # ------------------------------------------------------------
#     # runoff ylim
#     # ------------------------------------------------------------
#     runoff1_max = 0

#     if runoff1 is not None:

#         runoff1_max = max([
#             float(cfs_to_m3(data).max().values)
#             for data in runoff1.data_vars.values()])

#     runoff1_padding = runoff1_max * 0.20

#     ax1.set_ylim(0, runoff1_max + runoff1_padding)

#     ax1.margins(y=0)

#     # ------------------------------------------------------------
#     # rainfall ylim
#     # ------------------------------------------------------------
#     if e1 in rainfall_events:

#         ds_rain1 = rainfall_events[e1]

#         rain1_max = max([
#             float(data.max().values)
#             for data in ds_rain1.data_vars.values()
#         ])

#         rain1_padding = rain1_max * 0.40

#         ax1_rain.set_ylim(
#             rain1_max + rain1_padding,
#             0
#         )

#         ax1_rain.margins(y=0)

#     # ============================================================
#     # EVENT 2
#     # ============================================================
#     runoff2 = get_runoff_event(runoff_trees, e2)

#     plot_event(ax2,ax2_rain,e2,rainfall_events, runoff2,color_map)

#     ax2.set_title(e2, fontsize=13)

#     # ------------------------------------------------------------
#     # runoff ylim
#     # ------------------------------------------------------------
#     runoff2_max = 0

#     if runoff2 is not None:

#         runoff2_max = max([
#             float(cfs_to_m3(data).max().values)
#             for data in runoff2.data_vars.values()
#         ])

#     runoff2_padding = runoff2_max * 0.20

#     ax2.set_ylim(0,runoff2_max + runoff2_padding)

#     ax2.margins(y=0)

#     # ------------------------------------------------------------
#     # rainfall ylim
#     # ------------------------------------------------------------
#     if e2 in rainfall_events:

#         ds_rain2 = rainfall_events[e2]

#         rain2_max = max([
#             float(data.max().values)
#             for data in ds_rain2.data_vars.values()])

#         rain2_padding = rain2_max * 0.40

#         ax2_rain.set_ylim(rain2_max + rain2_padding,0)

#         ax2_rain.margins(y=0)

#     # ============================================================
#     # formatting
#     # ============================================================
#     for ax in axes:

#         ax.set_xlabel("Time", fontsize=11)

#         ax.tick_params(axis="both",labelsize=10)

#         ax.set_ylabel(
#             "Runoff (m³/s)",
#             fontsize=11
#         )

#     ax1_rain.set_ylabel(
#         "Rainfall (mm/hr)",
#         color="blue",
#         fontsize=11
#     )

#     ax2_rain.set_ylabel(
#         "Rainfall (mm/hr)",
#         color="blue",
#         fontsize=11
#     )

#     # ============================================================
#     # layout
#     # ============================================================
#     plt.tight_layout(rect=[0, 0.24, 1, 1])

#     # ============================================================
#     # colorbar under subplot 1
#     # ============================================================
#     sm1 = plt.cm.ScalarMappable(
#         cmap=cmap,
#         norm=norm
#     )

#     sm1.set_array([])

#     bbox1 = ax1.get_position()

#     cax1 = fig.add_axes([
#         bbox1.x0 + 0.03,
#         bbox1.y0 - 0.14,
#         bbox1.width - 0.06,
#         0.025
#     ])

#     cb1 = fig.colorbar(
#         sm1,
#         cax=cax1,
#         orientation="horizontal"
#     )

#     cb1.set_label(
#         "Contributing area (km$^2$)",
#         fontsize=10
#     )

#     cb1.ax.tick_params(labelsize=9)

#     cb1.set_ticks([norm.vmin, norm.vmax])

#     cb1.set_ticklabels([
#         f"{norm.vmin:.2f}",
#         f"{norm.vmax:.2f}"
#     ])

#     # ============================================================
#     # colorbar under subplot 2
#     # ============================================================
#     sm2 = plt.cm.ScalarMappable(
#         cmap=cmap,
#         norm=norm
#     )

#     sm2.set_array([])

#     bbox2 = ax2.get_position()

#     cax2 = fig.add_axes([
#         bbox2.x0 + 0.03,
#         bbox2.y0 - 0.14,
#         bbox2.width - 0.06,
#         0.025
#     ])

#     cb2 = fig.colorbar(
#         sm2,
#         cax=cax2,
#         orientation="horizontal"
#     )

#     cb2.set_label(
#         "Contributing area (km$^2$)",
#         fontsize=10
#     )

#     cb2.ax.tick_params(labelsize=9)

#     cb2.set_ticks([norm.vmin, norm.vmax])

#     cb2.set_ticklabels([
#         f"{norm.vmin:.2f}",
#         f"{norm.vmax:.2f}"
#     ])

#     # ============================================================
#     # dates under each subplot
#     # ============================================================
#     if runoff1 is not None:

#         times1 = []

#         for data in runoff1.data_vars.values():
#             times1.extend(data.time.values)

#         if len(times1) > 0:

#             date1 = pd.to_datetime(
#                 max(times1)
#             ).strftime("%d-%b-%Y")

#             fig.text(
#                 bbox1.x1,
#                 bbox1.y0 - 0.075,
#                 date1,
#                 ha="right",
#                 va="top",
#                 fontsize=10,
#                 fontweight="bold"
#             )

#     if runoff2 is not None:

#         times2 = []

#         for data in runoff2.data_vars.values():
#             times2.extend(data.time.values)

#         if len(times2) > 0:

#             date2 = pd.to_datetime(
#                 max(times2)
#             ).strftime("%d-%b-%Y")

#             fig.text(
#                 bbox2.x1,
#                 bbox2.y0 - 0.075,
#                 date2,
#                 ha="right",
#                 va="top",
#                 fontsize=10,
#                 fontweight="bold"
#             )

#     # ============================================================
#     # save
#     # ============================================================
#     outpath = os.path.join(
#         output_dir,
#         f"event_pair_{i+1}.pdf"
#     )

#     plt.savefig(
#         outpath,
#         bbox_inches="tight"
#     )

#     plt.close(fig)

#     print(f"Saved: {outpath}")
