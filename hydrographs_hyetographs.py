import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
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

flume_order = config["flume_order"]

print(flume_order)

# ------------------------------------------------------------
# paths
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


def cfs_to_m3(cfs):
    return cfs * 0.0283168


def get_runoff_event(tree_list, event_label):
    for tree in tree_list:
        if event_label in tree:
            return tree[event_label].to_dataset()
    return None


# ------------------------------------------------------------
# color map (uses contributing area)
# ------------------------------------------------------------
def build_color_map(flume_master_df):

    df = flume_master_df[["Flume", "Contributing_area_km2"]].drop_duplicates()
    df["Contributing_area_km2"] = pd.to_numeric(df["Contributing_area_km2"], errors="coerce")

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

    flume_raingauges = pd.read_csv(FLUME_RAINGAUGES_PATH, sep=r"\s+|\t+|,", engine="python")

    flume_watersheds = pd.read_csv(FLUME_WATERSHEDS_PATH, sep=r"\s+|\t+|,", engine="python")

    flume_raingauges.columns = flume_raingauges.columns.str.strip()
    flume_watersheds.columns = flume_watersheds.columns.str.strip()

    flume_master = flume_raingauges.merge(flume_watersheds, on="Flume", how="left")

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
    
        rain_max = max(float(data.max().values) for data in ds_rain.data_vars.values())
    
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

        ax.plot(runoff.time, runoff.values, color=color_map.get(flume_id, "black"), linewidth=1)

    ax.set_ylabel("Runoff (m³/s)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


# ------------------------------------------------------------
# main script
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

    # ------------------------------------------------------------
    # fix event numbering
    # ------------------------------------------------------------
    if int(e1.split("_")[1]) > 178:
        e1 = f"event_{int(e1.split('_')[1]) + 1}"
    if int(e2.split("_")[1]) > 178:
        e2 = f"event_{int(e2.split('_')[1]) + 1}"

    # ------------------------------------------------------------
    # figure + grid
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(2, 4, figure=fig)

    event_list = [e1, e2]

    for col_idx, event_label in enumerate(event_list):

        # ------------------------------------------------------------
        # hydrographs - hyetographs 
        # ------------------------------------------------------------
        if col_idx == 0:
            ax_hydro = fig.add_subplot(gs[0, 0:2])
        else:
            ax_hydro = fig.add_subplot(gs[0, 2:4])

        ax_rain = ax_hydro.twinx()

        runoff_ds = get_runoff_event(runoff_trees, event_label)

        if runoff_ds is None:
            continue

        plot_event(ax_hydro, ax_rain, event_label, rainfall_events, runoff_ds, color_map)

        ax_hydro.set_title(event_label, fontsize=13)
        ax_hydro.set_xlabel("Time")
        ax_hydro.set_ylabel("Runoff (m³/s)")
        ax_rain.set_ylabel("Rainfall (mm/hr)", color="blue")

        runoff_max = max([float(cfs_to_m3(data).max().values) for data in runoff_ds.data_vars.values()])
        ax_hydro.set_ylim(0, runoff_max * 1.5)

        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        
        # cbar = fig.colorbar(sm, ax=ax_hydro,orientation="horizontal", fraction=0.05, pad=0.12)
        # bbox = ax_hydro.get_position()
        # cax = fig.add_axes([ bbox.x0 + 0.03, bbox.y0 - 0.14, bbox.width - 0.06, 0.025 ])
        # cbar.set_label("Contributing Area (km²)")
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        bbox = ax_hydro.get_position()
        cax = fig.add_axes([bbox.x0 + 0.03, bbox.y0 - 0.04, bbox.width - 0.06, 0.025])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("Contributing Area (km²)")
        full_date = pd.to_datetime(runoff_ds.time.values[-1]).strftime("%d-%b-%Y")

        # write the date under hydrograph
        fig.text(bbox.x1 + 0.01, bbox.y0 - 0.001, full_date, ha="right", va="top", fontsize=11, color="black", fontweight="bold")

        if event_label in rainfall_events:
            rain_max = max([float(data.max().values)
                for data in rainfall_events[event_label].data_vars.values()])
            ax_rain.set_ylim(rain_max * 1.5, 0)

        # ------------------------------------------------------------
        # FC matrices
        # ------------------------------------------------------------
        ax_fc_sim = fig.add_subplot(gs[1, col_idx * 2])
        ax_fc_seq = fig.add_subplot(gs[1, col_idx * 2 + 1])

        # ------------------------------------------------------------
        # compute FC matrices
        # ------------------------------------------------------------
        df_event = runoff_ds.to_dataframe().reindex(columns=flume_labels)

        fc_sim = df_event.corr().fillna(0).to_numpy().copy()
        fc_seq = compute_fc_seq_for_event(df_event, flume_labels, df_edges_seq).copy()

        np.fill_diagonal(fc_sim, 0)
        np.fill_diagonal(fc_seq, 0)

        im1 = ax_fc_sim.matshow(fc_sim, vmin=-1, vmax=1, cmap="coolwarm_r")
        ax_fc_sim.set_title("FC$_{sync}$", fontsize=10, fontweight = "bold", loc = "left")

        im2 = ax_fc_seq.matshow(fc_seq,  vmin=-1, vmax=1, cmap="coolwarm_r")
        ax_fc_seq.set_title("FC$_{seq}$", fontsize=10, fontweight = "bold", loc = "left")

        for ax in [ax_fc_sim, ax_fc_seq]:
            ax.set_xticks(np.arange(len(flume_labels)))
            ax.set_yticks(np.arange(len(flume_labels)))
            ax.set_xticklabels(flume_order[::-1], rotation=90, fontsize=6)
            ax.set_yticklabels(flume_order, fontsize=6)
            ax.xaxis.set_ticks_position("bottom")

    # colorbars
        div1 = make_axes_locatable(ax_fc_sim)
        cax1 = div1.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax1)

        div2 = make_axes_locatable(ax_fc_seq)
        cax2 = div2.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax2)

    plt.tight_layout()

    outpath = os.path.join(output_dir, f"hydrographs_and_fc_eventpair_{i+1}.pdf")
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {outpath}")
