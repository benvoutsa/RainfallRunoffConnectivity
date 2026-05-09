import numpy as np
import pandas as pd
import datatree as dtree
from itertools import permutations
from pathlib import Path
import matplotlib.pyplot as plt

import yaml

PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"

FLUME_COORDS_FILE = DATA_DIR / "flume_coordinates.csv"
FLUME_AREA_FILE = DATA_DIR / "Flume_watersheds.csv"
EDGE_LIST_FILE = DATA_DIR / "df_sc_seq_extended.csv"
SC_SEQ_FILE = DATA_DIR / "adj_seq.csv"

DISTANCE_SCALE_KM = 1000

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

flume_order = config["flume_order"]

# ------------------ load data ------------------------------------

def load_flume_coordinates() -> pd.DataFrame:
    """Load flume coordinate table."""
    return pd.read_csv(FLUME_COORDS_FILE)

def load_contributing_areas() -> pd.DataFrame:
    """Load flume contributing area table."""
    return pd.read_csv(FLUME_AREA_FILE, index_col=0)

def load_edge_list() -> pd.DataFrame:
    """Load sequential edge list."""
    return pd.read_csv(EDGE_LIST_FILE, index_col=0)

def load_sc_seq() -> pd.DataFrame:
    """Load sequential structural connectivity data."""
    return pd.read_csv(SC_SEQ_FILE, index_col=0)

def load_runoff_events(file_path):
    """Load runoff events as a DataTree object from a given file"""
    return dtree.open_datatree(file_path, format="NETCDF4")

# ------------------  SC matrix functions  -------------------------

def euclidean_distance(flume_pair, df_flume_coordinates):
    flume1 = df_flume_coordinates[df_flume_coordinates["Flume"] == flume_pair[0]]
    flume2 = df_flume_coordinates[df_flume_coordinates["Flume"] == flume_pair[1]]
    distance = np.sqrt((flume1['East'].values - flume2['East'].values)**2 +
                       (flume1['North'].values - flume2['North'].values)**2)
    return np.round(distance[0], 2).item()


def compute_sc_sim(df_flume_coordinates, df_contributing_area):
    """
    Compute the SC simulated adjacency matrix (weighted by distance and contributing area)
    """
    edges_eucl_distance = list(permutations(df_flume_coordinates["Flume"], 2))
    df_sc_sim = pd.DataFrame(edges_eucl_distance, columns=['flume_1', 'flume_2'])

    # Distance weights
    distances = [euclidean_distance(pair, df_flume_coordinates) for pair in edges_eucl_distance]
    df_sc_sim['weight_dist'] = [1 / np.round(dist / DISTANCE_SCALE_KM, 2).item() for dist in distances]

    # Contributing area weights
    # contr_area_weights = []
    # for i, row in df_sc_sim.iterrows():
    #     area1 = df_contributing_area[df_contributing_area["Flume"] == row['flume_1']]["Contributing_area_km2"].iloc[0]
    #     area2 = df_contributing_area[df_contributing_area["Flume"] == row['flume_2']]["Contributing_area_km2"].iloc[0]
    #     contr_area_weights.append(min(area1, area2) / max(area1, area2))
    # df_sc_sim['weight_contr_area'] = contr_area_weights

    # Combined weight (the final results include only the inverse Euclidean distance)
    df_sc_sim['weight'] = df_sc_sim['weight_dist'] #+ \
                          #0.5 * df_sc_sim['weight_contr_area']

    flume_order_rev = flume_order[::-1]

    # Create adjacency matrix
    flume_labels = [f"flume_{i}" for i in flume_order_rev] 
    flume_indices = {flume: idx for idx, flume in enumerate(flume_labels)}
    adj_matrix = np.zeros((len(flume_labels), len(flume_labels)))

    for _, row in df_sc_sim.iterrows():
        i = flume_indices[f"flume_{int(row['flume_1'])}"]
        j = flume_indices[f"flume_{int(row['flume_2'])}"]
        adj_matrix[i, j] = row['weight']

    adj_matrix = pd.DataFrame(adj_matrix, index=flume_labels, columns=flume_labels)
    
    return adj_matrix, flume_labels


# --------------------------- FC matrix function ----------------------

def compute_fc_seq_for_event(df_complete, flume_labels, df_edges_seq):
    """
    Compute sequential functional connectivity (FC) for a single event
    """
    T = len(df_complete)
    n_flumes = len(flume_labels)
    fc_seq = np.zeros((n_flumes, n_flumes))
    print(df_complete)
    for idx in range(len(df_edges_seq)):
        flume1 = df_edges_seq['flume 1'].iloc[idx]
        flume2 = df_edges_seq['flume 2'].iloc[idx]
        #print(flume1, flume2)
        flume1 = f"flume_{flume1}"; flume2 = f"flume_{flume2}"
        if flume1 in flume_labels and flume2 in flume_labels:
            i1 = flume_labels.index(flume1)
            i2 = flume_labels.index(flume2)
            print(i1, i2)
              
            if not (np.isnan(df_complete[flume1].iloc[0]) or np.isnan(df_complete[flume2].iloc[0])):
                t_delay = int(round(df_edges_seq['time-delay mins'].iloc[idx], 0))
                if T > t_delay:
                    runoff_flume1 = df_complete[flume1].iloc[0:(T - t_delay)]
                    runoff_flume2 = df_complete[flume2].iloc[t_delay:T]
                    corr_val = np.corrcoef(runoff_flume1, runoff_flume2)[0, 1]
                    fc_seq[i1, i2] = np.round(corr_val, 2)
                    print(corr_val, np.round(corr_val, 2))
                    print("\n")

    return np.nan_to_num(fc_seq)


# ------------------ plot SC matrices ------------------------------


def plot_sc_matrices(sc_sim, sc_seq, labels=None, flume_order=None, figsize=(6, 10)):

    """
    Simple SC matrix plotting (2 rows × 1 column) in grayscale.
    
    """
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    cmap = "Greys"

    matrices = [sc_sim, sc_seq]
    titles = ["SC$_{sync}$", "SC$_{seq}$"]

    for ax, mat, title in zip(axes, matrices, titles):

        im = ax.matshow(mat, cmap=cmap, vmin=0, vmax=1)

        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)

        n = mat.shape[0]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))

        # labels
        if flume_order is not None:
            ax.set_xticklabels(flume_order[::-1], rotation=90, fontsize=7)
            ax.set_yticklabels(flume_order, fontsize=7)
        elif labels is not None:
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)
        else:
            ax.set_xticklabels(np.arange(n), rotation=90, fontsize=7)
            ax.set_yticklabels(np.arange(n), fontsize=7)

        ax.xaxis.set_ticks_position("bottom")

        # colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.show()
    return fig, axes


# ------------------ SC-FC helper functions -------------------------

def remove_diagonal(A):
    """Remove diagonal from square matrix for correlation calculations"""
    return A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)


def scfc_correlation(adj_matrix, fc_matrix):
    """Compute SC–FC correlation"""
    adj_flat = remove_diagonal(np.array(adj_matrix)).flatten()
    fc_flat = remove_diagonal(np.array(fc_matrix)).flatten()
    return np.corrcoef(adj_flat, fc_flat)[0, 1]

def run_scfc_analysis(runoff_file: Path):
    """
    Run SC–FC analysis for all events in a single NetCDF file
    """
    df_coords = load_flume_coordinates()
    df_areas = load_contributing_areas()
    df_edges_seq = load_edge_list()
    df_sc_seq = load_sc_seq()
    runoff_tree = load_runoff_events(runoff_file)

    # Structural adjacency
    adj_sim, flume_labels = compute_sc_sim(df_coords, df_areas)
    
    scfc_results = {"event": [], "scfc_sync": [], "scfc_seq": []}
    for node in runoff_tree.descendants:
        xr_event = runoff_tree[node.name]
        ds_event = xr_event.to_dataset()

        df_event = ds_event.to_dataframe().reindex(columns=flume_labels)
        
        # Functional connectivity
        fc_sim = df_event.corr().reindex(index=flume_labels, columns=flume_labels).fillna(0)
        fc_sim = fc_sim.to_numpy()
        fc_seq = compute_fc_seq_for_event(df_event, flume_labels, df_edges_seq)
        
        # Correlations
        scfc_sim = scfc_correlation(adj_sim, fc_sim)
        scfc_seq_val = scfc_correlation(df_sc_seq.values, fc_seq)
        scfc_seq_val = scfc_seq_val*(len(xr_event.data_vars)/len(flume_labels))

        scfc_results["event"].append(node.name)
        scfc_results["scfc_sync"].append(scfc_sim)
        scfc_results["scfc_seq"].append(scfc_seq_val)

    return pd.DataFrame(scfc_results).fillna(0)
