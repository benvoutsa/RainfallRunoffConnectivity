"""
SC–FC analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sc_fc_functions import *

#from sc_fc_functions import load_flume_coordinates, load_contributing_areas, load_edge_list, load_sc_seq, load_runoff_events
#from sc_fc_functions import compute_sc_sim, load_sc_seq

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

flume_order = config["flume_order"]

PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


RUNOFF_FILES = [
    DATA_DIR / "runoff_events_2000_2006.nc",
    DATA_DIR / "runoff_events_2007_2013.nc",
    DATA_DIR / "runoff_events_2014_2024.nc"
]

df_flume_coordinates = load_flume_coordinates()
df_contributing_area = load_contributing_areas()
SC_SIM, flume_labels = compute_sc_sim(df_flume_coordinates, df_contributing_area)
SC_SEQ = load_sc_seq()

OUTPUT_FILE = RESULTS_DIR / "scfc_results_updated.csv"
OUTPUT_FIGURE = RESULTS_DIR / "sc_matrices.pdf"

def main():

    fig, axes = plot_sc_matrices(SC_SIM, SC_SEQ, labels=flume_labels, flume_order=flume_order, figsize=(6, 10))
    plt.savefig(OUTPUT_FIGURE, bbox_inches="tight", dpi=300)
    
    all_results = []

    for runoff_file in RUNOFF_FILES:
        print(f"Processing {runoff_file} ...")
        df_results = run_scfc_analysis(runoff_file)
        all_results.append(df_results)

    # Combine all results into one DataFrame
    df_all = pd.concat(all_results, ignore_index=True)
    #df_all.to_csv(OUTPUT_FILE, index=False)
    print(f"All results saved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
