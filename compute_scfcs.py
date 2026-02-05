"""
SC–FC analysis
"""

import pandas as pd
import numpy as np

from sc_fc_functions import *
#from sc_fc_functions import load_flume_coordinates, load_contributing_areas, load_edge_list, load_sc_seq, load_runoff_events
#from sc_fc_functions import compute_sc_sim, compute_fc_seq_for_event, remove_diagonal


PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "scfc_results.csv"

def main(output_file: Path = OUTPUT_FILE):
    """
    Run SC–FC analysis and save results to CSV.
    """
    print("Running SC–FC analysis...")
    df_results = run_scfc_analysis()
    df_results.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to '{output_file}'.")


if __name__ == "__main__":
    main()
