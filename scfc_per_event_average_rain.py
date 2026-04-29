"""
SC/FC timeseries scatter plot across events, split into 3 periods.

- Plots SC/FC_sync and SC/FC_seq for each event as markers
- Bars for average rainfall on secondary y-axis
- Vertical lines highlight specific events
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ---------------------------------------------------------------------
# Project paths (portable)
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(".")

DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "df_scfcs_all.csv"
OUTPUT_FIG = FIG_DIR / "scfcs_scatter_plot_average_rain.pdf"

# ---------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------

#COL_EVENT_LABEL = "event label"
COL_SCFC_SYNC = "scfc_sim"   
COL_SCFC_SEQ = "scfc_seq"
COL_AVG_RAIN = "average_rainfall"

# ---------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------

def plot_scfc_timeseries(df):
    event_labels = df[COL_EVENT_LABEL].tolist()
    event_indices = np.arange(len(event_labels))
    #event_labels = event_indices
    
    # Split into 3 equal periods
    n_events = len(event_labels)
    thirds = [0, n_events // 3, 2 * n_events // 3, n_events]

    fig, axes = plt.subplots(3, 1, figsize=(28, 15), sharex=False)

    colors = ["lightblue", "lightcoral"]
    labels = ["SC/FC$_{sync}$", "SC/FC$_{seq}$"]

    # Define events to highlight with vertical lines
    #highlight_events = ["event_8", "event_9", "event_14", "event_17", "event_56", "event_112"]
    highlight_indices = [8, 9, 14, 17, 56, 112]
    #[event_indices[event_labels.index(ev)] for ev in highlight_events]

    for i in range(3):
        start, mid, end = thirds[i], thirds[i+1], thirds[i+2] if i < 2 else thirds[3]
        ax = axes[i]

        x = event_indices[start:end]
        scfc_sync = df[COL_SCFC_SYNC][start:end]
        scfc_seq = df[COL_SCFC_SEQ][start:end]
        rain = df[COL_AVG_RAIN][start:end]

        # Plot markers
        ax.plot(x, scfc_sync, color=colors[0], marker='s', markersize=10,
                markeredgecolor='black', linestyle='', label=labels[0])
        ax.plot(x, scfc_seq, color=colors[1], marker='s', markersize=10,
                markeredgecolor='black', linestyle='', label=labels[1])

        ax.set_ylabel("SC/FC", fontsize=25)
        ax.set_ylim(-0.5, 1.0)
        ax.tick_params(axis='y', labelsize=16)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)

        # Secondary y-axis for rainfall
        ax_rain = ax.twinx()
        ax_rain.bar(x, rain, color='blue', alpha=0.6, width=0.5)
        ax_rain.set_ylim(120, 0)
        ax_rain.set_ylabel("average rain (mm)", color='b', fontsize=20)
        ax_rain.tick_params(axis='y', labelcolor='b', labelsize=16)

        # Highlight vertical lines
        for idx in highlight_indices:
            if start <= idx < end:
                ax.axvline(idx, color='black', linestyle='--', linewidth=2.5)

        # Title for each period
        periods = ["2000 - 2007", "2008 - 2015", "2016 - 2024"]
        ax.set_title(periods[i], fontsize=22, pad=15)

        # Legend only on the middle subplot
        if i == 1:
            custom_lines = [Line2D([0], [0], color=c, marker='s', markersize=18,
                                   markeredgecolor='black', linestyle='') for c in colors]
            ax.legend(custom_lines, labels, loc='lower left', fontsize=20, ncol=2, edgecolor='black')

    axes[2].set_xlabel("event index", fontsize=25)

    plt.tight_layout(pad=3.0, w_pad=0.5)
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    plt.show()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    df_scfcs = pd.read_csv(INPUT_FILE, index_col=0)
    plot_scfc_timeseries(df_scfcs)

if __name__ == "__main__":
    main()




