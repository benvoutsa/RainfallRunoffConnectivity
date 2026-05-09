import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# -----------------------------------------------------------------------------
# Rainfall - Runoff Scatter Plots
# -----------------------------------------------------------------------------
# This script:
#   1. Loads rainfall/runoff data
#   2. Fits a power-law relationship
#   3. Creates a combined publication-style figure:
#        (a) SC/FC_sync
#        (b) SC/FC_seq
#        (c) Difference plot
#   4. Saves the figure in /results
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Paths and settings
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "df_scfcs_all.csv"
OUTPUT_DIR = PROJECT_ROOT / "results"
FIGURE_NAME = PROJECT_ROOT / "rainfall_runoff_scatterplots.pdf"

# Create results directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df_scfcs = pd.read_csv(DATA_FILE, index_col=0)

# -----------------------------------------------------------------------------
# Prepare data for power-law fit
# -----------------------------------------------------------------------------
x = df_scfcs["average_rainfall"].values
y = df_scfcs["1"].values

# Keep only positive values for log-log fitting
mask = (x > 0) & (y > 0)
x_pos = x[mask]
y_pos = y[mask]

# Log-transform
log_x = np.log10(x_pos)
log_y = np.log10(y_pos)

# Power-law fit: y = a * x^b
b, log_a = np.polyfit(log_x, log_y, 1)
a = 10**log_a

print(f"Fitted parameters: a = {a:.3f}, b = {b:.3f}")

# Smooth fitted curve
x_fit = np.linspace(x_pos.min(), x_pos.max(), 300)
y_fit = a * x_fit**b

# -----------------------------------------------------------------------------
# Additional variables
# -----------------------------------------------------------------------------
df_scfcs["scfc_diff"] = df_scfcs["scfc_seq"] - df_scfcs["scfc_sync"]

# -----------------------------------------------------------------------------
# Figure layout
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))

gs = fig.add_gridspec(
    nrows=2, ncols=2,
    height_ratios=[1, 1.1],
    hspace=0.32,
    wspace=0.18
)

# Top panels
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)

# Bottom panel
ax3 = fig.add_axes([0.15, 0.15, 0.60, 0.30])

# -----------------------------------------------------------------------------
# Common plotting settings
# -----------------------------------------------------------------------------
scatter_kwargs = dict(s=45, edgecolor="black", linewidth=0.5)

xticks = [1, 10, 100]
xtick_labels = [r"$10^{%d}$" % int(np.log10(val)) for val in xticks]

# -----------------------------------------------------------------------------
# (a) SC/FC_sync
# -----------------------------------------------------------------------------
sc1 = ax1.scatter(df_scfcs["average_rainfall"], df_scfcs["1"],
    c=df_scfcs["scfc_sync"], cmap="magma", **scatter_kwargs)

ax1.plot(x_fit, y_fit, color="blue", linestyle="--",
    linewidth=2, label=fr"$Q = {a:.2f} \cdot rain^{{{b:.2f}}}$")

ax1.set_xscale("log")
ax1.set_yscale("log")

ax1.set_xlabel("average rainfall (mm)", fontsize=15)
ax1.set_ylabel(r"Q$_{flume\ 1}$ (m$^3$)", fontsize=15)

ax1.legend(fontsize=12)

cbar1 = fig.colorbar(sc1, ax=ax1)
cbar1.set_label(r"SC/FC$_{sync}$", fontsize=13)
cbar1.ax.tick_params(labelsize=11)

ax1.set_title("(a)", loc="left", fontsize=16, fontweight="normal")

# -----------------------------------------------------------------------------
# (b) SC/FC_seq
# -----------------------------------------------------------------------------
sc2 = ax2.scatter(df_scfcs["average_rainfall"], df_scfcs["1"],
    c=df_scfcs["scfc_seq"], cmap="magma", **scatter_kwargs)

ax2.plot(x_fit, y_fit,
    color="blue",linestyle="--",
    linewidth=2, label=fr"$Q = {a:.2f} \cdot rain^{{{b:.2f}}}$")

ax2.set_xscale("log")
ax2.set_yscale("log")

ax2.set_xlabel("average rainfall (mm)", fontsize=15)
ax2.set_ylabel(r"Q$_{flume\ 1}$ (m$^3$)", fontsize=15)

ax2.legend(fontsize=12)

cbar2 = fig.colorbar(sc2, ax=ax2)
cbar2.set_label(r"SC/FC$_{seq}$", fontsize=13)
cbar2.ax.tick_params(labelsize=11)

ax2.set_title("(b)", loc="left", fontsize=16, fontweight="normal")

# -----------------------------------------------------------------------------
# Formatting top panels
# -----------------------------------------------------------------------------
for ax in [ax1, ax2]:

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    ax.tick_params(axis="both", labelsize=13)

# -----------------------------------------------------------------------------
# (c) Difference plot
# -----------------------------------------------------------------------------
norm = Normalize(vmin=df_scfcs["1"].min(), vmax=df_scfcs["1"].max())
cmap = cm.plasma

sc3 = ax3.scatter(df_scfcs["average_rainfall"], df_scfcs["scfc_diff"],
    c=df_scfcs["1"], s=df_scfcs["1"] * 0.0008, cmap=cmap,
    edgecolor="black", linewidth=0.5)

ax3.axhline(0, color="magenta", linestyle="--", linewidth=2)
ax3.set_xlabel("average rainfall (mm)", fontsize=15)
ax3.set_ylabel(r"(SC-FC)$_{seq}$ - (SC-FC)$_{sync}$", fontsize=15)
ax3.set_ylim(-0.6, 0.5)
ax3.tick_params(axis="both", labelsize=13)

ax3.set_title("(c)", loc="left", fontsize=16, fontweight="normal")

# -----------------------------------------------------------------------------
# Bubble-size legend
# -----------------------------------------------------------------------------
legend_values = [100000, 200000, 400000, 600000, 800000]

legend_handles = [plt.scatter([], [], s=val * 0.0008, color=cmap(norm(val)), alpha=0.7, edgecolors="black", label=f"{int(val):,}") for val in legend_values]

ax3.legend(
    handles=legend_handles,
    title=r"Q$_{flume\ 1}$ (m$^3$)",
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    fontsize=11,
    title_fontsize=12,
    labelspacing=2.3,
    borderaxespad=0.5,
    borderpad=1.2,
    frameon=True,
    framealpha=1,
    edgecolor="black",
    markerscale=0.99,
    handletextpad=0.65,
    ncol=1
)

plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.savefig(os.path.join(OUTPUT_DIR, FIGURE_NAME),
    dpi=300,bbox_inches="tight")
plt.show()
