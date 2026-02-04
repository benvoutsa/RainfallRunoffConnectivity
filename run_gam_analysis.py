"""
run_gam_analysis.py

GAM analysis of SC–FC correlations and discharge at flume 1.

- Fits additive and tensor GAMs
- Plots partial effects and 2D interaction contour
- Uses paths from scfc.config
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pygam import LinearGAM, s, te
from config import DATA_DIR, RESULTS_DIR

# --- Load SC–FC results ---
df_scfcs_file = DATA_DIR / "df_scfcs_all.csv"
df_scfcs = pd.read_csv(df_scfcs_file, index_col=0)

# Rename columns for clarity
df_scfcs.rename(columns={"1": "discharge_flume_1"}, inplace=True)

# --- Prepare predictor ranges for plotting ---
scfc_seq_range = np.linspace(df_scfcs['scfc_seq'].min(), df_scfcs['scfc_seq'].max(), 100)
scfc_sim_range = np.linspace(df_scfcs['scfc_sim'].min(), df_scfcs['scfc_sim'].max(), 100)
grid_seq, grid_sim = np.meshgrid(scfc_seq_range, scfc_sim_range)

# --- Fit additive GAM (raw discharge) ---
X_add = df_scfcs[['scfc_seq', 'scfc_sim']].values
y_add = df_scfcs['discharge_flume_1'].values
gam_add = LinearGAM(s(0) + s(1)).fit(X_add, y_add)

# --- Fit tensor GAM on log-transformed discharge (handle zeros) ---
df_log = df_scfcs.copy()
df_log['discharge_nozero'] = df_log['discharge_flume_1'].replace(0, 1e-3)
df_log['log_discharge'] = np.log(df_log['discharge_nozero'])
X_tensor = df_log[['scfc_seq', 'scfc_sim']].values
y_tensor = df_log['log_discharge'].values
gam_tensor = LinearGAM(te(0, 1)).fit(X_tensor, y_tensor)

# --- Create figure ---
fig, axs = plt.subplots(1, 3, figsize=(28, 8))

# --- (a) Partial effect on Q_flume1 for scfc_sim (term 1, additive) ---
X_sim = np.column_stack([np.repeat(df_scfcs['scfc_seq'].mean(), 100), scfc_sim_range])
pdep_sim, confi_sim = gam_add.partial_dependence(term=1, X=X_sim, width=0.95)
axs[0].plot(scfc_sim_range, pdep_sim, color='blue', label='Partial effect')
axs[0].fill_between(scfc_sim_range, confi_sim[:, 0], confi_sim[:, 1], alpha=0.3, color='blue')
axs[0].set_xlabel('SC/FC$_{sync}$', fontsize=20)
axs[0].set_ylabel('Partial effect on Q$_{flume1}$', fontsize=20)
axs[0].set_title('(a)', fontsize=24, loc='left')
axs[0].grid()
axs[0].set_ylim(-1.8e5, 1.8e5)
sf0 = ScalarFormatter(useMathText=True)
sf0.set_scientific(True)
sf0.set_powerlimits((-1, 1))
axs[0].yaxis.set_major_formatter(sf0)
axs[0].yaxis.get_offset_text().set_fontsize(18)

# --- (b) Partial effect on Q_flume1 for scfc_seq (term 0, additive) ---
X_seq = np.column_stack([scfc_seq_range, np.repeat(df_scfcs['scfc_sim'].mean(), 100)])
pdep_seq, confi_seq = gam_add.partial_dependence(term=0, X=X_seq, width=0.95)
axs[1].plot(scfc_seq_range, pdep_seq, color='red', label='Partial effect')
axs[1].fill_between(scfc_seq_range, confi_seq[:, 0], confi_seq[:, 1], alpha=0.3, color='red')
axs[1].set_xlabel('SC/FC$_{seq}$', fontsize=20)
axs[1].set_title('(b)', fontsize=24, loc='left')
axs[1].grid()
axs[1].set_ylim(-1.8e5, 1.8e5)
sf1 = ScalarFormatter(useMathText=True)
sf1.set_scientific(True)
sf1.set_powerlimits((-1, 1))
axs[1].yaxis.set_major_formatter(sf1)
axs[1].yaxis.get_offset_text().set_fontsize(18)

# --- (c) Tensor GAM 2D interaction contour on log(Q_flume1) ---
X_grid = np.column_stack([grid_seq.ravel(), grid_sim.ravel()])
pdep_tensor, _ = gam_tensor.partial_dependence(term=0, X=X_grid, width=0.95)
partial_effect_tensor = pdep_tensor.reshape(grid_seq.shape)

cf = axs[2].contourf(grid_seq, grid_sim, partial_effect_tensor, levels=20, cmap='RdBu_r', alpha=0.8)
axs[2].contour(grid_seq, grid_sim, partial_effect_tensor, levels=10, colors='black', linewidths=0.5)
axs[2].scatter(df_scfcs['scfc_seq'], df_scfcs['scfc_sim'], c='gray', s=55, edgecolor='black', label='Data points')
axs[2].set_title('(c)', fontsize=24, loc='left', pad=20)
axs[2].set_xlabel('SC/FC$_{seq}$', fontsize=20)
axs[2].set_ylabel('SC/FC$_{sync}$', fontsize=20)

# Colorbar
cbar = fig.colorbar(cf, ax=axs[2], shrink=0.85)
cbar.set_label('Partial effect on log(Q$_{flume1}$)', fontsize=18)
cbar.ax.tick_params(labelsize=14)

# General formatting
for ax in axs:
    ax.tick_params(axis='both', labelsize=18)

plt.tight_layout()

# --- Save figure ---
output_file = RESULTS_DIR / "GAM_plots.pdf"
plt.savefig(output_file, format='pdf', bbox_inches='tight')
plt.show()

