import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
from statsmodels.multivariate.manova import MANOVA
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

# -------------------------------
# Paths and settings
# -------------------------------
DATA_FILE = "data/rainfall/rainfall_features.csv"  # adjust path
SCFCS_FILE = "data/df_scfcs_all.csv"
OUTPUT_DIR = "results"
FEATURE_COLS = ['numberofgauges', 'durationmin', 'averageintensity', 
                'maxrollingintensity', 'dayswithoutrain']
CUSTOM_COLORS = ['crimson', 'green', 'blueviolet', 'goldenrod']
MAX_D = 2

# -------------------------------
# Load and Scale Data
# -------------------------------
df_raw = pd.read_csv(DATA_FILE, index_col=0)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_raw[FEATURE_COLS])
df_scaled = pd.DataFrame(data_scaled, columns=FEATURE_COLS, index=df_raw.index)
df_scaled.head()

# -------------------------------
# Hierarchical Clustering
# -------------------------------
Z = linkage(data_scaled, method='ward', metric="euclidean")
clusters = fcluster(Z, MAX_D, criterion='distance')
#print(len(clusters))
#print(len(df_scaled))
df_scaled['cluster'] = clusters
#print(df_scaled)
print(f"Number of clusters after cut (height {MAX_D}): {len(set(clusters))}")

# -------------------------------
# Dendrogram / Clustermap
# -------------------------------
xticklabels = ['number of gauges', 'duration', 'average intensity', 
               'max rolling intensity', 'days with no rain']

# Set custom dendrogram colors
set_link_color_palette(CUSTOM_COLORS)

# Create clustermap
g = sns.clustermap(
    df_scaled[FEATURE_COLS],
    row_linkage=Z,              # IMPORTANT: reuse same linkage
    col_cluster=False,
    cmap='coolwarm',
    figsize=(12, 12),
    cbar_pos=(1.4, .3, .02, .4),
    xticklabels=xticklabels,
    yticklabels=['Event ' + str(i + 1) for i in range(len(df_scaled))]
)

# -------------------------------
# Reordered labels
# -------------------------------
reordered = g.dendrogram_row.reordered_ind

yticks = [
    'Event ' + str(i + 1)
    for i in reordered
]

yticks_5 = [
    '' if i % 5 != 0 else yticks[i]
    for i in range(len(yticks))
]

g.ax_heatmap.set_yticklabels(
    yticks_5,
    fontsize=11,
    rotation=0
)

# -------------------------------
# Manual layout adjustments
# -------------------------------
g.ax_row_dendrogram.set_position([0.1, 0.1, 0.2, 0.8])

g.ax_heatmap.set_position([0.302, 0.1, 0.18, 0.8])

# -------------------------------
# Overlay dendrogram
# -------------------------------
ax_dendro = g.ax_row_dendrogram

dendrogram(
    Z,
    ax=ax_dendro,
    color_threshold=MAX_D,
    orientation='left',
    no_labels=True
)

ax_dendro.invert_yaxis()

# -------------------------------
# X labels formatting
# -------------------------------
g.ax_heatmap.set_xticklabels(
    xticklabels,
    fontsize=13,
    rotation=90
)

# Optional:
# remove y labels entirely
# g.ax_heatmap.set_yticks([])
# g.ax_heatmap.set_yticklabels([])

plt.tight_layout()

plt.savefig("results/dendrogram.pdf", dpi=300,bbox_inches='tight')

plt.show()

# -------------------------------
# Silhouette Analysis
# -------------------------------
silhouette_scores = []
range_clusters = range(2, 11)
for k in range_clusters:
    cluster_labels = fcluster(Z, k, criterion='maxclust')
    silhouette_scores.append(silhouette_score(data_scaled, cluster_labels))

plt.figure(figsize=(8,5))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Different Numbers of Clusters")
plt.show()

#optimal_clusters = range_clusters[np.argmax(silhouette_scores)]
#print(f"Optimal number of clusters based on silhouette score: {optimal_clusters}")

# -------------------------------
# Boxplots of Scaled Features by Cluster
# -------------------------------

BOXPLOT_FEATURES = ['numberofgauges', 'durationmin','averageintensity']
BOXPLOT_LABELS = ['number of gauges', 'duration', 'average intensity']

BOX_COLORS = ['red', 'forestgreen', 'dodgerblue']

# Melt dataframe for seaborn
df_melted = df_scaled.melt(id_vars='cluster', value_vars=BOXPLOT_FEATURES, var_name='feature', value_name='value')

# Replace internal names with pretty labels
label_map = dict(zip(BOXPLOT_FEATURES, BOXPLOT_LABELS))
df_melted['feature'] = df_melted['feature'].map(label_map)

# Create figure
fig, ax = plt.subplots(figsize=(8, 4))

sns.boxplot(data=df_melted, x='cluster', y='value', hue='feature', palette=BOX_COLORS, width=0.6, dodge=True, ax=ax)

# Improve box appearance
for patch in ax.patches:
    patch.set_edgecolor('black')
    patch.set_linewidth(1.2)

# Improve whiskers/lines
for line in ax.lines:
    line.set_color('black')
    line.set_linewidth(1.2)

# Labels
ax.set_ylabel('scaled rainfall features', fontsize=13)
ax.set_xlabel('rainfall cluster', fontsize=13, labelpad=28)

# Legend
ax.legend(title='', fontsize=11, ncol=3, loc='upper center',bbox_to_anchor=(0.5, -0.08))
plt.tight_layout()
plt.savefig("results/rainfall_boxplots_in_clusters.pdf", dpi=300, bbox_inches='tight')

plt.show()

# -------------------------------
#  MANOVA Test
# -------------------------------
formula = 'Q("numberofgauges") + Q("durationmin") + Q("averageintensity") + ' \
          'Q("maxrollingintensity") + Q("dayswithoutrain") ~ C(cluster)'
manova = MANOVA.from_formula(formula, data=df_scaled)
print(manova.mv_test())


# -------------------------------
# SC-FC Boxplots by Cluster
# -------------------------------
df_scfcs = pd.read_csv(SCFCS_FILE, index_col=0)
#print(df_scfcs)

# df2 contais 'cluster', 'scfc_sync', 'scfc_seq'
df2 = pd.DataFrame({
    'cluster': df_scaled['cluster'].values,
    'scfc_sync': df_scfcs["scfc_sync"].values,
    'scfc_seq': df_scfcs["scfc_seq"].values
})

df_scfc_melted =  df2.melt(id_vars=['cluster'], 
                    value_vars=['scfc_sync', 'scfc_seq'],
                    var_name='feature', value_name='value')
print(df_scfc_melted)
df_scfc_melted['feature'] = df_scfc_melted['feature'].astype(str).str.strip()
hue_order = ['scfc_sync', 'scfc_seq']
palette = {'scfc_sync': 'royalblue', 'scfc_seq': 'red'}
print(df_scfc_melted)
fig, ax = plt.subplots(figsize=(8, 4))

sns.boxplot(data=df_scfc_melted, x='cluster', y='value', hue='feature', hue_order = hue_order, palette=palette, width=0.5, dodge=True, ax=ax)

for patch in ax.patches:
    patch.set_edgecolor('black')
    patch.set_linewidth(1.2)

for line in ax.lines:
    line.set_color("black")
    line.set_linewidth(1.2)

ax.set_ylabel('SC-FC', fontsize=13)
ax.set_xlabel('rainfall cluster', labelpad=32, fontsize=13)

legend_labels = ['(SC-FC)$_{sync}$', '(SC-FC)$_{seq}$']

ax.legend(
    handles=[plt.Line2D([0], [0], color=palette[k], lw=6) for k in hue_order],
    labels=legend_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.07),
    fontsize=12,
    ncol=2
)

ax.set_ylim(-0.27, 0.8)

plt.tight_layout()
plt.savefig("results/scfc_boxplots_in_clusters.pdf")
plt.show()

# -------------------------------
#  SC-FC Scatter Plots by Cluster
# -------------------------------


n_clusters = len(set(clusters))
cluster_nums = np.sort(np.unique(clusters))
cluster_map = dict(zip(clusters, CUSTOM_COLORS[:n_clusters]))

print(n_clusters)

fig, axes = plt.subplots(1, n_clusters, figsize=(12,2), sharex=True, sharey=True)
cluster_colors = CUSTOM_COLORS
print(cluster_colors)

if n_clusters == 1:
    axes = [axes]
  
# Map cluster label → color
cluster_color_map = dict(zip(cluster_nums, cluster_colors))
print(cluster_color_map)
print(cluster_nums)
# -----------------------------
# Axis limits (shared scaling)
# -----------------------------
all_vals = pd.concat([df2['scfc_sync'], df2['scfc_seq']])
buffer = 0.05
min_val, max_val = all_vals.min() - buffer, all_vals.max() + buffer

# -----------------------------
# Scatter plots per cluster
# -----------------------------
for ax, cl in zip(axes, cluster_nums):
    print(ax, cl)
    print(cluster_color_map[cl])
    data = df2[df2['cluster'] == cl]

    ax.scatter(
        data['scfc_sync'],
        data['scfc_seq'],
        color=cluster_color_map[cl],
        edgecolor='black',
        alpha=0.9,
        s=100
    )

    # diagonal reference line
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color='black',
        linestyle='--',
        linewidth=2
    )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.grid(True)

    if cl == 1:
        ax.set_ylabel('(SC-FC)$_{seq}$', fontsize=14)

    ax.tick_params(labelsize=12)

# shared x-label
fig.supxlabel('(SC-FC)$_{sync}$', fontsize=14)

# -----------------------------
# Colorbar (categorical)
# -----------------------------
cmap = ListedColormap(cluster_colors[:n_clusters])


# centers each cluster at integer positions
bounds = np.arange(n_clusters + 1) - 0.5
norm = BoundaryNorm(bounds, cmap.N)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# space for colorbar
fig.subplots_adjust(right=0.85)

cbar_ax = fig.add_axes([0.92, 0.3, 0.012, 0.65])

cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

cbar.set_ticks(np.arange(n_clusters))
cbar.set_ticklabels([f'Cluster {c}' for c in cluster_nums])
cbar.ax.tick_params(labelsize=12)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("results/scfc_scatter_in_clusters.pdf")
plt.show()
