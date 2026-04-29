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
Z = linkage(data_scaled, method='ward')
clusters = fcluster(Z, MAX_D, criterion='distance')
df_scaled['cluster'] = clusters

# -------------------------------
# Dendrogram / Clustermap
# -------------------------------
xticklabels = ['number of gauges', 'duration', 'average intensity', 
               'max rolling intensity', 'days with no rain']

set_link_color_palette(CUSTOM_COLORS)
g = sns.clustermap(
    df_scaled[FEATURE_COLS],
    method='ward',
    cmap='coolwarm',
    figsize=(12, 12),
    cbar_pos=(1.4, .3, .02, .4),
    xticklabels=xticklabels,
    yticklabels=['Event ' + str(i+1) for i in range(len(df_scaled))],
    col_cluster=False
)

# Adjust dendrogram and labels
reordered = g.dendrogram_row.reordered_ind
yticks = ['Event ' + str(i+1) for i in reordered]
yticks_5 = ['' if i%5!=0 else yticks[i] for i in range(len(yticks))]
g.ax_heatmap.set_yticklabels(yticks_5, fontsize=11)
plt.savefig("results/dendogram.pdf")
plt.show()

# -------------------------------
# Silhouette Analysis
# -------------------------------
scores = []
for k in range(2, 11):
    cluster_labels = fcluster(linkage(data_scaled, method='ward'), k, criterion='maxclust')
    scores.append(silhouette_score(data_scaled, cluster_labels))

plt.figure(figsize=(8,5))
plt.plot(range(2, 11), scores, marker='o', linestyle='--')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Different Numbers of Clusters")
plt.show()

optimal_clusters = np.argmax(scores) + 2
print("Optimal number of clusters:", optimal_clusters)

# -------------------------------
#  Boxplots of Scaled Features by Cluster
# -------------------------------
colors = ['red', 'forestgreen', 'dodgerblue']
df_melted = df_scaled.melt(id_vars='cluster', 
                           value_vars=['numberofgauges','durationmin','averageintensity'],
                           var_name='feature', value_name='value')

plt.figure(figsize=(8,4))
sns.boxplot(data=df_melted, x='cluster', y='value', hue='feature', palette=colors)
plt.ylabel('Scaled Rainfall Features')
plt.xlabel('Cluster')
plt.legend(title='Feature')
plt.savefig("results/scfc_boxplots_in_clusters.pdf")
plt.show()

# -------------------------------
#  MANOVA Test
# -------------------------------
formula = 'Q("numberofgauges") + Q("durationmin") + Q("averageintensity") + ' \
          'Q("maxrollingintensity") + Q("dayswithoutrain") ~ C(cluster)'
manova = MANOVA.from_formula(formula, data=df_scaled)
print(manova.mv_test())

# -------------------------------
#  SC/FC Scatter Plots by Cluster
# -------------------------------
df_scfcs = pd.read_csv(SCFCS_FILE, index_col=0)

print(len(df_scaled), len(df_scfcs))
print(df_scaled.index.equals(df_scfcs.index))

# df2 contais 'cluster', 'scfc_sync', 'scfc_seq'
df2 = pd.DataFrame({
    'cluster': df_scaled['cluster'],
    'scfc_sync': df_scfcs["scfc_sync"],
    'scfc_seq': df_scfcs["scfc_seq"]
})

clusters = sorted(df2['cluster'].unique())
n_clusters = len(clusters)
cluster_map = dict(zip(clusters, CUSTOM_COLORS[:n_clusters]))

print(n_clusters)

fig, axes = plt.subplots(1, n_clusters, figsize=(14,2), sharex=True, sharey=True)
if n_clusters==1: axes=[axes]

all_vals = pd.concat([df2['scfc_sync'], df2['scfc_seq']])
min_val, max_val = all_vals.min()-0.05, all_vals.max()+0.05

# for i, cl in enumerate(clusters):
#     ax = axes[i]
#     data = df2[df2['cluster']==cl]
#     ax.scatter(data['scfc_sync'], data['scfc_seq'], color=cluster_map[cl], edgecolor='black', s=80)
#     ax.plot([min_val,max_val],[min_val,max_val],'--',color='black')
#     ax.set_xlim(min_val,max_val)
#     ax.set_ylim(min_val,max_val)
#     if i==0: ax.set_ylabel('SC/FC_seq')
#     ax.set_xlabel('SC/FC_sync')
#     ax.grid(True)

for i, cl in enumerate(clusters):
    ax = axes[i]
    data = df2[df2['cluster'] == cl]

    ax.scatter(data['scfc_sync'], data['scfc_seq'], color=CUSTOM_COLORS[i],
               alpha=0.7, edgecolors='black',s=80)

    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel(r'(SC-FC)$_{sync}$', fontsize=14)
    if i == 0:
        ax.set_ylabel(r'(SC-FC)$_{seq}$', fontsize=14)
    ax.grid(True)

# add colorbar
cmap = mpl.colors.ListedColormap(CUSTOM_COLORS[:n_clusters])
norm = mpl.colors.BoundaryNorm(range(len(clusters) + 1),cmap.N)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
# ---- Colorbar axis ----
cax = fig.add_axes([0.91, 0.30, 0.012, 0.62])

cbar = fig.colorbar(sm, cax=cax, ticks=[i + 0.5 for i in range(len(clusters))])
cbar.set_ticklabels([f"Cluster {c}" for c in clusters])
  
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("results/scfc_scatter_in_clusters.pdf")
plt.show()
