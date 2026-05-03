# make_ridge_plots.R
# -------------------------------
# Load libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggridges)
library(viridis)
library(scales)
library(readr)
library(yaml)

source("ridge_plots_lib.R")  # load functions

# -------------------------------
# Define relative paths
# -------------------------------
data_path <- file.path("data", "df_scfcs_all.csv")
                       #"df_scfcs_clusters_and_runoff.csv")
config_path <- "config.yaml"

# -------------------------------
# Read config.yaml
# -------------------------------
config <- read_yaml(config_path)
cluster_labels <- config$plotting$cluster_labels
flume_order <- paste0("flume_", config$plotting$flume_order)

# -------------------------------
# Read contributing area from YAML (optional)
# -------------------------------
if(!is.null(config$plotting$contributing_area)){
  ca_list <- config$plotting$contributing_area
  ca_df <- data.frame(
    flume_name = paste0("flume_", names(ca_list)),
    contributing_area = as.numeric(unlist(ca_list)),
    stringsAsFactors = FALSE
  )
  ca_df$ca_km2 <- ca_df$contributing_area / 1e6
} else {
  ca_df <- NULL
}

# -------------------------------
# Read dataset
# -------------------------------
data <- read_main_dataset(data_path)

# -------------------------------
# Filter clusters
# -------------------------------
data <- data %>%
  mutate(cluster = as.numeric(as.character(cluster))) %>%
  filter(cluster %in% as.numeric(names(cluster_labels))) %>%
  mutate(cluster = factor(cluster, levels = names(cluster_labels)))

# -------------------------------
# Pivot flume columns
# -------------------------------
long_filtered <- pivot_flumes(data, flume_prefix = "flume_")

# -------------------------------
# Merge contributing area
# -------------------------------
long_filtered <- merge_contributing_area(long_filtered, ca_df)

# -------------------------------
# Generate Q ridges
# -------------------------------
Q_plot <- plot_Q_ridges(long_filtered, cluster_labels, flume_order)
ggsave(filename = "results/corr_ridges.pdf", plot = Q_plot, width = 25, height = 20, units = "cm")

# -------------------------------
# Pivot SC/FC correlation
# -------------------------------
long_fc <- data %>%
  pivot_longer(
    cols = c(scfc_seq, scfc_sync),
    names_to = "corr",
    values_to = "value"
  )

# -------------------------------
# Generate correlation ridges
# -------------------------------
corr_plot <- plot_corr_ridges(long_fc, cluster_labels)
ggsave(filename = "results/Q_ridges.pdf", plot = corr_plot, width = 25, height = 5, units = "cm")
