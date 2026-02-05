# ============================================================
# Ridge plots – shared library
# ============================================================

# ---------------- Libraries & global setup ---------------- #
library(ggplot2)
library(ggridges)
library(tidyr)
library(dplyr)
library(viridis)
library(scales)
library(yaml)

theme_set(theme_minimal())

# ---------------- Load configuration ---------------- #

config <- yaml::read_yaml("config.yaml")

CLUSTER_LABELS <- unlist(config$clusters$labels)
FLUME_ORDER <- config$plotting$flume_order

# ---------------- Utility functions ---------------- #

# Fancy scientific labels for log-scale axes
fancy_scientific <- function(l) {
  l <- format(l, scientific = TRUE)
  l <- gsub("^(.*)e", "'\\1'e", l)
  l <- gsub("e", "%*%10^", l)
  parse(text = l)
}

# Identify flume columns (numeric column names)
get_flume_columns <- function(df, max_flume = 105) {
  flume_cols <- as.character(1:max_flume)
  flume_cols[flume_cols %in% colnames(df)]
}

# ---------------- I/O helpers ---------------- #

read_main_dataset <- function(path) {
  read.table(
    path,
    sep = ",",
    header = TRUE,
    stringsAsFactors = FALSE
  )
}

# ---------------- Ridge plot functions ---------------- #

plot_Q_ridges <- function(
  df,
  flume_order = FLUME_ORDER,
  cluster_labels = CLUSTER_LABELS
) {

  stopifnot("cluster" %in% colnames(df))

  flume_cols <- get_flume_columns(df)

  long_df <- df %>%
    pivot_longer(
      cols = all_of(flume_cols),
      names_to = "flume",
      values_to = "value"
    ) %>%
    mutate(
      flume_num = as.numeric(flume),
      flume_name_f = factor(flume, levels = flume_order),
      value_m3 = value
    ) %>%
    filter(value_m3 != 0)

  ggplot(
    long_df,
    aes(
      x = value_m3,
      y = flume_name_f,
      group = flume_name_f,
      fill = flume_num
    )
  ) +
    geom_density_ridges(
      quantile_lines = TRUE,
      quantiles = 2,
      jittered_points = TRUE
    ) +
    facet_wrap(
      ~cluster,
      nrow = 1,
      labeller = labeller(cluster = cluster_labels)
    ) +
    scale_fill_viridis(
      option = "A",
      trans = pseudo_log_trans(sigma = 0.001)
    ) +
    scale_x_log10(
      oob = squish_infinite,
      labels = fancy_scientific
    ) +
    labs(
      x = "Accumulated Q",
      y = ""
    )
}

plot_correlation_ridges <- function(
  df,
  cluster_labels = CLUSTER_LABELS
) {

  required_cols <- c("corr_seq", "corr_sim", "cluster")
  if (!all(required_cols %in% colnames(df))) {
    message("Correlation columns not found — skipping correlation plot.")
    return(NULL)
  }

  long_corr <- df %>%
    pivot_longer(
      cols = c("corr_seq", "corr_sim"),
      names_to = "corr",
      values_to = "value"
    )

  ggplot(long_corr, aes(x = abs(value), y = corr, group = corr)) +
    geom_density_ridges(
      fill = "red",
      panel_scaling = TRUE,
      quantile_lines = TRUE,
      quantiles = 2
    ) +
    facet_wrap(
      ~cluster,
      nrow = 1,
      labeller = labeller(cluster = cluster_labels)
    ) +
    labs(
      x = "Correlation",
      y = ""
    )
}
