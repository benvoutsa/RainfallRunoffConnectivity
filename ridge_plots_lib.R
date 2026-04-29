# ridge_plots_lib.R
# -------------------------------
# Load required libraries (install if missing)
# -------------------------------
packages <- c("dplyr","tidyr","ggplot2","ggridges","viridis","scales","readr")
invisible(lapply(packages, function(pkg){
  if(!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}))

# -------------------------------
# Read dataset safely
# -------------------------------
read_main_dataset <- function(path) {
  df <- readr::read_csv(path, col_types = cols())
  colnames(df) <- make.names(colnames(df))  # safe R names
  return(df)
}

# -------------------------------
# Pivot flume columns (X-prefixed numeric columns)
# -------------------------------
pivot_flumes <- function(df, flume_prefix = "X") {
  flume_cols <- grep(paste0("^", flume_prefix, "\\d+"), colnames(df), value = TRUE)
  
  df_long <- df %>%
    pivot_longer(
      cols = all_of(flume_cols),
      names_to = "flume_name",
      values_to = "value"
    ) %>%
    filter(value != 0) %>%
    mutate(
      flume_num = as.numeric(sub(paste0("^", flume_prefix), "", flume_name))
    )
  
  return(df_long)
}

# -------------------------------
# Merge contributing area if provided
# -------------------------------
merge_contributing_area <- function(df_long, ca_df = NULL) {
  if (!is.null(ca_df)) {
    df_long <- merge(df_long, ca_df, by = "flume_name")
    if(!"ca_km2" %in% colnames(df_long)) {
      df_long$ca_km2 <- df_long$contributing_area / 1e6
    }
  }
  return(df_long)
}

# -------------------------------
# Fancy scientific labels for plotting
# -------------------------------
fancy_scientific <- function(l) {
  l <- format(l, scientific = TRUE)
  l <- gsub("^(.*)e", "'\\1'e", l)
  l <- gsub("e", "%*%10^", l)
  parse(text = l)
}

# -------------------------------
# Plot Q ridges
# -------------------------------
plot_Q_ridges <- function(df_long, cluster_labels, flume_order, ca_fill = TRUE) {
  
  df_long <- df_long %>%
    mutate(
      cluster = factor(cluster, levels = names(cluster_labels)),
      flume_name_f = factor(flume_name, levels = flume_order)
    )
  
  Q_plot <- ggplot(df_long, aes(
    x = value,
    y = flume_name_f,
    group = flume_name_f,
    fill = if(ca_fill & "ca_km2" %in% colnames(df_long)) ca_km2 else NA
  )) +
    geom_density_ridges(quantile_lines = TRUE, quantiles = 2) +
    facet_wrap(~cluster, nrow = 1, labeller = labeller(cluster = cluster_labels)) +
    scale_x_log10(oob = scales::squish_infinite, labels = fancy_scientific) +
    labs(y = "flume", x = expression("accumulated discharge (m"^3*")")) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5),
      legend.position = "right"
    )
  
  if(ca_fill & "ca_km2" %in% colnames(df_long)) {
    Q_plot <- Q_plot +
      scale_fill_viridis(option = "A", trans = pseudo_log_trans(sigma = 0.0001))
  }
  
  return(Q_plot)
}

# -------------------------------
# Plot SC/FC correlation ridges
# -------------------------------
plot_corr_ridges <- function(df_long, cluster_labels) {
  df_long <- df_long %>%
    mutate(cluster = factor(cluster, levels = names(cluster_labels)))
  
  corr_plot <- ggplot(df_long, aes(x = abs(value), y = corr, group = corr, fill = corr)) +
    geom_density_ridges(panel_scaling = TRUE, quantile_lines = TRUE, quantiles = 2) +
    facet_wrap(~cluster, nrow = 1, labeller = labeller(cluster = cluster_labels)) +
    scale_fill_manual(values = c("scfc_seq" = "red", "scfc_sim" = "royalblue")) +
    labs(y = "", x = "SC-FC correlation") +
    theme_minimal(base_size = 15)
  
  return(corr_plot)
}
