# ============================================================
# Generate ridge plots
# ============================================================

source("R/ridge_plots_lib.R")

DATA_PATH <- "data/df_scaled_rainfall_features_and_clusters_and_runoff.csv"
FIG_DIR <- "figures"

dir.create(FIG_DIR, showWarnings = FALSE)

# ---------------- Load data ---------------- #
data <- read_main_dataset(DATA_PATH)

# ---------------- Q ridges ---------------- #
Q_plot <- plot_Q_ridges(data)

ggsave(
  filename = file.path(FIG_DIR, "Q_ridges.pdf"),
  plot = Q_plot,
  width = 25,
  height = 20,
  units = "cm"
)

# ---------------- Correlation ridges ---------------- #
corr_plot <- plot_correlation_ridges(data)

if (!is.null(corr_plot)) {
  ggsave(
    filename = file.path(FIG_DIR, "corr_ridges.pdf"),
    plot = corr_plot,
    width = 25,
    height = 5,
    units = "cm"
  )
}
