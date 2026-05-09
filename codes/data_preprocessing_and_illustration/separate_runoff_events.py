
import xarray as xr
from runoff_functions import load_runoff_csv, process_all_runoff_files, split_runoff_events, process_event_to_dataset
from datatree import DataTree
from pathlib import Path

# -------------------------------
# Configurable paths
# -------------------------------

PROJECT_ROOT = Path(".")

DATA_DIR = PROJECT_ROOT / "data" / "runoff"
RESULTS_DIR = PROJECT_ROOT / "data" / "runoff"

# Runoff files (ordered chronologically)
RUNOFF_FILES = [
    DATA_DIR / "runoff_data_2000_2006.csv",
    DATA_DIR / "runoff_data_2007_2013.csv",
    DATA_DIR / "runoff_data_2014_2024.csv"
]

# ------------------ Main function -------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for runoff_file in RUNOFF_FILES:
        # Infer period name from filename
        period = runoff_file.stem.replace("runoff_data_", "")
        output_file = RESULTS_DIR / f"runoff_{period}.nc"

        # Load and split events for this period only
        df_runoff = load_runoff_csv(runoff_file)
        events = split_runoff_events(df_runoff)

        root = DataTree(name=f"runoff_{period}")

        for i, df_event in enumerate(events):
            ds_event = process_event_to_dataset(df_event)
            if ds_event is None:
                continue

            # Add minimal metadata
            ds_event.attrs.update({
                "event_index": i,
                "period": period,
            })

            # Integer event index in hierarchy
            root[f"event_{i}"] = DataTree(ds_event)

        # Write NetCDF
        root.to_netcdf(output_file)
        print(f"Saved {output_file}")

if __name__ == "__main__":
    main()
