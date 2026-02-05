"""
runoff helper functions
"""
import numpy as np
import pandas as pd
import datatree as dtree
from datetime import timedelta

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
MIN_RUNOFF_CFS = config["min_runoff_cfs"]
DOWNSAMPLE_TIMESTEP = config["downsample_timestep"]
MERGE_THRESHOLD_MIN = config["merge_threshold_min"]

# ------------------ Load runoff unprocessed data -------------------------

def load_runoff_csv(file_path: str) -> pd.DataFrame:
    """
    Load a runoff CSV and compute Real_Time index.
    Removes invalid or very small flows.
    """
    df = pd.read_csv(file_path, sep=",", comment='#', skiprows=34)
    
    # Parse timestamp
    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Start_Time'])
    df['Real_Time'] = df['Timestamp'] + pd.to_timedelta(df['Elapsed_Time (min.)'], unit='m')
    
    # Filter data
    df = df.drop(df[(df['Runoff Rate(cfs)'] == 0) & (df['Elapsed_Time (min.)'] != 0)].index)
    df = df[df['Runoff Rate(cfs)'] > MIN_RUNOFF_CFS]
    
    return df[['Flume', 'Runoff Rate(cfs)', 'Accumulated_Volume(cf)', 'Real_Time']]


# --------------------- Utility --------------------- #

def downsample_runoff_events(ds_event, timestep=DOWNSAMPLE_TIMESTEP):
    """
    Downsample xarray Dataset along all dimensions
    """
    return xr.Dataset(
        {var_name: ds_event[var_name].isel(
            {dim: slice(None, None, timestep) for dim in ds_event.dims}
        ) for var_name in ds_event.data_vars},
        coords={dim: ds_event.coords[dim][::timestep] for dim in ds_event.dims}
    )

# --------------------- Event processing --------------------- #

def split_runoff_events(df_runoff, merge_threshold_min=MERGE_THRESHOLD_MIN):
    """
    Split a runoff DataFrame into individual events based on time gaps.
    Returns a list of DataFrames (each representing an event)
    """
    df_dict = {i: df for i, (_, df) in enumerate(df_runoff.groupby('Real_Time'))}
    
    current_group = []
    merged_groups = []

    for i, df in df_dict.items():
        if not current_group:
            current_group.append(df)
        else:
            last_time = current_group[-1]['Real_Time'].iloc[-1]
            current_time = df['Real_Time'].iloc[-1]
            if (current_time - last_time) <= timedelta(minutes=merge_threshold_min):
                current_group.append(df)
            else:
                merged_groups.append(pd.concat(current_group))
                current_group = [df]
    
    # Append any remaining group
    if current_group:
        merged_groups.append(pd.concat(current_group))
    
    # Sort events chronologically
    return sorted(merged_groups, key=lambda df: df['Real_Time'].dt.date.min())

def process_event_to_dataset(df_event, timestep=DOWNSAMPLE_TIMESTEP):
    """
    Convert a runoff DataFrame for a single event into xarray.Dataset:
    - Interpolate missing times
    - Remove zero tails
    - Downsample
    """
    event_flumes = np.unique(df_event['Flume'])
    event_time = pd.date_range(start=df_event.Real_Time.min(), end=df_event.Real_Time.max(), freq='15s')

    # Create dictionary of runoff per flume
    event_runoff_data = {
        f"flume_{int(flume)}": (["time"],
            df_event.loc[df_event['Flume'] == flume]
            .set_index('Real_Time')
            .reindex(event_time)['Runoff Rate(cfs)']
            .interpolate(method='linear')
            .fillna(0).values
        )
        for flume in event_flumes
    }

    # Remove flumes with all zeros
    event_runoff_data = {k: v for k, v in event_runoff_data.items() if v[1].sum() > 0}
    if not event_runoff_data:
        return None

    # Trim zero tails
    runoff_df = pd.DataFrame({k: v[1] for k, v in event_runoff_data.items()}, index=event_time)
    non_zero_mask = ~(runoff_df == 0).all(axis=1)
    first_idx = non_zero_mask.idxmax()
    last_idx = non_zero_mask[::-1].idxmax()
    runoff_df_trimmed = runoff_df.loc[first_idx:last_idx]

    # Recreate xarray.Dataset
    ds_event = xr.Dataset({
        col: (["time"], runoff_df_trimmed[col].values)
        for col in runoff_df_trimmed.columns
    }, coords={"time": runoff_df_trimmed.index})

    # Downsample
    return downsample_runoff_events(ds_event, timestep=timestep)

def process_all_runoff_files(runoff_files):
    """
    Process multiple runoff CSVs and return a list of xarray Datasets
    """
    
    ds_runoff_events = []

    for file in runoff_files:
        df_runoff = load_runoff_csv(file)
        events = split_runoff_events(df_runoff)
        for df_event in events:
            ds_event = process_event_to_dataset(df_event)
            if ds_event is not None:
                ds_runoff_events.append(ds_event)

    return ds_runoff_events

# ------------------------  Functions for preprocessed runoff data ----------------------------

def load_runoff_trees(runoff_files):
    """Load multiple runoff event DataTrees."""
    return [
        dtree.open_datatree(f, format="NETCDF4")
        for f in runoff_files
    ]

def get_all_event_labels(runoff_trees):
    """Return a list of all event labels across all trees."""
    labels = []
    for tree in runoff_trees:
        labels.extend([node.name for node in tree.descendants])
    return labels

def load_runoff_dates(runoff_date_files):
    """Load and concatenate runoff event dates."""
    dfs = [pd.read_csv(f, index_col=0) for f in runoff_date_files]
    runoff_dates = pd.concat(dfs, ignore_index=True)

    runoff_dates["start_time"] = pd.to_datetime(
        runoff_dates["start_time"], format="mixed"
    )
    runoff_dates["end_time"] = pd.to_datetime(
        runoff_dates["end_time"], format="mixed"
    )

    return runoff_dates
