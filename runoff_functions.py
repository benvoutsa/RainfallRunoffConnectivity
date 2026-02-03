"""
runoff helper functions
"""

import pandas as pd
import datatree as dtree

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
