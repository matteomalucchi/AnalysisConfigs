import os

import numpy as np


def plot_regions_names(cat_plot_name, namesuffix=""):
    cat_plot_name = cat_plot_name.replace("MC", "")
    cat_plot_name = cat_plot_name.replace("_transform", "")
    if "SPREAD" in cat_plot_name:
        cat_plot_name = cat_plot_name.split("_SPREAD")[0]
    cat_plot_name = cat_plot_name.replace("control_region", "CR")
    cat_plot_name = cat_plot_name.replace("signal_region", "SR")
    cat_plot_name = cat_plot_name.replace("_", " ")

    if "Run2" in cat_plot_name:
        cat_plot_name = cat_plot_name.replace("Run2", "")
        namesuffix += r" $D_{HH}$"
    
    cat_plot_name += namesuffix
    return cat_plot_name


def print_event_counts(cat_col_dict, sample_label, outputdir, weight_column="weight"):
    """Print a table with the number of events and the sum of the weights
    (weight_column) in each region, and save it to <outputdir>/event_counts_<sample_label>.txt"""
    rows = []
    for category in sorted(cat_col_dict.keys()):
        columns = cat_col_dict[category]
        if weight_column in columns:
            n_events = len(columns[weight_column])
            weights = np.asarray(columns[weight_column], dtype=np.float64)
            sum_w = weights[np.isfinite(weights)].sum()
        else:
            n_events = len(next(iter(columns.values()))) if columns else 0
            sum_w = float(n_events)
        rows.append((category, n_events, sum_w))

    width = max([len("Region")] + [len(r[0]) for r in rows])
    header = f"{'Region':<{width}} | {'N events':>12} | {'Sum of weights':>16}"
    lines = [header, "-" * len(header)]
    lines += [f"{c:<{width}} | {n:>12d} | {w:>16.3f}" for c, n, w in rows]
    table = "\n".join(lines)

    print(f"\nNumber of events per region ({sample_label}, weight: {weight_column}):")
    print(table + "\n")
    os.makedirs(outputdir, exist_ok=True)
    out_file = os.path.join(outputdir, f"event_counts_{sample_label}.txt")
    with open(out_file, "w") as f:
        f.write(f"# weight column: {weight_column}\n")
        f.write(table + "\n")
    print(f"Event counts saved in {out_file}")


# TODO: add a function to plot the columns names
def plot_columns_names():
    return
