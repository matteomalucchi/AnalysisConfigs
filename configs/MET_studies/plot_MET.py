import logging

logger = logging.getLogger("matplotlib")
logger.setLevel(logging.WARNING)  # suppress INFO
logger.propagate = False

import os
import numpy as np
from collections import defaultdict
import re
from multiprocessing import Pool
import argparse
from hist import Hist
from coffea.util import load, save

# from coffea import hist as coffea_hist

from utils.plot.get_columns_from_files import get_columns_from_files
from utils.plot.weighted_quantile import weighted_quantile
from plot_config import (
    total_var_dict,
    response_var_name_dict,
    qT_bins,
    PV_bins,
    met_dict_names,
    u_dict_names,
    N_bins,
    R_bin_edges,
    u_bin_edges,
)
from utils.plot.HEPPlotter import HEPPlotter


parser = argparse.ArgumentParser(description="Plot MET distributions from coffea files")
parser.add_argument(
    "-i",
    "--input-dir",
    type=str,
    required=True,
    help="Input directory for data with coffea files",
)
parser.add_argument(
    "--histo",
    action="store_true",
    default=False,
    help="If set, will plot 1d and 2d histograms of the recoil variables",
)
parser.add_argument(
    "-v",
    "--variables",
    action="store_true",
    default=False,
    help="If set, the inputs are read from variables instead from columns or parquet files ",
)
parser.add_argument(
    "-w",
    "--workers",
    type=int,
    default=1,
    help="Number of workers for multiprocessing (default: 1, no multiprocessing)",
)
parser.add_argument(
    "--novars",
    action="store_true",
    help="If true, old save format without saved variations is expected",
    default=False,
)
parser.add_argument(
    "-m",
    "--max-parquet",
    type=int,
    help="Maximum number of parquet files to load",
    default=None,
)
parser.add_argument(
    "-l",
    "--load",
    type=str,
    help="Path to precomputed histograms coffea file to load and plot",
    default=None,
)
parser.add_argument("-o", "--output", type=str, help="Output directory", default="")

args = parser.parse_args()


YEARS = ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix", "2024"]

BIN_VARIABLES = {
    "ll_pt": {
        "bin_edges": qT_bins,
        "label": "Z q$_{\\mathrm{T}}$ [GeV]",
        "name_plot": "Z_qT",
    },
    "PV_npvs": {"bin_edges": PV_bins, "label": "#PV", "name_plot": "nPV"},
}

outputdir = args.output if args.output else "plots_MET"

# Create output directory if it does not exist
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

met_histograms_dir = os.path.join(outputdir, "met_histograms")
if not os.path.exists(met_histograms_dir):
    os.makedirs(met_histograms_dir)

response_dir = os.path.join(outputdir, "response_curves")
if not os.path.exists(response_dir):
    os.makedirs(response_dir)

histograms_dir = os.path.join(outputdir, "1d_response_histograms")
if not os.path.exists(histograms_dir):
    os.makedirs(histograms_dir)
histograms_2d_dir = os.path.join(outputdir, "2d_response_histograms")
if not os.path.exists(histograms_2d_dir):
    os.makedirs(histograms_2d_dir)


def extract_labels(var_name):
    y_var_name, x_var_name = var_name.split("VS")
    y_label = (
        y_var_name
        if y_var_name not in response_var_name_dict
        else response_var_name_dict[y_var_name]
    )
    x_label = [
        v["label"] for k, v in BIN_VARIABLES.items() if v["name_plot"] == x_var_name
    ][0]
    return y_label, x_label


def save_dict_to_file(dict_to_save, output_path):
    """Save multiple dictionaries to a single coffea file.

    Parameters
    ----------
    dict_to_save : dict
        Dictionary to save.
    output_path : str
        Output file path.
    """

    save(dict_to_save, output_path)
    print(f"Saved histograms to {output_path}")


def extract_year_tag(s):
    match = re.search(r"(19|20)\d{2}(?:_[A-Za-z0-9]+)?", s)
    return match.group() if match else None


def run_plot(plotter):
    """Run a HEPPlotter instance."""
    plotter.run()


def weighted_mean(x, w):
    """
    Compute the weighted mean and its standard error.

    Parameters
    ----------
    x : array-like
        Input values.
    w : array-like
        Weights associated with the values.

    Returns
    -------
    mean_w : float
        Weighted mean of x.
    sem_w : float
        Standard error of the weighted mean, accounting for effective statistics.
    """
    mean_w = np.average(x, weights=w)
    n_eff = (np.sum(w)) ** 2 / np.sum(w**2)  # Effective number of entries
    variance_w = np.sum(w * (x - mean_w) ** 2) / np.sum(w)
    sem_w = np.sqrt(variance_w / n_eff)
    return mean_w, sem_w


def weighted_std_dev(x, w):
    """
    Compute the weighted standard deviation and its uncertainty.

    Parameters
    ----------
    x : array-like
        Input values.
    w : array-like
        Weights associated with the values.

    Returns
    -------
    std_w : float
        Weighted standard deviation of x.
    std_err_w : float
        Error estimate of the weighted standard deviation.
    """
    mean_w = np.average(x, weights=w)
    var_w = np.sum(w * (x - mean_w) ** 2) / np.sum(w)
    std_w = np.sqrt(var_w)

    n_eff = (np.sum(w) ** 2) / np.sum(w**2)
    std_err_w = std_w / np.sqrt(2 * (n_eff - 1))
    return std_w, std_err_w


def compute_u_info(u_i, weights_i, distribution_name, all_responses, bin_var):
    """
    Compute mean, quantile resolution, and std. dev for a given distribution
    and fill the results into the histogram dictionary.

    Parameters
    ----------
    u_i : array-like
        Distribution values in one bin.
    weights_i : array-like
        Weights corresponding to u_i.
    distribution_name : str
        Name of the distribution (e.g. 'u_perp', 'u_paral').
    all_responses : dict
        Dictionary where results (values/errors) are stored.
    bin_var: str
        Name of the variable used for binning
    """
    mean_u_i, err_mean_u_i = weighted_mean(u_i, weights_i)
    all_responses[f"{distribution_name}_meanVS{bin_var}"]["data"]["y"][0].append(
        mean_u_i
    )
    all_responses[f"{distribution_name}_meanVS{bin_var}"]["data"]["y"][1].append(
        err_mean_u_i
    )

    all_responses[f"{distribution_name}_quantile_resolutionVS{bin_var}"]["data"]["y"][
        0
    ].append(
        (
            float(weighted_quantile(u_i, 0.84, weights_i))
            - float(weighted_quantile(u_i, 0.16, weights_i))
            # np.quantile(u_i, 0.84) - np.quantile(u_i, 0.16)
        )
        / 2.0,
    )
    # TODO: compute error on quantile resolution
    all_responses[f"{distribution_name}_quantile_resolutionVS{bin_var}"]["data"]["y"][
        1
    ].append(0)

    stddev_u_i, err_stddev_u_i = weighted_std_dev(u_i, weights_i)
    all_responses[f"{distribution_name}_stddev_resolutionVS{bin_var}"]["data"]["y"][
        0
    ].append(stddev_u_i)
    all_responses[f"{distribution_name}_stddev_resolutionVS{bin_var}"]["data"]["y"][
        1
    ].append(err_stddev_u_i)


def create_hist(
    hists_dict,
    dict_info_x,
    dict_info_y,
    weights,
    style,
):
    """
    Create and fill a 2D histogram.

    Parameters
    ----------
    hists_dict : dict
        Dictionary where histograms are stored.
    dict_info_x: dict
        Dictionary where info about the variable on the x-axis are stored.
    dict_info_y: dict
        Dictionary where info about the variable on the y-axis are stored.
    weights : array-like
        Event weights.
    style : dict
        Plotting style metadata to attach to the histogram.
    """
    h = (
        Hist.new.Var(
            dict_info_x["bin_edges"],
            name=dict_info_x["name_plot"],
            label=dict_info_x["label"],
            flow=False,
        )
        .Var(
            dict_info_y["bin_edges"],
            name=dict_info_y["name_plot"],
            label=dict_info_y["label"],
            flow=False,
        )
        .Weight()
    )
    h.style = style
    if "rescale_array" in dict_info_y:
        # rescale u by the average response in each bin
        bin_indices = np.digitize(dict_info_x["array"], dict_info_x["bin_edges"]) - 1
        # avoid out of range indices
        bin_indices = np.clip(bin_indices, 0, len(dict_info_y["rescale_array"]) - 1)
        dict_info_y["array"] = (
            dict_info_y["array"] / np.array(dict_info_y["rescale_array"])[bin_indices]
        )

    h.fill(dict_info_x["array"], dict_info_y["array"], weight=weights)
    hists_dict[f"{dict_info_y['name_plot']}VS{dict_info_x['name_plot']}"] = h


def create_reponses_info(bin_var_array, u_dict, weights, bin_var):
    """
    Build response summaries and histograms for each MET type.

    Parameters
    ----------
    bin_var_array : array-like
        Array of the variable to use for binning.
    u_dict : dict
        Dictionary with variables for each MET type (u_perp, u_paral, response).
    weights : array-like
        Event weights.
    bin_var: str
        Name of the variable used for binning

    Returns
    -------
    responses_dict : dict
        Nested dictionary with response summaries (mean, stddev, quantiles).
    hists_dict : dict
        Nested dictionary with histograms for each variable and MET type.
    """

    bin_edges = BIN_VARIABLES[bin_var]["bin_edges"]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    inds = np.digitize(bin_var_array, bin_edges)

    name_bin_var = BIN_VARIABLES[bin_var]["name_plot"]

    all_responses = {}
    all_hists = {}

    u_info_list = [
        "R",
        "u_perp",
        "u_perp_scaled",
        "u_paral",
        "u_paral_scaled",
    ]
    metric_list = ["mean", "quantile_resolution", "stddev_resolution"]

    for met_type, style in met_dict_names.items():
        met_type = f"u{met_type}"
        print(met_type)
        assert met_type in u_dict, f"MET type {met_type} not found in u_dict"

        for var_name in u_dict[met_type]:
            if "u_perp" in var_name:
                u_perp_arr = u_dict[met_type][var_name]
            elif "u_paral" in var_name:
                u_par_arr = u_dict[met_type][var_name]
            elif "response" in var_name:
                R_arr = u_dict[met_type][var_name]

        all_responses[met_type] = defaultdict(defaultdict)

        for i in range(1, len(bin_edges)):
            weights_i = weights[np.where(inds == i)[0]]
            if i == 1:
                # initialize the dicts and add style info
                for var in u_info_list:
                    for metric in metric_list:
                        complete_name = f"{var}_{metric}VS{name_bin_var}"
                        if complete_name not in all_responses[met_type]:
                            all_responses[met_type][complete_name] = {
                                "data": {"x": [[], []], "y": [[], []]},
                                "style": style,
                            }
                        all_responses[met_type][complete_name]["data"]["x"][
                            0
                        ] = bin_centers.tolist()
                        all_responses[met_type][complete_name]["data"]["x"][1] = (
                            (bin_edges[1:] - bin_edges[:-1]) / 2.0
                        ).tolist()

            # check if the bin is empty and put nan
            if sum(weights_i) < 1e-6:
                for var in u_info_list:
                    for metric in metric_list:
                        complete_name = f"{var}_{metric}VS{name_bin_var}"
                        if complete_name not in all_responses[met_type]:
                            all_responses[met_type][complete_name] = {
                                "data": {"x": [[], []], "y": [[], []]},
                                "style": style,
                            }
                        all_responses[met_type][complete_name]["data"]["y"][0].append(
                            np.nan
                        )
                        all_responses[met_type][complete_name]["data"]["y"][1].append(0)
                continue

            # Define quantities for this bin
            R_i = R_arr[np.where(inds == i)[0]]
            av_R_i, _ = weighted_mean(R_i, weights_i)
            u_perp_i = u_perp_arr[np.where(inds == i)[0]]
            u_perp_scaled_i = u_perp_i / av_R_i
            u_par_i = u_par_arr[np.where(inds == i)[0]]
            u_par_scaled_i = u_par_i / av_R_i

            u_info_dict = {
                "R": R_i,
                "u_perp": u_perp_i,
                "u_perp_scaled": u_perp_scaled_i,
                "u_paral": u_par_i,
                "u_paral_scaled": u_par_scaled_i,
            }

            for var, u_arr in u_info_dict.items():
                compute_u_info(
                    u_arr, weights_i, var, all_responses[met_type], name_bin_var
                )

        # Create 2D histograms of bin variable vs response, u_perp, u_paral
        all_hists[met_type] = {}

        array_dict = {
            "R": {
                "array": R_arr,
                "label": response_var_name_dict["R"],
                "name_plot": "R",
                "bin_edges": R_bin_edges,
            },
            "u_perp": {
                "array": u_perp_arr,
                "label": response_var_name_dict["u_perp"],
                "name_plot": "u_perp",
                "bin_edges": u_bin_edges,
            },
            "u_perp_scaled": {
                "array": u_perp_arr,
                "label": response_var_name_dict["u_perp_scaled"],
                "name_plot": "u_perp_scaled",
                "bin_edges": u_bin_edges,
                "rescale_array": all_responses[met_type][f"R_meanVS{name_bin_var}"][
                    "data"
                ]["y"][0],
            },
            "u_paral": {
                "array": u_par_arr,
                "label": response_var_name_dict["u_paral"],
                "name_plot": "u_paral",
                "bin_edges": u_bin_edges,
            },
            "u_paral_scaled": {
                "array": u_par_arr,
                "label": response_var_name_dict["u_paral_scaled"],
                "name_plot": "u_paral_scaled",
                "bin_edges": u_bin_edges,
                "rescale_array": all_responses[met_type][f"R_meanVS{name_bin_var}"][
                    "data"
                ]["y"][0],
            },
        }

        bin_var_dict = {"array": bin_var_array} | BIN_VARIABLES[bin_var]
        for var in array_dict:
            create_hist(
                all_hists[met_type],
                bin_var_dict,
                array_dict[var],
                weights,
                style,
            )

    # change the gerarchy of the keys
    responses_dict = {}
    for met_type in all_responses:
        for var_name in all_responses[met_type]:
            if var_name not in responses_dict:
                responses_dict[var_name] = {}
            responses_dict[var_name][met_type] = all_responses[met_type][var_name]

    hists_dict = {}
    for met_type in all_hists:
        for var_name in all_hists[met_type]:
            if var_name not in hists_dict:
                hists_dict[var_name] = {}
            hists_dict[var_name][met_type] = all_hists[met_type][var_name]

    return responses_dict, hists_dict


def create_met_histos(col_var, category):
    """
    Build MET comparison histograms for a given category.

    Parameters
    ----------
    col_var : dict
        Dictionary of input variables (columns).
    category : str
        Category name (e.g. event selection).
    """

    for quantity_name, var_dict in total_var_dict.items():

        hist_1d_dict = {}
        ref_var = var_dict["reference"]

        for i, variable in enumerate(var_dict["variables"]):
            col_num = col_var[variable]
            weight = col_var["weight"]
            var_name = variable.split("_")[0]

            # Build numerator histogram only
            hist_num = Hist.new.Reg(
                N_bins,
                var_dict["range"][0],
                var_dict["range"][1],
                name=var_name,
                flow=False,
            ).Weight()
            hist_num.fill(col_num, weight=weight)

            # Store into dict expected by cms_plotter
            hist_1d_dict[var_name] = {
                "data": hist_num,
                "style": {
                    "is_reference": (variable == ref_var),
                    "color": (var_dict["colors"][i] if "colors" in var_dict else None),
                },
            }

        # Output name
        output_name = os.path.join(met_histograms_dir, f"{category}_{quantity_name}")

        info = {
            "series_dict": hist_1d_dict,
            "output_base": output_name,
            "xlabel": var_dict["plot_name"],
            "ylabel": "Events",
            "y_log": var_dict["log"],
            "ratio_label": var_dict.get("ratio_label", "Ratio"),
        }

        return {quantity_name: info}


def plot_reponses(responses_dict, cat, year):
    """
    Plot response curves (mean, stddev, quantile resolutions) vs the bin variable.

    Parameters
    ----------
    responses_dict : dict
        Response summary information from `create_reponses_info`.
    cat : str
        Category name.
    year : str
        Year string for labeling.
    """

    plotters = []

    for var_name in responses_dict:
        y_label, x_label = extract_labels(var_name)

        p = (
            HEPPlotter()
            .set_plot_config(lumitext=f"{year} (13.6 TeV)")
            .set_options(split_legend=False, set_ylim=False)
            .set_output(f"{response_dir}/{cat}_{var_name}")
            .set_data(responses_dict[var_name], plot_type="graph")
            .set_labels(x_label, y_label)
        )
        if "R" in var_name and "mean" in var_name:
            p = p.add_line(
                orientation="h",
                y=1.0,
                color="black",
                linestyle="--",
            )
        plotters.append(p)

    if args.workers > 1:
        print(f"Plotting responses in parallel in category {cat}")
        with Pool(args.workers) as pool:
            pool.map(run_plot, plotters)
    else:
        for i, p in enumerate(plotters):
            print(
                f"Plotting response for {list(responses_dict.keys())[i]} in category {cat}"
            )
            p.run()


def plot_2d_response_histograms(hists_dict, cat, year):
    """
    Plot 2D histograms (bin variable vs response) for each MET type.

    Parameters
    ----------
    hists_dict : dict
        Histogram dictionary from `create_reponses_info`.
    cat : str
        Category name.
    year : str
        Year string for labeling.
    """

    plotters = []
    for var_name in hists_dict:
        y_label, x_label = extract_labels(var_name)

        for met_type in hists_dict[var_name]:
            series_dict = {
                f"{var_name} {met_type}": {
                    "data": hists_dict[var_name][met_type],
                    "style": {"label": f"{var_name} {met_type}"},
                }
            }

            p = (
                HEPPlotter()
                .set_plot_config(
                    figsize=(14, 13), lumitext=f"{met_type}     {year} (13.6 TeV)"
                )
                .set_options(legend=False, cbar_log=True)
                .set_output(f"{histograms_2d_dir}/2d_histo_{cat}_{var_name}_{met_type}")
                .set_data(
                    series_dict,
                    plot_type="2d",
                )
                .set_labels(x_label, y_label)
            )
            plotters.append(p)

    if args.workers > 1:
        print(f"Plotting 2d histograms in parallel in category {cat}")
        with Pool(args.workers) as pool:
            pool.map(run_plot, plotters)
    else:
        for p in plotters:
            print(
                f"Plotting 2d histogram for {p.output_base.split('/')[-1]} in category {cat}"
            )
            p.run()


def plot_1d_response_histograms(hists_dict, cat, year):
    """
    Plot 1D histograms of variables in each bin.

    Parameters
    ----------
    hists_dict : dict
        Histogram dictionary from `create_reponses_info`.
    cat : str
        Category name.
    year : str
        Year string for labeling.
    """

    # for each bin, plot the distribution of the variable
    plotters = []

    for var_name in hists_dict:
        hist_1d_dict = {}
        var_label, _ = extract_labels(var_name)
        for met_type in hists_dict[var_name]:
            hist = hists_dict[var_name][met_type]
            bin_edges = hist.axes[0].edges
            bin_var = hist.axes[0].name
            label = [
                v["label"]
                for k, v in BIN_VARIABLES.items()
                if v["name_plot"] == bin_var
            ][0]

            for i in range(len(bin_edges) - 1):
                bin_name = f"{bin_var}:{i}"
                if bin_name not in hist_1d_dict.keys():
                    hist_1d_dict[bin_name] = {}

                bin_edges_string = f"{bin_var}{bin_edges[i]}-{bin_edges[i+1]}"
                output_name = f"{histograms_dir}/{cat}_{var_name}_{bin_edges_string}"

                # Select the bin corresponding to the current bin
                hist_1d_u = hist[{bin_var: i}]
                hist_1d_dict[bin_name][met_type] = {
                    "data": hist_1d_u,
                    "style": hist.style,
                }

                hist_1d_dict[bin_name]["infos"] = {
                    "output_name": output_name,
                    "lumitext_prefix": f"{bin_edges[i]} < {label} < {bin_edges[i+1]}",
                }

        for bin_name, hist_met_dict in hist_1d_dict.items():
            lumitext_prefix = hist_met_dict["infos"]["lumitext_prefix"]
            output_name = hist_met_dict["infos"]["output_name"]
            hist_met_dict.pop("infos")

            p = (
                HEPPlotter()
                .set_plot_config(
                    figsize=(14, 13),
                    lumitext=f"{lumitext_prefix}      {year} (13.6 TeV)",
                )
                .set_output(output_name)
                .set_labels(var_label, "Events")
                .set_options(y_log=False, split_legend=False, set_ylim_ratio=0.5)
                .set_data(hist_met_dict, plot_type="1d")
                .add_line(
                    orientation="v",
                    x=1.0 if "R" in var_name else 0.0,
                    color="black",
                    linestyle="--",
                )
            )
            plotters.append(p)

    if args.workers > 1:
        print(f"Plotting 1d histograms in parallel in category {cat}")
        with Pool(args.workers) as pool:
            pool.map(run_plot, plotters)
    else:
        for p in plotters:
            print(
                f"Plotting 1d histogram for {p.output_base.split('/')[-1]} in category {cat}"
            )
            p.run()


def plot_histo_met(met_dict, year):
    """
    Plot MET histograms using either multiprocessing or sequential execution.

    Parameters
    ----------
    met_dict : dict
        Dictionary of histogram plotting configurations.
    year : str
        Year string for labeling.
    """

    plotters = []
    for info in met_dict.values():
        print(f"Plotting MET histogram: {info['output_base']}")
        p = (
            HEPPlotter()
            .set_plot_config(figsize=(14, 13), lumitext=f"{year} (13.6 TeV)")
            .set_output(info["output_base"])
            .set_labels(info["xlabel"], info["ylabel"], ratio_label=info["ratio_label"])
            .set_options(
                y_log=info["y_log"],
                set_ylim=True,
                split_legend=False,
                set_ylim_ratio=0.5,
            )
            .set_data(info["series_dict"], plot_type="1d")
        )
        plotters.append(p)
    if args.workers > 1:
        with Pool(args.workers) as pool:
            pool.map(run_plot, plotters)
    else:
        for p in plotters:
            p.run()


def do_plots(responses_dict, hists_dict, met_dict, category, year):
    """Plot all the required plots.

    Parameters
    ----------
    responses_dict : dict
        Response summary information from `create_reponses_info`.
    hists_dict : dict
        Histogram dictionary from `create_reponses_info`.
    met_dict : dict
        Dictionary of histogram plotting configurations.
    category : str
        Category name.
    year : str
        Year string for labeling.
    bin_vars : list
        Variables used for binning.
    """

    # plot per-bin response histograms
    if args.histo:
        plot_1d_response_histograms(hists_dict, category, year)
        plot_2d_response_histograms(hists_dict, category, year)

    # plot response curves
    plot_reponses(responses_dict, category, year)
    # plot MET histograms
    plot_histo_met(met_dict, year)


def main():
    inputfiles_data = [
        os.path.join(args.input_dir, file)
        for file in os.listdir(args.input_dir)
        if file.endswith(".coffea")
    ]

    if args.load:
        # load precomputed histograms
        print(f"Loading precomputed histograms from {args.load}")
        loaded_dict = load(args.load)
        responses_dict = loaded_dict["responses"]
        hists_dict = loaded_dict["hists"]
        met_dict = loaded_dict["met_histos"]
        year = loaded_dict.get("year", "unknown_year")
        category = loaded_dict.get("category", "unknown_category")

        do_plots(responses_dict, hists_dict, met_dict, category, year)

        return

    if not args.variables:
        print("Loading the columns...")
        cat_col, total_datasets_list = get_columns_from_files(
            inputfiles_data,
            "nominal",
            None,
            True,
            args.novars,
            max_num_parquet_files=args.max_parquet,
        )

        # get year from dataset
        year = ""
        for dataset in total_datasets_list:
            # dataset are of the shape name_year_yearsuffix or name_year
            y = extract_year_tag(dataset)

            year = ", ".join([year, y]) if year else y

        print(f"Total datasets found: {total_datasets_list}")
        print(f"Year: {year}")

        for category, col_var in cat_col.items():
            responses_dict = {}
            hists_dict = {}
            for bin_var in BIN_VARIABLES.keys():
                met_dict = {}
                print(f"Processing category: {category} and bin in variable {bin_var}")
                bin_var_array = col_var[bin_var]

                u_dict = {}
                for var in col_var:
                    coll = var.split("_")[0]
                    if (
                        any(
                            x in var
                            for x in ["u_perp_predict", "u_paral_predict", "response"]
                        )
                        and coll in u_dict_names
                    ):

                        if coll not in u_dict:
                            u_dict[coll] = {}
                        u_dict[coll][var] = col_var[var]
                    elif "weight" in var:
                        weights = col_var[var]

                # build response info
                new_responses_dict, new_hists_dict = create_reponses_info(
                    bin_var_array, u_dict, weights, bin_var
                )
                responses_dict |= new_responses_dict
                hists_dict |= new_hists_dict

                # build MET histograms
                met_dict |= create_met_histos(col_var, category)

            # save all the histograms
            save_dict_to_file(
                {
                    "responses": responses_dict,
                    "hists": hists_dict,
                    "met_histos": met_dict,
                    "year": year,
                    "category": category,
                },
                os.path.join(
                    outputdir, f"histograms_{category.replace(' ','_')}.coffea"
                ),
            )

            # plot all the required plots
            do_plots(responses_dict, hists_dict, met_dict, category, year)

    else:
        raise ValueError("Not implemented")
        # read input variables
        accumulator = load(inputfiles_data[0])

        for var, histos_dict in accumulator["variables"].items():
            for sample in histos_dict:
                for dataset in histos_dict[sample]:
                    print(f"Variable: {var}, Sample: {sample}", f"Dataset: {dataset}")
                    hist_obj = histos_dict[sample][dataset]
                    histo = hist_obj[{"cat": list(hist_obj.axes["cat"])[0]}][
                        {"variation": list(hist_obj.axes["variation"])[0]}
                    ]

    print("Finished plotting!")


if __name__ == "__main__":

    main()
