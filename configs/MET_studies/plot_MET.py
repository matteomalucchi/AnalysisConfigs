import logging

logger = logging.getLogger("matplotlib")
logger.setLevel(logging.WARNING)  # suppress INFO
logger.propagate = False

import os
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
import argparse
from hist import Hist
from coffea.util import load

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
import helpers

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
    "-s",
    "--slice",
    action="store_true",
    default=False,
    help="If set, will plot all 1d and 2d histograms, and the response graph sliced in each bin of the binning variables. Very heavy output, use with caution!",
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
        "label": r"Z q$_{\mathrm{T}}$ [GeV]",
        "name_plot": "Z_qT",
    },
    "PV_npvs": {"bin_edges": PV_bins, "label": "#PV", "name_plot": "nPV"},
}

MASK = False

outputdir = args.output if args.output else "plots_MET"

# Create output directories if it does not exist
met_histograms_dir = os.path.join(outputdir, "met_histograms")
os.makedirs(met_histograms_dir, exist_ok=True)

# Response curves directories
response_dir = os.path.join(outputdir, "response_curves")
response_slice_dir = os.path.join(response_dir, "slice")
os.makedirs(response_slice_dir, exist_ok=True)

# 1D response histograms directories
histograms_1d_dir = os.path.join(outputdir, "1d_response_histograms")
histograms_1d_slice_dir = os.path.join(histograms_1d_dir, "1d_slice")
os.makedirs(histograms_1d_slice_dir, exist_ok=True)
histograms_1d_project_dir = os.path.join(histograms_1d_dir, "1d_project")
os.makedirs(histograms_1d_project_dir, exist_ok=True)

# 2D response histograms directories
histograms_2d_dir = os.path.join(outputdir, "2d_response_histograms")
histograms_2d_profile_dir = os.path.join(histograms_2d_dir, "2d_profile")
os.makedirs(histograms_2d_profile_dir, exist_ok=True)
histograms_2d_histo_dir = os.path.join(histograms_2d_dir, "2d_histo")
os.makedirs(histograms_2d_histo_dir, exist_ok=True)


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
    mean_u_i, err_mean_u_i = helpers.weighted_mean(u_i, weights_i)
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

    stddev_u_i, err_stddev_u_i = helpers.weighted_std_dev(u_i, weights_i)
    all_responses[f"{distribution_name}_stddev_resolutionVS{bin_var}"]["data"]["y"][
        0
    ].append(stddev_u_i)
    all_responses[f"{distribution_name}_stddev_resolutionVS{bin_var}"]["data"]["y"][
        1
    ].append(err_stddev_u_i)


def create_hist(
    hists_dict,
    dict_info_x_list,
    dict_info_y,
    weights,
    style,
):
    """
    Create and fill an N-D histogram plus N-D profile graphs (mean, std dev,
    quantile resolution) for each response variable.

    Parameters
    ----------
    hists_dict : dict
        Dictionary where histograms are stored.
    dict_info_x_list : list of dict or dict
        List of dicts with info about each binning (x) axis. For backward
        compatibility a single dict is also accepted and wrapped automatically.
    dict_info_y : dict
        Dictionary with info about the variable on the y-axis.
    weights : array-like
        Event weights.
    style : dict
        Plotting style metadata to attach to the histogram.
    """
    # ---- Backward compatibility: accept a single dict as before --------------
    if isinstance(dict_info_x_list, dict):
        dict_info_x_list = [dict_info_x_list]

    n_dims = len(dict_info_x_list)

    # ---- Build the Hist object dynamically -----------------------------------
    h_builder = Hist.new
    for dict_info_x in dict_info_x_list:
        h_builder = h_builder.Var(
            dict_info_x["bin_edges"],
            name=dict_info_x["name_plot"],
            label=dict_info_x["label"],
            flow=False,
        )

    hist = h_builder.Var(
        dict_info_y["bin_edges"],
        name=dict_info_y["name_plot"],
        label=dict_info_y["label"],
        flow=False,
    ).Weight()
    hist.style = style

    # ---- N-D rescaling of the y-array ----------------------------------------
    # rescale_array is now an N-D array with shape (n_bins_x0, n_bins_x1, ...)
    # built from the mean response in each N-D bin.
    if "rescale_array" in dict_info_y:
        rescale_array = np.array(dict_info_y["rescale_array"])

        # One bin index per dimension
        bin_indices = []
        for d, dict_info_x in enumerate(dict_info_x_list):
            idx = np.digitize(dict_info_x["array"], dict_info_x["bin_edges"]) - 1
            idx = np.clip(idx, 0, rescale_array.shape[d] - 1)
            bin_indices.append(idx)

        # Index into the N-D rescale array per event
        scale_per_event = rescale_array[tuple(bin_indices)]
        dict_info_y["array"] = dict_info_y["array"] / scale_per_event

    # ---- Fill the N-D histogram ----------------------------------------------
    x_arrays = [d["array"] for d in dict_info_x_list]
    hist.fill(*x_arrays, dict_info_y["array"], weight=weights)

    x_names = "VS".join(d["name_plot"] for d in dict_info_x_list)
    key_base = f"{dict_info_y['name_plot']}VS{x_names}"
    hists_dict[key_base] = hist

    # ---- N-D profile graphs --------------------------------------------------

    all_bin_edges = [d["bin_edges"] for d in dict_info_x_list]
    all_bin_centers = [(e[1:] + e[:-1]) / 2.0 for e in all_bin_edges]
    n_bins_per_dim = [len(e) - 1 for e in all_bin_edges]

    # Digitise all dimensions once (1-indexed from np.digitize)
    all_inds = [
        np.digitize(dict_info_x_list[d]["array"], all_bin_edges[d])
        for d in range(n_dims)
    ]

    y_arr = dict_info_y["array"]  # already rescaled if needed

    # Allocate output arrays
    shape = tuple(n_bins_per_dim)
    mean_vals = np.full(shape, np.nan)
    mean_errs = np.full(shape, np.nan)
    std_vals = np.full(shape, np.nan)
    std_errs = np.full(shape, np.nan)
    qres_vals = np.full(shape, np.nan)
    qres_errs = np.full(shape, np.nan)

    # Iterate over every N-D bin combination
    for multi_idx in np.ndindex(*shape):
        # multi_idx is 0-indexed; np.digitize is 1-indexed
        mask = np.ones(len(y_arr), dtype=bool)
        for d, idx in enumerate(multi_idx):
            mask &= all_inds[d] == (idx + 1)

        w_i = weights[mask]
        y_i = y_arr[mask]

        if w_i.sum() < 1e-6:
            continue  # leave as NaN

        av, av_err = helpers.weighted_mean(y_i, w_i)
        mean_vals[multi_idx] = av
        mean_errs[multi_idx] = av_err

        sd, sd_err = helpers.weighted_std_dev(y_i, w_i)
        std_vals[multi_idx] = sd
        std_errs[multi_idx] = sd_err
        qr, qr_err = (
            (
                float(weighted_quantile(y_i, 0.84, w_i))
                - float(weighted_quantile(y_i, 0.16, w_i))
            )
            / 2.0
        ), np.zeros_like(sd_err)
        qres_vals[multi_idx] = qr
        qres_errs[multi_idx] = qr_err

    # Package and store each metric
    for metric_name, vals, errs in [
        ("mean", mean_vals, mean_errs),
        ("stddev_resolution", std_vals, std_errs),
        ("quantile_resolution", qres_vals, qres_errs),
    ]:
        # Build a Hist with the same N-D binning axes but Double() storage
        # (no y-axis — the metric value IS the "count")
        h_profile_builder = Hist.new
        for dict_info_x in dict_info_x_list:
            h_profile_builder = h_profile_builder.Var(
                dict_info_x["bin_edges"],
                name=dict_info_x["name_plot"],
                label=dict_info_x["label"],
                flow=False,
            )
        h_profile = h_profile_builder.Double()
        h_profile.style = style

        # Directly assign the pre-computed metric values into the histogram view.
        # For Double() storage the view is just an N-D array of floats.
        h_profile.view()[:] = np.where(np.isnan(vals), 0.0, vals)

        # Store errors as a plain attribute (Hist has no native error storage
        # for Double(), but it's accessible for plotting)
        h_profile.errors = np.where(np.isnan(errs), 0.0, errs)
        h_profile.has_nan_mask = np.isnan(vals)  # track empty bins separately

        profile_key = f"{dict_info_y['name_plot']}_{metric_name}VS{x_names}"
        hists_dict[profile_key] = h_profile


def create_responses_info(bin_var_arrays, u_dict, weights, bin_vars):
    """
    Build response summaries and histograms for each MET type, binned in N dimensions.

    Parameters
    ----------
    bin_var_arrays : list of array-like
        List of arrays for each binning variable (one per dimension).
    u_dict : dict
        Dictionary with variables for each MET type (u_perp, u_paral, response).
    weights : array-like
        Event weights.
    bin_vars : list of str
        Names of the variables used for binning (one per dimension).

    Returns
    -------
    responses_dict : dict
        Nested dictionary with response summaries (mean, stddev, quantiles),
        saved as a function of each binning variable (marginalizing over others).
    hists_dict : dict
        Nested dictionary with N-D histograms for each variable and MET type.
    """
    # --- Validate inputs -------------------------------------------------------
    if isinstance(bin_vars, str):
        bin_vars = [bin_vars]
        bin_var_arrays = [bin_var_arrays]

    assert len(bin_vars) == len(
        bin_var_arrays
    ), "bin_vars and bin_var_arrays must have the same length"

    n_dims = len(bin_vars)

    # Per-dimension bin edges, centers, names
    all_bin_edges = [BIN_VARIABLES[bv]["bin_edges"] for bv in bin_vars]
    all_bin_centers = [(e[1:] + e[:-1]) / 2.0 for e in all_bin_edges]
    all_name_bin_var = [BIN_VARIABLES[bv]["name_plot"] for bv in bin_vars]

    # Digitize each dimension independently
    all_inds = [np.digitize(bin_var_arrays[d], all_bin_edges[d]) for d in range(n_dims)]

    # Number of bins per dimension (excluding under/overflow)
    n_bins = [len(e) - 1 for e in all_bin_edges]

    u_info_list = ["R", "u_perp", "u_perp_scaled", "u_paral", "u_paral_scaled"]
    metric_list = ["mean", "quantile_resolution", "stddev_resolution"]

    all_responses = {}
    all_hists = {}

    for met_type, style in met_dict_names.items():
        met_type = f"u{met_type}"
        print(met_type)
        assert met_type in u_dict, f"MET type {met_type} not found in u_dict"

        # ---- Unpack per-event arrays for this MET type ------------------------
        for var_name in u_dict[met_type]:
            if "u_perp" in var_name:
                u_perp_arr = u_dict[met_type][var_name]
            elif "u_paral" in var_name:
                u_par_arr = u_dict[met_type][var_name]
            elif "response" in var_name:
                R_arr = u_dict[met_type][var_name]

        all_responses[met_type] = defaultdict(defaultdict)

        # ---- Response vs each binning variable (marginalise over the others) --
        for dim, (name_bin_var, bin_edges, bin_centers) in enumerate(
            zip(all_name_bin_var, all_bin_edges, all_bin_centers)
        ):
            inds_dim = all_inds[dim]  # digitised indices for this dimension

            for i in range(1, len(bin_edges)):  # 1-indexed, as digitize returns
                # Build mask: exact bin i in *this* dimension, any valid bin in others
                mask = inds_dim == i
                for d2 in range(n_dims):
                    if d2 == dim:
                        continue
                    mask &= (all_inds[d2] >= 1) & (all_inds[d2] <= n_bins[d2])

                weights_i = weights[mask]

                if i == 1:
                    # Initialise dict entries and attach x-axis data
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

                # Empty bin → append NaN
                if sum(weights_i) < 1e-6:
                    for var in u_info_list:
                        for metric in metric_list:
                            complete_name = f"{var}_{metric}VS{name_bin_var}"
                            if complete_name not in all_responses[met_type]:
                                all_responses[met_type][complete_name] = {
                                    "data": {"x": [[], []], "y": [[], []]},
                                    "style": style,
                                }
                            all_responses[met_type][complete_name]["data"]["y"][
                                0
                            ].append(np.nan)
                            all_responses[met_type][complete_name]["data"]["y"][
                                1
                            ].append(0)
                    continue

                # Quantities for events in this bin slice
                R_i = R_arr[mask]
                av_R_i, _ = helpers.weighted_mean(R_i, weights_i)
                u_perp_i = u_perp_arr[mask]
                u_perp_scaled_i = u_perp_i / av_R_i
                u_par_i = u_par_arr[mask]
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

        # ---- N-D histograms ---------------------------------------------------
        all_hists[met_type] = {}

        # ---- Build N-D rescale array from the N-D bin loop ----------------------
        # Shape: (n_bins_dim0, n_bins_dim1, ...)
        rescale_array_nd = np.full(tuple(n_bins), np.nan)

        for multi_idx in np.ndindex(*n_bins):
            mask = np.ones(len(R_arr), dtype=bool)
            for d, idx in enumerate(multi_idx):
                mask &= all_inds[d] == (idx + 1)  # digitize is 1-indexed

            w_i = weights[mask]
            R_i = R_arr[mask]

            if w_i.sum() < 1e-6:
                continue  # leave as NaN

            av_R_i, _ = helpers.weighted_mean(R_i, w_i)
            rescale_array_nd[multi_idx] = av_R_i

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
                "rescale_array": rescale_array_nd,  # N-D array, shape (n_bins_dim0, n_bins_dim1, ...)
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
                "rescale_array": rescale_array_nd,  # same N-D array
            },
        }

        # Build a list of bin-variable dicts (one per dimension) and pass to
        # create_hist, which is expected to handle N-D bin axes.
        bin_var_dicts = [
            {"array": bin_var_arrays[d]} | BIN_VARIABLES[bin_vars[d]]
            for d in range(n_dims)
        ]

        for var in array_dict:
            create_hist(
                all_hists[met_type],
                bin_var_dicts,
                array_dict[var],
                weights,
                style,
            )

    # ---- Invert key hierarchy: var_name → met_type ---------------------------
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
    met_dict = {}
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
        met_dict[quantity_name] = info

    # Add histograms of the difference between PuppiMET_pt and RawPuppiMET-Type1CorrMET_pt/RawPuppiMET-Type1CorrMETUncorrected_pt
    met_diff = {}
    for variable in [
        "PuppiMET_pt",
        "RawPuppiMET-Type1CorrMET_pt",
        "RawPuppiMET-Type1CorrMETUncorrected_pt",
    ]:
        try:
            met_diff[variable] = col_var[variable]
        except KeyError:
            print(
                f"Variable {variable} not found in the file, skipping the met difference plot"
            )
            met_diff = {}
            break

    for met_type1 in [
        "RawPuppiMET-Type1CorrMET_pt",
        "RawPuppiMET-Type1CorrMETUncorrected_pt",
    ]:
        rel_diff_perc = (
            (met_diff["PuppiMET_pt"] - met_diff[met_type1]) / met_diff["PuppiMET_pt"] * 100
        )
        hist_diff = Hist.new.Reg(
            1000,
            -25,
            25,
            name="met_diff",
            flow=False,
        ).Double()
        hist_diff.fill(rel_diff_perc)
        # compute the percentage which is less than 0.5%
        num_less_0p5 = len(np.where(abs(rel_diff_perc) < 0.5)[0])
        num_events = len(rel_diff_perc)
        frac_0p5 = num_less_0p5 / num_events
        print(
            f"Number of events with diff less than 0.5%: {num_less_0p5} out of {num_events}, fraction: {frac_0p5}"
        )

        hist_diff_dict = {
            "met_diff": {
                "data": hist_diff,
                "style": {
                    "is_reference": False,
                    "legend_name": f"(PuppiMET - {met_type1.replace('_pt', '')})/PuppiMET \n {frac_0p5*100:.6f}% of the time < 0.5%",
                },
            }
        }

        met_dict[f"met_diff_PuppiMET_{met_type1.replace('_pt', '')}"] = {
            "series_dict": hist_diff_dict,
            "output_base": os.path.join(
                met_histograms_dir,
                f"{category}_met_diff_PuppiMET_{met_type1.replace('_pt', '')}",
            ),
            "xlabel": "MET $p_{T}$ relative difference [%]",
            "ylabel": "Events",
            "y_log": True,
            # "set_ylim": False,
            "ratio_label": "Ratio",
        }

    return met_dict


def plot_responses(responses_dict, cat, year, hists_dict=None):
    """
    Plot response curves (mean, stddev, quantile resolutions) vs the bin variable.
    Additionally, if hists_dict is provided, slices 2D profiles to produce
    response-vs-variable-2 graphs in each bin of variable-1 (and vice versa).

    Parameters
    ----------
    responses_dict : dict
        Response summary information from `create_responses_info`.
    cat : str
        Category name.
    year : str
        Year string for labeling.
    hists_dict : dict, optional
        Histogram dictionary from `create_responses_info`. If provided,
        2D profile objects are extracted and sliced along each axis to
        produce 1D response graphs in each bin of the other axis.
    """

    # -----------------------------------------------------------------------

    plotters = []

    # ---- 1. Original 1D response graphs from responses_dict ----------------
    for var_name in responses_dict:
        y_label, x_labels = helpers.extract_labels(
            var_name, response_var_name_dict, BIN_VARIABLES
        )
        x_label = x_labels[0] if isinstance(x_labels, list) else x_labels

        p = (
            HEPPlotter()
            .set_plot_config(lumitext=f"{year} (13.6 TeV)")
            .set_options(split_legend=False, set_ylim=False)
            .set_output(f"{response_dir}/{cat}_{var_name}")
            .set_data(responses_dict[var_name], plot_type="graph")
            .set_labels(x_label, y_label)
        )
        if "R" in var_name and "mean" in var_name:
            p = p.add_line(orientation="h", y=1.0, color="black", linestyle="--")
        
        plotters.append(p)

    # ---- 2. 2D profile slices → 1D response graphs -------------------------
    if hists_dict is not None and args.slice:
        for var_name in hists_dict:

            # Check that at least one entry is a valid 2D profile
            if not any(
                helpers.is_profile(h) and len(h.axes) == 2
                for h in hists_dict[var_name].values()
            ):
                continue

            y_label, x_labels = helpers.extract_labels(
                var_name, response_var_name_dict, BIN_VARIABLES
            )

            # Use axes from the first valid profile (all met_types share the same axes)
            h_ref = next(
                h
                for h in hists_dict[var_name].values()
                if helpers.is_profile(h) and len(h.axes) == 2
            )
            ax0, ax1 = h_ref.axes[0], h_ref.axes[1]

            # Slice along ax0 → graph vs ax1, and vice versa
            for fixed_ax, free_ax in [(ax0, ax1), (ax1, ax0)]:

                # x_label is the free axis label
                free_ax_label = next(
                    (
                        v["label"]
                        for v in BIN_VARIABLES.values()
                        if v["name_plot"] == free_ax.name
                    ),
                    free_ax.name,
                )

                # Derive y-label from the metric part of var_name
                metric_part = var_name.split("_")[1] if "_" in var_name else var_name
                y_label_metric = (
                    y_label
                    if y_label not in response_var_name_dict
                    else response_var_name_dict.get(metric_part, y_label)
                )

                for bin_idx in range(len(fixed_ax)):
                    lo = fixed_ax.edges[bin_idx]
                    hi = fixed_ax.edges[bin_idx + 1]
                    bin_label = f"{lo:.4g} < {fixed_ax.label} < {hi:.4g}"
                    suffix = f"{fixed_ax.name}{lo:.2g}_{hi:.2g}"

                    # Collect all MET types into a single graph_dict
                    graph_dict = {}
                    for met_type, h in hists_dict[var_name].items():
                        if not helpers.is_profile(h) or len(h.axes) != 2:
                            continue
                        graph_dict[met_type] = helpers.profile2d_to_graph_dict(
                            h, fixed_ax, free_ax, bin_idx, h.style
                        )

                    if not graph_dict:
                        continue

                    output = (
                        f"{response_slice_dir}/{cat}_{var_name}"
                        f"_vs{free_ax.name}_{suffix}"
                    )

                    p = (
                        HEPPlotter()
                        .set_plot_config(
                            figsize=(14, 13),
                            lumitext=f"{bin_label}     {year} (13.6 TeV)",
                            lumitext_font_size=20,
                            cmstext_font_size=20,
                        )
                        .set_options(split_legend=False, set_ylim=False)
                        .set_output(output)
                        .set_data(graph_dict, plot_type="graph")
                        .set_labels(free_ax_label, y_label_metric)
                    )
                    if "R" in var_name and "mean" in var_name:
                        p = p.add_line(
                            orientation="h",
                            y=1.0,
                            color="black",
                            linestyle="--",
                        )
                    plotters.append(p)

    # ---- Dispatch ----------------------------------------------------------
    if args.workers > 1:
        print(f"Plotting responses in parallel in category {cat}")
        with Pool(args.workers) as pool:
            pool.map(helpers.run_plot, plotters)
    else:
        for p in plotters:
            print(
                f"Plotting response for {p.output_base.split('/')[-1]} in category {cat}"
            )
            p.run()


def plot_2d_response_histograms(hists_dict, cat, year):
    """
    Plot 2D histograms (bin variable vs response) for each MET type.
    For N-D histograms all combinations of extra dimensions are sliced,
    producing one 2D plot per combination (plus projected plots).
    Profile objects (mean / stddev / quantile_resolution) are converted to
    equivalent 2D Hist objects so HEPPlotter can handle them identically.

    Parameters
    ----------
    hists_dict : dict
        Histogram dictionary from `create_responses_info`.
    cat : str
        Category name.
    year : str
        Year string for labeling.
    """

    plotters = []

    for var_name in hists_dict:
        y_label, x_labels = helpers.extract_labels(
            var_name, response_var_name_dict, BIN_VARIABLES
        )

        for met_type in hists_dict[var_name]:
            hist = hists_dict[var_name][met_type]

            # ----------------------------------------------------------------
            # Profile objects — convert each 2-axis slice to a Hist2D
            # ----------------------------------------------------------------
            if helpers.is_profile(hist):

                n_axes = len(hist.axes)
                if n_axes < 2:
                    continue  # nothing to plot as 2D

                # Keep first two axes as the 2D plane, slice/project the rest
                free_axes = [hist.axes[0].name, hist.axes[1].name]

                for (
                    slice_dict,
                    label_parts,
                    suffix,
                    mode,
                ) in helpers.iter_slice_combinations(hist, free_axes):
                    # Apply slice_dict to the profile
                    h_sliced = hist[slice_dict] if slice_dict else hist

                    # Convert to a proper 2D Hist
                    h_2d = helpers.profile_to_hist2d(
                        h_sliced, h_sliced.axes[0], h_sliced.axes[1]
                    )

                    extra_label = (
                        ("   " + ",  ".join(label_parts)) if label_parts else ""
                    )
                    out_suffix = f"_{suffix}" if suffix else ""
                    output = f"{histograms_2d_profile_dir}/2d_profile_{cat}_{var_name}_{met_type}{out_suffix}"

                    series_dict = {
                        f"{var_name} {met_type}": {
                            "data": h_2d,
                            "style": {"label": f"{var_name} {met_type}"},
                        }
                    }

                    p = (
                        HEPPlotter()
                        .set_plot_config(
                            figsize=(14, 13),
                            lumitext=f"{met_type}{extra_label}     {year} (13.6 TeV)",
                            lumitext_font_size=20,
                            cmstext_font_size=20,
                        )
                        .set_options(legend=False, cbar_log=False)
                        .set_output(output)
                        .set_data(series_dict, plot_type="2d")
                        .set_labels(x_labels[0], x_labels[1], y_label)
                    )
                    plotters.append(p)

                continue

            # ----------------------------------------------------------------
            # Regular Hist objects — slice/project all dims beyond the 2D plane
            # ----------------------------------------------------------------
            for it, (y_idx, x_idx) in enumerate(zip([-1, -1], [-3, -2])):
                # y is the response axis
                y_axis_name = hist.axes[y_idx].name
                # x is the bin variable axis
                x_axis_name = hist.axes[x_idx].name

                free_axes = [x_axis_name, y_axis_name]

                for (
                    slice_dict,
                    label_parts,
                    suffix,
                    mode,
                ) in helpers.iter_slice_combinations(hist, free_axes):
                    h_2d = hist[slice_dict] if slice_dict else hist

                    extra_label = (
                        ("   " + ",  ".join(label_parts)) if label_parts else ""
                    )
                    out_suffix = f"_{suffix}" if suffix else ""
                    output = f"{histograms_2d_histo_dir}/2d_histo_{cat}_{var_name}_{met_type}{out_suffix}"

                    if mode == "project":
                        h_2d = h_2d.project(*free_axes)
                    
                    # Skip slices that are not requested
                    if not args.slice and mode == "slice":
                        continue

                    series_dict = {
                        f"{var_name} {met_type}": {
                            "data": h_2d,
                            "style": {"label": f"{var_name} {met_type}"},
                        }
                    }

                    p = (
                        HEPPlotter()
                        .set_plot_config(
                            figsize=(15, 14),
                            lumitext=f"{met_type}{extra_label}     {year} (13.6 TeV)",
                            lumitext_font_size=20,
                            cmstext_font_size=20,
                        )
                        .set_options(legend=False, cbar_log=True)
                        .set_output(output)
                        .set_data(series_dict, plot_type="2d")
                        .set_labels(x_labels[it], y_label)
                    )
                    plotters.append(p)

    if args.workers > 1:
        print(f"Plotting 2d histograms in parallel in category {cat}")
        with Pool(args.workers) as pool:
            pool.map(helpers.run_plot, plotters)
    else:
        for p in plotters:
            print(
                f"Plotting 2d histogram for {p.output_base.split('/')[-1]} in category {cat}"
            )
            p.run()


def plot_1d_response_histograms(hists_dict, cat, year):
    """
    Plot 1D histograms of variables for every combination of bins.
    For N-D histograms all combinations of the extra (non-y) axes are sliced,
    producing one 1-D plot per combination.
    Profile objects are skipped here (they are handled in plot_2d).

    Parameters
    ----------
    hists_dict : dict
        Histogram dictionary from `create_responses_info`.
    cat : str
        Category name.
    year : str
        Year string for labeling.
    """
    plotters = []

    for var_name in hists_dict:
        hist_1d_dict = {}
        var_label, _ = helpers.extract_labels(
            var_name, response_var_name_dict, BIN_VARIABLES
        )

        for met_type in hists_dict[var_name]:
            hist = hists_dict[var_name][met_type]

            # Skip profile objects — they are 2-D plots, not 1-D distributions
            if helpers.is_profile(hist):
                continue

            # The y-variable axis is always the last one.
            # Slice over every combination of the binning axes (all but last).
            y_axis_name = hist.axes[-1].name
            free_axes = [y_axis_name]

            for (
                slice_dict,
                label_parts,
                suffix,
                mode,
            ) in helpers.iter_slice_combinations(hist, free_axes):
                # Apply the slice_dict: integer entries fix a bin, slice(None) sums the axis.
                # Boost-histogram / Hist supports this mixed indexing natively:
                h_out = (
                    hist[{k: v for k, v in slice_dict.items()}] if slice_dict else hist
                )

                if mode == "project":
                    h_out = h_out.project(y_axis_name)

                # Skip slices that are not requested
                if not args.slice and mode == "slice":
                    continue

                # Build a unique key for this bin combination
                bin_name = suffix if suffix else "inclusive"

                if bin_name not in hist_1d_dict:
                    hist_1d_dict[bin_name] = {}

                extra_label = ",  ".join(label_parts)
                out_suffix = f"_{suffix}" if suffix else ""
                output_name = (
                    f"{histograms_1d_slice_dir}/{cat}_{var_name}{out_suffix}"
                    if mode == "slice"
                    else f"{histograms_1d_project_dir}/{cat}_{var_name}{out_suffix}"
                )

                hist_1d_dict[bin_name][met_type] = {
                    "data": h_out,
                    "style": hist.style,
                }
                hist_1d_dict[bin_name]["infos"] = {
                    "output_name": output_name,
                    "lumitext_prefix": extra_label,
                }

        for bin_name, hist_met_dict in hist_1d_dict.items():
            lumitext_prefix = hist_met_dict["infos"]["lumitext_prefix"]
            output_name = hist_met_dict["infos"]["output_name"]
            hist_met_dict.pop("infos")

            p = (
                HEPPlotter()
                .set_plot_config(
                    figsize=(14, 14),
                    lumitext=f"{lumitext_prefix}      {year} (13.6 TeV)",
                    lumitext_font_size=20,
                    cmstext_font_size=20,
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
            pool.map(helpers.run_plot, plotters)
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
        p = (
            HEPPlotter()
            .set_plot_config(figsize=(14, 14), lumitext=f"{year} (13.6 TeV)")
            .set_output(info["output_base"])
            .set_labels(info["xlabel"], info["ylabel"], ratio_label=info["ratio_label"])
            .set_options(
                y_log=info["y_log"],
                set_ylim=info.get("set_ylim", True),
                split_legend=False,
                set_ylim_ratio=0.5,
            )
            .set_data(info["series_dict"], plot_type="1d")
        )
        plotters.append(p)
    if args.workers > 1:
        print(f"Plotting MET histograms in parallel")
        with Pool(args.workers) as pool:
            pool.map(helpers.run_plot, plotters)
    else:
        for p in plotters:
            print(f"Plotting MET histogram  {p.output_base.split('/')[-1]}")
            p.run()


def do_plots(responses_dict, hists_dict, met_dict, category, year):
    """Plot all the required plots.

    Parameters
    ----------
    responses_dict : dict
        Response summary information from `create_responses_info`.
    hists_dict : dict
        Histogram dictionary from `create_responses_info`.
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
    plot_responses(responses_dict, category, year, hists_dict)
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
        print("Finished plotting!")

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
            y = helpers.extract_year_tag(dataset)

            year = ", ".join([year, y]) if year else y

        print(f"Total datasets found: {total_datasets_list}")
        print(f"Year: {year}")

        for category, col_var in cat_col.items():
            responses_dict = {}
            hists_dict = {}
            if MASK:
                # define a mask for the events
                mask = col_var["ll_pt"] > 100
                for var in col_var:
                    col_var[var] = col_var[var][mask]

            print(f"Processing category: {category}")
            bin_var_arrays = [col_var[bin_var] for bin_var in BIN_VARIABLES.keys()]

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
            new_responses_dict, new_hists_dict = create_responses_info(
                bin_var_arrays, u_dict, weights, list(BIN_VARIABLES.keys())
            )
            responses_dict |= new_responses_dict
            hists_dict |= new_hists_dict

            # build MET histograms
            met_dict = create_met_histos(col_var, category)

            # save all the histograms
            helpers.save_dict_to_file(
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
