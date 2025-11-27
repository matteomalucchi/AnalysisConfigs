import os
import sys
import numpy as np
from multiprocessing import Pool
from hist import Hist


import configs.HH4b_common.dnn_input_variables as dnn_input_variables
from utils.inference_session_onnx import get_model_session
from utils.get_DNN_input_list import get_DNN_input_list

from utils.plot.get_era_lumi import get_era_lumi
from utils.plot.get_columns_from_files import get_columns_from_files
from utils.plot.weighted_quantile import weighted_quantile
from utils.plot.plot_names import plot_regions_names
from utils.plot.args_plot import args

from utils.plot.HEPPlotter import HEPPlotter


if not args.output:
    if not args.test:
        args.output = "plots_2bVS4b"
    else:
        args.output = "test_2bVS4b"
        
NUMBER_OF_BINS = 20
PAD_VALUE = -999
BLIND_VALUE = 0.9
ARCTANH_BINS = False
VARIABLES_TEST = ["score", "weight", "prob"]
DEBUG = args.test

input_dir = os.path.dirname(args.input_data[0])
log_scale = not args.linear
outputdir = args.output + f"_{args.normalisation}"
# outputdir = os.path.join(input_dir, args.output) + f"_{args.normalisation}"


# To mix categories with Run2 and SPANet, put first the Run2 category
# because first the name of the variables is try with the Run2 string
# and after without it
cat_dict = {}

# The first category is the one that will be used as the reference in the ratio
if args.run2:
    cat_dict |= {
        f"CR{args.region_suffix}Run2": [
            [
                f"4b{args.region_suffix}_control_regionRun2",
                f"2b{args.region_suffix}_control_region_postWRun2",
                f"2b{args.region_suffix}_control_region_preWRun2",
            ]
        ],
        # f"SR{args.region_suffix}_blindRun2": [
        #     f"4b{args.region_suffix}_signal_region_blindRun2",
        #     f"2b{args.region_suffix}_signal_region_postW_blindRun2",
        #     f"2b{args.region_suffix}_signal_region_preW_blindRun2",
        # ],
        f"SR{args.region_suffix}Run2": [
            [
                f"4b{args.region_suffix}_signal_regionRun2",
                f"2b{args.region_suffix}_signal_region_postWRun2",
                f"2b{args.region_suffix}_signal_region_preWRun2",
            ]
        ],
        f"SR{args.region_suffix}Run2_SPREAD": [
            [
                f"2b{args.region_suffix}_signal_region_postWRun2",
                f"2b{args.region_suffix}_signal_region_postWRun2_SPREAD",
            ]
        ],
    }
else:
    cat_dict |= {
        f"CR{args.region_suffix}": [
            [
                f"4b{args.region_suffix}_control_region",
                f"2b{args.region_suffix}_control_region_postW",
                f"2b{args.region_suffix}_control_region_preW",
            ]
        ],
        f"SR{args.region_suffix}": [
            [
                f"4b{args.region_suffix}_signal_region",
                f"2b{args.region_suffix}_signal_region_postW",
                f"2b{args.region_suffix}_signal_region_preW",
            ]
        ],
        # f"SR{args.region_suffix}_blind": [
        #     f"4b{args.region_suffix}_signal_region_blind",
        #     f"2b{args.region_suffix}_signal_region_postW_blind",
        #     f"2b{args.region_suffix}_signal_region_preW_blind",
        # ],
        #
        # Special case for the 2b morphed with the spread of the morphing weights
        # Keyword: "SPREAD"
        #
        f"SR{args.region_suffix}_SPREAD": [
            [
                f"2b{args.region_suffix}_signal_region_postW",
                f"2b{args.region_suffix}_signal_region_postW_SPREAD",
            ]
        ],
        # f"CR{args.region_suffix}_2b_Run2SPANet": [f"2b{args.region_suffix}_control_region_preWRun2", f"2b{args.region_suffix}_control_region_preW"],
        # f"CR{args.region_suffix}_4b_Run2SPANet": [f"4b{args.region_suffix}_control_regionRun2", f"4b{args.region_suffix}_control_region"],
    }

if args.comparison:
    # Compare distributions for DATA and MC
    # Keyword: "DATAMC"
    cat_dict |= {
        f"SR{args.region_suffix}_4b_Run2SPANet_DATAMC": [
            [
                f"2b{args.region_suffix}_signal_region_postWRun2",
                f"4b{args.region_suffix}_signal_regionRun2_MC",
            ],
            [
                f"2b{args.region_suffix}_signal_region_postW",
                f"4b{args.region_suffix}_signal_region_MC",
            ],
        ],
        f"SR{args.region_suffix}_4b_DATAMC": [
            [
                f"4b{args.region_suffix}_signal_region",
                f"2b{args.region_suffix}_signal_region_postW",
                f"4b{args.region_suffix}_signal_region_MC",
            ],
        ],
    }

if args.test:
    cat_dict = {
        f"CR": [
            [
                f"4b_control_region",
                f"2b_control_region_postW",
                f"2b_control_region_preW",
            ]
        ],
        f"SR{args.region_suffix}_SPREAD": [
            [
                f"2b{args.region_suffix}_signal_region_postW",
                f"2b{args.region_suffix}_signal_region_postW_SPREAD",
            ]
        ],
    }


color_list_orig = [[("black",), ("blue", "dodgerblue"), ("red",)]]
color_list_spread = [[("red", "red")] + [("green",)] * 20 + [("orange",)] + [("blue",)]]
color_list_alt = [[("purple",), ("darkorange", "orange"), ("green",)]]
color_list_DATAMC = [
    [("red",), ("darkorange",), ("purple",)],
    [("blue",), ("dodgerblue",)],
    [("green",), ("limegreen",)],
]

## Load the onnx model
if args.onnx_model:
    (
        model_session_SIG_BKG_DNN,
        input_name_SIG_BKG_DNN,
        output_name_SIG_BKG_DNN,
    ) = get_model_session(args.onnx_model, "SIG_BKG_DNN")
    # load the variables for the DNN
    dnn_variables = getattr(dnn_input_variables, args.input_variables)
    dnn_input_list = get_DNN_input_list(args.run2, dnn_variables)
    print(f"Input list for DNN: {dnn_input_list}")


if not os.path.exists(outputdir):
    os.makedirs(outputdir)

# load the data
if args.input_data[0].endswith(".coffea"):
    inputfiles = args.input_data
else:
    # get list of coffea files
    inputfiles = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.endswith(".coffea") and "DATA" in file
    ]

filter_lambda = (lambda x: ("weight" in x or "score" in x)) if args.spread else None
cat_col_data, total_datasets_list = get_columns_from_files(inputfiles, "nominal", filter_lambda, debug=False, novars=args.novars)

cat_col_mc = None
if args.input_mc:
    if args.input_mc[0].endswith(".coffea"):
        inputfiles_mc = args.input_mc
    else:
        # get list of coffea files
        inputfiles_mc = [
            os.path.join(input_dir, file)
            for file in os.listdir(input_dir)
            if file.endswith(".coffea") and "DATA" not in file
        ]

    cat_col_mc, _ = get_columns_from_files(inputfiles_mc, "nominal", filter_lambda, debug=False, novars=args.novars)

    if args.run2:
        cols_sig_mc = cat_col_mc[f"4b{args.region_suffix}_signal_regionRun2"]
    else:
        cols_sig_mc = cat_col_mc[f"4b{args.region_suffix}_signal_region"]

    if args.input_mc[0].endswith(".coffea") and any(
        ["score" in col for col in cols_sig_mc]
    ):
        CONST_SIG_BINNING = True if len(inputfiles_mc) == 1 else False
    else:
        CONST_SIG_BINNING = False

    for col in cols_sig_mc:
        print(col)
        if "score" in col:
            print("\n WEIGHTED")
            CONSTANT_SIGNAL_BINS = weighted_quantile(
                cols_sig_mc[col],
                np.linspace(0, 1, NUMBER_OF_BINS + 1),
                weights=cols_sig_mc["weight"],
            )
            CONSTANT_SIGNAL_BLIND_BINS = CONSTANT_SIGNAL_BINS[
                CONSTANT_SIGNAL_BINS < BLIND_VALUE
            ]
            print(f"Constant signal bins: {CONSTANT_SIGNAL_BINS}")
            score_hist = np.histogram(
                cols_sig_mc[col],
                bins=CONSTANT_SIGNAL_BINS,
                weights=cols_sig_mc["weight"],
            )
            print(f"Constant signal bins histogram: {score_hist}")
else:
    CONST_SIG_BINNING = False


def plot_weights(weights_list, suffix, lumi, era_string):

    hist_1d_dict = {}
    mean_std_list = []
    for i, weights in enumerate(weights_list):
        var_name = f"Morphing weights" + (f" {i}" if len(weights_list) > 1 else "")
        hist_w = Hist.new.Var(
            np.logspace(-3, 2, 100),
            name=var_name,
            flow=False,
        ).Double()
        hist_w.fill(weights)
        hist_1d_dict[var_name] = {
            "data": hist_w,
        }
        mean_std_list.append(
            f"mean: {np.mean(weights):.2f}, std: {np.std(weights):.2f}"
        )

    output_base = os.path.join(outputdir, f"weights_{suffix}")

    p = (
        HEPPlotter()
        .set_plot_config(
            lumitext=f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)",
            figsize=[13, 13],
        )
        .set_output(output_base)
        .set_labels(
            "Morphing weights",
            "Events",
        )
        .set_options(y_log=True, x_log=True, set_ylim=False, legend=False)
        .set_data(hist_1d_dict, plot_type="1d")
        .add_annotation(
            x=0.05, y=0.95, s="\n".join(mean_std_list), fontsize=20, ha="left", va="top"
        )
        .run()
    )


def get_median_hist(
    hist_1d_dict, histos_spread, bins, var, k, color_list, cat_plot_name_alt
):
    
    h_median_spread = np.median([h.values() for h in histos_spread], axis=0)

    histo_median_spread = Hist.new.Var(
        bins,
        name=var,
        flow=False,
    ).Double()
    histo_median_spread.fill(bins[:-1], weight=h_median_spread)

    hist_1d_dict[cat_plot_name_alt] = {
        "data": histo_median_spread,
        "style": {
            "histtype_ratio": "step",
            "color": color_list[k][-2][0],
            "linewidth": 2,
            "label": cat_plot_name_alt,
            "appear_in_legend_ratio": False,
            "edges_ratio": False,
            "plot_errors": False,
        },
    }
    return hist_1d_dict


def get_median_quantiles(hist_1d_dict, histos_spread, bins, var, k, color_list):
    # plot the 16th and 84th percentiles of the spread
    for h in hist_1d_dict.values():
        is_ref = h.get("style", {}).get("is_reference", False)
        if is_ref:
            hist_ref = h["data"]
            break

    ratios_spread = []
    for h in histos_spread:
        ratios_spread.append(h.values() / hist_ref.values())

    hist_1d_ratio_dict = {}
    for idx, quantile in enumerate([16, 84]):
        ratio_quantile = np.percentile(ratios_spread, quantile, axis=0)
        ratio_quantile_histo = Hist.new.Var(
            bins,
            name=var,
            flow=False,
        ).Double()
        ratio_quantile_histo.fill(bins[:-1], weight=ratio_quantile)

        hist_1d_ratio_dict[f"ratio_{quantile}"] = {
            "data": ratio_quantile_histo,
            "style": {
                "histtype_ratio": "step",
                "color": color_list[k][-1][0],
                "linewidth": 2,
                "legend_name_ratio": (r"$\pm \sigma_{k-folds}$" if idx == 0 else None),
                "appear_in_legend_ratio": True if quantile == 16 else False,
                "edges_ratio": False,
                "plot_errors": False,
            },
        }
    return hist_1d_ratio_dict


def plot_single_var_from_columns(
    var,
    col_dict,
    weight_dict,
    cats_name,
    cat_lists,
    dir_cat,
    chi_squared,
    color_list,
    lumi,
    era_string,
):
    histos_spread = []

    for k, cat_list in enumerate(cat_lists):
        hist_1d_dict = {}
        for i, cat in enumerate(cat_list):

            if "SPREAD" in cats_name:
                if "SPREAD" in cat and i == 1:
                    cat_plot_name = plot_regions_names(cat, " (k-folds)")
                elif "SPREAD" in cat:
                    cat_plot_name = cat
                else:
                    cat_plot_name = plot_regions_names(cat, " (mean weight per event)")
                cat_plot_name_alt = plot_regions_names(cat, " (median per bin)")
            else:
                if "MC" in cat:
                    kl = (
                        os.path.basename(inputfiles_mc[0])
                        .split("kl-")[-1]
                        .split("_")[0]
                        .replace("p", ".")
                    )
                    namesuffix = r" ($\kappa_\lambda$=" + kl + ")"
                else:
                    namesuffix = ""
                cat_plot_name = plot_regions_names(cat, namesuffix)

            weights_den = weight_dict[cat]
            weights_num = weight_dict[cat_list[0]]

            col_den = col_dict[cat]
            col_num = col_dict[cat_list[0]]

            # remove padded values
            weights_den = weights_den[col_den != PAD_VALUE]
            weights_num = weights_num[col_num != PAD_VALUE]
            col_den = col_den[col_den != PAD_VALUE]
            col_num = col_num[col_num != PAD_VALUE]

            # WARNING: the weights are different for the additional jets because
            # the number of events is different since it's computed after the masking of the PAD_VALUE
            if args.normalisation == "num_events":
                norm_factor_den = len(weights_den) / len(weights_num)
                norm_factor_num = 1.0
            elif args.normalisation == "sum_weights":
                norm_factor_den = weights_num.sum() / weights_den.sum()
                norm_factor_num = 1.0
            elif args.normalisation == "density":
                norm_factor_den = 1 / weights_den.sum()
                norm_factor_num = 1 / weights_num.sum()
            else:
                raise ValueError(
                    f"Unknown normalisation type {args.normalisation}. Use num_events or sum_weights"
                )

            print(
                f"Plotting from columns {var} for {cat} with norm {norm_factor_den} and weights sum {weights_den.sum()}"
            )

            # normalize the weights
            weights_den = weights_den * norm_factor_den
            weights_num = weights_num * norm_factor_num

            # fix the bins and range
            if "score" in var and CONST_SIG_BINNING:
                bins = (
                    CONSTANT_SIGNAL_BINS
                    if "blind" not in cat
                    else CONSTANT_SIGNAL_BLIND_BINS
                )
            elif type(col_num[0]) == np.int64:
                # this is a categorical variable, use the number of bins
                bins = np.arange(
                    col_num.min() - 0.5,
                    col_num.max() + 1.5,
                    1,
                )
            else:
                # compute the range of the 4b category considering the 0.1% and 99.9% quantile
                range_4b = (
                    tuple(np.quantile(col_den, [0.001, 0.999])) if i == 0 else range_4b
                )

                mask_num_range4b = (col_num >= range_4b[0]) & (col_num <= range_4b[1])
                weights_num = weights_num[mask_num_range4b]
                col_num = col_num[mask_num_range4b]

                mask_den_range4b = (col_den >= range_4b[0]) & (col_den <= range_4b[1])
                weights_den = weights_den[mask_den_range4b]
                col_den = col_den[mask_den_range4b]

                bins = np.linspace(range_4b[0], range_4b[1], NUMBER_OF_BINS + 1)

                if ARCTANH_BINS:
                    # transform the bins to arctanh space
                    bins = np.arctanh(np.linspace(-0.1, 0.999, NUMBER_OF_BINS + 1))

            if "TRANSFORM" in var:
                bins = np.linspace(bins[0], bins[-1], NUMBER_OF_BINS + 1)

            histo = Hist.new.Var(bins, name=var, flow=False).Weight()
            histo.fill(col_den, weight=weights_den)

            # Store into dict expected by cms_plotter
            if "SPREAD" in cats_name:
                # spread histograms
                if i == 0:
                    legend_name_ratio = r"$\pm \sigma_{stat}$"
                else:
                    legend_name_ratio = None
                histos_spread.append(histo)
                style_dict = {
                    "is_reference": (i == 0),
                    "histtype": "errorbar" if i == 0 else "step",
                    "histtype_ratio": "step",
                    "color": color_list[k][i][0],
                    "appear_in_legend": True if i < 2 else False,
                    "appear_in_legend_ratio": True if i < 1 else False,
                    "legend_name_ratio": legend_name_ratio,
                    "edges_ratio": False,
                    "plot_errors": True if i == 0 else False,
                    "linewidth": 1,
                }
            else:
                if len(color_list[k][i]) > 1:
                    style_dict = {
                        "is_reference": (i == 0),
                        "histtype": "fill",
                        "edgecolor": color_list[k][i][0],
                        "facecolor": color_list[k][i][1],
                    }
                else:
                    style_dict = {
                        "is_reference": (i == 0),
                        "histtype": "errorbar" if i == 0 else "step",
                        "color": color_list[k][i][0],
                    }

            hist_1d_dict[cat_plot_name] = {
                "data": histo,
                "style": style_dict,
            }

            del col_den, col_num

    if "SPREAD" in cats_name:
        # plot the median of the spread
        hist_1d_dict = get_median_hist(
            hist_1d_dict.copy(),
            histos_spread,
            bins,
            var,
            k,
            color_list,
            cat_plot_name_alt,
        )

        # plot the 16th and 84th percentiles of the spread
        hist_1d_ratio_dict = get_median_quantiles(
            hist_1d_dict, histos_spread, bins, var, k, color_list
        )

        p = (
            HEPPlotter(debug=DEBUG)
            .set_plot_config(
                lumitext=f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)",
                figsize=[13, 13],
            )
            .set_output(os.path.join(dir_cat, f"{var}"))
            .set_labels(
                var.replace("Run2", ""),
                "Events",
            )
            .set_options(
                y_log=log_scale,
                reference_to_den=False,
                legend_ratio=True,
                set_ylim=False,
            )
            .set_data(hist_1d_dict, plot_type="1d")
            .add_ratio_hists(hist_1d_ratio_dict)
        )

    else:
        p = (
            HEPPlotter(debug=DEBUG)
            .set_plot_config(
                lumitext=f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)",
                figsize=[13, 13],
            )
            .set_output(os.path.join(dir_cat, f"{var}"))
            .set_labels(
                var.replace("Run2", ""),
                "Events",
                ratio_label="Data/Pred." if not "DATAMC" in cats_name else "Bkg/Sig",
            )
            .set_options(
                y_log=log_scale,
                reference_to_den=False,
            )
            .set_data(hist_1d_dict, plot_type="1d")
        )
        if chi_squared:
            p = p.add_chi_square()

    p.run()


def main(cat_cols, lumi, era_string):

    print(f"CATEGORIES ARE:")
    print(cat_dict)
    cat_col_DATA = cat_cols[0]
    cat_col_MC = cat_cols[1]

    col_dict = {}
    for cats_name, cat_lists in cat_dict.items():
        plot_categories = False
        cat_lists_final = []
        for cat_list in cat_lists:
            print(f"categories being analysed {cats_name}", cat_list)
            if args.spread:
                if "SPREAD" not in cats_name:
                    continue

            chi_squared = True
            color_list = color_list_orig
            if "Run2SPANet" in cats_name:
                chi_squared = False
                color_list = color_list_alt
            if "DATAMC" in cats_name:
                chi_squared = False
                color_list = color_list_DATAMC
            if "SPREAD" in cats_name:
                chi_squared = False
                color_list = color_list_spread

            # check if the categories are in the accumulator
            try:
                for cat in cat_list:
                    cat_col_DATA[cat.replace("_MC", "").replace("_SPREAD", "")]
            except KeyError:
                print(
                    f"KeyError: {cat} not in {cat_col_DATA.keys()}, skipping {cats_name}"
                )
                continue

            vars_tot = list(cat_col_DATA[cat_list[0]].keys())
            vars_tot = [v for v in vars_tot if "year" not in v]
            if "SPREAD" in cats_name:
                vars_tot = [v for v in vars_tot if "weight" in v or "score" in v]

            if args.test:
                # vars_tot = vars_tot[:3]
                # vars_tot=[v for v in vars_tot if "prob" in v or "weight" in v]
                vars_tot = [
                    v
                    for v in vars_tot
                    if any(test_var in v for test_var in VARIABLES_TEST)
                ]

            print("vars_tot", vars_tot)

            vars_to_plot = []

            for v in vars_tot:
                var_name = v.replace("Run2", "")
                if "_N" in v:
                    continue
                v_pref = v.split("_")[0]
                if v_pref + "_N" in vars_tot:
                    N = cat_col_DATA[cat_list[0]][v_pref + "_N"][0]
                    try:
                        assert (cat_col_DATA[cat_list[0]][v_pref + "_N"] == N).all()
                    except AssertionError:
                        print(
                            f"WARNING: Variables {v_pref} have different N values: {cat_col_DATA[cat_list[0]][v_pref + '_N']}. Skipping..."
                        )
                        # skip the variable
                        continue

                    for idx in range(N):
                        if f"{var_name}_{idx}" not in col_dict:
                            col_dict[f"{var_name}_{idx}"] = {}
                        vars_to_plot.append(f"{var_name}_{idx}")
                        for cat in cat_list:
                            if "SPREAD" in cat:
                                continue
                            if cat in col_dict[f"{var_name}_{idx}"]:
                                continue
                            if "MC" in cat:
                                cat_col = cat_col_MC
                            else:
                                cat_col = cat_col_DATA

                            print(v, cat)
                            try:
                                col_dict[f"{var_name}_{idx}"][cat] = cat_col[
                                    cat.replace("_MC", "")
                                ][v][
                                    np.arange(len(cat_col[cat.replace("_MC", "")][v]))
                                    % N
                                    == idx
                                ]
                            except KeyError:
                                col_dict[f"{var_name}_{idx}"][cat] = cat_col[
                                    cat.replace("_MC", "")
                                ][var_name][
                                    np.arange(
                                        len(cat_col[cat.replace("_MC", "")][var_name])
                                    )
                                    % N
                                    == idx
                                ]
                else:
                    if var_name not in col_dict:
                        col_dict[var_name] = {}
                    if "weight" not in v:
                        vars_to_plot.append(var_name)
                    for cat in cat_list:
                        if "SPREAD" in cat:
                            continue
                        if cat in col_dict[var_name]:
                            continue
                        if "MC" in cat:
                            if "morph" in v:
                                continue
                            cat_col = cat_col_MC
                        else:
                            cat_col = cat_col_DATA

                        # swap the dict keys
                        print(v, cat)
                        try:
                            col_dict[var_name][cat] = cat_col[cat.replace("_MC", "")][v]
                        except KeyError:
                            col_dict[var_name][cat] = cat_col[cat.replace("_MC", "")][
                                var_name
                            ]

            cat_list_final = cat_list.copy()

            # compute the DNN score if onnx model is given
            if args.onnx_model:
                v = f"events_sig_bkg_dnn_score"  # {'Run2' if args.run2 else ''}"
                if not v in col_dict.keys():
                    col_dict[v] = {}
                if not v in vars_to_plot:
                    vars_to_plot.append(v)

                for cat in cat_list:
                    if "SPREAD" in cat:
                        continue
                    if not cat in col_dict[v].keys():
                        col_dict[v][cat] = {}
                    else:
                        continue
                    input_variables_array = []
                    for input_var in dnn_input_list:
                        input_variables_array.append(
                            np.array(
                                col_dict[input_var.replace("Run2", "")][cat],
                                dtype=np.float32,
                            )
                        )
                    input_variables_array = np.stack(input_variables_array, axis=-1)
                    inputs_complete = {input_name_SIG_BKG_DNN[0]: input_variables_array}
                    print("inputs_complete", inputs_complete)
                    outputs = model_session_SIG_BKG_DNN.run(
                        output_name_SIG_BKG_DNN, inputs_complete
                    )
                    # print("events_sig_bkg_dnn_score", col_dict["events_sig_bkg_dnn_score"][cat])
                    col_dict[v][cat] = outputs[0][:, -1]
                    print("outputs", cat, outputs[0].shape, outputs[0])
                    del input_variables_array, inputs_complete, outputs

            # if args.run2:
            #     var_SPREAD = "events_bkg_morphing_spread_dnn_weightsRun2"
            # else:
            var_SPREAD = "events_bkg_morphing_spread_dnn_weights"

            try:
                for cat in cat_list:
                    if "SPREAD" in cat:
                        for i in range(len(col_dict[var_SPREAD][cat_list_final[0]][0])):
                            for v in vars_tot:
                                if "score" in v:
                                    col_dict[v.replace("Run2", "")][f"{cat}_{i}"] = (
                                        col_dict[v][cat_list_final[0]]
                                    )

                            col_dict["weight"][f"{cat}_{i}"] = col_dict[var_SPREAD][
                                cat_list_final[0]
                            ][:, i]

                            cat_list_final.append(f"{cat}_{i}")

                            if cat in cat_list_final:
                                # remove the original cat
                                cat_list_final.remove(cat)
                # col_dict[var_SPREAD]
            except KeyError:
                print(f"{var_SPREAD} not in {col_dict.keys()}, skipping")
                continue

            dir_cat = f"{outputdir}/{cats_name}_columns"
            if not os.path.exists(dir_cat):
                os.makedirs(dir_cat)

            vars_to_plot += [f"{v}_TRANSFORM" for v in vars_to_plot if "score" in v]

            cat_lists_final.append(cat_list_final)

            plot_categories = True

        if plot_categories:
            if args.workers > 1:
                with Pool(args.workers) as p:
                    p.starmap(
                        plot_single_var_from_columns,
                        [
                            (
                                var,
                                col_dict[var.replace("_TRANSFORM", "")],
                                col_dict["weight"],
                                cats_name,
                                cat_lists_final,
                                dir_cat,
                                chi_squared,
                                color_list,
                                lumi,
                                era_string,
                            )
                            for var in vars_to_plot
                        ],
                    )
            else:
                for var in vars_to_plot:
                    plot_single_var_from_columns(
                        var,
                        col_dict[var.replace("_TRANSFORM", "")],
                        col_dict["weight"],
                        cats_name,
                        cat_lists_final,
                        dir_cat,
                        chi_squared,
                        color_list,
                        lumi,
                        era_string,
                    )


if __name__ == "__main__":
            
    lumi, era_string = get_era_lumi(total_datasets_list)

    # plot the weights
    for category in cat_col_data.keys():
        if "postW" in category:
            weights = cat_col_data[category]["weight"]
            plot_weights([weights], category, lumi, era_string)

    main([cat_col_data, cat_col_mc], lumi, era_string)

    print(f"\nPlots saved in {outputdir}")
