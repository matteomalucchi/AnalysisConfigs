import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats.distributions import chi2
import mplhep as hep
from multiprocessing import Pool

import configs.HH4b_common.dnn_input_variables as dnn_input_variables
from utils.inference_session_onnx import get_model_session
from utils.get_DNN_input_list import get_DNN_input_list

from utils.plot.get_era_lumi import get_era_lumi
from utils.plot.get_columns_from_files import get_columns_from_files
from utils.plot.weighted_quantile import weighted_quantile
from utils.plot.plot_names import plot_regions_names
from utils.plot.args_plot import args

hep.style.use("CMS")

if not args.output:
    args.output = "plots_2bVS4b"

if args.test:
    # args.workers = 1
    args.output = "test"

NUMBER_OF_BINS = 20
PAD_VALUE = -999
BLIND_VALUE = 0.9
ARCTANH_BINS=False
VARIABLES_TEST=["score", "weight", "prob"]


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
    }

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


color_list_orig = [[("black",), ("blue", "dodgerblue"), ("red",)]]
color_list_spread = [
    [("green", "red")] + [("green",)] * 20 + [("orange",)] + [("blue",)]
]
color_list_alt = [[("purple",), ("darkorange", "orange"), ("green",)]]
color_list_DATAMC = [
    [("red",), ("darkorange",), ("purple",)],
    [("blue",), ("dodgerblue",)],
    [("green",), ("limegreen",)],
]


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
cat_col_data, total_datasets_list = get_columns_from_files(inputfiles, filter_lambda)

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
        
    cat_col_mc, _ = get_columns_from_files(inputfiles_mc, filter_lambda)

    if args.run2:
        cols_sig_mc = cat_col_mc[f"4b{args.region_suffix}_signal_regionRun2"]
    else:
        cols_sig_mc = cat_col_mc[f"4b{args.region_suffix}_signal_region"]
        
    if args.input_mc[0].endswith(".coffea") and any(["score" in col for col in cols_sig_mc]):
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
    fig, ax = plt.subplots(figsize=[13, 13])
    for i, weights in enumerate(weights_list):
        ax.hist(
            weights,
            bins=np.logspace(-3, 2, 100),
            histtype="step",
            label="Morphing weights "
            + (f"{i}" if len(weights_list) > 1 else "")
            + "\nmean: {:.2f}\nstd: {:.2f}".format(np.mean(weights), np.std(weights)),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("Morphing weights")
    ax.set_ylabel("Events")

    hep.cms.lumitext(f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax)

    fig.savefig(os.path.join(outputdir, f"weights_{suffix}.png"))
    plt.close(fig)


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
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=[13, 13],
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    print(var, cat_lists)
    ratios_spread = []
    histos_spread = []

    min_histo_value = 1e10

    for k, cat_list in enumerate(cat_lists):
        for i, cat in enumerate(cat_list):
            print("cat", cat)

            if "SPREAD" in cats_name:
                if "SPREAD" in cat:
                    cat_plot_name = plot_regions_names(cat, " (k-folds)")
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
                print("categorical variable")
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

                print(f"range_4b {range_4b}")

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

                # print(f"weights_den {weights_den}", type(weights_den))
                # print(f"weights_num {weights_num}")
                # print(f"col_num {col_num}", type(col_num))
                # print(f"col_den {col_den}")

            idx_den = np.digitize(col_den, bins)
            idx_num = np.digitize(col_num, bins)

            if "TRANSFORM" in var:
                bins = np.linspace(bins[0], bins[-1], NUMBER_OF_BINS + 1)
            bins_center = (bins[1:] + bins[:-1]) / 2

            h_den = []
            h_num = []
            err_den = []
            err_num = []

            for j in range(1, len(bins)):
                h_den.append(np.sum(weights_den[idx_den == j]))
                h_num.append(np.sum(weights_num[idx_num == j]))
                err_den.append(np.sqrt(np.sum(weights_den[idx_den == j] ** 2)))
                err_num.append(np.sqrt(np.sum(weights_num[idx_num == j] ** 2)))
                # print('weights_den[idx_den == j]', weights_den[idx_den == j])

            h_den = np.array(h_den)
            h_num = np.array(h_num)
            err_den = np.array(err_den)
            err_num = np.array(err_num)

            # print("h_den", h_den, len(h_den))
            # print("h_num", h_num, len(h_num))
            # print("err_den", err_den)
            # print("err_num", err_num)

            chi2_norm = None
            if i > 0 and chi_squared:
                # compute the chi square between the two histograms (divide by the error on data)
                chi2s = ((h_den - h_num) / np.where(err_num == 0, 1, err_num)) ** 2
                chi2_value = np.sum(chi2s)
                ndof = len(h_den) - 1
                chi2_norm = chi2_value / ndof
                pvalue = chi2.sf(chi2_value, ndof)
                print("chi2", chi2s, chi2_value, ndof, chi2_norm)

            ratio = h_num / h_den

            print("ratio", ratio)
            if i == 0:
                ratio_err = err_num / h_num
                ax.errorbar(
                    bins_center,
                    h_den,
                    yerr=err_den,
                    label=cat_plot_name,
                    color=color_list[k][i][0],
                    fmt=".",
                    zorder=10,
                )

                ax_ratio.axhline(y=1, color="black", linestyle="--")
                if "SPREAD" in cats_name:
                    ax_ratio.step(
                        bins,
                        1 - np.append(ratio_err, ratio_err[-1]),
                        where="post",
                        color=color_list[k][i][1],
                        label=r"$\pm \sigma_{stat}$",
                        linewidth=2,
                    )
                    ax_ratio.step(
                        bins,
                        1 + np.append(ratio_err, ratio_err[-1]),
                        where="post",
                        color=color_list[k][i][1],
                        linewidth=2,
                    )
                else:
                    ax_ratio.fill_between(
                        bins_center,
                        1 - ratio_err,
                        1 + ratio_err,
                        color=color_list[k][i][0],
                        alpha=0.2,
                    )

            else:
                ratio_err = np.sqrt(
                    (err_num / h_den) ** 2 + (h_num * err_den / h_den**2) ** 2
                )

                ## plot the histogram
                ax.step(
                    bins,
                    np.append(h_den, h_den[-1]),
                    where="post",
                    label=cat_plot_name if "SPREAD" not in cat or i == 1 else None,
                    color=color_list[k][i][0],
                    linewidth=2,
                    zorder=0,
                )

                # plot the first and last bin edges
                x0 = bins[0]
                x1 = bins[-1]
                y0 = h_den[0]
                y1 = h_den[-1]
                ax.plot(
                    [x0, x0], [1e-10, y0], color=color_list[k][i][0], linewidth=2,
                )  # first bin edge
                ax.plot(
                    [x1, x1], [1e-10, y1], color=color_list[k][i][0],linewidth=2,
                )  # last bin edge

                # get the minimum value of the histogram
                min_histo_value = min(min_histo_value, h_den.min())

                if len(color_list[k][i]) > 1:
                    ax.fill_between(
                        bins,
                        np.append(h_den, h_den[-1]),
                        step="post",
                        alpha=0.5,
                        color=color_list[k][i][1],
                    )

                if "SPREAD" in cat:
                    histos_spread.append(h_den)
                    # plot the spread of the DNN score as histogram in the ratio
                    ax_ratio.step(
                        bins,
                        np.append(ratio, ratio[-1]),
                        where="post",
                        color=color_list[k][i][0],
                        # linestyle="--",
                        linewidth=1,
                        zorder=0,
                    )
                    ratios_spread.append(ratio)
                else:
                    ax_ratio.errorbar(
                        bins_center,
                        ratio,
                        yerr=ratio_err,
                        fmt=".",
                        color=color_list[k][i][0],
                    )

            if chi2_norm:
                ax.text(
                    0.05,
                    0.95 - 0.05 * i,
                    r"$\chi^2$/ndof= {:.1f},".format(chi2_norm)
                    + f"  p-value= {pvalue:.2f}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    color=color_list[k][i][0],
                    fontsize=20,
                )

            del col_den, col_num

    if "SPREAD" in cats_name:
        # plot the median of the spread
        h_median_spread = np.median(histos_spread, axis=0)
        ax.step(
            bins,
            np.append(h_median_spread, h_median_spread[-1]),
            where="post",
            color=color_list[k][-2][0],
            linewidth=1,
            label=cat_plot_name_alt,
            zorder=1,
        )
        x0 = bins[0]
        x1 = bins[-1]
        y0 = h_median_spread[0]
        y1 = h_median_spread[-1]
        ax.plot(
            [x0, x0], [ax.get_ylim()[0], y0], color=color_list[k][-2][0]
        )  # first bin edge
        ax.plot(
            [x1, x1], [ax.get_ylim()[0], y1], color=color_list[k][-2][0]
        )  # last bin edge

        if len(color_list[k][-2]) > 1:
            ax.fill_between(
                bins,
                np.append(h_median_spread, h_median_spread[-1]),
                step="post",
                alpha=0.5,
                color=color_list[k][-2][1],
            )

        ratio_median = h_num / h_median_spread
        ax_ratio.step(
            bins,
            np.append(ratio_median, ratio_median[-1]),
            where="post",
            color=color_list[k][-2][0],
            linewidth=2,
        )

        # plot the 16th and 84th percentiles of the spread
        ratio_16 = np.percentile(ratios_spread, 16, axis=0)
        ratio_84 = np.percentile(ratios_spread, 84, axis=0)
        ax_ratio.step(
            bins,
            np.append(ratio_16, ratio_16[-1]),
            where="post",
            color=color_list[k][-1][0],
            linewidth=1,
            label=r"$\pm \sigma_{k-folds}$",
        )
        ax_ratio.step(
            bins,
            np.append(ratio_84, ratio_84[-1]),
            where="post",
            color=color_list[k][-1][0],
            linewidth=1,
        )

    ax.legend(loc="upper right")
    ax_ratio.legend(loc="upper left")
    ax.set_yscale("log" if log_scale else "linear")

    hep.cms.lumitext(f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax)

    var_plot_name = var.replace("Run2", "")
    ax_ratio.set_xlabel(var_plot_name)
    ax.set_ylabel("Events")
    ax_ratio.set_ylabel("Data/Pred." if not "DATAMC" in cats_name else "Bkg/Sig")

    ax.grid()
    ax_ratio.grid()
    if "SPREAD" in cat:
        ax_ratio.set_ylim(0.75, 1.25)
    elif not "DATAMC" in cat:
        ax_ratio.set_ylim(0.5, 1.5)

    ax.set_ylim(
        top=(
            1.3 * ax.get_ylim()[1]
            if not log_scale
            else ax.get_ylim()[1] ** (1.3 if args.normalisation != "density" else -1.3)
        ),
        bottom=(
            min_histo_value * 0.9
            if not log_scale
            else min_histo_value ** (0.9 if args.normalisation != "density" else -0.9)
        ),
    )
    fig.savefig(
        os.path.join(dir_cat, f"{var}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_from_columns(cat_cols, lumi, era_string):

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
                vars_tot=[v for v in vars_tot if any(test_var in v for test_var in VARIABLES_TEST)]
                
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

            print("vars_to_plot", vars_to_plot)
            print("col_dict", col_dict)
            print("cat_list_final", cat_list_final)

            for col in col_dict:
                for cat in col_dict[col]:
                    print(
                        col,
                        cat,
                        col_dict[col][cat],
                        len(col_dict[col][cat]),
                    )
            cat_lists_final.append(cat_list_final)

            plot_categories = True

        if plot_categories:
            if args.workers>1:
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
        # del col_dict


if __name__ == "__main__":

    # print(cat_col_data)
    for k in cat_col_data.keys():
        for kk in cat_col_data[k].keys():
            print(k, kk, len(cat_col_data[k][kk]))
    lumi, era_string = get_era_lumi(total_datasets_list)

    # plot the weights
    for category in cat_col_data.keys():
        if "postW" in category:
            weights = cat_col_data[category]["weight"]
            plot_weights([weights], category, lumi, era_string)

    plot_from_columns([cat_col_data, cat_col_mc], lumi, era_string)

    print(f"\nPlots saved in {outputdir}")
