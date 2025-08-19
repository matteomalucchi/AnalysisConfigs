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

import matplotlib

matplotlib.rcParams["agg.path.chunksize"] = 10000  # or try 5000, depending on size

if not args.output:
    args.output = "plots_DNN_data_and_mc"


if args.test:
    args.workers = 1
    args.output = "test"

NUMBER_OF_BINS = 20
PAD_VALUE = -999
BLIND_VALUE = 0.96

input_dir_data = os.path.dirname(args.input_data[0])

log_scale = not args.linear
outputdir =args.output + f"_{args.normalisation}"
# outputdir = os.path.join(input_dir_data, args.output) + f"_{args.normalisation}"


# To mix categories with Run2 and SPANet, put first the Run2 category
# because first the name of the variables is try with the Run2 string
# and after without it
# First region: data 4b (blinded)
# Second region: mc 4b (unblinded)
# Third region: data 2b reweighted (unblinded)
cat_dict = {}
if args.run2:
    cat_dict[f"CR{args.region_suffix}Run2"] = [
        f"4b{args.region_suffix}_control_regionRun2",
        f"4b{args.region_suffix}_control_regionRun2",
        f"2b{args.region_suffix}_control_region_postWRun2",
    ]
    # cat_dict[f"SR{args.region_suffix}_blindRun2"] = [f"4b{args.region_suffix}_signal_region_blindRun2", f"4b{args.region_suffix}_signal_regionRun2", f"2b{args.region_suffix}_signal_region_postWRun2"]
    cat_dict[f"SR{args.region_suffix}Run2"] = [
        f"4b{args.region_suffix}_signal_regionRun2",
        f"4b{args.region_suffix}_signal_regionRun2",
        f"2b{args.region_suffix}_signal_region_postWRun2",
    ]
else:
    cat_dict[f"CR{args.region_suffix}"] = [
        f"4b{args.region_suffix}_control_region",
        f"4b{args.region_suffix}_control_region",
        f"2b{args.region_suffix}_control_region_postW",
        # f"2b{args.region_suffix}_control_region_preW"
    ]
    # f"SR{args.region_suffix}_blind": [f"4b{args.region_suffix}_signal_region_blind", f"4b{args.region_suffix}_signal_region", f"2b{args.region_suffix}_signal_region_postW"],
    cat_dict[f"SR{args.region_suffix}"] = [
        f"4b{args.region_suffix}_signal_region",
        f"4b{args.region_suffix}_signal_region",
        f"2b{args.region_suffix}_signal_region_postW",
    ]
    # f"2b{args.region_suffix}_signal_region_preW"
    #    f"CR{args.region_suffix}_2b_Run2SPANet": [f"2b{args.region_suffix}_control_region_preWRun2", f"2b{args.region_suffix}_control_region_preW"],
    #    f"CR{args.region_suffix}_4b_Run2SPANet": [f"4b{args.region_suffix}_control_regionRun2", f"4b{args.region_suffix}_control_region"],

if args.test:
    if args.run2:
        cat_dict = {
            f"SR{args.region_suffix}Run2": [
                f"4b{args.region_suffix}_signal_regionRun2",
                f"4b{args.region_suffix}_signal_regionRun2",
                f"2b{args.region_suffix}_signal_region_postWRun2",
            ]
        }
    else:
        cat_dict = {
            f"SR{args.region_suffix}": [
                f"4b{args.region_suffix}_signal_region",
                f"4b{args.region_suffix}_signal_region",
                f"2b{args.region_suffix}_signal_region_postW",
            ]
        }


## Load the onnx model
if args.onnx_model:
    (
        model_session_SIG_BKG_DNN,
        input_name_SIG_BKG_DNN,
        output_name_SIG_BKG_DNN,
    ) = get_model_session(args.onnx_model, "SIG_BKG_DNN")
    # load the variables for the DNN
    # get the list name from the string args.input_variables
    dnn_variables = getattr(dnn_input_variables, args.input_variables)
    dnn_input_list = get_DNN_input_list(args.run2, dnn_variables)
    print(f"Input list for DNN: {dnn_input_list}")


color_list_orig = [
    ("black",),
    ("red",),
    ("blue", "dodgerblue"),
]
color_list_alt = [("purple",), ("darkorange", "orange"), ("green",)]


if not os.path.exists(outputdir):
    os.makedirs(outputdir)

if args.input_data[0].endswith(".coffea"):
    inputfiles_data = args.input_data
else:
    # get list of coffea files
    inputfiles_data = [
        os.path.join(input_dir_data, file)
        for file in os.listdir(input_dir_data)
        if file.endswith(".coffea") and "DATA" in file
    ]

inputfiles_mc = args.input_mc

filter_lambda = (
    (
        lambda x: (
            "weight" in x
            or ("score" in x and ("Run2" in x if args.run2 else "Run2" not in x))
        )
    )
    if not args.onnx_model
    else None
)

## Collecting MC dataset
cat_col_mc, total_datasets_list_mc = get_columns_from_files(
    inputfiles_mc, filter_lambda
)

## Collecting DATA dataset
cat_col_data, total_datasets_list_data = get_columns_from_files(
    inputfiles_data, filter_lambda
)


def plot_single_var_from_columns(
    var,
    col_dict,
    weight_dict,
    cat_list,
    dir_cat,
    chi_squared=True,
    color_list=color_list_orig,
    ratio2b4b=1,
    lumi=5.79,
    era_string="22 E",
):
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=[13, 13],
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )
    weights_plotted = False
    print(var)

    var_plot_name = var.replace("Run2", "")

    plotdict = {}

    # the range for the score can be inclusively all the range.
    # I still have to define it this way to make up for different ranges of blinded histograms.
    # range_4b is basically a global variable. It is calculated only once.
    # Same as bin_edges. Both depend on the MC signal
    if "UNIFORM" in var:
        bin_edges = np.linspace(0, 1, NUMBER_OF_BINS + 1)
    else:
        bin_edges = weighted_quantile(
            col_dict[cat_list[1]]["MC"],
            np.linspace(0, 1, NUMBER_OF_BINS + 1),
            weights=weight_dict[cat_list[1]]["MC"],
        )
    print(f"bin_edges {bin_edges}")

    # this is the mc data that I want to add to the histogram of the reweighted data
    mc_signal = f"{cat_list[1]}_MC"
    for i, cat in enumerate(cat_list):
        # I only want the following columns:
        # postW data
        # signal data
        # signal MC
        for data_mc in ["MC", "DATA"]:
            # we dont need the reweighted MC region
            if data_mc == "MC" and "postW" in cat:
                continue
            if data_mc == "MC" and "blind" in cat:
                continue
            if i == 1 and data_mc == "DATA":
                continue

            weights_den = weight_dict[cat][data_mc]
            weights_num = weight_dict[cat_list[0]][data_mc]

            col_den = col_dict[cat][data_mc]
            col_num = col_dict[cat_list[0]][data_mc]

            # remove padded values
            weights_den = weights_den[col_den != PAD_VALUE]
            weights_num = weights_num[col_num != PAD_VALUE]
            col_den = col_den[col_den != PAD_VALUE]
            col_num = col_num[col_num != PAD_VALUE]

            norm_factor_den = weights_num.sum() / weights_den.sum()
            print("Difference between the different normings")
            print(norm_factor_den)
            print(ratio2b4b)

            print(f"weights_den {weights_den}", type(weights_den))
            print(f"weights_num {weights_num}")
            print(f"col_num {col_num}", type(col_num))
            print(f"col_den {col_den}")

            print("bin_edges", bin_edges, len(bin_edges))
            bins_center = (bin_edges[1:] + bin_edges[:-1]) / 2
            print("bins_center", bins_center, len(bins_center))

            if "TRANSFORM" in var:
                bin_edges_plotting = np.linspace(
                    bin_edges[0], bin_edges[-1], NUMBER_OF_BINS + 1
                )
                bins_center_plotting = (
                    bin_edges_plotting[1:] + bin_edges_plotting[:-1]
                ) / 2
            else:
                bin_edges_plotting = bin_edges
                bins_center_plotting = bins_center

            print(f"Found something to plot {cat}_{data_mc}")
            # Here we save the MC and the bg reweighted
            # The signal was already plotted and we don't need it anymore
            plotdict[f"{cat}_{data_mc}"] = {
                "bin_edges": bin_edges,
                "bins_center": bins_center,
                "bin_edges_plotting": bin_edges_plotting,
                "bins_center_plotting": bins_center_plotting,
                "color": color_list[i],
                "col_den": col_den,
                "col_num": col_num,
                "weights_den": weights_den,
                "weights_num": weights_num,
            }
            del col_den, col_num

    # print(plotdict)
    for region, values in plotdict.items():
        print(f"Plotting region {region}")
        # Trying to add up the reweighted data from 2b with the MC signal
        # MC is still plotted independently.

        # Essentially the idea is:
        # -> events are appended with np.concatenate()
        # -> histograms are added binwise with +
        namesuffix = ""
        if "postW" in region:
            # Applying reweighting weight to 2b reweighted signal
            values["weights_den"] = values["weights_den"] * ratio2b4b
            values["col_den_onlybg"] = values["col_den"]
            values["weights_den_onlybg"] = values["weights_den"]
            values["col_den"] = np.concatenate(
                (values["col_den"], plotdict[mc_signal]["col_den"])
            )
            values["weights_den"] = np.concatenate(
                (values["weights_den"], plotdict[mc_signal]["weights_den"])
            )

            idx_den_onlybg = np.digitize(values["col_den_onlybg"], values["bin_edges"])
            values["h_den_onlybg"] = []
            values["err_den_onlybg"] = []
            for j in range(1, len(values["bin_edges"])):
                values["h_den_onlybg"].append(
                    np.sum(values["weights_den_onlybg"][idx_den_onlybg == j])
                )
                values["err_den_onlybg"].append(
                    np.sqrt(
                        np.sum(values["weights_den_onlybg"][idx_den_onlybg == j] ** 2)
                    )
                )
            values["h_den_onlybg"] = np.array(values["h_den_onlybg"])
            values["err_den_onlybg"] = np.array(values["err_den_onlybg"])

        if "postW" in region or "MC" in region:
            kl = (
                os.path.basename(inputfiles_mc[0])
                .split("kl-")[-1]
                .split("_")[0]
                .replace("p", ".")
            )
            if "postW" in region:
                namesuffix = r" + $\kappa_\lambda$=" + kl
                savesuffix = f"plus_kl_{kl}"
            if "MC" in region:
                namesuffix = r" ($\kappa_\lambda$=" + kl + ")"
                savesuffix = f"kl_{kl}"

        cat_plot_name = plot_regions_names(region, namesuffix).replace("Run2", "_DHH")
        # Filling the histograms
        idx_den = np.digitize(values["col_den"], values["bin_edges"])
        idx_num = np.digitize(values["col_num"], values["bin_edges"])
        # print(f"bin_edges {values['bin_edges']}")
        # print(len(values["col_den"]))
        # print(len(values["weights_den"]))
        # print("idx_den", idx_den, len(idx_den))
        # print("idx_num", idx_num, len(idx_num))

        values["h_den"] = []
        values["h_num"] = []
        values["err_den"] = []
        values["err_num"] = []

        for j in range(1, len(values["bin_edges"])):
            values["h_den"].append(np.sum(values["weights_den"][idx_den == j]))
            values["h_num"].append(np.sum(values["weights_num"][idx_num == j]))
            values["err_den"].append(
                np.sqrt(np.sum(values["weights_den"][idx_den == j] ** 2))
            )
            values["err_num"].append(
                np.sqrt(np.sum(values["weights_num"][idx_num == j] ** 2))
            )
            # print(
            #     'values["weights_den"][idx_den == j]',
            #     values["weights_den"][idx_den == j],
            # )

        values["h_den"] = np.array(values["h_den"])
        values["h_num"] = np.array(values["h_num"])
        values["err_den"] = np.array(values["err_den"])
        values["err_num"] = np.array(values["err_num"])

        # print("h_den", values["h_den"], len(values["h_den"]))
        # print("h_num", values["h_num"], len(values["h_num"]))
        # print("err_den", values["err_den"])
        # print("err_num", values["err_num"])

        ratio = values["h_num"] / values["h_den"]
        ratio_err = np.sqrt(
            (values["err_num"] / values["h_den"]) ** 2
            + (values["h_num"] * values["err_den"] / values["h_den"] ** 2) ** 2
        )
        print("These are ratio and ratio error")
        print(ratio[0])
        print(ratio_err)
        print(values["bin_edges"])

        # Thanks chatGPT. Try to mask the bins, where both edges are above BLIND_VALUE:
        if not "control" in region:
            mask_blind = ~(
                (bin_edges[:-1] > BLIND_VALUE) & (bin_edges[1:] > BLIND_VALUE)
            )
        else:
            mask_blind = ~(
                (bin_edges[:-1] > 1) & (bin_edges[1:] > 1)
            )  # hacked - so not blinded

        # Reference dataset (data in 4b)
        if not "postW" in region and "DATA" in region:
            print("Found signal region DATA")
            ratio = values["h_num"][mask_blind] / values["h_den"][mask_blind]
            ratio_err = values["err_num"][mask_blind] / values["h_num"][mask_blind]
            # print("ratio_err", ratio_err)

            ax.errorbar(
                values["bins_center_plotting"][mask_blind],
                values["h_den"][mask_blind],
                yerr=values["err_den"][mask_blind],
                label=cat_plot_name,
                color=values["color"][0],
                fmt=".",
            )
            ax_ratio.axhline(y=1, color=values["color"][0], linestyle="--")
            ax_ratio.fill_between(
                values["bins_center_plotting"][mask_blind],
                1 - ratio_err,
                1 + ratio_err,
                color="grey",
                alpha=0.5,
            )
        else:
            chi2_norm = None
            if "postW" in region and chi_squared:
                # compute the chi square between the two histograms (divide by the error on data)
                chi2_value = np.sum(
                    (
                        (values["h_den"][mask_blind] - values["h_num"][mask_blind])
                        / np.where(
                            values["err_num"][mask_blind] == 0,
                            1,
                            values["err_num"][mask_blind],
                        )
                    )
                    ** 2
                )
                ndof = len(values["h_den"][mask_blind]) - 1
                chi2_norm = chi2_value / ndof
                pvalue = chi2.sf(chi2_value, ndof)

                ax.text(
                    0.05,
                    0.95 - 0.05,
                    r"$\chi^2$/ndof= {:.1f},".format(chi2_norm)
                    + f"  p-value= {pvalue:.2f} (blind)",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    color=values["color"][0],
                    fontsize=20,
                )
                # Calculating binwise s/sqrt(b).
                # our background (reweighted 2b contains the signal at this point.
                # Therefore the function needs to be:
                # s/np.sqrt(bg-s) with s being the MC_signal and bg being the reweighted data
                # Assuming the mc did already go through
                s = plotdict[mc_signal]["h_den"]
                b = values["h_den"]
                s_err = plotdict[mc_signal]["err_den"]
                b_err = values["err_den"]
                sob_list = s / np.sqrt(b - s)
                sob = np.sqrt(np.sum(sob_list**2))

                dds = -(s - 2 * b) * (2 * (b - s) ** (-3 / 2))  # derivative d(sob)/ds
                ddb = -(s / 2) * (b - s) ** (-3 / 2)  # derivative d(sob)/db
                sob_err_list = np.sqrt((dds * s_err) ** 2 + (ddb * b_err) ** 2)
                sob_err_sq = np.sum((sob_list * sob_err_list / sob) ** 2)
                sob_err = np.sqrt(sob_err_sq)
                print("====== S/B list bin-by-bin: =======")
                print(sob_list)
                print("Errors also as a list")
                print(sob_err_list)
                print("S/B and errors combined")
                print(f"sob: {sob}, error: {sob_err}")

                ########## Easier approach #############
                # Calculating binwise s/sqrt(b).h_den_onlybg
                # our background (reweighted 2b contains the signal at this point.
                # Therefore the function needs to be:
                # s/np.sqrt(bg-s) with s being the MC_signal and bg being the reweighted data
                # Assuming the mc did already go through
                s = plotdict[mc_signal]["h_den"]
                b = values["h_den_onlybg"]
                s_err = plotdict[mc_signal]["err_den"]
                b_err = values["err_den_onlybg"]
                sob_list = s / np.sqrt(b)
                sob = np.sqrt(np.sum(sob_list**2))

                dds = 1 / np.sqrt(b)  # derivative d(sob)/ds
                ddb = -(s / 2) * (b) ** (-3 / 2)  # derivative d(sob)/db
                sob_err_list = np.sqrt((dds * s_err) ** 2 + (ddb * b_err) ** 2)
                sob_err_sq = np.sum((sob_list * sob_err_list / sob) ** 2)
                sob_err = np.sqrt(sob_err_sq)
                print("====== S/B list bin-by-bin ALTERNATIVE APPROACH: =======")
                print(sob_list)
                print("Errors also as a list")
                print(sob_err_list)
                print("S/B and errors combined")
                print(f"sob: {sob}, error: {sob_err}")

                ax.text(
                    0.05,
                    0.95 - 0.15,
                    r"$s/\sqrt{{{{b}}}}$ = {:.2f} $\pm$ {:.2f}".format(sob, sob_err),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    color="k",
                    fontsize=20,
                )

                for plot_var, plot_var_err, axis_name, plot_name in zip(
                    [sob_list, s, b],
                    [
                        sob_err_list,
                        s_err,
                        b_err,
                    ],
                    [r"$s/\sqrt{{{{b}}}}$", "sig", "bkg"],
                    ["sob", "sig", "bkg"],
                ):

                    # plot the sob for each bin
                    fig_events, ax_events = plt.subplots(figsize=[13, 13])
                    ax_events.errorbar(
                        values["bins_center_plotting"],
                        plot_var,
                        yerr=plot_var_err,
                        fmt=".",
                        label=cat_plot_name,
                        color=values["color"][0],
                    )
                    ax_events.fill_between(
                        values["bins_center_plotting"],
                        plot_var - plot_var_err,
                        plot_var + plot_var_err,
                        color="grey",
                        alpha=0.5,
                    )
                    ax_events.legend(loc="upper left")
                    ax_events.set_yscale("linear")
                    hep.cms.lumitext(
                        f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)",
                        ax=ax_events,
                    )
                    ax_events.set_xlabel(var_plot_name)
                    ax_events.set_ylabel(axis_name)
                    ax_events.grid()
                    fig_events.savefig(
                        os.path.join(dir_cat, f"{var}_{plot_name}.png"),
                        bbox_inches="tight",
                        dpi=300,
                    )
                    plt.close(fig_events)

                ax_ratio.errorbar(
                    values["bins_center_plotting"][mask_blind],
                    ratio[mask_blind],
                    yerr=ratio_err[mask_blind],
                    fmt=".",
                    label=cat_plot_name,
                    color=values["color"][0],
                )

                ## Save the histogram
                np.savez(
                    os.path.join(
                        dir_cat,
                        f"hist_columns_{region}_{var_plot_name}_{savesuffix}.npz".replace(
                            "Run2", "_DHH"
                        ),
                    ),
                    counts=np.append(values["h_den"], values["h_den"][-1]),
                    count_err=values["err_den"],
                    bin_edges=values["bin_edges_plotting"],
                    plot=var_plot_name,
                    num_events=len(values["col_den"]),
                )
                np.savez(
                    os.path.join(
                        dir_cat,
                        f"hist_columns_{region}_{var_plot_name}.npz".replace(
                            "Run2", "_DHH"
                        ),
                    ),
                    counts=np.append(
                        values["h_den_onlybg"], values["h_den_onlybg"][-1]
                    ),
                    count_err=values["err_den_onlybg"],
                    bin_edges=values["bin_edges_plotting"],
                    sob=sob_list,
                    sob_err=sob_err_list,
                    plot=var_plot_name,
                    num_events=len(values["col_den"]),
                )
            else:  # For mainly MC and 4b_data, as there we are not calculating the sob
                ## Save the histogram
                np.savez(
                    os.path.join(
                        dir_cat,
                        f"hist_columns_{region}_{var_plot_name}_{savesuffix}.npz".replace(
                            "Run2", "_DHH"
                        ),
                    ),
                    counts=np.append(values["h_den"], values["h_den"][-1]),
                    count_err=values["err_den"],
                    bin_edges=values["bin_edges_plotting"],
                    plot=var_plot_name,
                    num_events=len(values["col_den"]),
                )

            ## plot the histogram
            ax.step(
                values["bin_edges_plotting"],
                np.append(values["h_den"], values["h_den"][-1]),
                where="post",
                label=cat_plot_name,
                color=values["color"][0],
            )

            x0 = values["bin_edges_plotting"][0]
            x1 = values["bin_edges_plotting"][-1]
            y0 = values["h_den"][0]
            y1 = values["h_den"][-1]
            ax.plot(
                [x0, x0], [ax.get_ylim()[0], y0], color=values["color"][0]
            )  # first bin edge
            ax.plot(
                [x1, x1], [ax.get_ylim()[0], y1], color=values["color"][0]
            )  # last bin edge

            if len(values["color"]) > 1:
                ax.fill_between(
                    values["bin_edges_plotting"],
                    np.append(values["h_den"], values["h_den"][-1]),
                    step="post",
                    alpha=0.5,
                    color=values["color"][1],
                )

            # ax.hist(
            #     values["col_den"],
            #     bins=values["bin_edges_plotting"],
            #     histtype="step",
            #     label=cat_plot_name,
            #     weights=values["weights_den"],
            #     edgecolor=values["color"][0],
            #     facecolor=values["color"][1] if len(values["color"]) > 1 else None,
            #     fill=True if len(values["color"]) > 1 else False,
            #     alpha=0.5,
            # )

    del plotdict

    ax.legend(loc="upper right")
    ax.set_yscale("log" if log_scale else "linear")

    hep.cms.lumitext(f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax)

    ax_ratio.set_xlabel(var_plot_name)
    ax.set_ylabel("Events")
    ax_ratio.set_ylabel("Data/Pred.")

    ax.grid()
    ax_ratio.grid()
    ax_ratio.set_ylim(0.5, 1.5)
    ax.set_ylim(
        top=(1.3 * ax.get_ylim()[1] if not log_scale else ax.get_ylim()[1] ** 1.3)
    )
    print(f"Plotname: {os.path.join(dir_cat, f'{var}.png')}")
    fig.savefig(
        os.path.join(dir_cat, f"{var}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_from_columns(cat_cols, lumi, era_string):

    ## Calculating a ratio between the 2b-reweighted and the 4b region
    # Needs to be done here, as afterwards, we don't have access to all the regions anymore...
    # This will be calculated in the CR and also applied to SR. But in order to make sure that it makes sense, I also calculate both weights.
    print("Calculating ratios as weight from 2b-reweighted to 4b region")
    # HARDCODED:
    # - First region is CR
    # - In this region, 1st element is 4b, 3rd element is 2b-reweighted
    if args.normalisation == "sum_weights":
        op_norm = lambda x, y: sum(x) / sum(y)
    elif args.normalisation == "sum_weights":
        op_norm = lambda x, y: len(x) / len(y)
    else:
        raise ValueError(
            f"Unknown normalisation type {args.normalisation}. Use num_events or sum_weights"
        )

    if args.run2:
        if args.test:
            CRratio_4b_2bpostW_Run2 = 1
        else:
            CR_region_keys = cat_dict[f"CR{args.region_suffix}Run2"]
            CRratio_4b_2bpostW_Run2 = op_norm(
                cat_cols[0][CR_region_keys[0]]["weight"],
                cat_cols[0][CR_region_keys[2]]["weight"],
            )

        SR_region_keys = cat_dict[f"SR{args.region_suffix}Run2"]
        SRratio_4b_2bpostW_Run2 = op_norm(
            cat_cols[0][SR_region_keys[0]]["weight"],
            cat_cols[0][SR_region_keys[2]]["weight"],
        )

        print(f"CR ratio Run2: {CRratio_4b_2bpostW_Run2}")
        print(f"SR ratio Run2: {SRratio_4b_2bpostW_Run2}")
    else:
        if args.test:
            CRratio_4b_2bpostW = 1
        else:
            CR_region_keys = cat_dict[f"CR{args.region_suffix}"]
            print(CR_region_keys)

            CRratio_4b_2bpostW = op_norm(
                cat_cols[0][CR_region_keys[0]]["weight"],
                cat_cols[0][CR_region_keys[2]]["weight"],
            )

        SR_region_keys = cat_dict[f"SR{args.region_suffix}"]
        SRratio_4b_2bpostW = op_norm(
            cat_cols[0][SR_region_keys[0]]["weight"],
            cat_cols[0][SR_region_keys[2]]["weight"],
        )

        print(f"CR ratio: {CRratio_4b_2bpostW}")
        print(f"SR ratio: {SRratio_4b_2bpostW}")

    # cat_dict defined on top (global variable)
    for cats_name, cat_list in cat_dict.items():
        if "Run2SPANet" in cats_name:
            chi_squared = False
            color_list = color_list_alt
        else:
            chi_squared = True
            color_list = color_list_orig
        dir_cat = f"{outputdir}/{cats_name}_columns"
        if not os.path.exists(dir_cat):
            os.makedirs(dir_cat)
        # From here on we have to extract for signal an MC:
        # :param: data_mc means that we make the dictionary one longer such that for each category we save the data and the MC values.
        col_dict = {}
        for data_mc, cat_col in zip(["DATA", "MC"], cat_cols):
            print(data_mc)
            print(cat_col.keys())
            vars_tot = list(cat_col[cat_list[0]].keys())
            if args.test:
                vars_tot = vars_tot[:3]
            # print("vars_tot", vars_tot)
            vars_to_plot = []
            # vars_tot = [v for v in vars_tot if "add" in v or "weight"  in v]
            for v in vars_tot:
                if "_N" in v:
                    continue
                v_pref = v.split("_")[0]
                if v_pref + "_N" in vars_tot:
                    N = cat_col[cat_list[0]][v_pref + "_N"][0]
                    try:
                        assert (cat_col[cat_list[0]][v_pref + "_N"] == N).all()
                    except AssertionError:
                        print(
                            f"WARNING: Variables {v_pref} have different N values: {cat_col[cat_list[0]][v_pref + '_N']}. Skipping..."
                        )
                        # skip the variable
                        continue

                    for idx in range(N):
                        if not f"{v}_{idx}" in col_dict.keys():
                            col_dict[f"{v}_{idx}"] = {}
                        vars_to_plot.append(f"{v}_{idx}")
                        for cat in cat_list:
                            if "2b" in cat and data_mc == "MC":
                                print(f"Skipping {v} for {cat} in {data_mc}")
                                continue  # skip 2b MC
                            if not cat in col_dict[f"{v}_{idx}"].keys():
                                col_dict[f"{v}_{idx}"][cat] = {}
                            try:
                                col_dict[f"{v}_{idx}"][cat][data_mc] = cat_col[cat][v][
                                    np.arange(len(cat_col[cat][v])) % N == idx
                                ]
                            except KeyError:
                                col_dict[f"{v}_{idx}"][cat][data_mc] = cat_col[cat][
                                    v.replace("Run2", "")
                                ][
                                    np.arange(len(cat_col[cat][v.replace("Run2", "")]))
                                    % N
                                    == idx
                                ]
                else:
                    if not v in col_dict.keys():
                        col_dict[v] = {}
                    if v != "weight":
                        vars_to_plot.append(v)
                    for cat in cat_list:
                        if "2b" in cat and data_mc == "MC":
                            print(f"Skipping {v} for {cat} in {data_mc}")
                            continue  # skip 2b MC
                        if not cat in col_dict[v].keys():
                            col_dict[v][cat] = {}
                        try:
                            col_dict[v][cat][data_mc] = cat_col[cat][v]
                        except KeyError:
                            col_dict[v][cat][data_mc] = cat_col[cat][
                                v.replace("Run2", "")
                            ]
                        if v == "weight":
                            # Note that the total luminosity is hardcoded here for 2022postEE
                            col_dict[v][cat][data_mc] = col_dict[v][cat][data_mc] * (
                                float(lumi) / (5.79 + 17.6 + 2.88)
                                if data_mc == "MC"
                                else 1
                            )

            # compute the DNN score if onnx model is given
            if args.onnx_model:
                if any(["score" in v for v in vars_to_plot]):
                    print("Found score variables and onnx model")
                    print("The score will be overwritten by the onnx model")
                    # raise ValueError(
                    #     "onnx model and score variables are not compatible"
                    # )

                v = f"events_sig_bkg_dnn_score{'Run2' if args.run2 else ''}"
                if not v in col_dict.keys():
                    col_dict[v] = {}
                if not v in vars_to_plot:
                    vars_to_plot.append(v)

                for cat in cat_list:
                    if "2b" in cat and data_mc == "MC":
                        print(f"Skipping {v} for {cat} in {data_mc}")
                        continue  # skip 2b MC
                    if not cat in col_dict[v].keys():
                        col_dict[v][cat] = {}
                    # if data_mc in col_dict[v][cat].keys():
                    #     continue
                    input_variables_array = []
                    for input_var in dnn_input_list:
                        input_variables_array.append(
                            np.array(
                                col_dict[input_var][cat][data_mc], dtype=np.float32
                            )
                        )
                    input_variables_array = np.stack(input_variables_array, axis=-1)
                    inputs_complete = {input_name_SIG_BKG_DNN[0]: input_variables_array}
                    outputs = model_session_SIG_BKG_DNN.run(
                        output_name_SIG_BKG_DNN, inputs_complete
                    )
                    # print("events_sig_bkg_dnn_score", col_dict["events_sig_bkg_dnn_score"][cat][data_mc])
                    col_dict[v][cat][data_mc] = outputs[0][:, -1]
                    # print("outputs", cat, data_mc, outputs[0].shape, outputs[0])
                    del input_variables_array, inputs_complete, outputs

        vars_to_plot_final = vars_to_plot.copy()
        vars_to_plot_final += [f"{v}_TRANSFORM" for v in vars_to_plot if "score" in v]
        vars_to_plot_final += [f"{v}_UNIFORM" for v in vars_to_plot if "score" in v]
        vars_to_plot_final = [v for v in vars_to_plot_final if "score" in v]

        print("col_dict", col_dict)
        print("vars_to_plot_final", vars_to_plot_final)

        with Pool(args.workers) as p:
            print(f"Category name: {cats_name}")
            if "SR" in cats_name:
                ratio2b4b = (
                    SRratio_4b_2bpostW_Run2
                    if "Run2" in cats_name
                    else SRratio_4b_2bpostW
                )
            else:
                ratio2b4b = (
                    CRratio_4b_2bpostW_Run2
                    if "Run2" in cats_name
                    else CRratio_4b_2bpostW
                )

            p.starmap(
                plot_single_var_from_columns,
                [
                    (
                        var,
                        col_dict[var.replace("_TRANSFORM", "").replace("_UNIFORM", "")],
                        col_dict["weight"],
                        cat_list,
                        dir_cat,
                        chi_squared,
                        color_list,
                        ratio2b4b,
                        lumi,
                        era_string,
                    )
                    for var in vars_to_plot_final
                ],
            )
        del col_dict


if __name__ == "__main__":

    print("cat_col_data")
    for key, value in cat_col_data.items():
        print(key, value.keys())

    ## Generating the lumi and era_string for the plots:
    lumi, era_string = get_era_lumi(total_datasets_list_data)

    ############# Actual plotting command. Now a list with [datastuff, mcstuff] ######################
    plot_from_columns(
        [cat_col_data, cat_col_mc],
        lumi,
        era_string,
    )

    print(f"\nPlots saved in {outputdir}")
