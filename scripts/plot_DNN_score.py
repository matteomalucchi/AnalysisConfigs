import os
import sys
import numpy as np
import awkward as ak
from hist import Hist
from multiprocessing import Pool

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
        args.output = "plots_DNN_data_and_mc"
    else:
        args.output = "test_DNN_data_and_mc"

NUMBER_OF_BINS = 20
PAD_VALUE = -999
BLINDED_SIGNAL_FRACTION = 0.2

input_dir_data = os.path.dirname(args.input_data[0])

log_scale = not args.linear
outputdir = args.output + f"_{args.normalisation}"
# outputdir = os.path.join(input_dir_data, args.output) + f"_{args.normalisation}"


# To mix categories with Run2 and SPANet, put first the Run2 category
# because first the name of the variables is try with the Run2 string
# and after without it
# First region: data 4b (blinded)
# Second region: data 2b reweighted (unblinded)
# Third region: mc 4b (unblinded)
cat_dict = {}
if args.run2:
    cat_dict[f"CR{args.region_suffix}Run2"] = [
        f"4b{args.region_suffix}_control_regionRun2",
        f"2b{args.region_suffix}_control_region_postWRun2",
        f"4b{args.region_suffix}_control_regionRun2",
    ]
    # cat_dict[f"SR{args.region_suffix}_blindRun2"] = [f"4b{args.region_suffix}_signal_region_blindRun2", f"4b{args.region_suffix}_signal_regionRun2", f"2b{args.region_suffix}_signal_region_postWRun2"]
    cat_dict[f"SR{args.region_suffix}Run2"] = [
        f"4b{args.region_suffix}_signal_regionRun2",
        f"2b{args.region_suffix}_signal_region_postWRun2",
        f"4b{args.region_suffix}_signal_regionRun2",
    ]
else:
    cat_dict[f"CR{args.region_suffix}"] = [
        f"4b{args.region_suffix}_control_region",
        f"2b{args.region_suffix}_control_region_postW",
        f"4b{args.region_suffix}_control_region",
        # f"2b{args.region_suffix}_control_region_preW"
    ]
    # f"SR{args.region_suffix}_blind": [f"4b{args.region_suffix}_signal_region_blind", f"4b{args.region_suffix}_signal_region", f"2b{args.region_suffix}_signal_region_postW"],
    cat_dict[f"SR{args.region_suffix}"] = [
        f"4b{args.region_suffix}_signal_region",
        f"2b{args.region_suffix}_signal_region_postW",
        f"4b{args.region_suffix}_signal_region",
    ]
    # f"2b{args.region_suffix}_signal_region_preW"
    #    f"CR{args.region_suffix}_2b_Run2SPANet": [f"2b{args.region_suffix}_control_region_preWRun2", f"2b{args.region_suffix}_control_region_preW"],
    #    f"CR{args.region_suffix}_4b_Run2SPANet": [f"4b{args.region_suffix}_control_regionRun2", f"4b{args.region_suffix}_control_region"],

if args.test:
    if args.run2:
        cat_dict = {
            f"SR{args.region_suffix}Run2": [
                f"4b{args.region_suffix}_signal_regionRun2",
                f"2b{args.region_suffix}_signal_region_postWRun2",
                f"4b{args.region_suffix}_signal_regionRun2",
            ]
        }
    else:
        cat_dict = {
            f"SR{args.region_suffix}": [
                f"4b{args.region_suffix}_signal_region",
                f"2b{args.region_suffix}_signal_region_postW",
                f"4b{args.region_suffix}_signal_region",
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
    ("blue", "dodgerblue"),
    ("red", "red"),
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
    inputfiles_mc,
    sel_var="nominal",
    filter_lambda=filter_lambda,
    debug=False,
    novars=args.novars,
)

## Collecting DATA dataset
cat_col_data, total_datasets_list_data = get_columns_from_files(
    inputfiles_data,
    sel_var="nominal",
    filter_lambda=filter_lambda,
    debug=False,
    novars=args.novars,
)


def compute_sob(hist_1d_dict):
    # Calculating binwise s/sqrt(b).
    # our background (reweighted 2b contains the signal at this point.
    # Therefore the function needs to be:
    # s/np.sqrt(bg-s) with s being the MC_signal and bg being the reweighted data
    # Assuming the mc did already go through
    # s = plotdict[mc_signal]["h_den"]
    # b = values["h_den"]
    # s_err = plotdict[mc_signal]["err_den"]
    # b_err = values["err_den"]
    # sob_list = s / np.sqrt(b - s)
    # sob = np.sqrt(np.sum(sob_list**2))

    # dds = -(s - 2 * b) * (2 * (b - s) ** (-3 / 2))  # derivative d(sob)/ds
    # ddb = -(s / 2) * (b - s) ** (-3 / 2)  # derivative d(sob)/db
    # sob_err_list = np.sqrt((dds * s_err) ** 2 + (ddb * b_err) ** 2)
    # sob_err_sq = np.sum((sob_list * sob_err_list / sob) ** 2)
    # sob_err = np.sqrt(sob_err_sq)
    # print("====== S/B list bin-by-bin: =======")
    # print(sob_list)
    # print("Errors also as a list")
    # print(sob_err_list)
    # print("S/B and errors combined")
    # print(f"sob: {sob}, error: {sob_err}")

    ########## Easier approach #############
    # Calculating binwise s/sqrt(b).h_den_onlybg
    # our background (reweighted 2b contains the signal at this point.
    # Therefore the function needs to be:
    # s/np.sqrt(bg-s) with s being the MC_signal and bg being the reweighted data
    # Assuming the mc did already go through
    for idx in range(len(hist_1d_dict["Sig+Bkg"]["data"])):
        if "DATA" in hist_1d_dict["Sig+Bkg"]["style"]["legend_name"][idx]:
            b_hist = hist_1d_dict["Sig+Bkg"]["data"][idx]
        else:
            s_hist = hist_1d_dict["Sig+Bkg"]["data"][idx]

    s = s_hist.values()
    b = b_hist.values()
    s_err = np.sqrt(s_hist.variances())
    b_err = np.sqrt(b_hist.variances())

    sob_list = s / np.sqrt(b)
    sob = np.sqrt(np.sum(sob_list**2))

    dds = 1 / np.sqrt(b)  # derivative d(sob)/ds
    ddb = -(s / 2) * (b) ** (-3 / 2)  # derivative d(sob)/db
    sob_err_list = np.sqrt((dds * s_err) ** 2 + (ddb * b_err) ** 2)
    sob_err_sq = np.sum((sob_list * sob_err_list / sob) ** 2)
    sob_err = np.sqrt(sob_err_sq)

    return sob, sob_err, sob_list, sob_err_list, s, s_err, b, b_err


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
    print(f"\n\nPlotting variable {var} in {dir_cat}")

    var_plot_name = var.replace("Run2", "")

    plotdict = {}

    mc_signal = ""
    for cat in cat_list:
        if "_MC" in cat:
            kl = (
                os.path.basename(inputfiles_mc[0])
                .split("kl-")[-1]
                .split("_")[0]
                .replace("p", ".")
            )
            namesuffix = r" ($\kappa_\lambda$=" + kl + ")"
            # mc_signal_region = plot_regions_names(cat, namesuffix)
            mc_signal = cat
            break
    if mc_signal == "":
        raise ValueError("No MC signal found in cat_list")

    # the range for the score can be inclusively all the range.
    # I still have to define it this way to make up for different ranges of blinded histograms.
    # range_4b is basically a global variable. It is calculated only once.
    # Same as bin_edges. Both depend on the MC signal
    if "UNIFORM" in var:
        bin_edges = np.linspace(0, 1, NUMBER_OF_BINS + 1)
    else:
        bin_edges = weighted_quantile(
            col_dict[mc_signal],
            np.linspace(0, 1, NUMBER_OF_BINS + 1),
            weights=weight_dict[mc_signal],
        )
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0

    hist_1d_dict = {}

    # this is the mc data that I want to add to the histogram of the reweighted data
    for i, cat in enumerate(cat_list):
        style_dict = {}
        col_den = col_dict[cat]
        col_den = col_den[col_den != PAD_VALUE]
        weights_den = weight_dict[cat]
        weights_den = weights_den[col_den != PAD_VALUE]
        if "DATA" in cat and "2b" in cat:

            weights_den = weights_den * ratio2b4b
        bins_center = (bin_edges[1:] + bin_edges[:-1]) / 2

        if "TRANSFORM" in var:
            bin_edges_plotting = np.linspace(
                bin_edges[0], bin_edges[-1], NUMBER_OF_BINS + 1
            )
        else:
            bin_edges_plotting = None

        namesuffix = ""
        if "MC" in cat:
            kl = (
                os.path.basename(inputfiles_mc[0])
                .split("kl-")[-1]
                .split("_")[0]
                .replace("p", ".")
            )
            namesuffix = r" ($\kappa_\lambda$=" + kl + ")"
            # savesuffix = f"kl_{kl}"

        cat_plot_name = plot_regions_names(cat, namesuffix).replace("Run2", "_DHH")
        print(cat_plot_name)

        print(f"Found something to plot {cat} -> {cat_plot_name}")

        # Mask the blinded region
        if "4b" in cat and "signal" in cat and "DATA" in cat:
            # check if the blinded fraction can divide the bin edges
            np.testing.assert_almost_equal(
                (len(bin_edges) - 1) * BLINDED_SIGNAL_FRACTION % 1,
                0,
                decimal=3,
                err_msg=f"BLINDED_SIGNAL_FRACTION {BLINDED_SIGNAL_FRACTION} cannot divide the bin edges of length {len(bin_edges)-1}",
            )

            blind_value = bin_edges[
                int((len(bin_edges) - 1) * (1 - BLINDED_SIGNAL_FRACTION))
            ]
            mask_blind = col_den <= blind_value
            
            col_den=col_den[mask_blind]
            weights_den=weights_den[mask_blind]
            
        # histogram of the denominator
        histo = Hist.new.Var(bin_edges, name=var_plot_name, flow=False).Weight()
        histo.fill(col_den, weight=weights_den)
        print(cat_plot_name)
        print(histo)

        if i == 0:
            hist_1d_dict[cat_plot_name] = {
                "data": histo,
                "style": {
                    "is_reference": (i == 0),
                    "histtype": "errorbar" if i == 0 else "step",
                    "color": color_list[i][0],
                    "bin_edges_plotting": bin_edges_plotting,
                },
            }

        else:
            if "Sig+Bkg" not in hist_1d_dict:

                hist_1d_dict["Sig+Bkg"] = {
                    "data": [],
                    "style": {
                        "is_reference": (i == 0),
                        "histtype": "fill",
                        "stack": i != 0,
                        "edgecolor": [],
                        "facecolor": [],
                        "alpha": [],
                        "legend_name": [],
                        "bin_edges_plotting": [],
                    },
                }
            hist_1d_dict["Sig+Bkg"]["data"].append(histo)
            hist_1d_dict["Sig+Bkg"]["style"]["edgecolor"].append(color_list[i][0])
            hist_1d_dict["Sig+Bkg"]["style"]["facecolor"].append(color_list[i][1])
            hist_1d_dict["Sig+Bkg"]["style"]["alpha"].append(
                0.5 if "DATA" in cat else 1
            )
            hist_1d_dict["Sig+Bkg"]["style"]["legend_name"].append(cat_plot_name)
            hist_1d_dict["Sig+Bkg"]["style"]["bin_edges_plotting"].append(
                bin_edges_plotting
            )
    print(f"bin_edges {bin_edges}")

    sob, sob_err, sob_list, sob_err_list, s, s_err, b, b_err = compute_sob(hist_1d_dict)
    sob_string = r"$s/\sqrt{{{{b}}}}$ = {:.3f} $\pm$ {:.3f}".format(sob, sob_err)

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
        graph_dict = {
            f"{plot_name}": {
                "data": {
                    "x": [bins_center, (bin_edges[1:] - bin_edges[:-1]) / 2],
                    "y": [plot_var, plot_var_err],
                },
                "style": {
                    "appear_in_legend": False,
                },
            }
        }

        p_sob = (
            HEPPlotter(debug=True)
            .set_plot_config(
                lumitext=f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)",
                figsize=[13, 13],
            )
            .set_output(os.path.join(dir_cat, f"{var}_{plot_name}"))
            .set_options(set_ylim=False)
            .set_labels(
                var.replace("Run2", ""),
                axis_name,
            )
            .set_data(
                graph_dict,
                plot_type="graph",
            )
        )
        p_sob = p_sob.add_annotation(x=0.05, y=0.95 - 0.15, s=sob_string, fontsize=20)
        p_sob.run()

    p = (
        HEPPlotter(debug=True)
        .set_plot_config(
            lumitext=f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)",
            figsize=[13, 13],
        )
        .set_output(os.path.join(dir_cat, f"{var}"))
        .set_labels(
            var.replace("Run2", ""),
            "Events",
            ratio_label="Data/Pred.",
        )
        .set_options(
            y_log=log_scale,
            reference_to_den=False,
        )
        .add_annotation(x=0.05, y=0.95 - 0.15, s=sob_string, fontsize=20)
        .set_data(hist_1d_dict, plot_type="1d")
    )

    if chi_squared:
        p = p.add_chi_square()

    p.run()

    # save the histogram
    # np.savez(
    #     os.path.join(dir_cat, f"hist_columns_{var_plot_name}_{savesuffix}.npz".replace("Run2", "_DHH")),
    #     counts=np.append(hist_1d_dict[mc_signal_region]["data"].values(), hist_1d_dict[mc_signal_region]["data"].values()[-1]),
    #     count_err=np.sqrt(hist_1d_dict[mc_signal_region]["data"].variances()),
    #     bin_edges=bin_edges,
    #     plot=var_plot_name,
    #     num_events=len(col_den),
    #     sob=sob_list,
    #     sob_err=sob_err_list,
    # )


def main(cat_cols, lumi, era_string):

    ## Calculating a ratio between the 2b-reweighted and the 4b region
    # Needs to be done here, as afterwards, we don't have access to all the regions anymore...
    # This will be calculated in the CR and also applied to SR. But in order to make sure that it makes sense, I also calculate both weights.
    print("Calculating ratios as weight from 2b-reweighted to 4b region")
    # HARDCODED:
    # - First region is CR
    # - In this region, 1st element is 4b, 3rd element is 2b-reweighted
    if args.normalisation == "sum_weights":
        op_norm = lambda x, y: sum(x) / sum(y)
    elif args.normalisation == "num_events":
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
                cat_cols[0][CR_region_keys[1]]["weight"],
            )

        SR_region_keys = cat_dict[f"SR{args.region_suffix}Run2"]
        SRratio_4b_2bpostW_Run2 = op_norm(
            cat_cols[0][SR_region_keys[0]]["weight"],
            cat_cols[0][SR_region_keys[1]]["weight"],
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
                cat_cols[0][CR_region_keys[1]]["weight"],
            )

        SR_region_keys = cat_dict[f"SR{args.region_suffix}"]
        SRratio_4b_2bpostW = op_norm(
            cat_cols[0][SR_region_keys[0]]["weight"],
            cat_cols[0][SR_region_keys[1]]["weight"],
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
        col_list_data_mc = []
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
                            cat_data_mc = f"{cat}_{data_mc}"
                            if "2b" in cat and data_mc == "MC":
                                print(f"Skipping {v} for {cat} in {data_mc}")
                                continue  # skip 2b MC
                            # if not cat in col_dict[f"{v}_{idx}"].keys():
                            #     col_dict[f"{v}_{idx}"][cat] = {}
                            try:
                                col_dict[f"{v}_{idx}"][cat_data_mc] = cat_col[cat][v][
                                    np.arange(len(cat_col[cat][v])) % N == idx
                                ]
                            except KeyError:
                                col_dict[f"{v}_{idx}"][cat_data_mc] = cat_col[cat][
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
                        cat_data_mc = f"{cat}_{data_mc}"
                        if "2b" in cat and data_mc == "MC":
                            print(f"Skipping {v} for {cat} in {data_mc}")
                            continue  # skip 2b MC
                        if not cat_data_mc in col_list_data_mc:
                            col_list_data_mc.append(cat_data_mc)
                        # if not cat in col_dict[v].keys():
                        #     col_dict[v][cat] = {}
                        try:
                            col_dict[v][cat_data_mc] = cat_col[cat][v]
                        except KeyError:
                            col_dict[v][cat_data_mc] = cat_col[cat][
                                v.replace("Run2", "")
                            ]
                        if v == "weight":
                            # Note that the total luminosity is hardcoded here for 2022postEE
                            col_dict[v][cat_data_mc] = col_dict[v][cat_data_mc] * (
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
                    cat_data_mc = f"{cat}_{data_mc}"
                    if "2b" in cat and data_mc == "MC":
                        print(f"Skipping {v} for {cat} in {data_mc}")
                        continue  # skip 2b MC
                    # if not cat in col_dict[v].keys():
                    # col_dict[v][cat] = {}
                    # if data_mc in col_dict[v][cat].keys():
                    #     continue
                    input_variables_array = []
                    for input_var in dnn_input_list:
                        input_variables_array.append(
                            np.array(col_dict[input_var][cat_data_mc], dtype=np.float32)
                        )
                    input_variables_array = np.stack(input_variables_array, axis=-1)
                    inputs_complete = {input_name_SIG_BKG_DNN[0]: input_variables_array}
                    outputs = model_session_SIG_BKG_DNN.run(
                        output_name_SIG_BKG_DNN, inputs_complete
                    )
                    # print("events_sig_bkg_dnn_score", col_dict["events_sig_bkg_dnn_score"][cat][data_mc])
                    col_dict[v][cat_data_mc] = outputs[0][:, -1]
                    # print("outputs", cat, data_mc, outputs[0].shape, outputs[0])
                    del input_variables_array, inputs_complete, outputs

        vars_to_plot_final = vars_to_plot.copy()
        vars_to_plot_final += [f"{v}_TRANSFORM" for v in vars_to_plot if "score" in v]
        vars_to_plot_final += [f"{v}_UNIFORM" for v in vars_to_plot if "score" in v]
        vars_to_plot_final = [v for v in vars_to_plot_final if "score" in v]

        print("col_dict", col_dict)
        print("vars_to_plot_final", vars_to_plot_final)

        print(f"Category name: {cats_name}")
        if "SR" in cats_name:
            ratio2b4b = (
                SRratio_4b_2bpostW_Run2 if "Run2" in cats_name else SRratio_4b_2bpostW
            )
        else:
            ratio2b4b = (
                CRratio_4b_2bpostW_Run2 if "Run2" in cats_name else CRratio_4b_2bpostW
            )
        if args.workers > 1:
            with Pool(args.workers) as p:
                p.starmap(
                    plot_single_var_from_columns,
                    [
                        (
                            var,
                            col_dict[
                                var.replace("_TRANSFORM", "").replace("_UNIFORM", "")
                            ],
                            col_dict["weight"],
                            col_list_data_mc,
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
        else:
            for var in vars_to_plot_final:
                plot_single_var_from_columns(
                    var,
                    col_dict[var.replace("_TRANSFORM", "").replace("_UNIFORM", "")],
                    col_dict["weight"],
                    col_list_data_mc,
                    dir_cat,
                    chi_squared,
                    color_list,
                    ratio2b4b,
                    lumi,
                    era_string,
                )
        del col_dict


if __name__ == "__main__":

    print("cat_col_data")
    for key, value in cat_col_data.items():
        print(key, value.keys())

    ## Generating the lumi and era_string for the plots:
    lumi, era_string = get_era_lumi(total_datasets_list_data)

    ############# Actual plotting command. Now a list with [datastuff, mcstuff] ######################
    main(
        [cat_col_data, cat_col_mc],
        lumi,
        era_string,
    )

    print(f"\nPlots saved in {outputdir}")
