import os
import sys
import numpy as np
import awkward as ak
import importlib.util
from multiprocessing import Pool
from hist import Hist


import configs.HH4b_common.dnn_input_variables as dnn_input_variables
from utils_configs.inference_session_onnx import get_model_session
from utils_configs.get_DNN_input_list import get_DNN_input_list

from utils_configs.plot.get_era_lumi import get_era_lumi
from utils_configs.plot.get_columns_from_files import get_columns_from_files
from utils_configs.plot.weighted_quantile import weighted_quantile
from utils_configs.plot.plot_names import plot_regions_names
from utils_configs.plot.args_plot import args

from utils_configs.plot.HEPPlotter import HEPPlotter
from utils_configs.plot.morphing_model_comparison import (
    MODEL_LABELS,
    MODEL_SUFFIXES,
    compare_with_stored,
    evaluate_morphing_weight,
    model_suffix,
    plot_weight_comparison,
    strip_model_suffix,
)


if not args.output:
    if args.compare_morphing_models:
        # next to the models: <dir of the models>/plots_avg_vs_med_<normalisation>
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(args.compare_morphing_models[0])),
            "plots_avg_vs_med",
        )
    elif not args.test:
        args.output = "plots_2bVS4b"
    else:
        args.output = "test_2bVS4b"

NUMBER_OF_BINS = 20
PAD_VALUE = -999
BLIND_VALUE = 0.9
ARCTANH_BINS = False
# Columns that are a working-point index (integers stored as floats): one bin per index, the same in all the regions
WP_INDEX_VARIABLES = ("btagBBTXbb_morph_5wp", "btagBBTXbb_CDF_rescaled_WPs")
VARIABLES_TEST = ["score", "weight", "prob"]
DEBUG = args.test

# -i can be a directory (with or without trailing slash) or coffea files
if os.path.isdir(args.input_data[0]):
    input_dir = args.input_data[0]
else:
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
elif args.boosted:
    new_definition = True
    if new_definition:
        sub_regions = {
            "signal":    "boosted_signal_region_D",
            "sig_bkg_preW":  "boosted_signal_sideband_region_C",
            "sig_bkg_postW": "boosted_signal_sideband_region_C_postW",
            "control":    "boosted_control_region_B",
            "con_bkg_preW":  "boosted_control_sideband_region_A",
            "con_bkg_postW": "boosted_control_sideband_region_A_postW",
        }
        if args.morphing_direction == "btag":
            sub_regions["sig_bkg_preW"] = "boosted_control_region_B"
            sub_regions["sig_bkg_postW"] = "boosted_control_region_B_postW"
            sub_regions["control"] = "boosted_signal_sideband_region_C"
        cat_dict |= {
            f"CR_TT_{args.region_suffix}": [
                [
                    sub_regions["control"],
                    sub_regions["con_bkg_postW"],
                    sub_regions["con_bkg_preW"],
                    sub_regions["control"] + "_TT",
                ]
            ],
            f"SR_TT_{args.region_suffix}": [
                [
                    sub_regions["signal"],
                    sub_regions["sig_bkg_postW"],
                    sub_regions["sig_bkg_preW"],
                    sub_regions["signal"] + "_TT",
                ]
            ],
            f"CR{args.region_suffix}": [
                [
                    sub_regions["control"],
                    sub_regions["con_bkg_postW"],
                    sub_regions["con_bkg_preW"],
                ]
            ],
            f"SR{args.region_suffix}": [
                [
                    sub_regions["signal"],
                    sub_regions["sig_bkg_postW"],
                    sub_regions["sig_bkg_preW"],
                ]
            ]
        }
    else:
        sub_regions = {
            "signal":    "boosted_signal_region_A",
            "sig_bkg_preW":  "boosted_signal_sideband_region_B",
            "sig_bkg_postW": "boosted_signal_sideband_region_B_postW",
            "control":    "boosted_control_region_C",
            "con_bkg_preW":  "boosted_control_sideband_region_D",
            "con_bkg_postW": "boosted_control_sideband_region_D_postW",
        }
        if args.morphing_direction == "btag":
            sub_regions["sig_bkg_preW"] = "boosted_control_region_C"
            sub_regions["sig_bkg_postW"] = "boosted_control_region_C_postW"
            sub_regions["control"] = "boosted_signal_sideband_region_B"
        cat_dict |= {
            f"CR_TT_{args.region_suffix}": [
                [
                    sub_regions["control"],
                    sub_regions["con_bkg_postW"],
                    sub_regions["con_bkg_preW"],
                    sub_regions["control"] + "_TT",
                ]
            ],
            f"SR_TT_{args.region_suffix}": [
                [
                    sub_regions["signal"],
                    sub_regions["sig_bkg_postW"],
                    sub_regions["sig_bkg_preW"],
                    sub_regions["signal"] + "_TT",
                ]
            ],
            f"CR{args.region_suffix}": [
                [
                    sub_regions["control"],
                    sub_regions["con_bkg_postW"],
                    sub_regions["con_bkg_preW"],
                ]
            ],
            f"SR{args.region_suffix}": [
                [
                    sub_regions["signal"],
                    sub_regions["sig_bkg_postW"],
                    sub_regions["sig_bkg_preW"],
                ]
            ]
        }

elif args.mixed:
    cat_dict |= {
        f"CR{args.region_suffix}": [
            [
                f"4b{args.region_suffix}_control_region",
                f"4b{args.region_suffix}_control_region_postW",
                f"4b{args.region_suffix}_control_region_preW",
            ]
        ],
        f"SR{args.region_suffix}": [
            [
                f"4b{args.region_suffix}_signal_region",
                f"4b{args.region_suffix}_signal_region_postW",
                f"4b{args.region_suffix}_signal_region_preW",
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

if args.compare_morphing_models:
    if not args.boosted:
        raise ValueError("--compare-morphing-models is only for --boosted")
    # Same events (postW regions) reweighted with the two models, compared to
    # the target region (first category = reference of the ratio). The last
    # category is the ttbar MC of the target region: it is added to each
    # reweighted curve, whose yield is scaled to (data - ttbar MC), as in the
    # CR_TT / SR_TT plots. Only these plots are made.
    cat_dict = {
        f"CR_MODELCMP{args.region_suffix}": [
            [sub_regions["control"]]
            + [sub_regions["con_bkg_postW"] + s for s in MODEL_SUFFIXES]
            + [sub_regions["control"] + "_TT"]
        ],
        f"SR_MODELCMP{args.region_suffix}": [
            [sub_regions["signal"]]
            + [sub_regions["sig_bkg_postW"] + s for s in MODEL_SUFFIXES]
            + [sub_regions["signal"] + "_TT"]
        ],
    }

if args.skip_signal_region:
    if not args.boosted:
        raise ValueError("--skip-signal-region is only for --boosted")
    # no plot (and no loading) of the signal region data
    cat_dict = {name: cats for name, cats in cat_dict.items() if not name.startswith("SR")}

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
color_list_BOOSTED = [
    [("black",), ("red", "salmon"), ("gray",)]
]
color_list_MODELCMP = [[("black",), ("red",), ("blue",)]]
color_list_BOOSTED_TT = [
    [("black",), ("red", "salmon"), ("gray",), ("darkorange", "navajowhite")]
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


## Load the two morphing models to compare
if args.compare_morphing_models:
    morphing_sessions = {
        suffix: get_model_session(path, f"morphing_{suffix}")
        for suffix, path in zip(MODEL_SUFFIXES, args.compare_morphing_models)
    }
    morphing_input_list = get_DNN_input_list(
        args.run2, getattr(dnn_input_variables, args.input_variables)
    )
    print(f"Input list for the morphing models: {morphing_input_list}")
    for _suffix, (_session, _, _) in morphing_sessions.items():
        _n_model = _session.get_inputs()[0].shape[1]
        if _n_model != len(morphing_input_list):
            raise ValueError(
                f"The {_suffix} model has {_n_model} inputs but "
                f"--input-variables {args.input_variables} has "
                f"{len(morphing_input_list)}: give the variables of the "
                f"morphing training (a name in dnn_input_variables.py, e.g. "
                f"bkg_morphing_boosted_dnn_input_variables_other_group_txbb_wp)"
            )

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
        if file.endswith(".coffea") and ("DATA" in file or "Mixed" in file)
    ]
if not inputfiles:
    raise FileNotFoundError(
        f"No DATA/Mixed .coffea files found in {input_dir} (from -i {args.input_data})"
    )

# Optional debug restriction: only keep columns matching one of --only-vars
# ("weight" and "*_N" are always kept because the plotting code needs them).
# This filter is pushed all the way down into Arrow's to_table(columns=...)
# in the parquet loader, so unmatched columns are never read from disk.
if args.only_vars:
    if len(args.only_vars)==1 and args.only_vars[0].endswith(".py"):
        def _load_only_vars_from_py(filepath):
           spec = importlib.util.spec_from_file_location("only_vars_module", filepath)
           module = importlib.util.module_from_spec(spec)
           spec.loader.exec_module(module)
           variables_dict = module.dnn_input_variables
           only_vars = []
           for collection, feature, *_ in variables_dict.values():
               # strip the ":pos" suffix used for multi-jet inputs (e.g.
               # "JetGoodVBFEnergyOrdered:0") so the filter matches the real
               # per-jet column name ("JetGoodVBFEnergyOrdered_pt"), same as
               # get_DNN_input_list.py does when building the DNN input list.
               coll = collection.split(":")[0]
               only_vars.append(f"{coll}_{feature}")
           return only_vars

        args.only_vars = _load_only_vars_from_py(args.only_vars[0])
    def _keep_var(x):
        return "weight" in x or "_N" in x or any(s in x for s in args.only_vars)
else:
    def _keep_var(x):
        return True

filter_lambda = (
    (lambda x: _keep_var(x) and ("weight" in x or "score" in x))
    if args.spread
    else (lambda x: _keep_var(x) and "prov" not in x and "events_sigma" not in x and "era" not in x)
)

# In the boosted case the files contain many more categories (e.g. the
# *_cat2 sub-categories) than the ones used for the reweighting plots.
# Only load the base categories actually referenced in cat_dict.
if args.boosted:
    needed_categories = {
        strip_model_suffix(cat.replace("_MC", "").replace("_TT", "").replace("_SPREAD", ""))
        for cat_lists in cat_dict.values()
        for cat_list in cat_lists
        for cat in cat_list
    }
    print(f"Loading only categories: {sorted(needed_categories)}")
    category_filter = lambda c: c in needed_categories
else:
    category_filter = None

cat_col_data, total_datasets_list = get_columns_from_files(inputfiles, "nominal", filter_lambda, debug=False, novars=args.novars, filter_mixed=args.mixed, category_filter=category_filter)
cat_col_tt = None
inputfiles_tt = [
    os.path.join(input_dir, file)
    for file in os.listdir(input_dir)
    if file.endswith(".coffea") and ("TT" in file)
]
if inputfiles_tt:
    cat_col_tt, _ = get_columns_from_files(
        inputfiles_tt, "nominal",
        lambda x: _keep_var(x) and "prov" not in x and "era" not in x,
        debug=False, novars=args.novars, filter_mixed=args.mixed,
        category_filter=category_filter,
    )
else:
    if args.compare_morphing_models:
        raise FileNotFoundError(
            f"--compare-morphing-models adds the ttbar MC, but no TT .coffea files "
            f"were found in {input_dir}"
        )
    print("WARNING: no ttbar ('TT') files found in input_mc, ttbar stacking disabled")

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

    cat_col_mc, _ = get_columns_from_files(inputfiles_mc, "nominal", lambda x: _keep_var(x) and "prov" not in x and "era" not in x, debug=False, novars=args.novars, filter_mixed=args.mixed, category_filter=category_filter)

    if args.run2:
        cols_sig_mc = cat_col_mc[f"4b{args.region_suffix}_signal_regionRun2"]
    elif args.boosted:
        cols_sig_mc = cat_col_mc[sub_regions["signal"]]
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

    cleaned_weights = []
    mean_std_list = []
    for i, weights in enumerate(weights_list):
        if any(weights == np.inf):
            print(f"There are infinite values in the weights at {ak.local_index(weights[weights == np.inf])}")
            print(f"We remove them")
            weights = weights[weights < np.inf]
        var_name = f"Morphing weights" + (f" {i}" if len(weights_list) > 1 else "")
        # Remove corrupted weights
        nan_events = np.isnan(weights)
        if nan_events.any():
            print(f"Found invalid weights in position: {np.where(nan_events)}")
            print("These weights are removed for plotting")
            weights = weights[~nan_events]
        # Calculating limits of the weights:
        min_w = np.min(weights)
        max_w = np.max(weights)
        print(f"minimal weight in the collection is: {min_w} at index {np.where(weights == min_w)}")
        print(f"maximal weight in the collection is: {max_w} at index {np.where(weights == max_w)}")
        cleaned_weights.append((var_name, weights, min_w, max_w))
        mean_std_list.append(
            f"mean: {np.mean(weights):.2f}, std: {np.std(weights):.2f}"
        )

    output_base = os.path.join(outputdir, f"weights_{suffix}")

    # Produce both a log-x and a linear-x version of the weight distribution,
    # each with bin edges matched to that axis scale.
    for axis_scale in ("log", "linear"):
        hist_1d_dict = {}
        for var_name, weights, min_w, max_w in cleaned_weights:
            if axis_scale == "log":
                # negative/zero weights cannot be shown on a log axis
                pos_weights = weights[weights > 0]
                if pos_weights.size == 0:
                    print(f"WARNING: no positive weights for {var_name}, skipping log-x plot")
                    continue
                log_min = np.floor(np.log10(np.min(pos_weights)))
                log_max = np.ceil(np.log10(np.max(pos_weights)))
                if log_max <= log_min:
                    # avoid degenerate (non-ascending) bin edges when all
                    # positive weights fall within a single decade
                    log_max = log_min + 1
                bins = np.logspace(log_min, log_max, 100)
            else:
                if max_w <= min_w:
                    print(f"WARNING: degenerate range for {var_name}, skipping linear-x plot")
                    continue
                bins = np.linspace(min_w, max_w, 100)

            hist_w = Hist.new.Var(
                bins,
                name=var_name,
                flow=False,
            ).Double()
            hist_w.fill(weights)
            hist_1d_dict[var_name] = {
                "data": hist_w,
            }

        if not hist_1d_dict:
            continue

        p = (
            HEPPlotter()
            .set_plot_config(
                lumitext=f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)", cmstext=args.cmstext,
                figsize=[13, 13],
            )
            .set_output(f"{output_base}_{axis_scale}x")
            .set_labels(
                "Morphing weights",
                "Events",
            )
            .set_options(y_log=True, x_log=(axis_scale == "log"), set_ylim=False, legend=False)
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
        # model comparison with the ttbar MC as last category
        has_tt = "MODELCMP" in cats_name and cat_list[-1].endswith("_TT")
        model_stash = {}
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
                cat_plot_name = plot_regions_names(strip_model_suffix(cat), namesuffix)
                if model_suffix(cat):
                    cat_plot_name = cat_plot_name.replace(
                        " SPANet", f" ({MODEL_LABELS[model_suffix(cat)]}) SPANet"
                    )

            weights_den = weight_dict[cat]
            weights_num = weight_dict[cat_list[0]]

            col_den = col_dict[cat]
            col_num = col_dict[cat_list[0]]

            # remove padded values
            noninf_den = np.abs(weights_den) < np.inf
            noninf_num = np.abs(weights_num) < np.inf
            mask_den = (col_den != PAD_VALUE) & noninf_den
            mask_num = (col_num != PAD_VALUE) & noninf_num
            weights_den = weights_den[mask_den]
            weights_num = weights_num[mask_num]
            col_den = col_den[mask_den]
            col_num = col_num[mask_num]

            ### This is a bad workaround for the case where a few weights are Nan, very high or strongly negative.
            if args.mask_large_weights:
                if "postW" in cats_name:
                    n_before = len(weights_den)
                    bad_weights = np.isnan(weights_den) | (weights_den > 100) | (weights_den < 0)
                    n_cut = int(np.sum(bad_weights))
                    if n_cut > 0:
                        print(
                            f"[mask_large_weights] {cat}, variable {var}: removing "
                            f"{n_cut}/{n_before} events ({100 * n_cut / n_before:.3f}%) "
                            f"with NaN, weight > 100, or weight < 0"
                        )
                    weights_den = weights_den[~bad_weights]
                    col_den = col_den[~bad_weights]

            # WARNING: the weights are different for the additional jets because
            # the number of events is different since it's computed after the masking of the PAD_VALUE
            if has_tt:
                # Two reweighted curves (mean / median model) + the ttbar MC of
                # the target region (last category). Each QCD estimate is
                # scaled to (data - ttbar MC), the data and the ttbar MC keep
                # their own normalisation.
                norm_factor_num = 1.0
                if model_suffix(cat):
                    w_tt = weight_dict[cat_list[-1]]
                    c_tt = col_dict[cat_list[-1]]
                    m_tt = (c_tt != PAD_VALUE) & (np.abs(w_tt) < np.inf)
                    norm_factor_den = (
                        weights_num.sum() - w_tt[m_tt].sum()
                    ) / weights_den.sum()
                else:
                    norm_factor_den = 1.0
            elif "TT" in cats_name and i != 2:
                # TTbar-inclusive closure check compares absolute yields. The
                # TT MC (i == 3) and the data (i == 0) keep their own
                # normalization. The data-driven QCD estimate (i == 1) is
                # scaled to (data - ttbar MC) in this region, so that
                # QCD_est + ttbar_MC reproduces the data by construction
                # instead of double-counting the ttbar already contained in
                # the reweighted data.
                norm_factor_num = 1.0
                if i == 1:
                    w_tt = weight_dict[cat_list[3]]
                    c_tt = col_dict[cat_list[3]]
                    m_tt = (c_tt != PAD_VALUE) & (np.abs(w_tt) < np.inf)
                    norm_factor_den = (
                        weights_num.sum() - w_tt[m_tt].sum()
                    ) / weights_den.sum()
                else:
                    norm_factor_den = 1.0
            elif args.normalisation == "num_events":
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
            elif any(wp_var in var for wp_var in WP_INDEX_VARIABLES):
                # working-point index: one bin per index, over the values of all the regions of the plot, so that the
                # bins are the same in all the regions (the cuts of the quantile map move events between the indices)
                wp_values = np.concatenate([np.asarray(col_dict[c], dtype=np.float64) for c in cat_list])
                wp_values = wp_values[wp_values != PAD_VALUE]
                bins = np.arange(np.floor(wp_values.min()) - 0.5, np.ceil(wp_values.max()) + 1.5, 1)
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

            # Guard against degenerate binning: if the selected column is
            # (nearly) constant or empty, the 0.1%/99.9% quantile range
            # collapses and the bin edges are not strictly ascending. Skip
            # the whole variable instead of crashing the run.
            if not (np.all(np.isfinite(bins)) and np.all(np.diff(bins) > 0)):
                print(
                    f"WARNING: degenerate bin edges for {var} in {cats_name} "
                    f"(range {bins[0]}..{bins[-1]}); skipping variable"
                )
                return

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
                hist_1d_dict[cat_plot_name] = {
                    "data": histo,
                    "style": style_dict,
                }
            elif has_tt and model_suffix(cat):
                # stash the reweighted background of this model; the ttbar MC
                # (last category) is added to it below
                model_stash[cat] = (histo, cat_plot_name, color_list[k][i][0])
            elif has_tt and cat.endswith("_TT"):
                for stash_cat, (histo_bkg, name_bkg, color_bkg) in model_stash.items():
                    hist_1d_dict[f"{name_bkg} + TT"] = {
                        "data": histo_bkg + histo,
                        "style": {
                            "is_reference": False,
                            "histtype": "step",
                            "color": color_bkg,
                            "linewidth": 2,
                        },
                    }
                model_stash = {}
            elif "TT" in cats_name and i == 1:
                # stash the reweighted background; it is combined with the
                # TT MC histogram (i == 3) into one stacked entry below,
                # instead of being plotted as its own overlaid curve
                histo_bkg_stack = histo
                cat_plot_name_bkg_stack = cat_plot_name
                color_bkg_stack = color_list[k][i]
            elif "TT" in cats_name and i == 3:
                color_tt_stack = color_list[k][i]
                hist_1d_dict[f"{cat_plot_name_bkg_stack} + TT"] = {
                    "data": [histo_bkg_stack, histo],
                    "style": {
                        "is_reference": False,
                        "histtype": "fill",
                        "stack": True,
                        # keep TT last (top of stack) regardless of its
                        # integral relative to the morphed background
                        "stack_sort": False,
                        "legend_name": [cat_plot_name_bkg_stack, cat_plot_name],
                        "edgecolor": [color_bkg_stack[0], color_tt_stack[0]],
                        "facecolor": [color_bkg_stack[-1], color_tt_stack[-1]],
                        "alpha": [0.5, 0.5],
                    },
                }
                del histo_bkg_stack, cat_plot_name_bkg_stack, color_bkg_stack
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
                lumitext=f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)", cmstext=args.cmstext,
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
                set_ylim_ratio= 0.5
            )
            .set_data(hist_1d_dict, plot_type="1d")
            .add_ratio_hists(hist_1d_ratio_dict)
        )

    else:
        p = (
            HEPPlotter(debug=DEBUG)
            .set_plot_config(
                lumitext=f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)", cmstext=args.cmstext,
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
                set_ylim_ratio= 0.5
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
    cat_col_TT = cat_cols[2]

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
            if args.boosted and "DATAMC" not in cats_name and "SPREAD" not in cats_name:
                color_list = color_list_BOOSTED
            if "TT" in cats_name:
                color_list = color_list_BOOSTED_TT
            if "MODELCMP" in cats_name:
                chi_squared = False
                color_list = color_list_MODELCMP

            if "TT" in cats_name and cat_col_TT is None:
                print(f"WARNING: no ttbar columns loaded, skipping {cats_name}")
                continue

            # check if the categories are in the accumulator
            try:
                for cat in cat_list:
                    cat_col_DATA[strip_model_suffix(cat.replace("_MC", "").replace("_TT", "").replace("_SPREAD", ""))]
            except KeyError:
                print(
                    f"KeyError: {cat} not in {cat_col_DATA.keys()}, skipping {cats_name}"
                )
                continue

            if "SPREAD" in cats_name:
                vars_tot = list(cat_col_DATA[cat_list[0]].keys())
                vars_tot = [v for v in vars_tot if "weight" in v or "score" in v]
            else:
                vars_tot = list(cat_col_DATA[strip_model_suffix(cat_list[1])].keys())
                vars_tot = [v for v in vars_tot if "year" not in v]


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
                            elif "TT" in cat:
                                cat_col = cat_col_TT
                            else:
                                cat_col = cat_col_DATA

                            print(v, cat)
                            base_cat = strip_model_suffix(cat.replace("_MC", "").replace("_TT", ""))
                            try:
                                col_dict[f"{var_name}_{idx}"][cat] = cat_col[
                                    base_cat
                                ][v][
                                    np.arange(len(cat_col[base_cat][v]))
                                    % N
                                    == idx
                                ]
                            except KeyError:
                                col_dict[f"{var_name}_{idx}"][cat] = cat_col[
                                    base_cat
                                ][var_name][
                                    np.arange(
                                        len(cat_col[base_cat][var_name])
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
                        elif "TT" in cat:
                            cat_col = cat_col_TT
                        else:
                            cat_col = cat_col_DATA

                        # swap the dict keys
                        print(v, cat)
                        base_cat = strip_model_suffix(cat.replace("_MC", "").replace("_TT", ""))
                        try:
                            col_dict[var_name][cat] = cat_col[base_cat][v]
                        except KeyError:
                            col_dict[var_name][cat] = cat_col[base_cat][
                                var_name
                            ]

            cat_list_final = cat_list.copy()

            # re-evaluate the weight of the two morphing models on the same
            # events (the columns of the model categories are those of postW)
            model_weights = {}
            if args.compare_morphing_models:
                for cat in cat_list:
                    suffix = model_suffix(cat)
                    if not suffix:
                        continue
                    session, input_name, output_name = morphing_sessions[suffix]
                    w_model = evaluate_morphing_weight(
                        session,
                        input_name,
                        output_name,
                        [
                            col_dict[v.replace("Run2", "")][cat]
                            for v in morphing_input_list
                        ],
                    )
                    compare_with_stored(
                        f"{cats_name} {suffix}", w_model, np.asarray(col_dict["weight"][cat])
                    )
                    col_dict["weight"][cat] = w_model
                    model_weights[suffix] = w_model

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
                    # SvB classifiers output (N, n_classes) -> take the last
                    # class' score; single-output regression/reweighting
                    # models (e.g. the bkg-morphing "avg_w" output) are
                    # already 1D and used as-is.
                    col_dict[v][cat] = outputs[0][:, -1] if outputs[0].ndim > 1 else outputs[0]
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

            if len(model_weights) == 2:
                plot_weight_comparison(
                    model_weights[MODEL_SUFFIXES[0]],
                    model_weights[MODEL_SUFFIXES[1]],
                    cats_name,
                    dir_cat,
                    lumi,
                    era_string,
                    args.cmstext,
                )

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
        weights = cat_col_data[category]["weight"]
        if np.any(weights >= np.inf):
            print("Found infinite values in weights")
        weights = weights[weights < np.inf]
        if "postW" in category:
            plot_weights([weights], category, lumi, era_string)
    # plot the weights
    if cat_col_tt:
        for category in cat_col_tt.keys():
            weights = cat_col_tt[category]["weight"]
            if np.any(weights >= np.inf):
                print("Found infinite values in weights")
            weights = weights[weights < np.inf]
            if "postW" in category:
                plot_weights([weights], f"{category}_ttbar", lumi, era_string)

    main([cat_col_data, cat_col_mc, cat_col_tt], lumi, era_string)

    print(f"\nPlots saved in {outputdir}")
