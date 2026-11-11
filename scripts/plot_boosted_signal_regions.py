"""Plot the boosted background prediction (reweighted QCD estimate + ttbar MC)
in each boosted signal-region category (catvbf, cat1, cat2, cat3) and, as an additional comparison,
in the full (inclusive) signal region D (category "full").

The QCD estimate is the reweighted data of boosted_control_region_B_<cat>_postW,
scaled with one global factor such that QCD estimate + ttbar MC matches the data in C:
    (sum_w(data, C) - sum_w(ttbar MC, C)) / sum_w(data, boosted_control_sideband_region_A_postW)
with C = boosted_signal_sideband_region_C.
The ttbar component is taken from the TT MC in boosted_signal_region_D_<cat>.
Data is only shown in the categories passed with --unblind (default: cat3, full).

With --region sideband the same plots are made for the mass sideband C (closure check): the QCD estimate is the
reweighted data of boosted_control_sideband_region_A_<cat>_postW (same global scale factor), the ttbar component is
the TT MC and the data are taken from boosted_signal_sideband_region_C_<cat>. Data is always shown (C is not blinded)
and --unblind is ignored. For "full" the prediction matches the data in total by construction.

With --region txbbwp the sideband closure is made in bins of the TXbb working point (categories txbbwp<i>, bin i is
(TXBB_WP_EDGES[i], TXBB_WP_EDGES[i+1]]): the data are taken from boosted_signal_sideband_region_C_txbbwp<i> (nominal
TXbb) and the QCD estimate from boosted_control_sideband_region_A_txbbwp<i>_postW (morphed TXbb), both with the same
global scale factor. Besides the per-variable plots, closure_summary.png/pdf shows the data/prediction yield per bin.
The input must be made with the option boosted_txbb_wp_categories.

Example:
    python scripts/plot_boosted_signal_regions.py -i /path/to/output_dir -o plots_boosted_SR
    python scripts/plot_boosted_signal_regions.py -i /path/to/output_dir -r sideband -o plots_boosted_SB
    python scripts/plot_boosted_signal_regions.py -i /path/to/output_dir -r txbbwp -o plots_boosted_TXWP
"""

import os
import argparse
import numpy as np
from multiprocessing import Pool
from hist import Hist

from utils_configs.plot.get_era_lumi import get_era_lumi
from utils_configs.plot.get_columns_from_files import get_columns_from_files
from utils_configs.plot.HEPPlotter import HEPPlotter


# Pseudo-category for the full (inclusive) signal region D / control region B
FULL_SR = "full"

parser = argparse.ArgumentParser(description="Plot boosted background prediction in the signal-region categories")
parser.add_argument("-i", "--input-dir", type=str, required=True, help="Directory with the coffea output files")
parser.add_argument("-o", "--output", type=str, default=None,
                    help="Output directory (default: plots_boosted_SR, plots_boosted_SB with --region sideband, "
                         "plots_boosted_TXWP with --region txbbwp)")
parser.add_argument("-r", "--region", type=str, choices=["signal", "sideband", "txbbwp"], default="signal",
                    help="'signal': QCD from B_postW + ttbar MC of D vs data of D; "
                         "'sideband': QCD from A_postW + ttbar MC of C vs data of C (closure check); "
                         "'txbbwp': the same closure check in bins of the TXbb working point")
parser.add_argument("-c", "--categories", type=str, nargs="+", default=None,
                    help=f"Categories to plot ('{FULL_SR}' = inclusive region). Default: catvbf cat1 cat2 cat3 {FULL_SR}, "
                         f"or the TXbb working point bins txbbwp0 txbbwp1 ... with --region txbbwp")
parser.add_argument("-u", "--unblind", type=str, nargs="*", default=["cat3", "full"], help="Categories in which data is shown")
parser.add_argument("-w", "--workers", type=int, default=8, help="Number of workers")
parser.add_argument("-l", "--linear", action="store_true", help="Linear y scale", default=False)
parser.add_argument("-ov", "--only-vars", type=str, nargs="+", default=None, help="Only plot columns containing one of these substrings")
parser.add_argument("-mw", "--mask-large-weights", action="store_true", default=False,
                    help="Remove reweighted events with NaN, weight > 100 or weight < 0 (also in the normalisation region)")
parser.add_argument("-cms", "--cmstext", type=str, default="Private", help="Text shown on CMS plot")
args = parser.parse_args()

TXBB_WP = args.region == "txbbwp"
if TXBB_WP:
    # single source of truth for the bin edges of the txbbwp<i> categories
    from configs.HH4b_common.config_files.configurator_tools import TXBB_WP_EDGES

    DEFAULT_CATEGORIES = [f"txbbwp{i}" for i in range(len(TXBB_WP_EDGES) - 1)]
else:
    DEFAULT_CATEGORIES = ["catvbf", "cat1", "cat2", "cat3", FULL_SR]
# bins that don't exist in the input (e.g. the lowest bin for region_setting 40GeV) are dropped for the default list
DEFAULT_CATEGORIES_USED = args.categories is None
if DEFAULT_CATEGORIES_USED:
    args.categories = DEFAULT_CATEGORIES

NUMBER_OF_BINS = 20
PAD_VALUE = -999

SIG_REGION_FULL = "boosted_signal_region_D"
BKG_REGION_FULL = "boosted_control_region_B_postW"
NORM_NUM_REGION = "boosted_signal_sideband_region_C"
NORM_DEN_REGION = "boosted_control_sideband_region_A_postW"

COLOR_DATA = "black"
COLOR_QCD = ("red", "salmon")
COLOR_TT = ("darkorange", "navajowhite")

SKIP_VARS = ("weight", "events_event", "year")

SIDEBAND = args.region in ("sideband", "txbbwp")
REGION_TAG = {"signal": "SR", "sideband": "SB", "txbbwp": "TXWP"}[args.region]
# C is not blinded, so data is always shown there
UNBLIND = set(args.categories) if SIDEBAND else set(args.unblind)


def category_label(category):
    """Readable label of a category: the TXbb range for the txbbwp<i> bins, the category name otherwise."""
    if TXBB_WP and category.startswith("txbbwp"):
        i = int(category[len("txbbwp"):])
        return f"{TXBB_WP_EDGES[i]} < TXbb ≤ {TXBB_WP_EDGES[i + 1]}"
    return category


def sig_region(category):
    """Region of the data and of the ttbar MC (D for the signal region, C for the sideband)."""
    base = NORM_NUM_REGION if SIDEBAND else SIG_REGION_FULL
    return base if category == FULL_SR else f"{base}_{category}"


def bkg_region(category):
    """Region of the reweighted data used as QCD estimate (B_postW for the signal region, A_postW for the sideband)."""
    if SIDEBAND:
        return NORM_DEN_REGION if category == FULL_SR else f"boosted_control_sideband_region_A_{category}_postW"
    return BKG_REGION_FULL if category == FULL_SR else f"boosted_control_region_B_{category}_postW"


outputdir = args.output or f"plots_boosted_{REGION_TAG}"
os.makedirs(outputdir, exist_ok=True)


def _keep_var(x):
    if "weight" in x:
        return True
    if "prov" in x or "era" in x or "events_sigma" in x:
        return False
    return args.only_vars is None or any(s in x for s in args.only_vars)


def _list_files(selector):
    return sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".coffea") and selector(f)
    )


data_categories = {NORM_NUM_REGION, NORM_DEN_REGION}
data_categories |= {bkg_region(c) for c in args.categories}
data_categories |= {sig_region(c) for c in args.categories if c in UNBLIND}
tt_categories = {NORM_NUM_REGION} | {sig_region(c) for c in args.categories}

print(f"Loading data categories: {sorted(data_categories)}")
cat_col_data, total_datasets_list = get_columns_from_files(
    _list_files(lambda f: "DATA" in f), "nominal", _keep_var,
    category_filter=lambda c: c in data_categories,
)
print(f"Loading ttbar categories: {sorted(tt_categories)}")
cat_col_tt, _ = get_columns_from_files(
    _list_files(lambda f: "_TT" in f), "nominal", _keep_var,
    category_filter=lambda c: c in tt_categories,
)

if DEFAULT_CATEGORIES_USED and TXBB_WP:
    available = [
        c for c in args.categories
        if all(r in cat_col_data for r in (bkg_region(c), sig_region(c))) and sig_region(c) in cat_col_tt
    ]
    if dropped := [c for c in args.categories if c not in available]:
        print(f"WARNING: categories {dropped} not found in the input, dropping them")
    if not available:
        raise KeyError("None of the txbbwp categories found in the input, was it made with boosted_txbb_wp_categories?")
    args.categories = available
    UNBLIND = set(args.categories)


def clean_weights(cols, is_reweighted):
    """Return a mask removing events with non-finite (and optionally large/negative) weights."""
    w = cols["weight"]
    mask = np.isfinite(w)
    if is_reweighted and args.mask_large_weights:
        mask &= (w <= 100) & (w >= 0)
    n_cut = int((~mask).sum())
    if n_cut:
        print(f"  removing {n_cut}/{len(w)} events ({100 * n_cut / len(w):.3f}%) with bad weights")
    return mask


masks_data = {cat: clean_weights(cols, "postW" in cat) for cat, cols in cat_col_data.items()}
masks_tt = {cat: clean_weights(cols, False) for cat, cols in cat_col_tt.items()}

# Global normalisation of the reweighted QCD estimate: QCD_est(A_postW) + ttbar MC(C) = data(C)
sum_data = cat_col_data[NORM_NUM_REGION]["weight"][masks_data[NORM_NUM_REGION]].sum()
sum_tt = cat_col_tt[NORM_NUM_REGION]["weight"][masks_tt[NORM_NUM_REGION]].sum()
sum_den = cat_col_data[NORM_DEN_REGION]["weight"][masks_data[NORM_DEN_REGION]].sum()
NORM_FACTOR = (sum_data - sum_tt) / sum_den
print(
    f"Normalisation factor (data - ttbar)({NORM_NUM_REGION}) / {NORM_DEN_REGION} = "
    f"({sum_data} - {sum_tt}) / {sum_den} = {NORM_FACTOR:.5f}"
)


def get_bins(col):
    if np.issubdtype(col.dtype, np.integer):
        return np.arange(col.min() - 0.5, col.max() + 1.5, 1)
    low, high = np.quantile(col, [0.001, 0.999])
    return np.linspace(low, high, NUMBER_OF_BINS + 1)


def make_hist(col, weight, bins, var):
    histo = Hist.new.Var(bins, name=var, flow=False).Weight()
    histo.fill(col, weight=weight)
    return histo


def get_column(cat_col, masks, region, var):
    col = cat_col[region][var][masks[region]]
    w = cat_col[region]["weight"][masks[region]]
    pad = col != PAD_VALUE
    return col[pad], w[pad]


def plot_single_var(category, var, lumi, era_string):
    col_qcd, w_qcd = get_column(cat_col_data, masks_data, bkg_region(category), var)
    col_tt, w_tt = get_column(cat_col_tt, masks_tt, sig_region(category), var)
    w_qcd = w_qcd * NORM_FACTOR
    show_data = category in UNBLIND

    if len(col_qcd) == 0:
        print(f"WARNING: no background events for {var} in {category}, skipping")
        return
    # binning from the background prediction, so it doesn't depend on unblinding
    bins = get_bins(col_qcd)
    if not (np.all(np.isfinite(bins)) and np.all(np.diff(bins) > 0)):
        print(f"WARNING: degenerate bin edges for {var} in {category}, skipping")
        return

    hist_1d_dict = {}
    if show_data:
        col_data, w_data = get_column(cat_col_data, masks_data, sig_region(category), var)
        hist_1d_dict["Data"] = {
            "data": make_hist(col_data, w_data, bins, var),
            "style": {
                "is_reference": True,
                "histtype": "errorbar",
                "color": COLOR_DATA,
            },
        }
    hist_1d_dict["QCD (reweighted) + TT"] = {
        "data": [make_hist(col_qcd, w_qcd, bins, var), make_hist(col_tt, w_tt, bins, var)],
        "style": {
            "is_reference": False,
            "histtype": "fill",
            "stack": True,
            # keep TT on top of the stack
            "stack_sort": False,
            "legend_name": ["QCD (reweighted)", r"$t\bar{t}$"],
            "edgecolor": [COLOR_QCD[0], COLOR_TT[0]],
            "facecolor": [COLOR_QCD[1], COLOR_TT[1]],
            "alpha": [0.5, 0.5],
        },
    }

    dir_cat = os.path.join(outputdir, f"{REGION_TAG}_{category}")
    os.makedirs(dir_cat, exist_ok=True)
    p = (
        HEPPlotter()
        .set_plot_config(
            lumitext=f"{REGION_TAG} {category_label(category)}, {era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)", cmstext=args.cmstext,
            figsize=[13, 13],
        )
        .set_output(os.path.join(dir_cat, var))
        .set_labels(var, "Events", ratio_label="Data/Pred.")
        .set_options(y_log=not args.linear, reference_to_den=False, set_ylim_ratio=0.5)
        .set_data(hist_1d_dict, plot_type="1d")
    )
    if show_data:
        p = p.add_chi_square()
    p.run()


def print_yields():
    print(f"\n{'category':<10} {'QCD (rew.)':>14} {'TT':>12} {'total':>14} {'data':>10}")
    for category in args.categories:
        bkg_reg, sig_reg = bkg_region(category), sig_region(category)
        y_qcd = cat_col_data[bkg_reg]["weight"][masks_data[bkg_reg]].sum() * NORM_FACTOR
        y_tt = cat_col_tt[sig_reg]["weight"][masks_tt[sig_reg]].sum()
        y_data = (
            f"{cat_col_data[sig_reg]['weight'][masks_data[sig_reg]].sum():10.1f}"
            if category in UNBLIND else f"{'blind':>10}"
        )
        print(f"{category:<10} {y_qcd:14.2f} {y_tt:12.2f} {y_qcd + y_tt:14.2f} {y_data}")
    print()


def _yield_and_error(cat_col, masks, region, scale=1.0):
    """Sum of weights and its statistical uncertainty sqrt(sum w^2)."""
    w = cat_col[region]["weight"][masks[region]] * scale
    return w.sum(), np.sqrt((w ** 2).sum())


def plot_closure_summary(lumi, era_string):
    """Data / prediction yield in each category (TXbb working point bin), with statistical uncertainties only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = args.categories
    qcd = np.array([_yield_and_error(cat_col_data, masks_data, bkg_region(c), NORM_FACTOR) for c in categories])
    tt = np.array([_yield_and_error(cat_col_tt, masks_tt, sig_region(c)) for c in categories])
    data = np.array([_yield_and_error(cat_col_data, masks_data, sig_region(c)) for c in categories])
    pred = qcd[:, 0] + tt[:, 0]
    pred_err = np.hypot(qcd[:, 1], tt[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = data[:, 0] / pred
        ratio_err = ratio * np.hypot(data[:, 1] / data[:, 0], pred_err / pred)

    print(f"{'bin':<22} {'data':>12} {'pred':>12} {'data/pred':>18}")
    for c, d, p, r, re in zip(categories, data[:, 0], pred, ratio, ratio_err):
        print(f"{category_label(c):<22} {d:12.1f} {p:12.1f} {r:10.3f} +- {re:.3f}")

    x = np.arange(len(categories))
    fig, (ax, ax_ratio) = plt.subplots(
        2, 1, figsize=(9, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    ax.bar(x, qcd[:, 0], width=0.8, color=COLOR_QCD[1], edgecolor=COLOR_QCD[0], label="QCD (reweighted)")
    ax.bar(x, tt[:, 0], width=0.8, bottom=qcd[:, 0], color=COLOR_TT[1], edgecolor=COLOR_TT[0], label=r"$t\bar{t}$")
    ax.errorbar(x, pred, yerr=pred_err, fmt="none", ecolor="gray", capsize=3, label="Pred. stat. unc.")
    ax.errorbar(x, data[:, 0], yerr=data[:, 1], fmt="o", color=COLOR_DATA, label="Data")
    if not args.linear:
        ax.set_yscale("log")
    ax.set_ylabel("Events")
    ax.set_title(f"{REGION_TAG} closure, {era_string}, {lumi}" + r" $fb^{-1}$ (13.6 TeV)", loc="right", fontsize=11)
    ax.legend()

    ax_ratio.errorbar(x, ratio, yerr=ratio_err, fmt="o", color=COLOR_DATA)
    ax_ratio.axhline(1, color="gray", linestyle="--")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_ylabel("Data/Pred.")
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels([category_label(c) for c in categories], rotation=20, ha="right")
    ax_ratio.set_xlabel("Subleading Higgs TXbb bin")

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outputdir, f"closure_summary.{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    lumi, era_string = get_era_lumi(total_datasets_list)

    missing = [
        c for c in args.categories
        if bkg_region(c) not in cat_col_data or sig_region(c) not in cat_col_tt
    ]
    if missing:
        raise KeyError(f"Missing background or ttbar columns for categories {missing}")

    print_yields()
    if TXBB_WP:
        plot_closure_summary(lumi, era_string)

    jobs = []
    for category in args.categories:
        vars_bkg = set(cat_col_data[bkg_region(category)].keys())
        vars_tt = set(cat_col_tt[sig_region(category)].keys())
        variables = sorted(
            v for v in vars_bkg & vars_tt
            if not any(s in v for s in SKIP_VARS) and not v.endswith("_N")
        )
        jobs += [(category, var, lumi, era_string) for var in variables]

    if args.workers > 1:
        with Pool(args.workers) as p:
            p.starmap(plot_single_var, jobs)
    else:
        for job in jobs:
            plot_single_var(*job)

    print(f"\nPlots saved in {outputdir}")
