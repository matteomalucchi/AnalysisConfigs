"""Find the TXbb cuts in the low-TXbb regions (A, B) that map onto the TXbb working points of the high-TXbb regions (C, D).

The boosted ABCD background model is trained on A and C (lower mass sideband) and applied to B and D (signal region).
The TXbb shape is not used, only the working-point (WP) index, so the TXbb of A and B has to be rescaled to the range of C and D.
The linear rescaling of `btagBBTXbb_morph` (workflow.py) does not guarantee the same WP occupancies. Here a quantile map is used instead:

    1. the weighted CDF of the subleading-Higgs TXbb of the data in region C is evaluated at the WPs -> percentiles p_i
    2. the cuts in region A are the TXbb values at the same percentiles p_i of the weighted CDF of the data in region A

so that the fraction of A in each WP bin equals the fraction of C in that WP bin. The cuts are written to a JSON file that is read by the
boosted workflow (config option `txbb_cdf_wp_cuts`) to define `btagBBTXbb_CDF_rescaled_WPs`.

Only the upper WPs (0.8, 0.95, 0.975, 0.99) are matched: the lowest WP (0.65) is the lower edge of region C, i.e. percentile 0, which
corresponds to the lowest TXbb of region A (the baseline cut) and is therefore trivial.

Needs the ABCD regions (columns in `boosted_signal_sideband_region_C` and `boosted_control_sideband_region_A`) produced WITHOUT the
`txbb_cdf_wp_cuts` option; the nominal TXbb used here does not depend on it.
"""
import argparse
import json
import logging
import os

import numpy as np

from utils_configs.plot.get_columns_from_files import get_columns_from_files
from utils_configs.plot.weighted_quantile import weighted_quantile


logging.basicConfig(format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

# TXbb working points (VL, L, M, T, XT) as in `generate_btag_workingpoints`. VL is the lower edge of region C (50GeV setting).
WPS = [0.65, 0.8, 0.95, 0.975, 0.99]
UPPER_WPS = WPS[1:]
# linear rescaling of the workflow (`btagBBTXbb_morph`) for the 50GeV setting, only used for the comparison
LINEAR_B_LOW, LINEAR_B_THRESH = 0.05, 0.65
PAD_VALUE = -999
MIN_EVENTS_TOP_WP = 50


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input-data", type=str, required=True, help="Directory with the coffea output files (one per dataset)")
    parser.add_argument("-o", "--output", type=str, default="./txbb_cdf_wp_cuts", help="Output directory")
    parser.add_argument("--data-pattern", type=str, default="DATA", help="Substring selecting the data files")
    parser.add_argument("--region-c", type=str, default="boosted_signal_sideband_region_C", help="High-TXbb region defining the percentiles of the WPs")
    parser.add_argument("--region-a", type=str, default="boosted_control_sideband_region_A", help="Low-TXbb region in which the cuts are searched")
    parser.add_argument("--txbb-column", type=str, default="HiggsSubLeading_btagBBTXbb", help="Subleading Higgs TXbb column")
    parser.add_argument("--weight-column", type=str, default="weight")
    parser.add_argument("--region-setting", choices=["50GeV"], default="50GeV",
                        help="Only the 50GeV setting is supported (the lowest WP, 0.65, has to be the lower edge of region C)")
    parser.add_argument("--no-plot", action="store_true", default=False, help="Do not make the diagnostic plot")
    return parser.parse_args(argv)


# ======================= Loading =======================
def load_regions(input_dir, data_pattern, regions, columns):
    files = sorted(
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.endswith(".coffea") and f != "output_all.coffea" and data_pattern in f
    )
    if not files:
        raise ValueError(f"No files containing '{data_pattern}' found in {input_dir}")
    logger.info(f"Data files: {[os.path.basename(f) for f in files]}")
    cat_col, _ = get_columns_from_files(
        files, "nominal", lambda c: c in columns, category_filter=lambda c: c in regions,
    )
    out = {}
    for region in regions:
        if region not in cat_col:
            raise ValueError(f"Category {region} not found in {files[0]} (and others)")
        missing = [c for c in columns if c not in cat_col[region]]
        if missing:
            raise ValueError(f"Columns {missing} not found in category {region}. Available: {sorted(cat_col[region])}")
        out[region] = {c: np.asarray(cat_col[region][c], dtype=np.float64) for c in columns}
    return out


def clean(txbb, weights):
    """Drop padded / non-finite entries."""
    ok = np.isfinite(txbb) & np.isfinite(weights) & (txbb > PAD_VALUE + 1)
    if not ok.all():
        logger.warning(f"Dropping {(~ok).sum()} of {len(ok)} entries with padded or non-finite values")
    return txbb[ok], weights[ok]


# ======================= Quantile map =======================
def weighted_cdf_at(values, weights, points):
    """Weighted CDF P(x <= point) at each of the points. Same convention as the WPs: a jet passes a WP if x > WP."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.concatenate([[0.0], np.cumsum(weights)])
    idx = np.searchsorted(values, points, side="right")
    return cumulative[idx] / cumulative[-1]


def quantile_map(txbb_c, w_c, txbb_a, w_a, wps):
    """Cuts in A with the same percentiles as the wps in C. Returns (percentiles, cuts_A)."""
    percentiles = weighted_cdf_at(txbb_c, w_c, np.asarray(wps))
    # old_style: inverse of the weighted ECDF (cumulative weight normalised to 1 at the last event)
    cuts_a = weighted_quantile(txbb_a, percentiles, w_a, old_style=True)
    return percentiles, np.asarray(cuts_a)


def bin_fractions(values, weights, edges):
    """Weighted fraction in the bins (edges[i], edges[i+1]], the last bin is open on the right."""
    edges = np.asarray(edges, dtype=np.float64)
    idx = np.searchsorted(edges, values, side="left") - 1  # values in (edges[i], edges[i+1]] -> i
    total = weights.sum()
    return np.array([weights[idx == i].sum() / total for i in range(len(edges) - 1)])


def linear_morph(txbb):
    return np.where(
        txbb > LINEAR_B_THRESH, txbb,
        (txbb - LINEAR_B_LOW) * (1 - LINEAR_B_THRESH) / (LINEAR_B_THRESH - LINEAR_B_LOW) + LINEAR_B_THRESH,
    )


def closure_table(txbb_c, w_c, txbb_a, w_a, cuts_a):
    """Fraction of C and of A (new quantile map and old linear morph) in each of the 5 WP bins (VL, L, M, T, XT)."""
    edges_c = np.array(WPS + [np.inf])
    frac_c = bin_fractions(txbb_c, w_c, edges_c)
    edges_a = np.concatenate([[txbb_a.min() - 1e-9], cuts_a, [np.inf]])  # the first bin (VL) starts at the lowest TXbb of A
    frac_a_new = bin_fractions(txbb_a, w_a, edges_a)
    frac_a_old = bin_fractions(linear_morph(txbb_a), w_a, edges_c)
    return frac_c, frac_a_new, frac_a_old


def print_table(percentiles, cuts_a, frac_c, frac_a_new, frac_a_old, txbb_a):
    lines = ["", f"{'WP bin':<22}{'C':>10}{'A new (CDF)':>14}{'A old (linear)':>16}"]
    names = [f"({lo}, {hi}]" for lo, hi in zip(WPS[:-1], WPS[1:])] + [f"> {WPS[-1]}"]
    for name, c, new, old in zip(names, frac_c, frac_a_new, frac_a_old):
        lines.append(f"{name:<22}{c:>10.4f}{new:>14.4f}{old:>16.4f}")
    lines.append("")
    lines.append(f"{'WP (C)':<10}{'percentile in C':>18}{'cut in A':>12}")
    for wp, p, cut in zip(UPPER_WPS, percentiles, cuts_a):
        lines.append(f"{wp:<10}{p:>18.5f}{cut:>12.5f}")
    lines.append(f"(lowest TXbb in A: {txbb_a.min():.4f})")
    logger.info("\n".join(lines))


# ======================= Plot =======================
def make_plot(txbb_c, w_c, txbb_a, w_a, percentiles, cuts_a, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not available, skipping the plot")
        return

    def ecdf(values, weights):
        order = np.argsort(values)
        return values[order], np.cumsum(weights[order]) / weights.sum()

    fig, ax = plt.subplots(figsize=(7, 5))
    xc, yc = ecdf(txbb_c, w_c)
    xa, ya = ecdf(txbb_a, w_a)
    ax.plot(xa, ya, label="Region A (data)", color="tab:blue")
    ax.plot(xc, yc, label="Region C (data)", color="tab:orange")
    for wp, p, cut in zip(UPPER_WPS, percentiles, cuts_a):
        ax.plot([wp, cut], [p, p], color="grey", ls="--", lw=0.8)
        ax.plot([wp], [p], "o", color="tab:orange", ms=4)
        ax.plot([cut], [p], "o", color="tab:blue", ms=4)
    ax.set_xlabel("Subleading Higgs TXbb")
    ax.set_ylabel("Weighted CDF")
    ax.set_ylim(0, 1.02)
    ax.set_title("Quantile map of the TXbb working points from C to A")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot written to {path}")


# ======================= Main =======================
def main(argv=None):
    args = get_args(argv)
    columns = [args.txbb_column, args.weight_column]
    data = load_regions(args.input_data, args.data_pattern, [args.region_c, args.region_a], columns)
    txbb_c, w_c = clean(data[args.region_c][args.txbb_column], data[args.region_c][args.weight_column])
    txbb_a, w_a = clean(data[args.region_a][args.txbb_column], data[args.region_a][args.weight_column])
    logger.info(f"Region C: {len(txbb_c)} events, sum w = {w_c.sum():.1f}, TXbb in [{txbb_c.min():.4f}, {txbb_c.max():.4f}]")
    logger.info(f"Region A: {len(txbb_a)} events, sum w = {w_a.sum():.1f}, TXbb in [{txbb_a.min():.4f}, {txbb_a.max():.4f}]")
    if txbb_c.min() < WPS[0] - 1e-6:
        raise ValueError(f"Region C has TXbb values below the lowest WP {WPS[0]} (min {txbb_c.min():.4f}): is this the 50GeV setting?")
    if txbb_a.max() > WPS[0] + 1e-6:
        raise ValueError(f"Region A has TXbb values above {WPS[0]} (max {txbb_a.max():.4f}): is this the 50GeV setting?")

    percentiles, cuts_a = quantile_map(txbb_c, w_c, txbb_a, w_a, UPPER_WPS)
    if np.any(np.diff(cuts_a) < 0):
        raise ValueError(f"The cuts in A are not monotonic: {cuts_a}")
    n_top = int((txbb_a > cuts_a[-1]).sum())
    if n_top < MIN_EVENTS_TOP_WP:
        logger.warning(f"Only {n_top} raw events in region A above the highest cut {cuts_a[-1]:.4f}: the top WP is statistically limited")

    frac_c, frac_a_new, frac_a_old = closure_table(txbb_c, w_c, txbb_a, w_a, cuts_a)
    print_table(percentiles, cuts_a, frac_c, frac_a_new, frac_a_old, txbb_a)

    os.makedirs(args.output, exist_ok=True)
    result = {
        "region_setting": args.region_setting,
        "wps_C": UPPER_WPS,
        "percentiles": [float(p) for p in percentiles],
        "cuts_A": [float(c) for c in cuts_a],
        "region_C": args.region_c,
        "region_A": args.region_a,
        "n_events_C": int(len(txbb_c)),
        "n_events_A": int(len(txbb_a)),
        "sum_w_C": float(w_c.sum()),
        "sum_w_A": float(w_a.sum()),
    }
    out_json = os.path.join(args.output, "txbb_cdf_wp_cuts.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Cuts written to {out_json}")
    if not args.no_plot:
        make_plot(txbb_c, w_c, txbb_a, w_a, percentiles, cuts_a, os.path.join(args.output, "txbb_cdf_wp_cuts.png"))


if __name__ == "__main__":
    main()
