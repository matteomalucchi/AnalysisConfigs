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

Leading Higgs (`--higgs leading`): the same algorithm on `HiggsLeading_btagBBTXbb`, written to `txbb_cdf_wp_cuts_leading.json` (config option
`txbb_cdf_wp_cuts_leading`). Differences with the subleading Higgs: in region C the leading TXbb is also above the lowest WP (it is
at least the subleading one), but in region A it spans the full range (up to 1), so the map is applied by region in the workflow (decided
by the subleading TXbb), not by value, and the check "A below the lowest WP" is dropped. `--subleading-cuts <json>` additionally prints the joint
(leading, subleading) working-point occupancy of C and A, a diagnostic of the fact that the two maps are marginal.

Conditional leading map (`--higgs leading --conditional --subleading-cuts <json>`): the marginal maps of the two Higgs candidates do not preserve
the joint occupancy (in C the leading index is never below the subleading one, in the mapped A it can be). Here the leading cuts are derived
separately for each subleading working-point index s (Rosenblatt-style, the subleading map is unchanged): the percentiles of the WPs 0.8/0.95/0.975/0.99
are those of the leading TXbb in C restricted to s, the cuts are the leading TXbb values at the same percentiles in A restricted to s. A percentile
of exactly 0 (the WP lies below the leading TXbb that C can have for this s) is stored as the sentinel -1.0, "always exceeded", so that the leading
index is at least s by construction. `--subleading-groups` can pool several s (default: one group per s, no pooling). Written to
`txbb_cdf_wp_cuts_leading_cond.json` (config option `txbb_cdf_wp_cuts_leading`, nested form).
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
    parser.add_argument("--higgs", choices=["subleading", "leading"], default="subleading",
                        help="Higgs candidate whose TXbb working points are mapped (default: subleading)")
    parser.add_argument("--txbb-column", type=str, default=None,
                        help="TXbb column (default: HiggsSubLeading_btagBBTXbb or HiggsLeading_btagBBTXbb, following --higgs)")
    parser.add_argument("--subleading-cuts", type=str, default=None,
                        help="Only with --higgs leading: JSON of the subleading map (find_txbb_cdf_wp_cuts.py). Used for the diagnostic of the "
                             "joint (leading, subleading) working-point occupancy of C and A")
    parser.add_argument("--subleading-column", type=str, default="HiggsSubLeading_btagBBTXbb", help="Subleading TXbb column (joint diagnostic and --conditional)")
    parser.add_argument("--conditional", action="store_true", default=False,
                        help="Only with --higgs leading (needs --subleading-cuts): one set of leading cuts per subleading working-point index")
    parser.add_argument("--subleading-groups", type=str, default="1;2;3;4;5",
                        help="Only with --conditional: groups of subleading indices sharing one map, groups separated by ';' and indices by ',', "
                             "contiguous partition of 1..5 (default '1;2;3;4;5': no pooling, e.g. '1;2;3,4,5' pools 3-5)")
    parser.add_argument("--min-events-c", type=int, default=50,
                        help="Only with --conditional: warn if region C has fewer events than this in a group (nothing is pooled automatically)")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Only with --conditional: number of bootstrap replicas of region C to estimate the spread of the cuts (0: none)")
    parser.add_argument("--weight-column", type=str, default="weight")
    parser.add_argument("--region-setting", choices=["50GeV"], default="50GeV",
                        help="Only the 50GeV setting is supported (the lowest WP, 0.65, has to be the lower edge of region C)")
    parser.add_argument("--no-plot", action="store_true", default=False, help="Do not make the diagnostic plot")
    args = parser.parse_args(argv)
    if args.txbb_column is None:
        args.txbb_column = "HiggsLeading_btagBBTXbb" if args.higgs == "leading" else "HiggsSubLeading_btagBBTXbb"
    if args.subleading_cuts and args.higgs != "leading":
        parser.error("--subleading-cuts is only used with --higgs leading")
    if args.conditional:
        if args.higgs != "leading":
            parser.error("--conditional is only implemented for --higgs leading")
        if not args.subleading_cuts:
            parser.error("--conditional needs --subleading-cuts (the subleading index s is defined with these cuts in region A)")
        try:
            args.groups = parse_groups(args.subleading_groups)
        except ValueError as e:
            parser.error(str(e))
    return args


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


def print_table(percentiles, cuts_a, frac_c, frac_a_new, frac_a_old, txbb_a, frac_a_nominal=None):
    header = f"{'WP bin':<22}{'C':>10}{'A new (CDF)':>14}{'A old (linear)':>16}"
    if frac_a_nominal is not None:
        header += f"{'A nominal WPs':>16}"
    lines = ["", header]
    names = [f"({lo}, {hi}]" for lo, hi in zip(WPS[:-1], WPS[1:])] + [f"> {WPS[-1]}"]
    for i, (name, c, new, old) in enumerate(zip(names, frac_c, frac_a_new, frac_a_old)):
        line = f"{name:<22}{c:>10.4f}{new:>14.4f}{old:>16.4f}"
        if frac_a_nominal is not None:
            line += f"{frac_a_nominal[i + 1]:>16.4f}"
        lines.append(line)
    if frac_a_nominal is not None:
        lines.append(f"(A nominal WPs: the plain digitisation of A, without the map; {frac_a_nominal[0]:.4f} of A is below the lowest WP "
                     f"{WPS[0]} and has no counterpart in C)")
    lines.append("")
    lines.append(f"{'WP (C)':<10}{'percentile in C':>18}{'cut in A':>12}")
    for wp, p, cut in zip(UPPER_WPS, percentiles, cuts_a):
        lines.append(f"{wp:<10}{p:>18.5f}{cut:>12.5f}")
    lines.append(f"(lowest TXbb in A: {txbb_a.min():.4f})")
    logger.info("\n".join(lines))


def nominal_wp_fractions(values, weights):
    """Weighted fraction of the events in the 6 bins of the nominal WP index: 0 (<= 0.65), 1 (0.65, 0.8], ... 5 (> 0.99)."""
    idx = np.searchsorted(np.asarray(WPS), values, side="left")  # number of WPs strictly below the value
    return np.array([weights[idx == i].sum() / weights.sum() for i in range(len(WPS) + 1)])


def wp_index_low(x, cuts_a):
    """Index of the region A/B rule of the workflow: 1 + number of cuts exceeded (the baseline TXbb >= 0.05 gives the 1). float32 like the workflow."""
    x = np.asarray(x, dtype=np.float32)
    return 1 + sum((x > np.float32(c)).astype(int) for c in cuts_a)


def wp_index_nominal(x):
    x = np.asarray(x, dtype=np.float32)
    return sum((x > np.float32(wp)).astype(int) for wp in WPS)


def joint_table(il, isub):
    """Joint (leading index, subleading index) occupancy as fractions of the (raw) events, 6x6 (indices 0..5)."""
    t = np.zeros((6, 6))
    np.add.at(t, (il, isub), 1)
    return t / t.sum()


def violation_fraction(t):
    """Fraction of the events with leading index < subleading index (impossible in C, where leading TXbb >= subleading TXbb)."""
    return float(sum(t[i, j] for i in range(6) for j in range(6) if i < j))


def print_joint(name, t, t_ref=None):
    lines = ["", f"Joint WP occupancy {name}, rows = leading index, columns = subleading index (fractions of the events)"]
    lines.append("      " + "".join(f"{j:>9d}" for j in range(6)))
    for i in range(6):
        lines.append(f"{i:>5d} " + "".join(f"{t[i, j]:>9.4f}" for j in range(6)))
    line = f"fraction with leading index < subleading index: {violation_fraction(t):.4f}"
    if t_ref is not None:
        line += f";  sum over cells of |this - C|: {np.abs(t - t_ref).sum():.4f}"
    lines.append(line)
    logger.info("\n".join(lines))


def joint_occupancy(lead_c, sub_c, lead_a, sub_a, cuts_lead, cuts_sub):
    """Diagnostic: joint (leading, subleading) WP index occupancy of C (nominal indices) and of A (mapped indices), raw events."""
    il_c, is_c = wp_index_nominal(lead_c), wp_index_nominal(sub_c)
    il_a, is_a = wp_index_low(lead_a, cuts_lead), wp_index_low(sub_a, cuts_sub)
    tables = []
    for name, il, isub in (("C (nominal)", il_c, is_c), ("A (mapped, marginal leading map)", il_a, is_a)):
        t = joint_table(il, isub)
        tables.append(t)
        print_joint(name, t, tables[0] if len(tables) > 1 else None)
    return tables


# ======================= Conditional leading map =======================
SENTINEL = -1.0  # cut that every event exceeds (the TXbb is never below it): used where the percentile in C is 0
N_SUB = len(WPS)  # subleading indices 1..5
PERCENTILE_EPS = 1e-9


def parse_groups(text):
    """'1;2;3,4,5' -> [[1], [2], [3, 4, 5]]: a contiguous partition of the subleading indices 1..5."""
    try:
        groups = [[int(i) for i in part.split(",")] for part in text.split(";")]
    except ValueError:
        raise ValueError(f"Cannot parse --subleading-groups '{text}'")
    if [i for g in groups for i in g] != list(range(1, N_SUB + 1)):
        raise ValueError(f"--subleading-groups must be a contiguous partition of 1..{N_SUB} in increasing order, got {groups}")
    return groups


def clean_joint(lead, sub, weights):
    """Drop entries where the leading or subleading TXbb is padded / non-finite."""
    ok = np.isfinite(lead) & np.isfinite(sub) & np.isfinite(weights) & (lead > PAD_VALUE + 1) & (sub > PAD_VALUE + 1)
    if not ok.all():
        logger.warning(f"Dropping {(~ok).sum()} of {len(ok)} entries with padded or non-finite values")
    return lead[ok], sub[ok], weights[ok]


def prepare_a(lead_a, s_a, w_a, groups):
    """Leading TXbb of A (and weights) sorted, per group of subleading indices: the cuts in A only depend on the percentiles."""
    prepared = []
    for g in groups:
        mask = np.isin(s_a, g)
        if not mask.any():
            raise ValueError(f"Region A has no events with subleading index in {g}")
        order = np.argsort(lead_a[mask], kind="stable")
        prepared.append((lead_a[mask][order], w_a[mask][order]))
    return prepared


def conditional_cuts(lead_c, s_c, w_c, a_prepared, groups, allow_empty=False, warn=True):
    """Leading cuts in A given the subleading index. Returns (percentiles, cuts), both [N_SUB, 4], row s - 1.

    For every group of subleading indices: the percentiles of the WPs 0.8/0.95/0.975/0.99 are those of the leading TXbb in C restricted to
    the group, the cuts the leading TXbb values at the same percentiles in A restricted to the group. A percentile of 0 gives the sentinel.
    With `allow_empty` a group without events in C gives NaN rows (bootstrap replicas).
    """
    n_wp = len(UPPER_WPS)
    percentiles = np.full((N_SUB, n_wp), np.nan)
    cuts = np.full((N_SUB, n_wp), np.nan)
    for g, (values_a, weights_a) in zip(groups, a_prepared):
        mask = np.isin(s_c, g)
        if not mask.any():
            if allow_empty:
                continue
            raise ValueError(f"Region C has no events with subleading index in {g}")
        p = weighted_cdf_at(lead_c[mask], w_c[mask], np.asarray(UPPER_WPS))
        real = p >= PERCENTILE_EPS
        c = np.full(n_wp, SENTINEL)
        if real.any():
            c[real] = weighted_quantile(values_a, p[real], weights_a, values_sorted=True, old_style=True)
        if warn:
            # the WPs at or below the lowest WP that the leading TXbb can exceed for the lowest s of the group must have percentile 0 in C
            for k in range(n_wp):
                if k <= min(g) - 2 and p[k] > PERCENTILE_EPS:
                    logger.warning(f"Group {g}: percentile {p[k]:.5f} at WP {UPPER_WPS[k]} should be 0 (leading TXbb >= subleading in C)")
        for s_idx in g:
            percentiles[s_idx - 1], cuts[s_idx - 1] = p, c
    return percentiles, cuts


def leading_index_conditional(lead, s, cuts_table):
    """Leading index in A/B with the conditional table (float32 like the workflow): (lead >= 0.05) + number of the cuts of row s exceeded."""
    lead32 = np.asarray(lead, dtype=np.float32)
    idx = (lead32 >= np.float32(LINEAR_B_LOW)).astype(int)
    for r in range(N_SUB):
        mask = np.asarray(s) == r + 1
        idx[mask] += sum((lead32[mask] > np.float32(c)).astype(int) for c in cuts_table[r])
    return idx


def bootstrap_cuts(lead_c, s_c, w_c, a_prepared, groups, n_replicas, seed=1):
    """Resample region C (A fixed): cuts of every replica, [n, N_SUB, 4], NaN where a group has no event in the replica."""
    rng = np.random.default_rng(seed)
    out = np.full((n_replicas, N_SUB, len(UPPER_WPS)), np.nan)
    for i in range(n_replicas):
        pick = rng.integers(0, len(lead_c), len(lead_c))
        out[i] = conditional_cuts(lead_c[pick], s_c[pick], w_c[pick], a_prepared, groups, allow_empty=True, warn=False)[1]
    return out


def print_conditional_table(percentiles, cuts, n_c, n_a, boot=None):
    lines = ["", f"Conditional leading cuts in A, one row per subleading index s (-1 = always exceeded)"]
    header = f"{'s':<3}{'C events':>10}{'A events':>10}  " + "".join(f"{'p(' + str(wp) + ')':>11}" for wp in UPPER_WPS) + "   " + "".join(f"{'cut ' + str(wp):>12}" for wp in UPPER_WPS)
    lines.append(header)
    for r in range(N_SUB):
        lines.append(f"{r + 1:<3}{n_c[r]:>10d}{n_a[r]:>10d}  " + "".join(f"{p:>11.4f}" for p in percentiles[r]) + "   " + "".join(f"{c:>12.4f}" for c in cuts[r]))
    if boot is not None:
        lines.append("")
        lines.append(f"Bootstrap of region C ({boot.shape[0]} replicas): mean +- std of the cuts, valid replicas")
        for r in range(N_SUB):
            valid = int((~np.isnan(boot[:, r, 0])).sum())
            if valid == 0:
                lines.append(f"{r + 1:<3}no replica with events in C")
                continue
            cells = []
            for k in range(len(UPPER_WPS)):
                col = boot[:, r, k]
                col = col[~np.isnan(col)]
                cells.append("      -1      " if np.all(col == SENTINEL) else f"{col.mean():>7.4f}+-{col.std():.4f}")
            lines.append(f"{r + 1:<3}" + "  ".join(cells) + f"   ({valid}/{boot.shape[0]})")
    logger.info("\n".join(lines))


def print_per_s_closure(il_c, s_c, il_a, s_a):
    """Leading-index fractions (1..5) in C and in A for every s: equal by construction up to the quantile interpolation."""
    lines = ["", "Leading index fractions given the subleading index s (indices 1..5): C | A (conditional map)"]
    for s_idx in range(1, N_SUB + 1):
        mc, ma = s_c == s_idx, s_a == s_idx
        fc = np.bincount(il_c[mc], minlength=6)[1:] / max(mc.sum(), 1)
        fa = np.bincount(il_a[ma], minlength=6)[1:] / max(ma.sum(), 1)
        lines.append(f"s={s_idx} (C {int(mc.sum()):>6d}, A {int(ma.sum()):>6d}):  " + " ".join(f"{x:.4f}" for x in fc) + "  |  " + " ".join(f"{x:.4f}" for x in fa))
    logger.info("\n".join(lines))


# ======================= Plot =======================
def make_plot(txbb_c, w_c, txbb_a, w_a, percentiles, cuts_a, path, higgs="subleading"):
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
    ax.set_xlabel(f"{higgs.capitalize()} Higgs TXbb")
    ax.set_ylabel("Weighted CDF")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Quantile map of the {higgs} TXbb working points from C to A")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot written to {path}")


def make_plot_conditional(lead_c, s_c, w_c, lead_a, s_a, w_a, percentiles, cuts, path):
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

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for r, ax in enumerate(axes.flat[:N_SUB]):
        mc, ma = s_c == r + 1, s_a == r + 1
        if mc.any():
            ax.plot(*ecdf(lead_c[mc], w_c[mc]), label=f"C (n={int(mc.sum())})", color="tab:orange")
        if ma.any():
            ax.plot(*ecdf(lead_a[ma], w_a[ma]), label=f"A (n={int(ma.sum())})", color="tab:blue")
        for wp, p, cut in zip(UPPER_WPS, percentiles[r], cuts[r]):
            if np.isnan(p) or cut == SENTINEL:
                continue
            ax.plot([wp, cut], [p, p], color="grey", ls="--", lw=0.8)
            ax.plot([wp], [p], "o", color="tab:orange", ms=4)
            ax.plot([cut], [p], "o", color="tab:blue", ms=4)
        ax.set_title(f"subleading index s = {r + 1}")
        ax.set_xlabel("Leading Higgs TXbb")
        ax.set_ylabel("Weighted CDF")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("Conditional quantile map of the leading-Higgs working points, C to A")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot written to {path}")


# ======================= Main =======================
def main(argv=None):
    args = get_args(argv)
    if args.conditional:
        return main_conditional(args)
    leading = args.higgs == "leading"
    columns = [args.txbb_column, args.weight_column]
    sub_cuts = None
    if args.subleading_cuts:
        with open(args.subleading_cuts) as f:
            sub_cuts = [float(c) for c in json.load(f)["cuts_A"]]
        columns.append(args.subleading_column)
    data = load_regions(args.input_data, args.data_pattern, [args.region_c, args.region_a], columns)
    txbb_c, w_c = clean(data[args.region_c][args.txbb_column], data[args.region_c][args.weight_column])
    txbb_a, w_a = clean(data[args.region_a][args.txbb_column], data[args.region_a][args.weight_column])
    logger.info(f"{args.higgs.capitalize()} Higgs, column {args.txbb_column}")
    logger.info(f"Region C: {len(txbb_c)} events, sum w = {w_c.sum():.1f}, TXbb in [{txbb_c.min():.4f}, {txbb_c.max():.4f}]")
    logger.info(f"Region A: {len(txbb_a)} events, sum w = {w_a.sum():.1f}, TXbb in [{txbb_a.min():.4f}, {txbb_a.max():.4f}]")
    if txbb_c.min() < WPS[0] - 1e-6:
        raise ValueError(f"Region C has TXbb values below the lowest WP {WPS[0]} (min {txbb_c.min():.4f}): is this the 50GeV setting?")
    if leading:
        # the leading TXbb in A is not bounded by the lowest WP (it is at least the subleading one): the map is applied by region
        frac_above = float(w_a[txbb_a > WPS[0]].sum() / w_a.sum())
        logger.info(f"Fraction of region A with the leading TXbb above {WPS[0]}: {frac_above:.4f}")
    elif txbb_a.max() > WPS[0] + 1e-6:
        raise ValueError(f"Region A has TXbb values above {WPS[0]} (max {txbb_a.max():.4f}): is this the 50GeV setting?")

    percentiles, cuts_a = quantile_map(txbb_c, w_c, txbb_a, w_a, UPPER_WPS)
    if np.any(np.diff(cuts_a) < 0):
        raise ValueError(f"The cuts in A are not monotonic: {cuts_a}")
    n_top = int((txbb_a > cuts_a[-1]).sum())
    if n_top < MIN_EVENTS_TOP_WP:
        logger.warning(f"Only {n_top} raw events in region A above the highest cut {cuts_a[-1]:.4f}: the top WP is statistically limited")

    frac_c, frac_a_new, frac_a_old = closure_table(txbb_c, w_c, txbb_a, w_a, cuts_a)
    frac_a_nominal = nominal_wp_fractions(txbb_a, w_a) if leading else None
    print_table(percentiles, cuts_a, frac_c, frac_a_new, frac_a_old, txbb_a, frac_a_nominal)

    if sub_cuts is not None:
        cols = {}
        for region in (args.region_c, args.region_a):
            lead = np.asarray(data[region][args.txbb_column], dtype=np.float64)
            sub = np.asarray(data[region][args.subleading_column], dtype=np.float64)
            ok = (lead > PAD_VALUE + 1) & (sub > PAD_VALUE + 1) & np.isfinite(lead) & np.isfinite(sub)
            cols[region] = (lead[ok], sub[ok])
        joint_occupancy(*cols[args.region_c], *cols[args.region_a], cuts_a, sub_cuts)

    os.makedirs(args.output, exist_ok=True)
    suffix = "_leading" if leading else ""
    result = {
        "higgs": args.higgs,
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
    out_json = os.path.join(args.output, f"txbb_cdf_wp_cuts{suffix}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Cuts written to {out_json}")
    if not args.no_plot:
        make_plot(txbb_c, w_c, txbb_a, w_a, percentiles, cuts_a, os.path.join(args.output, f"txbb_cdf_wp_cuts{suffix}.png"), args.higgs)


def main_conditional(args):
    with open(args.subleading_cuts) as f:
        sub_cuts = [float(c) for c in json.load(f)["cuts_A"]]
    columns = [args.txbb_column, args.subleading_column, args.weight_column]
    data = load_regions(args.input_data, args.data_pattern, [args.region_c, args.region_a], columns)
    lead_c, sub_c, w_c = clean_joint(*(data[args.region_c][c] for c in columns))
    lead_a, sub_a, w_a = clean_joint(*(data[args.region_a][c] for c in columns))
    logger.info(f"Conditional leading map, columns {args.txbb_column} (leading) and {args.subleading_column}, subleading cuts {sub_cuts}")
    logger.info(f"Region C: {len(lead_c)} events, sum w = {w_c.sum():.1f}; region A: {len(lead_a)} events, sum w = {w_a.sum():.1f}")
    if lead_c.min() < WPS[0] - 1e-6 or sub_c.min() < WPS[0] - 1e-6:
        raise ValueError(f"Region C has TXbb values below the lowest WP {WPS[0]} (leading min {lead_c.min():.4f}, subleading min {sub_c.min():.4f}): is this the 50GeV setting?")
    if sub_a.max() > WPS[0] + 1e-6:
        raise ValueError(f"Region A has a subleading TXbb above {WPS[0]} (max {sub_a.max():.4f}): is this the 50GeV setting?")

    # subleading index: nominal in C, mapped with the subleading cuts in A (same float32 rules as the workflow)
    s_c, s_a = wp_index_nominal(sub_c), wp_index_low(sub_a, sub_cuts)
    if s_c.min() < 1:
        raise ValueError("Region C has events with the subleading index 0 (subleading TXbb not above the lowest WP)")
    n_c, n_a = np.bincount(s_c, minlength=N_SUB + 1)[1:], np.bincount(s_a, minlength=N_SUB + 1)[1:]
    for g in args.groups:
        if n_c[[i - 1 for i in g]].sum() < args.min_events_c:
            logger.warning(f"Only {int(n_c[[i - 1 for i in g]].sum())} events in C for the subleading indices {g}: the cuts of this group are statistically limited")
    il_c = wp_index_nominal(lead_c)
    n_order = int((il_c < s_c).sum())
    logger.info(f"Region C: {n_order} events with the leading index below the subleading one (expected 0)")

    a_prepared = prepare_a(lead_a, s_a, w_a, args.groups)
    percentiles, cuts = conditional_cuts(lead_c, s_c, w_c, a_prepared, args.groups)
    if np.any(np.diff(cuts, axis=1) < 0):
        raise ValueError(f"The conditional cuts are not monotonic in a row: {cuts}")
    boot = bootstrap_cuts(lead_c, s_c, w_c, a_prepared, args.groups, args.bootstrap) if args.bootstrap > 0 else None
    print_conditional_table(percentiles, cuts, n_c, n_a, boot)

    # joint occupancy: C nominal, A with the marginal leading map (previous option) and with the conditional one
    il_a_cond = leading_index_conditional(lead_a, s_a, cuts)
    marginal_cuts = quantile_map(lead_c, w_c, lead_a, w_a, UPPER_WPS)[1]
    il_a_marg = (lead_a.astype(np.float32) >= np.float32(LINEAR_B_LOW)).astype(int) + sum((lead_a.astype(np.float32) > np.float32(c)).astype(int) for c in marginal_cuts)
    t_c = joint_table(il_c, s_c)
    print_joint("C (nominal)", t_c)
    print_joint("A (mapped, marginal leading map)", joint_table(il_a_marg, s_a), t_c)
    t_cond = joint_table(il_a_cond, s_a)
    print_joint("A (mapped, conditional leading map)", t_cond, t_c)
    print_per_s_closure(il_c, s_c, il_a_cond, s_a)
    n_viol = int((il_a_cond < s_a).sum())
    logger.info(f"Region A, conditional map: {n_viol} events with the leading index below the subleading one"
                + (" (expected 0 without pooling)" if all(len(g) == 1 for g in args.groups) else " (pooled groups: not guaranteed)"))
    if n_viol and all(len(g) == 1 for g in args.groups):
        raise ValueError(f"{n_viol} events have the leading index below the subleading one although no subleading indices are pooled")

    os.makedirs(args.output, exist_ok=True)
    result = {
        "higgs": "leading",
        "conditional": True,
        "conditioned_on": "subleading working-point index s (1..5): nominal in C, 1 + number of subleading_cuts_A exceeded in A",
        "region_setting": args.region_setting,
        "subleading_groups": args.groups,
        "wps_C": UPPER_WPS,
        "sentinel": SENTINEL,
        "percentiles": [[float(x) for x in row] for row in percentiles],
        "cuts_A": [[float(x) for x in row] for row in cuts],
        "subleading_cuts_A": sub_cuts,
        "region_C": args.region_c,
        "region_A": args.region_a,
        "n_events_C": [int(x) for x in n_c],
        "n_events_A": [int(x) for x in n_a],
        "sum_w_C": float(w_c.sum()),
        "sum_w_A": float(w_a.sum()),
        "min_events_c": args.min_events_c,
    }
    if boot is not None:
        result["bootstrap"] = {
            "n_replicas": int(boot.shape[0]),
            "cuts_mean": [[None if np.all(np.isnan(boot[:, r, k])) else float(np.nanmean(boot[:, r, k])) for k in range(len(UPPER_WPS))] for r in range(N_SUB)],
            "cuts_std": [[None if np.all(np.isnan(boot[:, r, k])) else float(np.nanstd(boot[:, r, k])) for k in range(len(UPPER_WPS))] for r in range(N_SUB)],
        }
    out_json = os.path.join(args.output, "txbb_cdf_wp_cuts_leading_cond.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Cuts written to {out_json}")
    if not args.no_plot:
        make_plot_conditional(lead_c, s_c, w_c, lead_a, s_a, w_a, percentiles, cuts, os.path.join(args.output, "txbb_cdf_wp_cuts_leading_cond.png"))


if __name__ == "__main__":
    main()
