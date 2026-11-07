"""Optimise the boosted category cuts (cat1, catvbf, cat2, cat3) on a ggF and a VBF discriminant
plus the subleading TXbb score.

The category logic mirrors configs/HH4b_common/custom_cut_functions_common.py (hh4b_boosted_category_*),
with the BDT scores replaced by a ggF and a VBF discriminant:
    cat1   : txbb > txbb_c1 & D_ggF > bdt_ggf_c1
    catvbf : !cat1 & txbb > txbb_vbf & D_VBF > bdt_vbf
    cat2   : !cat1 & !catvbf & ( (txbb_c2a < txbb < txbb_c1 & D_ggF > bdt_ggf_c1) | (txbb > txbb_c1 & bdt_ggf_c2b < D_ggF < bdt_ggf_c1) )
    cat3   : !cat1 & !catvbf & !cat2 & txbb > txbb_c2a & D_ggF > bdt_ggf_bkg

With --no-txbb-cut all TXbb cuts are set to NO_TXBB_CUT (-1, always passed) and only the score cuts are optimised:
    cat1   : D_ggF > bdt_ggf_c1
    catvbf : !cat1 & D_VBF > bdt_vbf
    cat2   : !cat1 & !catvbf & bdt_ggf_c2b < D_ggF < bdt_ggf_c1
    cat3   : !cat1 & !catvbf & !cat2 & D_ggF > bdt_ggf_bkg
The existing category functions give exactly these categories with txbb_c1 = txbb_vbf = txbb_c2a = -1, so the output
cuts can be used in the config as they are.

Processes:
    - ggF / VBF signal and ttbar: MC in boosted_signal_region_D (TXbb not morphed)
    - QCD: data in boosted_control_region_B_postW (morphed TXbb, morphing weights), normalised with
      (sum(data in boosted_signal_sideband_region_C) - sum(ttbar MC in boosted_signal_sideband_region_C))
      / sum(data in boosted_control_sideband_region_A_postW), i.e. the same estimate as in build_datacards.py

Figure of merit (lower is better), evaluated per category (and optionally per mass bin), combined over all categories:
    - limit       : expected 95% CL upper limit on the signal strength from the Asimov exclusion significance
                    (generalises 2*sqrt(B)/S to small B and includes the background uncertainty)
    - significance: 1 / combined Asimov discovery significance
    - simple      : 2 / sqrt(sum_i S_i^2 / B_i), i.e. the combination of 2*sqrt(B)/S over the categories
The ggF and VBF figures of merit are combined as F_ggF^(1-w) * F_VBF^w (w = --vbf-weight).

The cuts are optimised by coordinate descent over parameter pairs on a grid. The events are binned once
on the grid edges, so each evaluation is only a sum over bins.
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


parser = argparse.ArgumentParser(description="Optimise the boosted category cuts on the sig/bkg DNN scores and the subleading TXbb", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input-data", type=str, required=True, help="Directory with the coffea output files (one per dataset)")
parser.add_argument("-o", "--output", type=str, default="./category_optimisation", help="Output directory")
parser.add_argument(
    "-d",
    "--discriminant",
    choices=["ratio", "raw", "bdt"],
    default="ratio",
    help="ratio: X/(X+QCD+TTbar) for X=ggF,VBF; raw: the softmax outputs themselves (also separate ggF from VBF); "
         "bdt: the current boosted BDT scores (validation / baseline)",
)
parser.add_argument("-m", "--metric", choices=["limit", "significance", "simple"], default="limit", help="Figure of merit")
parser.add_argument("-w", "--vbf-weight", type=float, default=0.5, help="Weight w of the VBF figure of merit: F = F_ggF^(1-w) * F_VBF^w")
parser.add_argument("--ggf-pattern", type=str, default="c2-0p00-kl-1p00-kt-1p00", help="Substring selecting the ggF signal dataset(s)")
parser.add_argument("--vbf-pattern", type=str, default="CV-1-C2V-1-C3-1_", help="Substring selecting the VBF signal dataset(s), e.g. 'CV-1-C2V-0-C3-1_' for kappa_2V=0")
parser.add_argument("--other-signal-as-bkg", action="store_true", default=False, help="Count the other HH signal as background when computing the figure of merit of one signal")
parser.add_argument("--mass-column", type=str, default="HiggsSubLeading_mass", help="Fit variable, used for the mass bins")
parser.add_argument("--mass-bins", type=int, default=1, help="Number of equal-width bins in the fit variable (1: counting experiment)")
parser.add_argument("--bkg-syst", type=float, default=0.0, help="Relative systematic uncertainty on the background per bin")
parser.add_argument("--no-bkg-stat", action="store_true", default=False, help="Do not include the background MC/data statistical uncertainty in the figure of merit")
parser.add_argument("--min-qcd-events", type=int, default=10, help="Minimum number of raw QCD (B_postW data) events per category")
parser.add_argument("--min-bkg", type=float, default=0.0, help="Minimum weighted total background per category")
parser.add_argument("--n-score-points", type=int, default=40, help="Number of grid points for the score cuts (quantiles of the signal)")
parser.add_argument("--txbb-grid", type=float, nargs=3, default=[0.65, 0.995, 0.005], help="TXbb grid: start stop step")
parser.add_argument("--txbb-values", type=float, nargs="+", default=None,
                    help="Explicit TXbb cut values (overrides --txbb-grid), e.g. the TXbb working points")
parser.add_argument("--txbb-wps", action="store_true", default=False,
                    help="Restrict the TXbb cuts to the working-point edges 0.8 0.95 0.975 0.99 (shorthand for --txbb-values). "
                         "The bkg morphing only sees the digitised TXbb (btagBBTXbb_morph_5wp), so cuts between the edges are not modelled")
parser.add_argument("--txbb-scan", action="store_true", default=False,
                    help="Scan all TXbb cut combinations on the grid (score cuts re-optimised for each) and plot the figure of merit "
                         "as a heatmap. Only sensible for a coarse TXbb grid, e.g. with --txbb-wps")
parser.add_argument("--no-txbb-cut", action="store_true", default=False,
                    help="Do not cut on the subleading TXbb in the signal-region categories: all TXbb cuts are set to -1 "
                         "(always passed) and only the DNN/BDT score cuts are optimised")
parser.add_argument("--scan-cat3", action=argparse.BooleanOptionalAction, default=True,
                    help="Also optimise the loose ggF cut of cat3 (with --no-scan-cat3 it is fixed to the loosest grid point)")
parser.add_argument("--max-iter", type=int, default=10, help="Maximum number of coordinate-descent iterations")
parser.add_argument("--holdout", action="store_true", default=False, help="Optimise on even event numbers, report on odd ones (weights x2) to check for overtuning")
parser.add_argument("--cache", type=str, default=None, help="npz file with the loaded events: read if it exists, else written after loading (delete it when the inputs change)")
parser.add_argument("--region-setting", choices=["40GeV", "50GeV"], default="50GeV", help="Region setting for the BDT baseline cuts")
parser.add_argument("--signal-region", type=str, default="boosted_signal_region_D")
parser.add_argument("--qcd-region", type=str, default="boosted_control_region_B_postW")
parser.add_argument("--norm-num-region", type=str, default="boosted_signal_sideband_region_C")
parser.add_argument("--norm-denom-region", type=str, default="boosted_control_sideband_region_A_postW")
parser.add_argument("--profile", action="store_true", default=False,
                    help="Profile of the figure of merit for each cut (all other cuts re-optimised at every point), with the TXbb cuts "
                         "restricted to the working points vs free on --profile-txbb-grid. Shows the cost of the restriction")
parser.add_argument("--profile-free-step", type=float, default=0.01, help="Step of the scanned TXbb value in the free profiles")
parser.add_argument("--profile-score-stride", type=int, default=2, help="Use every n-th score grid point as scanned value in the profiles")
parser.add_argument("--workers", type=int, default=32, help="Worker processes for the profiles")
parser.add_argument("--profile-2d", action="store_true", default=False,
                    help="2D map of the figure of merit in (txbb_c1, txbb_c2a) with txbb_vbf at the optimum and the score cuts re-optimised per cell")
parser.add_argument("--profile-txbb-grid", type=float, nargs=3, default=[0.65, 0.995, 0.005], help="TXbb grid of the 1D profiles: start stop step")
parser.add_argument("--profile-2d-step", type=float, default=0.01, help="TXbb step of the 2D map")
args = parser.parse_args()
TXBB_WPS = [0.8, 0.95, 0.975, 0.99]
# TXbb cut value meaning "no cut": below the range of the (morphed) TXbb score, so `txbb > NO_TXBB_CUT` always holds
NO_TXBB_CUT = -1.0
if args.no_txbb_cut:
    incompatible = [opt for opt, on in [("--txbb-wps", args.txbb_wps), ("--txbb-values", args.txbb_values is not None),
                                        ("--txbb-scan", args.txbb_scan), ("--profile", args.profile),
                                        ("--profile-2d", args.profile_2d)] if on]
    if incompatible:
        parser.error(f"--no-txbb-cut cannot be combined with {', '.join(incompatible)}")
    # A single TXbb grid point at NO_TXBB_CUT: every TXbb cut is snapped to it and stays there
    args.txbb_values = [NO_TXBB_CUT]
if args.txbb_wps:
    if args.txbb_values is not None:
        parser.error("--txbb-wps and --txbb-values are mutually exclusive")
    args.txbb_values = TXBB_WPS

CATEGORIES = ["none", "cat1", "catvbf", "cat2", "cat3"]
# Parameter names follow _get_params in configs/HH4b_common/custom_cuts_common.py, so that the optimised cuts
# can be copied directly into the `optimised_signal_cuts` config option
PARAMS = ["txbb_c1", "bdt_ggf_c1", "txbb_vbf", "bdt_vbf", "txbb_c2a", "bdt_ggf_c2b", "bdt_ggf_bkg"]
# Which grid axis each parameter lives on
PARAM_AXIS = {"txbb_c1": "t", "txbb_vbf": "t", "txbb_c2a": "t", "bdt_ggf_c1": "g", "bdt_ggf_c2b": "g", "bdt_ggf_bkg": "g", "bdt_vbf": "v"}
Z_CL = 1.96  # Z of the expected 95% CL limit (2*sqrt(B)/S convention uses ~2)

SCORE_COLUMNS = {
    "ratio": ["events_sig_bkg_dnn_score_ggF", "events_sig_bkg_dnn_score_VBF", "events_sig_bkg_dnn_score_QCD", "events_sig_bkg_dnn_score_TTbar"],
    "raw": ["events_sig_bkg_dnn_score_ggF", "events_sig_bkg_dnn_score_VBF"],
    "bdt": ["events_boosted_bdt_score", "events_boosted_bdt_vbf_score"],
}


# ======================= Loading =======================
def load_group(files, categories, columns):
    """Load the columns of a group of datasets, merged over the datasets: {category: {column: array}}."""
    if not files:
        return {}
    cat_col, _ = get_columns_from_files(
        files, "nominal", lambda c: c in columns, category_filter=lambda c: c in categories,
    )
    for category in categories:
        if category not in cat_col:
            raise ValueError(f"Category {category} not found for {files[0]} (and others)")
        missing = [c for c in columns if c not in cat_col[category]]
        if missing:
            raise ValueError(f"Columns {missing} not found in category {category}. Available: {sorted(cat_col[category])}")
    return cat_col


def get_discriminants(cols):
    if args.discriminant == "bdt":
        return cols["events_boosted_bdt_score"], cols["events_boosted_bdt_vbf_score"]
    ggf, vbf = cols["events_sig_bkg_dnn_score_ggF"], cols["events_sig_bkg_dnn_score_VBF"]
    if args.discriminant == "raw":
        return ggf, vbf
    bkg = cols["events_sig_bkg_dnn_score_QCD"] + cols["events_sig_bkg_dnn_score_TTbar"]
    return ggf / (ggf + bkg), vbf / (vbf + bkg)


def build_process(cols, txbb_column, scale=1.0):
    d_ggf, d_vbf = get_discriminants(cols)
    return {
        "t": np.asarray(cols[txbb_column], dtype=np.float64),
        "g": np.asarray(d_ggf, dtype=np.float64),
        "v": np.asarray(d_vbf, dtype=np.float64),
        "m": np.asarray(cols[args.mass_column], dtype=np.float64),
        "w": np.asarray(cols["weight"], dtype=np.float64) * scale,
        "event": np.asarray(cols["events_event"]),
    }


def load_processes():
    files = sorted(os.path.join(args.input_data, f) for f in os.listdir(args.input_data) if f.endswith(".coffea") and f != "output_all.coffea")
    groups = {
        "ggF": [f for f in files if "GluGlu" in f and args.ggf_pattern in f],
        "VBF": [f for f in files if "VBFHH" in f and args.vbf_pattern in f],
        "TT": [f for f in files if "TTto" in f],
        "DATA": [f for f in files if "DATA" in f],
    }
    for name, group in groups.items():
        logger.info(f"{name}: {[os.path.basename(f) for f in group]}")
        if not group and name != "TT":
            raise ValueError(f"No files found for {name}")

    columns = ["weight", "events_event", args.mass_column, "HiggsSubLeading_btagBBTXbb", "HiggsSubLeading_btagBBTXbb_morph"] + SCORE_COLUMNS[args.discriminant]
    processes = {}
    for name in ["ggF", "VBF", "TT"]:
        cat_col = load_group(groups[name], [args.signal_region], columns)
        if cat_col:
            processes[name] = build_process(cat_col[args.signal_region], "HiggsSubLeading_btagBBTXbb")

    # QCD: data-driven from the reweighted B region, normalised with (C - ttbar MC in C) / A_postW
    norm = load_group(groups["DATA"], [args.norm_num_region, args.norm_denom_region], ["weight"])
    sum_num, sum_denom = norm[args.norm_num_region]["weight"].sum(), norm[args.norm_denom_region]["weight"].sum()
    norm_tt = load_group(groups["TT"], [args.norm_num_region], ["weight"])
    sum_tt = norm_tt[args.norm_num_region]["weight"].sum() if norm_tt else 0.0
    norm_factor = (sum_num - sum_tt) / sum_denom
    logger.info(f"QCD normalisation ({args.norm_num_region} - TT) / {args.norm_denom_region} = "
                f"({sum_num:.1f} - {sum_tt:.1f}) / {sum_denom:.1f} = {norm_factor:.5f}")
    cat_col = load_group(groups["DATA"], [args.qcd_region], columns)
    # In the control region the TXbb is morphed into the signal region range (as in the category cuts)
    processes["QCD"] = build_process(cat_col[args.qcd_region], "HiggsSubLeading_btagBBTXbb_morph", scale=norm_factor)
    for name, proc in processes.items():
        logger.info(f"{name}: {len(proc['w'])} events, yield {proc['w'].sum():.3f}")
    return processes


def select_events(proc, parity):
    """Select even (parity=0) or odd (parity=1) event numbers, doubling the weights. parity=None: all events."""
    if parity is None:
        return proc
    mask = (proc["event"] % 2) == parity
    out = {k: v[mask] for k, v in proc.items()}
    out["w"] = out["w"] * 2
    return out


# ======================= Categories =======================
def assign_categories(t, g, v, cuts):
    """Category index (see CATEGORIES) for each entry. Works on values with value cuts, or on grid indices
    with cuts = index - 0.5 (then x > cut <=> index >= k)."""
    cat1 = (t > cuts["txbb_c1"]) & (g > cuts["bdt_ggf_c1"])
    catvbf = ~cat1 & (t > cuts["txbb_vbf"]) & (v > cuts["bdt_vbf"])
    cat2 = ~cat1 & ~catvbf & (
        ((t > cuts["txbb_c2a"]) & (t < cuts["txbb_c1"]) & (g > cuts["bdt_ggf_c1"]))
        | ((t > cuts["txbb_c1"]) & (g > cuts["bdt_ggf_c2b"]) & (g < cuts["bdt_ggf_c1"]))
    )
    cat3 = ~cat1 & ~catvbf & ~cat2 & (t > cuts["txbb_c2a"]) & (g > cuts["bdt_ggf_bkg"])
    category = np.zeros(np.shape(t), dtype=np.int64)
    for i, mask in enumerate([cat1, catvbf, cat2, cat3], start=1):
        category[mask] = i
    return category


def event_yields(processes, cuts, mass_edges):
    """Yields with exact (unbinned) cuts: {proc: (sumw, sumw2, n)} each of shape (n_categories, n_mass_bins)."""
    n_mass = len(mass_edges) - 1
    yields = {}
    for name, proc in processes.items():
        category = assign_categories(proc["t"], proc["g"], proc["v"], cuts)
        idx = category * n_mass + mass_index(proc["m"], mass_edges)
        size = len(CATEGORIES) * n_mass
        yields[name] = tuple(
            np.bincount(idx, weights=w, minlength=size).reshape(len(CATEGORIES), n_mass)
            for w in (proc["w"], proc["w"] ** 2, np.ones_like(proc["w"]))
        )
    return yields


def mass_index(m, mass_edges):
    return np.clip(np.searchsorted(mass_edges, m, side="right") - 1, 0, len(mass_edges) - 2)


class BinnedYields:
    """Events binned on the cut grid, so that any cut on the grid is a sum over bins."""

    def __init__(self, processes, grids, mass_edges):
        self.names = list(processes)
        self.n_mass = len(mass_edges) - 1
        # Bin k of an axis covers [grid[k-1], grid[k]), bin 0 is everything below grid[0]
        shape = tuple(len(grids[a]) + 1 for a in "tgv")
        it, ig, iv = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
        self.index = {"t": it.ravel(), "g": ig.ravel(), "v": iv.ravel()}
        n_bins = int(np.prod(shape))
        columns = []
        for name in self.names:
            proc = processes[name]
            flat = np.ravel_multi_index(
                tuple(np.searchsorted(grids[a], proc[a], side="right") for a in "tgv"), shape
            ) * self.n_mass + mass_index(proc["m"], mass_edges)
            for w in (proc["w"], proc["w"] ** 2, np.ones_like(proc["w"])):
                columns.append(np.bincount(flat, weights=w, minlength=n_bins * self.n_mass).reshape(n_bins, self.n_mass))
        # (n_bins, n_proc * 3 * n_mass)
        self.histogram = np.concatenate(columns, axis=1)
        self.n_bins = n_bins

    def yields(self, cut_indices):
        cuts = {p: cut_indices[p] - 0.5 for p in PARAMS}
        category = assign_categories(self.index["t"], self.index["g"], self.index["v"], cuts)
        one_hot = np.zeros((len(CATEGORIES), self.n_bins))
        one_hot[category, np.arange(self.n_bins)] = 1.0
        summed = (one_hot @ self.histogram).reshape(len(CATEGORIES), len(self.names), 3, self.n_mass)
        return {name: tuple(summed[:, i, j, :] for j in range(3)) for i, name in enumerate(self.names)}


class FastYields:
    """Same yields as BinnedYields, but from tail cumulative sums over the (t, g, v) grid bins: every category is a
    combination of boxes in index space (inclusion-exclusion), so a yield evaluation is a few lookups instead of a sum
    over all bins. Needed for fine TXbb grids. Requires valid() cuts (txbb_c2a <= txbb_c1, bkg <= c2b <= c1)."""

    def __init__(self, processes, grids, mass_edges):
        self.names = list(processes)
        self.n_mass = len(mass_edges) - 1
        self.shape = tuple(len(grids[a]) + 1 for a in "tgv")
        n_cols = len(self.names) * 3 * self.n_mass
        # C[i, j, k] = sum over bins with t >= i, g >= j, v >= k; padded with zeros at the upper ends
        self.C = np.zeros(tuple(n + 1 for n in self.shape) + (n_cols,))
        n_bins = int(np.prod(self.shape))
        col = 0
        for name in self.names:
            proc = processes[name]
            flat = np.ravel_multi_index(
                tuple(np.searchsorted(grids[a], proc[a], side="right") for a in "tgv"), self.shape
            ) * self.n_mass + mass_index(proc["m"], mass_edges)
            for w in (proc["w"], proc["w"] ** 2, np.ones_like(proc["w"])):
                counts = np.bincount(flat, weights=w, minlength=n_bins * self.n_mass)
                self.C[:-1, :-1, :-1, col:col + self.n_mass] = counts.reshape(self.shape + (self.n_mass,))
                col += self.n_mass
        for axis in range(3):
            flipped = np.flip(self.C, axis=axis)
            np.cumsum(flipped, axis=axis, out=flipped)
        self.zero = np.zeros(n_cols)

    def box(self, t0, t1, g0, g1, v0, v1):
        """Sum over t0 <= t < t1, g0 <= g < g1, v0 <= v < v1 (bin indices)."""
        if t0 >= t1 or g0 >= g1 or v0 >= v1:
            return self.zero
        C = self.C
        return (C[t0, g0, v0] - C[t1, g0, v0] - C[t0, g1, v0] - C[t0, g0, v1]
                + C[t1, g1, v0] + C[t1, g0, v1] + C[t0, g1, v1] - C[t1, g1, v1])

    def yields(self, cut_indices):
        # x > cut <=> bin index >= k (see BinnedYields.yields)
        a1, b1 = cut_indices["txbb_c1"], cut_indices["bdt_ggf_c1"]
        a2, d = cut_indices["txbb_vbf"], cut_indices["bdt_vbf"]
        a3, b2, b3 = cut_indices["txbb_c2a"], cut_indices["bdt_ggf_c2b"], cut_indices["bdt_ggf_bkg"]
        nt, ng, nv = self.shape
        B = self.box
        cat1 = B(a1, nt, b1, ng, 0, nv)
        # catvbf = (t >= a2 & v >= d) minus its overlap with cat1
        vbf_box_cat1 = B(max(a1, a2), nt, b1, ng, d, nv)
        catvbf = B(a2, nt, 0, ng, d, nv) - vbf_box_cat1
        # cat2 = (a3 <= t < a1 & g >= b1) + (t >= a1 & b2 <= g < b1), both disjoint from cat1, minus their overlap with catvbf
        cat2 = (B(a3, a1, b1, ng, 0, nv) + B(a1, nt, b2, b1, 0, nv)
                - B(max(a3, a2), a1, b1, ng, d, nv) - B(max(a1, a2), nt, b2, b1, d, nv))
        # cat3 = (t >= a3 & g >= b3) minus cat1, catvbf and cat2, which (for valid cuts) overlap it as follows
        catvbf_in_q = B(max(a2, a3), nt, b3, ng, d, nv) - vbf_box_cat1
        cat3 = B(a3, nt, b3, ng, 0, nv) - cat1 - catvbf_in_q - cat2
        none = self.C[0, 0, 0] - cat1 - catvbf - cat2 - cat3
        summed = np.stack([none, cat1, catvbf, cat2, cat3]).reshape(len(CATEGORIES), len(self.names), 3, self.n_mass)
        return {name: tuple(summed[:, i, j, :] for j in range(3)) for i, name in enumerate(self.names)}


# ======================= Figure of merit =======================
def z_discovery(s, b, var):
    """Squared Asimov discovery significance with background uncertainty (Cowan)."""
    s, b, var = np.broadcast_arrays(s, b, var)
    z2 = np.zeros_like(s)
    ok = (b > 0) & (s > 0)
    no_unc = ok & (var <= 1e-12 * b**2)
    z2[no_unc] = 2 * ((s + b) * np.log1p(s / b) - s)[no_unc]
    unc = ok & ~no_unc
    s_, b_, v_ = s[unc], b[unc], var[unc]
    # Profile likelihood ratio test & Asimov
    z2[unc] = 2 * ((s_ + b_) * np.log((s_ + b_) * (b_ + v_) / (b_**2 + (s_ + b_) * v_))
                   - b_**2 / v_ * np.log1p(v_ * s_ / (b_ * (b_ + v_))))
    return np.clip(z2, 0, None)


def z_exclusion(s, b, var):
    """Squared Asimov exclusion significance (of s+b, assuming b only) with background uncertainty."""
    s, b, var = np.broadcast_arrays(s, b, var)
    z2 = np.zeros_like(s)
    ok = (b > 0) & (s > 0)
    no_unc = ok & (var <= 1e-12 * b**2)
    z2[no_unc] = 2 * (s[no_unc] - b[no_unc] * np.log1p(s[no_unc] / b[no_unc]))
    unc = ok & ~no_unc
    s_, b_, v_ = s[unc], b[unc], var[unc]
    x = np.sqrt((s_ + b_) ** 2 - 4 * s_ * b_ * v_ / (b_ + v_))
    # b - s + x and b + s - x cancel catastrophically for s >> b (e.g. a bin with b ~ 1e-20 from cancelling weights),
    # which gave log(0) = -inf and a Z^2 that is not monotonic in s. Use the exact rationalised forms instead:
    #   x - (s - b) = 4 s b^2 / ((b + v) (x + s - b)),   (b + s) - x = 4 s b v / ((b + v) (b + s + x))
    b_minus_s_plus_x = b_ - s_ + x
    big = s_ > b_
    b_minus_s_plus_x[big] = 4 * s_[big] * b_[big] ** 2 / ((b_[big] + v_[big]) * (x[big] + s_[big] - b_[big]))
    b_plus_s_minus_x = 4 * s_ * b_ * v_ / ((b_ + v_) * (b_ + s_ + x))
    z2[unc] = (2 * (s_ - b_ * np.log((b_ + s_ + x) / (2 * b_)) - b_**2 / v_ * np.log(b_minus_s_plus_x / (2 * b_)))
               - b_plus_s_minus_x * (1 + b_ / v_))
    return np.clip(z2, 0, None)


def expected_limit(s, b, var):
    """Signal strength mu for which the combined Asimov exclusion significance is Z_CL."""
    if np.sum(s) <= 0:
        return np.inf
    lo, hi = 0.0, 1.0
    while np.sum(z_exclusion(hi * s, b, var)) < Z_CL**2:
        hi *= 2
        if hi > 1e9:
            return np.inf
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if np.sum(z_exclusion(mid * s, b, var)) < Z_CL**2:
            lo = mid
        else:
            hi = mid
    return hi


def figure_of_merit(s, b, var):
    """Lower is better. s, b, var have shape (n_categories, n_mass_bins) without the 'none' category."""
    if args.metric == "limit":
        return expected_limit(s, b, var)
    if args.metric == "significance":
        z2 = np.sum(z_discovery(s, b, var))
        return 1 / np.sqrt(z2) if z2 > 0 else np.inf
    ok = b > 0
    sum_s2_b = np.sum(s[ok] ** 2 / b[ok])
    return 2 / np.sqrt(sum_s2_b) if sum_s2_b > 0 else np.inf


def passes_requirements(yields):
    bkg = yields["QCD"][0][1:] + (yields["TT"][0][1:] if "TT" in yields else 0)
    return not (np.any(yields["QCD"][2][1:].sum(axis=1) < args.min_qcd_events) or np.any(bkg.sum(axis=1) < args.min_bkg))


def evaluate(yields, enforce=True):
    """Combined figure of merit and the per-signal ones. Infinite if a category fails the statistics requirements (and enforce)."""
    qcd_w, qcd_w2 = yields["QCD"][0][1:], yields["QCD"][1][1:]
    bkg, bkg_w2 = qcd_w.copy(), qcd_w2.copy()
    if "TT" in yields:
        bkg += yields["TT"][0][1:]
        bkg_w2 += yields["TT"][1][1:]
    if enforce and not passes_requirements(yields):
        return np.inf, np.inf, np.inf
    foms = {}
    for signal, other in [("ggF", "VBF"), ("VBF", "ggF")]:
        b, var = bkg, (0 if args.no_bkg_stat else bkg_w2) + (args.bkg_syst * bkg) ** 2
        if args.other_signal_as_bkg:
            b = b + yields[other][0][1:]
        foms[signal] = figure_of_merit(yields[signal][0][1:], b, var)
    combined = foms["ggF"] ** (1 - args.vbf_weight) * foms["VBF"] ** args.vbf_weight
    return combined, foms["ggF"], foms["VBF"]


# ======================= Optimisation =======================
def make_grids(processes):
    efficiencies = np.linspace(0.99, 0.01, args.n_score_points)
    if args.txbb_values is not None:
        grids = {"t": np.unique(np.asarray(args.txbb_values, dtype=np.float64))}
    else:
        start, stop, step = args.txbb_grid
        grids = {"t": np.round(np.arange(start, stop + 0.5 * step, step), 6)}
    for axis, signal in [("g", "ggF"), ("v", "VBF")]:
        # Cut values at fixed signal efficiencies: the scores pile up close to 1 for signal
        values, weights = processes[signal][axis], processes[signal]["w"]
        grids[axis] = np.unique(weighted_quantile(values, 1 - efficiencies, weights=weights))
    return grids


def initial_cuts(grids, processes):
    if args.discriminant == "bdt":
        # Baseline: the BDT cuts the input was produced with, else the current ones in custom_cuts_common.py
        try:
            with open(os.path.join(args.input_data, "config.json")) as f:
                category_cuts = json.load(f)["categories"]["categories"][f"{args.signal_region}_cat1"]
            params = next(c["params"] for c in category_cuts if c["name"] == "hh4b_boosted_category_1")
            logger.info("Starting from the BDT cuts in config.json of the input")
        except (OSError, KeyError, StopIteration) as e:
            from configs.HH4b_common.custom_cuts_common import _get_params
            logger.info(f"No BDT cuts found in config.json of the input ({e!r}), starting from _get_params({args.region_setting})")
            params = _get_params(args.region_setting)
        params = dict(params)
        if args.no_txbb_cut:
            params.update({p: NO_TXBB_CUT for p in TXBB_PARAMS})
        return {p: params[p] for p in PARAMS}

    def at_eff(axis, signal, eff):
        return weighted_quantile(processes[signal][axis], [1 - eff], weights=processes[signal]["w"])[0]
    return {
        "txbb_c1": 0.95, "bdt_ggf_c1": at_eff("g", "ggF", 0.3), "txbb_vbf": 0.85, "bdt_vbf": at_eff("v", "VBF", 0.3),
        "txbb_c2a": 0.85, "bdt_ggf_c2b": at_eff("g", "ggF", 0.6), "bdt_ggf_bkg": grids["g"][0],
    }


def to_indices(cuts, grids):
    """Snap value cuts to the nearest grid point: x > cut <=> bin index >= k + 1, cut = grid[k]."""
    return {p: int(np.argmin(np.abs(grids[PARAM_AXIS[p]] - cuts[p]))) + 1 for p in PARAMS}


def to_values(indices, grids):
    return {p: float(grids[PARAM_AXIS[p]][indices[p] - 1]) for p in PARAMS}


def valid(indices):
    return indices["txbb_c2a"] <= indices["txbb_c1"] and indices["bdt_ggf_bkg"] <= indices["bdt_ggf_c2b"] <= indices["bdt_ggf_c1"]


def optimise(binned, grids, start, fixed=(), verbose=True):
    """Coordinate descent over parameter pairs. Parameters in `fixed` keep their start value."""
    if args.no_txbb_cut:
        # No TXbb axis to scan: pair the score cuts with each other instead, since they compete for the same events
        pairs = [("bdt_ggf_c1", "bdt_vbf"), ("bdt_ggf_c1", "bdt_ggf_c2b"), ("bdt_vbf", "bdt_ggf_c2b")]
        if args.scan_cat3:
            pairs.append(("bdt_ggf_c2b", "bdt_ggf_bkg"))
    else:
        pairs = [("txbb_c1", "bdt_ggf_c1"), ("txbb_vbf", "bdt_vbf"), ("txbb_c2a", "bdt_ggf_c2b")]
        if args.scan_cat3:
            pairs.append(("bdt_ggf_bkg",))
    pairs = [tuple(p for p in pair if p not in fixed) for pair in pairs]
    pairs = [pair for pair in pairs if pair]
    best = dict(start)
    best_fom = evaluate(binned.yields(best))[0]
    if verbose:
        logger.info(f"Start: F = {best_fom:.4g}")
    for iteration in range(args.max_iter):
        improved = False
        for pair in pairs:
            ranges = [range(1, len(grids[PARAM_AXIS[p]]) + 1) for p in pair]
            for point in np.array(np.meshgrid(*ranges, indexing="ij")).reshape(len(pair), -1).T:
                trial = dict(best, **{p: int(k) for p, k in zip(pair, point)})
                if not valid(trial):
                    continue
                fom = evaluate(binned.yields(trial))[0]
                if fom < best_fom * (1 - 1e-6):
                    best, best_fom, improved = trial, fom, True
            if verbose:
                logger.info(f"Iteration {iteration}, scanned {pair}: F = {best_fom:.4g}, cuts {to_values(best, grids)}")
        if not improved:
            break
    return best


TXBB_PARAMS = ["txbb_c1", "txbb_vbf", "txbb_c2a"]


def scan_txbb(binned, grids, start, processes, mass_edges):
    """For every valid combination of the TXbb cuts on the grid, optimise the score cuts (on the optimisation
    sample) and evaluate the figure of merit with exact cuts on `processes`.
    Returns a list of {txbb_c1, txbb_vbf, txbb_c2a, F, F_ggF, F_VBF, cuts}."""
    n_t = len(grids["t"])
    results = []
    for k_c1 in range(1, n_t + 1):
        for k_vbf in range(1, n_t + 1):
            for k_c2a in range(1, k_c1 + 1):  # txbb_c2a <= txbb_c1
                trial = dict(start, txbb_c1=k_c1, txbb_vbf=k_vbf, txbb_c2a=k_c2a)
                if not valid(trial):
                    continue
                best = optimise(binned, grids, trial, fixed=TXBB_PARAMS, verbose=False)
                cuts = to_values(best, grids)
                combined, fom_ggf, fom_vbf = evaluate(event_yields(processes, cuts, mass_edges))
                results.append({p: cuts[p] for p in TXBB_PARAMS} | {"F": combined, "F_ggF": fom_ggf, "F_VBF": fom_vbf, "cuts": cuts})
                logger.info(f"TXbb scan c1={cuts['txbb_c1']}, vbf={cuts['txbb_vbf']}, c2a={cuts['txbb_c2a']}: "
                            f"F = {combined:.4g} (ggF {fom_ggf:.4g}, VBF {fom_vbf:.4g})")
    return results


def plot_txbb_scan(scan, txbb_values, output_file):
    """Heatmap of the figure of merit over the TXbb cuts: one column per txbb_vbf, x = txbb_c1, y = txbb_c2a.
    Rows: combined, ggF and VBF figure of merit, each shown relative to the best cell of the row (1 = best)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = list(txbb_values)
    n = len(values)
    rows = [("F", "combined F"), ("F_ggF", "F ggF"), ("F_VBF", "F VBF")]
    fig, axes = plt.subplots(len(rows), n, figsize=(3.6 * n + 1.5, 3.4 * len(rows)), squeeze=False, constrained_layout=True)
    # Single-hue sequential scale, dark = best (lowest F)
    cmap = matplotlib.colormaps["Blues_r"].copy()
    cmap.set_bad("#f0f0f0")
    for r, (key, label) in enumerate(rows):
        finite = [s[key] for s in scan if np.isfinite(s[key])]
        f_best = min(finite) if finite else np.nan
        vmax = min(np.nanpercentile([f / f_best for f in finite], 90), 2.0) if finite else 2.0
        for c, vbf in enumerate(values):
            ax = axes[r, c]
            grid = np.full((n, n), np.nan)  # [c2a index, c1 index]
            status = np.full((n, n), "", dtype=object)
            for s in scan:
                if s["txbb_vbf"] != vbf:
                    continue
                i, j = values.index(s["txbb_c2a"]), values.index(s["txbb_c1"])
                if np.isfinite(s[key]):
                    grid[i, j] = s[key] / f_best
                    status[i, j] = f"{grid[i, j]:.3f}\n({s[key]:.3g})"
                else:
                    status[i, j] = "fails\nstats"
            image = ax.imshow(np.ma.masked_invalid(grid), origin="lower", cmap=cmap, vmin=1.0, vmax=vmax, aspect="auto")
            for i in range(n):
                for j in range(n):
                    text = status[i, j] or ("c2a > c1" if i > j else "")
                    if not text:
                        continue
                    dark_cell = np.isfinite(grid[i, j]) and (grid[i, j] - 1.0) < 0.5 * (vmax - 1.0)
                    ax.text(j, i, text, ha="center", va="center", fontsize=8,
                            color="white" if dark_cell else "#333333")
            ax.set_xticks(range(n), [f"{v:g}" for v in values])
            ax.set_yticks(range(n), [f"{v:g}" for v in values])
            ax.set_xlabel("txbb_c1")
            if c == 0:
                ax.set_ylabel(f"{label}\n\ntxbb_c2a")
            ax.set_title(f"txbb_vbf = {vbf:g}", fontsize=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.colorbar(image, ax=axes[r, :].tolist(), label=f"{label} / best ({f_best:.4g})", shrink=0.9)
    fig.suptitle(f"Figure of merit ({args.metric}, lower is better) per TXbb cut combination, score cuts re-optimised per cell")
    for ext in ("png", "pdf"):
        fig.savefig(f"{output_file}.{ext}", dpi=150)
    plt.close(fig)
    logger.info(f"Saved TXbb scan heatmap to {output_file}.png/.pdf")


# ======================= Profile scans around the optimum =======================
COLOR_FREE, COLOR_RESTRICTED = "#2a78d6", "#eb6834"
COLOR_INK, COLOR_MUTED = "#333333", "#8a8a8a"
FOM_KEYS = ("F", "F_ggF", "F_VBF")
# Filled in __main__ before the worker pool is forked: {"restricted": (FastYields, grids), "free": (FastYields, grids)}
PROFILE_CONTEXT = {}


def fine_txbb_grid(start, stop, step):
    values = np.round(np.arange(start, stop + 0.5 * step, step), 6)
    return values[values < 1.0]


def profile_point(processes_opt, processes_eval, grids, cuts, fixed, mass_edges):
    """Figure of merit of `cuts` as they are, and with the cuts not in `fixed` re-optimised (on processes_opt).
    The events are binned on a TXbb grid containing only the TXbb cuts of `cuts`, so any TXbb value is exact."""
    as_is = evaluate(event_yields(processes_eval, cuts, mass_edges))
    point_grids = dict(grids, t=np.unique([cuts[p] for p in TXBB_PARAMS]))
    binned = BinnedYields(processes_opt, point_grids, mass_edges)
    best = optimise(binned, point_grids, to_indices(cuts, point_grids), fixed=fixed, verbose=False)
    reopt_cuts = to_values(best, point_grids)
    reopt = evaluate(event_yields(processes_eval, reopt_cuts, mass_edges))
    return dict(zip(FOM_KEYS, as_is)), dict(zip(FOM_KEYS, reopt)), reopt_cuts


def make_valid(cuts, param):
    """Move the other cuts so that the ordering constraints hold after `param` was changed."""
    c = dict(cuts)
    if param == "txbb_c1":
        c["txbb_c2a"] = min(c["txbb_c2a"], c["txbb_c1"])
    elif param == "txbb_c2a":
        c["txbb_c1"] = max(c["txbb_c1"], c["txbb_c2a"])
    elif param == "bdt_ggf_c1":
        c["bdt_ggf_c2b"] = min(c["bdt_ggf_c2b"], c["bdt_ggf_c1"])
        c["bdt_ggf_bkg"] = min(c["bdt_ggf_bkg"], c["bdt_ggf_c2b"])
    elif param == "bdt_ggf_c2b":
        c["bdt_ggf_c1"] = max(c["bdt_ggf_c1"], c["bdt_ggf_c2b"])
        c["bdt_ggf_bkg"] = min(c["bdt_ggf_bkg"], c["bdt_ggf_c2b"])
    elif param == "bdt_ggf_bkg":
        c["bdt_ggf_c2b"] = max(c["bdt_ggf_c2b"], c["bdt_ggf_bkg"])
        c["bdt_ggf_c1"] = max(c["bdt_ggf_c1"], c["bdt_ggf_c2b"])
    return c


def descend(approach, start_cuts, fixed=()):
    """Coordinate descent on the grid of `approach` ("restricted": TXbb on the working points, "free": fine TXbb grid).
    Returns (cuts, {F, F_ggF, F_VBF})."""
    yields_obj, grids = PROFILE_CONTEXT[approach]
    best = optimise(yields_obj, grids, to_indices(start_cuts, grids), fixed=fixed, verbose=False)
    return to_values(best, grids), dict(zip(FOM_KEYS, evaluate(yields_obj.yields(best))))


def _profile_task(task):
    """Minimum of F over all cuts except `param`, which is fixed to `value`, from several starting points."""
    approach, param, value, starts = task
    best_cuts, best_f = None, {k: np.inf for k in FOM_KEYS}
    for start in starts:
        cuts, f = descend(approach, make_valid(dict(start, **{param: value}), param), fixed=(param,))
        if f["F"] < best_f["F"]:
            best_cuts, best_f = cuts, f
    return {"approach": approach, "param": param, "value": float(value), "cuts": best_cuts, **best_f}


def profile_compare(start_cuts):
    """Profiles of the figure of merit for each cut: at every value of the cut, all other cuts are re-optimised,
    once with the TXbb cuts restricted to the working points and once with them free on the fine grid.
    The restricted configurations are a subset of the free ones, and are used as starting points of the free
    minimisation, so free <= restricted holds point by point (up to the coordinate descent reaching it)."""
    from multiprocessing import Pool

    restricted_opt, f_restricted = descend("restricted", start_cuts)
    free_opt, f_free = descend("free", restricted_opt)
    if f_free["F"] > f_restricted["F"]:
        free_opt, f_free = restricted_opt, f_restricted
    logger.info(f"Profiles: restricted optimum F = {f_restricted['F']:.4g}, free optimum F = {f_free['F']:.4g} "
                f"({f_free['F'] / f_restricted['F'] - 1:+.2%}), free cuts {free_opt}")

    _, grids_restricted = PROFILE_CONTEXT["restricted"]
    _, grids_free = PROFILE_CONTEXT["free"]
    free_txbb_values = fine_txbb_grid(args.profile_txbb_grid[0], args.profile_txbb_grid[1], args.profile_free_step)
    free_txbb_values = grids_free["t"][np.isin(np.round(grids_free["t"], 6), np.round(free_txbb_values, 6))]

    def values_for(param, approach):
        axis = PARAM_AXIS[param]
        if axis == "t":
            return grids_restricted["t"] if approach == "restricted" else free_txbb_values
        return grids_free[axis][::args.profile_score_stride]

    with Pool(args.workers) as pool:
        tasks = [("restricted", p, v, [restricted_opt]) for p in PARAMS for v in values_for(p, "restricted")]
        restricted = pool.map(_profile_task, tasks)
        logger.info(f"Profiles: {len(restricted)} restricted points done")
        by_point = {(r["param"], round(r["value"], 6)): r["cuts"] for r in restricted if r["cuts"] is not None}
        tasks = []
        for p in PARAMS:
            for v in values_for(p, "free"):
                starts = [restricted_opt, free_opt]
                if (p, round(float(v), 6)) in by_point:
                    starts.append(by_point[(p, round(float(v), 6))])
                tasks.append(("free", p, v, starts))
        free = pool.map(_profile_task, tasks)
        logger.info(f"Profiles: {len(free)} free points done")

    profiles = {p: {"restricted": [r for r in restricted if r["param"] == p], "free": [r for r in free if r["param"] == p]}
                for p in PARAMS}
    for p in PARAMS:
        for approach in ("restricted", "free"):
            finite = [r for r in profiles[p][approach] if np.isfinite(r["F"])]
            if finite:
                best = min(finite, key=lambda r: r["F"])
                logger.info(f"Profile {p} ({approach}): minimum F = {best['F']:.4g} at {best['value']:.5g}")
    return {"restricted_optimum": {"cuts": restricted_opt, **f_restricted},
            "free_optimum": {"cuts": free_opt, **f_free}, "profiles": profiles}


def profile_2d(processes_opt, processes_eval, grids, opt_cuts, mass_edges):
    """F in (txbb_c1, txbb_c2a), txbb_vbf at the optimum, score cuts re-optimised per cell."""
    start, stop, _ = args.profile_txbb_grid
    values = fine_txbb_grid(start, stop, args.profile_2d_step)
    cells = []
    for c1 in values:
        for c2a in values[values <= c1 + 1e-9]:
            cuts = dict(opt_cuts, txbb_c1=float(c1), txbb_c2a=float(c2a))
            _, reopt, reopt_cuts = profile_point(processes_opt, processes_eval, grids, cuts, TXBB_PARAMS, mass_edges)
            cells.append({"txbb_c1": float(c1), "txbb_c2a": float(c2a), "reopt": reopt, "reopt_cuts": reopt_cuts})
        logger.info(f"2D map: txbb_c1 = {c1:.3f} done")
    return {"values": values.tolist(), "cells": cells}


def _shade_failing(ax, x, f):
    """Light band over the x ranges where the figure of merit is infinite (statistics requirement not met)."""
    if len(x) < 2:
        return
    half = 0.5 * np.min(np.diff(x))
    for xi, fi in zip(x, f):
        if not np.isfinite(fi):
            ax.axvspan(xi - half, xi + half, color="#e9e9e9", lw=0, zorder=0)


def plot_profile_compare(result, output_file):
    """One panel per cut: F (relative to the restricted optimum) as a function of the cut, all other cuts
    re-optimised at every point. Orange: TXbb cuts restricted to the working points (discrete for the TXbb cuts),
    blue: TXbb cuts free on the fine grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f_ref = result["restricted_optimum"]["F"]
    profiles = result["profiles"]
    params = [p for p in PARAMS if profiles[p]["free"] or profiles[p]["restricted"]]
    ncols = 4
    nrows = int(np.ceil((len(params) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.7 * nrows), squeeze=False, constrained_layout=True)
    axes = axes.ravel()

    def series(points):
        x = np.array([p["value"] for p in points])
        f = np.array([p["F"] for p in points]) / f_ref
        return x, np.where(np.isfinite(f), f, np.nan)

    all_f = np.concatenate([series(profiles[p][a])[1] for p in params for a in ("free", "restricted")])
    y_low = min(0.95, np.nanmin(all_f) - 0.01)
    y_top = min(np.nanmax(all_f) * 1.02, 1.6)

    for ax, param in zip(axes, params):
        is_txbb = PARAM_AXIS[param] == "t"
        if is_txbb:
            for wp in TXBB_WPS:
                ax.axvline(wp, color=COLOR_MUTED, lw=0.8, ls=":", zorder=1)
        ax.axhline(1.0, color=COLOR_MUTED, lw=0.8, zorder=1)
        x_free, f_free = series(profiles[param]["free"])
        x_res, f_res = series(profiles[param]["restricted"])
        _shade_failing(ax, x_free, f_free)
        ax.plot(x_free, f_free, color=COLOR_FREE, lw=2, zorder=2, label="TXbb free (fine grid)")
        # Discrete for the TXbb cuts (only the working points are allowed), a curve for the score cuts
        ax.plot(x_res, f_res, color=COLOR_RESTRICTED, lw=0 if is_txbb else 1.5, marker="o", ms=8 if is_txbb else 4,
                mec="white", mew=1.5, zorder=3, label="TXbb restricted to WPs")
        lines = []
        for x, f, name, color in ((x_res, f_res, "restricted", COLOR_RESTRICTED), (x_free, f_free, "free", COLOR_FREE)):
            if np.isfinite(f).any():
                k = int(np.nanargmin(f))
                lines.append(f"{name} min {f[k]:.3f} at {x[k]:.4g}")
        ax.text(0.98, 0.96, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=8, color=COLOR_INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
        ax.set_ylim(y_low, y_top)
        ax.set_xlabel(param)
        ax.set_ylabel("F / F_opt(restricted)")
        ax.grid(True, color="#eeeeee", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    legend_ax = axes[len(params)]
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="upper left", frameon=False)
    f_free_opt = result["free_optimum"]["F"]
    legend_ax.text(0.0, 0.62,
                   f"Every point: all other cuts re-optimised.\n"
                   f"F_opt(restricted) = {f_ref:.4g}\n"
                   f"F_opt(free) = {f_free_opt:.4g} ({f_free_opt / f_ref - 1:+.1%})\n"
                   "dotted lines: TXbb working points\n"
                   "grey bands: fails min. QCD statistics\n\n"
                   "In-sample values: the free curve has more\n"
                   "freedom to tune to fluctuations. F away from\n"
                   "the WP edges also assumes the background\n"
                   "model is valid there (needs a retraining).",
                   transform=legend_ax.transAxes, va="top", fontsize=9, color=COLOR_INK)
    for ax in axes[len(params):]:
        ax.axis("off")
    fig.suptitle(f"Profiled figure of merit ({args.metric}, lower is better): TXbb cuts restricted to the WPs vs free")
    for ext in ("png", "pdf"):
        fig.savefig(f"{output_file}.{ext}", dpi=150)
    plt.close(fig)
    logger.info(f"Saved profile comparison to {output_file}.png/.pdf")


def plot_profile_2d(map_2d, opt_cuts, f_opt, output_file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.array(map_2d["values"])
    n = len(values)
    grid = np.full((n, n), np.nan)  # [c2a, c1]
    for cell in map_2d["cells"]:
        i, j = np.argmin(np.abs(values - cell["txbb_c2a"])), np.argmin(np.abs(values - cell["txbb_c1"]))
        if np.isfinite(cell["reopt"]["F"]):
            grid[i, j] = cell["reopt"]["F"] / f_opt
    step = values[1] - values[0] if n > 1 else 0.01
    edges = np.append(values - 0.5 * step, values[-1] + 0.5 * step)

    fig, ax = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    cmap = matplotlib.colormaps["Blues_r"].copy()
    cmap.set_bad("#f0f0f0")
    finite = grid[np.isfinite(grid)]
    vmax = min(np.percentile(finite, 90), 1.5) if finite.size else 1.5
    mesh = ax.pcolormesh(edges, edges, np.ma.masked_invalid(grid), cmap=cmap, vmin=min(1.0, finite.min()) if finite.size else 1.0, vmax=vmax)
    levels = [lvl for lvl in (1.0, 1.01, 1.02, 1.05, 1.1, 1.2) if finite.size and finite.min() < lvl < vmax]
    if levels:
        contours = ax.contour(values, values, grid, levels=levels, colors=COLOR_INK, linewidths=0.8)
        ax.clabel(contours, fmt=lambda v: f"{v:g}", fontsize=8)
    for wp in TXBB_WPS:
        ax.axvline(wp, color="white", lw=1.2, ls="--")
        ax.axhline(wp, color="white", lw=1.2, ls="--")
    ax.plot(opt_cuts["txbb_c1"], opt_cuts["txbb_c2a"], "*", ms=16, color="#eb6834", mec="white", mew=1.5, label="optimum (input cuts)")
    if finite.size:
        i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
        ax.plot(values[j], values[i], "X", ms=11, color=COLOR_INK, mec="white", mew=1.5,
                label=f"map minimum ({values[j]:g}, {values[i]:g}): {grid[i, j]:.3f}")
    ax.set_xlabel("txbb_c1")
    ax.set_ylabel("txbb_c2a")
    ax.set_title(f"F / F_opt in (txbb_c1, txbb_c2a), txbb_vbf = {opt_cuts['txbb_vbf']:g}, score cuts re-optimised\n"
                 "dashed: TXbb working points, grey: fails min. statistics or c2a > c1", fontsize=10)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.colorbar(mesh, ax=ax, label=f"F / F_opt (F_opt = {f_opt:.4g})")
    for ext in ("png", "pdf"):
        fig.savefig(f"{output_file}.{ext}", dpi=150)
    plt.close(fig)
    logger.info(f"Saved 2D map to {output_file}.png/.pdf")


def report(title, processes, cuts, mass_edges):
    yields = event_yields(processes, cuts, mass_edges)
    combined, fom_ggf, fom_vbf = evaluate(yields, enforce=False)
    passes = passes_requirements(yields)
    lines = [
        f"===== {title} =====",
        f"F = {combined:.4g} (ggF: {fom_ggf:.4g}, VBF: {fom_vbf:.4g}, metric: {args.metric})"
        + ("" if passes else "  WARNING: fails the minimum background statistics requirement"),
        "cuts: " + ", ".join(f"{p}={cuts[p]:.5g}" for p in PARAMS),
    ]
    header = f"{'category':>8} {'ggF':>9} {'VBF':>9} {'QCD':>10} {'+-':>8} {'n_QCD':>7} {'TT':>9} {'S_ggF/sqrtB':>11} {'S_VBF/sqrtB':>11}"
    lines.append(header)
    table = {}
    for i, category in enumerate(CATEGORIES[1:], start=1):
        row = {name: float(yields[name][0][i].sum()) for name in yields}
        qcd_unc = float(np.sqrt(yields["QCD"][1][i].sum()))
        n_qcd = int(yields["QCD"][2][i].sum())
        b = row["QCD"] + row.get("TT", 0.0)
        lines.append(
            f"{category:>8} {row['ggF']:9.4f} {row['VBF']:9.4f} {row['QCD']:10.3f} {qcd_unc:8.3f} {n_qcd:7d} {row.get('TT', 0.0):9.4f} "
            f"{row['ggF'] / np.sqrt(b) if b > 0 else np.nan:11.4f} {row['VBF'] / np.sqrt(b) if b > 0 else np.nan:11.4f}"
        )
        table[category] = row | {"QCD_unc": qcd_unc, "n_QCD": n_qcd}
    logger.info("\n" + "\n".join(lines))
    return {"F": combined, "F_ggF": fom_ggf, "F_VBF": fom_vbf, "passes_requirements": passes, "cuts": cuts, "yields": table}


if __name__ == "__main__":
    CACHE_KEYS = ["discriminant", "mass_column", "ggf_pattern", "vbf_pattern", "signal_region",
              "qcd_region", "norm_num_region", "norm_denom_region"]
    cache_meta = json.dumps({k: getattr(args, k) for k in CACHE_KEYS}, sort_keys=True)

    if args.cache and os.path.exists(args.cache):
        logger.info(f"Reading the events from the cache {args.cache}")
        with np.load(args.cache) as cache:
            stored = str(cache["__meta__"]) if "__meta__" in cache.files else None
            if stored != cache_meta:
                raise ValueError(f"Cache {args.cache} was made with different settings:\n"
                                 f"  cache: {stored}\n  now:   {cache_meta}\n"
                                 f"Delete it or use another --cache file.")
            processes_all = {}
            for key in cache.files:
                if key == "__meta__":
                    continue
                name, field = key.split("/")
                processes_all.setdefault(name, {})[field] = cache[key]
    else:
        processes_all = load_processes()
        if args.cache:
            np.savez(args.cache, __meta__=cache_meta,
                     **{f"{name}/{field}": arr for name, proc in processes_all.items() for field, arr in proc.items()})
            logger.info(f"Saved the events to the cache {args.cache}")

    all_mass = np.concatenate([p["m"] for p in processes_all.values()])
    mass_edges = np.linspace(all_mass.min(), all_mass.max() + 1e-6, args.mass_bins + 1)

    if args.holdout:
        processes_opt = {n: select_events(p, 0) for n, p in processes_all.items()}
        processes_test = {n: select_events(p, 1) for n, p in processes_all.items()}
    else:
        processes_opt = processes_all

    grids = make_grids(processes_opt)
    logger.info(f"Grid sizes: TXbb {len(grids['t'])}, ggF {len(grids['g'])}, VBF {len(grids['v'])}")
    binned = BinnedYields(processes_opt, grids, mass_edges)

    start_cuts = initial_cuts(grids, processes_opt)
    if args.txbb_values is not None:
        # Only the given TXbb values are allowed, also for the starting point
        snapped = to_values(to_indices(start_cuts, grids), grids)
        start_cuts = dict(start_cuts, **{p: snapped[p] for p in TXBB_PARAMS})
    results = {"args": vars(args), "start": report("Start (exact cuts)", processes_all, start_cuts, mass_edges)}

    best = optimise(binned, grids, to_indices(start_cuts, grids))
    best_cuts = to_values(best, grids)
    results["optimised"] = report("Optimised", processes_all, best_cuts, mass_edges)
    if args.holdout:
        results["optimised_train"] = report("Optimised, optimisation half", processes_opt, best_cuts, mass_edges)
        results["optimised_test"] = report("Optimised, held-out half", processes_test, best_cuts, mass_edges)
        results["start_test"] = report("Start, held-out half", processes_test, start_cuts, mass_edges)

    os.makedirs(args.output, exist_ok=True)
    if args.no_txbb_cut:
        suffix = "_notxbb"
    else:
        suffix = "_txbbwps" if args.txbb_wps else ("_txbbvalues" if args.txbb_values is not None else "")
    if args.txbb_scan:
        scan = scan_txbb(binned, grids, best, processes_all, mass_edges)
        results["txbb_scan"] = scan
        finite = [s for s in scan if np.isfinite(s["F"])]
        if finite:
            scan_best = min(finite, key=lambda s: s["F"])
            results["txbb_scan_best"] = report("Best of the TXbb scan", processes_all, scan_best["cuts"], mass_edges)
            if scan_best["F"] < results["optimised"]["F"] * (1 - 1e-6):
                logger.warning(f"The TXbb scan found better cuts (F = {scan_best['F']:.4g}) than the coordinate descent "
                               f"(F = {results['optimised']['F']:.4g}): use txbb_scan_best")
        plot_txbb_scan(scan, grids["t"], os.path.join(args.output, f"txbb_scan_{args.discriminant}_{args.metric}{suffix}"))

    if args.profile or args.profile_2d:
        f_opt = results["optimised"]["F"]
        if args.profile:
            # In-sample on all events: restricted (TXbb on the WPs) vs free (fine TXbb grid), same score grids
            for approach, t_values in (("restricted", np.asarray(TXBB_WPS, dtype=np.float64)),
                                       ("free", fine_txbb_grid(*args.profile_txbb_grid))):
                approach_grids = dict(grids, t=t_values)
                PROFILE_CONTEXT[approach] = (FastYields(processes_all, approach_grids, mass_edges), approach_grids)
            results["profile_compare"] = profile_compare(best_cuts)
            plot_profile_compare(results["profile_compare"],
                                 os.path.join(args.output, f"profile_compare_{args.discriminant}_{args.metric}{suffix}"))
        if args.profile_2d:
            results["profile_2d"] = profile_2d(processes_opt, processes_all, grids, best_cuts, mass_edges)
            plot_profile_2d(results["profile_2d"], best_cuts, f_opt,
                            os.path.join(args.output, f"profile_2d_{args.discriminant}_{args.metric}{suffix}"))

    output_file = os.path.join(args.output, f"optimised_cuts_{args.discriminant}_{args.metric}{suffix}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Saved results to {output_file}")
