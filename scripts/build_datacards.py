import argparse
import copy
import json
import logging
import os
import numpy as np
import re
import yaml
from collections import defaultdict
from pathlib import Path

import hist
from coffea.util import load
from pocket_coffea.utils.stat import (
    Datacard,
    DataProcess,
    DataProcesses,
    MCProcess,
    MCProcesses,
    Systematics,
    SystematicUncertainty,
)
from pocket_coffea.utils.stat.shape_manipulation import (
    add_envelope_variation,
    add_quadrature_variation,
)
from pocket_coffea.utils.histogram import rebin_hist

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


parser = argparse.ArgumentParser(description="Build datacards from pocket-coffea outputs")
parser.add_argument(
    "-i",
    "--input-data",
    type=str,
    nargs="+",
    required=True,
    help="Input directory for data with coffea files or coffea files themselves (needs a '/' in the end, if not a single file)",
)
parser.add_argument(
    "-r2",
    "--run2",
    action="store_true",
    help="If running with Run2 method",
    default=False,
)
parser.add_argument(
    "-sn",
    "--separate-normalisation",
    action="store_true",
    help="If true, normalise signal region and control region separately. Otherwise, normalise everything to control region.",
    default=False,
)
parser.add_argument(
    "-o", "--output", type=str, help="Output directory", default="./datacards"
)
parser.add_argument(
    "-b",
    "--boosted",
    action="store_true",
    help="If a boosted analysis is being performed",
    default=False,
)
parser.add_argument(
    "-sb",
    "--single-bin",
    action="store_true",
    help="rebins fit histogram to only have a single bin",
    default=False,
)
parser.add_argument(
    "-nb",
    "--nbins",
    type=int,
    help="rebins fit histogram to this number of bins (must divide the original number of bins). Default: keep original binning",
    default=None,
)
parser.add_argument(
    "-mu",
    "--max-bkg-unc",
    type=float,
    help="Minimum-statistics rebinning: merge adjacent bins (per category, left to right) until the relative statistical "
         "uncertainty of the total background in every bin is at most this value (e.g. 0.3). Applied after --nbins.",
    default=None,
)
parser.add_argument(
    "-my",
    "--min-bkg-yield",
    type=float,
    help="Minimum-statistics rebinning: additionally require at least this total background yield per bin (e.g. 1.0)",
    default=None,
)
parser.add_argument(
    "-fr",
    "--fit-range",
    type=float,
    nargs=2,
    metavar=("LOW", "HIGH"),
    help="Cut the fit variable histogram to the range [LOW, HIGH] (e.g. 60 200). Events outside are dropped. "
         "Applied before --nbins and the minimum-statistics rebinning.",
    default=None,
)
parser.add_argument(
    "-qu",
    "--qcd-norm-unc",
    type=float,
    help="Additional flat normalisation uncertainty (lnN, e.g. 0.1 for 10%%) on the data-driven background, "
         "uncorrelated between categories (e.g. for the non-closure of the background estimate)",
    default=None,
)
parser.add_argument(
    "-fs",
    "--fit-sidebands",
    action="store_true",
    help="Boosted only: also build one-bin datacards for every category in the QCD sideband "
         "(boosted_signal_sideband_region_C_<cat>, QCD from A_<cat>_postW) and in the ttbar sideband "
         "(boosted_signal_ttbar_region_E_<cat>, QCD from F_<cat>_postW). Needs a coffea output produced with "
         "boosted_fit_sidebands=True. Implies --rate-params.",
    default=False,
)
parser.add_argument(
    "-rp",
    "--rate-params",
    action="store_true",
    help="Boosted only: free rateParams per category for QCD (rate_QCD_<cat>) and ttbar (rate_ttbar_<cat>), shared "
         "between the signal region and the sidebands of the same category. The pure normalisation lnNs of these "
         "processes (QCD norm, ttbar cross section, lumi on ttbar) are then dropped, as the rateParams absorb them.",
    default=False,
)
parser.add_argument(
    "--qcd-ttsb",
    choices=["lnN", "shared"],
    default="lnN",
    help="QCD normalisation in the ttbar sideband: 'lnN' = own lnN of --qcd-ttsb-unc around the ABCD prediction, "
         "'shared' = the same rate_QCD_<cat> as in the signal region",
)
parser.add_argument(
    "--qcd-ttsb-unc",
    type=float,
    default=0.3,
    help="lnN uncertainty on the QCD prediction in the ttbar sideband (with --qcd-ttsb lnN)",
)

args = parser.parse_args()
if args.fit_sidebands:
    args.rate_params = True
if (args.fit_sidebands or args.rate_params) and not args.boosted:
    parser.error("--fit-sidebands / --rate-params are only implemented for the boosted analysis (--boosted)")
if args.fit_range is not None and args.fit_range[0] >= args.fit_range[1]:
    parser.error(f"--fit-range: LOW must be smaller than HIGH, got {args.fit_range}")
if args.single_bin:
    args.nbins = 1

LUMI_YAML = Path(__file__).parent.parent / "configs" / "HH4b_common" / "params" / "lumi_systematics.yaml"
with open(LUMI_YAML) as f:
    lumi_cfg = yaml.safe_load(f)
lumi_sys_values = lumi_cfg["lumi_systematics"]

# Values taken from previous datacard:
# https://gitlab.cern.ch/cms-analysis/hig/HIG-24-010/datacards/-/blob/master/AN-23-184/merged_channel/vbf/combined.txt?ref_type=heads
# Recommendations in case of update:
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHHRecommendations
# Asymmetric lnN values are written to the datacard as value[0]/value[1], which combine reads as
# kappa(-1 sigma)/kappa(+1 sigma): give them in the order (down, up).
BR_HBB = (0.975, 1.025)  # <-- not found
PDF_HIGGS_QQHH = 1.027  # 1.027<-- not found
PDF_HIGGS_GGHH = 1.021  # 1.0192
ALPHAS_GGHH = 1.021  # 1.0149
ALPHAS_QQHH = 1.021  # <-- not found
SCALE_GGHH = (0.951, 1.021)  # (0.9515, 1.0207)
MTOP_GGHH = (0.82, 1.04)  # (0.84, 1.05)
SCALE_QQHH = (0.9997, 1.0005)  # TODO: check value, LHCHWG HH recommendations (VBF HH: +0.03% / -0.04%)
# ttbar inclusive cross section (TOP PAG NNLO+NNLL reference), only needed as long as ttbar has no rateParam
SCALE_TTBAR = (0.964, 1.024)  # TODO: check value
PDF_GG = 1.03  # TODO: check value (PDF + alpha_s) <-- Do I need one for VBF too?

# Theory acceptance uncertainties, built from the lhe_scale / lhe_pdf weight variations
# (normalised to the inclusive cross section, so they do not overlap with the lnN above)
LHE_SCALE_VARIATIONS = [f"lhe_scale_{var}{shift}" for var in ("muR", "muF", "muRmuF") for shift in ("Up", "Down")]
LHE_PDF_EIGENVECTORS = [f"lhe_pdf_eigenvector{i}Up" for i in range(1, 101)]

input_dir = os.path.dirname(f"{args.input_data[0]}/")

if not os.path.exists(args.output):
    os.makedirs(args.output)


# Define signal, background, and data datasets
# == Layout: {"signal"/"data"/"background": {process_name: [dataset names as in the coffea file]}}
# Data datasets listed under "background" are taken from bkg_region_name and reweighted (data-driven background);
# they are listed with their plain names, the "_background" suffix is added internally.
if not args.boosted:
    sig_bkg_dict = {
            "signal": {
                "ggHH_kl_1_kt_1_13p6TeV_hbbhbb": ["GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE"],
                "ggHH_kl_2p45_kt_1_13p6TeV_hbbhbb": ["GluGlutoHHto4B_spanet_kl-2p45_kt-1p00_c2-0p00_2022_postEE"],
                "ggHH_kl_5_kt_1_13p6TeV_hbbhbb": ["GluGlutoHHto4B_spanet_kl-5p00_kt-1p00_c2-0p00_2022_postEE"],
                },
            "data": {
                # This needs to have this name, and only be one category
                "data_obs": ["DATA_JetMET_JMENano_E_2022_postEE_EraE", "DATA_JetMET_JMENano_F_2022_postEE_EraF", "DATA_JetMET_JMENano_G_2022_postEE_EraG"]
                },
            "background": {
                "bbbb_background": ["DATA_JetMET_JMENano_E_2022_postEE_EraE", "DATA_JetMET_JMENano_F_2022_postEE_EraF", "DATA_JetMET_JMENano_G_2022_postEE_EraG"]
                }
            }
else:
    sig_bkg_dict = {
            "signal": {
                # ggF samples must match the ggf basis of hh_model_13p6TeV:model_default (kl = 1, 2.45, 5)
                "ggHH_kl_1_kt_1_13p6TeV_hbbhbb":    ["GluGluHHto4B_Par-c2-0p00-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8_2024"],
                "ggHH_kl_2p45_kt_1_13p6TeV_hbbhbb": ["GluGluHHto4B_Par-c2-0p00-kl-2p45-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8_2024"],
                "ggHH_kl_5_kt_1_13p6TeV_hbbhbb":    ["GluGluHHto4B_Par-c2-0p00-kl-5p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8_2024"],
                "qqHH_CV_1_C2V_1_kl_1_13p6TeV_hbbhbb":             ["VBFHHto4B_Par-CV-1-C2V-1-C3-1_TuneCP5_13p6TeV_madgraph-pythia8_2024"],
                "qqHH_CV_1_C2V_0_kl_1_13p6TeV_hbbhbb":              ["VBFHHto4B_Par-CV-1-C2V-0-C3-1_TuneCP5_13p6TeV_madgraph-pythia8_2024"],
                "qqHH_CV_m0p758_C2V_1p44_kl_m19p3_13p6TeV_hbbhbb":  ["VBFHHto4B_Par-CV-m0p758-C2V-1p44-C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8_2024"],
                "qqHH_CV_1p74_C2V_1p37_kl_14p4_13p6TeV_hbbhbb":     ["VBFHHto4B_Par-CV-1p74-C2V-1p37-C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8_2024"],
                "qqHH_CV_m0p962_C2V_0p959_kl_m1p43_13p6TeV_hbbhbb": ["VBFHHto4B_Par-CV-m0p962-C2V-0p959-C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8_2024"],
                "qqHH_CV_m1p60_C2V_2p72_kl_m1p36_13p6TeV_hbbhbb":   ["VBFHHto4B_Par-CV-m1p60-C2V-2p72-C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8_2024"],
            },
            "data": {
                # This needs toTTto2L2Nu_2024 have this name, and only be one category
                "data_obs": ["DATA_JetMET0_HH4bBoosted_2024_EraC", "DATA_JetMET0_HH4bBoosted_2024_EraD", "DATA_JetMET0_HH4bBoosted_2024_EraE", "DATA_JetMET0_HH4bBoosted_2024_EraF", "DATA_JetMET0_HH4bBoosted_2024_EraG", "DATA_JetMET0_HH4bBoosted_2024_EraH", "DATA_JetMET0_HH4bBoosted_2024_EraI", "DATA_JetMET1_HH4bBoosted_2024_EraC", "DATA_JetMET1_HH4bBoosted_2024_EraD", "DATA_JetMET1_HH4bBoosted_2024_EraE", "DATA_JetMET1_HH4bBoosted_2024_EraF", "DATA_JetMET1_HH4bBoosted_2024_EraG", "DATA_JetMET1_HH4bBoosted_2024_EraH", "DATA_JetMET1_HH4bBoosted_2024_EraI"]
            },
            "background": {
                "QCD_background": ["DATA_JetMET0_HH4bBoosted_2024_EraC", "DATA_JetMET0_HH4bBoosted_2024_EraD", "DATA_JetMET0_HH4bBoosted_2024_EraE", "DATA_JetMET0_HH4bBoosted_2024_EraF", "DATA_JetMET0_HH4bBoosted_2024_EraG", "DATA_JetMET0_HH4bBoosted_2024_EraH", "DATA_JetMET0_HH4bBoosted_2024_EraI", "DATA_JetMET1_HH4bBoosted_2024_EraC", "DATA_JetMET1_HH4bBoosted_2024_EraD", "DATA_JetMET1_HH4bBoosted_2024_EraE", "DATA_JetMET1_HH4bBoosted_2024_EraF", "DATA_JetMET1_HH4bBoosted_2024_EraG", "DATA_JetMET1_HH4bBoosted_2024_EraH", "DATA_JetMET1_HH4bBoosted_2024_EraI"],
                "TTto2L2Nu": ["TTto2L2Nu_2024"],
                "TTto4Q": ["TTto4Q_2024"],
                "TTtoLNu2Q": ["TTtoLNu2Q_2024"],
                }
            }

# Define the new region name
# region_info: {datacard region: {"kind": "SR" / "QCDsb" / "TTsb", "category": category}}
region_info = {}
if args.boosted:
    region_name_list = ["catvbf", "cat1", "cat2", "cat3"]
    sig_region_name = ["boosted_signal_region_D_catvbf", "boosted_signal_region_D_cat1", "boosted_signal_region_D_cat2", "boosted_signal_region_D_cat3"]
    bkg_region_name = ["boosted_control_region_B_catvbf_postW", "boosted_control_region_B_cat1_postW", "boosted_control_region_B_cat2_postW", "boosted_control_region_B_cat3_postW"]
    region_definitions = {}
    # The global normalisation of the QCD template is always taken from C / A_postW (inclusive)
    norm_regions = {
        "sig_control_region_name": "boosted_signal_sideband_region_C",
        "bkg_control_region_name": "boosted_control_sideband_region_A_postW",
    }
    for name, sig, bkg in zip(region_name_list, sig_region_name, bkg_region_name):
        region_definitions[f"HH4b_merged_{name}"] = {
                "fit_variable": "SubLeadingHiggsMass",
                "sig_region_name": sig,
                "bkg_region_name": bkg,
                **norm_regions,
                }
        region_info[f"HH4b_merged_{name}"] = {"kind": "SR", "category": name}
        if args.fit_sidebands:
            # One-bin sideband cards (rebinned to a single bin below)
            region_definitions[f"HH4b_qcdsb_{name}"] = {
                    "fit_variable": "SubLeadingHiggsMass",
                    "sig_region_name": f"boosted_signal_sideband_region_C_{name}",
                    "bkg_region_name": f"boosted_control_sideband_region_A_{name}_postW",
                    **norm_regions,
                    }
            region_info[f"HH4b_qcdsb_{name}"] = {"kind": "QCDsb", "category": name}
            region_definitions[f"HH4b_ttsb_{name}"] = {
                    "fit_variable": "SubLeadingHiggsMass",
                    "sig_region_name": f"boosted_signal_ttbar_region_E_{name}",
                    "bkg_region_name": f"boosted_control_ttbar_region_F_{name}_postW",
                    **norm_regions,
                    }
            region_info[f"HH4b_ttsb_{name}"] = {"kind": "TTsb", "category": name}

else:
    region_definitions = {
            "ggHH": {
                "fit_variable": "sig_bkg_dnn_score",
                "sig_region_name": "4b_signal_region",
                "bkg_region_name": "2b_signal_region_postW",
                "sig_control_region_name": "4b_control_region",
                "bkg_control_region_name": "2b_control_region_postW",
                }
            }
    region_info = {"ggHH": {"kind": "SR", "category": "ggHH"}}


def get_year(pocketcoffea_year, year_type=0):
    """Get year string for datacard from PocketCoffea version.

    :param year_type: There are three different ways, how year is used.
                      - separated by pre/post eras (year_type=0)
                      - actual year (year_type=1)
                      - separated by parking/no-parking (year_type=2)
    """
    # Different categories for how to separated years/datataking periods
    dict_year_type = [
            {"2022_preEE": "2022", "2022_postEE": "2022EE", "2023_preBPix": "2023", "2023_postBPix": "2023BPIX", "2024": "2024"},
            {"2022_preEE": "2022", "2022_postEE": "2022", "2023_preBPix": "2023", "2023_postBPix": "2023", "2024": "2024"},
            {"2022_preEE": "2022", "2022_postEE": "2022", "2023_preBPix": "2023", "2023_postBPix": "2023ParkingHH", "2024": "2024"}  # <-- Probably wrong. But used for background datadriven
            ]
    try:
        return dict_year_type[year_type][pocketcoffea_year]
    except:
        raise ValueError(f"Wrong values for either input year: {pocketcoffea_year} or year_type: {year_type}")


# Jet types (prefix of the PocketCoffea jet variations, e.g. "AK8PFPuppi_JES_Total") reported as
# the AK4 / AK8 jet uncertainties. Variations of any other jet type (e.g. the merged "AK4Jet"
# collections) are not used in the datacards, so that two different variations never get the
# same datacard name.
JET_TYPE_TAGS = {"AK4PFPuppi": "AK4", "AK4PFchs": "AK4", "AK8PFPuppi": "AK8"}


def is_unused_jet_variation(pocketcoffea_name):
    """True for JES/JER variations of a jet type not listed in JET_TYPE_TAGS"""
    jet_type, _, jet_variation = pocketcoffea_name.partition("_")
    return jet_type.startswith("AK") and jet_variation in ("JES_Total", "JER") and jet_type not in JET_TYPE_TAGS


def get_uncertainty_name(pocketcoffea_name, year):
    """Get conventional datacard name for systematic uncertainties"""
    year_tag = get_year(year, year_type=0)
    # Jet variations: "<jet type>_JES_Total" / "<jet type>_JER"
    jet_type, _, jet_variation = pocketcoffea_name.partition("_")
    if jet_type.startswith("AK") and jet_variation in ("JES_Total", "JER"):
        jet_tag = JET_TYPE_TAGS[jet_type]
        # In the boosted analysis AK4 and AK8 jets are varied separately (not seen in the
        # datacards of HIG-24-010); the resolved analysis keeps the plain names for AK4 jets
        jet_prefix = "" if (not args.boosted and jet_tag == "AK4") else f"{jet_tag}_"
        if jet_variation == "JES_Total":
            return f"CMS_scale_j_{jet_prefix}{year_tag}"
        return f"CMS_res_j_{jet_prefix}{year_tag}"

    energy = "13p6TeV" if str(year).startswith("202") else "13TeV"
    unc_map = {
        "pileup": f"CMS_pileup_{year_tag}",
        "sf_partonshower_isr": "ps_isr",
        "sf_partonshower_fsr": "ps_fsr",
        "lumi": f"lumi_{energy}_{get_year(year, year_type=1)}",
        }
    if pocketcoffea_name in unc_map.keys():
        return unc_map[pocketcoffea_name]
    # Variations connected to b-tag SF
    elif "btag" in pocketcoffea_name:
        variation = pocketcoffea_name.split("_")[-1]
        if "heavy" in pocketcoffea_name:
            return f"CMS_btag_fixedWP_bc_{variation}_{year}"
        elif "light" in pocketcoffea_name:
            return f"CMS_btag_fixedWP_light_{variation}_{year}"
        else:
            raise ValueError(f"Unclear flavour type for b-tag correction {pocketcoffea_name}. Should contain 'light' or 'heavy' in name")
    else:
        raise ValueError(f"Did not found name mapping for {pocketcoffea_name}")

def add_variation_axis(histogram):
    """Return a histogram with an extra variation axis ('nominal'), preserving values+variances."""
    # build new axes
    axes = list(histogram.axes)
    var_axis = hist.axis.StrCategory(["nominal"], name="variation", label="Variation")

    cat_idx = next((i for i, ax in enumerate(axes) if getattr(ax, "name", None) == "cat"), None)
    if cat_idx is None:
        raise ValueError(f"Histogram has no 'cat' axis: {histogram.axes}")
    var_idx = cat_idx + 1
    new_axes = axes[:var_idx] + [var_axis] + axes[var_idx:]

    # create new histogram
    try:
        new_hist = hist.Hist(*new_axes, storage=hist.storage.Weight())
    except:
        breakpoint()

    # --- copy content (the variation axis only has 'nominal' at index 0) ---
    sl_new = [slice(None)] * new_hist.ndim
    sl_new[var_idx] = 0
    new_hist.view(flow=True)[tuple(sl_new)] = histogram.view(flow=True)

    return new_hist


def fold_flow(histogram, axis=-1):
    """Return a copy of the histogram with under-/overflow of `axis` added to the first/last bin.

    The Datacard only reads the in-range bins, so without folding events outside the axis range
    (e.g. a score of exactly 1.0 for Regular(0, 1)) would silently be dropped.
    """
    new_hist = histogram.copy()
    traits = new_hist.axes[axis].traits
    view = new_hist.view(flow=True)
    view_ax = np.moveaxis(view, axis, -1)  # still a view into new_hist
    for field in ("value", "variance"):
        if traits.underflow:
            view_ax[field][..., 1] += view_ax[field][..., 0]
            view_ax[field][..., 0] = 0
        if traits.overflow:
            view_ax[field][..., -2] += view_ax[field][..., -1]
            view_ax[field][..., -1] = 0
    return new_hist


def cut_range(histogram, low, high):
    """Return the histogram restricted to [low, high] on the variable axis (last axis).

    Slicing in hist moves the removed content into the flow bins, so these are zeroed afterwards
    to really drop the events outside the range. If `low`/`high` are not on a bin edge, the bin
    containing them is kept.
    """
    axis = histogram.axes[-1]
    new_hist = histogram[{axis.name: slice(hist.loc(low), hist.loc(high))}]
    edges = new_hist.axes[-1].edges
    if not (np.isclose(edges[0], low) and np.isclose(edges[-1], high)):
        logger.warning(f"Fit range [{low}, {high}] does not match the bin edges of {axis.name}, using [{edges[0]}, {edges[-1]}]")
    traits = new_hist.axes[-1].traits
    view_ax = np.moveaxis(new_hist.view(flow=True), -1, 0)
    for field in ("value", "variance"):
        if traits.underflow:
            view_ax[field][0] = 0
        if traits.overflow:
            view_ax[field][-1] = 0
    return new_hist


def compute_min_stat_edges(histograms, category, bkg_datasets, meta_dict, max_rel_unc=None, min_yield=None):
    """Bin edges for which every bin of the total background fulfills the minimum statistics requirement.

    Adjacent bins of the variable axis (last axis) are merged from left to right until the summed nominal
    background has a yield > 0, >= `min_yield` and a relative stat. uncertainty <= `max_rel_unc`.
    A remaining tail not fulfilling the requirement is merged into the last bin.

    :param histograms: histograms of a single variable, layout {sample: {dataset: hist}}
    :param bkg_datasets: background datasets (as in the histograms) entering the total background
    """
    values, variances, edges = 0.0, 0.0, None
    for dataset in bkg_datasets:
        sample = meta_dict[dataset]["sample"]
        if dataset not in histograms.get(sample, {}):
            continue
        histogram = histograms[sample][dataset]
        selection = {"cat": category}
        if "variation" in histogram.axes.name:
            selection["variation"] = "nominal"
        # Flow bins were already folded in create_new_region
        view = histogram[selection].view(flow=False)
        values = values + view["value"]
        variances = variances + view["variance"]
        edges = histogram.axes[-1].edges
    if edges is None:
        raise ValueError(f"No background histograms found for category {category}")

    def passes(val, var):
        if val <= 0:
            return False
        if min_yield is not None and val < min_yield:
            return False
        if max_rel_unc is not None and np.sqrt(var) / val > max_rel_unc:
            return False
        return True

    new_edges = [edges[0]]
    acc_val, acc_var = 0.0, 0.0
    for i in range(len(values)):
        acc_val += values[i]
        acc_var += variances[i]
        if passes(acc_val, acc_var):
            new_edges.append(edges[i + 1])
            acc_val, acc_var = 0.0, 0.0
    if not np.isclose(new_edges[-1], edges[-1]):
        if len(new_edges) > 1:
            new_edges[-1] = edges[-1]
        else:
            new_edges.append(edges[-1])
    new_edges = np.array(new_edges)

    # Report the resulting background statistics per bin
    idx = np.searchsorted(edges, new_edges[:-1] + 0.5 * np.min(np.diff(edges))) - 1
    bin_val = np.add.reduceat(np.asarray(values), idx)
    bin_unc = np.sqrt(np.add.reduceat(np.asarray(variances), idx))
    logger.info(f"Min-stat rebinning for {category}: {len(edges) - 1} -> {len(new_edges) - 1} bins, edges {new_edges.tolist()}")
    logger.info(f"  bkg yield per bin: {np.round(bin_val, 2).tolist()}")
    logger.info(f"  bkg rel. unc. per bin: {np.round(bin_unc / np.where(bin_val > 0, bin_val, np.nan), 3).tolist()}")
    return new_edges


def process_yield(histograms, datasets, category, variation, meta_dict):
    """Total yield of a process (summed over its datasets) in `category` for a single variation"""
    total = 0.0
    for dataset in datasets:
        histogram = histograms[meta_dict[dataset]["sample"]][dataset]
        # Flow bins were already folded in create_new_region and are not read by the Datacard
        total += histogram[{"cat": category, "variation": variation}].values(flow=False).sum()
    return total


def compute_norm_factor(histograms, num_label, denom_label, mc_bkg_datasets=()):
    """Global normalisation factor (sum(data, num_label) - sum(MC bkg, num_label)) / sum(data, denom_label).

    The reweighted data in denom_label only estimates the data-driven (QCD) part, so the MC backgrounds
    (e.g. ttbar) are subtracted in num_label: data-driven estimate + MC backgrounds = data in num_label.

    :param histograms: histograms of a single variable, layout {sample: {dataset: hist}}
    :param mc_bkg_datasets: MC background datasets subtracted from the data in num_label

    Returns the factor and its relative statistical uncertainty (from the data in both regions
    and the MC background statistics, added in quadrature).
    """
    sum_num = var_num = 0.0
    sum_mc = var_mc = 0.0
    sum_denom = var_denom = 0.0
    for datasets in histograms.values():
        for dataset, histogram in datasets.items():
            if dataset in mc_bkg_datasets:
                selection = {"cat": num_label}
                if "variation" in histogram.axes.name:
                    selection["variation"] = "nominal"
                total = histogram[selection].sum(flow=True)
                sum_mc += total.value
                var_mc += total.variance
                continue
            if "DATA" not in dataset or dataset.endswith("_background"):
                continue
            total_num = histogram[{"cat": num_label}].sum(flow=True)
            total_denom = histogram[{"cat": denom_label}].sum(flow=True)
            sum_num += total_num.value
            var_num += total_num.variance
            sum_denom += total_denom.value
            var_denom += total_denom.variance
    if sum_denom == 0:
        raise ValueError(f"No data events in normalisation region {denom_label!r}")
    if sum_num - sum_mc <= 0:
        raise ValueError(f"Non-positive data - MC bkg in normalisation region {num_label!r}: {sum_num} - {sum_mc}")
    factor = (sum_num - sum_mc) / sum_denom
    rel_unc = np.sqrt((var_num + var_mc) / (sum_num - sum_mc)**2 + var_denom / sum_denom**2)
    logger.info(f"Normalisation factor ({num_label} - MC bkg) / {denom_label} = ({sum_num} - {sum_mc}) / {sum_denom} = {factor:.5f} "
                f"+- {rel_unc:.2%} (stat.)")
    return factor, rel_unc


def duplicate_category(histo, src_label, new_label, axis=0, scale=None):
    """Copy the histogram for a category from one region into a new one.

    Essentially all the required datasets/regions for the analysis are to be copied into the newly created region.
    If `scale` is given, the copied category is scaled by it and a 'nominal' variation axis is added
    (needed, because the scaled data is treated as an MC process in the datacard).
    """
    cat_axis = histo.axes[axis]
    if not isinstance(cat_axis, hist.axis.StrCategory):
        raise TypeError(f"Axis {axis} is not a StrCategory axis")

    if src_label not in cat_axis:
        raise ValueError(f"Source label {src_label!r} not found in category axis")

    # build a new histoogram with the extended categories
    new_cats = list(cat_axis) + [new_label]
    new_axes = list(histo.axes)
    new_axes[axis] = hist.axis.StrCategory(new_cats, name=cat_axis.name, label=cat_axis.label)

    new_histo = hist.Hist(*new_axes, storage=hist.storage.Weight())

    # ---- copy over each old category one by one ----
    for cat in cat_axis:
        idx_old = histo.axes[axis].index(cat)
        idx_new = new_histo.axes[axis].index(cat)

        sl_old = [slice(None)] * histo.ndim
        sl_new = [slice(None)] * new_histo.ndim
        sl_old[axis] = idx_old
        sl_new[axis] = idx_new

        vals = histo.view(flow=True)[tuple(sl_old)]
        new_histo.view(flow=True)[tuple(sl_new)] = vals

    # ---- duplicate requested category ----
    idx_src = histo.axes[axis].index(src_label)
    idx_dst = new_histo.axes[axis].index(new_label)

    sl_src = [slice(None)] * histo.ndim
    sl_dst = [slice(None)] * histo.ndim
    sl_src[axis] = idx_src
    sl_dst[axis] = idx_dst

    # Copy, so that scaling does not modify the input histogram in place
    vals = histo.view(flow=True)[tuple(sl_src)].copy()
    if scale is None:
        new_histo.view(flow=True)[tuple(sl_dst)] = vals
    else:
        vals["value"] *= scale
        vals["variance"] *= scale**2

        new_histo.view(flow=True)[tuple(sl_dst)] = vals
        if "variation" not in new_histo.axes.name:
            new_histo = add_variation_axis(new_histo)

    return fold_flow(new_histo)


def add_theory_acceptance_variations(histograms, processes_by_production, meta_dict):
    """Add the QCD scale envelope and the PDF quadrature sum as shape variations
    `CMS_bbbb_hadronic_QCDScaleacc_{production}` / `CMS_bbbb_hadronic_PDFacc_{production}`.

    Must be called on the histograms in their final binning (envelope and quadrature sum are
    not additive over bins). Productions whose histograms lack the lhe_scale / lhe_pdf variations
    (i.e. processed without these weights) are skipped with a warning.
    Returns the new histograms and a dict {variation name: processes} for the shape systematics.
    """
    theory_shapes = {}
    for production, processes in processes_by_production.items():
        datasets = [d for p in processes for d in sig_bkg_dict["signal"][p]]
        if not datasets:
            continue
        samples = sorted({meta_dict[d]["sample"] for d in datasets})
        for variation_name, source_variations, combine in (
            (f"QCDscale_{production}_ACCEPT", LHE_SCALE_VARIATIONS, add_envelope_variation),
            (f"pdf_Higgs_{production}_ACCEPT", LHE_PDF_EIGENVECTORS, add_quadrature_variation),
        ):
            missing = [d for d in datasets
                       if not set(source_variations) <= set(histograms[meta_dict[d]["sample"]][d].axes["variation"])]
            if missing:
                logger.warning(f"No LHE variations for {missing}: {variation_name} not added")
                continue
            histograms = combine(histograms, variation_name, samples, source_variations, datasets=datasets)
            theory_shapes[variation_name] = processes
    return histograms, theory_shapes


def create_new_region(coffea_file, cat_name, sig_region_name, bkg_region_name, fit_variable, sig_control_region_name, bkg_control_region_name):
    """Create a new region in the coffea file, that contains all the regions we need for the analysis."""

    for key in ["sumw", "sumw2"]:
        # Only works, if all MC are in the signal region (will hopefully stay this way...)
        coffea_file[key][cat_name] = copy.deepcopy(coffea_file[key][sig_region_name])
    # == adding cutflow ==
    presel_samples = [k for k in coffea_file["cutflow"]["presel"].keys() if not k.endswith("_background")]
    for key in presel_samples:
        coffea_file["cutflow"]["presel"][f"{key}_background"] = copy.deepcopy(coffea_file["cutflow"]["presel"][key])
    # deepcopy, as otherwise the "_background" entries below would also be added to the signal region cutflow
    coffea_file["cutflow"][cat_name] = copy.deepcopy(coffea_file["cutflow"][sig_region_name])
    for key in coffea_file["cutflow"][bkg_region_name].keys():
        if "DATA" in key:
            coffea_file["cutflow"][cat_name][f"{key}_background"] = {}
            for subkey, subvalue in coffea_file["cutflow"][bkg_region_name][key].items():
                coffea_file["cutflow"][cat_name][f"{key}_background"][f"{subkey}_background"] = subvalue
    # == variables ==
    # One global factor (from the fit variable, flow included) for all eras and variables
    mc_bkg_datasets = [d for datasets in sig_bkg_dict["background"].values() for d in datasets if "DATA" not in d]
    if args.separate_normalisation:
        norm_factor, norm_rel_unc = compute_norm_factor(coffea_file["variables"][fit_variable], sig_region_name, bkg_region_name, mc_bkg_datasets)
    else:
        norm_factor, norm_rel_unc = compute_norm_factor(coffea_file["variables"][fit_variable], sig_control_region_name, bkg_control_region_name, mc_bkg_datasets)
    for column in coffea_file["variables"].keys():
        sample_list = [s for s in coffea_file["variables"][column] if not s.endswith("_background")]
        for sample in sample_list:
            for dataset in list(coffea_file["variables"][column][sample]):
                histogram = coffea_file["variables"][column][sample][dataset]
                # Copy the histgam of Data and MC from the signal region.
                coffea_file["variables"][column][sample][dataset] = duplicate_category(histogram, sig_region_name, cat_name, axis=0)  # If Data, also copy from the background region
                if "DATA" in dataset:
                    bkg_sample, bkg_dataset = f"{sample}_background", f"{dataset}_background"
                    # Extend the background histogram from earlier passes if it exists, else start from the data histogram
                    base = coffea_file["variables"][column].get(bkg_sample, {}).get(bkg_dataset, histogram)
                    coffea_file["variables"][column].setdefault(bkg_sample, {})[bkg_dataset] = duplicate_category(
                        base, bkg_region_name, cat_name, axis=0, scale=norm_factor)
    # == datasets_metadata : by_datataking_period ==
    for period, entries in coffea_file["datasets_metadata"]["by_datataking_period"].items():
        for sample, dataset in list(entries.items()):
            if "DATA" in sample and not sample.endswith("_background"):
                entries[f"{sample}_background"] = {f"{d}_background" for d in dataset}
    # == datasets_metadata : by_dataset ==
    orig_dset_list = list(coffea_file["datasets_metadata"]["by_dataset"].keys())
    for dataset in orig_dset_list:
        if "DATA" in dataset and not dataset.endswith("_background"):
            temp_dset = coffea_file["datasets_metadata"]["by_dataset"][dataset].copy()
            temp_dset["sample"] = f"{temp_dset['sample']}_background"
            temp_dset["primaryDataset"] = f"{temp_dset['primaryDataset']}_background"
            coffea_file["datasets_metadata"]["by_dataset"][f"{dataset}_background"] = temp_dset
    return coffea_file, norm_rel_unc



def add_rate_params(card_path, rate_params, low=0, high=5):
    """Append rateParam lines to a datacard written by the PocketCoffea Datacard.

    The PocketCoffea `has_rateParam` option writes one global rateParam per process (bin "*"), while here the
    rateParams are per category and shared only between the signal region and the sidebands of that category.
    The bin and process names are read back from the card, so that the year suffix of the process names does
    not matter. Parameters with the same name in different cards are merged by combineCards.py.

    :param rate_params: list of (parameter name, list of process names as in sig_bkg_dict)
    """
    with open(card_path) as f:
        lines = f.read().splitlines()
    tokens = [line.split() for line in lines]
    bin_name = next(t[1] for t in tokens if len(t) > 1 and t[0] == "bin")
    process_rows = [t[1:] for t in tokens if t and t[0] == "process"]
    card_processes = next(row for row in process_rows if not all(re.fullmatch(r"-?\d+", x) for x in row))
    new_lines = []
    for param, processes in rate_params:
        matched = sorted({p for p in card_processes if any(p == q or p.startswith(f"{q}_") for q in processes)})
        if not matched:
            logger.warning(f"rateParam {param}: none of {processes} found in {card_path}")
        for process in matched:
            new_lines.append(f"{param} rateParam {bin_name} {process} 1 [{low},{high}]")
    with open(card_path, "a") as f:
        f.write("\n" + "\n".join(new_lines) + "\n")
    logger.info(f"Added rateParams to {card_path}:\n  " + "\n  ".join(new_lines))


if __name__ == "__main__":
    # I want to use the coffea file with all outputs. Therefore, it should be merged beforehand.
    if "output_all.coffea" in input_dir:
        coffea_file = input_dir
    else:
        coffea_list = [file for file in os.listdir(input_dir) if file.endswith(".coffea")]
        if "output_all.coffea" in coffea_list:
            coffea_file = os.path.join(input_dir, "output_all.coffea")
        else:
            raise NameError(f"No combined coffea file found in {coffea_list}")

    # -- Load Coffea file and config.json --
    coffea_file = load(coffea_file)

    # Data-driven backgrounds are listed with the plain data dataset names. Their reweighted copies
    # (created in create_new_region) carry the "_background" suffix, so point to those instead.
    for name, datasets in sig_bkg_dict["background"].items():
        sig_bkg_dict["background"][name] = [f"{d}_background" if "DATA" in d else d for d in datasets]

    missing_regions = sorted({
        value[key] for value in region_definitions.values() for key in ("sig_region_name", "bkg_region_name")
        if value[key] not in coffea_file["sumw"] and value[key] not in coffea_file["cutflow"]
    })
    if missing_regions:
        raise KeyError(f"Regions {missing_regions} not found in the coffea file"
                       + (" (was it produced with boosted_fit_sidebands=True?)" if args.fit_sidebands else ""))

    card_paths = defaultdict(list)
    for region_name, value in region_definitions.items():
        kind = region_info[region_name]["kind"]
        category = region_info[region_name]["category"]
        coffea_file, norm_rel_unc = create_new_region(coffea_file,
                                        cat_name=region_name,
                                        **value
                                        )

        # -- Histograms --
        histograms_dict = {}
        for key, sob_hist in coffea_file["variables"].items():
            if value["fit_variable"] in key:
                if args.fit_range is not None:
                    sob_hist = {
                        sample: {dataset: cut_range(h, *args.fit_range) for dataset, h in datasets.items()}
                        for sample, datasets in sob_hist.items()
                    }
                if kind != "SR":
                    # Sidebands: only the yield enters the fit -> one bin (flow already folded in create_new_region)
                    histograms_dict[key] = {
                        sample: {dataset: h[{h.axes[-1].name: hist.rebin(h.axes[-1].size)}] for dataset, h in datasets.items()}
                        for sample, datasets in sob_hist.items()
                    }
                elif args.nbins is None:
                    histograms_dict[key] = sob_hist
                else:
                    # Rebin the variable axis (last axis) of every histogram into args.nbins bins.
                    # Flow bins were already folded in create_new_region, so no events are lost.
                    def rebin_factor(h):
                        size = h.axes[-1].size
                        if size % args.nbins != 0:
                            raise ValueError(f"Cannot rebin {size} bins of {key} into {args.nbins} bins (not a divisor)")
                        return size // args.nbins
                    histograms_dict[key] = {
                        sample: {
                            dataset: h[{h.axes[-1].name: hist.rebin(rebin_factor(h))}]
                            for dataset, h in datasets.items()
                        }
                        for sample, datasets in sob_hist.items()
                    }
            # TODO: Make sure, that the correct signal histograms are chosen
            # if "sig_bkg" in key:
            #     histograms_dict[key] = sob_hist
            #     # "SoB": coffea_file["variables"]["sig_bkg_dnn_score"],
        # -- Create Processes
        meta_dict = coffea_file['datasets_metadata']['by_dataset']

        for datasets in sig_bkg_dict["signal"].values():
            for dataset in datasets:
                if dataset not in meta_dict.keys():
                    raise Exception(f"Signal dataset {dataset} not found in file")
        for datasets in sig_bkg_dict["data"].values():
            for dataset in datasets:
                if dataset not in meta_dict.keys():
                    raise Exception(f"Data dataset {dataset} not found in file")
        for datasets in sig_bkg_dict["background"].values():
            for dataset in datasets:
                if dataset not in meta_dict.keys():
                    raise Exception(f"Background dataset {dataset} not found in file")

        logger.info(f"These are the MC samples: {sig_bkg_dict['signal']}")
        logger.info(f"These are the Data samples: {sig_bkg_dict['data']}")
        logger.info(f"These are the Data background samples: {sig_bkg_dict['background']}")

        # -- Filling metadata into the respective objects --
        mc_process = []
        for name, datasets in sig_bkg_dict["signal"].items():
            mc_process.append(MCProcess(
                    name=name,
                    samples={meta_dict[dataset]["sample"] for dataset in datasets},
                    years={meta_dict[dataset]["year"] for dataset in datasets},
                    is_signal=True,
                    ))
        bg_process = []
        for name, datasets in sig_bkg_dict["background"].items():
            bg_process.append(MCProcess(
                    name=name,
                    samples={meta_dict[dataset]["sample"] for dataset in datasets},
                    years={meta_dict[dataset]["year"] for dataset in datasets},
                    is_signal=False,
                    ))
        mc_processes = MCProcesses(mc_process + bg_process)

        if len(sig_bkg_dict["data"].keys()) > 1 or "data_obs" not in sig_bkg_dict["data"].keys():
            raise Exception("Only one single data process is allowed with fixed name 'data_obs'")
        for name, datasets in sig_bkg_dict["data"].items():
            data_process = DataProcess(
                    name=name,
                    samples={meta_dict[dataset]["sample"] for dataset in datasets},
                    years={meta_dict[dataset]["year"] for dataset in datasets},
                    )
        data_processes = DataProcesses([data_process])

        # -- Systematics --
        # common_systematics = [
        #     "JES_Total_AK4PFPuppi", "JER_AK4PFPuppi"
        # ]

        signal_processes = list(sig_bkg_dict["signal"])
        signal_processes_by_production = {
            "ggHH": [p for p in signal_processes if p.startswith("ggHH")],
            "qqHH": [p for p in signal_processes if p.startswith("qqHH")],
        }
        for hist_cat, sob_hist in histograms_dict.items():
            # The final binning is fixed before the theory variations are built, as the
            # envelope and the quadrature sum are not additive over bins.
            bins_edges = None
            if kind == "SR" and (args.max_bkg_unc is not None or args.min_bkg_yield is not None):
                bkg_datasets = [d for datasets in sig_bkg_dict["background"].values() for d in datasets]
                bins_edges = compute_min_stat_edges(sob_hist, region_name, bkg_datasets, meta_dict,
                                                    max_rel_unc=args.max_bkg_unc, min_yield=args.min_bkg_yield)
                os.makedirs(f"{args.output}/{hist_cat}", exist_ok=True)
                with open(f"{args.output}/{hist_cat}/bin_edges_{region_name}.json", "w") as f:
                    json.dump({"edges": bins_edges.tolist(), "max_bkg_unc": args.max_bkg_unc,
                               "min_bkg_yield": args.min_bkg_yield}, f, indent=2)
                sob_hist = rebin_hist(bins_edges, sob_hist)
            sob_hist, theory_shapes = add_theory_acceptance_variations(sob_hist, signal_processes_by_production, meta_dict)

            systematics_list = []
            card_year = meta_dict[sig_bkg_dict["data"]["data_obs"][0]]["year"]
            # MC backgrounds (ttbar); the data-driven background has no variations (at the moment)
            bg_mc = {k: v for k, v in sig_bkg_dict["background"].items() if "DATA" not in v[0]}

            # Shape systematics from the histogram variations (JES, JER, pileup, PS, ...):
            # one nuisance per variation, on every MC process (signal and MC background)
            # whose histograms carry it, so that e.g. JES is correlated between signal and ttbar.
            processes_by_variation = defaultdict(set)
            for process, datasets in {**sig_bkg_dict["signal"], **bg_mc}.items():
                for dataset in datasets:
                    variations_updown = sob_hist[meta_dict[dataset]["sample"]][dataset].axes["variation"]
                    if "nominal" not in variations_updown:
                        raise ValueError(f"Variations of {dataset} do not contain 'nominal': {list(variations_updown)}")
                    for variation in {re.sub(r"(Up|Down)$", "", v) for v in variations_updown}:
                        # The raw LHE variations only enter through the combined theory shapes
                        if variation == "nominal" or variation.startswith("lhe_") or variation in theory_shapes:
                            continue
                        if is_unused_jet_variation(variation):
                            continue
                        processes_by_variation[variation].add(process)
            for variation, processes in sorted(processes_by_variation.items()):
                # combine cannot handle a non-positive Up/Down yield for a process with a positive nominal
                # yield ("Bogus norm", kappa <= 0). This happens for low-stat MC, where all events migrate
                # out of the category under the variation: drop the systematic for that process here.
                all_process_datasets = {**sig_bkg_dict["signal"], **bg_mc}
                for process in sorted(processes):
                    datasets = all_process_datasets[process]
                    nominal = process_yield(sob_hist, datasets, region_name, "nominal", meta_dict)
                    if nominal <= 0:
                        continue
                    shifted = {shift: process_yield(sob_hist, datasets, region_name, f"{variation}{shift}", meta_dict)
                               for shift in ("Up", "Down")}
                    if min(shifted.values()) <= 0:
                        logger.warning(f"Dropping shape variation {variation} for process {process} in {region_name}: "
                                       f"non-positive yield (nominal {nominal:.4g}, Up {shifted['Up']:.4g}, Down {shifted['Down']:.4g})")
                        processes.discard(process)
                if not processes:
                    continue
                logger.info(f"Shape variation {variation} on processes {sorted(processes)}")
                systematics_list.append(SystematicUncertainty(name=variation, datacard_name=get_uncertainty_name(variation, card_year), typ="shape", processes=sorted(processes), years=[card_year], value=1.0))

            # ====== Other systematics / basically theory =======
            # With rateParams the ttbar normalisation is free: lumi only on the signal
            lumi_processes = ([] if args.rate_params else list(bg_mc.keys())) + list(sig_bkg_dict["signal"].keys())
            systematics_list.append(SystematicUncertainty(name="lumi", datacard_name=get_uncertainty_name("lumi", card_year), typ="lnN", processes=lumi_processes, value=lumi_sys_values[card_year], years=[card_year]))
            # Inclusive cross-section uncertainties, each only on the processes it belongs to
            ggHH_processes = signal_processes_by_production["ggHH"]
            qqHH_processes = signal_processes_by_production["qqHH"]
            ttbar_processes = list(bg_mc.keys())
            # The inclusive ttbar cross-section uncertainties are absorbed by the free rate_ttbar_<cat>
            ttbar_xs_processes = [] if args.rate_params else ttbar_processes
            lnN_uncertainties = [
                ("BR_hbb", signal_processes, BR_HBB),
                ("pdf_alphas_ggHH", ggHH_processes, ALPHAS_GGHH),
                ("pdf_alphas_qqHH", qqHH_processes, ALPHAS_QQHH),
                ("QCDscale_ggHH", ggHH_processes, SCALE_GGHH),
                ("mtop_ggHH", ggHH_processes, MTOP_GGHH),
                ("pdf_Higgs_qqHH", qqHH_processes, PDF_HIGGS_QQHH),
                ("pdf_Higgs_ggHH", ggHH_processes, PDF_HIGGS_GGHH),
                ("QCDscale_qqHH", qqHH_processes, SCALE_QQHH),
                ("QCDscale_ttbar", ttbar_xs_processes, SCALE_TTBAR),
                ("pdf_gg", ttbar_xs_processes, PDF_GG),
            ]
            for name, processes, value in lnN_uncertainties:
                if processes:
                    systematics_list.append(SystematicUncertainty(name=name, datacard_name=name, typ="lnN", processes=processes, value=value, years=[card_year]))

            # ====== Data-driven background normalisation =======
            # Statistical uncertainty of the normalisation factor. The factor of the control regions is shared
            # by all categories (correlated), with --separate-normalisation each category has its own factor.
            # The per-bin statistics of the template are covered by autoMCStats.
            data_driven_processes = [k for k, v in sig_bkg_dict["background"].items() if "DATA" in v[0]]
            analysis_tag = "boosted" if args.boosted else "resolved"
            qcd_has_rate_param = args.rate_params and not (kind == "TTsb" and args.qcd_ttsb == "lnN")
            if not qcd_has_rate_param:
                # Without a free rateParam the normalisation factor from C / A_postW and its stat. uncertainty are used
                norm_stat_name = f"CMS_bbbb_{analysis_tag}_QCD_norm_stat"
                if args.separate_normalisation:
                    norm_stat_name += f"_{region_name}"
                systematics_list.append(SystematicUncertainty(name=norm_stat_name, datacard_name=norm_stat_name, typ="lnN", processes=data_driven_processes, value=round(1 + norm_rel_unc, 4), years=[card_year]))
            if kind == "TTsb" and args.qcd_ttsb == "lnN":
                # QCD under the ttbar peak: ABCD prediction with its own normalisation uncertainty
                ttsb_name = f"CMS_bbbb_{analysis_tag}_QCD_ttsb_norm_{category}"
                systematics_list.append(SystematicUncertainty(name=ttsb_name, datacard_name=ttsb_name, typ="lnN", processes=data_driven_processes, value=1 + args.qcd_ttsb_unc, years=[card_year]))
            if args.qcd_norm_unc is not None and not qcd_has_rate_param:
                norm_name = f"CMS_bbbb_{analysis_tag}_QCD_norm_{region_name}"
                systematics_list.append(SystematicUncertainty(name=norm_name, datacard_name=norm_name, typ="lnN", processes=data_driven_processes, value=1 + args.qcd_norm_unc, years=[card_year]))

            # Theory acceptance uncertainties (shape), built in add_theory_acceptance_variations
            for name, processes in theory_shapes.items():
                systematics_list.append(SystematicUncertainty(name=name, datacard_name=name, typ="shape", processes=processes, years=[card_year], value=1.0))




            systematics = Systematics(systematics_list)

            _label = "run3"
            _datacard_name = f"datacard_combined_{_label}"
            _workspace_name = f"workspace_{_label}.root"

            auto_mc_stats = {
                "threshold": 10,
                "include_signal": 0,
                "hist_mode": 1,
            }

            datacard = Datacard(
                    histograms=sob_hist,
                    bins_edges=bins_edges,
                    datasets_metadata=coffea_file["datasets_metadata"],
                    cutflow=coffea_file["cutflow"],
                    systematics=systematics,
                    # This might have to change. Right now I am binding the year to the data year...
                    years={meta_dict[dataset]["year"] for dataset in sig_bkg_dict["data"]["data_obs"]},
                    mc_processes=mc_processes,
                    mcstat=auto_mc_stats,
                    data_processes=data_processes,
                    category=region_name,
                    single_year=True,
                    )
            card_name = f"{region_name}_{_label}.txt"
            datacard.dump(directory=f"{args.output}/{hist_cat}", card_name=card_name, shapes_name=f"shapes_{region_name}_{_label}.root")
            card_paths[hist_cat].append((region_name, card_name))

            if args.rate_params:
                rate_params = [(f"rate_ttbar_{category}", list(bg_mc.keys()))]
                if qcd_has_rate_param:
                    rate_params.append((f"rate_QCD_{category}", data_driven_processes))
                add_rate_params(f"{args.output}/{hist_cat}/{card_name}", rate_params)

    # Command to combine the cards of all regions (signal region + sidebands) into one card per fit variable
    for hist_cat, cards in card_paths.items():
        script = f"{args.output}/{hist_cat}/combine_cards.sh"
        with open(script, "w") as f:
            f.write("#!/bin/bash\n# Combine the datacards of all regions (rateParams with the same name are merged)\n")
            f.write('cd "$(dirname "$0")"\n')
            f.write("combineCards.py " + " ".join(f"{region}={card}" for region, card in cards) + " > datacard_combined.txt\n")
        logger.info(f"Written {script}")
