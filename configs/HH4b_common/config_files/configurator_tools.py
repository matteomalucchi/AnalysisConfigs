from collections import defaultdict
import copy

from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters.cuts import passthrough
from utils_configs.quantile_transformer import WeightedQuantileTransformer

import configs.HH4b_common.custom_cuts_common as cuts
from configs.HH4b_common.vbf_jet_collections import (
    VBF_BOOSTED_PAIR_COLL,
    VBF_SELECTED_PAIR_COLL,
    get_vbf_jet_collection,
)
import configs.VBF_HH4b.custom_cuts as vbf_cuts
from configs.HH4b_common.config_files.variables_dict import (
    variables_dict_jets,
    variables_dict_fatjets,
    variables_dict_higgs_mass,
    variables_dict_random_pt,
    variables_dict_vbf,
    variable_dict_bkg_morphing,
    get_variables_dict_sig_bkg_score,
)


def get_variables_dict(
    year,
    config_options_dict,
    JETS=False,
    CLASSIFICATION=False,
    RANDOM_PT=False,
    VBF_VARIABLES=False,
    BKG_MORPHING=False,
    SCORE=False,
    RUN2=False,
    SPANET=True,
    BOOSTED=False,
):
    """Function to create the variable dictionary for the PocketCoffea Configurator()."""
    variables_dict = {}
    if JETS:
        variables_dict.update(variables_dict_jets)
    if CLASSIFICATION:
        variables_dict.update(variables_dict_higgs_mass)
    if RANDOM_PT:
        variables_dict.update(variables_dict_random_pt)
    if VBF_VARIABLES:
        variables_dict.update(variables_dict_vbf)
    if BKG_MORPHING:
        variables_dict.update(variable_dict_bkg_morphing)
    if BOOSTED:
        variables_dict.update(variables_dict_fatjets)
    if SCORE:
        has_qt = False

        assert isinstance(
            year, list
        ), "Year must be a list of the years to be considered."

        for y in year:
            if "postEE" in y and config_options_dict["qt_postEE"]:
                params_qt = config_options_dict["qt_postEE"]
                print(f"Using postEE quantile transformation for year {y}")
            elif "preEE" in y and config_options_dict["qt_preEE"]:
                params_qt = config_options_dict["qt_preEE"]
                print(f"Using preEE quantile transformation for year {y}")
            else:
                print(f"Did not find a valid quantile transformation for year {y}")
                params_qt = None

            if params_qt:
                has_qt = True
                transformer = WeightedQuantileTransformer(
                    n_quantiles=0, output_distribution="uniform"
                )  # We read the quantiles and distribution anyway from the pickle file
                transformer.load(params_qt)
                transformed_bins = transformer.quantiles_
                transformed_bins[0] = 0.0
                transformed_bins[-1] = 1.0
                variables_dict.update(
                    get_variables_dict_sig_bkg_score(list(transformed_bins), y)
                )
            # bins_spanet_final = bins_spanet[::step]
        if not has_qt:
            variables_dict.update(get_variables_dict_sig_bkg_score(False))
    # Sort of lazy implementation. If neither SPANet nor RUN2 are active, no variables are saved.
    if (BOOSTED) and (not SPANET) and (not RUN2):
        print(" - Removing non-FatJetGood variables")
        variables_dict = {k: v for k, v in variables_dict.items() if "FatJetGood" in k}
    return variables_dict


# VBF pair assigned when a pairing (SPANet or Run 2) is applied, ordered by pt,
# with its mjj, deta and centrality: the VBF jets to use outside the workflow.
# When creating the SPANet training inputs the VBF jets are instead the first two
# of `JetGoodVBFMergedProvVBFPadded` (`SPANET_VBF_TRAINING_DEFAULT_COLUMNS_BTWP`)
VBF_SELECTED_PAIR_COLUMNS = {
    VBF_SELECTED_PAIR_COLL: [
        "provenance_vbf",
        "pt",
        "eta",
        "phi",
        "mass",
        "btagB",
        "index",
    ],
    "events": [
        f"mjj{VBF_SELECTED_PAIR_COLL}",
        f"deta{VBF_SELECTED_PAIR_COLL}",
        f"centralityHiggsLeading{VBF_SELECTED_PAIR_COLL}",
        f"centralityHiggsSubLeading{VBF_SELECTED_PAIR_COLL}",
    ],
}

SPANET_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP = [
    "provenance",
    "pt",
    "eta",
    "phi",
    "mass",
    "btagB_3wp",
    "btagB_5wp",
    "btagB",
]
SPANET_TRAINING_DEFAULT_COLUMNS_BTWP = {
    "JetGood": SPANET_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    "JetGoodPtFlatten": SPANET_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
}

SPANET_VBF_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP = [
    "provenance",
    "provenance_higgs",
    "provenance_z",
    "provenance_X",
    "provenance_vbf",
    "pt",
    "eta",
    "phi",
    "mass",
    "btagB_5wp",
    "btagB_3wp",
    "btagB",
]

SPANET_VBF_TRAINING_DEFAULT_COLUMNS_BTWP = {
    # merged collections with combined provenance
    "JetTotalSPANetPadded": SPANET_VBF_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    "JetTotalSPANetPtFlattenPadded": SPANET_VBF_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    "JetTotalSPANetPtFlattenHiggsMatchedPadded": SPANET_VBF_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    # collections with provenance_higgs and provenance_vbf saved separately
    "JetGoodProvHiggsPadded": SPANET_VBF_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    "JetGoodProvHiggsPtFlattenPadded": SPANET_VBF_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    "JetGoodVBFMergedProvVBFPadded": SPANET_VBF_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    "JetGoodVBFMergedProvVBFPtFlattenPadded": SPANET_VBF_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    # global collections
    "events": [
        "random_pt_weights",
        # merged collections with combined provenance
        "mjjJetTotalSPANetPadded",
        "detaJetTotalSPANetPadded",
        "centralityHiggsLeadingJetTotalSPANetPadded",
        "centralityHiggsSubLeadingJetTotalSPANetPadded",
        ## pt flatten
        "mjjJetTotalSPANetPtFlattenPadded",
        "detaJetTotalSPANetPtFlattenPadded",
        "centralityHiggsLeadingJetTotalSPANetPtFlattenPadded",
        "centralityHiggsSubLeadingJetTotalSPANetPtFlattenPadded",
        # collections with provenance_vbf saved separately
        "mjjJetGoodVBFMergedProvVBFPadded",
        "detaJetGoodVBFMergedProvVBFPadded",
        "centralityHiggsLeadingJetGoodVBFMergedProvVBFPadded",
        "centralityHiggsSubLeadingJetGoodVBFMergedProvVBFPadded",
        ## pt flatten
        "mjjJetGoodVBFMergedProvVBFPtFlattenPadded",
        "detaJetGoodVBFMergedProvVBFPtFlattenPadded",
        "centralityHiggsLeadingJetGoodVBFMergedProvVBFPtFlattenPadded",
        "centralityHiggsSubLeadingJetGoodVBFMergedProvVBFPtFlattenPadded",
    ],
}


def with_fw_momenta_columns(columns_dict, max_order_FW, fw_momenta_norms):
    """
    Return a copy of columns_dict with Fox-Wolfram moment columns appended to 'events'.

    Column names mirror what workflow_common.py writes:
        FW_H{i}_{norm}  and  FW_R{i}_{norm}  for i in range(max_order_FW).
    If max_order_FW <= 0 the dict is returned unchanged.
    """
    if max_order_FW <= 0:
        return columns_dict

    fw_cols = [
        f"FW_{kind}{i}_{norm}"
        for norm in fw_momenta_norms
        for i in range(max_order_FW)
        for kind in ("H", "R")
    ]
    return {**columns_dict, "events": list(columns_dict["events"]) + fw_cols}

DEFAULT_JET_COLUMN_PARAMS = [
    "pt",
    "eta",
    "phi",
    "mass",
    "btagB",
    "btagB_5wp",
    "btagB_3wp",
]
DEFAULT_JET_COLUMNS = {
    "JetGood": DEFAULT_JET_COLUMN_PARAMS,
}

DEFAULT_JET_COLUMNS_DICT = {
    f"JetGood_{x}": ["JetGood", x] for x in DEFAULT_JET_COLUMN_PARAMS
}

DEFAULT_FATJET_COLUMN_PARAMS = [
    "pt",
    "eta",
    "phi",
    "mass",
    "mass_regr",
    "msoftdrop",
    "btagBB",
    "btagCC",
]
DEFAULT_FATJET_COLUMNS = {
    "FatJetGood": DEFAULT_FATJET_COLUMN_PARAMS,
    "FatJetGoodSelected": DEFAULT_FATJET_COLUMN_PARAMS,
}

DEFAULT_FATJET_COLUMNS_DICT = {
    f"FatJetGood_{x}": ["FatJetGood", x] for x in DEFAULT_FATJET_COLUMN_PARAMS
}

def get_columns_list(
    columns_dict=DEFAULT_JET_COLUMNS,
    flatten=True,
):
    """Function to create the column definition for the PocketCoffea Configurator().
    If any of the input options is set to `None`, the default option is used. To not save anything, use `[]`.

    :param: collection_dict: dict: dictionary with the collection name as key and the list of parameters to save as value.
    :param: flatten: bool: whether to flatten the columns or not.
    """
    columns = []
    for collection, attributes in columns_dict.items():
        columns.append(ColOut(collection, attributes, flatten))

    # add the event number if not present in the columns
    add_event_number = True
    for col in columns:
        for attr in col.collection:
            if attr == "event" and col.name == "events":
                add_event_number = False
    if add_event_number:
        columns.append(ColOut("events", ["event"], flatten))

    return columns


def unpack_dict(d):
    out = []
    for v in d.values():
        if isinstance(v, dict):
            out.extend(unpack_dict(v))
        else:
            out.append(v[:2])  # keep only first 2 elements
    return out


def create_DNN_columns_list(run2, flatten, columns_dict, btag=True):
    """Create the columns of the DNN input variables"""
    column_dict = defaultdict(set)

    unpacked_columns = unpack_dict(columns_dict)

    for x, y in unpacked_columns:
        column_dict[x.split(":")[0]].add(y)
    column_dict = {x: list(y) for x, y in column_dict.items()}
    if btag:
        if "JetGoodFromHiggsOrdered" in column_dict:
            column_dict["JetGoodFromHiggsOrdered"].append(
                "btagB"
            )
            column_dict["JetGoodFromHiggsOrdered"].append(
                "btagB_5wp"
            )
            column_dict["JetGoodFromHiggsOrdered"].append(
                "provenance"
            )
        if "JetGoodFromHiggsOrdered5Jets" in column_dict:
            column_dict["JetGoodFromHiggsOrdered5Jets"].append(
                "btagB"
            )
            column_dict["JetGoodFromHiggsOrdered5Jets"].append(
                "btagB_5wp"
            )
            column_dict["JetGoodFromHiggsOrdered5Jets"].append(
                "provenance"
            )
    column_list = get_columns_list(column_dict, flatten)
    return column_list


def define_single_category(
    category_name, wide_cr=False, ggf_vbf_threshold=False, vbf_jet_coll=None
):
    """
    Define a single category for the analysis.

    If `vbf_discriminator` is defined, we MUST have a `ggf_vbf_threshold`. Otherwise the scan will crash.
    But the workflow should be designed in a way, that we never call this function without a threshold if one is needed.

    The resolved VBF and ggF categories need `vbf_jet_coll`, the VBF jet
    collection given by `get_vbf_jet_collection` (the boosted ones always use
    `JetGoodVBFEnergyOrdered`).
    """
    resolved_vbf_category = "boosted" not in category_name and (
        "vbf" in category_name or "ggf" in category_name
    )
    if resolved_vbf_category and vbf_jet_coll is None:
        raise ValueError(
            f"The category {category_name} requires the VBF jet collection "
            "`vbf_jet_coll` (see `get_vbf_jet_collection`)"
        )
    cut_list = []
    # number of b jets
    if "4b" in category_name:
        cut_list.append(cuts.hh4b_4b_region)
    if "2b" in category_name:
        cut_list.append(cuts.hh4b_2b_region)

    if "boosted_group" in category_name:
        cut_list.append(cuts.hh4b_boosted_signal_region_other_group)
    elif "boosted" in category_name: # Elif mainly because I only want one region for the moment for testing.
        if "incl" not in category_name and "fail" not in category_name:
            cut_list.append(cuts.hh4b_vbf_pass_discriminator_region(ggf_vbf_threshold))
        elif "fail" in category_name:
            cut_list.append(cuts.hh4b_vbf_fail_discriminator_region(ggf_vbf_threshold))
        if "signal" in category_name:
            cut_list.append(cuts.hh4b_boosted_signal_region)
        if "ttbar" in category_name:
            cut_list.append(cuts.hh4b_boosted_ttbar_control_region)
        if "qcd" in category_name:
            if "A" in category_name:
                cut_list.append(cuts.hh4b_boosted_qcd_control_region_A)
            elif "B" in category_name:
                cut_list.append(cuts.hh4b_boosted_qcd_control_region_B)
            elif "C" in category_name:
                cut_list.append(cuts.hh4b_boosted_qcd_control_region_C)
            else:
                cut_list.append(cuts.hh4b_boosted_qcd_control_region)
        if "vbf" in category_name:
            cut_list.append(cuts.hh4b_vbf_2_jets(VBF_BOOSTED_PAIR_COLL))
            cut_list.append(cuts.hh4b_boosted_vbf_region)
    # mass cuts
    elif "VR1" not in category_name:
        if "control" in category_name:
            if not wide_cr:
                cut_list.append(cuts.hh4b_control_region)
            else:
                cut_list.append(cuts.hh4b_control_region_wide)
        if "signal" in category_name:
            cut_list.append(cuts.hh4b_signal_region)
    if "VR1" in category_name:
        if "control" in category_name:
            cut_list.append(cuts.hh4b_VR1_control_region)
        if "signal" in category_name:
            cut_list.append(cuts.hh4b_VR1_signal_region)

    # blind region
    if "blind" in category_name:
        cut_list.append(cuts.blinded)

    if "vbf" in category_name and "boosted" not in category_name:
        cut_list.append(cuts.hh4b_vbf_2_jets(vbf_jet_coll))
        if "best_candidates" not in category_name and "discriminator" not in category_name:
            raise ValueError("Unrecognized region name")
        # VBF preselection on the deta and mjj of the VBF jets
        if "best_candidates" in category_name:
            if "nokincut" in category_name:
                cut_list.append(
                    cuts.hh4b_vbf_best_candidates_6_jets_nokincut_region(
                        vbf_jet_coll
                    )
                )
            else:
                cut_list.append(
                    cuts.hh4b_vbf_best_candidates_6_jets_region(vbf_jet_coll)
                )
        # ggF vs VBF discriminator cut, which can be combined with the VBF
        # preselection
        if "discriminator" in category_name:
            if ggf_vbf_threshold is False or ggf_vbf_threshold is None:
                raise ValueError(
                    f"The category {category_name} requires a ggf_vbf_threshold"
                )
            if "pass" in category_name:
                cut_list.append(
                    cuts.hh4b_vbf_pass_discriminator_region(ggf_vbf_threshold)
                )
            elif "fail" in category_name:
                cut_list.append(
                    cuts.hh4b_vbf_fail_discriminator_region(ggf_vbf_threshold)
                )
            else:
                raise ValueError("Unrecognized region name")
    # ggF region: complement of the full VBF selection, i.e. events failing any
    # of: >= 2 VBF jets, mjj and deta cuts, ggF vs VBF discriminator cut (only
    # if a `ggf_vbf_threshold` is given)
    if "ggf" in category_name and "boosted" not in category_name:
        cut_list.append(
            cuts.hh4b_vbf_selection_region(
                vbf_jet_coll,
                ggf_vbf_threshold if ggf_vbf_threshold not in (False, None) else None,
                pass_vbf=False,
            )
        )
    if "high_score" in category_name:
        cut_list.append(cuts.hh4b_sig_bkg_score_cut(0.5))

    if len(cut_list) < 1:  # aka if no cut applied
        cut_list.append(passthrough)

    category_item = {category_name: cut_list}

    return category_item


def define_second_approach_categories(
    bkg_morphing_dnn=False, blind=False, btag_wp_category_years=("2024", "2025")
):
    """Regions of the second approach.

    Mass plane (`hh4b_mass_plane_region`, H1/H2 = HiggsLeading/HiggsSubLeading):
    SR = X_HH < 30 GeV or X_ZZ < 2.6 or X_ZH < 1.9 or X_HZ < 1.9, CR = inside
    52 < M_H1 < 180 GeV, 50 < M_H2 < 173 GeV and outside the SR.
    b-tag categories (`hh4b_btag_category`) of the 4 HH jets:
      - SR3T1M, CR3T1M: >= 3 T, 4th jet at least M
      - SR3T1L, CR3T1L: exactly 3 T, 4th jet L but not M
      - SR2M, CR2M: exactly 2 M (CR2M -> CR3T1M background training, weights
        applied in SR2M)
      - SR3MnT: the 3 jets leading in b-tag score pass M but fail T (validation,
        training for SR3T1L)
      - SR3M, CR3M, SR4M, CR4M: exactly 3 M, 4 M, only in `btag_wp_category_years`
    With `bkg_morphing_dnn`, SR2M_postW and CR2M_postW get the morphing weights on
    data; with `blind`, SR3T1M_blind and SR3T1L_blind apply the blinding cut.
    """
    signal = cuts.hh4b_mass_plane_region("signal")
    control = cuts.hh4b_mass_plane_region("control")
    regions = {"SR": signal, "CR": control}
    categories = {
        "SR3T1M": ["SR", "3T1M"],
        "SR3T1L": ["SR", "3T1L"],
        "CR3T1M": ["CR", "3T1M"],
        "CR3T1L": ["CR", "3T1L"],
        "SR2M": ["SR", "2M"],
        "CR2M": ["CR", "2M"],
        "SR3MnT": ["SR", "3MnT"],
    }
    categories_dict = {
        name: [regions[region], cuts.hh4b_btag_category(category)]
        for name, (region, category) in categories.items()
    }
    for category in ("3M", "4M"):
        for region, cut in regions.items():
            categories_dict[f"{region}{category}"] = [
                cut,
                cuts.hh4b_btag_category(category, years=list(btag_wp_category_years)),
            ]
    if bkg_morphing_dnn:
        for name in ("SR2M", "CR2M"):
            categories_dict[f"{name}_postW"] = categories_dict[name]
    if blind:
        for name in ("SR3T1M", "SR3T1L"):
            categories_dict[f"{name}_blind"] = categories_dict[name] + [cuts.blinded]
    return categories_dict


# b-tag categories of the high b-tag VBF regions of the second approach, by year
SECOND_APPROACH_VBF_BTAG_CATEGORIES = {
    "3T1M": ["2022", "2023"],
    "3T1L": ["2022", "2023"],
    "4M": ["2024", "2025"],
}


def define_vbf_ggf_categories(
    inclusive_categories,
    vbf_jet_coll,
    vbf_discriminator=False,
    ggf_vbf_threshold=0.95,
    vbf_kin_cuts=True,
):
    """VBF and ggF categories, defined in the same way in every approach.

    For each inclusive category `<name>` (b-tag and mass-plane cuts, which
    depend on the approach) two categories are defined:
      - `vbf_<name>`: VBF selection, i.e. >= 2 VBF jets in the first two jets
        of `vbf_jet_coll` (`JetGoodVBFSelectedPair` when a pairing is applied,
        `JetGoodVBFMergedProvVBFPadded` when creating the SPANet inputs),
        with mjj > 400 GeV and deta > 3.5 if `vbf_kin_cuts` and, with a
        `vbf_discriminator`, the ggF/VBF score >= `ggf_vbf_threshold`
      - `ggf_<name>`: events failing any of the cuts of the VBF selection
    so that `vbf_<name>` and `ggf_<name>` are orthogonal and cover `<name>`.
    """
    vbf_cut_list = [cuts.hh4b_vbf_2_jets(vbf_jet_coll)]
    if vbf_kin_cuts:
        vbf_cut_list.append(cuts.hh4b_vbf_best_candidates_6_jets_region(vbf_jet_coll))
    if vbf_discriminator:
        vbf_cut_list.append(
            cuts.hh4b_vbf_pass_discriminator_region(ggf_vbf_threshold)
        )
    # complement of the full VBF selection
    ggf_cut = cuts.hh4b_vbf_selection_region(
        vbf_jet_coll,
        ggf_vbf_threshold if vbf_discriminator else None,
        pass_vbf=False,
        kin_cuts=vbf_kin_cuts,
    )
    categories_dict = {}
    for name, cut_list in inclusive_categories.items():
        categories_dict[f"vbf_{name}"] = cut_list + vbf_cut_list
        categories_dict[f"ggf_{name}"] = cut_list + [ggf_cut]
    return categories_dict


def define_second_approach_vbf_categories(
    vbf_jet_coll,
    bkg_morphing_dnn=False,
    vbf_discriminator=False,
    ggf_vbf_threshold=0.95,
):
    """VBF and ggF regions of the second approach (`define_vbf_ggf_categories`
    without the mjj and deta cuts: the VBF selection is only the presence of
    the 2 VBF jets and the ggF/VBF score cut), in the SR and CR of the mass
    plane of:
      - the high b-tag categories of `SECOND_APPROACH_VBF_BTAG_CATEGORIES`
        (3T1M and 3T1L in 2022-2023, 4M in 2024-2025; empty in the other years):
        vbf_SR3T1M, vbf_CR3T1M, vbf_SR3T1L, vbf_CR3T1L, vbf_SR4M, vbf_CR4M
      - the 2M category for the background estimate: vbf_SR2M, vbf_CR2M
        (and vbf_SR2M_postW, vbf_CR2M_postW with `bkg_morphing_dnn`)
    and the corresponding ggF regions (ggf_SR3T1M, ..., ggf_CR2M_postW).
    """
    regions = {
        "SR": cuts.hh4b_mass_plane_region("signal"),
        "CR": cuts.hh4b_mass_plane_region("control"),
    }
    btag_categories = {
        category: cuts.hh4b_btag_category(category, years=years)
        for category, years in SECOND_APPROACH_VBF_BTAG_CATEGORIES.items()
    }
    btag_categories["2M"] = cuts.hh4b_btag_category("2M")

    inclusive_categories = {}
    for category, btag_cut in btag_categories.items():
        for region, region_cut in regions.items():
            inclusive_categories[f"{region}{category}"] = [region_cut, btag_cut]
    if bkg_morphing_dnn:
        for name in ("SR2M", "CR2M"):
            inclusive_categories[f"{name}_postW"] = inclusive_categories[name]
    return define_vbf_ggf_categories(
        inclusive_categories,
        vbf_jet_coll,
        vbf_discriminator,
        ggf_vbf_threshold,
        vbf_kin_cuts=False,
    )


def define_first_approach_vbf_categories(
    vbf_jet_coll,
    bkg_morphing_dnn=False,
    vbf_discriminator=False,
    ggf_vbf_threshold=0.95,
    mixeddata=False,
    expandCR=False,
):
    """VBF and ggF regions of the first approach (`define_vbf_ggf_categories`),
    in the signal and control regions of:
      - the 4b category: vbf_4b_signal_region, vbf_4b_control_region
      - the 2b category (4b with `mixeddata`) with the morphing weights, for
        the background estimate (only with `bkg_morphing_dnn`):
        vbf_2b_signal_region_postW, vbf_2b_control_region_postW
    and the corresponding ggF regions (ggf_4b_signal_region, ...,
    ggf_2b_control_region_postW).
    """
    inclusive_categories = {}
    inclusive_categories |= define_single_category("4b_signal_region")
    inclusive_categories |= define_single_category("4b_control_region")
    if bkg_morphing_dnn:
        n_b = "4b" if mixeddata else "2b"
        inclusive_categories |= define_single_category(f"{n_b}_signal_region_postW")
        inclusive_categories |= define_single_category(
            f"{n_b}_control_region_postW", expandCR
        )
    return define_vbf_ggf_categories(
        inclusive_categories, vbf_jet_coll, vbf_discriminator, ggf_vbf_threshold
    )


def define_categories(
    bkg_morphing_dnn=False,
    blind=False,
    spanet=False,
    run2=False,
    vr1=False,
    expandCR=False,
    mixeddata=False,
    btag_sf_comp=False,
    boosted=False,
    other_group=False,
    split_qcd=True,
    vbf_analysis=False,
    vbf_discriminator=False,
    ggf_vbf_threshold=0.95,  # Only needed if using a vbf_discriminator
    high_score_reg=False,
    approach="first",
):
    """Define the categories for the analysis.

    The resolved VBF and ggF categories cut on the VBF jet collection given by
    `get_vbf_jet_collection`: `JetGoodVBFSelectedPair` when a pairing (`spanet`
    or `run2`) is applied, `JetGoodVBFMergedProvVBFPadded` when creating the
    SPANet training inputs.
    """
    categories_dict = {}
    vbf_jet_coll = get_vbf_jet_collection(spanet, run2)

    if approach == "second" and not boosted:
        categories_dict |= define_second_approach_categories(bkg_morphing_dnn, blind)
        if vbf_analysis:
            categories_dict |= define_second_approach_vbf_categories(
                vbf_jet_coll, bkg_morphing_dnn, vbf_discriminator, ggf_vbf_threshold
            )
        return categories_dict

    if boosted:
        if vbf_analysis:
            is_vbf = "_vbf"
        else:
            is_vbf = ""
        if other_group:
            categories_dict |= define_single_category(f"boosted{is_vbf}_boosted_group_signal_region")
        elif not vbf_discriminator:
            categories_dict |= define_single_category(f"boosted{is_vbf}_incl_region")
            categories_dict |= define_single_category(f"boosted{is_vbf}_incl_signal_region", ggf_vbf_threshold=ggf_vbf_threshold)
            categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_A_region", ggf_vbf_threshold=ggf_vbf_threshold)
            categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_B_region", ggf_vbf_threshold=ggf_vbf_threshold)
            categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_C_region", ggf_vbf_threshold=ggf_vbf_threshold)
            if bkg_morphing_dnn:
                categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_A_region_postW", ggf_vbf_threshold=ggf_vbf_threshold)
                categories_dict |= (
                    define_single_category(f"boosted{is_vbf}_incl_qcd_C_region_postW" + "_blind", ggf_vbf_threshold=ggf_vbf_threshold)
                    if blind
                    else {}
                )
                categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_C_region_postW", ggf_vbf_threshold=ggf_vbf_threshold)
        else:
            categories_dict |= define_single_category(f"boosted{is_vbf}_signal_region", ggf_vbf_threshold=ggf_vbf_threshold)
            categories_dict |= define_single_category(f"boosted{is_vbf}_ttbar_region", ggf_vbf_threshold=ggf_vbf_threshold)
            categories_dict |= define_single_category(f"boosted{is_vbf}_pass_region", ggf_vbf_threshold=ggf_vbf_threshold)
            categories_dict |= define_single_category(f"boosted{is_vbf}_fail_region", ggf_vbf_threshold=ggf_vbf_threshold)
            if split_qcd:
                categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_A_region", ggf_vbf_threshold=ggf_vbf_threshold)
                categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_B_region", ggf_vbf_threshold=ggf_vbf_threshold)
                categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_C_region", ggf_vbf_threshold=ggf_vbf_threshold)
                if bkg_morphing_dnn:
                    categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_A_region_postW", ggf_vbf_threshold=ggf_vbf_threshold)
                    categories_dict |= (
                        define_single_category(f"boosted{is_vbf}_qcd_C_region_postW" + "_blind", ggf_vbf_threshold=ggf_vbf_threshold)
                        if blind
                        else {}
                    )
                    categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_C_region_postW", ggf_vbf_threshold=ggf_vbf_threshold)
            else:
                categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_region", ggf_vbf_threshold=ggf_vbf_threshold)
    elif not vr1:
        categories_dict |= define_single_category("4b_region")
        categories_dict |= define_single_category("4b_control_region")
        categories_dict |= define_single_category("4b_signal_region")
        categories_dict |= (
            define_single_category("4b_signal_region_blind")
            if blind
            else {}
        )
        if high_score_reg:
            categories_dict |= (
                define_single_category("4b_signal_region_high_score_blind")
                if blind
                else {}
            )
            categories_dict |= define_single_category("4b_signal_region_high_score")
        if not mixeddata:
            categories_dict |= define_single_category(f"2b_control_region_preW", expandCR)
            categories_dict |= define_single_category(f"2b_signal_region_preW", expandCR)
            categories_dict |= (
                define_single_category(f"2b_signal_region_preW_blind")
                if blind
                else {}
            )
            if high_score_reg:
                categories_dict |= define_single_category(f"2b_signal_region_preW_high_score", expandCR)
                categories_dict |= (
                    define_single_category("2b_signal_region_preW_high_score_blind", expandCR)
                    if blind
                    else {}
                )
        else:
            categories_dict |= define_single_category(f"4b_control_region_preW", expandCR)
            categories_dict |= define_single_category(f"4b_signal_region_preW", expandCR)
            categories_dict |= (
                define_single_category(f"4b_signal_region_preW_blind")
                if blind
                else {}
            )
            if high_score_reg:
                categories_dict |= define_single_category(f"4b_signal_region_preW_high_score", expandCR)
                categories_dict |= (
                    define_single_category(f"4b_signal_region_preW_blind_high_score")
                    if blind
                    else {}
                )
                categories_dict |= (
                    define_single_category("4b_signal_region_preW_high_score_blind", expandCR)
                    if blind
                    else {}
                )

        if bkg_morphing_dnn:
            if not mixeddata:
                categories_dict |= define_single_category(
                    f"2b_control_region_postW", expandCR
                )
                categories_dict |= (
                    define_single_category(f"2b_signal_region_postW_blind")
                    if blind
                    else {}
                )
                categories_dict |= define_single_category(
                    f"2b_signal_region_postW"
                )
                categories_dict |= (
                    define_single_category(f"2b_region_postW")
                )
                if high_score_reg:
                    categories_dict |= (
                        define_single_category("2b_signal_region_postW_high_score_blind")
                        if blind
                        else {}
                    )
                    categories_dict |= define_single_category(
                        "2b_signal_region_postW_high_score"
                    )
            else:
                categories_dict |= define_single_category(
                    f"4b_control_region_postW", expandCR
                )
                categories_dict |= (
                    define_single_category(f"4b_signal_region_postW_blind")
                    if blind
                    else {}
                )
                categories_dict |= define_single_category(
                    f"4b_signal_region_postW"
                )
                if high_score_reg:
                    categories_dict |= (
                        define_single_category(f"4b_signal_region_postW_high_score_blind")
                        if blind
                        else {}
                    )
                    categories_dict |= define_single_category(
                        f"4b_signal_region_postW_high_score"
                    )
        if vbf_analysis:
            if vbf_discriminator:
                categories_dict = {}
            categories_dict |= define_first_approach_vbf_categories(
                vbf_jet_coll,
                bkg_morphing_dnn,
                vbf_discriminator,
                ggf_vbf_threshold,
                mixeddata,
                expandCR,
            )
    else:
        categories_dict |= define_single_category(f"4b_VR1_control_region")
        categories_dict |= define_single_category(
            "2b_VR1_control_region_preW"
        )
        categories_dict |= define_single_category(f"4b_VR1_signal_region")
        categories_dict |= define_single_category(
            "2b_VR1_signal_region_preW"
        )
        if bkg_morphing_dnn:
            categories_dict |= define_single_category(
                "2b_VR1_control_region_postW"
            )
            categories_dict |= define_single_category(
                "2b_VR1_signal_region_postW"
            )

    if not spanet and not run2 and not boosted:
        # add the 2b control region post W for the old DNN
        categories_dict |= define_single_category("4b_region")

    if btag_sf_comp:
        btag_sf_categories = {}
        for key, value in categories_dict.items():
            btag_sf_categories[f"{key}_sf_btag"] = value
        categories_dict |= btag_sf_categories

    return categories_dict


def define_preselection(options):
    ## Define the preselection to apply
    # jet pt field on which the pt cuts of the resolved preselections are applied
    pt_type = options["preselection_pt_field"]
    no_btag = "no_btag" in options.keys() and options["no_btag"]
    if options["approach"] == "second":
        # second approach: its own year-dependent preselection
        preselection = [cuts.hh4b_presel_second_approach(pt_type, no_btag=no_btag)]
    elif no_btag:
        preselection = [cuts.hh4b_presel_nobtag(pt_type)]
    else:
        if options["vbf_presel"]:
            # block vbf_presel because it's done on the wrong jet collection
            raise ValueError("vbf_presel is not supported anymore!")
            if options["tight_cuts"]:
                preselection = [vbf_cuts.vbf_hh4b_presel_tight]
            else:
                preselection = [vbf_cuts.vbf_hh4b_presel]
        elif options["boosted_presel"]:
            preselection = [cuts.hh4b_boosted_presel]
        elif options["era_dependent_presel"]:
            preselection = [
                cuts.hh4b_presel_era_dependent(pt_type, options["tight_cuts"])
            ]
        else:
            if options["tight_cuts"]:
                preselection = [cuts.hh4b_presel_tight(pt_type)]
            else:
                preselection = [cuts.hh4b_presel(pt_type)]

    # Add the Jet Veto Map
    # Do this in the preselection to select jets based on
    # corrected pT after the Calibrators have run
    if not options["boosted_presel"] and not options["mixeddata"]:
        preselection.append(
            cuts.hh4b_JetVetoMap_regressed
            if options.get("jet_veto_map_regressed_pt", False)
            else cuts.hh4b_JetVetoMap
        )

    # Optional gen-level cut selecting X(->bb)X(->bb) decays (X = H or Z).
    # Needed only by the private ZZ/ZH samples, which are inclusive in the
    # decay channel. Off by default because it changes the yields of every
    # sample whose name contains "HH", "ZH" or "ZZ".
    if options["xx4b_presel"]:
        preselection.append(cuts.XX4b_presel)

    return preselection
