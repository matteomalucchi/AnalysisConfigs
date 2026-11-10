from collections import defaultdict
import copy
import re

from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters.cuts import passthrough
from utils_configs.quantile_transformer import WeightedQuantileTransformer

import configs.HH4b_common.custom_cuts_common as cuts
import configs.HH4b_common.custom_cut_functions_common as cuts_f
import configs.VBF_HH4b.custom_cuts as vbf_cuts
from configs.HH4b_common.config_files.variables_dict import (
    variables_dict_jets,
    variables_dict_fatjets,
    variables_dict_boosted_bdt,
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
    BDT=False,
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
    if BOOSTED and BDT:
        variables_dict.update(variables_dict_boosted_bdt)
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
                    get_variables_dict_sig_bkg_score(list(transformed_bins), y, BOOSTED)
                )
            # bins_spanet_final = bins_spanet[::step]
        if not has_qt:
            variables_dict.update(get_variables_dict_sig_bkg_score(False, "", BOOSTED))
    # Sort of lazy implementation. If neither SPANet nor RUN2 are active, no variables are saved.
    if (BOOSTED) and (not SPANET) and (not RUN2):
        print(" - Removing non-FatJetGood variables")
        variables_dict |= {k: v for k, v in variables_dict.items() if "FatJetGood" in k}
    return variables_dict


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
    # merged collections with provenance_higgs and provenance_vbf saved separately
    # "JetTotalSPANetSeparateProvHiggsVBFPadded": SPANET_VBF_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    # "JetTotalSPANetSeparateProvHiggsVBFPtFlattenPadded": SPANET_VBF_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    # "JetTotalSPANetSeparateProvHiggsVBFPtFlattenOnlyHiggsPadded": SPANET_VBF_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
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
    "mass_orig",
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


# TXbb working points (VL, L, M, T, XT, as in `generate_btag_workingpoints`) and the upper edge, defined next to the cut
# functions. The category suffix `txbbwp<i>` selects the bin (TXBB_WP_EDGES[i], TXBB_WP_EDGES[i + 1]].
TXBB_WP_EDGES = cuts_f.TXBB_WP_EDGES


def txbb_wp_bins(region_setting):
    """Indices of the TXbb WP bins that lie above the TXbb threshold of the signal regions (C, D)."""
    threshold = 0.8 if region_setting == "40GeV" else 0.65
    return [i for i in range(len(TXBB_WP_EDGES) - 1) if TXBB_WP_EDGES[i] >= threshold]


def define_single_category(category_name, wide_cr=False, ggf_vbf_threshold=False, full_sideband=False, region_setting="40GeV", lead_sublead="sublead", optimised_cuts=False, txbb_cdf_wps=False):
    """
    Define a single category for the analysis.

    If `vbf_discriminator` is defined, we MUST have a `ggf_vbf_threshold`. Otherwise the scan will crash.
    But the workflow should be designed in a way, that we never call this function without a threshold if one is needed.
    """
    cut_list = []
    # number of b jets
    if "4b" in category_name:
        cut_list.append(cuts.hh4b_4b_region)
    if "2b" in category_name:
        cut_list.append(cuts.hh4b_2b_region)

    elif "boosted" in category_name: # Elif mainly because I only want one region for the moment for testing.
        cut_list.append(cuts.hh4b_boosted_baseline(region_setting))
        if "boosted_signal" in category_name:
            cut_list.append(cuts.hh4b_boosted_TXbb_signal(region_setting))
            if "sideband" in category_name:
                if full_sideband:
                    cut_list.append(cuts.hh4b_boosted_mass_sideband_full(region_setting, lead_sublead))
                else:
                    cut_list.append(cuts.hh4b_boosted_mass_sideband_lower(region_setting, lead_sublead))
            else:
                cut_list.append(cuts.hh4b_boosted_mass_signal(region_setting, lead_sublead))
        elif "boosted_control" in category_name:
            cut_list.append(cuts.hh4b_boosted_TXbb_control(region_setting))
            if "sideband" in category_name:
                if full_sideband:
                    cut_list.append(cuts.hh4b_boosted_mass_sideband_full(region_setting, lead_sublead))
                else:
                    cut_list.append(cuts.hh4b_boosted_mass_sideband_lower(region_setting, lead_sublead))
            else:
                cut_list.append(cuts.hh4b_boosted_mass_signal(region_setting, lead_sublead))
        # TXbb working point bins, for the closure checks. The same bin edges are used in both regions, on the
        # nominal TXbb in the signal regions (C, D) and on the morphed TXbb in the control regions (A, B), or, with
        # `txbb_cdf_wps`, on the CDF-rescaled working-point index in all regions.
        wp_match = re.search(r"txbbwp(\d+)", category_name)
        if wp_match:
            wp_bin = int(wp_match.group(1))
            cut_list.append(cuts.hh4b_boosted_TXbb_bin(TXBB_WP_EDGES[wp_bin], TXBB_WP_EDGES[wp_bin + 1], morph="boosted_control" in category_name, txbb_cdf=txbb_cdf_wps))
        # The signal categories cut on the subleading TXbb. In the control regions (A, B) the TXbb is below the
        # signal threshold by construction, so there the categories cut on the morphed TXbb (`btagBBTXbb_morph`)
        # instead. The signal regions (C, D) always use the nominal TXbb.
        # With `txbb_cdf_wps` (option `txbb_cdf_wp_cuts`) all regions use the CDF-rescaled working-point index of the
        # subleading Higgs instead, so the cuts must sit on working points and no shape of the TXbb is used.
        if "cat1" in category_name:
            cut_list.append(cuts.hh4b_boosted_category_1(txbb_morph="boosted_control" in category_name, region_setting=region_setting, optimised_cuts=optimised_cuts, txbb_cdf=txbb_cdf_wps))
        elif "catvbf" in category_name:
            cut_list.append(cuts.hh4b_boosted_category_vbf(txbb_morph="boosted_control" in category_name, region_setting=region_setting, optimised_cuts=optimised_cuts, txbb_cdf=txbb_cdf_wps))
        elif "cat2" in category_name:
            cut_list.append(cuts.hh4b_boosted_category_2(txbb_morph="boosted_control" in category_name, region_setting=region_setting, optimised_cuts=optimised_cuts, txbb_cdf=txbb_cdf_wps))
        elif "cat3" in category_name:
            cut_list.append(cuts.hh4b_boosted_category_3(txbb_morph="boosted_control" in category_name, region_setting=region_setting, optimised_cuts=optimised_cuts, txbb_cdf=txbb_cdf_wps))

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
        cut_list.append(cuts.hh4b_vbf_2_jets)
        if "best_candidates" in category_name:
            if "nokincut" in category_name:
                cut_list.append(
                    cuts.hh4b_vbf_best_candidates_6_jets_nokincut_region
                )
            else:
                cut_list.append(cuts.hh4b_vbf_best_candidates_6_jets_region)
        elif "discriminator" in category_name:
            if "pass" in category_name:
                cut_list.append(cuts.hh4b_vbf_pass_discriminator_region)
            elif "fail" in category_name:
                cut_list.append(cuts.hh4b_vbf_fail_discriminator_region)
            else:
                raise ValueError("Unrecognized region name")

        else:
            raise ValueError("Unrecognized region name")
    if "high_score" in category_name:
        cut_list.append(cuts.hh4b_sig_bkg_score_cut(0.5))

    if len(cut_list) < 1:  # aka if no cut applied
        cut_list.append(passthrough)

    category_item = {category_name: cut_list}

    return category_item


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
    split_qcd=True,
    vbf_analysis=False,
    vbf_discriminator=False,
    ggf_vbf_threshold=0.95,  # Only needed if using a vbf_discriminator
    high_score_reg=False,
    full_sideband=False,
    lead_sublead="sublead",
    region_setting="40GeV",
    optimised_cuts=False,
    bdt_model="",
    morphing_direction="btag",
    txbb_wp_categories=False,
    txbb_cdf_wps=False,
):
    """Define the categories for the analysis."""
    categories_dict = {}

    if boosted:
        if morphing_direction not in ("btag", "mass"):
            raise ValueError(f"Unknown bkg_morphing_direction '{morphing_direction}', expected 'btag' or 'mass'")
        # All ABCD regions must share the same region definition, otherwise the regions are inconsistent
        region_kwargs = dict(full_sideband=full_sideband, region_setting=region_setting, lead_sublead=lead_sublead, optimised_cuts=optimised_cuts, txbb_cdf_wps=txbb_cdf_wps)
        # The signal categories cut on the DNN scores (optimised cuts) or, without optimised cuts, on the BDT scores.
        # If neither is available, only the ABCD regions are defined.
        signal_categories = bool(optimised_cuts) or bool(bdt_model)
        # The ABCD plane is defined by (thresholds depend on `region_setting`, "40GeV" / "50GeV"):
        #     - the subleading Higgs TXbb: above 0.8 / 0.65 are the signal regions (C, D), below the control regions (A, B)
        #     - the mass of the Higgs chosen by `lead_sublead`: inside 110-155 / 100-150 GeV are the mass signal
        #       regions (B, D), in the sideband (40 / 50 GeV below the window, or on both sides with `full_sideband`)
        #       the sideband regions (A, C)
        #
        #                  mass sideband   mass window
        #     high TXbb          C              D       <- signal region
        #     low TXbb           A              B
        #
        # The morphing is trained from A to the neighbouring region and applied to the region next to D:
        #     - "btag": trained A -> C (along TXbb), applied to B -> boosted_control_region_B_postW
        #     - "mass": trained A -> B (along mass), applied to C -> boosted_signal_sideband_region_C_postW
        categories_dict |= define_single_category("boosted_signal_region_D", **region_kwargs)
        categories_dict |= define_single_category("boosted_signal_sideband_region_C", **region_kwargs)
        categories_dict |= define_single_category("boosted_control_region_B", **region_kwargs)
        categories_dict |= define_single_category("boosted_control_sideband_region_A", **region_kwargs)
        if bkg_morphing_dnn:
            categories_dict |= define_single_category("boosted_control_sideband_region_A_postW", **region_kwargs)
            if signal_categories:
                categories_dict |= define_single_category("boosted_signal_region_D_cat1", **region_kwargs)
                categories_dict |= define_single_category("boosted_signal_region_D_cat2", **region_kwargs)
                categories_dict |= define_single_category("boosted_signal_region_D_cat3", **region_kwargs)
                categories_dict |= define_single_category("boosted_signal_region_D_catvbf", **region_kwargs)
                # Also define the region C subcategories, for closure check. 
                categories_dict |= define_single_category("boosted_signal_sideband_region_C_cat1", **region_kwargs)
                categories_dict |= define_single_category("boosted_signal_sideband_region_C_cat2", **region_kwargs)
                categories_dict |= define_single_category("boosted_signal_sideband_region_C_cat3", **region_kwargs)
                categories_dict |= define_single_category("boosted_signal_sideband_region_C_catvbf", **region_kwargs)
            if morphing_direction == "mass":
                categories_dict |= define_single_category("boosted_signal_sideband_region_C_postW", **region_kwargs)
                if signal_categories:
                    categories_dict |= define_single_category("boosted_signal_sideband_region_C_cat1_postW", **region_kwargs)
                    categories_dict |= define_single_category("boosted_signal_sideband_region_C_cat2_postW", **region_kwargs)
                    categories_dict |= define_single_category("boosted_signal_sideband_region_C_cat3_postW", **region_kwargs)
                    categories_dict |= define_single_category("boosted_signal_sideband_region_C_catvbf_postW", **region_kwargs)
            elif morphing_direction == "btag":
                categories_dict |= define_single_category("boosted_control_region_B_postW", **region_kwargs)
                if signal_categories:
                    categories_dict |= define_single_category("boosted_control_region_B_cat1_postW", **region_kwargs)
                    categories_dict |= define_single_category("boosted_control_region_B_cat2_postW", **region_kwargs)
                    categories_dict |= define_single_category("boosted_control_region_B_cat3_postW", **region_kwargs)
                    categories_dict |= define_single_category("boosted_control_region_B_catvbf_postW", **region_kwargs)
                    # Also define the region A subcategories, for closure check.
                    categories_dict |= define_single_category("boosted_control_sideband_region_A_cat1_postW", **region_kwargs)
                    categories_dict |= define_single_category("boosted_control_sideband_region_A_cat2_postW", **region_kwargs)
                    categories_dict |= define_single_category("boosted_control_sideband_region_A_cat3_postW", **region_kwargs)
                    categories_dict |= define_single_category("boosted_control_sideband_region_A_catvbf_postW", **region_kwargs)
                if txbb_wp_categories:
                    # Closure of the morphing (trained A -> C) in bins of the TXbb working point: the nominal TXbb in C
                    # and the morphed TXbb in A_postW, with the same bin edges
                    for wp_bin in txbb_wp_bins(region_setting):
                        categories_dict |= define_single_category(f"boosted_signal_sideband_region_C_txbbwp{wp_bin}", **region_kwargs)
                        categories_dict |= define_single_category(f"boosted_control_sideband_region_A_txbbwp{wp_bin}_postW", **region_kwargs)
    
                    

        # elif not vbf_discriminator:
        #     categories_dict |= define_single_category(f"boosted{is_vbf}_incl_region")
        #     categories_dict |= define_single_category(f"boosted{is_vbf}_incl_signal_region", ggf_vbf_threshold)
        #     categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_A_region", ggf_vbf_threshold)
        #     categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_B_region", ggf_vbf_threshold)
        #     categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_C_region", ggf_vbf_threshold)
        #     if bkg_morphing_dnn:
        #         categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_A_region_postW", ggf_vbf_threshold)
        #         categories_dict |= (
        #             define_single_category(f"boosted{is_vbf}_incl_qcd_C_region_postW" + "_blind", ggf_vbf_threshold)
        #             if blind
        #             else {}
        #         )
        #         categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_C_region_postW", ggf_vbf_threshold)
        # else:
        #     categories_dict |= define_single_category(f"boosted{is_vbf}_signal_region", ggf_vbf_threshold)
        #     categories_dict |= define_single_category(f"boosted{is_vbf}_ttbar_region", ggf_vbf_threshold)
        #     categories_dict |= define_single_category(f"boosted{is_vbf}_pass_region", ggf_vbf_threshold)
        #     categories_dict |= define_single_category(f"boosted{is_vbf}_fail_region", ggf_vbf_threshold)
        #     if split_qcd:
        #         categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_A_region", ggf_vbf_threshold)
        #         categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_B_region", ggf_vbf_threshold)
        #         categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_C_region", ggf_vbf_threshold)
        #         if bkg_morphing_dnn:
        #             categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_A_region_postW", ggf_vbf_threshold)
        #             categories_dict |= (
        #                 define_single_category(f"boosted{is_vbf}_qcd_C_region_postW" + "_blind", ggf_vbf_threshold)
        #                 if blind
        #                 else {}
        #             )
        #             categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_C_region_postW", ggf_vbf_threshold)
        #     else:
        #         categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_region", ggf_vbf_threshold)
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
            # NOTE: this region requires at least 6 jets
            categories_dict |= define_single_category(
                "vbf_best_candidates_6_jets_4b_region"
            )
            # NOTE: this region requires at least 6 jets
            categories_dict |= define_single_category(
                "vbf_best_candidates_6_jets_nokincut_4b_region"
            )

            if vbf_discriminator:
                # NOTE: this region requires at least 6 jets and that the vbf vs ggf score is above/below the threshold
                categories_dict |= define_single_category(
                    "vbf_pass_discriminator_4b_region"
                )
                categories_dict |= define_single_category(
                    "vbf_fail_discriminator_4b_region"
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
    if "no_btag" in options.keys() and options["no_btag"]:
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
        preselection.append(cuts.hh4b_JetVetoMap)

    # Optional gen-level cut selecting X(->bb)X(->bb) decays (X = H or Z).
    # Needed only by the private ZZ/ZH samples, which are inclusive in the
    # decay channel. Off by default because it changes the yields of every
    # sample whose name contains "HH", "ZH" or "ZZ".
    if options["xx4b_presel"]:
        preselection.append(cuts.XX4b_presel)

    return preselection
