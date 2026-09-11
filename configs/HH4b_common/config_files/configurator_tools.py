from collections import defaultdict
import copy

from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters.cuts import passthrough
from utils_configs.quantile_transformer import WeightedQuantileTransformer

import configs.HH4b_common.custom_cuts_common as cuts
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


SPANET_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP = [
    "provenance",
    "pt",
    "eta",
    "phi",
    "mass",
    "btagPNetB_3wp",
    "btagPNetB_5wp",
    "btagPNetB",
]
SPANET_TRAINING_DEFAULT_COLUMNS_BTWP = {
    "JetGood": SPANET_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    "JetGoodPtFlatten": SPANET_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
}

SPANET_VBF_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP = [
    "provenance",
    "provenance_higgs",
    "provenance_vbf",
    "pt",
    "eta",
    "phi",
    "mass",
    "btagPNetB_5wp",
    "btagPNetB_3wp",
    "btagPNetB",
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

SPANET_VBF_TRAINING_DEFAULT_COLUMNS_BTWP_RUN2 = copy.deepcopy(
    SPANET_VBF_TRAINING_DEFAULT_COLUMNS_BTWP
)
SPANET_VBF_TRAINING_DEFAULT_COLUMNS_BTWP_RUN2["events"] = [
    "random_pt_weights",
    # merged collections with combined provenance
    "mjjJetTotalSPANetPadded",
    "detaJetTotalSPANetPadded",
    ## pt flatten
    "mjjJetTotalSPANetPtFlattenPadded",
    "detaJetTotalSPANetPtFlattenPadded",
    # collections with provenance_vbf saved separately
    "mjjJetGoodVBFMergedProvVBFPadded",
    "detaJetGoodVBFMergedProvVBFPadded",
    ## pt flatten
    "mjjJetGoodVBFMergedProvVBFPtFlattenPadded",
    "detaJetGoodVBFMergedProvVBFPtFlattenPadded",
]

DEFAULT_JET_COLUMN_PARAMS = [
    "pt",
    "eta",
    "phi",
    "mass",
    "btagPNetB",
    "btagPNetB_5wp",
    "btagPNetB_3wp",
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
                "btagPNetB"
            )
            column_dict["JetGoodFromHiggsOrdered"].append(
                "btagPNetB_5wp"
            )
            column_dict["JetGoodFromHiggsOrdered"].append(
                "provenance"
            )
        if "JetGoodFromHiggsOrdered5Jets" in column_dict:
            column_dict["JetGoodFromHiggsOrdered5Jets"].append(
                "btagPNetB"
            )
            column_dict["JetGoodFromHiggsOrdered5Jets"].append(
                "btagPNetB_5wp"
            )
            column_dict["JetGoodFromHiggsOrdered5Jets"].append(
                "provenance"
            )
    column_list = get_columns_list(column_dict, flatten)
    return column_list


def define_single_category(category_name, wide_cr=False, ggf_vbf_threshold=False):
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
            cut_list.append(cuts.hh4b_vbf_2_jets)
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
    other_group=False,
    split_qcd=True,
    vbf_analysis=False,
    vbf_discriminator=False,
    ggf_vbf_threshold=0.95,  # Only needed if using a vbf_discriminator
    high_score_reg=False
):
    """Define the categories for the analysis."""
    categories_dict = {}

    if boosted:
        if vbf_analysis:
            is_vbf = "_vbf"
        else:
            is_vbf = ""
        if other_group:
            categories_dict |= define_single_category(f"boosted{is_vbf}_boosted_group_signal_region")
        elif not vbf_discriminator:
            categories_dict |= define_single_category(f"boosted{is_vbf}_incl_region")
            categories_dict |= define_single_category(f"boosted{is_vbf}_incl_signal_region", ggf_vbf_threshold)
            categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_A_region", ggf_vbf_threshold)
            categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_B_region", ggf_vbf_threshold)
            categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_C_region", ggf_vbf_threshold)
            if bkg_morphing_dnn:
                categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_A_region_postW", ggf_vbf_threshold)
                categories_dict |= (
                    define_single_category(f"boosted{is_vbf}_incl_qcd_C_region_postW" + "_blind", ggf_vbf_threshold)
                    if blind
                    else {}
                )
                categories_dict |= define_single_category(f"boosted{is_vbf}_incl_qcd_C_region_postW", ggf_vbf_threshold)
        else:
            categories_dict |= define_single_category(f"boosted{is_vbf}_signal_region", ggf_vbf_threshold)
            categories_dict |= define_single_category(f"boosted{is_vbf}_ttbar_region", ggf_vbf_threshold)
            categories_dict |= define_single_category(f"boosted{is_vbf}_pass_region", ggf_vbf_threshold)
            categories_dict |= define_single_category(f"boosted{is_vbf}_fail_region", ggf_vbf_threshold)
            if split_qcd:
                categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_A_region", ggf_vbf_threshold)
                categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_B_region", ggf_vbf_threshold)
                categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_C_region", ggf_vbf_threshold)
                if bkg_morphing_dnn:
                    categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_A_region_postW", ggf_vbf_threshold)
                    categories_dict |= (
                        define_single_category(f"boosted{is_vbf}_qcd_C_region_postW" + "_blind", ggf_vbf_threshold)
                        if blind
                        else {}
                    )
                    categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_C_region_postW", ggf_vbf_threshold)
            else:
                categories_dict |= define_single_category(f"boosted{is_vbf}_qcd_region", ggf_vbf_threshold)
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
    if "no_btag" in options.keys() and options["no_btag"]:
        preselection = [cuts.hh4b_presel_nobtag]
    else:
        if options["vbf_presel"]:
            # block vbf_presel because it's done on the wrong jet collection
            raise ValueError("vbf_presel is not spported anymore!")
            if options["tight_cuts"]:
                preselection = [vbf_cuts.vbf_hh4b_presel_tight]
            else:
                preselection = [vbf_cuts.vbf_hh4b_presel]
        elif options["boosted_presel"]:
            preselection = [cuts.hh4b_boosted_presel]
        else:
            if options["tight_cuts"]:
                preselection = [cuts.hh4b_presel_tight]
            else:
                preselection = [cuts.hh4b_presel]

    # Add the Jet Veto Map
    # Do this in the preselection to select jets based on
    # corrected pT after the Calibrators have run
    if not options["boosted_presel"] and not options["mixeddata"]:
        preselection.append(cuts.hh4b_JetVetoMap)
    return preselection
