import os
import cloudpickle
import utils_configs.quantile_transformer as quantile_transformer

from configs.HH4b_common.config_files.__config_file__ import (
    config_options_dict,
    onnx_model_dict,
)
import pocket_coffea.lib.calibrators.legacy.legacy_calibrators as legacy_cal
from pocket_coffea.lib.calibrators.common.common import JetsCalibrator


from pocket_coffea.lib.weights.common.common import common_weights

# from pocket_coffea.parameters.cuts import passthrough
# rom pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters import defaults
from pocket_coffea.parameters.histograms import *

# from collections import defaultdict
from pocket_coffea.utils.configurator import Configurator
from workflow import HH4bbQuarkMatchingProcessor

import configs.HH4b_common.custom_cuts_common as cuts
from configs.HH4b_common.config_files.configurator_tools import (
    SPANET_TRAINING_DEFAULT_COLUMNS_BTWP,
    DEFAULT_FATJET_COLUMNS,
    create_DNN_columns_list,
    define_categories,
    define_preselection,
    get_columns_list,
    get_variables_dict,
    define_single_category,
)
from configs.HH4b_common.custom_weights import (
    bkg_morphing_dnn_weight,
)
from configs.HH4b_common.params.CustomWeights import SF_btag_fixed_multiple_wp

localdir = os.path.dirname(os.path.abspath(__file__))


# Loading default parameters

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir)

# adding object preselection
year = ["2022_postEE", "2022_preEE"]  # , "2023_preBPix", "2023_postBPix"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/../HH4b_common/params/object_preselection_{config_options_dict['approach']}_approach.yaml",
    f"{localdir}/../HH4b_common/params/triggers.yaml",
    f"{localdir}/../HH4b_common/params/variations.yaml",
    f"{localdir}/../HH4b_common/params/btagging_multipleWP.yaml",
    f"{localdir}/../HH4b_common/params/btagging_sampleGroups.yaml",
    # f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_withoutVariations_withJERC.yaml",
    # f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_withVariations.yaml",
    f"{localdir}/../HH4b_common/params/jets_calibration_regression_json.yaml",
    update=True,
)


if config_options_dict["save_chunk"]:
    config_options_dict["dump_columns_as_arrays_per_chunk"] = config_options_dict["save_chunk"]


# score transform still in testing. So far hardcoded to be 2022_postEE...
variables_dict = {}
# Define the variables to save
variables_dict = get_variables_dict(
    year,
    config_options_dict,
    CLASSIFICATION=False,
    RANDOM_PT=False,
    VBF_VARIABLES=False,
    BKG_MORPHING=False,  # bool(onnx_model_dict["bkg_morphing_dnn"]),
    SCORE=bool(config_options_dict["sig_bkg_dnn"]),
    RUN2=config_options_dict["run2"],
    SPANET=bool(config_options_dict["spanet"]),
    BOOSTED=config_options_dict["boosted"],
)
# print(variables_dict)

## Define the preselection to apply
preselection = define_preselection(config_options_dict)


# Defining the used samples
sample_ggF_list = [
      "GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_skimmed",
      "GluGlutoHHto4B_spanet_kl-5p00_kt-1p00_c2-0p00_skimmed",
      "GluGlutoHHto4B_spanet_kl-2p45_kt-1p00_c2-0p00_skimmed",
      "GluGlutoHHto4B_spanet_kl-m2p00_kt-1p00_c2-0p00_skimmed",
      "GluGlutoHHto4B_spanet_kl-m1p00_kt-1p00_c2-0p00_skimmed",
      "GluGlutoHHto4B_spanet_kl-0p00_kt-0p00_c2-0p00_skimmed",
      "GluGlutoHHto4B_spanet_kl-3p50_kt-1p00_c2-0p00_skimmed",
      "GluGlutoHHto4B_spanet_kl-4p00_kt-1p00_c2-0p00_skimmed",
      "GluGlutoHHto4B_spanet_kl-3p00_kt-1p00_c2-0p00_skimmed",
      "GluGlutoHHto4B_spanet_kl-2p00_kt-1p00_c2-0p00_skimmed",
      "GluGlutoHHto4B_spanet_kl-1p50_kt-1p00_c2-0p00_skimmed",
      "GluGlutoHHto4B_spanet_kl-0p50_kt-1p00_c2-0p00_skimmed",
      "GluGluHHto4B_Par-c2-0p00-kl-0p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia9",
      "GluGluHHto4B_Par-c2-0p00-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
      "GluGluHHto4B_Par-c2-0p00-kl-2p45-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
      "GluGluHHto4B_Par-c2-0p00-kl-5p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
      "GluGluHHto4B_Par-c2-0p10-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
      "GluGluHHto4B_Par-c2-0p35-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
      "GluGluHHto4B_Par-c2-1p00-kl-0p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
      "GluGluHHto4B_Par-c2-2p24-kl-m20p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
      "GluGluHHto4B_Par-c2-3p00-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
      "GluGluHHto4B_Par-c2-m2p00-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
]
sample_vbf_list = [
      "VBFHHto4B_Par-CV-1-C2V-0-C3-1_TuneCP5_13p6TeV_madgraph-pythia8",
      "VBFHHto4B_Par-CV-1-C2V-1-C3-1_TuneCP5_13p6TeV_madgraph-pythia8",
      "VBFHHto4B_Par-CV-1p74-C2V-1p37-C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8",
      "VBFHHto4B_Par-CV-2p12-C2V-3p87-C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8",
      "VBFHHto4B_Par-CV-m0p012-C2V-0p030-C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8",
      "VBFHHto4B_Par-CV-m0p758-C2V-1p44-C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8",
      "VBFHHto4B_Par-CV-m0p962-C2V-0p959-C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8",
      "VBFHHto4B_Par-CV-m1p21-C2V-1p94-C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8",
      "VBFHHto4B_Par-CV-m1p60-C2V-2p72-C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8",
      "VBFHHto4B_Par-CV-m1p83-C2V-3p57-C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8",
]
sample_mixed_list = [
    # "MixedData_2022_preEE",
    # "MixedData_2022_postEE_EraE",
    "MixedData_2022_postEE_EraF",
    # "MixedData_2022_postEE_EraG",
    # "MixedData_2023_preBPix",
    # "MixedData_2023_postBPix"
        ]

if config_options_dict["mixeddata"]:
    sample_list = sample_mixed_list
else:
    sample_list = [
        # "DATA_JetMET_JMENano_C_skimmed",
        # "DATA_JetMET_JMENano_D_skimmed",
        # "DATA_JetMET_JMENano_E_skimmed",
        # "DATA_JetMET_JMENano_F_skimmed",
        # "DATA_JetMET_JMENano_G_skimmed",
        # "GluGlutoHHto4B_spanet_skimmed",
        # "GluGlutoHHto4B_spanet_skimmed_SM",
        # "GluGlutoHHto4B_spanet_skimmed",
        # "GluGlutoHHto4B",
        # "DATA_JetMET_JMENano_2023_Cv1_skimmed",
        # "DATA_JetMET_JMENano_2023_Cv2_skimmed",
        # "DATA_ParkingHH_2023_Cv3",
        # "DATA_ParkingHH_2023_Cv4",
        # "DATA_ParkingHH_2023_Dv1",
        # "DATA_ParkingHH_2023_Dv2",
        "DATA_ParkingHH",
    ] + sample_ggF_list


# Define the categories to save
categories_dict = define_categories(
    bkg_morphing_dnn=config_options_dict["bkg_morphing_dnn"],
    blind=config_options_dict["blind"],
    spanet=config_options_dict["spanet"],
    run2=config_options_dict["run2"],
    vr1=config_options_dict["vr1"],
    boosted=config_options_dict["boosted"],
    expandCR=config_options_dict["expandCR"],
    mixeddata=config_options_dict["mixeddata"],
)
# AKA if no model is applied
# print(onnx_model_dict)
if all([model == "" for model in onnx_model_dict.values()]) and not (config_options_dict["boosted"]):
    print("Didn't find any onnx model and not running boosted analysis. Will choose region for SPANet training")
    categories_dict = define_single_category("4b_region")

# print("categories_dict", categories_dict)

# VBF SPECIFIC REGIONS
# **{f"4b_semiTight_LeadingPt_region": [hh4b_4b_region, semiTight_leadingPt]},
# **{f"4b_semiTight_LeadingMjj_region": [hh4b_4b_region, semiTight_leadingMjj]},
# **{f"4b_semiTight_LeadingMjj_region": [hh4b_4b_region, semiTight_leadingMjj]}
# **{"4b_VBFtight_region": [hh4b_4b_region, vbf_wrapper()]},
#
# **{
#     f"4b_VBFtight_{list(ab[0].keys())[i]}_region": [
#         hh4b_4b_region,
#         vbf_wrapper(ab[i]),
#     ]
#     for i in range(0, 6)
# },
#
# **{"4b_VBF_generalSelection_region": [hh4b_4b_region, VBF_generalSelection_region]},
# **{"4b_VBF_region": [hh4b_4b_region, VBF_region]},
# **{f"4b_VBF_0{i}qvg_region": [hh4b_4b_region, VBF_region, qvg_regions[f"qvg_0{i}_region"]] for i in range(5, 10)},
# **{f"4b_VBF_0{i}qvg_generalSelection_region": [hh4b_4b_region, VBF_generalSelection_region, qvg_regions[f"qvg_0{i}_region"]] for i in range(5, 10)},

# Define the columns to save
total_input_variables = {}
column_list = []

assert not (config_options_dict["random_pt"] and config_options_dict["run2"])
if config_options_dict["dnn_variables"]:
    total_input_variables = (
        config_options_dict["sig_bkg_dnn_input_variables"]
        | config_options_dict["bkg_morphing_dnn_input_variables"]
        | {"year": ["events", "year"]} |
        {
            "jet_log_pt": ["JetGoodFromHiggsOrdered5Jets", "pt", "log_norm"],
            "jet_eta": ["JetGoodFromHiggsOrdered5Jets", "eta", "norm"],
            "jet_phi": ["JetGoodFromHiggsOrdered5Jets", "phi", "norm"],
            "jet_log_mass": ["JetGoodFromHiggsOrdered5Jets", "mass", "log_norm"],
            "jet_btag": ["JetGoodFromHiggsOrdered5Jets", "btagPNetB_5wp"],
            "h1_jet_log_pt": ["JetGoodFromHiggsOrderedLeading", "pt", "log_norm"],
            "h1_jet_eta": ["JetGoodFromHiggsOrderedLeading", "eta", "norm"],
            "h1_jet_phi": ["JetGoodFromHiggsOrderedLeading", "phi", "norm"],
            "h1_jet_log_mass": ["JetGoodFromHiggsOrderedLeading", "mass", "log_norm"],
            "h1_jet_btag": ["JetGoodFromHiggsOrderedLeading", "btagPNetB_5wp"],
            "h1_jet_prov": ["JetGoodFromHiggsOrderedLeading", "reco_provenance"],
            "h2_jet_log_pt": ["JetGoodFromHiggsOrderedSubLeading", "pt", "log_norm"],
            "h2_jet_eta": ["JetGoodFromHiggsOrderedSubLeading", "eta", "norm"],
            "h2_jet_phi": ["JetGoodFromHiggsOrderedSubLeading", "phi", "norm"],
            "h2_jet_log_mass": ["JetGoodFromHiggsOrderedSubLeading", "mass", "log_norm"],
            "h2_jet_btag": ["JetGoodFromHiggsOrderedSubLeading", "btagPNetB_5wp"],
            "h2_jet_prov": ["JetGoodFromHiggsOrderedSubLeading", "reco_provenance"],
            "a1_jet_log_pt": ["add_jet1pt", "pt", "log_norm"],
            "a1_jet_eta": ["add_jet1pt", "eta", "norm"],
            "a1_jet_phi": ["add_jet1pt", "phi", "norm"],
            "a1_jet_log_mass": ["add_jet1pt", "mass", "log_norm"],
            "a1_jet_btag": ["add_jet1pt", "btagPNetB_5wp"],
            "a1_jet_prov": ["add_jet1pt", "reco_provenance"],
            }
    )
    if config_options_dict["spanet"]:
        total_input_variables |= {
            "Delta_pairing_probabilities": ["events", "Delta_pairing_probabilities"],
            "Arctanh_Delta_pairing_probabilities": [
                "events",
                "Arctanh_Delta_pairing_probabilities",
            ],
            "Delta_pairing_probabilities_best_worst": ["events", "Delta_pairing_probabilities_best_worst"],
            "Arctanh_Delta_pairing_probabilities_best_worst": [
                "events",
                "Arctanh_Delta_pairing_probabilities_best_worst",
            ],
            "Binned_Arctanh_Delta_pairing_probabilities": [
                "events",
                "Binned_Arctanh_Delta_pairing_probabilities",
            ],
            "Padded_Arctanh_Delta_pairing_probabilities": [
                "events",
                "Padded_Arctanh_Delta_pairing_probabilities",
            ],
        }
    # print(total_input_variables)

    column_list = create_DNN_columns_list(
        False, not config_options_dict["save_chunk"], total_input_variables, btag=True
    )
elif all([model == "" for model in onnx_model_dict.values()]) and not (config_options_dict["boosted"]):
    if "wp" in config_options_dict["spanet_input_name_list"][-1]:
        print("Taking btag Working Points")
        column_list = get_columns_list(SPANET_TRAINING_DEFAULT_COLUMNS_BTWP, not config_options_dict["save_chunk"])
    else:
        column_list = get_columns_list(SPANET_TRAINING_DEFAULT_COLUMNS, not config_options_dict["save_chunk"])
    if config_options_dict["random_pt"]:
        column_list += get_columns_list({"events": ["random_pt_weights"]})
else:
    column_list = get_columns_list(flatten=not config_options_dict["save_chunk"])

# Add special columns
if config_options_dict["sig_bkg_dnn"]:
    column_list += get_columns_list({"events": ["sig_bkg_dnn_score"]})
if not any(
    ["DATA" in sample for sample in sample_list]
) and not any(["Mixed" in sample for sample in sample_list]):
    column_list += get_columns_list(
        {
            "events": [
                "correct_prediction",
                "correct_prediction_fully_matched",
                "mask_fully_matched",
            ]
        }
    )
if config_options_dict["boosted"]:
    column_list += get_columns_list(DEFAULT_FATJET_COLUMNS, not config_options_dict["save_chunk"])


save_separate_weights = False
if save_separate_weights:
    column_list += get_columns_list(
        {
            "events": [
                "weight_single_lumi",
                "weight_single_XS",
                "weight_single_pileup",
                "weight_single_sf_btag_fixed_multiple_wp",
            ]
        }
    )

bysample_bycategory_column_dict = {}
for sample in sample_list:
    bysample_bycategory_column_dict[sample] = {
        "inclusive": [],
        "bycategory": {},
    }
    for category in categories_dict.keys():
        bysample_bycategory_column_dict[sample]["bycategory"][category] = (
            column_list
            + (
                get_columns_list({"events": ["bkg_morphing_spread_dnn_weights"]})
                if "DATA" in sample
                and config_options_dict["bkg_morphing_spread_dnn"]
                and "postW" in category
                else []
            )
        )
# print("bysample_bycategory_column_dict", bysample_bycategory_column_dict)

# Define the weights to apply
bysample_bycategory_weight_dict = {}
for sample in sample_list:
    if "DATA" in sample.upper():
        bysample_bycategory_weight_dict[sample] = {"inclusive": [], "bycategory": {}}
        for category in categories_dict.keys():
            if "postW" in category:
                bysample_bycategory_weight_dict[sample]["bycategory"][category] = [
                    "bkg_morphing_dnn_weight"
                ]

# print("bysample_bycategory_weight_dict", bysample_bycategory_weight_dict)

if config_options_dict["boosted"]:
    skimming_cut_list = cuts.skimming_cut_list_boosted
else:
    skimming_cut_list = cuts.skimming_cut_list

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet_redirector.json",
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet_skimmed_pnfs_redirector.json",
            f"{localdir}/../HH4b_common/datasets/mixeddata_large.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET_pnfs_redirector.json",
            f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH_pnfs_redirector.json",
            f"{localdir}/../HH4b_common/datasets/background_TTtoX_pnfs_redirector.json",
            f"{localdir}/../HH4b_common/datasets/signal_VBFHHto4B_Par_2024_pnfs_redirector.json",
            f"{localdir}/../HH4b_common/datasets/signal_GluGluHHto4B_Par_2024_pnfs_redirector.json",
        ],
        "filter": {
            "samples": sample_list,
            "samples_exclude": [],
            "year": year,
        },
        "subsamples": {},
    },
    workflow=HH4bbQuarkMatchingProcessor,
    workflow_options=config_options_dict,
    skim=cuts.skimming_cut_list(config_options_dict),
    preselections=preselection,
    categories=categories_dict,
    weights_classes=common_weights + [bkg_morphing_dnn_weight, SF_btag_fixed_multiple_wp],
    calibrators=[JetsCalibrator] if not config_options_dict['mixeddata'] else [],
    weights={
        "common": {
            # "inclusive": ["genWeight", "lumi", "XS", "pileup", "sf_btag_fixed_multiple_wp"],
            # "inclusive": ["genWeight", "lumi", "XS", "pileup"],
            # "inclusive": ["genWeight", "lumi", "XS"],
            "inclusive": [],
            "bycategory": {
            },
        },
        "bysample": bysample_bycategory_weight_dict,
    },
    variations={
        "weights": {
            "common": {
                # "inclusive": ["XS", "lumi", "pileup", "sf_btag_fixed_multiple_wp"],
                "inclusive": [],
                "bycategory": {},
            },
            "bysample": {},
        },
        "shape": {
            "common": {
                # "inclusive": ["jet_calibration"],
                "inclusive": [],
                },
            }
    },
    variables=variables_dict,
    columns={
        "common": {
            "inclusive": [],
            "bycategory": {},
        },
        "bysample": bysample_bycategory_column_dict,
        # "bysample": {},
    },
)


cloudpickle.register_pickle_by_value(quantile_transformer)
