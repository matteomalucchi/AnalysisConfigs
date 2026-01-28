import os
import cloudpickle
import utils.quantile_transformer as quantile_transformer

from configs.HH4b_common.config_files.__config_file__ import (
    config_options_dict,
    onnx_model_dict,
)
from pocket_coffea.lib.calibrators.legacy.legacy_calibrators import (
    JetsCalibrator,
    JetsPtRegressionCalibrator,
)


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
    SPANET_TRAINING_DEFAULT_COLUMNS,
    SPANET_TRAINING_DEFAULT_COLUMNS_BTWP,
    DEFAULT_FATJET_COLUMNS,
    create_DNN_columns_list,
    define_categories,
    define_single_category,
    get_columns_list,
    get_variables_dict,
    define_preselection
)
from configs.HH4b_common.custom_weights import (
    bkg_morphing_dnn_weight,
    bkg_morphing_dnn_weightRun2,
)
from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_input_variables,
    bkg_morphing_boosted_dnn_input_variables,
    # bkg_morphing_dnn_input_variables_altOrder,
    sig_bkg_dnn_input_variables,
    sig_bkg_boosted_dnn_input_variables,
)
from configs.HH4b_common.params.CustomWeights import SF_btag_fixed_multiple_wp

localdir = os.path.dirname(os.path.abspath(__file__))


# Loading default parameters

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = ["2022_postEE", "2022_preEE"]  # , "2023_preBPix", "2023_postBPix"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/../HH4b_common/params/object_preselection.yaml",
    f"{localdir}/../HH4b_common/params/triggers.yaml",
    f"{localdir}/../HH4b_common/params/btagging_multipleWP.yaml",
    f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_withoutVariations_withJERC.yaml",
    # f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_withVariations.yaml",
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
    BKG_MORPHING=bool(config_options_dict["bkg_morphing_dnn"]),
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
    # "GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_skimmed",
    # "GluGlutoHHto4B_spanet_kl-m2p00_kt-1p00_c2-0p00_skimmed",
    # "GluGlutoHHto4B_spanet_kl-m1p00_kt-1p00_c2-0p00_skimmed",
    # "GluGlutoHHto4B_spanet_kl-5p00_kt-1p00_c2-0p00_skimmed",
    # "GluGlutoHHto4B_spanet_kl-2p45_kt-1p00_c2-0p00_skimmed",
    # "GluGlutoHHto4B_spanet_kl-0p00_kt-0p00_c2-0p00_skimmed",
    # "GluGlutoHHto4B_spanet_kl-3p50_kt-1p00_c2-0p00_skimmed",
    # "GluGlutoHHto4B_spanet_kl-4p00_kt-1p00_c2-0p00_skimmed",
    # "GluGlutoHHto4B_spanet_kl-3p00_kt-1p00_c2-0p00_skimmed",
    # "GluGlutoHHto4B_spanet_kl-2p00_kt-1p00_c2-0p00_skimmed",
    # "GluGlutoHHto4B_spanet_kl-1p50_kt-1p00_c2-0p00_skimmed",
    # "GluGlutoHHto4B_spanet_kl-0p50_kt-1p00_c2-0p00_skimmed",
    # "GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00",
    # "GluGlutoHHto4B_spanet_kl-m2p00_kt-1p00_c2-0p00",
    # "GluGlutoHHto4B_spanet_kl-m1p00_kt-1p00_c2-0p00",
    # "GluGlutoHHto4B_spanet_kl-5p00_kt-1p00_c2-0p00",
    # "GluGlutoHHto4B_spanet_kl-2p45_kt-1p00_c2-0p00",
    # "GluGlutoHHto4B_spanet_kl-0p00_kt-0p00_c2-0p00",
    # "GluGlutoHHto4B_spanet_kl-3p50_kt-1p00_c2-0p00",
    # "GluGlutoHHto4B_spanet_kl-4p00_kt-1p00_c2-0p00",
    # "GluGlutoHHto4B_spanet_kl-3p00_kt-1p00_c2-0p00",
    # "GluGlutoHHto4B_spanet_kl-2p00_kt-1p00_c2-0p00",
    # "GluGlutoHHto4B_spanet_kl-1p50_kt-1p00_c2-0p00",
    # "GluGlutoHHto4B_spanet_kl-0p50_kt-1p00_c2-0p00",
]
sample_list = [
    # "DATA_JetMET_JMENano_C_skimmed",
    # "DATA_JetMET_JMENano_D_skimmed",
    # "DATA_JetMET_JMENano_E_skimmed",
    # "DATA_JetMET_JMENano_F_skimmed",
    # "DATA_JetMET_JMENano_G_skimmed",
    # "DATA_JetMET_JMENano_E_boosted_skimmed",
    # "DATA_JetMET_JMENano_F_boosted_skimmed",
    # "DATA_JetMET_JMENano_G_boosted_skimmed",
    # "DATA_JetMET_JMENano_E",
    # "DATA_JetMET_JMENano_F",
    # "DATA_JetMET_JMENano_G",
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
    "TTtoLNu2Q_skimmed",
    "TTto2L2Nu_skimmed",
    "TTto4Q_skimmed",
] + sample_ggF_list

# Define the categories to save
categories_dict = define_categories(
    bkg_morphing_dnn=config_options_dict["bkg_morphing_dnn"],
    blind=config_options_dict["blind"],
    spanet=config_options_dict["spanet"],
    run2=config_options_dict["run2"],
    vr1=config_options_dict["vr1"],
    boosted=config_options_dict["boosted"],
    split_qcd=config_options_dict["split_qcd"] if config_options_dict["boosted"] else False,
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
column_listRun2 = []

assert not (config_options_dict["random_pt"] and config_options_dict["run2"])
if config_options_dict["dnn_variables"]:
    if config_options_dict["boosted"]:
        print("Using boosted DNN input variables")
        total_input_variables = (
            sig_bkg_boosted_dnn_input_variables
            | bkg_morphing_boosted_dnn_input_variables
            | {"year": ["events", "year"]}
        )
    else:
        total_input_variables = (
            sig_bkg_dnn_input_variables
            | bkg_morphing_dnn_input_variables
            | {"year": ["events", "year"]}
        )
    if config_options_dict["spanet"]:
        total_input_variables |= {
            "Delta_pairing_probabilities": ["events", "Delta_pairing_probabilities"],
            "Arctanh_Delta_pairing_probabilities": [
                "events",
                "Arctanh_Delta_pairing_probabilities",
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
        False, not config_options_dict["save_chunk"], total_input_variables, btag=False
    )
    column_listRun2 = create_DNN_columns_list(
        True, not config_options_dict["save_chunk"], total_input_variables, btag=False
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
    column_listRun2 = get_columns_list(flatten=not config_options_dict["save_chunk"])

# Add special columns
if config_options_dict["sig_bkg_dnn"] and config_options_dict["spanet"]:
    column_list += get_columns_list({"events": ["sig_bkg_dnn_score"]})
if config_options_dict["sig_bkg_dnn"] and config_options_dict["run2"]:
    column_listRun2 += get_columns_list({"events": ["sig_bkg_dnn_scoreRun2"]})
if config_options_dict["spanet"] and not any(
    ["DATA" in sample for sample in sample_list]
):
    column_list += get_columns_list(
        {
            "events": [
                "correct_prediction",
                "correct_prediction_fully_matched",
                "mask_fully_matched",
            ]
        }
    )
if config_options_dict["run2"] and not any(
    ["DATA" in sample for sample in sample_list]
):
    column_listRun2 += get_columns_list(
        {
            "events": [
                "correct_predictionRun2",
                "correct_prediction_fully_matchedRun2",
                "mask_fully_matched",
            ]
        }
    )
if config_options_dict["boosted"]:
    column_list += get_columns_list(DEFAULT_FATJET_COLUMNS, not config_options_dict["save_chunk"])



bysample_bycategory_column_dict = {}
for sample in sample_list:
    bysample_bycategory_column_dict[sample] = {
        "inclusive": [],
        "bycategory": {},
    }
    for category in categories_dict.keys():
        if "Run2" in category:
            bysample_bycategory_column_dict[sample]["bycategory"][category] = (
                column_listRun2
                + (
                    get_columns_list(
                        {"events": ["bkg_morphing_spread_dnn_weightsRun2"]}
                    )
                    if "DATA" in sample
                    and config_options_dict["bkg_morphing_spread_dnn"]
                    and "postW" in category
                    else []
                )
            )
        else:
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
    if "DATA" in sample:
        bysample_bycategory_weight_dict[sample] = {"inclusive": [], "bycategory": {}}
        for category in categories_dict.keys():
            if "postW" in category:
                if "Run2" in category:
                    bysample_bycategory_weight_dict[sample]["bycategory"][category] = [
                        "bkg_morphing_dnn_weightRun2"
                    ]
                else:
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
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b.json",
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet_redirector_boosted.json",
            f"{localdir}/../HH4b_common/datasets/background_TTtoX_skimmed.json",
            ## f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed.json",
            ## f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed_SM.json",
            ## f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed_separateSamples.json",
            ## f"{localdir}/../HH4b_common/datasets/DATA_JetMET_skimmed.json",
            # f"{localdir}/../HH4b_common/datasets/QCD.json",
            # f"{localdir}/../HH4b_common/datasets/SPANet_classification.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_local.json",
            # f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b_local.json",
            f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_JetMET.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET_boosted_skimmed_local.json",
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
    skim=skimming_cut_list,
    preselections=preselection,
    categories=categories_dict,
    weights_classes=common_weights
    + [bkg_morphing_dnn_weight, bkg_morphing_dnn_weightRun2, SF_btag_fixed_multiple_wp],
    calibrators=[JetsCalibrator, JetsPtRegressionCalibrator],
    weights={
        "common": {
            # "inclusive": ["genWeight", "lumi", "XS", "sf_btag_fixed_multiple_wp"],
            # "inclusive": ["genWeight", "lumi", "XS", "pileup"],
            "inclusive": ["genWeight", "lumi", "XS"],
            # "inclusive": [],
            "bycategory": {
            },
        },
        "bysample": bysample_bycategory_weight_dict,
    },
    variations={
        "weights": {
            "common": {
                # "inclusive": ["pileup"],  # , "sf_btag_fixed_multiple_wp"],
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
