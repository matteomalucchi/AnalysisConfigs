import os
import cloudpickle
import utils.quantile_transformer as quantile_transformer
from collections import defaultdict

from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
)
from pocket_coffea.parameters.histograms import *
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters import defaults
from pocket_coffea.lib.weights.common.common import common_weights
import pocket_coffea.lib.calibrators.legacy.legacy_calibrators as legacy_cal 
from pocket_coffea.lib.calibrators.common.common import JetsCalibrator


from configs.VBF_HH4b.workflow import VBFHH4bProcessor
from configs.VBF_HH4b.custom_cuts import vbf_hh4b_presel, vbf_hh4b_presel_tight

from configs.HH4b_common.custom_weights import (
    bkg_morphing_dnn_weight,
    bkg_morphing_dnn_weightRun2,
)
from configs.HH4b_common.config_files.configurator_tools import (
    get_variables_dict,
    get_columns_list,
    DEFAULT_JET_COLUMNS_DICT,
    create_DNN_columns_list,
    define_single_category,
    define_categories,
    define_preselection,
)

from configs.HH4b_common.config_files.__config_file__ import (
    config_options_dict,
)

import configs.HH4b_common.custom_cuts_common as cuts

BASELINE = False


localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = ["2022_postEE"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/../HH4b_common/params/object_preselection.yaml",
    f"{localdir}/../HH4b_common/params/triggers.yaml",
    f"{localdir}/../HH4b_common/params/btagging_multipleWP.yaml",
    # f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_withVariations.yaml",
    f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_withoutVariations_withJERC.yaml",
    update=True,
)

if config_options_dict["save_chunk"]:
    config_options_dict["dump_columns_as_arrays_per_chunk"] = config_options_dict[
        "save_chunk"
    ]

# Define the variables to save
variables_dict = get_variables_dict(
    year,
    config_options_dict,
    CLASSIFICATION=False,
    VBF_VARIABLES=False,
    BKG_MORPHING=False,  # bool(onnx_model_dict["bkg_morphing_dnn"]),
    SCORE=bool(config_options_dict["sig_bkg_dnn"]),
    RUN2=config_options_dict["run2"],
    SPANET=bool(config_options_dict["spanet"]),
)
# variables_dict = {}

## Define the preselection to apply
preselection = define_preselection(config_options_dict)


sample_ggF_list = [
    "GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-m2p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-m1p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-5p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-2p45_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-0p00_kt-0p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-3p50_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-4p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-3p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-2p00_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-1p50_kt-1p00_c2-0p00_skimmed",
    "GluGlutoHHto4B_spanet_kl-0p50_kt-1p00_c2-0p00_skimmed",
]

## Define the samples to process
sample_list = (
    [
        ## 2022 preEE
        # "DATA_JetMET_JMENano_C_skimmed",
        # "DATA_JetMET_JMENano_D_skimmed",
        ## 2022 postEE
        "DATA_JetMET_JMENano_E_skimmed",
        "DATA_JetMET_JMENano_F_skimmed",
        "DATA_JetMET_JMENano_G_skimmed",
    ]
    + sample_ggF_list
    + (
        [
        #     "GluGlutoHHto4B_spanet_skimmed",
        #     # "GluGlutoHHto4B",
        #     # "VBF_HHto4B",
        # "GluGlutoHHto4B_spanet"
        ]
    )
)


## Define the categories to save
categories_dict = define_categories(
    bkg_morphing_dnn=config_options_dict["bkg_morphing_dnn"],
    blind=config_options_dict["blind"],
    spanet=config_options_dict["spanet"],
    run2=config_options_dict["run2"],
    vr1=config_options_dict["vr1"],
)

# categories_dict=define_single_category("control_region")
# categories_dict|=define_single_category("signal_region")
if BASELINE:
    categories_dict = {"baseline": [passthrough]}

# print("categories_dict", categories_dict)

### VBF SPECIFIC REGIONS ###
# **{f"4b_semiTight_LeadingPt_region": [hh4b_4b_region, semiTight_leadingPt]},
# **{f"4b_semiTight_LeadingMjj_region": [hh4b_4b_region, semiTight_leadingMjj]},
# **{f"4b_semiTight_LeadingMjj_region": [hh4b_4b_region, semiTight_leadingMjj]}
# **{"4b_VBFtight_region": [hh4b_4b_region, VBFtight_region]},
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


## Define the columns to save
total_input_variables = {}

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

if config_options_dict["dnn_variables"]:
    total_input_variables |= (
        config_options_dict["sig_bkg_dnn_input_variables"]
        | config_options_dict["bkg_morphing_dnn_input_variables"]
        | {"year": ["events", "year"]}
    )
else:
    total_input_variables |= DEFAULT_JET_COLUMNS_DICT
if BASELINE:
    total_input_variables |= DEFAULT_JET_COLUMNS_DICT
# print(total_input_variables)

column_list = create_DNN_columns_list(
    False, not config_options_dict["save_chunk"], total_input_variables, btag=False
)
column_listRun2 = create_DNN_columns_list(
    True, not config_options_dict["save_chunk"], total_input_variables, btag=False
)

# print(column_list)

# Add special columns
if config_options_dict["sig_bkg_dnn"] and config_options_dict["spanet"]:
    column_list += get_columns_list({"events": ["sig_bkg_dnn_score"]})
if config_options_dict["sig_bkg_dnn"] and config_options_dict["run2"]:
    column_listRun2 += get_columns_list({"events": ["sig_bkg_dnn_scoreRun2"]})



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

## Define the weights to apply
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

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            # f"{localdir}/../HH4b_common/datasets/QCD.json",
            f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_local.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_local_rucio.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_SM_local_rucio_redirector.json",
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet_redirector.json",
            f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed.json",
            f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed_separateSamples.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_test.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET_skimmed.json",
        ],
        "filter": {
            "samples": sample_list,
            "samples_exclude": [],
            # "year": year,
        },
        "subsamples": {},
    },
    workflow=VBFHH4bProcessor,
    workflow_options=config_options_dict,
    skim=cuts.skimming_cut_list,
    preselections=preselection,
    categories=categories_dict,
    weights_classes=common_weights
    + ([bkg_morphing_dnn_weight, bkg_morphing_dnn_weightRun2] if not BASELINE else []),
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
            ],
            "bycategory": {},
        },
        "bysample": bysample_bycategory_weight_dict,
    },
    calibrators=[legacy_cal.JetsCalibrator, legacy_cal.JetsPtRegressionCalibrator],
    # calibrators=[JetsCalibrator],
    variations={
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory": {},
            },
            "bysample": {},
        },
        # "shape": {
        #     "common": {
        #         "inclusive": ["jet_calibration_with_pt_regression_legacy"],
        #         # "inclusive": [],
        #         },
        #     }
    },
    variables=variables_dict,
    columns={
        "common": {
            "inclusive": [],
            "bycategory": {},
        },
        "bysample": bysample_bycategory_column_dict,
    },
)
cloudpickle.register_pickle_by_value(quantile_transformer)
