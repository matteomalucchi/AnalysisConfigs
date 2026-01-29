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
    SPANET_TRAINING_DEFAULT_COLUMNS_BTWP,
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
    # bkg_morphing_dnn_input_variables_altOrder,
    sig_bkg_dnn_input_variables,
)

localdir = os.path.dirname(os.path.abspath(__file__))


# Loading default parameters

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir)

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
# Define the variables to save
variables_dict = {}

## Define the preselection to apply
preselection = define_preselection(config_options_dict)


# Defining the used samples
sample_ggF_list = [
    #  "GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_skimmed",
#     "GluGlutoHHto4B_spanet_kl-m2p00_kt-1p00_c2-0p00_skimmed",
#     "GluGlutoHHto4B_spanet_kl-m1p00_kt-1p00_c2-0p00_skimmed",
    #  "GluGlutoHHto4B_spanet_kl-5p00_kt-1p00_c2-0p00_skimmed",
    #  "GluGlutoHHto4B_spanet_kl-2p45_kt-1p00_c2-0p00_skimmed",
#     "GluGlutoHHto4B_spanet_kl-0p00_kt-0p00_c2-0p00_skimmed",
#     "GluGlutoHHto4B_spanet_kl-3p50_kt-1p00_c2-0p00_skimmed",
#     "GluGlutoHHto4B_spanet_kl-4p00_kt-1p00_c2-0p00_skimmed",
#     "GluGlutoHHto4B_spanet_kl-3p00_kt-1p00_c2-0p00_skimmed",
#     "GluGlutoHHto4B_spanet_kl-2p00_kt-1p00_c2-0p00_skimmed",
#     "GluGlutoHHto4B_spanet_kl-1p50_kt-1p00_c2-0p00_skimmed",
#     "GluGlutoHHto4B_spanet_kl-0p50_kt-1p00_c2-0p00_skimmed",
]
sample_list = [
    # "GluGlutoHHto4B_spanet_skimmed",
    # "GluGlutoHHto4B_spanet_skimmed_SM",
    # "GluGlutoHHto4B_spanet_skimmed",
    # "GluGlutoHHto4B",
    "GluGlutoHHto4B_spanet"
] + sample_ggF_list

# AKA if no model is applied
# print(onnx_model_dict)
if all([model == "" for model in onnx_model_dict.values()]):
    print("Didn't find any onnx model. Will choose region for SPANet training")
    categories_dict = define_single_category("4b_region")

# print("categories_dict", categories_dict)

# Define the columns to save
total_input_variables = {}
column_list = []
column_listRun2 = []
column_list = get_columns_list(SPANET_TRAINING_DEFAULT_COLUMNS_BTWP, not config_options_dict["save_chunk"])
if config_options_dict["random_pt"]:
    column_list += get_columns_list({"events": ["random_pt_weights"]})
        

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
            )
        else:
            bysample_bycategory_column_dict[sample]["bycategory"][category] = (
                column_list
            )
# print("bysample_bycategory_column_dict", bysample_bycategory_column_dict)

# Define the weights to apply
bysample_bycategory_weight_dict = {}

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet_redirector.json",
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b.json",
            f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed.json",
            f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed_SM.json",
            f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed_separateSamples.json",
            # f"{localdir}/../HH4b_common/datasets/QCD.json",
            # f"{localdir}/../HH4b_common/datasets/SPANet_classification.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_local.json",
            # f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b_local.json",
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
    weights_classes=common_weights
    + [bkg_morphing_dnn_weight, bkg_morphing_dnn_weightRun2],
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
