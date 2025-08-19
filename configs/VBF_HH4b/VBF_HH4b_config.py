import os
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
year = "2022_postEE"
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/jets_calibration_withoutVariations.yaml",
    update=True,
)


if config_options_dict["save_chunk"]:
    # workflow_options["dump_columns_as_arrays_per_chunk"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/training_samples/GluGlutoHHto4B_spanet_loose_03_17"
    pass

## Define the variables to save
# variables_dict = get_variables_dict(
#     CLASSIFICATION=CLASSIFICATION,
#     VBF_VARIABLES=False,
#     BKG_MORPHING=True if onnx_model_dict["BKG_MORPHING_DNN"] else False,
# )
variables_dict = {}

## Define the preselection to apply
preselection = (
    [
        (
            vbf_hh4b_presel
            if config_options_dict["tight_cuts"] is False
            else vbf_hh4b_presel_tight
        )
    ]
    if config_options_dict["vbf_presel"]
    else [
        (
            cuts.hh4b_presel
            if config_options_dict["tight_cuts"] is False
            else cuts.hh4b_presel_tight
        )
    ]
)

## Define the samples to process
sample_list = [
    # "DATA_JetMET_JMENano_C_skimmed",
    # "DATA_JetMET_JMENano_D_skimmed",
    # "DATA_JetMET_JMENano_E_skimmed",
    "DATA_JetMET_JMENano_F_skimmed",
    "DATA_JetMET_JMENano_G_skimmed",
] + (
    [
        # "GluGlutoHHto4B_spanet_skimmed",
        # "GluGlutoHHto4B",
        # "VBF_HHto4B",
    ]
    # if config_options_dict["sig_bkg_dnn"]
    # else []
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

print("categories_dict", categories_dict)

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
print(total_input_variables)

column_list = create_DNN_columns_list(
    False, not config_options_dict["save_chunk"], total_input_variables, btag=False
)
column_listRun2 = create_DNN_columns_list(
    True, not config_options_dict["save_chunk"], total_input_variables, btag=False
)

print(column_list)

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
print("bysample_bycategory_column_dict", bysample_bycategory_column_dict)

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

print("bysample_bycategory_weight_dict", bysample_bycategory_weight_dict)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            # f"{localdir}/../HH4b_common/datasets/QCD.json",
            f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_local.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_local_rucio.json",
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_SM_local_rucio_redirector.json",
            f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_test.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET_skimmed.json",
        ],
        "filter": {
            "samples": sample_list,
            "samples_exclude": [],
            # "year": [year],
        },
        "subsamples": {},
    },
    workflow=VBFHH4bProcessor,
    workflow_options=config_options_dict,
    skim=[
        get_HLTsel(primaryDatasets=["JetMET"]),
    ],
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
    variations={
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory": {},
            },
            "bysample": {},
        }
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
