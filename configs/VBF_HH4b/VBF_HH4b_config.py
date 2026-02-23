import os

import cloudpickle
from configs.HH4b_common.config_files.__config_file__ import (
    config_options_dict,
)
from pocket_coffea.lib.calibrators.common.common import JetsCalibrator
from pocket_coffea.lib.weights.common.common import common_weights
from pocket_coffea.parameters import defaults
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.utils.configurator import Configurator

import configs.HH4b_common.custom_cuts_common as cuts
import utils.quantile_transformer as quantile_transformer
from configs.HH4b_common.config_files.configurator_tools import (
    DEFAULT_JET_COLUMNS_DICT,
    SPANET_VBF_TRAINING_DEFAULT_COLUMNS_BTWP,
    SPANET_TRAINING_DEFAULT_COLUMNS_BTWP,
    create_DNN_columns_list,
    define_categories,
    define_preselection,
    get_columns_list,
    get_variables_dict,
)
from configs.HH4b_common.custom_weights import (
    bkg_morphing_dnn_weight,
    bkg_morphing_dnn_weightRun2,
)
from configs.VBF_HH4b.workflow import VBFHH4bProcessor

BASELINE = False


localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir)

# adding object preselection
year = ["2022_postEE"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/../HH4b_common/params/object_preselection.yaml",
    f"{localdir}/../HH4b_common/params/triggers.yaml",
    f"{localdir}/../HH4b_common/params/variations.yaml",
    f"{localdir}/../HH4b_common/params/btagging_multipleWP.yaml",
    f"{localdir}/../HH4b_common/params/btagging_sampleGroups.yaml",
    # f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_withVariations.yaml",
    # f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_withoutVariations_withJERC.yaml",
    # f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_onlyJEC.yaml",
    f"{localdir}/../HH4b_common/params/jets_calibration_regression_json.yaml",
    # f"{localdir}/../HH4b_common/params/jets_calibration_regression_json_onlyJEC.yaml",
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

# Define the preselection to apply
preselection = define_preselection(config_options_dict)


# Define the samples to process
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

sample_VBF_list=[
    "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4",
    "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2",
    "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3",
    "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43",
    "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94",
    "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36",
    "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39",
    "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96",
    "VBFHHto4B_CV_1_C2V_0_C3_1",
    "VBFHHto4B_CV_1_C2V_1_C3_1",
]
sample_list = (
    [
        # 2022 preEE
        # "DATA_JetMET_JMENano_C_skimmed",
        # "DATA_JetMET_JMENano_D_skimmed",
        # 2022 postEE
        # "DATA_JetMET_JMENano_E_skimmed",
        # "DATA_JetMET_JMENano_F_skimmed",
        # "DATA_JetMET_JMENano_G_skimmed",
    ]
    + sample_ggF_list
    + sample_VBF_list
    + (
        [
        #     "GluGlutoHHto4B_spanet_skimmed",
        #     # "GluGlutoHHto4B",
        # "GluGlutoHHto4B_spanet"
        ]
    )
)


# Define the categories to save
categories_dict = define_categories(
    bkg_morphing_dnn=config_options_dict["bkg_morphing_dnn"],
    blind=config_options_dict["blind"],
    spanet=config_options_dict["spanet"],
    run2=config_options_dict["run2"],
    vr1=config_options_dict["vr1"],
    vbf_analysis=config_options_dict["vbf_analysis"],
    vbf_discriminator=config_options_dict["vbf_discriminator"],
)

if BASELINE:
    categories_dict = {"baseline": [passthrough]}

column_list=[]
column_listRun2=[]

# Add SPANet training inputs
if not config_options_dict["spanet"] and not config_options_dict["run2"]:
    if not config_options_dict["vbf_analysis"]:
        column_list += get_columns_list(SPANET_TRAINING_DEFAULT_COLUMNS_BTWP, not config_options_dict["save_chunk"])
    else:
        column_list += get_columns_list(SPANET_VBF_TRAINING_DEFAULT_COLUMNS_BTWP, not config_options_dict["save_chunk"])
    
else:
    # Define the other columns to save
    total_input_columns = {}

    if config_options_dict["spanet"]:
        total_input_columns |= {
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
        total_input_columns |= (
            config_options_dict["sig_bkg_dnn_input_variables"]
            | config_options_dict["bkg_morphing_dnn_input_variables"]
            | {"year": ["events", "year"]}
        )
    else:
        total_input_columns |= DEFAULT_JET_COLUMNS_DICT
    if BASELINE:
        total_input_columns |= DEFAULT_JET_COLUMNS_DICT

    column_list += create_DNN_columns_list(
        False, not config_options_dict["save_chunk"], total_input_columns, btag=False
    )
    column_listRun2 += create_DNN_columns_list(
        True, not config_options_dict["save_chunk"], total_input_columns, btag=False
    )
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

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b_pnfs_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b.json",
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
    skim=cuts.skimming_cut_list(config_options_dict),
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
    calibrators=[JetsCalibrator],
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
        #         # "inclusive": ["jet_calibration_with_pt_regression_legacy"],
        #         # "inclusive": ["jet_calibration"]
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
