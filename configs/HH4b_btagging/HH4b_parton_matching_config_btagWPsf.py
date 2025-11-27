import os

import cloudpickle
from configs.HH4b_common.config_files.__config_file__ import (
    config_options_dict,
    onnx_model_dict,
)
from pocket_coffea.lib.calibrators.common import default_calibrators_sequence
from pocket_coffea.lib.calibrators.legacy.legacy_calibrators import (
    JetsCalibrator,
    JetsPtRegressionCalibrator,
)
from pocket_coffea.lib.weights.common.common import common_weights

# from pocket_coffea.parameters.cuts import passthrough
# rom pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters import defaults
from pocket_coffea.parameters.histograms import *
from pocket_coffea.parameters.histograms import Axis, HistConf

# from collections import defaultdict
from pocket_coffea.utils.configurator import Configurator
from workflow_btagSF_HH4b import HH4bbtagWPefficiencyProcessor

import configs.HH4b_common.custom_cuts_common as cuts
import utils.quantile_transformer as quantile_transformer
from configs.HH4b_common.config_files.configurator_tools import (
    SPANET_TRAINING_DEFAULT_COLUMNS,
    SPANET_TRAINING_DEFAULT_COLUMNS_BTWP,
    define_single_category,
    get_columns_list,
    define_preselection,
)
from configs.HH4b_common.custom_weights import (
    bkg_morphing_dnn_weight,
    bkg_morphing_dnn_weightRun2,
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
    f"{localdir}/../HH4b_common/params/object_preselection.yaml",
    f"{localdir}/../HH4b_common/params/triggers.yaml",
    f"{localdir}/../HH4b_common/params/variations.yaml",
    f"{localdir}/../HH4b_common/params/btagging_multipleWP.yaml",
    f"{localdir}/../HH4b_common/params/btagging_sampleGroups.yaml",
    f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_withoutVariations_withJERC.yaml",
    # f"{localdir}/../HH4b_common/params/jets_calibration_legacy_Calibrator_withVariations.yaml",
    update=True,
)

## Define the preselection to apply
preselection = define_preselection(config_options_dict | {"no_btag":True})

# Defining the used samples
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
] + sample_ggF_list

categories_dict = define_single_category("inclusive")
categories_dict |= define_single_category("inclusive_sf_btag")

# Define the columns to save
total_input_variables = {}
column_list = []
column_listRun2 = []

assert not (config_options_dict["random_pt"] and config_options_dict["run2"])
if all([model == "" for model in onnx_model_dict.values()]):
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

# print("bysample_bycategory_weight_dict", bysample_bycategory_weight_dict)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet_redirector.json",
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b.json",
            f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed.json",
            f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed_SM.json",
            f"{localdir}/../HH4b_common/datasets/GluGlutoHHto4B_spanet_skimmed_separateSamples.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET_skimmed.json",
            f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET.json",
        ],
        "filter": {
            "samples": sample_list,
            "samples_exclude": [],
            "year": year,
        },
        "subsamples": {},
    },
    workflow=HH4bbtagWPefficiencyProcessor,
    workflow_options=config_options_dict,
    skim=cuts.skimming_cut_list,
    preselections=preselection,
    categories=categories_dict,
    weights_classes=common_weights
    + [bkg_morphing_dnn_weight, bkg_morphing_dnn_weightRun2, SF_btag_fixed_multiple_wp],
    # calibrators=default_calibrators_sequence,
    calibrators=[JetsCalibrator, JetsPtRegressionCalibrator],
    weights={
        "common": {
            "inclusive": ["genWeight", "lumi", "XS", "pileup"],
            # "inclusive": ["genWeight", "lumi", "XS", "pileup"],
            # "inclusive": [],
            "bycategory": {"inclusive_sf_btag": ["sf_btag_fixed_multiple_wp"]
            },
        },
        "bysample": bysample_bycategory_weight_dict,
    },
    variations={
        "weights": {
            "common": {
                "inclusive": ["pileup"],
                # "inclusive": [],
            "bycategory": {"inclusive_sf_btag": ["sf_btag_fixed_multiple_wp"]
                }
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
    variables={
        # **count_hist(name="nJets", coll="JetGood",bins=10, start=2, stop=12),
        # **count_hist(name="nBJets", coll="BJetGood",bins=14, start=0, stop=14),
        # **count_hist(name="nLeptons", coll="LeptonGood",bins=3, start=0, stop=3),
        **{
            f"jet_btagPNetB_{i}": HistConf(
                [
                    Axis(
                        coll="JetGood",
                        field="btagPNetB",
                        bins=20,
                        start=0,
                        stop=1,
                        label=f"jet_btagPNetB_{i}",
                        pos=i
                    ),
                ]
            )
            for i in range(5)
        },
        "nJets_vs_HT": HistConf(
            [
                Axis(
                    coll="events",
                    field="nJetGood",
                    bins=[4, 5, 6, 7, 8, 9, 10],
                    label="nJets",
                ),
                Axis(
                    coll="events",
                    field="HT",
                    bins=20,
                    start=0,
                    stop=2000,
                    label="events_HT",
                ),
            ]
        ),
        ** jet_hists(name="jet", coll="JetGood", pos=0),
        **jet_hists(name="jet", coll="JetGood", pos=1),
        **jet_hists(name="jet", coll="JetGood", pos=2),
        **jet_hists(name="jet", coll="JetGood", pos=3),
        **jet_hists(name="jet", coll="JetGood", pos=4),
        # **jet_hists(name="bjet",coll="BJetGood", pos=0),
        # **jet_hists(name="bjet",coll="BJetGood", pos=1),
        # **jet_hists(name="bjet",coll="BJetGood", pos=2),
        # **jet_hists(name="bjet",coll="BJetGood", pos=3),
        # **jet_hists(name="bjet",coll="BJetGood", pos=4),
        **met_hists(coll="PuppiMET"),
    },
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
