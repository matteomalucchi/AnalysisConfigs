import os

import cloudpickle
from configs.HH4b_common.config_files.__config_file__ import (
    config_options_dict,
)
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
    get_L1sel,
    goldenJson,
    eventFlags,
    get_nPVgood,
)
from pocket_coffea.lib.calibrators.common.common import JetsCalibrator
from pocket_coffea.lib.weights.common.common import common_weights
from pocket_coffea.parameters import defaults
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.utils.configurator import Configurator

import configs.HH4b_common.custom_cuts_common as cuts
import utils_configs.quantile_transformer as quantile_transformer
from configs.HH4b_common.config_files.configurator_tools import (
    DEFAULT_JET_COLUMNS_DICT,
    SPANET_VBF_TRAINING_DEFAULT_COLUMNS_BTWP,
    SPANET_TRAINING_DEFAULT_COLUMNS_BTWP,
    DEFAULT_FATJET_COLUMNS,
    create_DNN_columns_list,
    define_categories,
    define_preselection,
    get_columns_list,
    get_variables_dict,
    with_fw_momenta_columns,
)
from configs.HH4b_common.custom_weights import (
    bkg_morphing_dnn_weight,
)
from configs.VBF_HH4b_boosted.workflow import VBFHH4bProcessor

BASELINE = False

localdir = os.path.dirname(os.path.abspath(__file__))


# Loading default parameters
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir)

# adding object preselection
# year = ["2022_postEE"]
year = ["2024"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/../HH4b_common/params/object_preselection_{config_options_dict['approach']}_approach.yaml",
    f"{localdir}/../HH4b_common/params/triggers_boosted.yaml",
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

print(config_options_dict)
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
    BOOSTED=config_options_dict["boosted"],
)
# variables_dict = {}

# Define the preselection to apply
preselection = define_preselection(config_options_dict)


# Define the samples to process
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
    # "GluGluHHto4B_Par-c2-0p00-kl-0p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia9",
    # "GluGluHHto4B_Par-c2-0p00-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
    # "GluGluHHto4B_Par-c2-0p00-kl-2p45-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
    # "GluGluHHto4B_Par-c2-0p00-kl-5p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
    # "GluGluHHto4B_Par-c2-0p10-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
    # "GluGluHHto4B_Par-c2-0p35-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
    # "GluGluHHto4B_Par-c2-1p00-kl-0p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
    # "GluGluHHto4B_Par-c2-2p24-kl-m20p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
    # "GluGluHHto4B_Par-c2-3p00-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
    # "GluGluHHto4B_Par-c2-m2p00-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
]

sample_VBF_list=[
    # "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4",
    # "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2",
    # "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3",
    # "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43",
    # "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94",
    # "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36",
    # "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39",
    # "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96",
    # "VBFHHto4B_CV_1_C2V_0_C3_1",
    # "VBFHHto4B_CV_1_C2V_1_C3_1",
    # "VBFHHto4B_Par-CV-1-C2V-0-C3-1_TuneCP5_13p6TeV_madgraph-pythia8",
    # "VBFHHto4B_Par-CV-1-C2V-1-C3-1_TuneCP5_13p6TeV_madgraph-pythia8",
    # "VBFHHto4B_Par-CV-1p74-C2V-1p37-C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8",
    # "VBFHHto4B_Par-CV-2p12-C2V-3p87-C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8",
    # "VBFHHto4B_Par-CV-m0p012-C2V-0p030-C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8",
    # "VBFHHto4B_Par-CV-m0p758-C2V-1p44-C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8",
    # "VBFHHto4B_Par-CV-m0p962-C2V-0p959-C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8",
    # "VBFHHto4B_Par-CV-m1p21-C2V-1p94-C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8",
    # "VBFHHto4B_Par-CV-m1p60-C2V-2p72-C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8",
    # "VBFHHto4B_Par-CV-m1p83-C2V-3p57-C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8",
]
sample_list = (
    [
        # 2022 preEE
        # "DATA_JetMET_JMENano_C_skimmed",
        # "DATA_JetMET_JMENano_D_skimmed",
        # 2022 postEE
        # "DATA_JetMET_JMENano_E",
        # "DATA_JetMET_JMENano_F",
        # "DATA_JetMET_JMENano_G",
        # "TTtoLNu2Q",
        # "TTto2L2Nu",
        # "TTto4Q",
        # "DATA_ParkingHH",
        "DATA_JetMET0_HH4bBoosted"
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

# # Define the categories to save
# categories_dict = define_categories(
#     bkg_morphing_dnn=config_options_dict["bkg_morphing_dnn"],
#     blind=config_options_dict["blind"],
#     spanet=config_options_dict["spanet"],
#     run2=config_options_dict["run2"],
#     vr1=config_options_dict["vr1"],
#     boosted=config_options_dict["boosted"],
#     other_group=True if config_options_dict["approach"] == "boosted" else False,
#     split_qcd=config_options_dict["split_qcd"] if config_options_dict["boosted"] else False,
#     # vbf_analysis=config_options_dict["vbf_analysis"],
#     vbf_analysis=config_options_dict["vbf_selection"] if "vbf_selection" in config_options_dict.keys() else config_options_dict["vbf_analysis"],
#     vbf_discriminator=config_options_dict["vbf_discriminator"],
#     ggf_vbf_threshold=config_options_dict["ggf_vbf_threshold"],
# )
# 
# if BASELINE:
#     categories_dict = {"baseline": [passthrough]}
# 
# column_list=[]
# 
# # Add SPANet training inputs
# if not config_options_dict["spanet"] and not config_options_dict["run2"] and not config_options_dict["boosted"]:
#     print("somehow we arrived at a wrong point")
#     if not config_options_dict["vbf_analysis"]:
#         column_list += get_columns_list(SPANET_TRAINING_DEFAULT_COLUMNS_BTWP, not config_options_dict["save_chunk"])
#     else:
#         column_list += get_columns_list(
#             with_fw_momenta_columns(
#                 SPANET_VBF_TRAINING_DEFAULT_COLUMNS_BTWP,
#                 config_options_dict["max_order_FW"],
#                 config_options_dict["FW_momenta_norms"],
#             ),
#             not config_options_dict["save_chunk"],
#         )
# else:
#     # Define the other columns to save
#     total_input_columns = {}
# 
#     if config_options_dict["spanet"]:
#         total_input_columns |= {
#             "Delta_pairing_probabilities": ["events", "Delta_pairing_probabilities"],
#             "Arctanh_Delta_pairing_probabilities": [
#                 "events",
#                 "Arctanh_Delta_pairing_probabilities",
#             ],
#             "Binned_Arctanh_Delta_pairing_probabilities": [
#                 "events",
#                 "Binned_Arctanh_Delta_pairing_probabilities",
#             ],
#             "Padded_Arctanh_Delta_pairing_probabilities": [
#                 "events",
#                 "Padded_Arctanh_Delta_pairing_probabilities",
#             ],
#         }
# 
#     if config_options_dict["dnn_variables"]:
#         # Be aware, that for boosted, you need a boosted sig/bkg and morphing
#         total_input_columns |= (
#             config_options_dict["sig_bkg_dnn_input_variables"]
#             | config_options_dict["bkg_morphing_dnn_input_variables"]
#             | {"year": ["events", "year"],
#                "vbf_jet_prov": ["JetGoodVBF", "provenance"],
#                "vbf_cand_jet_prov": ["JetGoodVBFCandidates", "provenance"],
#                "Higgs_leading_btag": ["HiggsLeading", "btagBB"],
#                "Higgs_subleading_btag": ["HiggsSubLeading", "btagBB"],
#               }
#         )
#         column_list += get_columns_list({})
#     elif config_options_dict["boosted"]:
#         column_list += get_columns_list(DEFAULT_FATJET_COLUMNS, not config_options_dict["save_chunk"])
# 
#     else:
#         total_input_columns |= DEFAULT_JET_COLUMNS_DICT
# 
#     column_list += create_DNN_columns_list(
#         False, not config_options_dict["save_chunk"], total_input_columns, btag=False
#     )
#     # Add special columns
#     if config_options_dict["sig_bkg_dnn"]:
#         column_list += get_columns_list({"events": ["sig_bkg_dnn_score"]})
# 
# bysample_bycategory_column_dict = {}
# for sample in sample_list:
#     bysample_bycategory_column_dict[sample] = {
#         "inclusive": [],
#         "bycategory": {},
#     }
#     for category in categories_dict.keys():
#         bysample_bycategory_column_dict[sample]["bycategory"][category] = (
#             column_list
#             + (
#                 get_columns_list({"events": ["bkg_morphing_spread_dnn_weights"]})
#                 if "DATA" in sample
#                 and config_options_dict["bkg_morphing_spread_dnn"]
#                 and "postW" in category
#                 else []
#             )
#         )
# 
# # Define the weights to apply
# bysample_bycategory_weight_dict = {}
# for sample in sample_list:
#     if "DATA" in sample:
#         bysample_bycategory_weight_dict[sample] = {"inclusive": [], "bycategory": {}}
#         for category in categories_dict.keys():
#             if "postW" in category:
#                 bysample_bycategory_weight_dict[sample]["bycategory"][category] = [
#                     "bkg_morphing_dnn_weight"
#                 ]
column_list = get_columns_list(
    {
        "JetGood": ["pt_regressed", "pt_default", "pt", "eta", "phi", "mass"],
        "JetGoodVBF": ["pt", "eta", "phi", "mass"],
        "Jet": ["pt_regressed", "pt_default", "eta"],
        "JetGoodVBFEnergyOrdered": ["pt", "eta", "phi", "mass"],
        "events": ["HT_jetJetGoodVBF", "HT_jetJetGoodVBF", "nJetGood", "nFatJetGoodSelected", "event", "boosted_bdt_score", "boosted_bdt_vbf_score", "mjjJetGoodVBF", "mjjJetGood", "detaJetGoodVBF", "HiggsLeadingByHiggsSubLeadingPt"],
        "JetGoodVBFNearHiggsLeading": ["pt", "eta", "phi", "mass"],
        "JetGoodVBFNearHiggsSubLeading": ["pt", "eta", "phi", "mass"],
        "FatJetGoodSelected": ["pt", "eta", "phi", "msoftdrop", "mass_regr", "mass", "btagBBTXbb"],
        "HiggsLeading": ["pt", "eta", "phi", "msoftdrop", "mass_regr", "mass", "btagBBTXbb", "btagBBTXbb_dig", "Tau3OverTau2", "dRclosestVBF", "massclosestVBF", "divHHmass"],
        "HiggsSubLeading": ["pt", "eta", "phi", "msoftdrop", "mass_regr", "mass", "btagBBTXbb", "Tau3OverTau2", "dRclosestVBF", "massclosestVBF", "divHHmass"],
        "HH": ["pt", "eta", "mass"],
        "PuppiMET": ["pt"],
        "PFMET": ["pt"],
    }
)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            # f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_JetMET_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_JetMET.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH.json",
            # f"{localdir}/../HH4b_common/datasets/background_TTtoX.json",
            # f"{localdir}/../HH4b_common/datasets/signal_VBFHHto4B_Par_2024.json",
            # f"{localdir}/../HH4b_common/datasets/signal_GluGluHHto4B_Par_2024.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET_boosted_test_24EraD.json",
        ],
        "filter": {
            "samples": sample_list,
            "samples_exclude": [],
            "year": year,
        },
        "subsamples": {},
    },
    workflow=VBFHH4bProcessor,
    workflow_options=config_options_dict,
    skim=[],  # cuts.skimming_cut_list(config_options_dict),
    preselections=[],  # preselection,
    categories={
        "MET_filter": [eventFlags],
        "Jet_Veto_map": [eventFlags, cuts.hh4b_JetVetoMap],
        "2AK8Jetsonly": [cuts.hh4b_boosted_2fatjets],
        "2AK8Jets": [eventFlags, cuts.hh4b_JetVetoMap, cuts.hh4b_boosted_2fatjets],
        "0lep": [eventFlags, cuts.hh4b_JetVetoMap, cuts.hh4b_boosted_2fatjets, cuts.hh4b_boosted_lepton_veto],
        "ak8bbcut": [eventFlags, cuts.hh4b_JetVetoMap, cuts.hh4b_boosted_2fatjets, cuts.hh4b_boosted_lepton_veto, cuts.hh4b_boosted_signal_region_other_group],
        },
    weights_classes=common_weights
    + [bkg_morphing_dnn_weight],
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
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
        #         # "inclusive": ["jet_calibration_with_pt_regression_legacy"],
        #         # "inclusive": ["jet_calibration"]
        #         },
        #     }
    },
    variables={},
    columns={
        "common": {
            "inclusive": [],
            "bycategory": {
                "ak8bbcut": column_list,
                },
        },
        "bysample": {},
    },
)
cloudpickle.register_pickle_by_value(quantile_transformer)
