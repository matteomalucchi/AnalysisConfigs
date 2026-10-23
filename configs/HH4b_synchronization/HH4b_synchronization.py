import os
import cloudpickle
import utils_configs.quantile_transformer as quantile_transformer

from configs.HH4b_synchronization.config_options import (
    config_options_dict,
    onnx_model_dict,
)
from pocket_coffea.lib.calibrators.common.common import JetsCalibrator


from pocket_coffea.lib.weights.common.common import common_weights

from pocket_coffea.parameters.cuts import passthrough

# rom pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters import defaults
from pocket_coffea.parameters.histograms import *

# from collections import defaultdict
from pocket_coffea.utils.configurator import Configurator
from configs.HH4b_synchronization.workflow import SyncHH4bProcessor

import configs.HH4b_synchronization.custom_cuts as sync_cuts
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
year = ["2024"]  # , "2023_preBPix", "2023_postBPix"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    # object preselection of the synchronization (second approach with lower jet pt cuts)
    f"{localdir}/params/object_preselection_synchronization.yaml",
    f"{localdir}/../HH4b_common/params/triggers.yaml",
    f"{localdir}/../HH4b_common/params/variations.yaml",
    f"{localdir}/../HH4b_common/params/btagging_multipleWP.yaml",
    f"{localdir}/../HH4b_common/params/btagging_sampleGroups.yaml",
    f"{localdir}/params/jets_calibration_regression_json_synchronization.yaml",
    update=True,
)



if config_options_dict["save_chunk"]:
    config_options_dict["dump_columns_as_arrays_per_chunk"] = config_options_dict[
        "save_chunk"
    ]


variables_dict = {}

## Define the preselection to apply
# synchronization: only >= 4 jets of the JetCuts collection (`JetCuts` object
# preselection, built by the synchronization workflow) with |eta| < 2.5;
# nothing else (no pt thresholds on the leading jets, b-tag requirements,
# lepton veto or jet veto map)
preselection = [sync_cuts.hh4b_4jets_presel]

sample_list = ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_synchronization"]


# Define the categories to save
categories_dict = {"baseline": [passthrough]}

total_input_variables = {}


# Synchronization "Events" tree: run/lumi/event, generator weight, trigger and
# L1 bits, noise filters, jet veto maps, MET (NanoAOD PuppiMET, not recomputed
# with the corrected jets), true pileup, the AK4 jets with pt > 15 GeV, tight ID
# and lepton cleaning, and the indices (the `index` field of the jets) of the 4
# jets of the HH pairing
sync_l1 = [
    "HTT280er",
    "HTT280er_QuadJet_70_55_40_35_er2p5",
    "HTT320er",
    "HTT320er_QuadJet_70_55_40_40_er2p5",
    "HTT320er_QuadJet_80_60_er2p1_45_40_er2p3",
    "HTT320er_QuadJet_80_60_er2p1_50_45_er2p3",
    "HTT360er",
    "Mu6_HTT240er",
]
sync_triggers = sorted(
    {t for y in year for trgs in parameters.HLT_triggers[y].values() for t in trgs}
)
sync_flags = sorted(
    {
        f
        for y in year
        for f in list(parameters.event_flags[y]) + list(parameters.event_flags_data[y])
    }
)
# NOTE: the regression branches are the UParT ones (2024-2025)
SYNC_JET_COLUMNS = [
    "index",
    "pt",
    "pt_default",
    "pt_regressed",
    "pt_raw_default",
    "mass",
    "mass_raw_default",
    "eta",
    "phi",
    "btagB",
    "jec_factor_default",
    "jer_factor_default",
    "jec_factor",
    "jer_factor",
    "UParTAK4RegPtRawCorr",
    "UParTAK4RegPtRawCorrNeutrino",
    # resolution of the regressed pt
    "pt_raw_res",
    # b-tag: quark vs gluon score and number of WPs passed (0-5: L, M,
    # T, XT, XXT of the year's algorithm)
    "btagQvG",
    "btagB_5wp",
    # jet ID inputs and other NanoAOD variables
    "area",
    "nConstituents",
    "nSVs",
    "chHEF",
    "neHEF",
    "chEmEF",
    "neEmEF",
    "muEF",
    "hfEmEF",
    "hfHEF",
    "chMultiplicity",
    "neMultiplicity",
    # per-jet jet veto map flag and jets given to the D_HH pairing
    "passes_jetvetomap",
    "used_in_dHH",
    # MC truth: flavour and Higgs provenance (gen matching)
    "hadronFlavour",
    "partonFlavour",
    "provenance_higgs",
]
column_list = get_columns_list(
    {
        "HLT": sync_triggers,
        "L1": sync_l1,
        "Flag": sync_flags,
        "PuppiMET": ["pt", "phi"],
        "Pileup": ["nTrueInt"],
        "events": [
            "run",
            "luminosityBlock",
            "genWeight",
            "passJetVetoMap",
            "ak4_h1b1_index",
            "ak4_h1b2_index",
            "ak4_h2b1_index",
            "ak4_h2b2_index",
        ],
        # saved jets, ordered by pt: JetSync (pt > 15 GeV, |eta| < 4.7, tight ID) and
        # JetCuts (the jets of the preselection: |eta| < 2.5, tight ID, no pt cut)
        "JetSync": SYNC_JET_COLUMNS,
        "JetCuts": SYNC_JET_COLUMNS,
    }
)
column_list += get_columns_list(
    {
        "events": [
            "weight_single_lumi",
            "weight_single_XS",
            "weight_single_pileup",
            # "weight_single_sf_btag_fixed_multiple_wp",
        ]
    }
)
# skimming_cut_list = configs.HH4b_common.custom_cuts_common.skimming_cut_list
skimming_cut_list = []

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/signal_ggF_HH4b_synchronization.json",
        ],
        "filter": {
            "samples": sample_list,
            "samples_exclude": [],
            "year": year,
        },
        "subsamples": {},
    },
    workflow=SyncHH4bProcessor,
    workflow_options=config_options_dict,
    skim=skimming_cut_list,
    preselections=preselection,
    categories=categories_dict,
    weights_classes=common_weights,  # + [bkg_morphing_dnn_weight, SF_btag_fixed_multiple_wp],
    calibrators=[JetsCalibrator] if not config_options_dict["mixeddata"] else [],
    weights={
        "common": {
            # "inclusive": ["genWeight", "lumi", "XS", "pileup", "sf_btag_fixed_multiple_wp"],
            "inclusive": ["genWeight", "lumi", "XS", "pileup"],
            "bycategory": {},
        },
        # "bysample": bysample_bycategory_weight_dict,
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
        },
    },
    variables={},  # variables_dict,
    columns={
        "common": {
            "inclusive": column_list,
            "bycategory": {},
        },
    },
)
