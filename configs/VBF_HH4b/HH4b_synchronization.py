import os
import cloudpickle
import utils_configs.quantile_transformer as quantile_transformer

from configs.HH4b_common.config_files.__config_file__ import (
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
from configs.VBF_HH4b.workflow_synchronization import SyncHH4bProcessor

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
year = ["2024"]  # , "2023_preBPix", "2023_postBPix"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    # object preselection of the synchronization (second approach with lower jet pt cuts)
    f"{localdir}/../HH4b_common/params/object_preselection_synchronization.yaml",
    f"{localdir}/../HH4b_common/params/triggers.yaml",
    f"{localdir}/../HH4b_common/params/variations.yaml",
    f"{localdir}/../HH4b_common/params/btagging_multipleWP.yaml",
    f"{localdir}/../HH4b_common/params/btagging_sampleGroups.yaml",
    f"{localdir}/../HH4b_common/params/jets_calibration_regression_json.yaml",
    update=True,
)

# no pt > 50 GeV cut for the jets in 2.5 < |eta| < 3.0
parameters.jets_calibration.forward_jets_mitigation.pt_cut_HE.apply = False


if config_options_dict["save_chunk"]:
    config_options_dict["dump_columns_as_arrays_per_chunk"] = config_options_dict[
        "save_chunk"
    ]


# score transform still in testing. So far hardcoded to be 2022_postEE...
variables_dict = {}

## Define the preselection to apply
# synchronization: only >= 4 jets of the full Jet collection with |eta| < 2.5,
# nothing else (no pt thresholds, jet ID, b-tag requirements, lepton veto or
# jet veto map)
preselection = [cuts.hh4b_4jets_presel]

sample_list = ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_synchronization"]


# Define the categories to save
categories_dict = {"baseline": [passthrough]}

total_input_variables = {}
column_list = get_columns_list(
    {
        "JetGood": [
            "provenance",
            "pt",
            "eta",
            "phi",
            "mass",
            "btagB_3wp",
            "btagB_5wp",
            "btagB",
        ],
    }
)
# Synchronization "Events" tree: trigger bits, noise filters, jet veto maps, the
# AK4 jets with pt > 15 GeV and tight ID, and the indices (the `index` field of
# the jets) of the 4 jets of the HH pairing
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
column_list += get_columns_list(
    {
        "HLT": sync_triggers,
        "Flag": sync_flags,
        "events": [
            "passJetVetoMap",
            "ak4_h1b1_index",
            "ak4_h1b2_index",
            "ak4_h2b1_index",
            "ak4_h2b2_index",
        ],
        # NOTE: the regression branches are the UParT ones (2024-2025)
        "JetSync": [
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
        ],
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
# skimming_cut_list = cuts.skimming_cut_list
skimming_cut_list = []

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_synchronization.json",
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
