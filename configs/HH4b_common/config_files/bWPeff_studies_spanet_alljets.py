import os

localdir = os.path.dirname(os.path.abspath(__file__))

import configs.HH4b_common.dnn_input_variables as dnn_vars

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict

onnx_model_dict["spanet"] = "/work/tharte/datasets/onnx_spanet_models_for_pairing_and_mass_sculpting_studies/spanet_hh4b_5jets_ptvary_loose_300_btag_wp_newLeptonVeto_L1Cut_UpdateJetVetoMap.onnx"

config_options_dict |= {
    "vbf_parton_matching": False,
    "tight_cuts": False,
    "save_chunk": False,
    "vbf_presel": False,
    "dnn_variables": False,
    "run2": False,
    "vr1": False,
    "random_pt": False,
    "rand_type": 0.3,
    "blind": False,
    "parton_jet_min_dR": 0.4,
    "max_num_jets_good": 5,
    "max_num_jets_spanet": 5,
    "max_num_jets_spanet_class": 4,
    "which_bquark": "last",
    "fifth_jet": "pt",
    "donotscale_sumgenweights": False,
    "pad_value": -999.0,
    "pad_value_spanet": 9999.0,
    "arctanh_delta_prob_bin_edge": 2.44,
    "add_jet_spanet": True,
    "spanet_input_name": dnn_vars.pairing_spanet_btagWP5,
    "only5jetsbSF": False,
} | onnx_model_dict
