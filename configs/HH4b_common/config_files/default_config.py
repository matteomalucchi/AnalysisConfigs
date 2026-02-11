from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_input_variables,
    sig_bkg_dnn_input_variables,
    pairing_spanet_btag
)

default_onnx_model_dict = {
    "spanet": "",
    "vbf_discriminator": "",
    "bkg_morphing_dnn": "",
    "sig_bkg_dnn": "",
    "bkg_morphing_spread_dnn": "",
}

default_config_options_dict = {
    "tight_cuts": False,
    "save_chunk": False,
    # VBF
    "vbf_parton_matching": False,
    "vbf_presel": False,
    "vbf_analysis": False,
    "which_vbf_quark": "with_status",  # "with_mothers_children"
    "max_num_jets_add_vbf": 2,
    #
    "dnn_variables": True,
    "run2": False,
    "vr1": False,
    "random_pt": False,
    "rand_type": 0.3,
    "blind": False,
    "sig_bkg_dnn_input_variables": sig_bkg_dnn_input_variables,
    "bkg_morphing_dnn_input_variables": bkg_morphing_dnn_input_variables,
    "parton_jet_min_dR": 0.4,
    "max_num_jets_good": 5,
    "max_num_jets_spanet": 5,
    "max_num_jets_spanet_class": 4,
    "which_bquark": "last",
    "fifth_jet": "pt",
    "donotscale_sumgenweights": False,
    "pad_value": -999.0,
    "arctanh_delta_prob_bin_edge": 2.44,
    "arctanh_delta_prob_pad_limit": 2.0,
    "add_jet_spanet": False,
    "spanet_input_name": pairing_spanet_btag,
    "old_wp_def": False,
    "qt_postEE": None,
    "qt_preEE": None,
    "only5jetsbSF": False,
    "noL1": False,
} | default_onnx_model_dict
