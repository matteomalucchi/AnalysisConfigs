import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict  |= {
    # "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DHH_method_20_runs_postEE/best_models/ratio/average_model_from_onnx.onnx", # --> training on postEE
    # "bkg_morphing_spread_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DHH_method_20_runs_postEE/best_models/ratio/all_ratios_model_onnx.onnx", # --> training on postEE
    # "sig_bkg_dnn": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/DHH_method_norm_e5drop75_postEE/state_dict/model_best_epoch_18.onnx",
}


config_options_dict |= {
    "vbf_parton_matching": False,
    "tight_cuts": False,
    "save_chunk": False,
    "vbf_presel": False,
    "dnn_variables": True,
    "run2": True,
    "vr1": False,
    "random_pt": False,
    "rand_type": 0.3,
    "blind": True if onnx_model_dict["sig_bkg_dnn"] else False,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_dnn_input_variables,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_dnn_input_variables,
    "parton_jet_min_dR": 0.4,
    "max_num_jets_good": 5,
    "which_bquark": "last",
    "fifth_jet": "pt",
    "donotscale_sumgenweights": False,
    "pad_value": -999.0,
    "arctanh_delta_prob_bin_edge": 2.44,
    "arctanh_delta_prob_pad_limit": 2.,
    "add_jet_spanet": False,
    # VBF
    "vbf_parton_matching": False,
    "vbf_presel": False,
    "vbf_analysis": True,
    "which_vbf_quark":"with_mothers_children",
    "max_num_jets_add_vbf": 2,
}| onnx_model_dict
