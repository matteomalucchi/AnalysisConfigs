from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_input_variables,
    sig_bkg_dnn_input_variables,
)

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict  |= {
    "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DHH_method_20_runs_postEE/best_models/ratio/average_model_from_onnx.onnx", # --> training on postEE
    # "bkg_morphing_spread_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DHH_method_20_runs_postEE/best_models/ratio/all_ratios_model_onnx.onnx", # --> training on postEE
    "sig_bkg_dnn": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/DHH_method_norm_e5drop75_postEE/state_dict/model_best_epoch_18.onnx",
}


config_options_dict |= {
    "higgs_parton_matching": False,
    "vbf_parton_matching": False,
    "tight_cuts": False,
    "classification": False,
    "save_chunk": False,
    "vbf_presel": False,
    "semi_tight_vbf": True,
    "dnn_variables": True,
    "run2": True,
    "vr1": False,
    "random_pt": False,
    "rand_type": 0.3,
    "blind": True if onnx_model_dict["sig_bkg_dnn"] else False,
    "sig_bkg_dnn_input_variables": sig_bkg_dnn_input_variables,
    "bkg_morphing_dnn_input_variables": bkg_morphing_dnn_input_variables,
    "parton_jet_min_dR": 0.4,
    "max_num_jets": 5,
    "which_bquark": "last",
    "fifth_jet": "pt",
    "donotscale_sumgenweights": False,
    "pad_value": -999.0,
    "arctanh_delta_prob_bin_edge": 2.44,
    "arctanh_delta_prob_pad_limit": 2.,
    "add_jet_spanet": False,
    "spanet_input_name_list": ["log_pt", "eta", "phi", "btag"],
    "qt_postEE": "/work/tharte/datasets/quantile_transformer/DHH_quantiles/SRRun2_qt/qt_events_sig_bkg_dnn_score_kl_1.00.pkl",
}| onnx_model_dict
