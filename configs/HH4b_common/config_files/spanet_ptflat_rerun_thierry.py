from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_input_variables,
    sig_bkg_dnn_input_variables,
)

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict  |= {
    "spanet": "/work/tharte/datasets/onnx_spanet_models_for_pairing_and_mass_sculpting_studies/hh4b_5jets_e300_s100_ptvary_wide_loose_btag.onnx",
    "vbf_ggf_dnn": "",
    "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/SPANET_ptFlat_20_runs_postEE/best_models/ratio/average_model_from_onnx.onnx", # --> trained on postEE only
    #"sig_bkg_dnn": "",
    "sig_bkg_dnn": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/SPANET_ptflat_norm_e5drop75_postEE/state_dict/model_best_epoch_23.onnx",
}


config_options_dict |= {
    "higgs_parton_matching": False,
    "vbf_parton_matching": False,
    "tight_cuts": False,
    "classification": False,
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/spanet_ptflat_rerun_thierry_histtest",
    "vbf_presel": False,
    "dnn_variables": True,
    "run2": False,
    "vr1": False,
    "random_pt": False,
    "rand_type": 0.3,
    # "blind": True if onnx_model_dict["sig_bkg_dnn"] else False,
    "blind": False,
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
    "qt_postEE": "/work/tharte/datasets/quantile_transformer/qt_events_sig_bkg_dnn_score_kl_1.00_postEE_21bins.pkl",
    "qt_preEE": None
} | onnx_model_dict
