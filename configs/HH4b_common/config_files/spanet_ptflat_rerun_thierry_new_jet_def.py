"""Config file for spanet_pT model with the correct definition for the 5th jet."""
import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict |= {
    "spanet": "/work/tharte/datasets/onnx_spanet_models_for_pairing_and_mass_sculpting_studies/hh4b_5jets_e300_s100_ptvary_wide_loose_btag.onnx",
    "vbf_discriminator": "",
    "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/SPANET_ptFlat_20_runs_postEE_new_jet5_def/best_models/ratio/average_model_from_onnx.onnx",  # --> trained on postEE only
    "sig_bkg_dnn": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/DNN_spanet_ptflat_e5drop75_postEE_allklambda_new_jet_def/state_dict/model_best_epoch_21.onnx",
}


config_options_dict |= {
    "vbf_parton_matching": False,
    "tight_cuts": False,
    "save_chunk": False,
    "vbf_presel": False,
    "dnn_variables": True,
    "run2": False,
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
    "add_jet_spanet": True,
    "spanet_input_name": dnn_vars.pairing_spanet_btag,
} | onnx_model_dict
