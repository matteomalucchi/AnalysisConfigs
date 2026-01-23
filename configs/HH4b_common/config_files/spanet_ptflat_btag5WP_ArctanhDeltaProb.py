from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_ArctanhDeltaProb_input_variables,
    sig_bkg_dnn_ArctanhDeltaProb_input_variables,
)

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict |= {
    "spanet": "/work/tharte/datasets/onnx_spanet_models_for_pairing_and_mass_sculpting_studies/spanet_hh4b_5jets_ptvary_loose_300_btag_5wp_s100_oldWPdef_start_n1.onnx",  # spanet pt vary 0.3, 1.7, btag 5 WP
    "bkg_morphing_dnn": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_btag5WP_postEE_ArctanhDeltaProb/best_models/average_model_from_onnx.onnx",  # btag_5WP, arctanh PD, only 2022_postEE, 20 k-folds, early stopping, 1e-5 minDelta, spanet pt vary 
    "sig_bkg_dnn": "/work/mmalucch/out_ML_pytorch/SvB_DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_btag5WP_postEE_ArctanhDeltaProb/run100/state_dict/model_best_epoch_5.onnx",
}


config_options_dict |= {
    "higgs_parton_matching": False,
    "vbf_parton_matching": False,
    "tight_cuts": False,
    "classification": False,
    "save_chunk": False,
    "vbf_presel": False,
    "dnn_variables": True,
    "run2": False,
    "vr1": False,
    "random_pt": False,
    "rand_type": 0.3,
    "blind": False,
    "bkg_morphing_dnn_input_variables": bkg_morphing_dnn_ArctanhDeltaProb_input_variables,
    "sig_bkg_dnn_input_variables": sig_bkg_dnn_ArctanhDeltaProb_input_variables,
    "parton_jet_min_dR": 0.4,
    "max_num_jets": 5,
    "which_bquark": "last",
    "fifth_jet": "pt",
    "donotscale_sumgenweights": False,
    "pad_value": -999.0,
    "arctanh_delta_prob_bin_edge": 2.44,
    "arctanh_delta_prob_pad_limit": 2.0,
    "add_jet_spanet": True,
    "spanet_input_name_list": ["log_pt", "eta", "phi", "btagPNetB_5wp"],
    "old_wp_def": True,
} | onnx_model_dict
