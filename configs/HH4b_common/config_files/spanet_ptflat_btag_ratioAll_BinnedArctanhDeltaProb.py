from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_BinnedArctanhDeltaProb_input_variables,
    sig_bkg_dnn_BinnedArctanhDeltaProb_input_variables,
)

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict |= {
    "spanet": "/work/tharte/datasets/output_spanet/spanet_hh4b_inclusive_5jets_100_pvary_loose_s300_bratio_all.onnx",  # spanet pt vary 0.3, 1.7, btag ratio all
    "bkg_morphing_dnn": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_btag_ratioAll_postEE_BinnedArctanhDeltaProb/best_models/average_model_from_onnx.onnx",
    "sig_bkg_dnn": "/work/mmalucch/out_ML_pytorch/DNN_ptFlatSPANet_btag_ratioAll_class_weights_e5drop75_postEE_allklambda_BinnedArctanhDeltaProb/state_dict/model_best_epoch_31.onnx",
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
    "bkg_morphing_dnn_input_variables": bkg_morphing_dnn_BinnedArctanhDeltaProb_input_variables,
    "sig_bkg_dnn_input_variables": sig_bkg_dnn_BinnedArctanhDeltaProb_input_variables,
    "parton_jet_min_dR": 0.4,
    "max_num_jets": 5,
    "which_bquark": "last",
    "fifth_jet": "pt",
    "donotscale_sumgenweights": False,
    "pad_value": -999.0,
    "arctanh_delta_prob_bin_edge": 3.2,
    "arctanh_delta_prob_pad_limit": 2.0,
    "add_jet_spanet": True,
    "spanet_input_name_list": ["log_pt", "eta", "phi", "btag_ratioAll"],
} | onnx_model_dict
