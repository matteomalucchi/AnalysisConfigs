from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_ArctanhDeltaProb_input_variables,
    sig_bkg_dnn_ArctanhDeltaProb_input_variables,
)

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict |= {
    "spanet": "/work/tharte/datasets/output_spanet/spanet_hh4b_inclusive_5jets_100_pvary_loose_s300_bratio_all.onnx",  # spanet pt vary 0.3, 1.7, btag ratio all
    "bkg_morphing_dnn": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_btag_ratioAll_postEE_ArctanhDeltaProb/best_models/average_model_from_onnx.onnx",  # btag_ratio_all, arctanh PD, only 2022_postEE, 20 k-folds, early stopping, 1e-5 minDelta, spanet pt vary 
    "sig_bkg_dnn": "/work/mmalucch/out_ML_pytorch/DNN_ptFlatSPANet_btag_ratioAll_class_weights_e5drop75_postEE_allklambda_ArctanhDeltaProb/state_dict/model_best_epoch_25.onnx",
}


config_options_dict |= {
    "vbf_parton_matching": False,
    "tight_cuts": False,
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/spanet_ptflat_btag_ratioAll_ArctanhDeltaProb",
    "vbf_presel": False,
    "dnn_variables": True,
    "run2": False,
    "vr1": False,
    "random_pt": False,
    "rand_type": 0.3,
    "blind": True if onnx_model_dict["sig_bkg_dnn"] else False,
    "bkg_morphing_dnn_input_variables": bkg_morphing_dnn_ArctanhDeltaProb_input_variables,
    "sig_bkg_dnn_input_variables": sig_bkg_dnn_ArctanhDeltaProb_input_variables,
    "parton_jet_min_dR": 0.4,
    "max_num_jets_good": 5,
    "which_bquark": "last",
    "fifth_jet": "pt",
    "donotscale_sumgenweights": False,
    "pad_value": -999.0,
    "arctanh_delta_prob_bin_edge": 2.44,
    "arctanh_delta_prob_pad_limit": 2.0,
    "add_jet_spanet": True,
    "spanet_input_name_list": ["log_pt", "eta", "phi", "btag_ratioAll"],
} | onnx_model_dict
