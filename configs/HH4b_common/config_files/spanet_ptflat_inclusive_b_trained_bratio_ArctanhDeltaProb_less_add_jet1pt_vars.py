from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_ArctanhDeltaProb_input_variables,
    sig_bkg_dnn_ArctanhDeltaProb_input_variables_less_add_jet1pt_vars,
)

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict |= {
    "spanet": "/work/tharte/datasets/onnx_spanet_models_for_pairing_and_mass_sculpting_studies/spanet_hh4b_inclusive_b_region_5jets_loose_ptvary_wide_bratio_300_inclusive_years_s100.onnx",  # spanet pt vary 0.3, 1.7
    #
    "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_inclusive_b_region_bratioAll_postEE_ArctanhDeltaProb/best_models/ratio/average_model_from_onnx.onnx",  # only 2022_postEE, 20 k-folds, early stopping, 1e-5 minDelta, spanet pt vary
    "sig_bkg_dnn": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_inclusive_b_region_bratioAll_postEE_ArctanhDeltaProb_less_add_jet1pt_vars_no_add_jet1pt_Leading_and_Subleading_params/run100/state_dict/model_best_epoch_9.onnx",
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
    "run2": False,
    "vr1": False,
    "random_pt": False,
    "rand_type": 0.3,
    "blind": True if onnx_model_dict["sig_bkg_dnn"] else False,
    "bkg_morphing_dnn_input_variables": bkg_morphing_dnn_ArctanhDeltaProb_input_variables,
    "sig_bkg_dnn_input_variables": sig_bkg_dnn_ArctanhDeltaProb_input_variables_less_add_jet1pt_vars,
    "parton_jet_min_dR": 0.4,
    "max_num_jets": 5,
    "which_bquark": "last",
    "fifth_jet": "pt",
    "donotscale_sumgenweights": False,
    "pad_value": -999.0,
    "arctanh_delta_prob_bin_edge": 2.44,
    "arctanh_delta_prob_pad_limit": 2.,
    "add_jet_spanet": True,
    "spanet_input_name_list": ["log_pt", "eta", "phi", "btag_ratioAll"],
} | onnx_model_dict
