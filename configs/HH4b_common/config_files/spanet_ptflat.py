from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_input_variables,
    sig_bkg_dnn_input_variables,
)

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict |= {
    "spanet": "/work/tharte/datasets/onnx_spanet_models_for_pairing_and_mass_sculpting_studies/hh4b_5jets_e300_s100_ptvary_wide_loose_btag.onnx",  # spanet pt vary 0.3, 1.7
    # "spanet": "/work/tharte/datasets/onnx_spanet_models_for_pairing_and_mass_sculpting_studies/hh4b_5jets_e300_s160_btag.onnx", # thierry's model, spanet baseline
    # "spanet": "params/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx", # ruth's model
    #
    # "vbf_ggf_dnn":"/t3home/rcereghetti/ML_pytorch/out/20241212_223142_SemitTightPtLearningRateConstant/models/model_28.onnx",
    #
    ### SPANET pt vary 0.3, 1.7
    # "bkg_morphing_dnn": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_oversample_split_20folds/batch06/best_models/average_model_from_onnx.onnx",  # full 2022, 20 k-folds, early stopping, 1e-5 minDelta, spanet pt vary
    # "bkg_morphing_dnn": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_postEE/best_models/average_model_from_onnx.onnx",  # only 2022_postEE, 20 k-folds, early stopping, 1e-5 minDelta, spanet pt vary
    "bkg_morphing_dnn": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_postEE_UpdatedDatasets/best_models/average_model_from_onnx.onnx",  # only 2022_postEE, 20 k-folds, early stopping, 1e-5 minDelta, spanet pt vary
    # "bkg_morphing_dnn": "/work/mmalucch/out_ML_pytorch/DNN_AN_VR1_1e-3_e20drop75_minDelta1em5_SPANet_oversample_split/state_dict/average_model_from_onnx.onnx",  # VR1 train, early stopping, 1e-5 minDelta, spanet pt vary
    ### SPANET pt vary 0.3, 1.7
    # "bkg_morphing_spread_dnn": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_postEE/best_models/all_ratios_model_onnx.onnx",  # only 2022_postEE, 20 k-folds, early stopping, 1e-5 minDelta, spanet, all ration spread model
    # "sig_bkg_dnn": "/work/mmalucch/out_ML_pytorch/DNN_ptFlatSPANet_class_weights_e5drop75/state_dict/model_best_epoch_25.onnx", # morphing full 2022, FixedPairingMorphing, FixedModelMorphing
    # "sig_bkg_dnn": "/work/mmalucch/out_ML_pytorch/DNN_ptFlatSPANet_class_weights_e5drop75_postEE/state_dict/model_best_epoch_13.onnx", # only 2022postEE, FixedPairingMorphing, FixedModelMorphing
    "sig_bkg_dnn": "/work/mmalucch/out_ML_pytorch/DNN_ptFlatSPANet_class_weights_e5drop75_postEE_allklambda/state_dict/model_best_epoch_16.onnx", # allklambda, only 2022postEE, FixedPairingMorphing, FixedModelMorphing
    # "sig_bkg_dnn": "/work/mmalucch/out_ML_pytorch/DNN_ptFlatSPANet_class_weights_e5drop75_WrongMorphing/state_dict/model_best_epoch_21.onnx", # pytorch training, FixedPairingMorphing, WrongModelMorphing
    # "sig_bkg_dnn": "/work/mmalucch/out_ML_pytorch/DNN_ptFlatSPANet_class_weights_e5drop75_WrongPairingMorphing/state_dict/model_best_epoch_14.onnx", # pytorch training, WrongPairingMorphing
    # "sig_bkg_dnn": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_SvsB/model_fold0.onnx", # soumya's model
}


config_options_dict |= {
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
} | onnx_model_dict
