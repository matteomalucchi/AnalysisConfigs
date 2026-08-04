import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict  |= {
    "spanet": "/work/tharte/datasets/onnx_spanet_models_for_pairing_and_mass_sculpting_studies/spanet_1_14_5_h4b_5jets_ptvary_loose_300_btag_wp_newLeptonVeto_3L1Cut_UpdateJetVetoMap.onnx",
    "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DNN_1_15_10_6_mixed_large_dataset_normalCR_bestworst_only_1e-3_e20drop95/best_models/average_model_from_onnx.onnx",
    # "sig_bkg_dnn": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/1_15_10_6_sig_bkg_mixed_dataset_large_with_btag_pd_normal_CR_with_bestworst_only/run100/state_dict/model_best_epoch_28.onnx",
    "sig_bkg_dnn": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/1_15_10_6_sig_bkg_mixed_dataset_large_with_btag_pd_normal_CR_with_bestworst_only_high_score/run100/state_dict/model_best_epoch_16.onnx",
}


config_options_dict |= {
    "dnn_variables": True,
    "run2": False,
    "max_num_jets_good": 5,
    "sig_bkg_dnn_input_variables": dnn_vars.bkg_morphing_dnn_input_variables_mixeddata_bestworst_only,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_dnn_input_variables_mixeddata_bestworst_only,
    "fifth_jet": "pt",
    "pad_value": -999.0,
    "add_jet_spanet": True,
    "spanet_input_name": dnn_vars.pairing_spanet_btagWP5,
    "qt_postEE": "",
    "max_num_jets_spanet_class": 5,
    "expandCR": False,
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/spanet_1_15_10_6_mixed_large_dataset_normalCR_bestworst_only_high_score",
}| onnx_model_dict
