import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict  |= {
    "spanet": "/work/tharte/datasets/onnx_spanet_models_for_pairing_and_mass_sculpting_studies/spanet_1_14_5_h4b_5jets_ptvary_loose_300_btag_wp_newLeptonVeto_3L1Cut_UpdateJetVetoMap.onnx",
    "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DNN_AN_1e-3_e10drop75_minDelta1em5_SPANet_3L1Cut_postEE_mixeddata_control_30_80_JetVetoMap_fix/best_models/average_model_from_onnx.onnx",
    # "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DNN_AN_1e-3_e10drop75_minDelta1em5_SPANet_3L1Cut_postEE_mixeddata_control_30_80/best_models/average_model_from_onnx.onnx",
    # "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DNN_AN_1e-3_e20drop95_minDelta1em5_SPANet_3L1Cut_postEE_mixeddata_control_30_80_best_worst/best_models/average_model_from_onnx.onnx",
    # "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DNN_AN_1e-3_e20drop95_minDelta1em5_SPANet_3L1Cut_postEE_mixeddata_control_30_80_different_model/best_models/average_model_from_onnx.onnx",
    # "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DNN_AN_1e-3_e10drop75_minDelta1em5_SPANet_3L1Cut_postEE_mixeddata_control_30_80_tryimprovement/best_models/average_model_from_onnx.onnx",
    # "sig_bkg_dnn": "/work/tharte/datasets/onnx_spanet_models_for_classification_sig_bkg/1_15_2_from_1_14_5b_spanet_hh4b_classifier_test_classifier_metric_jet_ptetaphimass_glob_arcanh_dr_ht_higgsleadsublead_HH_new_conversion_script_with_vbf_variables.onnx",
    "sig_bkg_dnn": "/work/tharte/datasets/onnx_spanet_models_for_classification_sig_bkg/spanet_hh4b_classifier_var000068_mixed_model_withBtag_withPD.onnx",
}


config_options_dict |= {
    "dnn_variables": True,
    "run2": False,
    "max_num_jets_good": 5,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_dnn_input_variables_spanet_mixeddata_btag_pd,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_dnn_input_variables_mixeddata,
    "fifth_jet": "pt",
    "pad_value": -999.0,
    "add_jet_spanet": True,
    "spanet_input_name": dnn_vars.pairing_spanet_btagWP5,
    "qt_postEE": "",
    "max_num_jets_spanet_class": 5,
    "mixeddata": True,
    "noL1": True,
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/spanet_ptflat_Update_newLeptonVeto_3L1Cut_UpdateJetVetoMap_mixeddata_control_30_80_var000068_btag_pd",
}| onnx_model_dict
