import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict  |= {
    "bkg_morphing_dnn": "/work/mmalucch/out_ML_pytorch/hh4b_bkg_morphing/DNN_AN_1e-3_e20drop75_minDelta1em5_run2_newUpdates_newLeptonVeto_OrOfL1Cut_UpdateJetVetoMap_postEE/best_models/average_model_from_onnx.onnx", 
    "sig_bkg_dnn": "/work/mmalucch/out_ML_pytorch/hh4b_sig_bkg_classifier/DNN_AN_1e-3_e20drop75_minDelta1em5_run2_newUpdates_newLeptonVeto_OrOfL1Cut_UpdateJetVetoMap_postEE/run100/state_dict/model_best_epoch_34.onnx",
}


config_options_dict |= {
    "dnn_variables": True,
    "run2": True,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_dnn_input_variables,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_dnn_input_variables,
    "fifth_jet": "pt",
    "pad_value": -999.0,
    "add_jet_spanet": True,
    "save_chunk":"/work/mmalucch/out_hh4b/full_analysis/out_MC_DATA_DHH_newUpdates_JECRegression_newLeptonVeto_OrOfL1Cut_UpdateJetVetoMap/sig_vs_bkg/parquet_files/",
    "qt_postEE": "/work/mmalucch/out_hh4b/full_analysis/out_MC_DATA_DHH_newUpdates_JECRegression_newLeptonVeto_OrOfL1Cut_UpdateJetVetoMap/bkg_morphing/quantile_transformer/SRRun2_qt/qt_events_sig_bkg_dnn_score_DHH_kl_1.00.pkl",
}| onnx_model_dict
