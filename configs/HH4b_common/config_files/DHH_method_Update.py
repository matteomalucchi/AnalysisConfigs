import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict  |= {
    "bkg_morphing_dnn": "/work/mmalucch/out_ML_pytorch/hh4b_bkg_morphing/DNN_AN_1e-3_e20drop75_minDelta1em5_run2_newUpdates_postEE/best_models/average_model_from_onnx.onnx", 
    "sig_bkg_dnn": "/work/mmalucch/out_ML_pytorch/hh4b_sig_bkg_classifier/DNN_AN_1e-3_e20drop75_minDelta1em5_run2_newUpdates_postEE/run100/state_dict/model_best_epoch_28.onnx",
}


config_options_dict |= {
    "dnn_variables": True,
    "run2": True,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_dnn_input_variables,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_dnn_input_variables,
    "fifth_jet": "pt",
    "pad_value": -999.0,
    "add_jet_spanet": True,
    "qt_postEE": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/out_hh4b/bkg_morphing_studies/out_DATA_MC_DHH_newUpdates_JECRegression_BkgMorphing/quantile_transformer/SRRun2_qt/qt_events_sig_bkg_dnn_score_DHH_kl_1.00.pkl",
}| onnx_model_dict
