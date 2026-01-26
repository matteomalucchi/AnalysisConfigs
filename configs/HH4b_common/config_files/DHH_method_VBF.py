from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_input_variables,
    sig_bkg_dnn_input_variables,
)

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict  |= {
    # "bkg_morphing_dnn": "/work/mmalucch/out_ML_pytorch/hh4b_bkg_morphing/DNN_AN_1e-3_e20drop75_minDelta1em5_run2_newUpdates_postEE/best_models/average_model_from_onnx.onnx", 
    # "sig_bkg_dnn": "/work/mmalucch/out_ML_pytorch/hh4b_sig_bkg_classifier/DNN_AN_1e-3_e20drop75_minDelta1em5_run2_newUpdates_postEE/run100/state_dict/model_best_epoch_28.onnx",
}


config_options_dict |= {
    "dnn_variables": True,
    "run2": True,
    "sig_bkg_dnn_input_variables": sig_bkg_dnn_input_variables,
    "bkg_morphing_dnn_input_variables": bkg_morphing_dnn_input_variables,
    "fifth_jet": "pt",
    "pad_value": -999.0,
    "add_jet_spanet": True,
    "qt_postEE": None,
    "vbf_parton_matching": True,
    "vbf_presel": False,
    "vbf_analysis": True,
    "which_vbf_quark":"with_status"
}| onnx_model_dict
