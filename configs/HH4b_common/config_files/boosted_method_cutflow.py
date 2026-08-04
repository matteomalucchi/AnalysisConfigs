from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_boosted_dnn_input_variables,
    sig_bkg_boosted_dnn_input_variables,
)

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict |= {
    "bkg_morphing_dnn": "/work/bevila_t/PostDoc/HH4b/Output/ML_trainings/bkg_reweigting_boosted/first_test/best_models/average_model_from_onnx.onnx", 
}

config_options_dict |= {
    "dnn_variables": True,
    "sig_bkg_dnn_input_variables": sig_bkg_boosted_dnn_input_variables,
    "bkg_morphing_dnn_input_variables": bkg_morphing_boosted_dnn_input_variables,
    "run2": False,
    "fifth_jet": "pt",
    "pad_value": -999.0,
    "add_jet_spanet": False,
    "boosted": True,
    "boosted_presel": True,
    "split_qcd": True,
}| onnx_model_dict | {  "spanet": False,}
