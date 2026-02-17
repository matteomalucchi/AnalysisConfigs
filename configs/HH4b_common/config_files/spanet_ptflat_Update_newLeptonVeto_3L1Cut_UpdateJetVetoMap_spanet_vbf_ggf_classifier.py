import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict  |= {
    "spanet": "/work/mmalucch/out_hh4b/VBF/out_ggf_vbf_spanet_input_SM_ptFlatten_NoParquet/vbf_ggf_pairing_classification.onnx",
    "vbf_discriminator": "/work/mmalucch/out_hh4b/VBF/out_ggf_vbf_spanet_input_SM_ptFlatten_NoParquet/vbf_ggf_pairing_classification.onnx",
    # "bkg_morphing_dnn": "",
    # "sig_bkg_dnn": "",
}


config_options_dict |= {
    "dnn_variables": True,
    "run2": False,
    "max_num_jets_good": 5,
    "max_num_jets_spanet": 9,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_dnn_input_variables_spanet,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_dnn_input_variables,
    "fifth_jet": "pt",
    "pad_value": -999.0,
    "add_jet_spanet": True,
    "spanet_input_name": dnn_vars.pairing_spanet_vbf_ggf_btagWP5,
    # VBF
    "vbf_parton_matching": False,
    "vbf_presel": False,
    "vbf_analysis": True,
    "which_vbf_quark":"with_mothers_children",
    "max_num_jets_add_vbf": 2,
}| onnx_model_dict
