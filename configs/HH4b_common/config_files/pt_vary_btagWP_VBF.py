from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_input_variables,
    sig_bkg_dnn_input_variables,
)

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict



config_options_dict |= {
    "dnn_variables": False,
    "run2": False,
    "sig_bkg_dnn_input_variables": None,
    "bkg_morphing_dnn_input_variables": None,
    "max_num_jets_good": 5,
    "which_bquark": "last",
    "fifth_jet": "pt",
    "pad_value": -999.0,
    "add_jet_spanet": True,
    "qt_postEE": None,
    "random_pt": True,
    "rand_type": 0.3,
    # "save_chunk":"root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/mmalucch/out_hh4b/VBF/out_ggf_vbf_spanet_input_SM_ptFlatten/parquet_files/",
    # VBF
    "vbf_parton_matching": True,
    "vbf_presel": False,
    "vbf_analysis": True,
    "which_vbf_quark":"with_mothers_children"
}| onnx_model_dict
