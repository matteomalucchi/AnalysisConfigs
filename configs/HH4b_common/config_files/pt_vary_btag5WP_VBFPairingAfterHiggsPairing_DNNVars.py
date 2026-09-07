import configs.HH4b_common.dnn_input_variables as dnn_vars

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict |= {
    "spanet":"/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/spanet_vbf_models/vbf_ggf_all_Klambda_HiggsPairing.onnx",
    "vbf_discriminator":"/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/spanet_vbf_models/vbf_ggf_all_Klambda_VBFPairing_JetTotal_DNNVars_VBFNoKinCut_ClassLoss7_300e.onnx"

}

config_options_dict |= {
    "dnn_variables": True,
    "run2": False,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_dnn_input_variables,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_dnn_input_variables,
    "max_num_jets_good": 4,
    "which_bquark": "last",
    "fifth_jet": "pt",
    "pad_value": -999.0,
    "add_jet_spanet": True,
    "qt_postEE": None,
    "random_pt": True,
    "rand_type": 0.3,
    # "save_chunk":"root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/mmalucch/out_hh4b/VBF/out_ggf_vbf_spanet_input_AllKlambda_DetaMjjCentrality_VBFPairingAfterHiggsPairing_DNNVars/parquet_files/",
    "save_chunk":"root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/mmalucch/out_hh4b/VBF/out_ggf_vbf_spanet_input_AllKlambda_DetaMjjCentrality_VBFPairingAfterHiggsPairing_DNNVars/apply_ggf_vbf_discriminator/parquet_files/",
    "spanet_input_name": dnn_vars.pairing_spanet_btagWP5,
    # VBF
    # "vbf_discriminator_input_variables": dnn_vars.vbf_discriminator_boosted_dnn_input_variables,
    # "max_num_jets_vbf_discriminator": 2,
    "vbf_parton_matching": True,
    "vbf_presel": False,
    "vbf_analysis": True,
    "which_vbf_quark":"with_mothers_children",
    "max_num_jets_add_vbf": 3,
    "jets_add_vbf_order": "pt",
    "vbf_matching_after_higgs_pairing": True,
}| onnx_model_dict
