import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict  |= {
    # "vbf_discriminator": "/work/tharte/datasets/onnx_spanet_models_for_ggf_vs_vbf_boosted/1_16_4_boosted_vbf_ggf_classification_fixedworkflow.onnx"
    "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch_matteo/out/bkg_reweighting/boosted_vbf/1e-3_e20drop95_minDelta1em5_boosted_reweighting_btagging_reweight_50GeV_lower_band_txbbwps_reweight_shape/best_models/average_model_from_onnx.onnx",
    # "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch_matteo/out/bkg_reweighting/boosted_vbf/1e-4_e20drop95_minDelta1em5_unskimmed_boosted_multiclass_oversample_split_looser_cuts_novbf_cuts/best_models/average_model_from_onnx_multiclass.onnx",
    # "bkg_morphing_spread_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DHH_method_20_runs_postEE/best_models/ratio/all_ratios_model_onnx.onnx", # --> training on postEE
    "sig_bkg_dnn": "/work/tharte/datasets/ML_pytorch_matteo/out/sig_bkg_classifier/boosted/1e-4_e20drop95_minDelta1em5_boosted_signal_background_classifier_50GeV_noH2WP_txbbwps_shape_reweight/state_dict/model_best_epoch_74.onnx",
}


config_options_dict |= {
    "dnn_variables": True,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_boosted_dnn_input_variables_other_group_txbb_wp_noh2txbb,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_boosted_dnn_input_variables_other_group_njets,
    "vbf_discriminator_input_variables": dnn_vars.vbf_discriminator_boosted_dnn_input_variables,
    "max_num_jets_vbf_discriminator": 2,
    "run2": False,
    "fifth_jet": "pt",
    "pad_value": -999.0,
    "add_jet_spanet": False,
    "boosted": True,
    "boosted_presel": True,
    "split_qcd": True,
    "approach": "boosted",
    # VBF
    "vbf_parton_matching": False,
    "vbf_analysis": True,
    "vbf_selection": True,
    "ggf_vbf_threshold": 0.95,
    "which_vbf_quark": "with_mothers_children",
    "max_num_jets_add_vbf": 2,
    "TXbb_order": True,
    "boosted_full_sideband": False,
    "region_setting": "50GeV",  # This defines the size of the signal and control regions in the boosted topology
    "lead_sublead": "lead",
    # "optimised_signal_cuts": {"txbb_c1": 0.95, "bdt_ggf_c1": 0.97097, "txbb_vbf": 0.8, "bdt_vbf": 0.98826, "txbb_c2a": 0.8, "bdt_ggf_c2b": 0.85303, "bdt_ggf_bkg": 0.045441},
    "optimised_signal_cuts": False,  # {"txbb_c1": 0.95, "bdt_ggf_c1": 0.97097, "txbb_vbf": 0.8, "bdt_vbf": 0.98826, "txbb_c2a": 0.8, "bdt_ggf_c2b": 0.85303, "bdt_ggf_bkg": 0.045441},
    # "bdt_model": "/work/tharte/datasets/bdts/boosted/bdt_trainings_run3/ABCD_26Oct01_v15_2225_kf20median_abcd_smvbf_bal_xbb/trained_bdt.model",
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/1_16_17_boosted_other_group_50GeV_lower_sideband_txbbwps_shape_reweight",
} | onnx_model_dict
