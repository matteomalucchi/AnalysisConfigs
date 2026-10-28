import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict  |= {
    # "vbf_discriminator": "/work/tharte/datasets/onnx_spanet_models_for_ggf_vs_vbf_boosted/1_16_4_boosted_vbf_ggf_classification_fixedworkflow.onnx"
    # "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch_matteo/out/bkg_reweighting/boosted_vbf/1e-3_e20drop95_minDelta1em5_boosted_reweighting_btagging_reweight_50GeV_lower_band_njets/best_models/average_model_from_onnx.onnx",
    "bkg_morphing_dnn": "/work/zhiheng/out/reweighting_dnn/run00/best_models/all_ratios_model_onnx_btag.onnx",
    # "bkg_morphing_spread_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DHH_method_20_runs_postEE/best_models/ratio/all_ratios_model_onnx.onnx", # --> training on postEE
    # "sig_bkg_dnn": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/DHH_method_norm_e5drop75_postEE/state_dict/model_best_epoch_18.onnx",
}


config_options_dict |= {
    "dnn_variables": True,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_boosted_dnn_input_variables,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_boosted_dnn_input_variables_other_group_50GeV,
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
    "region_setting": "50GeV",  # This defines the size of the signal and control regions in th  e boosted topology
    "lead_sublead": "lead",
    # "bdt_model": "/work/tharte/datasets/bdts/boosted/bdt_trainings_run3/ABCD_26Oct01_v15_2225_kf20median_abcd_smvbf_bal_xbb/trained_bdt.model",
    # "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/spanet_1_16_9_boosted_other_group_50GeV_lower_sideband_leading_higgs_region_btagging_newbdt",
    "bdt_model": "/pnfs/psi.ch/cms/trivcat/store/user/tharte/datasets/bdts/boosted/bdt_trainings_run3/ABCD_26Oct01_v15_2225_kf20median_abcd_smvbf_bal_xbb/trained_bdt.model",
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/zhiheng/test/apply_SvB_bdt_with_syst",
} | onnx_model_dict
