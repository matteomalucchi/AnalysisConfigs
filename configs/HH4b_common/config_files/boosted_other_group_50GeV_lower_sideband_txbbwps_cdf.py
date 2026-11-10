import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


# Initial run (step 1 of configs/VBF_HH4b_boosted/README.md): event selection, the 4 ABCD regions and the column
# definitions, without any model. The TXbb working points of the low-TXbb regions (A, B) are quantile-mapped onto the ones
# of the high-TXbb regions (C, D), see `txbb_cdf_wp_cuts` below, which defines
# `HiggsSubLeading_btagBBTXbb_CDF_rescaled_WPs` next to the nominal `btagBBTXbb_morph_5wp`.
onnx_model_dict  |= {
    # "vbf_discriminator": "/work/tharte/datasets/onnx_spanet_models_for_ggf_vs_vbf_boosted/1_16_4_boosted_vbf_ggf_classification_fixedworkflow.onnx"
    # "bkg_morphing_dnn": "/pnfs/psi.ch/cms/trivcat/store/user/tharte/datasets/ML_pytorch_matteo/out/bkg_reweighting/boosted_vbf/1e-3_e20drop95_minDelta1em5_boosted_reweighting_btagging_reweight_50GeV_lower_band_txbbwps/best_models/average_model_from_onnx.onnx",
    # "bkg_morphing_spread_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DHH_method_20_runs_postEE/best_models/ratio/all_ratios_model_onnx.onnx", # --> training on postEE
    # "sig_bkg_dnn": "/pnfs/psi.ch/cms/trivcat/store/user/tharte/datasets/ML_pytorch_matteo/out/sig_bkg_classifier/boosted/1e-4_e20drop95_minDelta1em5_boosted_signal_background_classifier_50GeV_noH2WP_txbbwps/state_dict/model_best_epoch_103.onnx",
}


config_options_dict |= {
    "dnn_variables": True,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_boosted_dnn_input_variables_other_group_txbb_wp_noh2txbb,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_boosted_dnn_input_variables_other_group_txbb_wp,
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
    # Quantile map of the TXbb working points, from region C to region A (data), for the subleading Higgs:
    # the cuts in A (for the WPs 0.8, 0.95, 0.975, 0.99 of C) so that the fraction of A in each WP bin equals the one of C.
    # Derived with scripts/find_txbb_cdf_wp_cuts.py on
    # /work/zhiheng/BoostedAnalysis/Out/boosted_other_group_50GeV_lower_sideband_txbbwps_syst_raw
    # (output: <that directory>/txbb_cdf_wp_cuts/txbb_cdf_wp_cuts.json). Given inline so that the workers do not need to read the file.
    # Rerun the script and update these values if the region definition, the data or the selection change.
    "txbb_cdf_wp_cuts": {
        "region_setting": "50GeV",
        "wps_C": [0.8, 0.95, 0.975, 0.99],
        "percentiles": [0.7023135781228778, 0.9893602571648481, 0.9984606329515099, 0.9998641734957214],
        "cuts_A": [0.22481720647279396, 0.6110200870299926, 0.6442795881739497, 0.6494463093495494],
    },
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/zhiheng/HH4b/boosted_other_group_50GeV_lower_sideband_txbbwps_cdf_initial",
} | onnx_model_dict
