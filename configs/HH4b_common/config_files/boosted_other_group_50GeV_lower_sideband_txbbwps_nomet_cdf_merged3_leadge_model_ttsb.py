import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


# Preselection run with the ttbar sideband (regions E, F) on top of the "nomet + merged3 + lead >= sub cut" model run
# (boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_cdf_merged3_leadge_model.py): same selection, same CDF cuts, same morphing DNN (the
# reweighting is applied: postW weight in A_postW, B_postW, F_postW and the per-WP closure categories), same template (VBF_HH4b_boosted_config_sf.py).
# Differences to that config:
#   - `boosted_fit_sidebands: True`: adds `boosted_signal_ttbar_region_E` and `boosted_control_ttbar_region_F_postW` (both Higgs masses in 150-200 GeV).
#     F gets the same `lead >= sub` cut as A, B (`cut_sub_above_lead`). No signal categories (optimised_signal_cuts False), so no E/F per category.
#   - a signal-vs-background classifier is applied: the XGBoost BDT of ML_pytorch (38 inputs, no subleading Higgs mass, QCD/TTbar/VBF/ggF output).
#     Its scores are saved for the nominal masses and for the shifted ones: `*_qcdsb` (ABCD Higgs mass moved into the window) and `*_ttsb`
#     (ONLY the leading Higgs mass moved down by 50 GeV, the subleading one is not shifted), see VBFHH4bProcessor.get_sideband_mass_shifts.
onnx_model_dict  |= {
    # Morphing DNN trained on the MET-free inputs with working-point indices only (`..._txbb_cdf_wp_nomet`: nominal leading index `btagBBTXbb_5wp`
    # and the merged subleading CDF index), 20 folds, patience 15, median over the folds, with the ttbar subtraction (ML_pytorch:
    # configs/hh4b_bkg_reweighting/boosted_vbf/..._nomet_cdf_merged3_leadge_patience15.yml).
    "bkg_morphing_dnn": "/work/zhiheng/ML_pytorch/out/1e-3_e20drop95_minDelta1em5_boosted_reweighting_btagging_reweight_50GeV_lower_band_txbbwps_nomet_cdf_merged3_leadge_patience15/kfold_onnx/final_model_kfold.onnx",
    # XGBoost signal-vs-background classifier (ML_pytorch yml xgb_..._leadge_v2_txbb_nomass: subleading TXbb index as input, no subleading Higgs mass; 38 inputs,
    # best round 766, test loss 0.790). ONNX: input `InputVariables` [N, 38], output `Output` [N, 4] = QCD, TTbar, VBF, ggF probabilities (checked in onnxruntime).
    "sig_bkg_dnn": "/work/zhiheng/ML_pytorch/out/xgb_boosted_signal_background_classifier_50GeV_nomet_cdf_merged3_leadge_v2_txbb_nomass/state_dict/model_best_epoch_766.onnx",
}


config_options_dict |= {
    "dnn_variables": True,
    # Same 38 variables, in the same order, as the training of the XGBoost classifier
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_boosted_dnn_input_variables_other_group_txbb_cdf_wp_nomet_nosublmass,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_boosted_dnn_input_variables_other_group_txbb_cdf_wp_nomet,
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
    # Sidebands of the fit: ttbar sideband E, F (and the shifted-mass scores `_qcdsb`, `_ttsb`). Needs bkg_morphing_direction "btag" (default).
    "boosted_fit_sidebands": True,
    # Per working point closure categories (C vs A_postW), needs the morphing DNN
    "boosted_txbb_wp_categories": True,
    "region_setting": "50GeV",  # This defines the size of the signal and control regions in the boosted topology
    "lead_sublead": "lead",
    # No signal categories in this run (they would need TXbb cuts on the working points 0.8, 0.95, the ones below the merged top bin): the classifier
    # scores are saved as columns in all the categories instead.
    "optimised_signal_cuts": False,
    # Quantile map of the TXbb working points, from region C to region A (data), for the subleading Higgs (the original marginal map, derived on all
    # of A before the cut `cut_sub_above_lead`): the cuts in A (for the WPs 0.8, 0.95, 0.975, 0.99 of C). Derived with scripts/find_txbb_cdf_wp_cuts.py on
    # /work/zhiheng/BoostedAnalysis/Out/boosted_other_group_50GeV_lower_sideband_txbbwps_syst_raw
    # (output: <that directory>/txbb_cdf_wp_cuts/txbb_cdf_wp_cuts.json). Given inline so that the workers do not need to read the file.
    # MUST be the same as in the initial run config (the DNN was trained with this index).
    "txbb_cdf_wp_cuts": {
        "region_setting": "50GeV",
        "wps_C": [0.8, 0.95, 0.975, 0.99],
        "percentiles": [0.7023135781228778, 0.9893602571648481, 0.9984606329515099, 0.9998641734957214],
        "cuts_A": [0.22481720647279396, 0.6110200870299926, 0.6442795881739497, 0.6494463093495494],
    },
    # Merge the working points 3, 4, 5 (TXbb > 0.95) into the index 3 (subleading index 1, 2, 3)
    "txbb_cdf_max_wp": 3,
    # Remove, in A and B (also postW and txbbwp categories) and in the ttbar control region F, the events with nominal leading working-point index < subleading index
    "cut_sub_above_lead": True,
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/zhiheng/HH4b/boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_cdf_merged3_leadge_model_ttsb",
} | onnx_model_dict
