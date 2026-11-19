import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


# Second pass of the "nomet + LINEAR + merged3 + lead >= sub cut" chain (after
# boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_linear_merged3_leadge.py): applies the morphing DNN trained on its output. Defines the ABCD
# regions, the postW regions (A_postW, B_postW) and the closure categories per TXbb working point (`..._region_C_txbbwp<i>`,
# `..._region_A_txbbwp<i>_postW`, i = 0..2, the last one is the merged bin (0.95, 1.0]). The control regions A, B (and their postW and txbbwp
# categories) are cut with `cut_sub_above_lead`, as in the initial run. Same template as the initial run (VBF_HH4b_boosted_config_sf.py, TXbb
# working-point scale factor on MC). No signal-vs-background DNN and no signal category.
onnx_model_dict  |= {
    # Morphing DNN trained on the MET-free inputs with working-point indices only (`..._txbb_linear_wp_nomet`: nominal leading index
    # `btagBBTXbb_5wp` and the merged linear subleading index `btagBBTXbb_morph_5wp`), 20 folds, patience 15, median over the folds, with the
    # ttbar subtraction (ML_pytorch: configs/hh4b_bkg_reweighting/boosted_vbf/..._nomet_linear_merged3_leadge_patience15.yml).
    # Expected output of that training, check that it exists.
    "bkg_morphing_dnn": "/work/zhiheng/ML_pytorch/out/1e-3_e20drop95_minDelta1em5_boosted_reweighting_btagging_reweight_50GeV_lower_band_txbbwps_nomet_linear_merged3_leadge_patience15/kfold_onnx/kfold_morphing_ttbar_subtracted_median.onnx",
    # No signal-vs-background DNN (to be retrained)
    "sig_bkg_dnn": "",
}


config_options_dict |= {
    "dnn_variables": True,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_boosted_dnn_input_variables_other_group_txbb_wp_txbbwp_nomet_h1txbb_nomorph,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_boosted_dnn_input_variables_other_group_txbb_linear_wp_nomet,
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
    # Per working point closure categories (C vs A_postW), needs the morphing DNN
    "boosted_txbb_wp_categories": True,
    "region_setting": "50GeV",  # This defines the size of the signal and control regions in the boosted topology
    "lead_sublead": "lead",
    # No signal categories in this chain (no sig/bkg DNN yet)
    "optimised_signal_cuts": False,
    # LINEAR rescaling: no quantile map. MUST be the same as in the initial run config (the DNN was trained with this index).
    "txbb_cdf_wp_cuts": "",
    # Merge the working points 3, 4, 5 (TXbb > 0.95) into the index 3 of btagBBTXbb_morph_5wp (subleading index 1, 2, 3)
    "txbb_cdf_max_wp": 3,
    # Remove, in A and B (also postW and txbbwp categories), the events with nominal leading working-point index < (merged, linear) subleading index
    "cut_sub_above_lead": True,
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/zhiheng/HH4b/boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_linear_merged3_leadge_model",
} | onnx_model_dict
