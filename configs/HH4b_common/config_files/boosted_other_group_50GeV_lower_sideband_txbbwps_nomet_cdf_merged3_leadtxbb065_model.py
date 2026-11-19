import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


# Second pass of the "nomet + merged3 + leading TXbb > 0.65 preselection" chain (after boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_cdf_merged3_leadtxbb065.py):
# applies the morphing DNN trained on its output. Defines the ABCD regions, the postW regions (A_postW, B_postW) and the closure categories per TXbb
# working point (`..._region_C_txbbwp<i>`, `..._region_A_txbbwp<i>_postW`, i = 0..2, the last one is the merged bin (0.95, 1.0]). The preselection cut
# `HiggsLeading btagBBTXbb > 0.65` (`boosted_lead_txbb_min`) is applied to all events, as in the initial run (so also to D and to MC); there is no
# `cut_sub_above_lead`. Same template as the initial run (VBF_HH4b_boosted_config_sf.py, TXbb working-point scale factor on MC). No signal-vs-background
# DNN and no signal category.
onnx_model_dict  |= {
    # Morphing DNN trained on the MET-free inputs with working-point indices only (`..._txbb_cdf_wp_nomet`: nominal leading index `btagBBTXbb_5wp`
    # and the merged subleading CDF index), 20 folds, patience 15, median over the folds, with the ttbar subtraction (ML_pytorch:
    # configs/hh4b_bkg_reweighting/boosted_vbf/..._nomet_cdf_merged3_leadtxbb065_patience15.yml). Expected output of that training, check that it exists.
    "bkg_morphing_dnn": "/work/zhiheng/ML_pytorch/out/1e-3_e20drop95_minDelta1em5_boosted_reweighting_btagging_reweight_50GeV_lower_band_txbbwps_nomet_cdf_merged3_leadtxbb065_patience15/kfold_onnx/final_model_kfold.onnx",
    # No signal-vs-background DNN (to be retrained)
    "sig_bkg_dnn": "",
}


config_options_dict |= {
    "dnn_variables": True,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_boosted_dnn_input_variables_other_group_txbb_wp_txbbwp_nomet_h1txbb_nomorph,
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
    # Per working point closure categories (C vs A_postW), needs the morphing DNN
    "boosted_txbb_wp_categories": True,
    "region_setting": "50GeV",  # This defines the size of the signal and control regions in the boosted topology
    "lead_sublead": "lead",
    # No signal categories in this chain (no sig/bkg DNN yet); the TXbb cuts of the signal categories would have to be the working points
    # 0.8, 0.95 (the ones below the merged top bin).
    "optimised_signal_cuts": False,
    # Quantile map of the TXbb working points, from region C to region A (data), for the subleading Higgs (the original marginal map, derived on all
    # of A before any leading-TXbb cut): the cuts in A (for the WPs 0.8, 0.95, 0.975, 0.99 of C). Derived with scripts/find_txbb_cdf_wp_cuts.py on
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
    # NOT used in this run: the leading TXbb > 0.65 preselection cut replaces it
    "cut_sub_above_lead": False,
    # Preselection: leading Higgs TXbb > 0.65 (strict), all regions and samples (same as the initial run)
    "boosted_lead_txbb_min": 0.65,
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/zhiheng/HH4b/boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_cdf_merged3_leadtxbb065_model",
} | onnx_model_dict
