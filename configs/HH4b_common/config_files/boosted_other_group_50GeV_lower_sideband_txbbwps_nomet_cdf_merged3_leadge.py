import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


# Initial run (step 1 of configs/VBF_HH4b_boosted/README.md) built from
# boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_alex.py: the same selection and no MET inputs, with
#  - the subleading TXbb working points of A, B quantile-mapped onto the ones of C, D (marginal map, option `txbb_cdf_wp_cuts`, the
#    original 4 cuts), and the working points 3, 4, 5 (TXbb > 0.95) merged into one bin (`txbb_cdf_max_wp` = 3): the subleading index is 1, 2, 3;
#  - the leading Higgs TXbb left untouched and not merged: the DNN trains on working-point indices only, the nominal leading index
#    `HiggsLeading btagBBTXbb_5wp` (1..5 after the cut below, 5 bins) and the merged subleading index (1..3);
#  - the control regions A, B restricted to the events whose nominal leading working-point index is not below the subleading index
#    (`cut_sub_above_lead`). In C, D the leading index is never below the subleading one, so they are not cut.
# No model: the morphing DNN is trained on this output afterwards. Run it with the template VBF_HH4b_boosted_config_sf.py, which applies
# the TXbb working-point scale factor (MC only). There is no signal-vs-background DNN and no signal category.
onnx_model_dict  |= {
    # no models in the initial run
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
    "region_setting": "50GeV",  # This defines the size of the signal and control regions in the boosted topology
    "lead_sublead": "lead",
    # No signal categories in this chain (no sig/bkg DNN yet); the TXbb cuts of the signal categories would have to be the working points
    # 0.8, 0.95 (the ones below the merged top bin).
    "optimised_signal_cuts": False,
    # Quantile map of the TXbb working points, from region C to region A (data), for the subleading Higgs:
    # the cuts in A (for the WPs 0.8, 0.95, 0.975, 0.99 of C) so that the fraction of A in each WP bin equals the one of C.
    # Derived with scripts/find_txbb_cdf_wp_cuts.py on
    # /work/zhiheng/BoostedAnalysis/Out/boosted_other_group_50GeV_lower_sideband_txbbwps_syst_raw
    # (output: <that directory>/txbb_cdf_wp_cuts/txbb_cdf_wp_cuts.json). Given inline so that the workers do not need to read the file.
    # They were derived on ALL of A (before the removal of the events with a leading index below the subleading one), so after that cut
    # the top bin of A is below the one of C: re-derive them on the cut sample if that matters.
    "txbb_cdf_wp_cuts": {
        "region_setting": "50GeV",
        "wps_C": [0.8, 0.95, 0.975, 0.99],
        "percentiles": [0.7023135781228778, 0.9893602571648481, 0.9984606329515099, 0.9998641734957214],
        "cuts_A": [0.22481720647279396, 0.6110200870299926, 0.6442795881739497, 0.6494463093495494],
    },
    # Merge the working points 3, 4, 5 (TXbb > 0.95) into the index 3: the subleading index is 1 (0.65-0.8), 2 (0.8-0.95), 3 (> 0.95)
    "txbb_cdf_max_wp": 3,
    # Remove, in A and B, the events with nominal leading working-point index < subleading index
    "cut_sub_above_lead": True,
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/zhiheng/HH4b/boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_cdf_merged3_leadge_initial",
} | onnx_model_dict
