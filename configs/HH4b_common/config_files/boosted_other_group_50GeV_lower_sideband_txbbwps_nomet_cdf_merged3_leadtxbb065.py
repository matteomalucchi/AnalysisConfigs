import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


# Initial run built from boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_cdf_merged3_leadge.py: the same selection, inputs and
# merged subleading working points (txbb_cdf_max_wp = 3, original 4 inline cuts, subleading index 1, 2, 3; leading input = nominal index
# `HiggsLeading btagBBTXbb_5wp`, unmerged), except that the control regions A, B are no longer cut with `cut_sub_above_lead`. Instead
#  - a PRESELECTION cut `HiggsLeading btagBBTXbb > 0.65` (strict, `boosted_lead_txbb_min`) is applied to all events: A, B, C, D (so also the
#    signal region D) and to MC. In C, D it removes nothing (the leading TXbb is the larger one, TXbb_order, and the subleading one is > 0.65);
#  - events with the CDF-rescaled subleading index above the nominal leading index remain (e.g. leading 0.7 = index 1, subleading raw 0.62 = index 3).
#    Raw subleading TXbb above the leading one cannot happen (TXbb_order).
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
    # They were derived on ALL of A (before any leading-TXbb cut): re-derive them on the cut sample if that matters.
    "txbb_cdf_wp_cuts": {
        "region_setting": "50GeV",
        "wps_C": [0.8, 0.95, 0.975, 0.99],
        "percentiles": [0.7023135781228778, 0.9893602571648481, 0.9984606329515099, 0.9998641734957214],
        "cuts_A": [0.22481720647279396, 0.6110200870299926, 0.6442795881739497, 0.6494463093495494],
    },
    # Merge the working points 3, 4, 5 (TXbb > 0.95) into the index 3: the subleading index is 1 (0.65-0.8), 2 (0.8-0.95), 3 (> 0.95)
    "txbb_cdf_max_wp": 3,
    # NOT used in this run: the leading TXbb > 0.65 preselection cut replaces it
    "cut_sub_above_lead": False,
    # Preselection: leading Higgs TXbb > 0.65 (strict), all regions and samples
    "boosted_lead_txbb_min": 0.65,
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/zhiheng/HH4b/boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_cdf_merged3_leadtxbb065_initial",
} | onnx_model_dict
