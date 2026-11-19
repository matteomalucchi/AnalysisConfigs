import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


# Second pass of the "nomet + conditional CDF" chain (after boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_cdf_cond.py): applies the
# morphing DNN trained on its output. Defines the ABCD regions, the postW regions (A_postW, B_postW) and the closure categories per TXbb working
# point (`..._region_C_txbbwp<i>`, `..._region_A_txbbwp<i>_postW`, i = 0..4). Same template as the initial run (VBF_HH4b_boosted_config_sf.py,
# TXbb working-point scale factor on MC). No signal-vs-background DNN and no signal category (cat1, cat2, cat3, catvbf).
onnx_model_dict  |= {
    # Morphing DNN trained on the MET-free inputs with both CDF WP indices (`..._txbb_cdf_wp_both_nomet`), 20 folds, patience 15, median over
    # the folds, with the ttbar subtraction (ML_pytorch: configs/hh4b_bkg_reweighting/boosted_vbf/..._nomet_cdf_cond_patience15.yml). Expected output
    # of that training, check that it exists.
    "bkg_morphing_dnn": "/work/zhiheng/ML_pytorch/out/1e-3_e20drop95_minDelta1em5_boosted_reweighting_btagging_reweight_50GeV_lower_band_txbbwps_nomet_cdf_cond_patience15/kfold_onnx/final_model_kfold.onnx",
    # No signal-vs-background DNN (to be retrained)
    "sig_bkg_dnn": "",
}


config_options_dict |= {
    "dnn_variables": True,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_boosted_dnn_input_variables_other_group_txbb_wp_txbbwp_nomet_h1txbb_nomorph,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_boosted_dnn_input_variables_other_group_txbb_cdf_wp_both_nomet,
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
    # No signal categories in this chain (no sig/bkg DNN yet); the TXbb cuts of the signal categories would have to be working points
    # (0.8, 0.95, 0.975, 0.99) with the CDF-rescaled working-point indices.
    "optimised_signal_cuts": False,
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
    # Conditional quantile map of the TXbb working points of the LEADING Higgs, from region C to region A (data), applied by region
    # (A, B: subleading TXbb <= 0.65; C, D: nominal index). One row of 4 leading cuts per subleading working-point index s = 1..5
    # (rows of `cuts_A`; -1 = always exceeded, so the leading index is at least s). Derived with
    # `scripts/find_txbb_cdf_wp_cuts.py --higgs leading --conditional --subleading-cuts <subleading json>` on
    # /work/zhiheng/BoostedAnalysis/Out/boosted_other_group_50GeV_lower_sideband_txbbwps_cdf_initial
    # (output: <that directory>/txbb_cdf_wp_cuts_leading/txbb_cdf_wp_cuts_leading_cond.json), no pooling of the subleading indices.
    # The rows s = 4 (31 events in C) and s = 5 (3 events) are statistically limited. `subleading_cuts_A` must equal the cuts of
    # `txbb_cdf_wp_cuts` above (checked by the workflow). Given inline so that the workers do not need to read the file.
    # Rerun the script and update these values if the region definition, the data, the selection or the subleading cuts change.
    "txbb_cdf_wp_cuts_leading": {
        "region_setting": "50GeV",
        "conditional": True,
        "subleading_groups": [[1], [2], [3], [4], [5]],
        "wps_C": [0.8, 0.95, 0.975, 0.99],
        "percentiles": [
            [0.30211888967604816, 0.872415791846461, 0.948412442841502, 0.9855735170992465],
            [0.0, 0.6824794433902593, 0.8787160025300442, 0.9656862745098039],
            [0.0, 0.0, 0.39303482587064675, 0.835820895522388],
            [0.0, 0.0, 0.0, 0.5806451612903226],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "cuts_A": [
            [0.24720105886701194, 0.8305375729186372, 0.9203821186542726, 0.9704546919461418],
            [-1.0, 0.7685289300370262, 0.8984510996705892, 0.9620049683486714],
            [-1.0, -1.0, 0.7612184724997525, 0.9219112147146197],
            [-1.0, -1.0, -1.0, 0.8324324488639832],
            [-1.0, -1.0, -1.0, -1.0],
        ],
        "subleading_cuts_A": [0.22481720647279396, 0.6110200870299926, 0.6442795881739497, 0.6494463093495494],
    },
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/zhiheng/HH4b/boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_cdf_cond_model",
} | onnx_model_dict
