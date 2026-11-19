import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


# Initial run (step 1 of configs/VBF_HH4b_boosted/README.md): event selection, the 4 ABCD regions and the column
# definitions, without any model. The TXbb working points of the low-TXbb regions (A, B) are quantile-mapped onto the ones
# of the high-TXbb regions (C, D), see `txbb_cdf_wp_cuts` below, which defines
# `HiggsSubLeading_btagBBTXbb_CDF_rescaled_WPs` next to the nominal `btagBBTXbb_morph_5wp`.
# `_cdf_both`: the same quantile map is also applied to the LEADING Higgs (`txbb_cdf_wp_cuts_leading`, column
# `HiggsLeading_btagBBTXbb_CDF_rescaled_WPs`, by region), so that the morphing DNN does not see the continuous leading TXbb.
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
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/zhiheng/HH4b/boosted_other_group_50GeV_lower_sideband_txbbwps_cdf_both_initial",
} | onnx_model_dict
