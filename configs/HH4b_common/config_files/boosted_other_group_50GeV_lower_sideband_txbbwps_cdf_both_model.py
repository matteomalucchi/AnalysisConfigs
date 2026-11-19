import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


# Second pass (after `boosted_other_group_50GeV_lower_sideband_txbbwps_cdf_both.py`): the subleading-Higgs TXbb enters only as the
# CDF-rescaled working-point index (`btagBBTXbb_CDF_rescaled_WPs`), in the cuts of all regions (A, B, C, D), in the morphing
# DNN input and, with `--txbb-cdf`, in scripts/optimise_boosted_categories.py. No linear morph and no TXbb shape.
# This run applies only the morphing DNN: it defines the ABCD regions, the postW regions (A_postW, B_postW) and the
# closure categories per TXbb working point (`..._region_C_txbbwp<i>`, `..._region_A_txbbwp<i>_postW`, i = 0..4).
# There is no signal-vs-background DNN and no signal category (cat1, cat2, cat3, catvbf): the sig/bkg DNN is retrained
# afterwards on D vs B_postW of this run, then `sig_bkg_dnn` and `optimised_signal_cuts` are set (the TXbb cuts of the
# signal categories must then sit on working points 0.8, 0.95, 0.975, 0.99).
onnx_model_dict  |= {
    # Morphing DNN trained on the CDF-rescaled subleading TXbb working points and the nominal leading TXbb (variables
    # `bkg_morphing_boosted_dnn_input_variables_other_group_txbb_cdf_wp_both`), 20 folds, median over the folds, with the ttbar
    # subtraction (ML_pytorch: configs/hh4b_bkg_reweighting/boosted_vbf/..._lower_band_txbbwps_cdf_both.yml, run_training.sh
    # -k 20 --alpha -- --kfold-onnx median). The path is the expected output of that training, check that it exists.
    # The model trained on `btagBBTXbb_morph_5wp` cannot be used: its input has a different meaning in A.
    "bkg_morphing_dnn": "/work/zhiheng/ML_pytorch/out/1e-3_e20drop95_minDelta1em5_boosted_reweighting_btagging_reweight_50GeV_lower_band_txbbwps_cdf_both/kfold_onnx/final_model_kfold.onnx",
    # No signal-vs-background DNN for now (to be retrained)
    "sig_bkg_dnn": "",
}


config_options_dict |= {
    "dnn_variables": True,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_boosted_dnn_input_variables_other_group_txbb_wp_noh2txbb,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_boosted_dnn_input_variables_other_group_txbb_cdf_wp_both,
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
    # Quantile map of the TXbb working points, from region C to region A (data), for the subleading Higgs. See
    # `boosted_other_group_50GeV_lower_sideband_txbbwps_cdf_both.py` (same values, derived with scripts/find_txbb_cdf_wp_cuts.py).
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
    # No signal categories in this run (no sig/bkg DNN yet). Once it is retrained, set them here: with the CDF-rescaled WPs
    # the TXbb cuts must be working points (0.8, 0.95, 0.975, 0.99), e.g. the ones of the `_alex` config
    # {"txbb_c1": 0.95, "bdt_ggf_c1": 0.97097, "txbb_vbf": 0.8, "bdt_vbf": 0.98826, "txbb_c2a": 0.8, "bdt_ggf_c2b": 0.85303, "bdt_ggf_bkg": 0.045441},
    # and re-optimise them with scripts/optimise_boosted_categories.py --txbb-cdf.
    "optimised_signal_cuts": False,
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/zhiheng/HH4b/boosted_other_group_50GeV_lower_sideband_txbbwps_cdf_both_model",
} | onnx_model_dict
