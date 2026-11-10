import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


# Second pass (after `boosted_other_group_50GeV_lower_sideband_txbbwps_cdf.py`): the subleading-Higgs TXbb enters only as the
# CDF-rescaled working-point index (`btagBBTXbb_CDF_rescaled_WPs`), in the cuts of all regions (A, B, C, D), in the morphing
# DNN input and, with `--txbb-cdf`, in scripts/optimise_boosted_categories.py. No linear morph and no TXbb shape.
# This run applies only the morphing DNN: it defines the ABCD regions, the postW regions (A_postW, B_postW) and the
# closure categories per TXbb working point (`..._region_C_txbbwp<i>`, `..._region_A_txbbwp<i>_postW`, i = 0..4).
# There is no signal-vs-background DNN and no signal category (cat1, cat2, cat3, catvbf): the sig/bkg DNN is retrained
# afterwards on D vs B_postW of this run, then `sig_bkg_dnn` and `optimised_signal_cuts` are set (the TXbb cuts of the
# signal categories must then sit on working points 0.8, 0.95, 0.975, 0.99).
onnx_model_dict  |= {
    # Morphing DNN trained on the CDF-rescaled subleading TXbb working points and the nominal leading TXbb (variables
    # `bkg_morphing_boosted_dnn_input_variables_other_group_txbb_cdf_wp`), 20 folds, median over the folds, with the ttbar
    # subtraction (ML_pytorch: configs/hh4b_bkg_reweighting/boosted_vbf/..._lower_band_txbbwps_cdf.yml, run_training.sh
    # -k 20 --alpha -- --kfold-onnx median). The path is the expected output of that training, check that it exists.
    # The model trained on `btagBBTXbb_morph_5wp` cannot be used: its input has a different meaning in A.
    "bkg_morphing_dnn": "/work/zhiheng/ML_pytorch/out/1e-3_e20drop95_minDelta1em5_boosted_reweighting_btagging_reweight_50GeV_lower_band_txbbwps_cdf/kfold_onnx/final_model_kfold.onnx",
    # No signal-vs-background DNN for now (to be retrained)
    "sig_bkg_dnn": "",
}


config_options_dict |= {
    "dnn_variables": True,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_boosted_dnn_input_variables_other_group_txbb_wp_noh2txbb,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_boosted_dnn_input_variables_other_group_txbb_cdf_wp,
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
    # `boosted_other_group_50GeV_lower_sideband_txbbwps_cdf.py` (same values, derived with scripts/find_txbb_cdf_wp_cuts.py).
    "txbb_cdf_wp_cuts": {
        "region_setting": "50GeV",
        "wps_C": [0.8, 0.95, 0.975, 0.99],
        "percentiles": [0.7023135781228778, 0.9893602571648481, 0.9984606329515099, 0.9998641734957214],
        "cuts_A": [0.22481720647279396, 0.6110200870299926, 0.6442795881739497, 0.6494463093495494],
    },
    # No signal categories in this run (no sig/bkg DNN yet). Once it is retrained, set them here: with the CDF-rescaled WPs
    # the TXbb cuts must be working points (0.8, 0.95, 0.975, 0.99), e.g. the ones of the `_alex` config
    # {"txbb_c1": 0.95, "bdt_ggf_c1": 0.97097, "txbb_vbf": 0.8, "bdt_vbf": 0.98826, "txbb_c2a": 0.8, "bdt_ggf_c2b": 0.85303, "bdt_ggf_bkg": 0.045441},
    # and re-optimise them with scripts/optimise_boosted_categories.py --txbb-cdf.
    "optimised_signal_cuts": False,
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/zhiheng/HH4b/boosted_other_group_50GeV_lower_sideband_txbbwps_cdf_model",
} | onnx_model_dict
