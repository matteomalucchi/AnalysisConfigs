import configs.HH4b_common.dnn_input_variables as dnn_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


# Initial run (step 1 of configs/VBF_HH4b_boosted/README.md), LINEAR-rescaling counterpart of
# boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_cdf_merged3_leadge.py: the same selection and no MET inputs, with
#  - the subleading TXbb of A, B rescaled LINEARLY (`btagBBTXbb_morph`, [0.05, 0.65] -> [0.65, 1.0]; no quantile map: `txbb_cdf_wp_cuts` is empty),
#    its working-point index `HiggsSubLeading btagBBTXbb_morph_5wp`, with the working points 3, 4, 5 (TXbb > 0.95) merged into one bin
#    (`txbb_cdf_max_wp` = 3, applied to that column in linear mode): the subleading index is 1, 2, 3;
#  - the leading Higgs TXbb left untouched and not merged: the DNN trains on working-point indices only, the nominal leading index
#    `HiggsLeading btagBBTXbb_5wp` (1..5 after the cut below) and the merged linear subleading index (1..3);
#  - the control regions A, B restricted to the events whose nominal leading working-point index is not below the (merged, linear) subleading
#    index (`cut_sub_above_lead`). In C, D the leading index is never below the subleading one, so they are not cut.
# The category cuts (A/B on the continuous `btagBBTXbb_morph`, C/D on the nominal TXbb) are the same partition as the index.
# No model: the morphing DNN is trained on this output afterwards. Run it with the template VBF_HH4b_boosted_config_sf.py, which applies
# the TXbb working-point scale factor (MC only). There is no signal-vs-background DNN and no signal category.
onnx_model_dict  |= {
    # no models in the initial run
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
    "region_setting": "50GeV",  # This defines the size of the signal and control regions in the boosted topology
    "lead_sublead": "lead",
    # No signal categories in this chain (no sig/bkg DNN yet)
    "optimised_signal_cuts": False,
    # LINEAR rescaling: no quantile map (the column btagBBTXbb_CDF_rescaled_WPs is not defined)
    "txbb_cdf_wp_cuts": "",
    # Merge the working points 3, 4, 5 (TXbb > 0.95) into the index 3 of btagBBTXbb_morph_5wp: the subleading index is 1 (0.65-0.8), 2 (0.8-0.95), 3 (> 0.95)
    "txbb_cdf_max_wp": 3,
    # Remove, in A and B, the events with nominal leading working-point index < (merged) linear subleading index
    "cut_sub_above_lead": True,
    "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/zhiheng/HH4b/boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_linear_merged3_leadge_initial",
} | onnx_model_dict
