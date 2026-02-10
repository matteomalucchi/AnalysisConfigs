from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_input_variables,
    sig_bkg_dnn_input_variables,
    pairing_spanet_btagWP5,
)

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict  |= {
    "spanet": "/work/tharte/datasets/onnx_spanet_models_for_pairing_and_mass_sculpting_studies/spanet_1_14_5_h4b_5jets_ptvary_loose_300_btag_wp_newLeptonVeto_3L1Cut_UpdateJetVetoMap.onnx",
    "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch/out/bkg_reweighting/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_newUpdates_newLeptonVeto_3L1Cut_UpdateJetVetoMap_postEE/best_models/ratio/average_model_from_onnx.onnx",
    "sig_bkg_dnn": "/work/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_newUpdates_newLeptonVeto_3L1Cut_UpdateJetVetoMap_postEE/run100/state_dict/model_best_epoch_31.onnx",
}


config_options_dict |= {
    "dnn_variables": True,
    "run2": False,
    "max_num_jets": 5,
    "sig_bkg_dnn_input_variables": sig_bkg_dnn_input_variables,
    "bkg_morphing_dnn_input_variables": bkg_morphing_dnn_input_variables,
    "fifth_jet": "pt",
    "pad_value": -999.0,
    "add_jet_spanet": True,
    "spanet_input_name": pairing_spanet_btagWP5,
    "qt_postEE": "/work/tharte/datasets/samples_models_with_bkg_reweight/1_14_5_spanet_ptflat_Update_newLeptonVeto_3L1Cut_UpdateJetVetoMap/quantile_transformer/SRSpanet_qt/qt_events_sig_bkg_dnn_score_kl_1.00.pkl",
}| onnx_model_dict
