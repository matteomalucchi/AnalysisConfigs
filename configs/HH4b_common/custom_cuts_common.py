from . import custom_cut_functions_common as cuts_f
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
    get_L1sel,
    goldenJson,
    eventFlags,
    get_nPVgood,
)
from utils_configs.custom_cut_functions import get_custom_JetVetoMap_Mask

hh4b_presel = Cut(
    name="hh4b_presel",
    params={
        "njet": 4,
        "pt_jet0": 80,
        "pt_jet1": 60,
        "pt_jet2": 45,
        "pt_jet3": 35,
        "mean_pnet_jet": 0.65,
        "tight_cuts": False,
        "pt_type": "pt_default",
    },
    function=cuts_f.hh4b_presel_cuts,
)

hh4b_boosted_presel = Cut(
    name="hh4b_boosted_presel",
    params={
        "nfatjet": 2,
    },
    function=cuts_f.hh4b_boosted_presel_cuts,
)

hh4b_presel_tight = Cut(
    name="hh4b_presel_tight",
    params={
        "njet": 4,
        "pt_jet0": 80,
        "pt_jet1": 60,
        "pt_jet2": 45,
        "pt_jet3": 35,
        "mean_pnet_jet": 0.65,
        "tight_cuts": True,
        "pt_type": "pt_default",
    },
    function=cuts_f.hh4b_presel_cuts,
)

hh4b_presel_nobtag = Cut(
    name="hh4b_presel_nobtag",
    params={
        "njet": 4,
        "pt_jet0": 80,
        "pt_jet1": 60,
        "pt_jet2": 45,
        "pt_jet3": 35,
        "mean_pnet_jet": -999,
        "tight_cuts": False,
        "pt_type": "pt_default",
    },
    function=cuts_f.hh4b_presel_cuts,
)

hh4b_presel_parking = Cut(
    name="hh4b_presel_parking",
    params={
        "njet": 4,
        "pt_jet0": 35,
        "pt_jet1": 35,
        "pt_jet2": 35,
        "pt_jet3": 30,
        "mean_pnet_jet": 0.55,
        "tight_cuts": False,
        "pt_type": "pt_default",
    },
    function=cuts_f.hh4b_presel_cuts,
)


hh4b_2b_region = Cut(
    name="hh4b_2b_region",
    params={
        "third_pnet_jet": 0.2605,
        "fourth_pnet_jet": 0.2605,
    },
    function=cuts_f.hh4b_2b_cuts,
)
hh4b_4b_region = Cut(
    name="hh4b_4b_region",
    params={
        "third_pnet_jet": 0.2605,
        "fourth_pnet_jet": 0.2605,
    },
    function=cuts_f.hh4b_4b_cuts,
)

hh4b_signal_region = Cut(
    name="hh4b_signal_region",
    params={
        "radius_min": 0,
        "radius_max": 30,
        "higgs_lead_center": 125,
        "higgs_sublead_center": 120,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

hh4b_control_region = Cut(
    name="hh4b_control_region",
    params={
        "radius_min": 30,
        "radius_max": 55,
        "higgs_lead_center": 125,
        "higgs_sublead_center": 120,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

hh4b_boosted_signal_region = Cut(
    name="hh4b_boosted_signal_region",
    params={
        "pnet_cut": 0.65,
        "mass_min": 100,
        "mass_max": 150,
    },
    function=cuts_f.hh4b_boosted_SR_cuts,
)

hh4b_boosted_ttbar_control_region = Cut(
    name="hh4b_boosted_ttbar_control_region",
    params={
        "mass_min": 150,
        "mass_max": 200,
    },
    function=cuts_f.hh4b_boosted_ttbar_CR_cuts,
)

hh4b_boosted_qcd_control_region_tot = Cut(
    name="hh4b_boosted_qcd_control_region_tot",
    params={
        "pnet_cut": 0.65,
        "mass_min": 100,
        "mass_max": 150,
        "mass_max_sublead": 200,
    },
    function=cuts_f.hh4b_boosted_qcd_CR_cuts,
)

hh4b_boosted_qcd_control_region_A = Cut(
    name="hh4b_boosted_qcd_control_region_A",
    params={
        "pnet_cut_min": 0.05,
        "pnet_cut_max": 0.65,
        "mass_min_lead": 50,
        "mass_max_lead": 100,
        "mass_min_sublead": 50,
        "mass_max_sublead": 200,
    },
    function=cuts_f.hh4b_boosted_qcd_CR_cuts_X,
)

hh4b_boosted_qcd_control_region_B = Cut(
    name="hh4b_boosted_qcd_control_region_B",
    params={
        "pnet_cut_min": 0.65,
        "pnet_cut_max": 1.0,
        "mass_min_lead": 50,
        "mass_max_lead": 100,
        "mass_min_sublead": 50,
        "mass_max_sublead": 200,
    },
    function=cuts_f.hh4b_boosted_qcd_CR_cuts_X,
)

hh4b_boosted_qcd_control_region_C = Cut(
    name="hh4b_boosted_qcd_control_region_C",
    params={
        "pnet_cut_min": 0.05,
        "pnet_cut_max": 0.65,
        "mass_min_lead": 100,
        "mass_max_lead": 150,
        "mass_min_sublead": 50,
        "mass_max_sublead": 200,
    },
    function=cuts_f.hh4b_boosted_qcd_CR_cuts_X,
)

# hh4b_boosted_vbf_region = Cut(
#     name="hh4b_boosted_vbf_region",
#     params={
#         "vbf_pt": 25,
#         "vbf_gap_pt": 50,
#         "vbf_eta": 4.7,
#         "gap_eta_min": 2.5,
#         "gap_eta_max": 3.0,
#         "vbf_mjj": 400,
#         "vbf_delta_eta": 3.5,
#         "tight_cuts": False,
#     },
#     function=cuts_f.hh4b_boosted_vbf_cuts,
# )

hh4b_VR1_signal_region = Cut(
    name="hh4b_VR1_signal_region",
    params={
        "radius_min": 0,
        "radius_max": 30,
        "higgs_lead_center": 185,
        "higgs_sublead_center": 180,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

hh4b_VR1_control_region = Cut(
    name="hh4b_VR1_control_region",
    params={
        "radius_min": 30,
        "radius_max": 55,
        "higgs_lead_center": 185,
        "higgs_sublead_center": 180,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

blinded = Cut(
    name="blinded",
    params={
        "score": 0.9,
        "score_variable": "sig_bkg_dnn_score",
    },
    function=cuts_f.blinding_cuts,
)

hh4b_JetVetoMap = Cut(
    name="hh4b_JetVetoMap",
    params={
        "jet_type": "Jet",
        "pt_type": "pt_default",
    },
    function=get_custom_JetVetoMap_Mask,
)

hh4b_vbf_best_candidates_6_jets_region = Cut(
    name="hh4b_vbf_best_candidates_6_jets_region",
    params={
        "min_mjj": 400,
        "min_deta": 3.5,
        "jet_vbf_coll": "JetGoodVBFEnergyOrdered",
    },
    function=cuts_f.hh4b_vbf_eta_mjj_cuts,
)

hh4b_vbf_best_candidates_6_jets_nokincut_region = Cut(
    name="hh4b_vbf_best_candidates_6_jets_nokincut_region",
    params={
        "min_mjj": 0,
        "min_deta": 0,
        "jet_vbf_coll": "JetGoodVBFEnergyOrdered",
    },
    function=cuts_f.hh4b_vbf_eta_mjj_cuts,
)

hh4b_vbf_pass_discriminator_region = Cut(
    name="hh4b_vbf_pass_discriminator_region",
    params={
        "discriminator": "VBF_ggF_score",
        "pass": True,
        "threshold": 0.95,
    },
    function=cuts_f.hh4b_vbf_discriminator_cuts,
)

hh4b_vbf_fail_discriminator_region = Cut(
    name="hh4b_vbf_fail_discriminator_region",
    params={
        "discriminator": "VBF_ggF_score",
        "pass": False,
        "threshold": 0.95,
    },
    function=cuts_f.hh4b_vbf_discriminator_cuts,
)

hh4b_vbf_2_jets = Cut(
    name="hh4b_vbf_2_jets",
    params={
        "jet_vbf_coll": "JetGoodVBFEnergyOrdered",
    },
    function=cuts_f.hh4b_vbf_2_jets,
)

def skimming_cut_list(configs):
    skimlist = [
        eventFlags,
        goldenJson,
        get_nPVgood(1),
    ]
    if configs["boosted"]:
        skimlist.append(get_HLTsel(primaryDatasets=["Boosted"]))
    else:
        skimlist.append(get_HLTsel(primaryDatasets=["JetMET"]))
    if not configs["noL1"] and not configs["boosted"]:
        skimlist.append(get_L1sel(primaryDatasets=["JetMET"]))
    return skimlist
