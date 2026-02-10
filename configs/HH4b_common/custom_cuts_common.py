from . import custom_cut_functions_common as cuts_f
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
    get_L1sel,
    goldenJson,
    eventFlags,
    get_nPVgood,
)
from utils.custom_cut_functions import get_custom_JetVetoMap_Mask

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
        "Run2": False,
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
        "Run2": False,
        "radius_min": 30,
        "radius_max": 55,
        "higgs_lead_center": 125,
        "higgs_sublead_center": 120,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

hh4b_signal_region_run2 = Cut(
    name="hh4b_signal_region_run2",
    params={
        "Run2": True,
        "radius_min": 0,
        "radius_max": 30,
        "higgs_lead_center": 125,
        "higgs_sublead_center": 120,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

hh4b_control_region_run2 = Cut(
    name="hh4b_control_region_run2",
    params={
        "Run2": True,
        "radius_min": 30,
        "radius_max": 55,
        "higgs_lead_center": 125,
        "higgs_sublead_center": 120,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

hh4b_VR1_signal_region = Cut(
    name="hh4b_VR1_signal_region",
    params={
        "Run2": False,
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
        "Run2": False,
        "radius_min": 30,
        "radius_max": 55,
        "higgs_lead_center": 185,
        "higgs_sublead_center": 180,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

hh4b_VR1_signal_region_run2 = Cut(
    name="hh4b_VR1_signal_region_run2",
    params={
        "Run2": True,
        "radius_min": 0,
        "radius_max": 30,
        "higgs_lead_center": 185,
        "higgs_sublead_center": 180,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

hh4b_VR1_control_region_run2 = Cut(
    name="hh4b_VR1_control_region_run2",
    params={
        "Run2": True,
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

blindedRun2 = Cut(
    name="blindedRun2",
    params={
        "score": 0.9,
        "score_variable": "sig_bkg_dnn_scoreRun2",
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

hh4b_vbf_region = Cut(
    name="hh4b_vbf_region",
    params={
        "min_mjj": 400,
        "min_deta": 3.5,
    },
    function=cuts_f.hh4b_vbf_cuts,
)


def skimming_cut_list(configs):
    skimlist = [
        eventFlags,
        goldenJson,
        get_nPVgood(1),
        get_HLTsel(primaryDatasets=["JetMET"]),
    ]
    if not configs["noL1"]:
        skimlist.append(get_L1sel(primaryDatasets=["JetMET"]))
    return skimlist
