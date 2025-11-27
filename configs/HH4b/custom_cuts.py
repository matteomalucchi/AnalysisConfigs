from collections.abc import Iterable
import numpy as np
import awkward as ak

from pocket_coffea.lib.cut_definition import Cut
import configs.HH4b_common.custom_cut_functions_common as cuts_f

## Cuts to build the full cutflow for HH->4b analysis up to the signal region
four_jets_cut = Cut(
    name="four_jets_cut",
    params={
        "njet": 4,
        "jet_collection":"JetGood"
    },
    function=cuts_f.four_jets,
)

lepton_veto = Cut(
    name="lepton_veto",
    params={
        "njet": -999,
        "pt_jet0": -999,
        "pt_jet1": -999,
        "pt_jet2": -999,
        "pt_jet3": -999,
        "mean_pnet_jet": -999,
        "tight_cuts": False,
        "pt_type": "pt_default",
    },
    function=cuts_f.hh4b_presel_cuts,
)

jet_pt_cut = Cut(
    name="jet_pt_cut",
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

two_b_cut = Cut(
    name="two_b_cut",
    params={
        "njet": 4,
        "pt_jet0": -999,
        "pt_jet1": -999,
        "pt_jet2": -999,
        "pt_jet3": -999,
        "mean_pnet_jet": 0.65,
        "tight_cuts": False,
        "pt_type": "pt_default",
    },
    function=cuts_f.hh4b_presel_cuts,
)

third_btag_cut = Cut(
    name="third_btag_cut",
    params={
        "third_pnet_jet": 0.2605,
        "fourth_pnet_jet": -999,
    },
    function=cuts_f.hh4b_4b_cuts,
)

fourth_btag_cut = Cut(
    name="fourth_btag_cut",
    params={
        "third_pnet_jet": -999,
        "fourth_pnet_jet": 0.2605,
    },
    function=cuts_f.hh4b_4b_cuts,
)

signal_region_cut = Cut(
    name="signal_region_cut",
    params={
        "Run2": False,
        "radius_min": 0,
        "radius_max": 30,
        "higgs_lead_center": 125,
        "higgs_sublead_center": 120,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

signal_region_Run2_cut = Cut(
    name="signal_region_Run2_cut",
    params={
        "Run2": True,
        "radius_min": 0,
        "radius_max": 30,
        "higgs_lead_center": 125,
        "higgs_sublead_center": 120,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)


# Dhh cut above 30 GeV
dhh_above_30 = Cut(
    name="hh4b",
    params={
        "delta_dhh_cut": 30,
    },
    function=cuts_f.dhh_cuts,
)
