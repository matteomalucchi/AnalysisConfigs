import awkward as ak
import numpy as np
import copy

from pocket_coffea.lib.cut_functions import get_JetVetoMap_Mask


def four_jets(events, params, **kwargs):
    jet_collection=params["jet_collection"]
    mask = events[f"n{jet_collection}"] >= params["njet"]
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_presel_cuts(events, params, **kwargs):
    at_least_four_jets = events.nJetGood >= params["njet"]
    no_electron = events.nElectronGood == 0
    no_muon = events.nMuonGood == 0
    pt_type = params["pt_type"]

    mask_4jet_nolep = at_least_four_jets & no_electron & no_muon

    # convert false to None
    mask_4jet_nolep_none = ak.mask(mask_4jet_nolep, mask_4jet_nolep)
    jets_btag_order = (
        events[mask_4jet_nolep_none].JetGood
        if not params["tight_cuts"]
        else events[mask_4jet_nolep_none].JetGoodHiggs
    )

    jets_pt_order = jets_btag_order[
        ak.argsort(jets_btag_order[pt_type], axis=1, ascending=False)
    ]

    mask_pt_none = (
        (jets_pt_order[pt_type][:, 0] > params["pt_jet0"])
        & (jets_pt_order[pt_type][:, 1] > params["pt_jet1"])
        & (jets_pt_order[pt_type][:, 2] > params["pt_jet2"])
        & (jets_pt_order[pt_type][:, 3] > params["pt_jet3"])
    )
    # convert none to false
    mask_pt = ak.where(ak.is_none(mask_pt_none), False, mask_pt_none)

    mask_btag = (
        jets_btag_order.btagPNetB[:, 0] + jets_btag_order.btagPNetB[:, 1]
    ) / 2 > params["mean_pnet_jet"]

    mask_btag = ak.where(ak.is_none(mask_btag), False, mask_btag)

    mask = mask_pt & mask_btag

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_2b_cuts(events, params, **kwargs):
    jets_btag_order = events.JetGoodHiggs

    mask = (jets_btag_order.btagPNetB[:, 2] < params["third_pnet_jet"]) & (
        jets_btag_order.btagPNetB[:, 3] < params["fourth_pnet_jet"]
    )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_4b_cuts(events, params, **kwargs):
    jets_btag_order = events.JetGoodHiggs

    mask = (jets_btag_order.btagPNetB[:, 2] > params["third_pnet_jet"]) & (
        jets_btag_order.btagPNetB[:, 3] > params["fourth_pnet_jet"]
    )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_Rhh_cuts(events, params, **kwargs):
    Rhh = None
    if params["Run2"]:
        if "Rhh_Run2" in events.fields:
            Rhh = events.Rhh_Run2
        else:
            higgs_lead_mass = events.HiggsLeadingRun2.mass
            higgs_sublead_mass = events.HiggsSubLeadingRun2.mass
    else:
        if "Rhh" in events.fields:
            Rhh = events.Rhh
        else:
            higgs_lead_mass = events.HiggsLeading.mass
            higgs_sublead_mass = events.HiggsSubLeading.mass

    if Rhh is None:
        Rhh = np.sqrt(
            (higgs_lead_mass - params["higgs_lead_center"]) ** 2
            + (higgs_sublead_mass - params["higgs_sublead_center"]) ** 2
        )

    mask = (Rhh >= params["radius_min"]) & (Rhh < params["radius_max"])

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def blinding_cuts(events, params, **kwargs):
    """
    Function to apply a cut based on the dnn score.
    The idea is, to look at the data in the low score sideband to compare performance.
    """
    mask = events[params["score_variable"]] < params["score"]

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def dhh_cuts(events, params, **kwargs):

    mask = events.delta_dhh > params["delta_dhh_cut"]

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def get_hh4b_JetVetoMap_Mask(events, params, **kwargs):
    """Function to get the JetVetoMap mask for HH->4b analysis"""
    events_copy = copy.copy(events)
    events_copy["Jet"] = ak.with_field(
        events_copy["Jet"],
        events_copy["Jet"][params["pt_type"]],
        "pt",
    )

    mask = get_JetVetoMap_Mask(events_copy, params, **kwargs)

    return mask
