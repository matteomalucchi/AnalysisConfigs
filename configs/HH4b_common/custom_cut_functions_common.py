import awkward as ak
import numpy as np
import copy

from utils.basic_functions import add_fields


def mask_num_jets(events, params, **kwargs):
    jet_collection = "JetGood"
    mask = events[f"n{jet_collection}"] >= params["njet"]
    return ak.where(ak.is_none(mask), False, mask)


def lepton_veto(events, params, **kwargs):
    no_electron = events.nElectronGood == 0
    no_muon = events.nMuonGood == 0

    mask = no_electron & no_muon

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_presel_cuts(events, params, **kwargs):
    at_least_four_jets = mask_num_jets(events, params, **kwargs)
    pt_type = params["pt_type"]
    lepton_veto_mask = lepton_veto(events, params, **kwargs)

    mask_4jet_nolep = at_least_four_jets & lepton_veto_mask
    # convert false to None
    mask_4jet_nolep_none = ak.mask(mask_4jet_nolep, mask_4jet_nolep)

    jets_btag_order = (
        copy.copy(events[mask_4jet_nolep_none].JetGood)
        if not params["tight_cuts"]
        else copy.copy(events[mask_4jet_nolep_none].JetGoodHiggs)
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
    at_least_four_jets = mask_num_jets(events, {"njet": 4}, **kwargs)
    # convert false to None
    at_least_four_jets_none = ak.mask(at_least_four_jets, at_least_four_jets)

    jets_btag_order = events.JetGoodHiggs[at_least_four_jets_none]

    mask = (jets_btag_order.btagPNetB[:, 2] < params["third_pnet_jet"]) & (
        jets_btag_order.btagPNetB[:, 3] < params["fourth_pnet_jet"]
    )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_4b_cuts(events, params, **kwargs):
    at_least_four_jets = mask_num_jets(events, {"njet": 4}, **kwargs)
    # convert false to None
    at_least_four_jets_none = ak.mask(at_least_four_jets, at_least_four_jets)

    jets_btag_order = events.JetGoodHiggs[at_least_four_jets_none]

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


def hh4b_vbf_eta_mjj_cuts(events, params, **kwargs):
    jet_vbf = copy.copy(events[params["jet_vbf_coll"]])
    # do not count the None values
    mask_num_vbf_jets = ak.count(jet_vbf.pt, axis=1) >= 2
    mask_num_vbf_jets_none = ak.mask(mask_num_vbf_jets, mask_num_vbf_jets)

    jet_vbf = add_fields(
        jet_vbf[mask_num_vbf_jets_none], "all"
    )

    vbf_mjj = (jet_vbf[:, 0] + jet_vbf[:, 1]).mass
    vbf_deta = abs(jet_vbf[:, 0].eta - jet_vbf[:, 1].eta)

    mask = (vbf_mjj > params["min_mjj"]) & (vbf_deta > params["min_deta"])

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_vbf_discriminator_cuts(events, params, **kwargs):
    jet_vbf = copy.copy(events[params["jet_vbf_coll"]])
    # do not count the None values
    mask_num_vbf_jets = ak.count(jet_vbf.pt, axis=1) >= 2
    
    if params["pass"]:
        mask_discriminator = events[params["discriminator"]] >= params["threshold"]
    else:
        mask_discriminator = events[params["discriminator"]] < params["threshold"]
        
    mask = mask_num_vbf_jets & mask_discriminator

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def dhh_cuts(events, params, **kwargs):

    mask = events.delta_dhh > params["delta_dhh_cut"]

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)
