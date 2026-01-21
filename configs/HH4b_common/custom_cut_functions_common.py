import awkward as ak
import numpy as np


def four_jets(events, params, **kwargs):
    jet_collection = "JetGood"
    mask = events[f"n{jet_collection}"] >= params["njet"]
    return ak.where(ak.is_none(mask), False, mask)


def two_fat_jets(events, params, **kwargs):
    jet_collection = "FatJetGood"
    mask = events[f"n{jet_collection}"] >= params["nfatjet"]
    return ak.where(ak.is_none(mask), False, mask)


def lepton_veto(events, params, **kwargs):
    no_electron = events.nElectronGood == 0
    no_muon = events.nMuonGood == 0

    mask = no_electron & no_muon

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_presel_cuts(events, params, **kwargs):
    at_least_four_jets = four_jets(events, params, **kwargs)
    pt_type = params["pt_type"]
    lepton_veto_mask = lepton_veto(events, params, **kwargs)

    mask_4jet_nolep = at_least_four_jets & lepton_veto_mask
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


def hh4b_boosted_presel_cuts(events, params, **kwargs):
    # selection is already performed in the object preselection, 
    # here I just need to count the fat jets
    mask = events["nFatJetGoodSelected"] >= params["nfatjet"]

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_boosted_SR_cuts(events, params, **kwargs):
    # further splits after passing the boosted preselection, here I assume that the two candidate jets are present
    lead_jet, sublead_jet = events["FatJetGoodSelected"][:, 0], events["FatJetGoodSelected"][:, 1]

    # also the second jet has to pass the btag cut to end in the SR
    mask_btag = (
        sublead_jet["btagBB"] > params["pnet_cut"]
    )
    mask_btag = ak.where(ak.is_none(mask_btag), False, mask_btag)

    # this should be done with the regressed mass, GloParT or PNet? at the moment is PNet
    mask_mass = (
        (lead_jet.mass_regr > params["mass_min"]) 
        & (lead_jet.mass_regr < params["mass_max"])
    )
    mask_mass = ak.where(ak.is_none(mask_mass), False, mask_mass)

    mask = mask_btag & mask_mass

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_boosted_ttbar_CR_cuts(events, params, **kwargs):
    # further splits after passing the boosted preselection, here I assume that the two candidate jets are present  
    lead_jet, sublead_jet = events["FatJetGoodSelected"][:, 0], events["FatJetGoodSelected"][:, 1]

    # both jets has to be in the 150 < mass < 200 GeV window to be in the ttbar CR 
    mask_mass = (
        (lead_jet.mass_regr > params["mass_min"]) 
        & (lead_jet.mass_regr < params["mass_max"])
        & (sublead_jet.mass_regr > params["mass_min"])
        & (sublead_jet.mass_regr < params["mass_max"])
    )
    mask_mass = ak.where(ak.is_none(mask_mass), False, mask_mass)

    # Pad None values with False
    return ak.where(ak.is_none(mask_mass), False, mask_mass)


def hh4b_boosted_qcd_CR_cuts(events, params, **kwargs):
    # further splits after passing the boosted preselection, here I assume that the two candidate jets are present
    lead_jet, sublead_jet = events["FatJetGoodSelected"][:, 0], events["FatJetGoodSelected"][:, 1]

    # both jets has to be in the 50 < mass < 150 GeV window to be in the QCD CR 
    # the leading one has to be in the range 50 < m < 100 GeV or the subleading jet has to fail the btag cut
    mask_mass_lead = (
        (lead_jet.mass_regr > params["mass_min"]) 
        & (lead_jet.mass_regr < params["mass_max"])
    )
    mask_mass_lead = ak.where(ak.is_none(mask_mass_lead), False, mask_mass_lead)

    mask_mass_qcd = (
        (lead_jet.mass_regr < params["mass_max"])
        & (sublead_jet.mass_regr < params["mass_max"])
    )
    mask_mass_qcd = ak.where(ak.is_none(mask_mass_qcd), False, mask_mass_qcd)

    mask_btag_sublead = (sublead_jet["btagBB"] > params["pnet_cut"])
    mask_btag_sublead = ak.where(ak.is_none(mask_btag_sublead), False, mask_btag_sublead)

    mask = (~(mask_mass_lead) | ~(mask_btag_sublead)) & mask_mass_qcd


def hh4b_boosted_qcd_CR_cuts_X(events, params, **kwargs):
    # further splits after passing the boosted preselection, here I assume that the two candidate jets are present
    lead_jet, sublead_jet = events["FatJetGoodSelected"][:, 0], events["FatJetGoodSelected"][:, 1]

    # both jets has to be in the 50 < mass < 150 GeV window to be in the QCD CR 
    # the leading one has to be in the range 50 < m < 100 GeV or the subleading jet has to fail the btag cut
    mask_mass_lead = (
        (lead_jet.mass_regr > params["mass_min"]) 
        & (lead_jet.mass_regr < params["mass_max"])
    )
    mask_mass_lead = ak.where(ak.is_none(mask_mass_lead), False, mask_mass_lead)

    mask_mass_qcd = (
        (lead_jet.mass_regr < params["mass_max_qcd"])
        & (sublead_jet.mass_regr < params["mass_max_qcd"])
    )
    mask_mass_qcd = ak.where(ak.is_none(mask_mass_qcd), False, mask_mass_qcd)

    mask_btag_sublead = (
        (sublead_jet["btagBB"] >= params["pnet_cut_min"])
        & (sublead_jet["btagBB"] < params["pnet_cut_max"])
    )
    mask_btag_sublead = ak.where(ak.is_none(mask_btag_sublead), False, mask_btag_sublead)

    mask = (mask_mass_lead) & (mask_btag_sublead) & (mask_mass_qcd)

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

def hh4b_boosted_vbf_cuts(events, params, **kwargs):
    # candidate VBF jets are already selected in the object preselection and stored in the nDiJetVBFCandidates
    mask_vbf = (events.nDiJetVBFCandidates > 0)

    # Pad None values with False
    return ak.where(ak.is_none(mask_vbf), False, mask_vbf)


def hh4b_2b_cuts(events, params, **kwargs):
    at_least_four_jets = four_jets(events, {"njet": 4}, **kwargs)
    # convert false to None
    at_least_four_jets_none = ak.mask(at_least_four_jets, at_least_four_jets)
    
    jets_btag_order = events.JetGoodHiggs[at_least_four_jets_none]
    
    mask = (jets_btag_order.btagPNetB[:, 2] < params["third_pnet_jet"]) & (
        jets_btag_order.btagPNetB[:, 3] < params["fourth_pnet_jet"]
    )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_4b_cuts(events, params, **kwargs):
    at_least_four_jets = four_jets(events, {"njet": 4}, **kwargs)
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


def dhh_cuts(events, params, **kwargs):

    mask = events.delta_dhh > params["delta_dhh_cut"]

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)
