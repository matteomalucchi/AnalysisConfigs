import awkward as ak
import numpy as np
import copy

from utils_configs.basic_functions import add_fields


def mask_num_jets(events, params, **kwargs):
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


def hh4b_boosted_presel_cuts(events, params, **kwargs):
    # selection is already performed in the object preselection, 
    # here I just need to count the fat jets
    mask_fatjet = events["nFatJetGoodSelected"] >= params["nfatjet"]
    # Following the cuts of the other group:
    # lepton_veto_mask = lepton_veto(events, params, **kwargs)

    mask = mask_fatjet # & lepton_veto_mask

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

def hh4b_boosted_2fatjets(events, params, **kwargs):
    # selection is already performed in the object preselection, 
    # here I just need to count the fat jets
    # mask_fatjet = ak.num(events["FatJetGood"]) >= params["nfatjet"]
    mask_fatjet = events["nFatJetGoodSelected"] >= params["nfatjet"]
    # Following the cuts of the other group:
    # lepton_veto_mask = lepton_veto(events, params, **kwargs)

    mask = mask_fatjet # & lepton_veto_mask

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

def hh4b_boosted_lepton_veto(events, params, **kwargs):
    mask = lepton_veto(events, params, **kwargs)
    return ak.where(ak.is_none(mask), False, mask)

def hh4b_boosted_SR_cuts(events, params, **kwargs):
    # further splits after passing the boosted preselection, here I assume that the two candidate jets are present
    lead_jet, sublead_jet = events["FatJetGoodSelected"][:, 0], events["FatJetGoodSelected"][:, 1]

    # also the second jet has to pass the btag cut to end in the SR
    if "pnet_cut" in params:
        mask_btag = (
            sublead_jet["btagBB"] > params["pnet_cut"]
        )
    elif "bbtagTXbb" in params:
        mask_btag = (
            (lead_jet["btagBBTXbb"] >= params["bbtagTXbb"]) # | (lead_jet["btagBBPNetLegacy"] >= params["bbtagTXbb"])
        )
    mask_btag = ak.where(ak.is_none(mask_btag), False, mask_btag)

    # this should be done with the regressed mass, GloParT or PNet? at the moment is PNet
    if "mass_min" in params and "mass_max" in params:
        mask_mass = (
            (lead_jet.mass_regr > params["mass_min"]) 
            & (lead_jet.mass_regr < params["mass_max"])
        )
        mask_mass = ak.where(ak.is_none(mask_mass), False, mask_mass)
    else:
        mask_mass = ak.full_like(mask_btag, True)

    mask = mask_btag  # & mask_mass

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

    # the leading jet has to be in the range 50 < m < 100 GeV or the subleading jet has to fail the btag cut
    mask_mass_lead = (
        (lead_jet.mass_regr > params["mass_min"]) 
        & (lead_jet.mass_regr < params["mass_max"])
    )
    mask_mass_lead = ak.where(ak.is_none(mask_mass_lead), False, mask_mass_lead)

    mask_mass_qcd = (
        (lead_jet.mass_regr < params["mass_max"])
        & (sublead_jet.mass_regr < params["mass_max_sublead"])
    )
    mask_mass_qcd = ak.where(ak.is_none(mask_mass_qcd), False, mask_mass_qcd)

    mask_btag_sublead = (sublead_jet["btagBB"] > params["pnet_cut"])
    mask_btag_sublead = ak.where(ak.is_none(mask_btag_sublead), False, mask_btag_sublead)

    mask = (~(mask_mass_lead) | ~(mask_btag_sublead)) & mask_mass_qcd

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_boosted_qcd_CR_cuts_X(events, params, **kwargs):
    # further splits after passing the boosted preselection, here I assume that the two candidate jets are present
    lead_jet, sublead_jet = events["FatJetGoodSelected"][:, 0], events["FatJetGoodSelected"][:, 1]

    # the leading jet has to be in the range 50 < m < 100 GeV or the subleading jet has to fail the btag cut
    mask_mass_lead = (
        (lead_jet.mass_regr > params["mass_min_lead"]) 
        & (lead_jet.mass_regr < params["mass_max_lead"])
    )
    mask_mass_lead = ak.where(ak.is_none(mask_mass_lead), False, mask_mass_lead)

    mask_mass_sublead = (
        (sublead_jet.mass_regr > params["mass_min_sublead"])
        & (sublead_jet.mass_regr < params["mass_max_sublead"])
    )
    mask_mass_sublead = ak.where(ak.is_none(mask_mass_sublead), False, mask_mass_sublead)

    mask_btag_sublead = (
        (sublead_jet["btagBB"] >= params["pnet_cut_min"])
        & (sublead_jet["btagBB"] < params["pnet_cut_max"])
    )
    mask_btag_sublead = ak.where(ak.is_none(mask_btag_sublead), False, mask_btag_sublead)

    mask = (mask_mass_lead) & (mask_btag_sublead) & (mask_mass_sublead)

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

def hh4b_boosted_vbf_cuts(events, params, **kwargs):
    # candidate VBF jets are already selected in the object preselection and stored in the nDiJetVBFCandidates
    mask_vbf = (events.nDiJetVBFCandidates > 0)

    # Pad None values with False
    return ak.where(ak.is_none(mask_vbf), False, mask_vbf)


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

def sig_bkg_score_cut(events, params, **kwargs):
    mask = events[params["discriminator"]] >= params["threshold"]

    return ak.where(ak.is_none(mask), False, mask)

def hh4b_vbf_discriminator_cuts(events, params, **kwargs):
    if params["pass"]:
        mask_discriminator = events[params["discriminator"]] >= params["threshold"]
    else:
        mask_discriminator = events[params["discriminator"]] < params["threshold"]

    mask = mask_discriminator

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

def hh4b_vbf_2_jets(events, params, **kwargs):
    jet_vbf = copy.copy(events[params["jet_vbf_coll"]])
    # do not count the None values
    mask_num_vbf_jets = ak.count(jet_vbf.pt, axis=1) >= 2
    mask = mask_num_vbf_jets

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

def dhh_cuts(events, params, **kwargs):

    mask = events.delta_dhh > params["delta_dhh_cut"]

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

def xx4b_cuts(events, params, **kwargs):
    '''
    Check whether the event of an inclusive XX sample has two X bosons decaying to 2 b quarks.
    This is done by checking the GenPart collection for X bosons (pdgId == 23 or 25) and then
    checking if there are 4 b quarks (pdgId == 5) as daughters.
    '''
    # data (and MC without a GenPart collection) is left untouched
    if str(events.metadata.get("isMC", "True")) not in ("True", "true", "1"):
        return ak.ones_like(events.event, dtype=bool)
    if "GenPart" not in events.fields:
        return ak.ones_like(events.event, dtype=bool)
    # only the samples that are inclusive in the decay channel need this cut
    dataset = events.metadata["dataset"]
    if not any(tag in dataset for tag in ("ZZ", "ZH", "HH")):
        return ak.ones_like(events.event, dtype=bool)

    genpart = events["GenPart"]
    # genPartIdxMother is -1 for particles without a mother: indexing with it
    # would silently wrap around to the last GenPart, so mask those out
    has_mother = genpart.genPartIdxMother >= 0
    mother_idx = ak.where(has_mother, genpart.genPartIdxMother, 0)
    mother_pdg = genpart.pdgId[mother_idx]

    is_b_from_X = (
        has_mother
        & (abs(genpart.pdgId) == 5)
        & ((mother_pdg == 23) | (mother_pdg == 25))
    )
    has_4_bs = ak.sum(is_b_from_X, axis=1) >= 4

    # Pad None values with False
    return ak.where(ak.is_none(has_4_bs), False, has_4_bs)
