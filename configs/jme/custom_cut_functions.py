import awkward as ak
import numpy as np


def ptbin(events, params, **kwargs):
    # Mask to select events in a MatchedJets pt bin
    if params["pt_high"] == "Inf":
        mask = events.MatchedJets.pt > params["pt_low"]
    elif type(params["pt_high"]) != str:
        mask = (events.MatchedJets.JetPtRaw > params["pt_low"]) & (
            events.MatchedJets.JetPtRaw < params["pt_high"]
        )
    else:
        raise NotImplementedError

    assert not ak.any(ak.is_none(mask, axis=1)), f"None in ptbin"

    return mask


def etabin(events, params, **kwargs):
    # Mask to select events in a MatchedJets eta bin
    mask = (events.MatchedJets.eta > params["eta_low"]) & (
        events.MatchedJets.eta < params["eta_high"]
    )

    assert not ak.any(ak.is_none(mask, axis=1)), f"None in etabin"

    return mask


def etabin_neutrino(events, params, **kwargs):
    # Mask to select events in a MatchedJets eta bin
    mask = (events.MatchedJetsNeutrino.eta > params["eta_low"]) & (
        events.MatchedJetsNeutrino.eta < params["eta_high"]
    )

    assert not ak.any(ak.is_none(mask, axis=1)), f"None in etabin"

    return mask


def reco_etabin(events, params, **kwargs):
    # Mask to select events in a MatchedJets eta bin
    mask = (events.MatchedJets.RecoEta > params["eta_low"]) & (
        events.MatchedJets.RecoEta < params["eta_high"]
    )

    assert not ak.any(ak.is_none(mask, axis=1)), f"None in etabin"

    return mask


def reco_neutrino_etabin(events, params, **kwargs):
    # Mask to select events in a MatchedJets eta bin
    mask = (events.MatchedJetsNeutrino.RecoEta > params["eta_low"]) & (
        events.MatchedJetsNeutrino.RecoEta < params["eta_high"]
    )

    assert not ak.any(ak.is_none(mask, axis=1)), f"None in etabin"

    return mask


def reco_neutrino_abs_etabin(events, params, **kwargs):
    # Mask to select events in a MatchedJets eta bin
    mask = (abs(events.MatchedJetsNeutrino.RecoEta) > params["eta_low"]) & (
        abs(events.MatchedJetsNeutrino.RecoEta) < params["eta_high"]
    )

    assert not ak.any(ak.is_none(mask, axis=1)), f"None in etabin"

    return mask


def genjet_selection_flavsplit(events, jet_type, flavs):
    jets = events[jet_type]
    mask_flav = (
        jets.partonFlavour == flavs
        if type(flavs) == int
        else ak.any([jets.partonFlavour == flav for flav in flavs], axis=0)
    )
    # mask_flav = ak.any([jets.partonFlavour == flav for flav in flavs], axis=0)
    mask_flav = ak.mask(mask_flav, mask_flav)
    return jets[mask_flav]


def PV_presel_cuts(events, params, **kwargs):
    mask = abs(events.PV.z - events.GenVtx.z) < params["distance"]
    return ak.where(ak.is_none(mask), False, mask)


def jet_selection_nopu(events, jet_type, params, pt_cut="pt"):
    jets = events[jet_type]
    cuts = params.object_preselection[jet_type]

    mask_jets = (
        (getattr(jets, pt_cut) > cuts[pt_cut])
        & (np.abs(jets.eta) < cuts["eta"])
        & (jets.jetId >= cuts["jetId"])
    )

    return jets[mask_jets]
