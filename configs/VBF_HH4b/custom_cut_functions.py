from collections.abc import Iterable
import awkward as ak


def vbf_hh4b_presel_cuts(events, params, **kwargs):

    at_least_four_jetgood = events.nJetGood >= params["njetgood"]
    at_least_six_jetvbf = events.nJetVBF_generalSelection >= params["njetvbf"]  # HERE
    no_electron = events.nElectronGood == 0
    no_muon = events.nMuonGood == 0
    pt_type = params["pt_type"]

    mask_6jet_nolep = at_least_four_jetgood & no_electron & no_muon & at_least_six_jetvbf

    # convert false to None
    mask_6jet_nolep_none = ak.mask(mask_6jet_nolep, mask_6jet_nolep)
    jets_btag_order = (
        events[mask_6jet_nolep_none].JetGood
        if not params["tight_cuts"]
        else events[mask_6jet_nolep_none].JetGoodHiggs
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
