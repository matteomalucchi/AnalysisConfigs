import awkward as ak
import numpy as np
import copy

from utils_configs.basic_functions import add_fields
from utils_configs.jet_algorithms import get_btag_working_points


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
        jets_btag_order.btagB[:, 0] + jets_btag_order.btagB[:, 1]
    ) / 2 > params["mean_pnet_jet"]

    mask_btag = ak.where(ak.is_none(mask_btag), False, mask_btag)

    mask = mask_pt & mask_btag

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def get_hh4b_presel_era(year, sample, isMC):
    """Return the name of the resolved preselection to apply to a chunk.

    The preselection depends on the data-taking period and, for the 2023
    preBPix data, on the primary dataset (JetMET or ParkingHH):

    |      | 2022 | 2023 preBPix JetMET | 2023 preBPix ParkingHH | 2023 postBPix | 2024 |
    | ---- | ---- | ------------------- | ---------------------- | ------------- | ---- |
    | MC   | A    | /                   | B                      | B             | C    |
    | DATA | A    | A                   | B                      | B             | C    |

    The primary dataset is read from the sample name: data samples whose name
    contains "ParkingHH" are ParkingHH, all the other data samples are JetMET.
    """
    if year in ["2022_preEE", "2022_postEE"]:
        return "A"
    if year == "2023_preBPix":
        if not isMC and "ParkingHH" not in sample:
            return "A"
        return "B"
    if year == "2023_postBPix":
        return "B"
    if year == "2024":
        return "C"
    raise ValueError(f"No resolved preselection is defined for year {year}.")


def hh4b_presel_era_dependent_cuts(
    events, params, year, sample, isMC, processor_params, **kwargs
):
    """Resolved preselection chosen per chunk from its year and primary dataset.

    `params["presel_by_era"]` holds the cuts of each preselection (see
    `get_hh4b_presel_era`). Each one requires >= 4 jets, no leptons, a
    minimum pt on the 4 leading jets and one of the two b-tag requirements:
      - "mean_btag": the mean score of the 2 highest-score jets is above
        `mean_btag_jet`
      - "nbtag_wp": at least `nbtag` jets pass the `btag_wp` working point
    computed with the b-tag discriminant `btag_tagger`.
    """
    presel = params["presel_by_era"][get_hh4b_presel_era(year, sample, isMC)]
    pt_type = params["pt_type"]
    tagger = presel["btag_tagger"]

    at_least_four_jets = mask_num_jets(events, params, **kwargs)
    lepton_veto_mask = lepton_veto(events, params, **kwargs)

    mask_4jet_nolep = at_least_four_jets & lepton_veto_mask
    # convert false to None
    mask_4jet_nolep_none = ak.mask(mask_4jet_nolep, mask_4jet_nolep)

    jets = (
        copy.copy(events[mask_4jet_nolep_none].JetGood)
        if not params["tight_cuts"]
        else copy.copy(events[mask_4jet_nolep_none].JetGoodHiggs)
    )
    if tagger not in jets.fields:
        raise ValueError(
            f"The b-tag discriminant '{tagger}' of the preselection for year "
            f"{year} is not available in the jets. Available fields: {jets.fields}."
        )

    jets_pt_order = jets[ak.argsort(jets[pt_type], axis=1, ascending=False)]

    mask_pt_none = ak.ones_like(mask_4jet_nolep_none)
    for i, pt_cut in enumerate(presel["pt_jets"]):
        mask_pt_none = mask_pt_none & (jets_pt_order[pt_type][:, i] > pt_cut)
    # convert none to false
    mask_pt = ak.where(ak.is_none(mask_pt_none), False, mask_pt_none)

    if presel["btag_type"] == "mean_btag":
        btag_scores = ak.sort(jets[tagger], axis=1, ascending=False)
        mask_btag = (btag_scores[:, 0] + btag_scores[:, 1]) / 2 > presel[
            "mean_btag_jet"
        ]
    elif presel["btag_type"] == "nbtag_wp":
        threshold = get_btag_working_points(processor_params, year, tagger)[
            presel["btag_wp"]
        ]
        mask_btag = ak.sum(jets[tagger] > threshold, axis=1) >= presel["nbtag"]
    else:
        raise ValueError(f"Unknown preselection b-tag type {presel['btag_type']}.")

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
            (lead_jet["btagBBTXbb"] >= params["bbtagTXbb"]) & (sublead_jet["btagBBTXbb"] >= params["bbtagTXbb"]) # | (lead_jet["btagBBPNetLegacy"] >= params["bbtagTXbb"])
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


def hh4b_boosted_inclusive(events, params, **kwargs):
    cat_mask = hh4b_boosted_category_1(events, params) | hh4b_boosted_category_vbf(events, params) | hh4b_boosted_category_2(events, params) | hh4b_boosted_category_3(events, params)
    return cat_mask


def hh4b_boosted_TXbb(events, params, **kwargs):
    txbb = events["HiggsSubLeading"]["btagBBTXbb"]
    cat_mask = (txbb > params["threshold"])
    if params["signal"]:
        return cat_mask
    else:
        return ~cat_mask


# TXbb working points (VL, L, M, T, XT, as in `generate_btag_workingpoints`) and the upper edge.
# The category suffix `txbbwp<i>` selects the bin (TXBB_WP_EDGES[i], TXBB_WP_EDGES[i + 1]].
TXBB_WP_EDGES = [0.65, 0.8, 0.95, 0.975, 0.99, 1.0]


def txbb_wp_index(edge):
    """Position k of a TXbb cut value among the working points TXBB_WP_EDGES[:-1] (0.65 -> 0, 0.8 -> 1, ...).

    Raises a ValueError if the value is not a working point: the CDF-rescaled TXbb is a working-point index,
    so it can only express cuts on the working points.
    """
    for k, wp in enumerate(TXBB_WP_EDGES[:-1]):
        if abs(edge - wp) < 1e-9:
            return k
    raise ValueError(
        f"TXbb cut {edge} is not a working point. With the CDF-rescaled TXbb working points (`txbb_cdf_wp_cuts`) the "
        f"cuts must be one of {TXBB_WP_EDGES[:-1]} (see `--txbb-wps` of scripts/optimise_boosted_categories.py)"
    )


class _SubleadingTXbb:
    """Cuts on the TXbb of the subleading Higgs, the same call in all regions.

    - `params["txbb_cdf"]`: the CDF-rescaled working-point index `n` (`btagBBTXbb_CDF_rescaled_WPs`, number of the
      working points TXBB_WP_EDGES[:-1] exceeded, `old_wp_def=False`). It is the nominal index in C and D and the
      quantile-mapped one in A and B, so no shape of the TXbb is used. `gt(WP_k)` is `n > k`, `lt(WP_k)` is `n <= k`
      (this differs from `txbb < WP_k` only for a TXbb exactly equal to the working point).
    - otherwise the continuous TXbb: the linear morph (`btagBBTXbb_morph`) if `params["txbb_morph"]`
      (control regions A, B), the nominal one if not.
    """

    def __init__(self, events, params):
        self.cdf = bool(params.get("txbb_cdf", False))
        higgs = events["HiggsSubLeading"]
        if self.cdf:
            self.values = higgs["btagBBTXbb_CDF_rescaled_WPs"]
        elif params["txbb_morph"]:
            self.values = higgs["btagBBTXbb_morph"]
        else:
            self.values = higgs["btagBBTXbb"]

    def gt(self, edge):
        if self.cdf:
            return self.values > txbb_wp_index(edge)
        return self.values > edge

    def lt(self, edge):
        if self.cdf:
            return self.values <= txbb_wp_index(edge)
        return self.values < edge


def hh4b_boosted_TXbb_bin(events, params, **kwargs):
    # Same strict ">" convention as the working points: a jet is in the bin (low, high] if it passes the WP `low`
    # but not the WP `high`. The morphed TXbb is used in the control regions (A, B), unless the CDF-rescaled working
    # points are used (`txbb_cdf`): then the bin (TXBB_WP_EDGES[i], TXBB_WP_EDGES[i + 1]] is the working-point index i + 1
    # in all regions.
    if params.get("txbb_cdf", False):
        i = txbb_wp_index(params["low"])
        if abs(params["high"] - TXBB_WP_EDGES[i + 1]) > 1e-9:
            raise ValueError(f"TXbb bin ({params['low']}, {params['high']}] is not between two neighbouring working points")
        return events["HiggsSubLeading"]["btagBBTXbb_CDF_rescaled_WPs"] == i + 1
    if params["morph"]:
        txbb = events["HiggsSubLeading"]["btagBBTXbb_morph"]
    else:
        txbb = events["HiggsSubLeading"]["btagBBTXbb"]
    return (txbb > params["low"]) & (txbb <= params["high"])


def hh4b_boosted_mass(events, params, **kwargs):
    if params["higgs"] == "lead":
        mass = events["HiggsLeading"]["mass"]
    else:
        mass = events["HiggsSubLeading"]["mass"]
    cat_mask = (mass > params["lower"]) & (mass < params["upper"])
    if params["full_sideband"]:
        cat_mask_outer = (mass > (params["lower"] - params["sideband_size"])) & (mass < (params["upper"] + params["sideband_size"]))
    else:
        cat_mask_outer = (mass > params["lower"] - params["sideband_size"]) & (mass < params["lower"])
    if params["signal"]:
        return cat_mask
    else:
        return ~cat_mask & cat_mask_outer


def hh4b_boosted_category_1(events, params, **kwargs):
    txbb = _SubleadingTXbb(events, params)
    bdt_ggf = events[params["score_ggf"]]
    # cat_mask = (txbb > 0.945) & (bdt_ggf > 0.94)
    cat_mask = txbb.gt(params["txbb_c1"]) & (bdt_ggf > params["bdt_ggf_c1"])
    return cat_mask


def hh4b_boosted_category_vbf(events, params, **kwargs):
    txbb = _SubleadingTXbb(events, params)
    bdt_vbf = events[params["score_vbf"]]
    cat_mask = ~hh4b_boosted_category_1(events, params) & txbb.gt(params["txbb_vbf"]) & (bdt_vbf > params["bdt_vbf"])
    return cat_mask


def hh4b_boosted_category_2(events, params, **kwargs):
    txbb = _SubleadingTXbb(events, params)
    bdt_ggf = events[params["score_ggf"]]
    cat_mask_1 = (
            ~hh4b_boosted_category_1(events, params) & 
            ~hh4b_boosted_category_vbf(events, params) &
            txbb.gt(params["txbb_c2a"]) & txbb.lt(params["txbb_c1"]) & (bdt_ggf > params["bdt_ggf_c1"])
            )
    cat_mask_2 = (
            ~hh4b_boosted_category_1(events, params) &
            ~hh4b_boosted_category_vbf(events, params) &
            txbb.gt(params["txbb_c1"]) & (bdt_ggf > params["bdt_ggf_c2b"]) & (bdt_ggf < params["bdt_ggf_c1"])
            )
    cat_mask = cat_mask_1 | cat_mask_2
    return cat_mask


def hh4b_boosted_category_3(events, params, **kwargs):
    txbb = _SubleadingTXbb(events, params)
    bdt_ggf = events[params["score_ggf"]]
    cat_mask = (
            ~hh4b_boosted_category_1(events, params) &
            ~hh4b_boosted_category_vbf(events, params) &
            ~hh4b_boosted_category_2(events, params) &
            txbb.gt(params["txbb_c2a"]) & (bdt_ggf > params["bdt_ggf_bkg"])
            )
    return cat_mask


def hh4b_boosted_background_vbf(events, params, **kwargs):
    txbb = _SubleadingTXbb(events, params)
    bdt_vbf = events[params["score_vbf"]]
    cat_mask = (
            txbb.lt(params["txbb_c2a"]) & (bdt_vbf > params["bdt_ggf_bkg"])
            )
    return cat_mask


def hh4b_boosted_background_ggf(events, params, **kwargs):
    txbb = _SubleadingTXbb(events, params)
    bdt_ggf = events[params["score_ggf"]]
    cat_mask = (
            txbb.lt(params["txbb_c2a"]) & (bdt_ggf > params["bdt_ggf_bkg"])
            )
    return cat_mask


def hh4b_2b_cuts(events, params, **kwargs):
    at_least_four_jets = mask_num_jets(events, {"njet": 4}, **kwargs)
    # convert false to None
    at_least_four_jets_none = ak.mask(at_least_four_jets, at_least_four_jets)

    jets_btag_order = events.JetGoodHiggs[at_least_four_jets_none]

    mask = (jets_btag_order.btagB[:, 2] < params["third_pnet_jet"]) & (
        jets_btag_order.btagB[:, 3] < params["fourth_pnet_jet"]
    )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_4b_cuts(events, params, **kwargs):
    at_least_four_jets = mask_num_jets(events, {"njet": 4}, **kwargs)
    # convert false to None
    at_least_four_jets_none = ak.mask(at_least_four_jets, at_least_four_jets)

    jets_btag_order = events.JetGoodHiggs[at_least_four_jets_none]

    mask = (jets_btag_order.btagB[:, 2] > params["third_pnet_jet"]) & (
        jets_btag_order.btagB[:, 3] > params["fourth_pnet_jet"]
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

def zz4b_cuts(events, params, **kwargs):
    '''
    Check wether the event from the inclusive ZZ sample has two Z boson decaying to 2 b quarks. This is done by checking the GenPart collection for a Z bosons (pdgId == 23) and then checking if there are 4 b quarks (pdgId == 5) as daughters.
    '''
    # check that the sample is a ZZ sample, which is inclusive
    if not "ZZ" in events.metadata["dataset"]:
        return ak.ones_like(events.event, dtype=bool)
    else:
        genpart = copy.copy(events["GenPart"])
        mother_pdg = genpart.pdgId[genpart.genPartIdxMother]

        is_b_from_Z = (abs(genpart.pdgId) == 5) & (mother_pdg == 23)
        has_4_bs = ak.sum(is_b_from_Z, axis=1) >= 4

        # Pad None values with False
        return ak.where(ak.is_none(has_4_bs), False, has_4_bs)

def xx4b_cuts(events, params, **kwargs):
    '''
    Check wether the event from the inclusive XX sample has two X boson decaying to 2 b quarks. This is done by checking the GenPart collection for a X bosons (pdgId == 23 or 25) and then checking if there are 4 b quarks (pdgId == 5) as daughters.
    '''
    # check that the sample is a XX sample,
    if (("ZZ" not in events.metadata["dataset"]) and ("ZH" not in events.metadata["dataset"]) and ("HH" not in events.metadata["dataset"])) or (events.metadata["isMC"] == "False"):
        return ak.ones_like(events.event, dtype=bool)
    else:
        genpart = copy.copy(events["GenPart"])
        # genPartIdxMother is -1 for the particles without a stored mother:
        # awkward reads it as a negative index and silently wraps around to the
        # last GenPart of the event, so those particles are masked out
        has_mother = genpart.genPartIdxMother >= 0
        mother_pdg = genpart.pdgId[ak.where(has_mother, genpart.genPartIdxMother, 0)]

        is_b_from_X = (
            has_mother
            & (abs(genpart.pdgId) == 5)
            & ((mother_pdg == 23) | (mother_pdg == 25))
        )
        has_4_bs = ak.sum(is_b_from_X, axis=1) >= 4

        # Pad None values with False
        return ak.where(ak.is_none(has_4_bs), False, has_4_bs)
