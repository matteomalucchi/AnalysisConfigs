"""Observables used to evaluate the per-filter HH4b jet trigger scale factors.

These are specific to the set of trigger scale factors measured for the HH4b jet
trigger (see `scripts/convert_trigger_sf_to_correctionlib.py` and
`configs/HH4b_common/params/trigger_scale_factors.yaml`), not generic PocketCoffea
functionality: PocketCoffea's `pocket_coffea.lib.trigger_sf` only ships the
`register_trigger_sf_variable` mechanism and the `sf_trigger` weight applying
whichever observables are registered here.

Importing this module (and registering it with `cloudpickle.register_pickle_by_value`
so it is available to the workers) is enough to make `jet_pt`, `alljet_ht`,
`calojet_ht` and `atanh_btag_mean` available to the `trigger_scale_factors.yaml`
configuration.
"""

import awkward as ak
import numpy as np

from pocket_coffea.lib.trigger_sf import delta_r_table, register_trigger_sf_variable


def jets_in_ht_acceptance(jets, cfg):
    '''Mask of the jets entering the HT computation'''
    return (jets.pt >= float(cfg.get("pt", 30.0))) & (
        abs(jets.eta) < float(cfg.get("eta", 2.5))
    )


@register_trigger_sf_variable("jet_pt")
def jet_pt(events, cfg):
    '''pt of the `index`-th leading in pt jet (`index` starting from 1).

    Options: `collection` (default `JetGood`), `index` (default 1),
    `pad_value` (default 0., used for events with less than `index` jets).
    '''
    jets = events[cfg.get("collection", "JetGood")]
    index = int(cfg.get("index", 1))
    # The jets are explicitly sorted by pt: the collection may be sorted
    # differently by the analysis (e.g. by b-tagging score).
    pt = jets.pt[ak.argsort(jets.pt, axis=1, ascending=False)]
    return ak.fill_none(
        ak.pad_none(pt, index, axis=1)[:, index - 1], float(cfg.get("pad_value", 0.0))
    )


@register_trigger_sf_variable("alljet_ht")
def alljet_ht(events, cfg):
    '''Scalar sum of the pt of all the jets in the HT acceptance.

    Options: `collection` (default `Jet`), `pt` (default 30.), `eta` (default 2.5).
    '''
    jets = events[cfg.get("collection", "Jet")]
    return ak.sum(ak.where(jets_in_ht_acceptance(jets, cfg), jets.pt, 0.0), axis=1)


@register_trigger_sf_variable("calojet_ht")
def calojet_ht(events, cfg):
    '''Scalar sum of the pt of the jets in the HT acceptance not identified as muons.

    A jet is identified as a muon if it has a small muon and electromagnetic energy
    fraction and it is close to an isolated muon. N.B: following the definition used
    to derive the efficiencies, the muon overlap is checked *only* for the jets
    passing the energy fraction requirements.

    Options: `collection` (default `Jet`), `pt` (default 30.), `eta` (default 2.5),
    `muon_collection` (default `Muon`), `muon_iso_field` (default `pfRelIso04_all`),
    `muon_iso` (default 0.15), `muon_dr` (default 0.4), and the energy fraction
    thresholds `muEF` (default 0.5), `chEmEF` (default 0.5), `neEmEF` (default 0.8).
    '''
    jets = events[cfg.get("collection", "Jet")]
    mask = jets_in_ht_acceptance(jets, cfg)

    muon_like = (
        (jets.muEF < float(cfg.get("muEF", 0.5)))
        & (jets.chEmEF < float(cfg.get("chEmEF", 0.5)))
        & (jets.neEmEF < float(cfg.get("neEmEF", 0.8)))
    )
    muons = events[cfg.get("muon_collection", "Muon")]
    # Only isolated muons are considered
    muons = muons[
        muons[cfg.get("muon_iso_field", "pfRelIso04_all")]
        <= float(cfg.get("muon_iso", 0.15))
    ]
    close_to_muon = ak.any(
        delta_r_table(jets, muons) < float(cfg.get("muon_dr", 0.4)), axis=-1
    )

    return ak.sum(
        ak.where(mask & ~(muon_like & close_to_muon), jets.pt, 0.0), axis=1
    )


@register_trigger_sf_variable("atanh_btag_mean")
def atanh_btag_mean(events, cfg):
    '''atanh of the average b-tagging score of the `n` leading in b-tagging score jets.

    Options: `collection` (default `JetGood`), `field` (default `btagPNetB`),
    `n` (default 2), `pad_value` (default 0., used for events with less than `n` jets).
    '''
    jets = events[cfg.get("collection", "JetGood")]
    n = int(cfg.get("n", 2))
    score = jets[cfg.get("field", "btagPNetB")]
    score = score[ak.argsort(score, axis=1, ascending=False)]
    score = ak.fill_none(
        ak.pad_none(score, n, axis=1)[:, :n], float(cfg.get("pad_value", 0.0))
    )
    mean = ak.sum(score, axis=1) / n
    # The b-tagging score can be exactly 1: the mean is clipped to avoid infinities
    eps = float(cfg.get("epsilon", 1e-6))
    return np.arctanh(np.clip(ak.to_numpy(mean), -1 + eps, 1 - eps))
