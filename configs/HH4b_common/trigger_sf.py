"""Application of the HH4b jet-trigger scale factors, self-contained in AnalysisConfigs.

Deliberately independent of PocketCoffea's own `pocket_coffea.lib.trigger_sf` /
`pocket_coffea.lib.weights.common.common.SF_trigger` (added on the
`claude/trigger-sf-root-files-2ph6y5` branch): the trigger scale factor can be
applied, and its observables changed, by editing only this analysis
configuration, without needing that PocketCoffea branch merged (or even
present). It only relies on the stable `pocket_coffea.lib.weights.weights`
base classes.

The efficiency of a trigger path is often measured by factorizing it in the
efficiency of each of the filters composing the path (plus the efficiency of the
logical OR of the L1 seeds), each of them measured as a function of a different
observable (the leading jet pt, the HT, the b-tagging score...). The total
efficiency is the product of the per-filter efficiencies and therefore the total
scale factor is the product of the per-filter data/MC efficiency ratios::

    SF = prod_i eff_data_i(x_i) / prod_i eff_MC_i(x_i) = prod_i SF_i(x_i)

The per-filter scale factors are read from a correctionlib file (see
`scripts/convert_trigger_sf_to_correctionlib.py`), configured in the
`trigger_scale_factors` parameters (`params/trigger_scale_factors.yaml`): one
correction per filter, taking as input a `systematic` string (`nominal`, `up`,
`down`) and the value of the observable. The `up`/`down` variations of the total
scale factor are the product of the `up`/`down` variations of the single
filters (treated as fully correlated).
"""

import awkward as ak
import numpy as np
import correctionlib

from pocket_coffea.lib.weights.weights import WeightWrapper, WeightData

from .trigger_object_matching import delta_r_table

# Registry of the functions computing the observables used to evaluate the
# per-filter efficiencies, local to this analysis configuration. New observables
# can be added with the `register_trigger_sf_variable` decorator.
trigger_sf_variables = {}


def register_trigger_sf_variable(name):
    '''Decorator registering a function computing an observable for the trigger SF.

    The function must have the signature `f(events, cfg)`, where `cfg` is the
    configuration dictionary of the observable, and must return a flat array with
    one entry per event.
    '''

    def decorator(function):
        if name in trigger_sf_variables:
            raise ValueError(
                f"The trigger scale factor variable `{name}` has been already registered"
            )
        trigger_sf_variables[name] = function
        return function

    return decorator


def get_trigger_sf_variable(events, cfg):
    '''Compute the observable defined by the configuration `cfg`.

    :param events: awkward array of events
    :param cfg: configuration of the observable. The `name` key selects the
                function registered in `trigger_sf_variables`, the other keys are
                passed to the function as options.
    :returns: numpy array with one entry per event
    '''
    name = cfg["name"]
    if name not in trigger_sf_variables:
        raise ValueError(
            f"Trigger scale factor variable `{name}` not available. "
            f"Available variables: {list(trigger_sf_variables.keys())}"
        )
    out = trigger_sf_variables[name](events, cfg)
    return np.ascontiguousarray(ak.to_numpy(out), dtype=np.float64)


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


#############################################################################
# Trigger scale factor evaluation


def get_trigger_sf_config(params, year):
    '''Get the trigger scale factors configuration for the requested year'''
    if "trigger_scale_factors" not in params:
        raise Exception(
            "The `trigger_scale_factors` parameters are missing: they are needed "
            "to apply the trigger scale factor weight."
        )
    cfg = params["trigger_scale_factors"]
    if year not in cfg:
        raise Exception(
            f"The trigger scale factors are not configured for the year `{year}`. "
            f"Configured years: {[k for k in cfg.keys()]}"
        )
    return cfg[year]


def load_trigger_sf_correctionset(params, year):
    '''Load the correctionlib file with the per-filter trigger scale factors'''
    return correctionlib.CorrectionSet.from_file(get_trigger_sf_config(params, year)["file"])


def sf_trigger(params, events, year, correction_set=None):
    '''Compute the total trigger scale factor as the product of the per-filter ones.

    The `up` and `down` variations are computed by shifting coherently all the
    filters, as prescribed by the derivation of the efficiencies.

    :param params: parameters of the analysis
    :param events: awkward array of events
    :param year: data-taking period
    :param correction_set: (optional) already loaded correctionlib CorrectionSet.
                           If None the file in the parameters is loaded.
    :returns: tuple (sf, sf_up, sf_down) of per-event scale factors
    '''
    cfg = get_trigger_sf_config(params, year)
    if correction_set is None:
        correction_set = correctionlib.CorrectionSet.from_file(cfg["file"])

    variations = ["nominal", "up", "down"]
    sf = {variation: np.ones(len(events), dtype=np.float64) for variation in variations}

    for correction in cfg["corrections"]:
        x = get_trigger_sf_variable(events, correction["variable"])
        evaluator = correction_set[correction["name"]]
        for variation in variations:
            sf[variation] = sf[variation] * evaluator.evaluate(variation, x)

    return sf["nominal"], sf["up"], sf["down"]


# NOTE: named "sf_trigger_hh4b", not "sf_trigger", to avoid a duplicate-registration
# error against PocketCoffea's own `SF_trigger` weight (name "sf_trigger"), still
# present on the `claude/trigger-sf-root-files-2ph6y5` branch and pulled in by the
# `common_weights` import below. The two implementations are otherwise equivalent;
# this one is always the one actually used here, independently of what is or isn't
# in the PocketCoffea install. Rename back to "sf_trigger" once that branch's own
# copy is removed/decided on.
class SF_trigger(WeightWrapper):
    """Trigger scale factor computed as the product of the per-filter data/MC
    efficiency ratios, read from a correctionlib file.

    The configuration of the correctionlib file and of the observable used to
    evaluate each filter is taken from the `trigger_scale_factors` parameters.
    """

    name = "sf_trigger_hh4b"
    has_variations = True

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        # The correctionlib file is loaded lazily at the first call of `compute`:
        # the WeightWrapper objects are built for all the available weights, also
        # when the trigger scale factors are not configured for the analysis.
        self._correction_set = None

    def compute(self, events, size, shape_variation):
        if self._correction_set is None:
            self._correction_set = load_trigger_sf_correctionset(
                self._params, self._metadata["year"]
            )
        out = sf_trigger(
            self._params,
            events,
            year=self._metadata["year"],
            correction_set=self._correction_set,
        )
        return WeightData(
            name=self.name,
            nominal=out[0],
            up=out[1],
            down=out[2],
        )
