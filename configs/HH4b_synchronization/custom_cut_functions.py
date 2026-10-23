import awkward as ak


def mask_num_jets_in_eta(events, params, **kwargs):
    """At least `njet` jets of `jet_collection` with |eta| < `eta_max`."""
    jets = events[params["jet_collection"]]
    mask = ak.sum(abs(jets.eta) < params["eta_max"], axis=1) >= params["njet"]
    return ak.where(ak.is_none(mask), False, mask)
