import copy
import awkward as ak

from pocket_coffea.lib.cut_functions import get_JetVetoMap_Mask
from pocket_coffea.lib.jets import jet_selection


def get_custom_JetVetoMap_Mask(events, params, year, processor_params, **kwargs):
    """
    Custom function to get the JetVetoMap mask applying it on a specific pt type of the jets.
    Args:
        events: awkward array with events
        params: configuration parameters
        year: year of the data-taking period
        processor_params: processor configuration parameters
    """

    jet_type_default = "Jet"

    # create a copy of events to avoid modifying the original one
    # replace the jet_type_default collection with the jet_type one with the desired pt_type
    events_copy = copy.copy(events)
    events_copy[jet_type_default] = ak.with_field(
        events_copy[jet_type_default],
        events_copy[params["jet_type"]][params["pt_type"]],
        "pt",
    )

    # create a copy of params to avoid modifying the original one
    # and put in the collection the AK4PFPuppi to compute the jetId
    # because whatever jet_type is passed the jetId should be computed with AK4PFPuppi
    # since it's the one having the jetId stored
    processor_params_copy = copy.copy(processor_params)
    processor_params_copy.jets_calibration.collection[year] = {
        "AK4PFPuppi": jet_type_default
    }

    mask = get_JetVetoMap_Mask(
        events_copy, params, year, processor_params_copy, **kwargs
    )

    # remove copies
    del processor_params_copy
    del events_copy
    
    return mask


def custom_jet_selection(
    events,
    jet_type,
    params,
    year,
    leptons_collection="",
    jet_tagger="",
    pt_type="pt",
    pt_cut_name="pt",
):
    """
    Custom jet selection function to apply selection on different pt types.
    Args:
        events: awkward array with events
        jet_type: str, type of jet to select (e.g. "Jet")
        params: configuration parameters
        year: year of the data-taking period
        leptons_collection: str, type of leptons to consider for overlap removal
        jet_tagger: str, jet tagger to use
        pt_type: str, type of pt to apply the cut on (e.g. "pt", "pt_default", "pt_regressed")
        pt_cut_name: str, name of the pt cut in the params (e.g. "pt", "pt_tight")
    """
    jet_type_default = "Jet"

    # create a copy of params to avoid modifying the original one
    # and put in the collection the AK4PFPuppi to compute the jetId
    # because whatever jet_type is passed the jetId should be computed with AK4PFPuppi
    # since it's the one having the jetId stored
    # copy also the object_preselection to modify it
    params_copy = copy.copy(params)
    params_copy.object_preselection[jet_type_default]["pt"] = (
        params.object_preselection[jet_type][pt_cut_name]
    )
    params_copy.jets_calibration.collection[year] = {"AK4PFPuppi": jet_type_default}

    # create a copy of events to avoid modifying the original one
    # replace the jet_type_default collection with the jet_type one
    events_copy = copy.copy(events)
    events_copy[jet_type_default] = ak.with_field(
        events_copy[jet_type],
        events_copy[jet_type][pt_type],
        "pt",
    )

    _, mask = jet_selection(
        events_copy,
        jet_type,
        params_copy,
        year,
        leptons_collection,
        jet_tagger,
    )

    if "FatJet" in jet_type:
        events[jet_type] = ak.with_field(
            events[jet_type][mask], _["btagBB"], "btagBB",
        )
        events[jet_type] = ak.with_field(
            events[jet_type], _["btagCC"], "btagCC",
        )

        # remove copies
        del params_copy
        del events_copy
        
        return events[jet_type], mask

    # remove copies
    del params_copy
    del events_copy

    return events[jet_type][mask], mask
