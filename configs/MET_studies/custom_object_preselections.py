from utils.custom_cut_functions import custom_jet_selection


def jet_type1_selection(
    events,
    jet_type,
    params,
    year,
    leptons_collection="",
    jet_tagger="",
):

    jets = events[jet_type]

    _, mask = custom_jet_selection(
        events,
        jet_type,
        jet_type,
        params,
        year,
        leptons_collection,
        jet_tagger,
    )

    mask_EmEF = jets.EmEF < params.object_preselection[jet_type]["EmEF"]

    mask_jets = mask & mask_EmEF

    return jets[mask_jets]


def low_pt_jet_type1_selection(
    events,
    jet_type,
    params,
    year,
):

    jets = events[jet_type]
    cuts = params.object_preselection[jet_type]

    mask_jets = (
        (jets.pt > cuts["pt"])
        & (abs(jets.eta) < cuts["eta"])
        & (jets.EmEF < cuts["EmEF"])
    )

    return jets[mask_jets]

def muon_selection_custom(events, params):
    lepton_flavour = "Muon"

    leptons = events[lepton_flavour]
    cuts = params.object_preselection[lepton_flavour]

    # Requirements on pT and eta
    passes_eta = abs(leptons.eta) < cuts["eta"]
    passes_pt = leptons.pt > cuts["pt"]
    passes_dxy = abs(leptons.dxy) < cuts["dxy"]
    passes_dz = abs(leptons.dz) < cuts["dz"]
    passes_id = leptons[cuts["id"]] == True
    passes_iso = leptons.miniPFRelIso_all < cuts["iso"]
    passes_is_global = leptons.isGlobal == True
    passes_is_tracker = leptons.isTracker == True

    good_leptons = (
        passes_eta
        & passes_pt
        & passes_iso
        & passes_id
        & passes_dxy
        & passes_dz
        & passes_is_global
        & passes_is_tracker
    )

    return leptons[good_leptons]
