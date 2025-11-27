def jet_type1_selection(events, jet_type, params):
    jets = events[jet_type]
    cuts = params.object_preselection[jet_type]

    # same selection as in
    # https://github.com/nurfikri89/NanoSkimmer/blob/1b4db934993267761710ab2401caf43d7a19d710/modules/AddJEC.C#L394
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