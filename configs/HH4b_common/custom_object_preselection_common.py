import numpy as np
import awkward as ak


def lepton_selection(events, lepton_flavour, params):
    leptons = events[lepton_flavour]
    cuts = params.object_preselection[lepton_flavour]

    # Requirements on pT and eta
    passes_eta = abs(leptons.eta) < cuts["eta"]
    passes_pt = leptons.pt > cuts["pt"]

    # Requirements on the impact parameters
    if "eta_threshold" in cuts.keys():
        mask_highEta = leptons.eta > cuts["eta_threshold"]
        threshold_dxy = ak.where(mask_highEta, cuts["dxy_highEta"], cuts["dxy_lowEta"])
        threshold_dz = ak.where(mask_highEta, cuts["dz_highEta"], cuts["dz_lowEta"])
        passes_dxy = leptons.dxy < threshold_dxy
        passes_dz = leptons.dz < threshold_dz
    else:
        if "dxy" in cuts.keys():
            passes_dxy = leptons.dxy < cuts["dxy"]
        else:
            passes_dxy = ak.ones_like(leptons.dxy, dtype=bool)
        if "dz" in cuts.keys():
            passes_dz = leptons.dz < cuts["dz"]
        else:
            passes_dz = ak.ones_like(leptons.dz, dtype=bool)

    if lepton_flavour == "Electron":
        # Requirements on isolation and id
        if "iso" in cuts.keys():
            if isinstance(cuts["iso"], float):
                passes_iso = leptons.pfRelIso03_all < cuts["iso"]
            else:
                passes_iso = leptons[cuts["iso"]["type"]] < cuts["iso"]["max_value"]
        else:
            passes_iso = ak.ones_like(leptons.pfRelIso03_all, dtype=bool)
        passes_id = leptons[cuts["id"]["type"]] >= cuts["id"]["working_point"]
        good_leptons = (
            passes_eta & passes_pt & passes_iso & passes_dxy & passes_dz & passes_id
        )

    elif lepton_flavour == "Muon":
        # Requirements on isolation and id
        if cuts["iso"]["type"] == "pfIsoId":
            passes_iso = leptons[cuts["iso"]["type"]] >= cuts["iso"]["working_point"]
        else:
            passes_iso = (
                leptons[cuts["iso"]["type"]] < cuts["iso"]["max_value"]
            )  # e.g. pfRelIso03_all

        passes_id = leptons[cuts["id"]] == True

        good_leptons = (
            passes_eta & passes_pt & passes_iso & passes_dxy & passes_dz & passes_id
        )

    return leptons[good_leptons]


def jet_selection_nopu(
    events, jet_type, params, tight_cuts=False, semi_tight_vbf=False,
):
    jets = events[jet_type]
    cuts = params.object_preselection[jet_type]
    # Only jets that are more distant than dr to ALL leptons are tagged as good jets
    # Mask for  jets not passing the preselection

    pt_cut = "pt"
    if tight_cuts and "pt_tight" in cuts.keys():
        pt_cut = "pt_tight"

    if "eta_min" in cuts.keys() and "eta_max" in cuts.keys():
        mask_jets = (
            (jets.pt > cuts[pt_cut])
            & (np.abs(jets.eta) > cuts["eta_min"])
            & (np.abs(jets.eta) < cuts["eta_max"])
            & (jets.jetId >= cuts["jetId"])
            & (jets.btagPNetB >= cuts["btagPNetB"])
        )
    else:
        mask_jets = (
            (jets.pt > cuts[pt_cut])
            & (np.abs(jets.eta) < cuts["eta"])
            & (jets.jetId >= cuts["jetId"])
            & (jets.btagPNetB > cuts["btagPNetB"])
        )

    return jets[mask_jets]


def object_cleaning(object, cleaning_collection, dr_min=0.4):
    # here I create a deltaR matrix between jets and cleaning collection the output shape is (njets, ncleaning)
    dR = object[:, :, None].delta_r(cleaning_collection[:, None, :])

    # then I check if the jets are within dR min of ANY cleaning object
    dR_mask = dR < dr_min
    dR_mask_jets = ak.any(dR_mask, axis=2)

    # I then add to the mask the cleaning requirement
    cleaned_object = object[~dR_mask_jets]

    return cleaned_object


def clean_ak4_boosted(obj, ak8jets, muons, electrons, dr_jets, dr_lep):
    """Clean the AK4 collection to not be inside an AK8 or lepton cone."""
    # Clean From AK8
    cleaned_obj = object_cleaning(
        obj,
        ak8jets,
        dr_min=dr_jets
    )
    # Clean From Electrons
    cleaned_obj = object_cleaning(
        cleaned_obj,
        electrons,
        dr_min=dr_lep
    )
    # Clean From Muons
    cleaned_obj = object_cleaning(
        cleaned_obj,
        electrons,
        dr_min=dr_lep
    )
    return cleaned_obj
