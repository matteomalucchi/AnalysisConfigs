import numpy as np
import awkward as ak


def lepton_selection(events, lepton_flavour, params):
    leptons = events[lepton_flavour]
    cuts = params.object_preselection[lepton_flavour]

    # Requirements on pT and eta
    passes_eta = abs(leptons.eta) < cuts["eta"]
    passes_pt = leptons.pt > cuts["pt"]

    # Requirements on the impact parameters
    passes_dxy = leptons.dxy < cuts["dxy"]
    passes_dz = leptons.dz < cuts["dz"]

    if lepton_flavour == "Electron":
        # Requirements on isolation and id
        passes_iso = leptons.pfRelIso03_all < cuts["iso"]
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
    if semi_tight_vbf and "ptSemiTight" in cuts.keys():
        pt_cut = "ptSemiTight"

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
