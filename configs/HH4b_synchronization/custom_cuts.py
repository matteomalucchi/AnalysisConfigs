from pocket_coffea.lib.cut_definition import Cut

import configs.HH4b_synchronization.custom_cut_functions as cuts_f

# Preselection of the synchronization: >= 4 jets of the JetCuts collection
# (`JetCuts` object preselection, built by the synchronization workflow) with
# |eta| < 2.5, without pt thresholds on the leading jets, b-tag requirements,
# lepton veto or jet veto map
hh4b_4jets_presel = Cut(
    name="hh4b_4jets_presel",
    params={"njet": 4, "jet_collection": "JetCuts", "eta_max": 2.5},
    function=cuts_f.mask_num_jets_in_eta,
)

# >= 4 JetGood jets (object preselection: pt, |eta|, jet ID, lepton cleaning)
hh4b_4jetgood_presel = Cut(
    name="hh4b_4jetgood_presel",
    params={"njet": 4, "jet_collection": "JetGood", "eta_max": 2.5},
    function=cuts_f.mask_num_jets_in_eta,
)
