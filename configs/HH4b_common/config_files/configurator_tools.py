import sys
from collections import defaultdict

from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters.histograms import jet_hists, count_hist, parton_hists
from pocket_coffea.lib.hist_manager import HistConf, Axis
from pocket_coffea.parameters.cuts import passthrough
from utils.quantile_transformer import WeightedQuantileTransformer
import numpy as np

from utils.variables_helpers import jet_hists_dict, create_HistConf
from utils.variables_helpers import jet_hists_dict, create_HistConf
import configs.HH4b_common.custom_cuts_common as cuts
import configs.VBF_HH4b.custom_cuts as vbf_cuts

variables_dict_jets = {
    **jet_hists(coll="JetGoodFromHiggsOrdered", pos=0),
    **jet_hists(coll="JetGoodFromHiggsOrdered", pos=1),
    **jet_hists(coll="JetGoodFromHiggsOrdered", pos=2),
    **jet_hists(coll="JetGoodFromHiggsOrdered", pos=3),
    **jet_hists(coll="JetGoodFromHiggsOrderedRun2", pos=0),
    **jet_hists(coll="JetGoodFromHiggsOrderedRun2", pos=1),
    **jet_hists(coll="JetGoodFromHiggsOrderedRun2", pos=2),
    **jet_hists(coll="JetGoodFromHiggsOrderedRun2", pos=3),
    # **count_hist(coll="JetGood", bins=10, start=0, stop=10),
    # **count_hist(coll="JetGoodHiggs", bins=10, start=0, stop=10),
    # **count_hist(coll="ElectronGood", bins=3, start=0, stop=3),
    # **count_hist(coll="MuonGood", bins=3, start=0, stop=3),
    # **count_hist(coll="JetGoodHiggsMatched", bins=10, start=0, stop=10),
    # **count_hist(coll="bQuarkHiggsMatched", bins=10, start=0, stop=10),
    # **count_hist(coll="JetGoodMatched", bins=10, start=0, stop=10),
    # **count_hist(coll="bQuarkMatched", bins=10, start=0, stop=10),
    # **jet_hists(coll="JetGood", pos=0),
    # **jet_hists(coll="JetGood", pos=1),
    # **jet_hists(coll="JetGood", pos=2),
    # **jet_hists(coll="JetGood", pos=3),
    # **jet_hists(coll="JetGoodHiggsPtOrder", pos=0),
    # **jet_hists(coll="JetGoodHiggsPtOrder", pos=1),
    # **jet_hists(coll="JetGoodHiggsPtOrder", pos=2),
    # **jet_hists(coll="JetGoodHiggsPtOrder", pos=3),
    # **parton_hists(coll="bQuarkHiggsMatched", pos=0),
    # **parton_hists(coll="bQuarkHiggsMatched", pos=1),
    # **parton_hists(coll="bQuarkHiggsMatched", pos=2),
    # **parton_hists(coll="bQuarkHiggsMatched", pos=3),
    # **parton_hists(coll="bQuarkHiggsMatched"),
    # **parton_hists(coll="JetGoodHiggsMatched", pos=0),
    # **parton_hists(coll="JetGoodHiggsMatched", pos=1),
    # **parton_hists(coll="JetGoodHiggsMatched", pos=2),
    # **parton_hists(coll="JetGoodHiggsMatched", pos=3),
    # **parton_hists(coll="bQuarkMatched", pos=0),
    # **parton_hists(coll="bQuarkMatched", pos=1),
    # **parton_hists(coll="bQuarkMatched", pos=2),
    # **parton_hists(coll="bQuarkMatched", pos=3),
    # **parton_hists(coll="bQuarkMatched"),
    # **parton_hists(coll="JetGoodMatched", pos=0),
    # **parton_hists(coll="JetGoodMatched", pos=1),
    # **parton_hists(coll="JetGoodMatched", pos=2),
    # **parton_hists(coll="JetGoodMatched", pos=3),
    # **parton_hists(coll="JetGoodMatched"),
}


variables_dict_fatjets = {
    **jet_hists(coll="FatJetGood", pos=0),
    **jet_hists(coll="FatJetGood", pos=1),
}


variables_dict_higgs_mass = {
    "RecoHiggs1Mass": HistConf(
        [
            Axis(
                coll=f"HiggsLeading",
                field="mass",
                bins=240,
                start=0,
                stop=240,
                label=r"$M_{H_1}$ SPANet",
            )
        ],
    ),
    "RecoHiggs1Mass_Dhh": HistConf(
        [
            Axis(
                coll=f"HiggsLeadingRun2",
                field="mass",
                bins=240,
                start=0,
                stop=240,
                label=r"$M_{H_1}$ $D_{HH}$",
            )
        ],
    ),
    "RecoHiggs2Mass": HistConf(
        [
            Axis(
                coll=f"HiggsSubLeading",
                field="mass",
                bins=240,
                start=0,
                stop=240,
                label=r"$M_{H_2}$ SPANet",
            )
        ],
    ),
    "RecoHiggs2Mass_Dhh": HistConf(
        [
            Axis(
                coll=f"HiggsSubLeadingRun2",
                field="mass",
                bins=240,
                start=0,
                stop=240,
                label=r"$M_{H_2}$ $D_{HH}$",
            )
        ],
    ),
}


variables_dict_random_pt = {
    "Random_pt_Factor": HistConf(
        [
            Axis(
                coll=f"events",
                field="random_pt_weights",
                bins=50,
                start=0,
                stop=2,
                label=r"$pT$",
            )
        ],
    ),
}

variables_dict_vbf = {
    **count_hist(coll="JetGood", bins=10, start=0, stop=10),
    **jet_hists_dict(coll="JetGood", start=1, end=5),
    **create_HistConf(
        "JetGoodVBF", "eta", bins=60, start=-5, stop=5, label="JetGoodVBFeta"
    ),
    **create_HistConf(
        "JetGoodVBF",
        "btagPNetQvG",
        pos=0,
        bins=60,
        start=0,
        stop=1,
        label="JetGoodVBFQvG_0",
    ),
    **create_HistConf(
        "JetGoodVBF",
        "btagPNetQvG",
        pos=1,
        bins=60,
        start=0,
        stop=1,
        label="JetGoodVBFQvG_1",
    ),
    **create_HistConf(
        "events", "deltaEta", bins=60, start=5, stop=10, label="JetGoodVBFdeltaEta"
    ),
    **create_HistConf(
        "JetVBF_generalSelection",
        "eta",
        bins=60,
        start=-5,
        stop=5,
        label="JetVBFgeneralSelectionEta",
    ),
    **create_HistConf(
        "JetVBF_generalSelection",
        "btagPNetQvG",
        pos=0,
        bins=60,
        start=0,
        stop=1,
        label="JetVBFgeneralSelectionQvG_0",
    ),
    **create_HistConf(
        "JetVBF_generalSelection",
        "btagPNetQvG",
        pos=1,
        bins=60,
        start=0,
        stop=1,
        label="JetVBFgeneralSelectionQvG_1",
    ),
    **create_HistConf(
        "JetVBF_matched",
        "eta",
        bins=60,
        start=-5,
        stop=5,
        label="JetVBF_matched_eta",
    ),
    **create_HistConf(
        "events",
        "etaProduct",
        bins=5,
        start=-2.5,
        stop=2.5,
        label="JetVBF_matched_eta_product",
    ),
    **create_HistConf(
        "JetVBF_matched",
        "pt",
        bins=100,
        start=0,
        stop=1000,
        label="JetVBF_matched_pt",
    ),
    **create_HistConf(
        "JetVBF_matched",
        "btagPNetQvG",
        pos=0,
        bins=60,
        start=0,
        stop=1,
        label="JetVBF_matchedQvG_0",
    ),
    **create_HistConf(
        "JetVBF_matched",
        "btagPNetQvG",
        pos=1,
        bins=60,
        start=0,
        stop=1,
        label="JetVBF_matchedQvG_1",
    ),
    **create_HistConf(
        "quarkVBF_matched",
        "eta",
        bins=60,
        start=-5,
        stop=5,
        label="quarkVBF_matched_Eta",
    ),
    **create_HistConf(
        "quarkVBF_matched",
        "pt",
        bins=100,
        start=0,
        stop=1000,
        label="quarkVBF_matched_pt",
    ),
    **create_HistConf(
        "JetVBF_matched",
        "btagPNetB",
        bins=100,
        start=0,
        stop=1,
        label="JetGoodVBF_matched_btag",
    ),
    **create_HistConf(
        "events", "deltaEta_matched", bins=100, start=0, stop=10, label="deltaEta"
    ),
    **create_HistConf(
        "events", "jj_mass_matched", bins=100, start=0, stop=5000, label="jj_mass"
    ),
    **create_HistConf("HH", "mass", bins=100, start=0, stop=2500, label="HH_mass"),
    # variables from renato
    **create_HistConf(
        "events", "HH_deltaR", bins=50, start=0, stop=8, label="HH_deltaR"
    ),
    **create_HistConf(
        "events", "H1j1_deltaR", bins=50, start=0, stop=8, label="H1j1_deltaR"
    ),
    **create_HistConf(
        "events", "H1j2_deltaR", bins=50, start=0, stop=8, label="H1j2_deltaR"
    ),
    **create_HistConf(
        "events", "H2j1_deltaR", bins=50, start=0, stop=8, label="H2j1_deltaR"
    ),
    **create_HistConf(
        "events", "HH_centrality", bins=50, start=0, stop=1, label="HH_centrality"
    ),
    **create_HistConf("HH", "pt", bins=100, start=0, stop=800, label="HH_pt"),
    **create_HistConf("HH", "eta", bins=60, start=-6, stop=6, label="HH_eta"),
    **create_HistConf("HH", "phi", bins=60, start=-5, stop=5, label="HH_phi"),
    **create_HistConf("HH", "mass", bins=100, start=0, stop=2200, label="HH_mass"),
    **create_HistConf(
        "HiggsLeading", "pt", bins=100, start=0, stop=800, label="HiggsLeading_pt"
    ),
    **create_HistConf(
        "HiggsLeading", "eta", bins=60, start=-5, stop=5, label="HiggsLeading_eta"
    ),
    **create_HistConf(
        "HiggsLeading", "phi", bins=60, start=-5, stop=5, label="HiggsLeading_phi"
    ),
    **create_HistConf(
        "HiggsLeading", "mass", bins=100, start=0, stop=500, label="HiggsLeading_mass"
    ),
    **create_HistConf(
        "HiggsSubLeading", "pt", bins=100, start=0, stop=800, label="HiggsSubLeading_pt"
    ),
    **create_HistConf(
        "HiggsSubLeading", "eta", bins=60, start=-5, stop=5, label="HiggsSubLeading_eta"
    ),
    **create_HistConf(
        "HiggsSubLeading", "phi", bins=60, start=-5, stop=5, label="HiggsSubLeading_phi"
    ),
    **create_HistConf(
        "HiggsSubLeading",
        "mass",
        bins=100,
        start=0,
        stop=500,
        label="HiggsSubLeading_mass",
    ),
    **create_HistConf("Jet", "pt", bins=100, pos=0, start=0, stop=800, label="Jet_pt0"),
    **create_HistConf("Jet", "pt", bins=100, pos=1, start=0, stop=800, label="Jet_pt1"),
    **create_HistConf("Jet", "eta", bins=60, pos=0, start=-5, stop=5, label="Jet_eta0"),
    **create_HistConf("Jet", "eta", bins=60, pos=1, start=-5, stop=5, label="Jet_eta1"),
    **create_HistConf("Jet", "phi", bins=60, pos=0, start=-5, stop=5, label="Jet_phi0"),
    **create_HistConf("Jet", "phi", bins=60, pos=1, start=-5, stop=5, label="Jet_phi1"),
    **create_HistConf(
        "Jet", "mass", bins=100, pos=0, start=0, stop=150, label="Jet_mass0"
    ),
    **create_HistConf(
        "Jet", "mass", bins=100, pos=1, start=0, stop=150, label="Jet_mass1"
    ),
    **create_HistConf(
        "Jet", "btagPNetB", pos=0, bins=100, start=0, stop=1, label="Jet_btagPNetB0"
    ),
    **create_HistConf(
        "Jet", "btagPNetB", pos=1, bins=100, start=0, stop=1, label="Jet_btagPNetB1"
    ),
    **create_HistConf(
        "Jet", "btagPNetQvG", pos=0, bins=100, start=0, stop=1, label="Jet_btagPNetQvG0"
    ),
    **create_HistConf(
        "Jet", "btagPNetQvG", pos=1, bins=100, start=0, stop=1, label="Jet_btagPNetQvG1"
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "pt",
        bins=100,
        pos=0,
        start=0,
        stop=700,
        label="JetGoodFromHiggsOrdered_pt0",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "eta",
        bins=60,
        pos=0,
        start=-5,
        stop=5,
        label="JetGoodFromHiggsOrdered_eta0",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "phi",
        bins=60,
        pos=0,
        start=-5,
        stop=5,
        label="JetGoodFromHiggsOrdered_phi0",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "mass",
        bins=100,
        pos=0,
        start=0,
        stop=80,
        label="JetGoodFromHiggsOrdered_mass0",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "btagPNetB",
        bins=100,
        pos=0,
        start=0,
        stop=1,
        label="JetGoodFromHiggsOrdered_btagPNetB0",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "btagPNetQvG",
        bins=100,
        pos=0,
        start=0,
        stop=1,
        label="JetGoodFromHiggsOrdered_btagPNetQvG0",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "pt",
        bins=100,
        pos=1,
        start=0,
        stop=700,
        label="JetGoodFromHiggsOrdered_pt1",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "eta",
        bins=60,
        pos=1,
        start=-5,
        stop=5,
        label="JetGoodFromHiggsOrdered_eta1",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "phi",
        bins=60,
        pos=1,
        start=-5,
        stop=5,
        label="JetGoodFromHiggsOrdered_phi1",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "mass",
        bins=100,
        pos=1,
        start=0,
        stop=80,
        label="JetGoodFromHiggsOrdered_mass1",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "btagPNetB",
        bins=100,
        pos=1,
        start=0,
        stop=1,
        label="JetGoodFromHiggsOrdered_btagPNetB1",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "btagPNetQvG",
        bins=100,
        pos=1,
        start=0,
        stop=1,
        label="JetGoodFromHiggsOrdered_btagPNetQvG1",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "pt",
        bins=100,
        pos=2,
        start=0,
        stop=700,
        label="JetGoodFromHiggsOrdered_pt2",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "eta",
        bins=60,
        pos=2,
        start=-5,
        stop=5,
        label="JetGoodFromHiggsOrdered_eta2",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "phi",
        bins=60,
        pos=2,
        start=-5,
        stop=5,
        label="JetGoodFromHiggsOrdered_phi2",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "mass",
        bins=100,
        pos=2,
        start=0,
        stop=80,
        label="JetGoodFromHiggsOrdered_mass2",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "btagPNetB",
        bins=100,
        pos=2,
        start=0,
        stop=1,
        label="JetGoodFromHiggsOrdered_btagPNetB2",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "btagPNetQvG",
        bins=100,
        pos=2,
        start=0,
        stop=1,
        label="JetGoodFromHiggsOrdered_btagPNetQvG2",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "pt",
        bins=100,
        pos=3,
        start=0,
        stop=700,
        label="JetGoodFromHiggsOrdered_pt3",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "eta",
        bins=60,
        pos=3,
        start=-5,
        stop=5,
        label="JetGoodFromHiggsOrdered_eta3",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "phi",
        bins=60,
        pos=3,
        start=-5,
        stop=5,
        label="JetGoodFromHiggsOrdered_phi3",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "mass",
        bins=100,
        pos=3,
        start=0,
        stop=80,
        label="JetGoodFromHiggsOrdered_mass3",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "btagPNetB",
        bins=100,
        pos=3,
        start=0,
        stop=1,
        label="JetGoodFromHiggsOrdered_btagPNetB3",
    ),
    **create_HistConf(
        "JetGoodFromHiggsOrdered",
        "btagPNetQvG",
        bins=100,
        pos=3,
        start=0,
        stop=1,
        label="JetGoodFromHiggsOrdered_btagPNetQvG3",
    ),
    **create_HistConf(
        "FatJetGood",
        "pt",
        bins=100,
        pos=0,
        start=0,
        stop=1000,
        label="FatJetGood_pt0",
    ),
    **create_HistConf(
        "FatJetGood",
        "eta",
        bins=60,
        pos=0,
        start=-5,
        stop=5,
        label="FatJetGood_eta0",
    ),
    **create_HistConf(
        "FatJetGood",
        "phi",
        bins=60,
        pos=0,
        start=-5,
        stop=5,
        label="FatJetGood_phi0",
    ),
    **create_HistConf(
        "FatJetGood",
        "mass",
        bins=100,
        pos=0,
        start=50,
        stop=250,
        label="FatJetGood_mass0",
    ),
    **create_HistConf(
        "FatJetGood",
        "msoftdrop",
        bins=100,
        pos=0,
        start=0,
        stop=200,
        label="FatJetGood_msoftdrop0",
    ),
    **create_HistConf(
        "FatJetGood",
        "btagBB",
        bins=100,
        pos=0,
        start=0,
        stop=1,
        label="FatJetGood_btagBB0",
    ),
    **create_HistConf(
        "FatJetGood",
        "btagPNetQvG",
        bins=100,
        pos=0,
        start=0,
        stop=1,
        label="FatJetGood_btagCC0",
    ),
    **create_HistConf(
        "FatJetGood",
        "pt",
        bins=100,
        pos=1,
        start=0,
        stop=1000,
        label="FatJetGood_pt1",
    ),
    **create_HistConf(
        "FatJetGood",
        "eta",
        bins=60,
        pos=1,
        start=-5,
        stop=5,
        label="FatJetGood_eta1",
    ),
    **create_HistConf(
        "FatJetGood",
        "phi",
        bins=60,
        pos=1,
        start=-5,
        stop=5,
        label="FatJetGood_phi1",
    ),
    **create_HistConf(
        "FatJetGood",
        "mass",
        bins=100,
        pos=1,
        start=50,
        stop=250,
        label="FatJetGood_mass1",
    ),
    **create_HistConf(
        "FatJetGood",
        "msoftdrop",
        bins=100,
        pos=1,
        start=0,
        stop=200,
        label="FatJetGood_msoftdrop1",
    ),
    **create_HistConf(
        "FatJetGood",
        "btagBB",
        bins=100,
        pos=1,
        start=0,
        stop=1,
        label="FatJetGood_btagBB1",
    ),
    **create_HistConf(
        "FatJetGood",
        "btagPNetQvG",
        bins=100,
        pos=1,
        start=0,
        stop=1,
        label="FatJetGood_btagCC1",
    ),
    **create_HistConf(
        "JetVBFLeadingPtNotFromHiggs",
        "pt",
        bins=100,
        pos=0,
        start=0,
        stop=700,
        label="JetVBFLeadingPtNotFromHiggs_pt0",
    ),
    **create_HistConf(
        "JetVBFLeadingPtNotFromHiggs",
        "eta",
        bins=60,
        pos=0,
        start=-5,
        stop=5,
        label="JetVBFLeadingPtNotFromHiggs_eta0",
    ),
    **create_HistConf(
        "JetVBFLeadingPtNotFromHiggs",
        "phi",
        bins=60,
        pos=0,
        start=-5,
        stop=5,
        label="JetVBFLeadingPtNotFromHiggs_phi0",
    ),
    **create_HistConf(
        "JetVBFLeadingPtNotFromHiggs",
        "mass",
        bins=100,
        pos=0,
        start=0,
        stop=75,
        label="JetVBFLeadingPtNotFromHiggs_mass0",
    ),
    **create_HistConf(
        "JetVBFLeadingPtNotFromHiggs",
        "btagPNetB",
        bins=100,
        pos=0,
        start=0,
        stop=1,
        label="JetVBFLeadingPtNotFromHiggs_btagPNetB0",
    ),
    **create_HistConf(
        "JetVBFLeadingPtNotFromHiggs",
        "btagPNetQvG",
        bins=100,
        pos=0,
        start=0,
        stop=1,
        label="JetVBFLeadingPtNotFromHiggs_btagPNetQvG0",
    ),
    **create_HistConf(
        "JetVBFLeadingPtNotFromHiggs",
        "pt",
        bins=100,
        pos=1,
        start=0,
        stop=700,
        label="JetVBFLeadingPtNotFromHiggs_pt1",
    ),
    **create_HistConf(
        "JetVBFLeadingPtNotFromHiggs",
        "eta",
        bins=60,
        pos=1,
        start=-5,
        stop=5,
        label="JetVBFLeadingPtNotFromHiggs_eta1",
    ),
    **create_HistConf(
        "JetVBFLeadingPtNotFromHiggs",
        "phi",
        bins=60,
        pos=1,
        start=-5,
        stop=5,
        label="JetVBFLeadingPtNotFromHiggs_phi1",
    ),
    **create_HistConf(
        "JetVBFLeadingPtNotFromHiggs",
        "mass",
        bins=100,
        pos=1,
        start=0,
        stop=75,
        label="JetVBFLeadingPtNotFromHiggs_mass1",
    ),
    **create_HistConf(
        "JetVBFLeadingPtNotFromHiggs",
        "btagPNetB",
        bins=100,
        pos=1,
        start=0,
        stop=1,
        label="JetVBFLeadingPtNotFromHiggs_btagPNetB1",
    ),
    **create_HistConf(
        "JetVBFLeadingPtNotFromHiggs",
        "btagPNetQvG",
        bins=100,
        pos=1,
        start=0,
        stop=1,
        label="JetVBFLeadingPtNotFromHiggs_btagPNetQvG1",
    ),
    **create_HistConf(
        "JetVBFLeadingMjjNotFromHiggs",
        "pt",
        bins=100,
        pos=0,
        start=0,
        stop=700,
        label="JetVBFLeadingMjjNotFromHiggs_pt0",
    ),
    **create_HistConf(
        "JetVBFLeadingMjjNotFromHiggs",
        "eta",
        bins=60,
        pos=0,
        start=-5,
        stop=5,
        label="JetVBFLeadingMjjNotFromHiggs_eta0",
    ),
    **create_HistConf(
        "JetVBFLeadingMjjNotFromHiggs",
        "phi",
        bins=60,
        pos=0,
        start=-5,
        stop=5,
        label="JetVBFLeadingMjjNotFromHiggs_phi0",
    ),
    **create_HistConf(
        "JetVBFLeadingMjjNotFromHiggs",
        "mass",
        bins=100,
        pos=0,
        start=0,
        stop=75,
        label="JetVBFLeadingMjjNotFromHiggs_mass0",
    ),
    **create_HistConf(
        "JetVBFLeadingMjjNotFromHiggs",
        "btagPNetB",
        bins=100,
        pos=0,
        start=0,
        stop=1,
        label="JetVBFLeadingMjjNotFromHiggs_btagPNetB0",
    ),
    **create_HistConf(
        "JetVBFLeadingMjjNotFromHiggs",
        "btagPNetQvG",
        bins=100,
        pos=0,
        start=0,
        stop=1,
        label="JetVBFLeadingMjjNotFromHiggs_btagPNetQvG0",
    ),
    **create_HistConf(
        "JetVBFLeadingMjjNotFromHiggs",
        "pt",
        bins=100,
        pos=1,
        start=0,
        stop=700,
        label="JetVBFLeadingMjjNotFromHiggs_pt1",
    ),
    **create_HistConf(
        "JetVBFLeadingMjjNotFromHiggs",
        "eta",
        bins=60,
        pos=1,
        start=-5,
        stop=5,
        label="JetVBFLeadingMjjNotFromHiggs_eta1",
    ),
    **create_HistConf(
        "JetVBFLeadingMjjNotFromHiggs",
        "phi",
        bins=60,
        pos=1,
        start=-5,
        stop=5,
        label="JetVBFLeadingMjjNotFromHiggs_phi1",
    ),
    **create_HistConf(
        "JetVBFLeadingMjjNotFromHiggs",
        "mass",
        bins=100,
        pos=1,
        start=0,
        stop=75,
        label="JetVBFLeadingMjjNotFromHiggs_mass1",
    ),
    **create_HistConf(
        "JetVBFLeadingMjjNotFromHiggs",
        "btagPNetB",
        bins=100,
        pos=1,
        start=0,
        stop=1,
        label="JetVBFLeadingMjjNotFromHiggs_btagPNetB1",
    ),
    **create_HistConf(
        "JetVBFLeadingMjjNotFromHiggs",
        "btagPNetQvG",
        bins=100,
        pos=1,
        start=0,
        stop=1,
        label="JetVBFLeadingMjjNotFromHiggs_btagPNetQvG1",
    ),
    **create_HistConf(
        "events",
        "JetVBFLeadingPtNotFromHiggs_deltaEta",
        bins=11,
        start=0,
        stop=10,
        label="JetVBFLeadingPtNotFromHiggs_deltaEta",
    ),
    **create_HistConf(
        "events",
        "JetVBFLeadingMjjNotFromHiggs_deltaEta",
        bins=11,
        start=0,
        stop=10,
        label="JetVBFLeadingMjjNotFromHiggs_deltaEta",
    ),
    **create_HistConf(
        "events",
        "JetVBFLeadingPtNotFromHiggs_jjMass",
        bins=100,
        start=0,
        stop=2000,
        label="JetVBFLeadingPtNotFromHiggs_jjMass",
    ),
    **create_HistConf(
        "events",
        "JetVBFLeadingMjjNotFromHiggs_jjMass",
        bins=100,
        start=0,
        stop=2000,
        label="JetVBFLeadingMjjNotFromHiggs_jjMass",
    ),
}

variable_dict_bkg_morphing = {
    # SPANet pairing
    "RecoHiggs1Pt": HistConf(
        [
            Axis(
                coll=f"HiggsLeading",
                field="pt",
                bins=24,
                start=0,
                stop=600,
                label=r"$pT_{H_1}$",
            )
        ],
    ),
    "RecoHiggs2Pt": HistConf(
        [
            Axis(
                coll=f"HiggsSubLeading",
                field="pt",
                bins=24,
                start=0,
                stop=600,
                label=r"$pT_{H_2}$",
            )
        ],
    ),
    "RecoDiHiggsMass": HistConf(
        [
            Axis(
                coll=f"HH",
                field="mass",
                bins=16,
                start=200,
                stop=1000,
                label=r"$M_{HH}$",
            )
        ]
    ),
    "RecoHiggs1Mass": HistConf(
        [
            Axis(
                coll=f"HiggsLeading",
                field="mass",
                bins=16,
                start=80,
                stop=160,
                label=r"$M_{H_1}$",
            )
        ],
    ),
    "RecoHiggs2Mass": HistConf(
        [
            Axis(
                coll=f"HiggsSubLeading",
                field="mass",
                bins=16,
                start=80,
                stop=160,
                label=r"$M_{H_2}$",
            )
        ]
    ),
    "dRHiggs1": HistConf(
        [
            Axis(
                coll=f"HiggsLeading",
                field="dR",
                bins=16,
                start=0,
                stop=3,
                label=r"${H_1} \Delta R_{jj}$",
            )
        ],
    ),
    "dRHiggs2": HistConf(
        [
            Axis(
                coll=f"HiggsSubLeading",
                field="dR",
                bins=16,
                start=0,
                stop=4,
                label=r"${H_2} \Delta R_{jj}$",
            )
        ],
    ),
    # Run2 pairing
    "RecoHiggs1PtRun2": HistConf(
        [
            Axis(
                coll=f"HiggsLeadingRun2",
                field="pt",
                bins=24,
                start=0,
                stop=600,
                label=r"$pT_{H_1}$",
            )
        ],
    ),
    "RecoHiggs2PtRun2": HistConf(
        [
            Axis(
                coll=f"HiggsSubLeadingRun2",
                field="pt",
                bins=24,
                start=0,
                stop=600,
                label=r"$pT_{H_2}$",
            )
        ],
    ),
    "RecoDiHiggsMassRun2": HistConf(
        [
            Axis(
                coll=f"HHRun2",
                field="mass",
                bins=16,
                start=200,
                stop=1000,
                label=r"$M_{HH}$",
            )
        ]
    ),
    "RecoHiggs1MassRun2": HistConf(
        [
            Axis(
                coll=f"HiggsLeadingRun2",
                field="mass",
                bins=16,
                start=80,
                stop=160,
                label=r"$M_{H_1}$",
            )
        ],
    ),
    "RecoHiggs2MassRun2": HistConf(
        [
            Axis(
                coll=f"HiggsSubLeadingRun2",
                field="mass",
                bins=16,
                start=80,
                stop=160,
                label=r"$M_{H_2}$",
            )
        ]
    ),
    "dRHiggs1Run2": HistConf(
        [
            Axis(
                coll=f"HiggsLeadingRun2",
                field="dR",
                bins=16,
                start=0,
                stop=3,
                label=r"${H_1} \Delta R_{jj}$",
            )
        ],
    ),
    "dRHiggs2Run2": HistConf(
        [
            Axis(
                coll=f"HiggsSubLeadingRun2",
                field="dR",
                bins=16,
                start=0,
                stop=4,
                label=r"${H_2} \Delta R_{jj}$",
            )
        ],
    ),
    # common
    "dR_min": HistConf(
        [
            Axis(
                coll=f"events",
                field="dR_min",
                bins=16,
                start=0,
                stop=2,
                label=r"$min \Delta R_{jj}$",
            )
        ],
    ),
    "dR_max": HistConf(
        [
            Axis(
                coll=f"events",
                field="dR_max",
                bins=16,
                start=0,
                stop=4,
                label=r"$max \Delta R_{jj}$",
            )
        ],
    ),
}


def get_variables_dict_sig_bkg_score(transformed_bins, y=""):
    score_histograms = {
        "sig_bkg_dnn_score": HistConf(
            [
                Axis(
                    coll="events",
                    field="sig_bkg_dnn_score",
                    bins=20,
                    start=0,
                    stop=1,
                    label="Signal vs Background DNN score",
                )
            ],
            storage="weight",
        ),
        "sig_bkg_dnn_scoreRun2": HistConf(
            [
                Axis(
                    coll="events",
                    field="sig_bkg_dnn_scoreRun2",
                    bins=20,
                    start=0,
                    stop=1,
                    label=r"Signal vs Background DNN score D$_{HH}$-Method",
                )
            ],
            storage="weight",
        ),
    }
    if transformed_bins:
        score_histograms[f"sig_bkg_dnn_score_transformed{y}"] = HistConf(
            [
                Axis(
                    coll="events",
                    field="sig_bkg_dnn_score",
                    bins=transformed_bins,
                    type="variable",
                    start=0,
                    stop=1,
                    label=f"Signal vs Background DNN score transformed {y}",
                )
            ],
            storage="weight",
        )
        score_histograms[f"sig_bkg_dnn_score_transformedRun2{y}"] = HistConf(
            [
                Axis(
                    coll="events",
                    field="sig_bkg_dnn_scoreRun2",
                    bins=transformed_bins,
                    type="variable",
                    start=0,
                    stop=1,
                    label=r"Signal vs Background DNN score D$_{HH}$-Method transformed "
                    + y,
                )
            ],
            storage="weight",
        )
    return score_histograms


def get_variables_dict(
    year,
    config_options_dict,
    JETS=False,
    CLASSIFICATION=False,
    RANDOM_PT=False,
    VBF_VARIABLES=False,
    BKG_MORPHING=False,
    SCORE=False,
    RUN2=False,
    SPANET=True,
    BOOSTED=False,
):
    """Function to create the variable dictionary for the PocketCoffea Configurator()."""
    variables_dict = {}
    if JETS:
        variables_dict.update(variables_dict_jets)
    if CLASSIFICATION:
        variables_dict.update(variables_dict_higgs_mass)
    if RANDOM_PT:
        variables_dict.update(variables_dict_random_pt)
    if VBF_VARIABLES:
        variables_dict.update(variables_dict_vbf)
    if BKG_MORPHING:
        variables_dict.update(variable_dict_bkg_morphing)
    if BOOSTED:
        variables_dict.update(variables_dict_fatjets)
    if SCORE:
        has_qt = False

        assert isinstance(
            year, list
        ), "Year must be a list of the years to be considered."

        for y in year:
            if "postEE" in y and config_options_dict["qt_postEE"]:
                params_qt = config_options_dict["qt_postEE"]
                print(f"Using postEE quantile transformation for year {y}")
            elif "preEE" in y and config_options_dict["qt_preEE"]:
                params_qt = config_options_dict["qt_preEE"]
                print(f"Using preEE quantile transformation for year {y}")
            else:
                print(f"Did not find a valid quantile transformation for year {y}")
                params_qt = None

            if params_qt:
                has_qt = True
                transformer = WeightedQuantileTransformer(
                    n_quantiles=0, output_distribution="uniform"
                )  # We read the quantiles and distribution anyway from the pickle file
                transformer.load(params_qt)
                transformed_bins = transformer.quantiles_
                transformed_bins[0] = 0.0
                transformed_bins[-1] = 1.0
                variables_dict.update(
                    get_variables_dict_sig_bkg_score(list(transformed_bins), y)
                )
            # bins_spanet_final = bins_spanet[::step]
        if not has_qt:
            variables_dict.update(get_variables_dict_sig_bkg_score(False))
    # Sort of lazy implementation. If neither SPANet nor RUN2 are active, no variables are saved.
    # If not Run2, kick out all variables with Run2 in name
    # If not SPANet, kick out all variables without Run2 in name
    if (not RUN2) and (not BOOSTED):
        print(" - Removing Run2 variables")
        variables_dict = {k: v for k, v in variables_dict.items() if "Run2" not in k}
    elif (not SPANET) and (not BOOSTED):
        print(" - Removing non-Run2 variables")
        variables_dict = {k: v for k, v in variables_dict.items() if "Run2" in k}
    elif (BOOSTED) and (not SPANET) and (not RUN2):
        print(" - Removing non-FatJetGood variables")
        for k in variables_dict.items():
            print(k)
        variables_dict = {k: v for k, v in variables_dict.items() if "FatJetGood" in k}
    return variables_dict


SPANET_TRAINING_DEFAULT_COLUMN_PARAMS = [
    "provenance",
    "pt",
    "eta",
    "phi",
    "mass",
    "btagPNetB",
]
SPANET_TRAINING_DEFAULT_COLUMNS = {
    "JetGoodMatched": SPANET_TRAINING_DEFAULT_COLUMN_PARAMS,
    "JetGoodHiggsMatched": SPANET_TRAINING_DEFAULT_COLUMN_PARAMS,
    "JetGood": SPANET_TRAINING_DEFAULT_COLUMN_PARAMS,
    "JetGoodHiggs": SPANET_TRAINING_DEFAULT_COLUMN_PARAMS,
}

SPANET_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP = [
    "provenance",
    "pt",
    "eta",
    "phi",
    "mass",
    "btagPNetB_5wp",
    "btagPNetB_3wp",
    "btagPNetB",
]
SPANET_TRAINING_DEFAULT_COLUMNS_BTWP = {
    "JetGoodMatched": SPANET_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    "JetGoodHiggsMatched": SPANET_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    "JetGood": SPANET_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
    "JetGoodHiggs": SPANET_TRAINING_DEFAULT_COLUMN_PARAMS_BTWP,
}

DEFAULT_JET_COLUMN_PARAMS = [
    "pt",
    "eta",
    "phi",
    "mass",
    "btagPNetB",
]
DEFAULT_JET_COLUMNS = {
    "JetGood": DEFAULT_JET_COLUMN_PARAMS,
}

DEFAULT_JET_COLUMNS_DICT = {
    f"JetGood_{x}": ["JetGood", x] for x in DEFAULT_JET_COLUMN_PARAMS
}

DEFAULT_FATJET_COLUMN_PARAMS = [
    "pt",
    "eta",
    "phi",
    "mass",
    "mass_regr",
    "msoftdrop",
    "btagBB",
    "btagCC",
]
DEFAULT_FATJET_COLUMNS = {
    "FatJetGood": DEFAULT_FATJET_COLUMN_PARAMS,
    "FatJetGoodSelected": DEFAULT_FATJET_COLUMN_PARAMS,
}

DEFAULT_FATJET_COLUMNS_DICT = {
    f"FatJetGood_{x}": ["FatJetGood", x] for x in DEFAULT_FATJET_COLUMN_PARAMS
}

def get_columns_list(
    columns_dict=DEFAULT_JET_COLUMNS,
    flatten=True,
):
    """Function to create the column definition for the PocketCoffea Configurator().
    If any of the input options is set to `None`, the default option is used. To not save anything, use `[]`.

    :param: collection_dict: dict: dictionary with the collection name as key and the list of parameters to save as value.
    :param: flatten: bool: whether to flatten the columns or not.
    """
    columns = []
    for collection, attributes in columns_dict.items():
        columns.append(ColOut(collection, attributes, flatten))
        
    # add the event number if not present in the columns
    add_event_number = True
    for col in columns:
        for attr in col.collection:
            if attr == "event" and col.name == "events":
                add_event_number = False
    if add_event_number:
        columns.append(ColOut("events", ["event"], flatten))
    
    return columns


def create_DNN_columns_list(run2, flatten, columns_dict, btag=False):
    """Create the columns of the DNN input variables"""
    column_dict = defaultdict(set)
    for x, y in columns_dict.values():
        if run2:
            if x != "events":
                column_dict[x.split(":")[0] + "Run2"].add(y)
        else:
            column_dict[x.split(":")[0]].add(y)
    if run2:
        column_dict.update(
            {
                "events": set(
                    (
                        "era",
                        "HT",
                        "dR_min",
                        "dR_max",
                        "sigma_over_higgs1_reco_massRun2",
                        "sigma_over_higgs2_reco_massRun2",
                    )
                )
            }
        )
    column_dict = {x: list(y) for x, y in column_dict.items()}
    if btag:
        column_dict[f"JetGoodFromHiggsOrdered{'Run2' if run2 else ''}"].append(
            "btagPNetB"
        )
    column_list = get_columns_list(column_dict, flatten)
    print("porca madonna", column_list)
    return column_list


def define_single_category(category_name):
    """
    Define a single category for the analysis.
    """
    cut_list = []
    # number of b jets
    if "4b" in category_name:
        cut_list.append(cuts.hh4b_4b_region)
    if "2b" in category_name:
        cut_list.append(cuts.hh4b_2b_region)

    if "boosted" in category_name:
        if "signal" in category_name:
            cut_list.append(cuts.hh4b_boosted_signal_region)
        if "ttbar" in category_name:
            cut_list.append(cuts.hh4b_boosted_ttbar_control_region)
        if "qcd" in category_name:
            cut_list.append(cuts.hh4b_boosted_qcd_control_region)
    # mass cuts
    elif "VR1" not in category_name:
        if "control" in category_name:
            if "Run2" not in category_name:
                cut_list.append(cuts.hh4b_control_region)
            else:
                cut_list.append(cuts.hh4b_control_region_run2)
        if "signal" in category_name:
            if "Run2" not in category_name:
                cut_list.append(cuts.hh4b_signal_region)
            else:
                cut_list.append(cuts.hh4b_signal_region_run2)
    if "VR1" in category_name:
        if "control" in category_name:
            if "Run2" not in category_name:
                cut_list.append(cuts.hh4b_VR1_control_region)
            else:
                cut_list.append(cuts.hh4b_VR1_control_region_run2)
        if "signal" in category_name:
            if "Run2" not in category_name:
                cut_list.append(cuts.hh4b_VR1_signal_region)
            else:
                cut_list.append(cuts.hh4b_VR1_signal_region_run2)

    # blind region
    if "blind" in category_name:
        if "Run2" not in category_name:
            cut_list.append(cuts.blinded)
        else:
            cut_list.append(cuts.blindedRun2)

    if len(cut_list) < 1:  # aka if no cut applied
        cut_list.append(passthrough)
    category_item = {category_name: cut_list}
    return category_item


def define_categories(
    bkg_morphing_dnn=False, blind=False, spanet=False, run2=False, vr1=False, btag_sf_comp=False, boosted=False
):
    """
    Define the categories for the analysis.
    """
    categories_dict = {}
    if not vr1:
        if spanet:
            # categories_dict |= define_single_category("4b_region")
            categories_dict |= define_single_category("4b_control_region")
            categories_dict |= define_single_category("2b_control_region_preW")
            categories_dict |= (
                define_single_category("4b_signal_region" + "_blind") if blind else {}
            )
            categories_dict |= define_single_category("4b_signal_region")
            categories_dict |= (
                define_single_category("2b_signal_region_preW" + "_blind")
                if blind
                else {}
            )
            categories_dict |= define_single_category("2b_signal_region_preW")
            if bkg_morphing_dnn:
                categories_dict |= define_single_category("2b_control_region_postW")
                categories_dict |= (
                    define_single_category("2b_signal_region_postW" + "_blind")
                    if blind
                    else {}
                )
                categories_dict |= define_single_category("2b_signal_region_postW")
        if run2:
            categories_dict |= define_single_category("4b_regionRun2")
            categories_dict |= define_single_category("4b_control_regionRun2")
            categories_dict |= define_single_category("2b_control_region_preWRun2")
            categories_dict |= (
                define_single_category("4b_signal_region" + "_blind" + "Run2")
                if blind
                else {}
            )
            categories_dict |= define_single_category("4b_signal_region" + "Run2")
            categories_dict |= (
                define_single_category("2b_signal_region_preW" + "_blind" + "Run2")
                if blind
                else {}
            )
            categories_dict |= define_single_category("2b_signal_region_preW" + "Run2")
            if bkg_morphing_dnn:
                categories_dict |= define_single_category("2b_control_region_postWRun2")
                categories_dict |= (
                    define_single_category("2b_signal_region_postW" + "_blind" + "Run2")
                    if blind
                    else {}
                )
                categories_dict |= define_single_category(
                    "2b_signal_region_postW" + "Run2"
                )
        if boosted:
            categories_dict |= define_single_category("boosted_signal_region")
            categories_dict |= define_single_category("boosted_ttbar_region")
            categories_dict |= define_single_category("boosted_qcd_region")
    else:
        if spanet:
            categories_dict |= define_single_category("4b_VR1_control_region")
            categories_dict |= define_single_category("2b_VR1_control_region_preW")
            categories_dict |= define_single_category("4b_VR1_signal_region")
            categories_dict |= define_single_category("2b_VR1_signal_region_preW")
            if bkg_morphing_dnn:
                categories_dict |= define_single_category("2b_VR1_control_region_postW")
                categories_dict |= define_single_category("2b_VR1_signal_region_postW")
        if run2:
            categories_dict |= define_single_category("4b_VR1_control_regionRun2")
            categories_dict |= define_single_category("2b_VR1_control_region_preWRun2")
            categories_dict |= define_single_category("4b_VR1_signal_regionRun2")
            categories_dict |= define_single_category("2b_VR1_signal_region_preWRun2")
            if bkg_morphing_dnn:
                categories_dict |= define_single_category(
                    "2b_VR1_control_region_postWRun2"
                )
                categories_dict |= define_single_category(
                    "2b_VR1_signal_region_postWRun2"
                )

    if not spanet and not run2 and not boosted:
        # add the 2b control region post W for the old DNN
        categories_dict |= define_single_category("4b_region")

    if btag_sf_comp:
        btag_sf_categories = {}
        for key, value in categories_dict.items():
            btag_sf_categories[f"{key}_sf_btag"] = value
        categories_dict |= btag_sf_categories

    return categories_dict


def define_preselection(options):
    ## Define the preselection to apply
    if "no_btag" in options.keys() and options["no_btag"]:
        preselection = [cuts.hh4b_presel_nobtag]
    else:
        if options["vbf_presel"]:
            if options["tight_cuts"]:
                preselection = [vbf_cuts.vbf_hh4b_presel_tight]
            else:
                preselection = [vbf_cuts.vbf_hh4b_presel]
        elif options["boosted_presel"]:
            preselection = [cuts.hh4b_boosted_presel]
        else:
            if options["tight_cuts"]:
                preselection = [cuts.hh4b_presel_tight]
            else:
                preselection = [cuts.hh4b_presel]
    
    # Add the Jet Veto Map
    # Do this in the preselection to select jets based on
    # corrected pT after the Calibrators have run
    if not options["boosted_presel"]: # FIXME
        preselection.append(cuts.hh4b_JetVetoMap)
    
    return preselection
