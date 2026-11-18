from . import custom_cut_functions_common as cuts_f
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
    get_L1sel,
    goldenJson,
    eventFlags,
    get_nPVgood,
)
from utils_configs.custom_cut_functions import get_custom_JetVetoMap_Mask

def hh4b_presel(pt_type):
    return Cut(
        name="hh4b_presel",
        params={
            "njet": 4,
            "pt_jet0": 80,
            "pt_jet1": 60,
            "pt_jet2": 45,
            "pt_jet3": 35,
            "mean_pnet_jet": 0.65,
            "tight_cuts": False,
            "pt_type": pt_type,
        },
        function=cuts_f.hh4b_presel_cuts,
    )

hh4b_boosted_presel = Cut(
    name="hh4b_boosted_presel",
    params={
        "nfatjet": 2,
    },
    function=cuts_f.hh4b_boosted_presel_cuts,
)

hh4b_boosted_2fatjets = Cut(
    name="hh4b_boosted_2fatjets",
    params={
        "nfatjet": 2,
    },
    function=cuts_f.hh4b_boosted_2fatjets,
)

hh4b_boosted_lepton_veto = Cut(
    name="hh4b_boosted_lepton_veto",
    params={
    },
    function=cuts_f.hh4b_boosted_lepton_veto,
)

def hh4b_presel_tight(pt_type):
    return Cut(
        name="hh4b_presel_tight",
        params={
            "njet": 4,
            "pt_jet0": 80,
            "pt_jet1": 60,
            "pt_jet2": 45,
            "pt_jet3": 35,
            "mean_pnet_jet": 0.65,
            "tight_cuts": True,
            "pt_type": pt_type,
        },
        function=cuts_f.hh4b_presel_cuts,
    )

def hh4b_presel_nobtag(pt_type):
    return Cut(
        name="hh4b_presel_nobtag",
        params={
            "njet": 4,
            "pt_jet0": 80,
            "pt_jet1": 60,
            "pt_jet2": 45,
            "pt_jet3": 35,
            "mean_pnet_jet": -999,
            "tight_cuts": False,
            "pt_type": pt_type,
        },
        function=cuts_f.hh4b_presel_cuts,
    )

def hh4b_presel_parking(pt_type):
    return Cut(
        name="hh4b_presel_parking",
        params={
            "njet": 4,
            "pt_jet0": 35,
            "pt_jet1": 35,
            "pt_jet2": 35,
            "pt_jet3": 30,
            "mean_pnet_jet": 0.55,
            "tight_cuts": False,
            "pt_type": pt_type,
        },
        function=cuts_f.hh4b_presel_cuts,
    )


# Resolved preselection chosen per chunk from its year and primary dataset
# (see `get_hh4b_presel_era`), so that several years can run in one go.
# The |eta| < 2.5 requirement on the jets is in the object preselection.
def hh4b_presel_era_dependent(pt_type, tight_cuts):
    return Cut(
        name="hh4b_presel_era_dependent",
        params={
            "njet": 4,
            "tight_cuts": tight_cuts,
            "pt_type": pt_type,
            "presel_by_era": {
                "A": {
                    "pt_jets": [80, 60, 45, 35],
                    "btag_type": "mean_btag",
                    "btag_tagger": "btagPNetB",
                    "mean_btag_jet": 0.65,
                },
                "B": {
                    "pt_jets": [35, 35, 35, 30],
                    "btag_type": "mean_btag",
                    "btag_tagger": "btagPNetB",
                    "mean_btag_jet": 0.55,
                },
                "C": {
                    "pt_jets": [30, 30, 30, 30],
                    "btag_type": "nbtag_wp",
                    "btag_tagger": "btagUParTAK4B",
                    "btag_wp": "M",
                    "nbtag": 2,
                },
            },
        },
        function=cuts_f.hh4b_presel_era_dependent_cuts,
    )


hh4b_2b_region = Cut(
    name="hh4b_2b_region",
    params={
        "third_pnet_jet": 0.2605,
        "fourth_pnet_jet": 0.2605,
    },
    function=cuts_f.hh4b_2b_cuts,
)
hh4b_4b_region = Cut(
    name="hh4b_4b_region",
    params={
        "third_pnet_jet": 0.2605,
        "fourth_pnet_jet": 0.2605,
    },
    function=cuts_f.hh4b_4b_cuts,
)

hh4b_signal_region = Cut(
    name="hh4b_signal_region",
    params={
        "radius_min": 0,
        "radius_max": 30,
        "higgs_lead_center": 125,
        "higgs_sublead_center": 120,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

hh4b_control_region = Cut(
    name="hh4b_control_region",
    params={
        "radius_min": 30,
        "radius_max": 55,
        "higgs_lead_center": 125,
        "higgs_sublead_center": 120,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)
hh4b_control_region_wide = Cut(
    name="hh4b_control_region",
    params={
        "Run2": False,
        "radius_min": 30,
        "radius_max": 80,
        "higgs_lead_center": 125,
        "higgs_sublead_center": 120,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

def hh4b_boosted_baseline(region_setting="40GeV"):
    return Cut(
        name="hh4b_boosted_baseline",
        params={
            # "pnet_cut": 0.65,
            "bbtagTXbb": 0.30 if region_setting=="40GeV" else 0.05,
            # "mass_min": 100,
            # "mass_max": 150,
        },
        function=cuts_f.hh4b_boosted_SR_cuts,
)


hh4b_boosted_inclusive = Cut(
    name="hh4b_boosted_inclusive",
    params={},
    function=cuts_f.hh4b_boosted_inclusive,
)

def hh4b_boosted_TXbb_signal(region_setting):
    return Cut(
        name="hh4b_boosted_TXbb_signal",
        params={
            "signal": True,
            "threshold": 0.8 if region_setting=="40GeV" else 0.65
            },
        function=cuts_f.hh4b_boosted_TXbb,
)
def hh4b_boosted_TXbb_control(region_setting):
    return Cut(
        name="hh4b_boosted_TXbb_control",
        params={
            "signal": False,
            "threshold": 0.8 if region_setting=="40GeV" else 0.65
            },
        function=cuts_f.hh4b_boosted_TXbb,
)

def hh4b_boosted_mass_signal(region_setting, lead_sublead):
    return Cut(
        name="hh4b_boosted_mass_signal",
        params={
            "signal": True,
            "lower": 110 if region_setting=="40GeV" else 100,
            "upper": 155 if region_setting=="40GeV" else 150,
                "sideband_size": 40 if region_setting=="40GeV" else 50,
            "full_sideband": False,
            "higgs": lead_sublead
            },
        function=cuts_f.hh4b_boosted_mass,
)

def hh4b_boosted_mass_sideband_full(region_setting, lead_sublead):
    return Cut(
        name="hh4b_boosted_mass_sideband_full",
        params={
            "signal": False,
            "lower": 110 if region_setting=="40GeV" else 100,
            "upper": 155 if region_setting=="40GeV" else 150,
            "sideband_size": 40 if region_setting=="40GeV" else 50,
            "full_sideband": True,
            "higgs": lead_sublead
            },
        function=cuts_f.hh4b_boosted_mass,
)

def hh4b_boosted_mass_sideband_lower(region_setting, lead_sublead):
    return Cut(
        name="hh4b_boosted_mass_sideband_lower",
        params={
            "signal": False,
            "lower": 110 if region_setting=="40GeV" else 100,
            "upper": 155 if region_setting=="40GeV" else 150,
            "sideband_size": 40 if region_setting=="40GeV" else 50,
            "full_sideband": False,
            "higgs": lead_sublead
            },
        function=cuts_f.hh4b_boosted_mass,
)

def boosted_mass_window(region_setting):
    """Mass window of the boosted signal region and size of the mass sidebands (in GeV)."""
    if region_setting == "40GeV":
        return {"lower": 110, "upper": 155, "sideband_size": 40}
    return {"lower": 100, "upper": 150, "sideband_size": 50}


def hh4b_boosted_mass_ttbar(region_setting):
    """ttbar sideband: both Higgs candidate masses in [upper, upper + sideband_size] (150-200 GeV for 50GeV)."""
    return Cut(
        name="hh4b_boosted_mass_ttbar",
        params=boosted_mass_window(region_setting),
        function=cuts_f.hh4b_boosted_mass_ttbar,
    )


def _get_params(region_setting="50GeV"):
    if region_setting == "50GeV":
        return {"txbb_c1": 0.96, "bdt_ggf_c1": 0.965, "txbb_vbf": 0.855, "bdt_vbf": 0.9825, "txbb_c2a": 0.85, "bdt_ggf_c2b": 0.88, "bdt_ggf_bkg": 0.03}
    else:
        return {"txbb_c1": 0.945, "bdt_ggf_c1": 0.94, "txbb_vbf": 0.8, "bdt_vbf": 0.9825, "txbb_c2a": 0.85, "bdt_ggf_c2b": 0.755, "bdt_ggf_bkg": 0.03}


def _category_params(txbb_morph, region_setting, optimised_cuts, score_suffix):
    """Parameters of the boosted categories.

    `score_suffix` selects the DNN/BDT scores evaluated with shifted Higgs masses (e.g. "_qcdsb", "_ttsb"),
    which are used to define the categories in the mass sidebands. "" uses the nominal scores.
    """
    if optimised_cuts:
        cuts = dict(optimised_cuts)
        cuts["score_ggf"] = "sig_bkg_dnn_score_ggFvsBKG"
        cuts["score_vbf"] = "sig_bkg_dnn_score_VBFvsBKG"
    else:
        cuts = _get_params(region_setting)
        cuts["score_ggf"] = "boosted_bdt_score"
        cuts["score_vbf"] = "boosted_bdt_vbf_score"
    cuts["score_ggf"] += score_suffix
    cuts["score_vbf"] += score_suffix
    cuts["txbb_morph"] = txbb_morph
    return cuts


def hh4b_boosted_category_1(txbb_morph=True, region_setting="50GeV", optimised_cuts=False, score_suffix=""):
    return Cut(
        name="hh4b_boosted_category_1" + score_suffix,
        params=_category_params(txbb_morph, region_setting, optimised_cuts, score_suffix),
        function=cuts_f.hh4b_boosted_category_1,
    )

def hh4b_boosted_category_vbf(txbb_morph=True, region_setting="50GeV", optimised_cuts=False, score_suffix=""):
    return Cut(
        name="hh4b_boosted_category_vbf" + score_suffix,
        params=_category_params(txbb_morph, region_setting, optimised_cuts, score_suffix),
        function=cuts_f.hh4b_boosted_category_vbf,
    )

def hh4b_boosted_category_2(txbb_morph=True, region_setting="50GeV", optimised_cuts=False, score_suffix=""):
    return Cut(
        name="hh4b_boosted_category_2" + score_suffix,
        params=_category_params(txbb_morph, region_setting, optimised_cuts, score_suffix),
        function=cuts_f.hh4b_boosted_category_2,
    )

def hh4b_boosted_category_3(txbb_morph=True, region_setting="50GeV", optimised_cuts=False, score_suffix=""):
    return Cut(
        name="hh4b_boosted_category_3" + score_suffix,
        params=_category_params(txbb_morph, region_setting, optimised_cuts, score_suffix),
        function=cuts_f.hh4b_boosted_category_3,
    )

def hh4b_boosted_background(txbb_morph=True, region_setting="50GeV", optimised_cuts=False, score_suffix=""):
    return Cut(
        name="hh4b_boosted_background" + score_suffix,
        params=_category_params(txbb_morph, region_setting, optimised_cuts, score_suffix),
        function=cuts_f.hh4b_boosted_background,
    )


hh4b_boosted_signal_region = Cut(
    name="hh4b_boosted_signal_region",
    params={
        # "pnet_cut": 0.65,
        "pnet_cut": 0.60,
        "mass_min": 100,
        "mass_max": 150,
    },
    function=cuts_f.hh4b_boosted_SR_cuts,
)

hh4b_boosted_ttbar_control_region = Cut(
    name="hh4b_boosted_ttbar_control_region",
    params={
        "mass_min": 150,
        "mass_max": 200,
    },
    function=cuts_f.hh4b_boosted_ttbar_CR_cuts,
)
hh4b_control_region_wide_run2 = Cut(
    name="hh4b_control_region",
    params={
        "Run2": True,
        "radius_min": 30,
        "radius_max": 80,
        "higgs_lead_center": 125,
        "higgs_sublead_center": 120,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

hh4b_boosted_qcd_control_region_tot = Cut(
    name="hh4b_boosted_qcd_control_region_tot",
    params={
        "pnet_cut": 0.0,
        "mass_min": 100,
        "mass_max": 150,
        "mass_max_sublead": 200,
    },
    function=cuts_f.hh4b_boosted_qcd_CR_cuts,
)

hh4b_boosted_qcd_control_region_A = Cut(
    name="hh4b_boosted_qcd_control_region_A",
    params={
        "pnet_cut_min": 0.00,
        # "pnet_cut_max": 0.65,
        "pnet_cut_max": 0.60,
        "mass_min_lead": 40,
        "mass_max_lead": 100,
        "mass_min_sublead": 40,
        "mass_max_sublead": 200,
    },
    function=cuts_f.hh4b_boosted_qcd_CR_cuts_X,
)

hh4b_boosted_qcd_control_region_B = Cut(
    name="hh4b_boosted_qcd_control_region_B",
    params={
        # "pnet_cut_min": 0.65,
        "pnet_cut_min": 0.60,
        "pnet_cut_max": 1.0,
        "mass_min_lead": 40,
        "mass_max_lead": 100,
        "mass_min_sublead": 40,
        "mass_max_sublead": 200,
    },
    function=cuts_f.hh4b_boosted_qcd_CR_cuts_X,
)

hh4b_boosted_qcd_control_region_C = Cut(
    name="hh4b_boosted_qcd_control_region_C",
    params={
        "pnet_cut_min": 0.00,
        # "pnet_cut_max": 0.65,
        "pnet_cut_max": 0.60,
        "mass_min_lead": 100,
        "mass_max_lead": 150,
        "mass_min_sublead": 40,
        "mass_max_sublead": 200,
    },
    function=cuts_f.hh4b_boosted_qcd_CR_cuts_X,
)


hh4b_VR1_signal_region = Cut(
    name="hh4b_VR1_signal_region",
    params={
        "radius_min": 0,
        "radius_max": 30,
        "higgs_lead_center": 185,
        "higgs_sublead_center": 180,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

hh4b_VR1_control_region = Cut(
    name="hh4b_VR1_control_region",
    params={
        "radius_min": 30,
        "radius_max": 55,
        "higgs_lead_center": 185,
        "higgs_sublead_center": 180,
    },
    function=cuts_f.hh4b_Rhh_cuts,
)

blinded = Cut(
    name="blinded",
    params={
        "score": 0.9,
        "score_variable": "sig_bkg_dnn_score",
    },
    function=cuts_f.blinding_cuts,
)

hh4b_JetVetoMap = Cut(
    name="hh4b_JetVetoMap",
    params={
        "jet_type": "Jet",
        "pt_type": "pt_default",
    },
    function=get_custom_JetVetoMap_Mask,
)

hh4b_vbf_best_candidates_6_jets_region = Cut(
    name="hh4b_vbf_best_candidates_6_jets_region",
    params={
        "min_mjj": 400,
        "min_deta": 3.5,
        "jet_vbf_coll": "JetGoodVBFEnergyOrdered",
    },
    function=cuts_f.hh4b_vbf_eta_mjj_cuts,
)

hh4b_vbf_best_candidates_6_jets_nokincut_region = Cut(
    name="hh4b_vbf_best_candidates_6_jets_nokincut_region",
    params={
        "min_mjj": 0,
        "min_deta": 0,
        "jet_vbf_coll": "JetGoodVBFEnergyOrdered",
    },
    function=cuts_f.hh4b_vbf_eta_mjj_cuts,
)
def hh4b_sig_bkg_score_cut(thresh):
    return Cut(
    name="hh4b_sig_bkg_score_cut",
    params={
        "discriminator": "sig_bkg_dnn_score",
        "pass": True,
        "threshold": thresh,
    },
    function=cuts_f.sig_bkg_score_cut,
)

def hh4b_vbf_pass_discriminator_region(thresh):
    return Cut(
    name="hh4b_vbf_pass_discriminator_region",
    params={
        "discriminator": "VBF_ggF_score",
        "pass": True,
        "threshold": thresh,
    },
    function=cuts_f.hh4b_vbf_discriminator_cuts,
)

def hh4b_vbf_fail_discriminator_region(thresh):
    return Cut(
        name="hh4b_vbf_fail_discriminator_region",
        params={
            "discriminator": "VBF_ggF_score",
            "pass": False,
            "threshold": thresh,
        },
        function=cuts_f.hh4b_vbf_discriminator_cuts,
    )

hh4b_vbf_2_jets = Cut(
    name="hh4b_vbf_2_jets",
    params={
        "jet_vbf_coll": "JetGoodVBFEnergyOrdered",
    },
    function=cuts_f.hh4b_vbf_2_jets,
)

hh4b_boosted_vbf_region = Cut(
    name="hh4b_boosted_vbf_region",
    params={
        "min_mjj": 300,
        "min_deta": 3.5,
        "jet_vbf_coll": "JetGoodVBFEnergyOrdered",
    },
    function=cuts_f.hh4b_vbf_eta_mjj_cuts,
)

# some of the private sample don't ensure the correct decay, only the final state, so we need to apply a presel cut to select the correct decay
XX4b_presel = Cut(
    name="xx4b_presel",
    params={},
    function=cuts_f.xx4b_cuts,
)


def skimming_cut_list(configs):
    skimlist = [
        eventFlags,
        goldenJson,
        get_nPVgood(1),
    ]
    if not configs["mixeddata"] and not configs["approach"] == "boosted":
        skimlist.append(get_HLTsel())
    if not configs["noL1"] and not configs["mixeddata"] and not configs["boosted"]:
        skimlist.append(get_L1sel())
    return skimlist
