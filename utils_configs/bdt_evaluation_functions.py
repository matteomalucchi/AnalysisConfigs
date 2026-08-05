import numpy as np
import awkward as ak
import pandas as pd
import xgboost as xgb
# from collections import defaultdict

# from utils_configs.spanet_evaluation_functions import (
#     define_spanet_pairing_inputs,
# )



def disc_TXbb(txbb_array):
    bins = np.array([0, 0.8, 0.9, 0.94, 0.97, 0.99, 1])
    is_valid = ~ak.is_none(txbb_array)
    digitized = np.digitize(ak.fill_none(txbb_array, -1.0), bins)
    digitized = ak.Array(digitized)
    bin_indices = ak.mask(digitized, is_valid)
    return bin_indices


def to_col(arr, handle_none=None):
    if handle_none == "inf":
        arr = ak.fill_none(arr, np.inf)
    if handle_none == "99999":
        arr = ak.fill_none(arr, -99999)
    if handle_none == "0":
        arr = ak.fill_none(arr, 0)
    if handle_none == "NaN":
        arr = ak.fill_none(arr, np.nan)

    col = ak.to_numpy(arr, allow_missing=True)
    if np.ma.isMaskedArray(col):
        col.data[col.data == -999] = -99999
    else:
        col = np.where(col == -999, -99999, col)
    return col

def get_default_bdt_inputs(events):
    inputs = pd.DataFrame(
        {
            # dihiggs system
            "HHPt": to_col(events.HH.pt),
            "HHeta": to_col(events.HH.eta, handle_none="inf"),
            "HHmass": to_col(events.HH.mass),
            # met in the event
            "MET": to_col(events.PuppiMET.pt),
            # key_map("MET_RAW"): met_raw_pt),
            # fatjet tau32
            "H1T32": to_col(events.HiggsLeading.Tau3OverTau2, "99999"),
            "H2T32": to_col(events.HiggsSubLeading.Tau3OverTau2, "99999"),
            # fatjet mass
            "H1Mass": to_col(events.HiggsLeading.mass, "99999"),
            # fatjet kinematics
            "H1Pt": to_col(events.HiggsLeading.pt, "99999"),
            "H2Pt": to_col(events.HiggsSubLeading.pt, "99999"),
            "H1eta": to_col(events.HiggsLeading.eta, "99999"),
            # xbb
            "H1Xbb": to_col(events.HiggsLeading.btagBBTXbb),
            # ratios
            "H1Pt_HHmass": to_col(events.HiggsLeading.divHHmass),
            "H2Pt_HHmass": to_col(events.HiggsSubLeading.divHHmass),
            "H1Pt/H2Pt": to_col(events.HiggsLeadingByHiggsSubLeadingPt, "99999"),
            # vbf mjj and eta_jj
            "VBFjjMass": to_col(events.mjjJetGoodVBFEnergyOrdered, "NaN"),
            "VBFjjDeltaEta": to_col(events.detaJetGoodVBFEnergyOrdered),
            # AK4JetAway
            "H1AK4JetAway1dR": to_col(events.HiggsLeading.dRclosestVBF, "0"),
            "H2AK4JetAway2dR": to_col(events.HiggsSubLeading.dRclosestVBF, "0"),
            "H1AK4JetAway1mass": to_col(events.HiggsLeading.massclosestVBF, "NaN"),
            "H2AK4JetAway2mass": to_col(events.HiggsSubLeading.massclosestVBF, "NaN"),
        }
    )
    return inputs

def evaluate_bdt(bdt_model, bdt_events):
    bdt_model_xgb = xgb.XGBClassifier()
    bdt_model_xgb.load_model(bdt_model)
    bdt_scores = bdt_model_xgb.predict_proba(bdt_events)
    bg_tot = np.sum(bdt_scores[:, 2:], axis=1)
    bdt_score = bdt_scores[:, 0] / (bdt_scores[:, 1] + bg_tot)
    bdt_vbf_score = bdt_scores[:, 1] / (bdt_scores[:, 1] + bdt_scores[:, 2] + bdt_scores[:, 3])
    return bdt_score, bdt_vbf_score
