import numpy as np
import mplhep as hep


hep.style.use("CMS")
color_dict = list(hep.style.CMS["axes.prop_cycle"])
color_list = [cycle["color"] for cycle in color_dict] + [
    "black",
    "darkgreen",
    "blue",
    "lightgreen",
]
reshuffled_idx = [0, 1, 3, 4, 9, 8, 2, 5, 6, 7]
color_list = [color_list[i] for i in reshuffled_idx]
fmt_list = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">"]

met_list = [
    "RawPuppiMET",
    # "RawPuppiMET-Type1",
    "RawPuppiMET-Type1CorrMET",
    "RawPuppiMET-Type1JEC",
    "RawPuppiMET-Type1PNetCorrMET",
    "RawPuppiMET-Type1PNetPlusNeutrinoCorrMET",
    "PuppiMET",
    # "PuppiMET-Type1",
    "PuppiMET-Type1CorrMET",
    "PuppiMET-Type1JEC",
    "PuppiMET-Type1PNetCorrMET",
    "PuppiMET-Type1PNetPlusNeutrinoCorrMET",
]
met_dict_names = {
    met: {
        "color": color,
        "histtype": "step",
        "fmt": fmt,
        "is_reference": True if met == "RawPuppiMET" else False,
    }
    for met, color, fmt in zip(met_list, color_list, fmt_list)
}
# u_dict_names = {f"u{met}": infos for met, infos in met_dict_names.items()}

plot_met_list=[
    "RawPuppiMET",
    # "RawPuppiMET-Type1",
    # "RawPuppiMET-Type1JEC",
    "RawPuppiMET-Type1CorrMET",
    "RawPuppiMET-Type1PNetCorrMET",
    "RawPuppiMET-Type1PNetPlusNeutrinoCorrMET",
    "PuppiMET",
    # "PuppiMET-Type1",
    # "PuppiMET-Type1JEC",
    # "PuppiMET-Type1CorrMET",
    # "PuppiMET-Type1PNet",
    # "PuppiMET-Type1PNetPlusNeutrino",
]
# keep only the key in plot_met_list
met_dict_names= {met: infos for met, infos in met_dict_names.items() if met in plot_met_list}
u_dict_names= {f"u{met}": infos for met, infos in met_dict_names.items()}




total_var_dict = {
    "MET_comparison_pt": {
        "plot_name": r"MET $p_{\mathrm{T}}$ [GeV]",
        "variables": [met + "_pt" for met in met_dict_names],
        "range": (0, 200),
        "log": True,
        # "ratio_label": "MET / RawPuppiMET",
        "reference": "RawPuppiMET_pt",
        "colors": [style["color"] for style in met_dict_names.values()],
    },
    "MET_comparison_phi": {
        "plot_name": r"MET $\phi$",
        "variables": [met + "_phi" for met in met_dict_names],
        "range": (-3.14, 3.14),
        "log": False,
        # "ratio_label": "MET / RawPuppiMET",
        "reference": "RawPuppiMET_phi",
        "colors": [style["color"] for style in met_dict_names.values()],
    },
    "hadronic_recoil_comparison_pt": {
        "plot_name": r"u $p_{\mathrm{T}}$ [GeV]",
        "variables": ["u" + met + "_pt" for met in met_dict_names],
        "range": (0, 200),
        "log": True,
        # "ratio_label": "MET / RawPuppiMET",
        "reference": "uRawPuppiMET_pt",
        "colors": [style["color"] for style in met_dict_names.values()],
    },
    "hadronic_recoil_comparison_phi": {
        "plot_name": r"u $\phi$",
        "variables": ["u" + met + "_phi" for met in met_dict_names],
        "range": (-3.14, 3.14),
        "log": False,
        # "ratio_label": "MET / RawPuppiMET",
        "reference": "uRawPuppiMET_phi",
        "colors": [style["color"] for style in met_dict_names.values()],
    },
    "response_comparison": {
        "plot_name": r"$-u_{||}/q_{T}$",
        "variables": ["u" + met + "_response" for met in met_dict_names],
        "range": (-2, 2),
        "log": True,
        # "ratio_label": "MET / RawPuppiMET",
        "reference": "uRawPuppiMET_response",
        "colors": [style["color"] for style in met_dict_names.values()],
    },
    "u_paral_predict_comparison": {
        "plot_name": r"$u_{||}+q_{T}$ [GeV]",
        "variables": ["u" + met + "_u_paral_predict" for met in met_dict_names],
        "range": (-200, 200),
        "log": True,
        # "ratio_label": "MET / RawPuppiMET",
        "reference": "uRawPuppiMET_u_paral_predict",
        "colors": [style["color"] for style in met_dict_names.values()],
    },
    "u_perp_predict_comparison": {
        "plot_name": r"$u_{\perp}$ [GeV]",
        "variables": ["u" + met + "_u_perp_predict" for met in met_dict_names],
        "range": (-200, 200),
        "log": True,
        # "ratio_label": "MET / RawPuppiMET",
        "reference": "uRawPuppiMET_u_perp_predict",
        "colors": [style["color"] for style in met_dict_names.values()],
    },
}
met_hadronic_recoil_dict = {
    f"{met}_pt": {
        "plot_name": met + r" $p_{\mathrm{T}}$ [GeV]",
        "variables": [f"{met}_pt", f"u{met}_pt"],
        "range": (0, 200),
        "log": True,
        # "ratio_label": "u / MET",
        "reference": f"{met}_pt",
        "colors": [style["color"] for style in met_dict_names.values()],
    }
    for met in met_dict_names
}
total_var_dict.update(met_hadronic_recoil_dict)

response_var_name_dict = {
    "R": r"$-u_{\parallel}/q_{T}$",
    "R_mean": r"-$<u_{\parallel}/q_{T}>$",
    "R_quantile_resolution": r"q $\sigma(u_{\parallel}/q_{T})$",
    "R_stddev_resolution": r"std dev $\sigma(u_{\parallel}/q_{T})$",
    "u_perp": r"$u_{\perp}$ [GeV]",
    "u_perp_scaled": r"$u_{\perp}$ / (-$<u_{\parallel}/q_{T}>$)",
    "u_perp_mean": r"$<u_{\perp}>$ [GeV]",
    "u_perp_scaled_mean": r"$<u_{\perp}> / (-<u_{\parallel}/q_{T}>)$",
    "u_perp_quantile_resolution": r"q $\sigma(u_{\perp})$ [GeV]",
    "u_perp_scaled_quantile_resolution": r"q $\sigma(u_{\perp}) / (-<u_{\parallel}/q_{T}>)$",
    "u_perp_stddev_resolution": r"std dev $\sigma({u_{\perp}})$ [GeV]",
    "u_perp_scaled_stddev_resolution": r"std dev $\sigma({u_{\perp}}) / (-<u_{\parallel}/q_{T}>)$",
    "u_paral": r"$u_{\parallel}+q_{T}$ [GeV]",
    "u_paral_scaled": r"$(u_{\parallel}+q_{T})$ / (-$<u_{\parallel}/q_{T}>$)",
    "u_paral_mean": r"$<u_{\parallel}+q_{T}>$ [GeV]",
    "u_paral_scaled_mean": r"$<u_{\parallel}+q_{T}> / (-<u_{\parallel}/q_{T}>)$",
    "u_paral_quantile_resolution": r"q $\sigma(u_{\parallel}+q_{T})$ [GeV]",
    "u_paral_scaled_quantile_resolution": r"q $\sigma(u_{\parallel}+q_{T}) / (-<u_{\parallel}/q_{T}>)$",
    "u_paral_stddev_resolution": r"std dev $\sigma({u_{\parallel}+q_{T}})$ [GeV]",
    "u_paral_scaled_stddev_resolution": r"std dev $\sigma({u_{\parallel}+q_{T}}) / (-<u_{\parallel}/q_{T}>)$",
}


qT_bins = np.array(
    [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        80,
        90,
        100,
        125,
        150,
        200,
        300,
        400,
        500,
    ]
)

N_bins=40

R_bin_edges = np.linspace(-2, 2, N_bins)
u_bin_edges = np.linspace(-200, 200, N_bins)