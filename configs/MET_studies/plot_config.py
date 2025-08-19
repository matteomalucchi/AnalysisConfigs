var_dict = {
    r"MET $p_{\mathrm{T}}$ [GeV]": [
        "PuppiMET_pt",
        "PuppiMETPNet_pt",
        "PuppiMETPNetPlusNeutrino_pt",
    ],
    r"MET $\phi$": ["PuppiMET_phi", "PuppiMETPNet_phi", "PuppiMETPNetPlusNeutrino_phi"],
    r"MET_MinusMuons $p_{\mathrm{T}}$ [GeV]": [
        "PuppiMET_MuonGood_pt",
        "PuppiMETPNet_MuonGood_pt",
        "PuppiMETPNetPlusNeutrino_MuonGood_pt",
    ],
    r"MET_MinusMuons $\phi$": [
        "PuppiMET_MuonGood_phi",
        "PuppiMETPNet_MuonGood_phi",
        "PuppiMETPNetPlusNeutrino_MuonGood_phi",
    ],
    r"$-u_{||}/q_{T}$": [
        "PuppiMET_MuonGood_response",
        "PuppiMETPNet_MuonGood_response",
        "PuppiMETPNetPlusNeutrino_MuonGood_response",
    ],
    r"$u_{||}+q_{T}$ [GeV]": [
        "PuppiMET_MuonGood_u_paral_predict",
        "PuppiMETPNet_MuonGood_u_paral_predict",
        "PuppiMETPNetPlusNeutrino_MuonGood_u_paral_predict",
    ],
    r"$u_{\perp}$ [GeV]": [
        "PuppiMET_MuonGood_u_perp_predict",
        "PuppiMETPNet_MuonGood_u_perp_predict",
        "PuppiMETPNetPlusNeutrino_MuonGood_u_perp_predict",
    ],
    r"MET $p_{\mathrm{T}}$ [GeV]": ["PuppiMET_pt", "PuppiMET_MuonGood_pt"],
    r"METPNet $p_{\mathrm{T}}$ [GeV]": ["PuppiMETPNet_pt", "PuppiMETPNet_MuonGood_pt"],
    r"METPNetPlusNeutrino $p_{\mathrm{T}}$ [GeV]": [
        "PuppiMETPNetPlusNeutrino_pt",
        "PuppiMETPNetPlusNeutrino_MuonGood_pt",
    ],
}
ranges = {
    r"MET $p_{\mathrm{T}}$ [GeV]": (0, 200),
    r"MET $\phi$": (-3.14, 3.14),
    r"MET_MinusMuons $p_{\mathrm{T}}$ [GeV]": (0, 200),
    r"MET_MinusMuons $\phi$": (-3.14, 3.14),
    r"$-u_{||}/q_{T}$": (-1, 2),
    r"$u_{||}+q_{T}$ [GeV]": (-200, 200),
    r"$u_{\perp}$ [GeV]": (-200, 200),
    r"MET $p_{\mathrm{T}}$ [GeV]": (0, 200),
    r"METPNet $p_{\mathrm{T}}$ [GeV]": (0, 200),
    r"METPNetPlusNeutrino $p_{\mathrm{T}}$ [GeV]": (0, 200),
}
log_dict = {
    r"MET $p_{\mathrm{T}}$ [GeV]": True,
    r"MET $\phi$": False,
    r"MET_MinusMuons $p_{\mathrm{T}}$ [GeV]": True,
    r"MET_MinusMuons $\phi$": False,
    r"$-u_{||}/q_{T}$": True,
    r"$u_{||}+q_{T}$ [GeV]": True,
    r"$u_{\perp}$ [GeV]": True,
    r"MET $p_{\mathrm{T}}$ [GeV]": True,
    r"METPNet $p_{\mathrm{T}}$ [GeV]": True,
    r"METPNetPlusNeutrino $p_{\mathrm{T}}$ [GeV]": True,
}
color_list = [
    "black",
    "red",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]
var_name_dict={
                'u_perp_resolution': r'$\sigma u_{\perp}$ [GeV]',
            'u_perp_scaled_resolution': r'$\sigma u_{\perp} scaled$ [GeV]',
            'u_par_resolution': r'$\sigma u_{\parallel}$ [GeV]',
            'u_par_scaled_resolution': r'$\sigma u_{\parallel} scaled$ [GeV]',
            'R': r'-$<u_{||}>/<q_{T}>$',
}