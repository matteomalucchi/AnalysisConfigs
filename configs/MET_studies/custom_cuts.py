import awkward as ak
from pocket_coffea.lib.cut_definition import Cut

def dimuon(events, params, year, sample, **kwargs):

    # Masks for same-flavor (SF) and opposite-sign (OS)
    SF = ((events.nMuonGood == 2) & (events.nElectronGood == 0))
    OS = events.ll.charge == 0

    mask = (
        (ak.firsts(events.MuonGood.pt) > params["pt_leading_muon"])
        & OS & SF
        & (events.ll.mass > params["mll"]["low"])
        & (events.ll.mass < params["mll"]["high"])
    )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

def PV_presel_cuts(events, params, **kwargs):
    mask=  abs(events.PV.z - events.GenVtx.z) < params["distance"]
    return ak.where(ak.is_none(mask), False, mask)
    

dimuon_presel = Cut(
    name="dilepton",
    params={
        "pt_leading_muon": 20,
        "mll": {'low': 80, 'high': 100},
    },
    function=dimuon,
)

at_least_one_jet = Cut(
    name="at_least_one_jet",
    params={},
    function=lambda events, params, year, sample, **kwargs: events.nJetPuppiMET > 0,
)


PV_presel = Cut(
    name="PV_presel",
    params={
        "distance": 0.2
    },
    function=PV_presel_cuts,
)