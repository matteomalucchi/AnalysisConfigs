import awkward as ak


def dimuon(events, params, year, sample, **kwargs):

    # Masks for same-flavor (SF) and opposite-sign (OS)
    SF = (events.nMuonGood >= 2) & (events.nElectronGood == 0)
    OS = events.ll.charge == 0

    mask = (
        (ak.firsts(events.MuonGood.pt) > params["pt_leading_muon"])
        & OS
        & SF
        & (events.ll.mass > params["mll"]["low"])
        & (events.ll.mass < params["mll"]["high"])
        & (events.ll.deltaR > params["delta_r"])
    )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def PV_presel_cut(events, params, **kwargs):
    mask = abs(events.PV.z - events.GenVtx.z) < params["distance"]
    return ak.where(ak.is_none(mask), False, mask)

