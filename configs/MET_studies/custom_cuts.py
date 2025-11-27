import awkward as ak
from pocket_coffea.lib.cut_definition import Cut

from configs.MET_studies.custom_cuts_functions import dimuon, PV_presel_cut

dimuon_presel = Cut(
    name="dilepton",
    params={"pt_leading_muon": 26, "mll": {"low": 80, "high": 100}, "delta_r": 0.3},
    function=dimuon,
)

at_least_one_jet = Cut(
    name="at_least_one_jet",
    params={},
    function=lambda events, params, year, sample, **kwargs: events.nJetGood > 0,
)


PV_presel = Cut(
    name="PV_presel",
    params={"distance": 0.2},
    function=PV_presel_cut,
)
