from collections.abc import Iterable
import numpy as np
import awkward as ak

import configs.VBF_HH4b.custom_cut_functions as cuts_f
from pocket_coffea.lib.cut_definition import Cut

vbf_hh4b_presel = Cut(
    name="hh4b",
    params={
        "njetgood": 4,
        "njetvbf": 6,
        "pt_jet0": 80,
        "pt_jet1": 60,
        "pt_jet2": 45,
        "pt_jet3": 35,
        "mean_pnet_jet": 0.65,
        "tight_cuts": False,
        "pt_type": "pt_default",
    },
    function=cuts_f.vbf_hh4b_presel_cuts,
)

vbf_hh4b_presel_tight = Cut(
    name="hh4b",
    params={
        "njetgood": 4,
        "njetvbf": 6,
        "pt_jet0": 80,
        "pt_jet1": 60,
        "pt_jet2": 45,
        "pt_jet3": 35,
        "mean_pnet_jet": 0.65,
        "tight_cuts": True,
        "pt_type": "pt_default",
    },
    function=cuts_f.vbf_hh4b_presel_cuts,
)
