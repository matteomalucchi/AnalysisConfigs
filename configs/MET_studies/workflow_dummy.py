import awkward as ak
import numpy as np
import vector
import copy

vector.register_awkward()

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.jets import met_correction_after_jec
from pocket_coffea.lib.leptons import lepton_selection, get_dilepton
from pocket_coffea.lib.deltaR_matching import object_matching, deltaR_matching_nonunique

from configs.jme.workflow import QCDBaseProcessor
from configs.jme.custom_cut_functions import jet_selection_nopu
from utils_configs.basic_functions import add_fields
from configs.MET_studies.custom_object_preselections import (
    jet_type1_selection,
    muon_selection_custom,
    low_pt_jet_type1_selection,
)


class METProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    def apply_object_preselection(self, variation):
        pass
    def process_extra_after_presel(self, variation) -> ak.Array:
        breakpoint()

    def count_objects(self, variation):
        pass