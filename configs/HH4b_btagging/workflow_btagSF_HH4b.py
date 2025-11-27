import awkward as ak
import correctionlib
import numpy as np
import vector
from pocket_coffea.lib.deltaR_matching import object_matching
from pocket_coffea.workflows.base import BaseProcessorABC

from configs.HH4b_common.custom_object_preselection_common import jet_selection_nopu, lepton_selection
from utils.basic_functions import add_fields
from utils.parton_matching_function import get_parton_last_copy
from utils.reconstruct_higgs_candidates import (
    get_jets_no_higgs_from_idx,
    reconstruct_higgs_from_provenance,
    )
from configs.HH4b_common.workflow_common import HH4bCommonProcessor

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

vector.register_awkward()

era_dict = {
    "2022_preEE_C": 0,
    "2022_preEE_D": 1,
    "2022_postEE_E": 2,
    "2022_postEE_F": 3,
    "2022_postEE_G": 4,
    "2023_preBPix_Cv1": 5,
    "2023_preBPix_Cv2": 6,
    "2023_preBPix_Cv3": 7,
    "2023_preBPix_Cv4": 8,
    "2023_postBPix_Dv1": 9,
    "2023_postBPix_Dv2": 10,
    "2022_preEE_MC": -1,
    "2022_postEE_MC": -2,
    "2023_preBPix_MC": -3,
    "2023_postBPix_MC": -4,
}

year_dict = {
    "2022_preEE": 0,
    "2022_postEE": 1,
    "2023_preBPix": 2,
    "2023_postBPix": 3,
}


class HH4bbtagWPefficiencyProcessor(HH4bCommonProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)

        for key, value in self.workflow_options.items():
            setattr(self, key, value)

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation)

    def define_common_variables_after_presel(self, variation):
        super().define_common_variables_before_presel(variation=variation)

        self.events["JetGood"] = ak.with_field(
            self.events["JetGood"], abs(self.events["JetGood"]["eta"]), "abseta"
        )

        bTaggers = ["btagDeepFlavB", "btagPNetB", "btagRobustParTAK4B"]
        bTaggerCorrlibNames = ["deepJet", "particleNet", "robustParticleTransformer"]
        cset = correctionlib.CorrectionSet.from_file(self.params["jet_scale_factors"]["btagSF"][self._year]["file"])
        for btagName, btagAlgo in zip(bTaggerCorrlibNames, bTaggers):
            for wp in ["L", "M", "T", "XT", "XXT"]:
                fieldName = "BJetGood_" + btagName + "_" + wp
                wp_value = cset[btagName + "_wp_values"].evaluate(wp)
                self.events[fieldName] = self.btagging_custom(
                    self.events["JetGood"], btagAlgo, wp_value
                )

    def btagging_custom(self, Jet, btag, wp):
        return Jet[Jet[btag] > wp]

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        self.events["JetGood"] = self.generate_btag_workingpoints(
            self.events["JetGood"], 5
        )
        self.events["JetGood"] = self.generate_btag_workingpoints(
            self.events["JetGood"], 3
        )

        if self._isMC and not self.spanet:
            matched_jet_higgs_idx_not_none = self.get_true_pairing_and_compare()

        self.events["nJetGoodHiggsMatched"] = ak.num(
            self.events.JetGoodHiggsMatched, axis=1
        )
        self.events["nJetGoodMatched"] = ak.num(self.events.JetGoodMatched, axis=1)

        # HT : scalar sum of all jets with pT > 25 GeV inside | η | < 2.5
        self.events["HT"] = ak.sum(self.events.JetGood.pt, axis=1)
        if any(np.isnan(self.events["HT"])):
            raise Exception(
                f"NaN values in the column HT",
                f"Data: {self.events['HT']}"
            )
        if any(np.isnan(self.events["nJetGood"])):
            raise Exception(
                f"NaN values in the column HT",
                f"Data: {self.events['nJetGood']}"
            )
