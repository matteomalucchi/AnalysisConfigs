import copy
import logging

import awkward as ak
import correctionlib
import numpy as np
import vector

from configs.HH4b_common.workflow_common import HH4bCommonProcessor
from utils.inference_session_onnx import get_model_session
from utils.reconstruct_higgs_candidates import (
    reconstruct_higgs_from_idx,
)
from utils.spanet_evaluation_functions import get_best_pairings, get_pairing_information

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

    def fill_b_tag_wp_collections(self, variation):
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
                if self.only5jetsbSF:
                    self.events[fieldName] = self.btagging_custom(
                            self.events["JetGood"][:, :5], btagAlgo, wp_value
                    )
                else:
                    self.events[fieldName] = self.btagging_custom(
                            self.events["JetGood"], btagAlgo, wp_value
                    )
                self.events[fieldName] = self.events[fieldName][
                    ak.argsort(self.events[fieldName].pt, axis=1, ascending=False)
                ]

    def btagging_custom(self, Jet, btag, wp):
        return copy.copy(Jet[Jet[btag] > wp])

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        self.fill_b_tag_wp_collections(variation)
        self.events["JetGood"] = self.events["JetGood"][
            ak.argsort(self.events["JetGood"].pt, axis=1, ascending=False)
        ]
        self.events["JetGood"] = self.generate_btag_workingpoints(
            self.events["JetGood"], 5
        )
        self.events["JetGood"] = self.generate_btag_workingpoints(
            self.events["JetGood"], 3
        )

        if self._isMC and not self.spanet:
            matched_jet_higgs_idx_not_none = self.get_true_pairing_and_compare()
        elif self.spanet:
            # apply spanet model to get the pairing prediction for the b-jets from Higgs
            model_session_spanet, input_name_spanet, output_name_spanet = (
                get_model_session(self.spanet, "spanet")
            )

            # compute the pairing information using the SPANET model
            pairing_outputs = get_pairing_information(
                model_session_spanet,
                input_name_spanet,
                output_name_spanet,
                self.events,
                self.max_num_jets_spanet,
                self.spanet_input_name_list,
            )
            # Not needed anymore
            del model_session_spanet
            del input_name_spanet
            del output_name_spanet

            (
                pairing_predictions,
                self.events["best_pairing_probability"],
                self.events["second_best_pairing_probability"],
            ) = get_best_pairings(pairing_outputs)

            # get the probabilities difference between the best and second best jet assignment
            self.events["Delta_pairing_probabilities"] = (
                self.events.best_pairing_probability
                - self.events.second_best_pairing_probability
            )
            # apply arctanh transformation
            self.events["Arctanh_Delta_pairing_probabilities"] = np.arctanh(
                self.events["Delta_pairing_probabilities"]
            )
            arctanh_delta_prob_bin_edges = [
                np.min(self.events.Arctanh_Delta_pairing_probabilities) - 1,
                self.arctanh_delta_prob_bin_edge,
                np.max(
                    [
                        np.max(self.events.Arctanh_Delta_pairing_probabilities) + 1,
                        self.arctanh_delta_prob_bin_edge + 1,
                    ]
                ),
            ]
            self.events["Binned_Arctanh_Delta_pairing_probabilities"] = (
                np.digitize(
                    ak.to_numpy(self.events.Arctanh_Delta_pairing_probabilities),
                    arctanh_delta_prob_bin_edges,
                )
                - 1
            )
            self.events["Padded_Arctanh_Delta_pairing_probabilities"] = np.where(
                self.events.Arctanh_Delta_pairing_probabilities
                > self.arctanh_delta_prob_pad_limit,
                self.pad_value,
                self.events.Arctanh_Delta_pairing_probabilities,
            )

            (
                self.events["HiggsLeading"],
                self.events["HiggsSubLeading"],
                self.events["JetGoodFromHiggsOrdered"],
            ) = reconstruct_higgs_from_idx(self.events.JetGood, pairing_predictions)

            matched_jet_higgs_idx_not_none = self.events.JetGoodFromHiggsOrdered.index
            # Define distance parameter for selection:
            self.events["Rhh"] = np.sqrt(
                (self.events.HiggsLeading.mass - 125) ** 2
                + (self.events.HiggsSubLeading.mass - 120) ** 2
            )
            if self._isMC:
                matched_jet_higgs_idx_not_noneTrue = self.get_true_pairing_and_compare(
                    suffix="True",
                    pairing_predictions=pairing_predictions,
                    pairing_suffix="",
                )

            # if the 5th jet is matched, then the add jet should be order by btag
            # because we want to consider the leading in btag which the pairing discarded
            self.events["btag_order_add_jet"] = ak.any(
                ak.flatten(pairing_predictions, axis=-1) > 3, axis=-1
            )

        self.events["nJetGoodHiggsMatched"] = ak.num(
            self.events.JetGoodHiggsMatched, axis=1
        )
        self.events["nJetGoodMatched"] = ak.num(self.events.JetGoodMatched, axis=1)

        # HT : scalar sum of all jets with pT > 25 GeV inside | η | < 2.5
        self.events["HT"] = ak.sum(self.events.JetGood.pt, axis=1)
        if any(np.isnan(self.events["HT"])):
            raise Exception(
                "NaN values in the column HT",
                f"Data: {self.events['HT']}"
            )
        if any(np.isnan(self.events["nJetGood"])):
            raise Exception(
                "NaN values in the column HT",
                f"Data: {self.events['nJetGood']}"
            )
