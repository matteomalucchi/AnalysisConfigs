import awkward as ak
import sys
import numpy as np

from configs.HH4b_common.workflow_common import HH4bCommonProcessor
from utils_configs.reconstruct_higgs_candidates import (
    possible_higgs_reco,
    distance_pt_func
)
from utils_configs.basic_functions import add_fields

class HH4bCutflowProcessor(HH4bCommonProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
    
    # def modified_run2_matching_algorithm(self, jet_collection):  # -> ak.Array
    #     # implement the Run 2 pairing algorithm

    #     # this index are referred to the index of the JetGoodHiggs not the index of the jets
    #     comb_idx = np.array([[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]])

    #     higgs_candidates_unflatten = possible_higgs_reco(jet_collection, comb_idx)
    #     distance, max_pt = distance_pt_func(
    #         higgs_candidates_unflatten,
    #         1.04,
    #     )

    #     # order by distance and find the index of the smallest distance
    #     dist_order_idx = ak.argsort(distance, axis=1, ascending=True)
    #     dist_order = ak.sort(distance, axis=1, ascending=True)

    #     delta_dhh = abs(dist_order[:, 0] - dist_order[:, 1])

    #     # Do the pt ordering for all events and fill later only the ones where the dhh is smaller than 30
    #     pt_order_idx = ak.argsort(max_pt, axis=1, ascending=False)
    #     # pt_order = ak.sort(max_pt_list, axis=1, ascending=False)  # Maybe useful for debugging

    #     # Find idxes where the delta_dhh is smaller than 30 and if so, fill in by pt ordering.
    #     min_idx = ak.where(delta_dhh > 30, dist_order_idx[:, 0], pt_order_idx[:, 0])

    #     # get higgs candidates
    #     higgs_1 = add_fields(
    #         higgs_candidates_unflatten[np.arange(len(jet_collection)), min_idx][:, 0]
    #     )
    #     higgs_2 = add_fields(
    #         higgs_candidates_unflatten[np.arange(len(jet_collection)), min_idx][:, 1]
    #     )

    #     # get leadining and subleading higgs
    #     higgs_leading_index = np.where(higgs_1.pt > higgs_2.pt, 0, 1)
    #     higgs_lead = ak.where(higgs_leading_index == 0, higgs_1, higgs_2)
    #     higgs_sub = ak.where(higgs_leading_index == 0, higgs_2, higgs_1)

    #     # expand index to 3x2x2 for later
    #     higgs_leading_index_expanded = higgs_leading_index[
    #         :,
    #         np.newaxis,
    #         np.newaxis,
    #         np.newaxis,
    #     ] * np.ones((3, 2, 2))

    #     # expand the comb_idx
    #     comb_idx_tile = np.tile(comb_idx, (len(min_idx), 1, 1))
    #     comb_idx_reshape = np.reshape(comb_idx_tile, (len(min_idx), 3, 2, 2))
    #     # order indx according to the higgs candidate pt
    #     comb_idx_order = ak.to_numpy(
    #         np.where(
    #             higgs_leading_index_expanded == 0,
    #             comb_idx_reshape,
    #             comb_idx_reshape[:, :, ::-1, :],
    #         )
    #     )
    #     # reshape to 3x4
    #     comb_idx_order_reshape = np.reshape(comb_idx_order, (len(min_idx), 3, 4))
    #     # get the best index comb
    #     best_idx_expanded = ak.Array(
    #         comb_idx_order_reshape[np.arange(len(min_idx)), min_idx]
    #     )
    #     best_idx_origshape = ak.Array(comb_idx_order[np.arange(len(min_idx)), min_idx])

    #     # get the jets ordered like higg1_jet1, higg1_jet2, higg2_jet1, higg2_jet2
    #     jets_list = []
    #     for i in ((0, 1), (2, 3)):
    #         for j in (i, i[::-1]):
    #             jets_list.append(
    #                 ak.unflatten(
    #                     ak.where(
    #                         jet_collection[
    #                             np.arange(len(min_idx)), best_idx_expanded[:, i[0]]
    #                         ].pt
    #                         > jet_collection[
    #                             np.arange(len(min_idx)), best_idx_expanded[:, i[1]]
    #                         ].pt,
    #                         jet_collection[
    #                             np.arange(len(min_idx)), best_idx_expanded[:, j[0]]
    #                         ],
    #                         jet_collection[
    #                             np.arange(len(min_idx)), best_idx_expanded[:, j[1]]
    #                         ],
    #                     ),
    #                     1,
    #                 )
    #             )
    #     # concatenate the jets
    #     jets_ordered = ak.with_name(
    #         ak.concatenate(jets_list, axis=1),
    #         name="PtEtaPhiMLorentzVector",
    #     )

    #     return (
    #         best_idx_origshape,
    #         delta_dhh,
    #         higgs_lead,
    #         higgs_sub,
    #         jets_ordered,
    #     )
        
        
        
    def modified_run2_matching_algorithm(self, jet_collection):  # -> ak.Array

        # --------------------------------------------------
        # Step 1: build a safe jet collection (>=4 jets)
        # --------------------------------------------------
        n_jets = ak.num(jet_collection, axis=1)
        valid_evt = n_jets >= 4

        # find ONE real event with >= 4 jets (guaranteed in HH analyses)
        template_evt_idx = ak.argmax(valid_evt)
        template_event = jet_collection[template_evt_idx]
        # create a template with the template event repeated
        template_jets=ak.broadcast_arrays(template_event, jet_collection)[0]
        breakpoint()
        # replace bad events with the template event
        jet_collection_safe = ak.where(valid_evt,jet_collection,template_jets,        )

        # --------------------------------------------------
        # ORIGINAL Run-2 algorithm (unchanged)
        # --------------------------------------------------

        comb_idx = np.array([
            [[0, 1], [2, 3]],
            [[0, 2], [1, 3]],
            [[0, 3], [1, 2]],
        ])

        higgs_candidates_unflatten = possible_higgs_reco(
            jet_collection_safe, comb_idx
        )

        distance, max_pt = distance_pt_func(
            higgs_candidates_unflatten,
            1.04,
        )

        dist_order_idx = ak.argsort(distance, axis=1, ascending=True)
        dist_order = ak.sort(distance, axis=1, ascending=True)

        delta_dhh = abs(dist_order[:, 0] - dist_order[:, 1])

        pt_order_idx = ak.argsort(max_pt, axis=1, ascending=False)

        min_idx = ak.where(
            delta_dhh > 30,
            dist_order_idx[:, 0],
            pt_order_idx[:, 0],
        )

        higgs_1 = add_fields(
            higgs_candidates_unflatten[
                np.arange(len(jet_collection_safe)), min_idx
            ][:, 0]
        )
        higgs_2 = add_fields(
            higgs_candidates_unflatten[
                np.arange(len(jet_collection_safe)), min_idx
            ][:, 1]
        )

        higgs_leading_index = np.where(higgs_1.pt > higgs_2.pt, 0, 1)
        higgs_lead = ak.where(higgs_leading_index == 0, higgs_1, higgs_2)
        higgs_sub = ak.where(higgs_leading_index == 0, higgs_2, higgs_1)

        higgs_leading_index_expanded = (
            higgs_leading_index[:, None, None, None]
            * np.ones((3, 2, 2))
        )

        comb_idx_tile = np.tile(comb_idx, (len(min_idx), 1, 1))
        comb_idx_reshape = comb_idx_tile.reshape(len(min_idx), 3, 2, 2)

        comb_idx_order = ak.to_numpy(
            np.where(
                higgs_leading_index_expanded == 0,
                comb_idx_reshape,
                comb_idx_reshape[:, :, ::-1, :],
            )
        )

        comb_idx_order_reshape = comb_idx_order.reshape(len(min_idx), 3, 4)

        best_idx_expanded = ak.Array(
            comb_idx_order_reshape[np.arange(len(min_idx)), min_idx]
        )
        best_idx_origshape = ak.Array(
            comb_idx_order[np.arange(len(min_idx)), min_idx]
        )

        jets_list = []
        for i in ((0, 1), (2, 3)):
            for j in (i, i[::-1]):
                jets_list.append(
                    ak.unflatten(
                        ak.where(
                            jet_collection_safe[
                                np.arange(len(min_idx)), best_idx_expanded[:, i[0]]
                            ].pt
                            > jet_collection_safe[
                                np.arange(len(min_idx)), best_idx_expanded[:, i[1]]
                            ].pt,
                            jet_collection_safe[
                                np.arange(len(min_idx)), best_idx_expanded[:, j[0]]
                            ],
                            jet_collection_safe[
                                np.arange(len(min_idx)), best_idx_expanded[:, j[1]]
                            ],
                        ),
                        1,
                    )
                )

        jets_ordered = ak.with_name(
            ak.concatenate(jets_list, axis=1),
            name="PtEtaPhiMLorentzVector",
        )

        return (
            best_idx_origshape,
            delta_dhh,
            higgs_lead,
            higgs_sub,
            jets_ordered,
        )




    def process_extra_after_presel(self, variation):  # -> ak.Array
        super().process_extra_after_presel(variation=variation)
        # pass
        
        
        
        
        # reconstruct the higgs candidates for Run2 method
        # if self.run2:
        #     (
        #         pairing_predictions,
        #         self.events["delta_dhh"],
        #         self.events["HiggsLeadingRun2"],
        #         self.events["HiggsSubLeadingRun2"],
        #         self.events["JetGoodFromHiggsOrderedRun2"],
        #     ) = self.modified_run2_matching_algorithm(self.events["JetGoodHiggs"])

        #     self.events["Rhh_Run2"] = np.sqrt(
        #         (self.events.HiggsLeadingRun2.mass - 125) ** 2
        #         + (self.events.HiggsSubLeadingRun2.mass - 120) ** 2
        #     )

        #     # if the 5th jet is matched, then the add jet should be order by btag
        #     # because we want to consider the leading in btag which the pairing discarded
        #     # (useless for Run2 pairing because it's always 4 jets)
        #     self.events["btag_order_add_jet"] = ak.any(
        #         ak.flatten(pairing_predictions, axis=-1) > 3, axis=-1
        #     )
        
