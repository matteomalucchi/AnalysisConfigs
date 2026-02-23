import awkward as ak
import copy
import numpy as np

from utils.custom_cut_functions import custom_jet_selection
from utils.basic_functions import add_fields
from configs.HH4b_common.workflow_common import HH4bCommonProcessor
from utils.reconstruct_higgs_candidates import get_lead_mjj_jet_pair


class VBFHH4bProcessor(HH4bCommonProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)

    def process_extra_after_skim(self):
        super().process_extra_after_skim()

        if (
            self.vbf_parton_matching
            and self._isMC
            and "VBFHHto4B" in self.events.metadata["dataset"]
        ):
            # do truth matching to get VBF-jets
            self.do_vbf_parton_matching(
                which_vbf_quark=self.which_vbf_quark, jet_collection="Jet"
            )
        else:
            self.dummy_provenance_vbf()

        # # check that provenance fields are ortherogonal
        # if self.vbf_analysis and self._isMC:
        #     provenance_higgs = self.events.Jet.provenance_higgs
        #     provenance_vbf = self.events.Jet.provenance_vbf

        #     mask_both_not_none = (
        #         ~ak.is_none(provenance_higgs, axis=1)
        #         & ~ak.is_none(provenance_vbf, axis=1)
        #     )
        #     n_jets_both_not_none = ak.sum(mask_both_not_none, axis=1)
        #     n_events_with_jets_both_not_none = ak.sum(n_jets_both_not_none > 0)

        #     # if n_events_with_jets_both_not_none > 0:
        #     #     raise ValueError(
        #     #         f"Some jets are matched to both Higgs and VBF quarks in {n_events_with_jets_both_not_none} events!"
        #     )

        self.def_provenance_field()
        self.define_jet_collections()

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation=variation)
        if self.vbf_analysis:

            # get idx of good jets after preselection
            self.events["JetGoodClip"] = copy.copy(
                self.events.JetGood[:, : self.max_num_jets_good]
            )
            jet_good_idx_not_none = self.events.JetGoodClip.index

            # find the remaining jets to define the vbf candidates
            self.events["JetVBF"] = self.get_jets_not_from_idx(jet_good_idx_not_none)
            self.events["JetGoodVBF"], mask_jet_vbf = custom_jet_selection(
                self.events,
                "JetVBF",
                "JetVBF",
                self.params,
                year=self._year,
                pt_type="pt_default",
                pt_cut_name=self.pt_cut_name,
                forward_jet_veto=True,
            )

            # order in pt
            self.events["JetGoodVBF"] = self.events.JetGoodVBF[
                ak.argsort(self.events.JetGoodVBF.pt, axis=1, ascending=False)
            ]

            # define the Matched collection
            self.events["JetGoodVBFMatched"] = ak.mask(
                self.events["JetGoodVBF"],
                ~ak.is_none(self.events["JetGoodVBF"].provenance_vbf, axis=1),
            )

            # # Define VBF jets but removing only 4 JetGoodHiggs (like in the AN)
            # jet_goodhiggs_idx_not_none = self.events.JetGoodHiggs.index

            # # find the remaining jets to define the vbf candidates
            # self.events["JetVBFAN"] = self.get_jets_not_from_idx(
            #     jet_goodhiggs_idx_not_none
            # )
            # self.events["JetGoodVBFAN"], mask_jet_vbf = custom_jet_selection(
            #     self.events,
            #     "JetVBFAN",
            #     "JetVBF",
            #     self.params,
            #     year=self._year,
            #     pt_type="pt_default",
            #     pt_cut_name=self.pt_cut_name,
            #     forward_jet_veto=True,
            # )

            # # create the provenance field
            # for jet_coll in ["JetGood", "JetGoodHiggs", "JetGoodMatched", "JetGoodHiggsMatched"]:
            #     self.events[jet_coll] = ak.with_field(
            #         self.events[jet_coll],
            #         self.events[jet_coll].provenance_higgs,
            #         "provenance",
            #     )

            # for jet_coll in ["JetGoodVBF", "JetGoodVBFMatched"]:
            #     self.events[jet_coll] = ak.with_field(
            #         self.events[jet_coll],
            #         self.events[jet_coll].provenance_vbf,
            #         "provenance",
            #     )

    def count_objects(self, variation):
        super().count_objects(variation=variation)
        if self.vbf_analysis:
            self.events["nJetGoodVBF"] = ak.num(self.events.JetGoodVBF, axis=1)

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        if self.vbf_analysis:
            if self._isMC and self.vbf_parton_matching:
                self.events["nJetGoodVBFMatched"] = ak.num(
                    self.events.JetGoodVBFMatched, axis=1
                )

            # choose vbf jets as the two jets with the highest pt that are not from higgs decay
            self.events["JetVBFLeadingPtNotFromHiggs"] = self.events.JetGoodVBF[:, :2]

            # choose vbf jet candidates as the ones with the highest mjj that are not from higgs decay
            self.events["JetGoodVBFLeadingMjj"] = get_lead_mjj_jet_pair(self.events, "JetGoodVBF")
            
            # Get additional VBF jets
            mask_jet_vbf_lead_mjj_not_none = ak.values_astype(
                ~ak.is_none(self.events.JetGoodVBFLeadingMjj.pt, axis=1), "bool"
            )

            # this mask doesn't change the number of events 
            # but the elements from the array if they are None values
            jet_vbf_leading_mjj_idx_not_none = self.events[
                "JetGoodVBFLeadingMjj"
            ].index[mask_jet_vbf_lead_mjj_not_none]
            
            # Get the total idx to remove 
            jet_good_vbf_leading_mjj_idx_not_none = ak.concatenate(
                [
                    self.events.JetGoodClip.index,
                    jet_vbf_leading_mjj_idx_not_none,
                ],
                axis=1,
            )

            self.events["JetAdditionalVBF"] = self.get_jets_not_from_idx(
                jet_good_vbf_leading_mjj_idx_not_none
            )

            # get additional good VBF jets
            self.events["JetAdditionalGoodVBF"], _ = custom_jet_selection(
                self.events,
                "JetAdditionalVBF",
                "JetVBF",
                self.params,
                year=self._year,
                pt_type="pt_default",
                pt_cut_name=self.pt_cut_name,
                forward_jet_veto=True,
            )

            # order in energy because of the high eta
            self.events["JetAdditionalGoodVBF"] = ak.pad_none(
                self.events["JetAdditionalGoodVBF"][
                    ak.argsort(
                        self.events.JetAdditionalGoodVBF.energy, axis=1, ascending=False
                    )
                ],
                self.max_num_jets_add_vbf,
                axis=1,
                clip=True,
            )

            self.events["JetGoodPadded"] = ak.pad_none(
                self.events.JetGoodClip, self.max_num_jets_good, clip=True
            )

            # merge the 3 jet collections to feed to spanet training
            self.events["JetTotalSPANetPadded"] = ak.concatenate(
                [
                    self.events["JetGoodPadded"],
                    self.events["JetGoodVBFLeadingMjj"],
                    self.events["JetAdditionalGoodVBF"],
                ],
                axis=1,
            )
            self.events["JetTotalSPANetMatchedPadded"] = ak.mask(
                self.events["JetTotalSPANetPadded"],
                ~ak.is_none(self.events["JetTotalSPANetPadded"].provenance, axis=1),
            )

            # save the merged good VBF jets for convenience
            self.events["JetGoodVBFMergedPadded"] = ak.concatenate(
                [
                    self.events["JetGoodVBFLeadingMjj"],
                    self.events["JetAdditionalGoodVBF"],
                ],
                axis=1,
            )

            # create a new collection which is similar to the one of the AN
            self.events["JetGoodHiggsPlusVBF1mjj"] = ak.concatenate(
                [
                    self.events["JetGoodHiggs"],
                    self.events["JetGoodVBFLeadingMjj"],
                ],
                axis=1,
            )


            if self._isMC and self.random_pt:
                # flatten pt for all jets to train spanet 
                self.events["JetTotalSPANetPtFlattenPadded"] = copy.copy(
                    self.events["JetTotalSPANetPadded"]
                )
                self.flatten_pt(self.rand_type, "JetTotalSPANetPtFlattenPadded")
                # flatten pt only for jets matched to the Higgs for the training of spanet
                self.events["JetTotalSPANetPtFlattenHiggsMatchedPadded"] = ak.where(
                    ak.is_none(self.events["JetTotalSPANetPtFlattenPadded"].provenance_higgs, axis=1),
                    self.events["JetTotalSPANetPadded"],
                    self.events["JetTotalSPANetPtFlattenPadded"]
                )
                
            # Define mjj and delta eta of leading mjj vbf jet candidates
            for jet_coll in ["JetTotalSPANetPadded", "JetTotalSPANetPtFlattenPadded"]:
                vbf_mjj = (self.events[jet_coll][:, 5] + self.events[jet_coll][:, 6]).mass
                vbf_deta = abs(self.events[jet_coll][:, 5].eta - self.events[jet_coll][:, 6].eta)
                
                self.events[f"mjj{jet_coll}"]=vbf_mjj
                self.events[f"deta{jet_coll}"]=vbf_deta
                
        
        super().process_extra_after_presel(variation=variation)
