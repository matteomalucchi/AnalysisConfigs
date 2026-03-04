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
            # for jet_coll in ["JetGood", "JetGoodHiggs"]:
            #     self.events[jet_coll] = ak.with_field(
            #         self.events[jet_coll],
            #         self.events[jet_coll].provenance_higgs,
            #         "provenance",
            #     )

            # for jet_coll in ["JetGoodVBFAN"]:
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

            # choose vbf jets as the two jets with the highest pt that are not from higgs decay
            self.events["JetVBFLeadingPtNotFromHiggs"] = self.events.JetGoodVBF[:, :2]

            # choose vbf jet candidates as the ones with the highest mjj that are not from higgs decay
            self.events["JetGoodVBFLeadingMjj"] = get_lead_mjj_jet_pair(
                self.events, "JetGoodVBF"
            )

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

            # save the merged good VBF jets for convenience
            self.events["JetGoodVBFMergedPadded"] = ak.concatenate(
                [
                    self.events["JetGoodVBFLeadingMjj"],
                    self.events["JetAdditionalGoodVBF"],
                ],
                axis=1,
            )

            # create a new collection which is similar to the one of the AN
            # WARNING: JetGoodVBFLeadingMjj is not the same as in the AN if max_num_jets_good!=4
            self.events["JetGoodHiggsPlusVBF1mjj"] = ak.concatenate(
                [
                    self.events["JetGoodHiggs"],
                    self.events["JetGoodVBFLeadingMjj"],
                ],
                axis=1,
            )

            # collections with provenance_higgs and provenance_vbf saved separately
            self.events["JetGoodProvHiggsPadded"] = ak.with_field(
                self.events["JetGoodPadded"],
                self.events["JetGoodPadded"].provenance_higgs,
                "provenance",
            )
            self.events["JetGoodVBFMergedProvVBFPadded"] = ak.with_field(
                self.events["JetGoodVBFMergedPadded"],
                self.events["JetGoodVBFMergedPadded"].provenance_vbf,
                "provenance",
            )

            # create a combined jet collection with the provenance separate for higgs and vbf
            self.events["JetTotalSPANetSeparateProvHiggsVBFPadded"]= ak.concatenate(
                [
                    self.events["JetGoodProvHiggsPadded"],
                    self.events["JetGoodVBFMergedProvVBFPadded"],
                ],
                axis=1
                
            )

            if self._isMC and self.random_pt:
                # flatten pt for all jets to train spanet
                for jet_coll in [
                    "JetTotalSPANetPadded",
                    "JetGoodProvHiggsPadded",
                    "JetGoodVBFMergedProvVBFPadded",
                ]:
                    # add the ptflatten before padded
                    pt_flat_jet_coll = jet_coll.replace("Padded", "PtFlattenPadded")
                    self.events[pt_flat_jet_coll] = copy.copy(self.events[jet_coll])
                    self.flatten_pt(self.rand_type, pt_flat_jet_coll)

                # flatten pt only for jets matched to the Higgs for the training of spanet
                self.events["JetTotalSPANetPtFlattenHiggsMatchedPadded"] = ak.where(
                    ak.is_none(
                        self.events["JetTotalSPANetPtFlattenPadded"].provenance_higgs,
                        axis=1,
                    ),
                    self.events["JetTotalSPANetPadded"],
                    self.events["JetTotalSPANetPtFlattenPadded"],
                )
                
                # create a combined jet collection with the provenance separate for higgs and vbf
                self.events["JetTotalSPANetSeparateProvHiggsVBFPtFlattenPadded"] = ak.concatenate(
                    [
                        self.events["JetGoodProvHiggsPtFlattenPadded"],
                        self.events["JetGoodVBFMergedProvVBFPtFlattenPadded"],
                    ],
                    axis=1
                )
                self.events["JetTotalSPANetSeparateProvHiggsVBFPtFlattenOnlyHiggsPadded"]= ak.concatenate(
                    [
                        self.events["JetGoodProvHiggsPtFlattenPadded"],
                        self.events["JetGoodVBFMergedProvVBFPadded"],
                    ],
                    axis=1
                )
            
            
            # Define mjj and delta eta of leading mjj vbf jet candidates
            for jet_coll, jet_idx in zip(
                [
                    "JetTotalSPANetPadded",
                    "JetTotalSPANetPtFlattenPadded",
                    "JetGoodVBFMergedProvVBFPadded",
                    "JetGoodVBFMergedProvVBFPtFlattenPadded",
                ],
                [self.max_num_jets_good, self.max_num_jets_good, 0, 0],
            ):
                # the 2 leading jets in mjj are the ones right after the JetGood
                vbf_mjj = (
                    self.events[jet_coll][:, jet_idx]
                    + self.events[jet_coll][:, jet_idx + 1]
                ).mass
                vbf_deta = abs(
                    self.events[jet_coll][:, jet_idx].eta
                    - self.events[jet_coll][:, jet_idx + 1].eta
                )

                self.events[f"mjj{jet_coll}"] = vbf_mjj
                self.events[f"deta{jet_coll}"] = vbf_deta

        super().process_extra_after_presel(variation=variation)
