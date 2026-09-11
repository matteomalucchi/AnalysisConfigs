import awkward as ak
import copy

from utils_configs.basic_functions import add_fields
from configs.HH4b_common.workflow_common import HH4bCommonProcessor
from utils_configs.reconstruct_resonances import (
    reconstruct_resonances_from_idx,
    run2_matching_algorithm,
)


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
            self.dummy_provenance(jet_collection="Jet", name="provenance_vbf")

        self.def_provenance_field()
        self.define_jet_collections()

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation=variation)
        if self.vbf_analysis:
            self.define_vbf_jet_collections()

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        if self.vbf_analysis:
            if self.vbf_matching_after_higgs_pairing and self.spanet:
                # apply spanet model to get the pairing prediction for the b-jets from Higgs
                pairing_predictions, jet_coll_pairing, *_ = self.eval_spanet()
                (
                    self.events["HiggsLeading"],
                    self.events["HiggsSubLeading"],
                    self.events["JetGoodClip"],
                    jet_vbf,
                ) = reconstruct_resonances_from_idx(
                    self.events[jet_coll_pairing], pairing_predictions
                )

                # rebuild the VBF candidates out of the jets the pairing took
                self.select_jets_not_from_idx(
                    "JetGoodVBFCandidates",
                    self.events.JetGoodClip.index,
                    order_by="pt",
                )

            self.define_vbf_pair_collections()

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

            # create a new collection which is similar to the one of the AN
            self.events["JetGoodHiggsPlusVBF1mjjAN"] = ak.concatenate(
                [
                    self.events["JetGoodHiggs"],
                    self.events["JetGoodVBFLeadingMjjAN"],
                ],
                axis=1,
            )

            # collections with provenance_higgs and provenance_vbf saved separately
            padded = add_fields(self.events["JetGoodPadded"], "all")

            self.events["JetGoodProvHiggsPadded"] = ak.zip(
                {field: padded[field] for field in padded.fields}
                | {"provenance": padded.provenance_higgs},
                with_name="PtEtaPhiMLorentzVector",
            )

            # create a combined jet collection with the provenance separate for higgs and vbf
            self.events["JetTotalSPANetSeparateProvHiggsVBFPadded"] = ak.concatenate(
                [
                    self.events["JetGoodProvHiggsPadded"],
                    self.events["JetGoodVBFMergedProvVBFPadded"],
                ],
                axis=1,
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
                    self.events[jet_coll] = ak.with_field(
                        self.events[jet_coll],
                        self.events[jet_coll].pt,
                        "pt_orig",
                    )
                    self.events[jet_coll] = ak.with_field(
                        self.events[jet_coll],
                        self.events[jet_coll].mass,
                        "mass_orig",
                    )

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
                # with flattened pt for all jets
                self.events["JetTotalSPANetSeparateProvHiggsVBFPtFlattenPadded"] = (
                    ak.concatenate(
                        [
                            self.events["JetGoodProvHiggsPtFlattenPadded"],
                            self.events["JetGoodVBFMergedProvVBFPtFlattenPadded"],
                        ],
                        axis=1,
                    )
                )

                # create a combined jet collection with the provenance separate for higgs and vbf
                # with flattened pt only for jets used for Higgs matching
                self.events[
                    "JetTotalSPANetSeparateProvHiggsVBFPtFlattenOnlyHiggsPadded"
                ] = ak.concatenate(
                    [
                        self.events["JetGoodProvHiggsPtFlattenPadded"],
                        self.events["JetGoodVBFMergedProvVBFPadded"],
                    ],
                    axis=1,
                )

            if not (self.spanet and self.vbf_matching_after_higgs_pairing):
                # Compute the Run 2 pairing to compute the centrality
                (
                    pairing_predictions,
                    self.events["delta_dhh"],
                    self.events["HiggsLeading"],
                    self.events["HiggsSubLeading"],
                    self.events["JetGoodFromHiggsOrdered"],
                ) = run2_matching_algorithm(self.events["JetGoodHiggs"])

            # Define mjj, delta eta and centrality of leading mjj vbf jet candidates
            self.define_vbf_kinematics(
                [
                    "JetTotalSPANetPadded",
                    "JetTotalSPANetPtFlattenPadded",
                    "JetGoodVBFMergedProvVBFPadded",
                    "JetGoodVBFMergedProvVBFPtFlattenPadded",
                    "JetGoodVBFCandidates",
                ],
                [self.max_num_jets_good, self.max_num_jets_good, 0, 0],
            )

        super().process_extra_after_presel(variation=variation)
        if not self.vbf_analysis:
            self.events["JetGoodPtFlatten"] = copy.copy(self.events.JetGood)
            self.flatten_pt(self.rand_type, "JetGoodPtFlatten")
