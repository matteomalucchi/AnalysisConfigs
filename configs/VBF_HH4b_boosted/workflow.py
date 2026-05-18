import awkward as ak
import copy
import numpy as np

from utils_configs.custom_cut_functions import custom_jet_selection
from utils_configs.basic_functions import add_fields
from configs.HH4b_common.workflow_common import HH4bCommonProcessor
from utils_configs.reconstruct_higgs_candidates import get_lead_mjj_jet_pair
from utils_configs.reconstruct_higgs_candidates import run2_matching_algorithm

from configs.HH4b_common.custom_object_preselection_common import object_cleaning


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

        self.def_provenance_field()
        self.define_jet_collections()

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation=variation)
        if self.boosted:
            # Select FatJets for the boosted category
            self.events["FatJetGood"] = self.events.FatJet

            # ===== BB-tagging =====
            # here we propagate the btagging scores to the FatJetGood collection as is done in the pocket coffea jet_selection
            # if we're interested in other taggers, we need to add them here or swap to the mass correlated ones ("particleNetWithMass_HbbvsQCD", "particleNetWithMass_HccvsQCD")
            self.events["FatJetGood"] = ak.with_field(
                self.events["FatJetGood"],
                self.events["FatJetGood"]["particleNet_XbbVsQCD"],
                "btagBB",
            )
            self.events["FatJetGood"] = ak.with_field(
                self.events["FatJetGood"],
                self.events["FatJetGood"]["particleNet_XccVsQCD"],
                "btagCC",
            )
            # Add btag WP
            self.events["FatJetGood"] = self.generate_btag_workingpoints(
                self.events["FatJetGood"], 3
            )
            # jet ordered in btagging score
            self.events["FatJetGood"] = self.events["FatJetGood"][
                ak.argsort(self.events["FatJetGood"]["btagBB"], axis=1, ascending=False)
            ]
            # We do only take the masks from the leading and subleading jets. Then we apply the masks to FatJetGood
            # This does require the additional fields we add to the FatJet collection inside the function to be also added to the final collection
            _, mask_fat_lead = custom_jet_selection(
                self.events,
                jet_type="FatJet",
                jet_type_obj_presel="FatJetLeading",
                params=self.params,
                year=self._year,
                jet_tagger="PNet",
                pt_type="pt",
                pt_cut_name="pt",
            )
            _, mask_fat_sublead = custom_jet_selection(
                self.events,
                jet_type="FatJet",
                jet_type_obj_presel="FatJetSubLeading",
                params=self.params,
                year=self._year,
                jet_tagger="PNet",
                pt_type="pt",
                pt_cut_name="pt",
            )
            # ===== Regressions (needed for additional cuts) ===== 
            self.events["FatJetGood"] = ak.with_field(
                self.events["FatJetGood"],
                self.events["FatJetGood"].mass * self.events["FatJetGood"].particleNet_massCorr,
                "mass_regr"
            )
            self.events["FatJetGood"] = ak.with_field(
                self.events["FatJetGood"],
                self.events["FatJetGood"].pt * self.events["FatJetGood"].particleNet_massCorr,
                "pt_regr"
            )
            fatjet_obj_presel_lead = self.params.object_preselection["FatJetLeading"]
            fatjet_obj_presel_sublead = self.params.object_preselection["FatJetSubLeading"]
            if "mass_regr_min" in fatjet_obj_presel_lead.keys() and "mass_regr_max" in fatjet_obj_presel_lead.keys():
                mask_mass_regr = (
                    (self.events["FatJetGood"]["mass_regr"] >= fatjet_obj_presel_lead["mass_regr_min"]) &
                    (self.events["FatJetGood"]["mass_regr"] <= fatjet_obj_presel_lead["mass_regr_max"])
                )
                mask_fat_lead = mask_fat_lead & mask_mass_regr
            if "mass_regr_min" in fatjet_obj_presel_sublead.keys() and "mass_regr_max" in fatjet_obj_presel_sublead.keys():
                mask_mass_regr = (
                    (self.events["FatJetGood"]["mass_regr"] >= fatjet_obj_presel_sublead["mass_regr_min"]) &
                    (self.events["FatJetGood"]["mass_regr"] <= fatjet_obj_presel_sublead["mass_regr_max"])
                )
                mask_fat_sublead = mask_fat_sublead & mask_mass_regr

            # ===== Cutting and combining the two fatjets =====
            self.events["FatJetGoodLeading"] = self.events["FatJetGood"][mask_fat_lead][:, :1]
            lead_idx = ak.local_index(self.events["FatJetGood"], axis=1)[mask_fat_lead][:, :1]
            all_idx = ak.local_index(self.events["FatJetGood"], axis=1)
            not_lead_mask = all_idx != ak.firsts(lead_idx)
            mask_fat_sublead_excl = mask_fat_sublead & not_lead_mask
            self.events["FatJetGoodSubLeading"] = self.events["FatJetGood"][mask_fat_sublead_excl][:, :1]

            self.events["FatJetGoodSelected"] = ak.concatenate([self.events["FatJetGoodLeading"], self.events["FatJetGoodSubLeading"]], axis=1)
            self.events["nFatJetGoodSelected"] = ak.num(self.events["FatJetGoodSelected"], axis=1)
            self.events["FatJetGoodSelected"] = ak.pad_none(self.events["FatJetGoodSelected"], 2)

        if self.vbf_analysis:
            if not self.boosted:
                # get idx of good jets after preselection
                self.events["JetGoodClip"] = copy.copy(
                    self.events.JetGood[:, : self.max_num_jets_good]
                )
                jet_good_idx_not_none = self.events.JetGoodClip.index

                # find the remaining jets to define the vbf candidates
                self.events["JetVBF"] = self.get_jets_not_from_idx(jet_good_idx_not_none)
            else:
                self.events["JetVBF"] = copy.copy(self.events.JetGood)
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
            if not self.boosted:
                # Define VBF jets but removing only 4 JetGoodHiggs (like in the AN)
                jet_goodhiggs_idx_not_none = self.events.JetGoodHiggs.index

                # find the remaining jets to define the vbf candidates
                self.events["JetVBFAN"] = self.get_jets_not_from_idx(
                    jet_goodhiggs_idx_not_none
                )
                self.events["JetGoodVBFAN"], mask_jet_vbf = custom_jet_selection(
                    self.events,
                    "JetVBFAN",
                    "JetVBF",
                    self.params,
                    year=self._year,
                    pt_type="pt_default",
                    pt_cut_name=self.pt_cut_name,
                    forward_jet_veto=True,
                )

                # # create the provenance field separate for higgs and vbf
                for jet_coll in ["JetGoodHiggs"]:
                    self.events[jet_coll] = ak.with_field(
                        self.events[jet_coll],
                        self.events[jet_coll].provenance_higgs,
                        "provenance",
                    )
                for jet_coll in ["JetGoodVBFAN"]:
                    self.events[jet_coll] = ak.with_field(
                        self.events[jet_coll],
                        self.events[jet_coll].provenance_vbf,
                        "provenance",
                    )
                self.events["JetGoodVBFCandidates"] = self.events["JetGoodVBF"]

            if self.boosted:
                self.events["JetGoodVBFCandidates"] = object_cleaning(
                    self.events["JetGoodVBF"],
                    self.events["FatJetGoodSelected"],
                    dr_min=0.8
                )

                # The equivalent of this for the not-boosted is in the main workflow_common. But there it is after the preselection. So I am not sure, how to merge the two.
                self.events["JetGoodVBFEnergyOrdered"] = get_lead_mjj_jet_pair(
                    self.events, "JetGoodVBFCandidates"
                )
                vbf_pool = self.events["JetGoodVBFCandidates"]

                # Shortcut to the VBF jet preselection values
                jetvbf_obj_presel = self.params.object_preselection["JetVBF"]
                # looser VBF cuts
                mask_pt_vbf = ak.fill_none(vbf_pool.pt > jetvbf_obj_presel["pt"], False)
                # additional cuts for the region 2.5 < |eta| < 3.0
                central_or_forward = (np.abs(vbf_pool.eta) < jetvbf_obj_presel["gap_eta_min"]) | (np.abs(vbf_pool.eta) > jetvbf_obj_presel["gap_eta_max"])
                gap_higher_pt = (np.abs(vbf_pool.eta) >= jetvbf_obj_presel["gap_eta_min"]) & (np.abs(vbf_pool.eta) <= jetvbf_obj_presel["gap_eta_max"]) & (vbf_pool.pt > jetvbf_obj_presel["gap_pt"])
                within_max_eta = np.abs(vbf_pool.eta) < jetvbf_obj_presel["eta"]

                mask_eta_vbf = ak.fill_none(
                    (central_or_forward | gap_higher_pt) & within_max_eta,
                    False,
                )
                self.events["JetGoodVBFCandidates"] = vbf_pool[mask_pt_vbf & mask_eta_vbf]

                # build dijets for veto
                dijets = ak.combinations(self.events["JetGoodVBFCandidates"], 2, fields=["j_lead", "j_sublead"])
                dijets = ak.fill_none(dijets, [])
                d4 = dijets.j_lead + dijets.j_sublead
                for param in ["mass", "pt", "eta", "phi"]:
                    dijets = ak.with_field(dijets, getattr(d4, param), param)
                dijets = ak.with_field(dijets, np.abs(dijets.j_lead.eta - dijets.j_sublead.eta), "dEta")

                # Apply VBF veto conditions to select good dijets and create a mask
                good_pairs_mask = ak.fill_none(
                    (dijets.mass > jetvbf_obj_presel["mjj"]) & (dijets.dEta > jetvbf_obj_presel["delta_eta"]),
                    False,
                )

                self.events["DiJetVBFCandidates"] = dijets[good_pairs_mask]

    def count_objects(self, variation):
        super().count_objects(variation=variation)
        if self.vbf_analysis:
            self.events["nJetGoodVBF"] = ak.num(self.events.JetGoodVBF, axis=1)

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        if self.vbf_analysis and not self.boosted:

            # choose vbf jets as the two jets with the highest pt that are not from higgs decay
            self.events["JetVBFLeadingPtNotFromHiggs"] = self.events.JetGoodVBF[:, :2]

            # choose vbf jet candidates as the ones with the highest mjj that are not from higgs decay
            self.events["JetGoodVBFLeadingMjj"] = get_lead_mjj_jet_pair(
                self.events, "JetGoodVBF"
            )
            # choose vbf jet candidates as the ones with the highest mjj that are not from higgs decay
            self.events["JetGoodVBFLeadingMjjAN"] = get_lead_mjj_jet_pair(
                self.events, "JetGoodVBFAN"
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
            self.events.JetAdditionalGoodVBF = add_fields(
                self.events.JetAdditionalGoodVBF, "all"
            )

            # order in the additional VBF jets
            self.events["JetAdditionalGoodVBF"] = ak.pad_none(
                self.events["JetAdditionalGoodVBF"][
                    ak.argsort(
                        getattr(
                            self.events.JetAdditionalGoodVBF, self.jets_add_vbf_order
                        ),
                        axis=1,
                        ascending=False,
                    )
                ],
                self.max_num_jets_add_vbf,
                axis=1,
                clip=True,
            )

            # save the merged good VBF jets for convenience
            self.events["JetGoodVBFMergedPadded"] = ak.concatenate(
                [
                    self.events["JetGoodVBFLeadingMjj"],
                    self.events["JetAdditionalGoodVBF"],
                ],
                axis=1,
            )
            padded = add_fields(self.events["JetGoodVBFMergedPadded"], "all")
            self.events["JetGoodVBFMergedProvVBFPadded"] = ak.zip(
                {field: padded[field] for field in padded.fields}
                | {"provenance": padded.provenance_vbf},
                with_name="PtEtaPhiMLorentzVector",
            )

            # Define mjj,  delta eta and centrality of leading mjj vbf jet candidates
            for jet_coll, jet_idx in zip(["JetGoodVBFMergedProvVBFPadded"], [0]):
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
