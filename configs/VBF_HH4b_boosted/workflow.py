import awkward as ak
import pandas as pd
import copy
import numpy as np
import xgboost as xgb

from utils_configs.custom_cut_functions import custom_jet_selection
from utils_configs.basic_functions import add_fields
from configs.HH4b_common.workflow_common import HH4bCommonProcessor
from utils_configs.reconstruct_higgs_candidates import get_lead_mjj_jet_pair
from utils_configs.reconstruct_higgs_candidates import run2_matching_algorithm
from utils_configs.bdt_evaluation_functions import get_default_bdt_inputs, evaluate_bdt, disc_TXbb

from configs.HH4b_common.custom_object_preselection_common import object_cleaning, clean_ak4_boosted


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

            if self._year == "2024":
                self.events["FatJetGood"] = ak.with_field(
                    self.events["FatJetGood"],
                    (self.events["FatJetGood"]["globalParT3_Xbb"] / (self.events["FatJetGood"]["globalParT3_Xbb"] + self.events["FatJetGood"]["globalParT3_QCD"])),
                    "btagBBTXbb",
                )
                self.events["FatJetGood"] = ak.with_field(
                    self.events["FatJetGood"],
                    (self.events["FatJetGood"]["particleNetLegacy_Xbb"] / (self.events["FatJetGood"]["particleNetLegacy_Xbb"] + self.events["FatJetGood"]["particleNetLegacy_QCD"])),
                    "btagBBPNetLegacy",
                )
                self.events["FatJetGood"] = ak.with_field(
                    self.events["FatJetGood"],
                    self.events["FatJetGood"]["globalParT3_Xbb"],
                    "btagBB",
                )
            else:
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
            if not self.TXbb_order:
                self.events["FatJetGood"] = self.events["FatJetGood"][
                    ak.argsort(self.events["FatJetGood"]["btagBB"], axis=1, ascending=False)
                ]
            else:
                self.events["FatJetGood"] = self.events["FatJetGood"][
                    ak.argsort(self.events["FatJetGood"]["btagBBTXbb"], axis=1, ascending=False)
                ]
            # We do only take the masks from the leading and subleading jets. Then we apply the masks to FatJetGood
            # This does require the additional fields we add to the FatJet collection inside the function to be also added to the final collection
            _, mask_fat_lead = custom_jet_selection(
                self.events,
                jet_type="FatJetGood",
                jet_type_obj_presel="FatJetLeading",
                params=self.params,
                year=self._year,
                jet_tagger="PNet",
                pt_type="pt",
                pt_cut_name="pt",
            )
            _, mask_fat_sublead = custom_jet_selection(
                self.events,
                jet_type="FatJetGood",
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
            # == Cut on b-tag
            # mask_fat_lead = mask_fat_lead & (self.events["FatJetGood"]["btagBB"] > 0.65)
            # mask_fat_sublead = mask_fat_sublead & (self.events["FatJetGood"]["btagBB"] > 0.00) # used to be 0.05

            # ===== Cutting and combining the two fatjets =====
            self.events["FatJetGoodLeading"] = self.events["FatJetGood"][mask_fat_lead][:, :1]
            lead_idx = ak.local_index(self.events["FatJetGood"], axis=1)[mask_fat_lead][:, :1]
            all_idx = ak.local_index(self.events["FatJetGood"], axis=1)
            not_lead_mask = all_idx != ak.firsts(lead_idx)
            mask_fat_sublead_excl = mask_fat_sublead & not_lead_mask
            self.events["FatJetGoodSubLeading"] = self.events["FatJetGood"][mask_fat_sublead_excl][:, :1]

            self.events["FatJetGoodSelected"] = ak.concatenate([self.events["FatJetGoodLeading"], self.events["FatJetGoodSubLeading"]], axis=1)
            self.events["nFatJetGoodSelected"] = ak.num(self.events["FatJetGoodSelected"], axis=1)

            # mask_pt = self.events["FatJetGood"]["pt"] > 250
            # mask_eta = abs(self.events["FatJetGood"]["eta"]) < 2.5
            # mask_msd = self.events["FatJetGood"]["msoftdrop"] > 50

            # jetidtight = (
            #     (
            #         (np.abs(self.events["FatJetGood"].eta) <= 2.6)
            #         & (self.events["FatJetGood"].neHEF < 0.99)
            #         & (self.events["FatJetGood"].neEmEF < 0.9)
            #         & ((self.events["FatJetGood"].chMultiplicity + self.events["FatJetGood"].neMultiplicity) > 1)
            #         & (self.events["FatJetGood"].chHEF > 0.01)
            #         & (self.events["FatJetGood"].chMultiplicity > 0)
            #     )
            # | (
            # ((np.abs(self.events["FatJetGood"].eta) > 2.6) & (np.abs(self.events["FatJetGood"].eta) <= 2.7))
            # & (self.events["FatJetGood"].neHEF < 0.90)
            # & (self.events["FatJetGood"].neEmEF < 0.99)
            # )
            # | (((np.abs(self.events["FatJetGood"].eta) > 2.7) & (np.abs(self.events["FatJetGood"].eta) <= 3.0)) & (self.events["FatJetGood"].neHEF < 0.99))
            # | ((np.abs(self.events["FatJetGood"].eta) > 3.0) & (self.events["FatJetGood"].neMultiplicity >= 2) & (self.events["FatJetGood"].neEmEF < 0.4))
            # )
            # jetidtightlepveto = (
            # (np.abs(self.events["FatJetGood"].eta) <= 2.7) & jetidtight & (self.events["FatJetGood"].muEF < 0.8) & (self.events["FatJetGood"].chEmEF < 0.8)
            # ) | ((np.abs(self.events["FatJetGood"].eta) > 2.7) & jetidtight)

            # self.events["FatJetGoodTest"] = self.events["FatJetGood"][jetidtight & jetidtightlepveto & mask_pt & mask_eta & mask_msd]


        if self.vbf_analysis:
            if not self.boosted:
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
                self.events["JetGoodVBF"] = self.events.JetGoodVBF[
                    ak.argsort(self.events.JetGoodVBF.pt, axis=1, ascending=False)
                ]
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
                # self.events["JetGoodVBFCandidates"] = self.events["JetGoodVBF"]

            if self.boosted:
                self.events["JetGood"], mask_jet_vbf = custom_jet_selection(
                    self.events,
                    "Jet",
                    "JetBoosted",
                    self.params,
                    year=self._year,
                    pt_type="pt_default",
                    pt_cut_name=self.pt_cut_name,
                    forward_jet_veto=False,
                )
                self.events["JetGoodVBF"], mask_jet_vbf = custom_jet_selection(
                    self.events,
                    "Jet",
                    "JetVBF",
                    self.params,
                    year=self._year,
                    pt_type="pt_default",
                    pt_cut_name=self.pt_cut_name,
                    forward_jet_veto=False,
                )
                self.events["JetGoodVBF"] = self.events.JetGoodVBF[
                    ak.argsort(self.events.JetGoodVBF.pt, axis=1, ascending=False)
                ]
                self.events["JetVBF"] = copy.copy(self.events.Jet)
                self.events["HT_jetJetGoodVBF"] = ak.sum(self.events.JetGoodVBF.pt, axis=1)

                self.events["JetGoodCloseToFatJet"], mask_jet_close_to_fatjet = custom_jet_selection(
                    self.events,
                    "Jet",
                    "JetNearFatJet",
                    self.params,
                    year=self._year,
                    pt_type="pt_default",
                    pt_cut_name=self.pt_cut_name,
                    forward_jet_veto=False,
                )
                # order in pt
                # Clean From AK8
                # self.events["JetGoodVBFCandidates"] = clean_ak4_boosted(
                #         self.events["JetGoodBoosted"],
                #         self.events["FatJetGoodSelected"],
                #         self.events["Muon"][self.events["Muon"]["pt"] > 7],
                #         self.events["Electron"][self.events["Electron"]["pt"] > 5],
                #         dr_jets=1.2, dr_lep=0.4
                #         )
                # self.events["JetGoodCloseToFatJet"] = clean_ak4_boosted(
                #         self.events["JetGoodBoosted"][np.abs(self.events["JetGoodBoosted"]["eta"]) < 2.5],
                #         self.events["FatJetGoodSelected"],
                #         self.events["Muon"][self.events["Muon"]["pt"] > 7],
                #         self.events["Electron"][self.events["Electron"]["pt"] > 5],
                #         dr_jets=0.9, dr_lep=0.4
                #         )

                # The equivalent of this for the not-boosted is in the main workflow_common. But there it is after the preselection. So I am not sure, how to merge the two.
                # vbf_pool = self.events["JetGoodVBF"]

                # # Shortcut to the VBF jet preselection values
                # jetvbf_obj_presel = self.params.object_preselection["JetVBF"]
                # # looser VBF cuts
                # mask_pt_vbf = ak.fill_none(vbf_pool.pt > jetvbf_obj_presel["pt"], False)
                # # additional cuts for the region 2.5 < |eta| < 3.0
                # central_or_forward = (np.abs(vbf_pool.eta) < jetvbf_obj_presel["gap_eta_min"]) | (np.abs(vbf_pool.eta) > jetvbf_obj_presel["gap_eta_max"])
                # gap_higher_pt = (np.abs(vbf_pool.eta) >= jetvbf_obj_presel["gap_eta_min"]) & (np.abs(vbf_pool.eta) <= jetvbf_obj_presel["gap_eta_max"]) & (vbf_pool.pt > jetvbf_obj_presel["gap_pt"])
                # within_max_eta = np.abs(vbf_pool.eta) < jetvbf_obj_presel["eta"]

                # mask_eta_vbf = ak.fill_none(
                #     (central_or_forward | gap_higher_pt) & within_max_eta,
                #     False,
                # )
                # self.events["JetGoodVBF"] = ak.pad_none(vbf_pool[mask_pt_vbf & mask_eta_vbf], 2, clip=True)

                self.events["JetGoodVBFEnergyOrdered"] = get_lead_mjj_jet_pair(
                    self.events, "JetGoodVBF"
                )
                for jet_coll in ["JetGood", "JetGoodVBF", "JetGoodVBFEnergyOrdered"]:
                    padded = ak.pad_none(self.events[jet_coll], 2, axis=1)
                    vbf_mjj = (
                        padded[:, 0]
                        + padded[:, 1]
                    ).mass
                    vbf_deta = abs(
                        padded[:, 0].eta
                        - padded[:, 1].eta
                    )

                    self.events[f"mjj{jet_coll}"] = ak.fill_none(vbf_mjj, -999.0)
                    self.events[f"deta{jet_coll}"] = ak.fill_none(vbf_deta, -999.0)

                # # build dijets for veto
                # dijets = ak.combinations(self.events["JetGoodVBFCandidates"], 2, fields=["j_lead", "j_sublead"])
                # dijets = ak.fill_none(dijets, [])
                # d4 = dijets.j_lead + dijets.j_sublead
                # for param in ["mass", "pt", "eta", "phi"]:
                #     dijets = ak.with_field(dijets, getattr(d4, param), param)
                # dijets = ak.with_field(dijets, np.abs(dijets.j_lead.eta - dijets.j_sublead.eta), "dEta")

                # self.events["DiJetVBF"] = ak.pad_none(dijets,1,clip=True)[:,0]

    def count_objects(self, variation):
        super().count_objects(variation=variation)
        if self.vbf_analysis:
            self.events["nJetGoodVBF"] = ak.num(self.events.JetGoodVBF, axis=1)
            self.events["nJetGoodCloseToFatJet"] = ak.num(self.events.JetGoodCloseToFatJet, axis=1)
            self.events["nJetGoodVBFEnergyOrdered"] = ak.num(self.events.JetGoodVBFEnergyOrdered, axis=1)

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
        self.events["HiggsLeading"] = ak.with_field(
            self.events["HiggsLeading"],
            ak.fill_none(self.events.HiggsLeading.tau3 / self.events.HiggsLeading.tau2, -999),
            "Tau3OverTau2"
        )
        self.events["HiggsSubLeading"] = ak.with_field(
            self.events["HiggsSubLeading"],
            ak.fill_none(self.events.HiggsSubLeading.tau3 / self.events.HiggsSubLeading.tau2, -999),
            "Tau3OverTau2"
        )
        self.events["HiggsLeading"] = ak.with_field(
            self.events["HiggsLeading"],
            ak.fill_none(self.events.HiggsLeading.pt / self.events.HH.mass, -999),
            "divHHmass"
        )
        self.events["HiggsSubLeading"] = ak.with_field(
            self.events["HiggsSubLeading"],
            ak.fill_none(self.events.HiggsSubLeading.pt / self.events.HH.mass, -999),
            "divHHmass"
        )
        self.events["JetGoodVBFNearHiggsLeading"] = ak.firsts(self.events["JetGoodCloseToFatJet"][ak.argsort(self.events["JetGoodCloseToFatJet"].delta_r(self.events["HiggsLeading"]), ascending=True)])
        self.events["JetGoodVBFNearHiggsSubLeading"] = ak.firsts(self.events["JetGoodCloseToFatJet"][ak.argsort(self.events["JetGoodCloseToFatJet"].delta_r(self.events["HiggsSubLeading"]), ascending=True)])
        self.events["HiggsLeadingByHiggsSubLeadingPt"] = self.events.HiggsLeading.pt / self.events.HiggsSubLeading.pt
        self.events["HiggsLeading"] = ak.with_field(
            self.events["HiggsLeading"],
            ak.fill_none(self.events.HiggsLeading.delta_r(self.events.JetGoodVBFNearHiggsLeading), -999),
            "dRclosestVBF"
        )
        self.events["HiggsSubLeading"] = ak.with_field(
            self.events["HiggsSubLeading"],
            ak.fill_none(self.events.HiggsSubLeading.delta_r(self.events.JetGoodVBFNearHiggsSubLeading), -999),
            "dRclosestVBF"
        )
        self.events["HiggsLeading"] = ak.with_field(
            self.events["HiggsLeading"],
            ak.fill_none((self.events.HiggsLeading + self.events.JetGoodVBFNearHiggsLeading).mass, -999),
            "massclosestVBF",
        )
        self.events["HiggsSubLeading"] = ak.with_field(
            self.events["HiggsSubLeading"],
            ak.fill_none((self.events.HiggsSubLeading + self.events.JetGoodVBFNearHiggsSubLeading).mass, -999),
            "massclosestVBF",
        )

        if self.bdt_model:

            self.events["HiggsLeading"] = ak.with_field(
                self.events["HiggsLeading"],
                ak.fill_none(disc_TXbb(self.events.HiggsLeading.btagBBTXbb), -999),
                "btagBBTXbb_dig",
                )
            bdt_events = get_default_bdt_inputs(self.events)
            self.events["boosted_bdt_score"], self.events["boosted_bdt_vbf_score"] = evaluate_bdt(self.bdt_model, bdt_events)
