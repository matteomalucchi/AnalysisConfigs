import awkward as ak
import pandas as pd
import copy
import numpy as np
import xgboost as xgb

from utils_configs.custom_cut_functions import custom_jet_selection
from configs.HH4b_common.workflow_common import HH4bCommonProcessor
from utils_configs.reconstruct_resonances import get_lead_mjj_jet_pair
from utils_configs.bdt_evaluation_functions import get_default_bdt_inputs, evaluate_bdt, disc_TXbb

from configs.HH4b_common.custom_object_preselection_common import clean_ak4_boosted


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
            # Uncorrected softdrop mass for the comparison with the MiniAOD ntuples: the msoftdrop
            # calibrator overwrites msoftdrop in place, so rebuild it from the raw subjets
            subjets = self.events["FatJet"].subjets
            msoftdrop_raw = (subjets * (1 - subjets.rawFactor)).sum(axis=-1).mass
            self.events["FatJet"] = ak.with_field(
                self.events["FatJet"],
                ak.where(self.events["FatJet"].msoftdrop > 0, msoftdrop_raw, self.events["FatJet"].msoftdrop),
                "msoftdrop_raw",
            )

            # ===== BB-tagging =====
            # here we propagate the btagging scores to the FatJet collection as is done in the pocket coffea jet_selection
            # if we're interested in other taggers, we need to add them here or swap to the mass correlated ones ("particleNetWithMass_HbbvsQCD", "particleNetWithMass_HccvsQCD")

            if "2024" in self._year:
                self.events["FatJet"] = ak.with_field(
                    self.events["FatJet"],
                    (self.events["FatJet"]["globalParT3_Xbb"] / (self.events["FatJet"]["globalParT3_Xbb"] + self.events["FatJet"]["globalParT3_QCD"])),
                    "btagBBTXbb",
                )
                if self.region_setting == "40GeV":
                    b_low = 0.3
                    b_thresh = 0.8
                elif self.region_setting == "50GeV":
                    b_low = 0.05
                    b_thresh = 0.65
                self.events["FatJet"] = ak.with_field(self.events["FatJet"], ak.where(self.events["FatJet"]["btagBBTXbb"] > b_thresh, (self.events["FatJet"]["btagBBTXbb"]), (self.events["FatJet"]["btagBBTXbb"] - b_low) * (1-b_thresh) / (b_thresh - b_low) + b_thresh), "btagBBTXbb_morph")
                # self.events["FatJet"] = ak.with_field(
                #     self.events["FatJet"],
                #     (self.events["FatJet"]["particleNetLegacy_Xbb"] / (self.events["FatJet"]["particleNetLegacy_Xbb"] + self.events["FatJet"]["particleNetLegacy_QCD"])),
                #     "btagBBPNetLegacy",
                # )
                self.events["FatJet"] = ak.with_field(
                    self.events["FatJet"],
                    self.events["FatJet"]["globalParT3_Xbb"],
                    "btagBB",
                )
            else:
                self.events["FatJet"] = ak.with_field(
                    self.events["FatJet"],
                    self.events["FatJet"]["particleNet_XbbVsQCD"],
                    "btagBB",
                )
            self.events["FatJet"] = ak.with_field(
                self.events["FatJet"],
                self.events["FatJet"]["particleNet_XccVsQCD"],
                "btagCC",
            )
            self.events["FatJet"] = self.generate_btag_workingpoints(self.events["FatJet"], 5, "btagBBTXbb_morph")
            # jet ordered in btagging score
            if not self.TXbb_order:
                self.events["FatJet"] = self.events["FatJet"][
                    ak.argsort(self.events["FatJet"]["btagBB"], axis=1, ascending=False)
                ]
            else:
                self.events["FatJet"] = self.events["FatJet"][
                    ak.argsort(self.events["FatJet"]["btagBBTXbb"], axis=1, ascending=False)
                ]
            # ===== Regressions (needed for additional cuts) ===== 
            if self._year == "2024":
                self.events["FatJet"] = ak.with_field(self.events["FatJet"], self.events["FatJet"]["mass"], "mass_orig")
                self.events["FatJet"] = ak.with_field(
                    self.events["FatJet"],
                    self.events["FatJet"].mass_raw * self.events["FatJet"].globalParT3_massCorrGeneric,  # Check, if we should use globalParT3_massCorrGeneric
                    "mass"
                )
                self.events["FatJet"] = ak.with_field(
                    self.events["FatJet"],
                    self.events["FatJet"].mass_raw * self.events["FatJet"].globalParT3_massCorrX2p,  # Check, if we should use globalParT3_massCorrGeneric
                    "mass_x2p"
                )
            else:
                self.events["FatJet"] = ak.with_field(self.events["FatJet"], self.events["FatJet"]["mass"], "mass_orig")
                self.events["FatJet"] = ak.with_field(
                    self.events["FatJet"],
                    self.events["FatJet"].mass * self.events["FatJet"].particleNet_massCorr,
                    "mass"
                )
                # self.events["FatJetGood"] = ak.with_field(
                #     self.events["FatJetGood"],
                #     self.events["FatJetGood"].pt_raw * self.events["FatJetGood"].particleNet_massCorr,
                #     "pt_regr"
                # )

            # Select FatJets for the boosted category
            self.events["FatJetGood"] = copy.copy(self.events.FatJet)

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
            fatjet_obj_presel_lead = self.params.object_preselection["FatJetLeading"]
            fatjet_obj_presel_sublead = self.params.object_preselection["FatJetSubLeading"]
            if "mass_regr_min" in fatjet_obj_presel_lead.keys() and "mass_regr_max" in fatjet_obj_presel_lead.keys():
                mask_mass_regr = (
                    (self.events["FatJetGood"]["mass"] >= fatjet_obj_presel_lead["mass_regr_min"]) &
                    (self.events["FatJetGood"]["mass"] <= fatjet_obj_presel_lead["mass_regr_max"])
                )
                mask_fat_lead = mask_fat_lead & mask_mass_regr
            if "mass_regr_min" in fatjet_obj_presel_sublead.keys() and "mass_regr_max" in fatjet_obj_presel_sublead.keys():
                mask_mass_regr = (
                    (self.events["FatJetGood"]["mass"] >= fatjet_obj_presel_sublead["mass_regr_min"]) &
                    (self.events["FatJetGood"]["mass"] <= fatjet_obj_presel_sublead["mass_regr_max"])
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
            self.events["nFatJet"] = ak.num(self.events["FatJet"], axis=1)
            self.events["nFatJetGood"] = ak.num(self.events["FatJetGood"], axis=1)

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
                self.define_vbf_jet_collections()

            if self.boosted:
                self.events["JetGood"], mask_jet_vbf = custom_jet_selection(
                    self.events,
                    "Jet",
                    "JetBoosted",
                    self.params,
                    year=self._year,
                    pt_type="pt",
                    pt_cut_name=self.pt_cut_name,
                    forward_jet_veto=False,
                )
                self.events["JetGoodVBF"], mask_jet_vbf = custom_jet_selection(
                    self.events,
                    "Jet",
                    "JetVBF",
                    self.params,
                    year=self._year,
                    pt_type="pt",
                    pt_cut_name=self.pt_cut_name,
                    forward_jet_veto=False,
                )
                self.events["JetGoodVBF"] = self.events.JetGoodVBF[
                    ak.argsort(self.events.JetGoodVBF.pt, axis=1, ascending=False)
                ]
                # alias needed by count_objects of the common workflow
                self.events["JetGoodVBFCandidates"] = self.events["JetGoodVBF"]
                self.events["JetVBF"] = copy.copy(self.events.Jet)
                self.events["HT_jetJetGoodVBF"] = ak.sum(self.events.JetGoodVBF.pt, axis=1)
                self.events["HT_jetJetGood"] = ak.sum(self.events.JetGood.pt, axis=1)
                self.events["HT_jetJet"] = ak.sum(self.events.Jet.pt, axis=1)
                self.events["HT"] = ak.sum(self.events.Jet.pt, axis=1)

                self.events["JetGoodCloseToFatJet"], mask_jet_close_to_fatjet = custom_jet_selection(
                    self.events,
                    "Jet",
                    "JetNearFatJet",
                    self.params,
                    year=self._year,
                    pt_type="pt",
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

                self.events["JetGoodVBFEnergyOrderedPadded"] = get_lead_mjj_jet_pair(
                    self.events, "JetGoodVBF"
                )
                self.events["JetGoodVBFGood"] = self.events["JetGoodVBFEnergyOrderedPadded"][~ak.is_none(self.events["JetGoodVBFEnergyOrderedPadded"]["pt"], axis=1)]
                for jet_coll in ["JetGood", "JetGoodVBF", "JetGoodVBFGood"]:
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
            self.events["nJetGoodVBFEnergyOrderedPadded"] = ak.sum(~ak.is_none(self.events.JetGoodVBFEnergyOrderedPadded, axis=1), axis=1)
            self.events["nJetGoodVBFGood"] = ak.num(self.events.JetGoodVBF, axis=1)
            self.events["nJet"] = ak.num(self.events.JetGoodVBF, axis=1)

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        if self.vbf_analysis and not self.boosted:

            self.define_vbf_pair_collections()

            # Define mjj, delta eta and centrality of leading mjj vbf jet candidates
            # the Higgs candidates are only built after the common workflow runs,
            # so the centrality cannot be computed here
            self.define_vbf_kinematics(
                ["JetGoodVBFMergedProvVBFPadded"], [0], centrality=False
            )

        super().process_extra_after_presel(variation=variation)

        if self.bdt_model:
            bdt_events = get_default_bdt_inputs(self.events)
            self.events["boosted_bdt_score"], self.events["boosted_bdt_vbf_score"] = evaluate_bdt(self.bdt_model, bdt_events)

    def build_boosted_variables(self):
        """Define the Higgs variable used for a boosted analysis."""
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
        self.events["HiggsLeading"] = ak.with_field(
            self.events["HiggsLeading"],
            ak.fill_none(disc_TXbb(self.events.HiggsLeading.btagBBTXbb), -999),
            "btagBBTXbb_dig",
            )
        self.events["HiggsLeading"] = ak.with_field(
            self.events["HiggsLeading"],
            ak.fill_none(disc_TXbb(self.events.HiggsLeading.btagBBTXbb, "old"), -999),
            "btagBBTXbb_dig_old",
            )
