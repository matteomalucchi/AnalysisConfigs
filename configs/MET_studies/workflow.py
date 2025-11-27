import awkward as ak
import os
import cachetools
import numpy as np
import vector

vector.register_awkward()

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.jets import met_correction_after_jec
from pocket_coffea.lib.leptons import lepton_selection, get_dilepton
from pocket_coffea.lib.deltaR_matching import object_matching, deltaR_matching_nonunique

from configs.jme.workflow import QCDBaseProcessor
from configs.jme.custom_cut_functions import jet_selection_nopu
from utils.basic_functions import add_fields, align_by_eta
from configs.MET_studies.custom_selections import jet_type1_selection, muon_selection_custom


class METProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)
        self.only_physical_jet = self.workflow_options["only_physical_jet"]
        self.rescale_MET_with_regressed_pT = self.workflow_options[
            "rescale_MET_with_regressed_pT"
        ]
        self.jec_pt_threshold = self.workflow_options["jec_pt_threshold"]
        self.consider_all_jets = self.workflow_options["consider_all_jets"]
        self.add_low_pt_jets = self.workflow_options["add_low_pt_jets"]
        self.jet_regressed_option = self.workflow_options["jet_regressed_option"]

    def add_GenMET_plus_neutrino(self):
        # Add the neutrinos to the GetJets to compute the MET with neutrinos
        neutrinos = self.events["GenPart"][
            (abs(self.events.GenPart.pdgId) == 12)
            | (abs(self.events.GenPart.pdgId) == 14)
            | (abs(self.events.GenPart.pdgId) == 16)
        ]
        self.events["GenJetPlusNeutrino"] = QCDBaseProcessor.add_neutrinos_to_genjets(
            self, self.events["GenJetGood"], neutrinos
        )

        GenMETPlusNeutrino = met_correction_after_jec(
            self.events,
            "GenMET",
            self.events.GenJetGood,
            self.events.GenJetPlusNeutrino,
        )
        self.events["GenMETPlusNeutrino"] = ak.with_field(
            self.events["GenMET"], GenMETPlusNeutrino["pt"], "pt"
        )
        self.events["GenMETPlusNeutrino"] = ak.with_field(
            self.events["GenMETPlusNeutrino"], GenMETPlusNeutrino["phi"], "phi"
        )

    def get_low_pt_jets(self):
        # consider the lowpt jets collection for type 1 met correction
        jet_low_pt = ak.copy(self.events["CorrT1METJet"])
        jet_low_pt = ak.with_field(
            jet_low_pt,
            jet_low_pt.rawPt,
            "pt",
        )
        jet_low_pt = ak.with_field(
            jet_low_pt,
            ak.zeros_like(jet_low_pt.pt, dtype=np.float32),
            "mass",
        )
        jet_low_pt = ak.with_field(
            jet_low_pt,
            ak.zeros_like(jet_low_pt.pt, dtype=np.float32),
            "rawFactor",
        )
        jet_low_pt = ak.with_field(
            jet_low_pt,
            ak.ones_like(jet_low_pt.pt, dtype=np.float32),
            "PNetRegPtRawCorr",
        )
        jet_low_pt = ak.with_field(
            jet_low_pt,
            ak.ones_like(jet_low_pt.pt, dtype=np.float32),
            "PNetRegPtRawCorrNeutrino",
        )
        jet_low_pt = ak.with_field(
            jet_low_pt,
            ak.zeros_like(jet_low_pt.pt, dtype=np.float32) - 1,
            "btagPNetB",
        )
        jet_low_pt = ak.with_field(
            jet_low_pt,
            ak.zeros_like(jet_low_pt.pt, dtype=np.float32) - 1,
            "btagPNetCvL",
        )
        if "EmEF" not in jet_low_pt.fields:
            # EmEF is needed for the jet cleaning in the type1 met correction
            jet_low_pt = ak.with_field(
                jet_low_pt,
                ak.zeros_like(jet_low_pt.pt, dtype=np.float32) - 1,
                "EmEF",
            )
        jet_low_pt = add_fields(jet_low_pt, "all", four_vec="Momentum4D")

        return jet_low_pt

    def process_extra_after_skim(self):
        self.jet_good_list = ["JetGood"]

        # compute EmEF for the jets
        self.events["Jet"] = ak.with_field(
            self.events["Jet"],
            self.events["Jet"].chEmEF + self.events["Jet"].neEmEF,
            "EmEF",
        )
        if "pt_raw" not in self.events["Jet"].fields:
            self.events["Jet"] = ak.with_field(
                self.events["Jet"],
                self.events["Jet"].pt * (1 - self.events["Jet"].rawFactor),
                "pt_raw",
            )
            self.events["Jet"] = ak.with_field(
                self.events["Jet"],
                self.events["Jet"].mass * (1 - self.events["Jet"].rawFactor),
                "mass_raw",
            )

        if self.consider_all_jets:
            self.events["JetGood"] = ak.copy(self.events["Jet"])
        else:
            # keep only jets with pt_raw > 15 GeV and |eta| < 4.7
            self.events["JetGood"] = jet_selection_nopu(
                self.events, "Jet", self.params, "pt_raw"
            )
            if self.only_physical_jet:
                physisical_jet_mask = (
                    self.events["JetGood"].pt_raw * np.cosh(self.events["JetGood"].eta)
                    < (13.6 * 1000) / 2
                )
                self.events["JetGood"] = self.events["JetGood"][physisical_jet_mask]
            # consider only jets with regressed pt > 0
            reg_mask = self.events["JetGood"].PNetRegPtRawCorr > 0
            self.events["JetGood"] = self.events["JetGood"][reg_mask]

        # Create extra Jet collections for calibration
        self.events["JetGoodJEC"] = ak.copy(self.events["JetGood"])
        self.jet_good_list.append("JetGoodJEC")

        # Jets for type 1 met correction

        self.events["JetLowPtMuonSubtr"] = self.get_low_pt_jets()
        self.events["JetMuonSubtr"] = ak.copy(self.events["Jet"])
        self.events["JetPNetMuonSubtr"] = ak.copy(self.events["Jet"])
        self.events["JetPNetPlusNeutrinoMuonSubtr"] = ak.copy(self.events["Jet"])

        for jet_coll in [
            "JetMuonSubtr",
            "JetLowPtMuonSubtr",
            "JetPNetMuonSubtr",
            "JetPNetPlusNeutrinoMuonSubtr",
        ]:
            # Change the rawFactor to account for muon subtraction
            # This is not exactly the correct thing to do. We shouldn't be using the full
            # jet eta and phi but the eta and phi of muon-less jet p4. Not possible with
            # current NanoAODs.
            # This is the factor that is used to get the raw muon-less jet pT and mass
            self.events[jet_coll] = ak.with_field(
                self.events[jet_coll],
                1
                - (
                    (1 - self.events[jet_coll].rawFactor)
                    * (1 - self.events[jet_coll].muonSubtrFactor)
                ),
                "rawFactor",
            )

            if "muonSubtrDeltaEta" in self.events[jet_coll].fields:
                self.events[jet_coll] = ak.with_field(
                    self.events[jet_coll],
                    self.events[jet_coll].muonSubtrDeltaEta + self.events[jet_coll].eta,
                    "eta",
                )
            if "muonSubtrDeltaPhi" in self.events[jet_coll].fields:
                self.events[jet_coll] = ak.with_field(
                    self.events[jet_coll],
                    self.events[jet_coll].muonSubtrDeltaPhi + self.events[jet_coll].phi,
                    "phi",
                )

    def apply_object_preselection(self, variation):
        self.events["GenJetGood"] = self.events.GenJet[
            self.events.GenJet.pt > self.params.object_preselection["GenJet"]["pt"]
        ]
        saved_fields = [
            "pt",
            "mass",
            "phi",
            "eta",
            "EmEF",
            "pt_raw",
            "mass_raw",
            "rawFactor",
        ]

        for jet_name in [
            "JetMuonSubtr",
            "JetPNetMuonSubtr",
            "JetPNetPlusNeutrinoMuonSubtr",
        ]:
            jets_notNone = self.events[jet_name]
            if "PNet" in jet_name:
                if self.jet_regressed_option == "option_1":
                    pass

                elif (
                    self.jet_regressed_option == "option_2"
                    or self.jet_regressed_option == "option_5"
                ):
                    # add the jets without the regression back in the collection
                    jets_notNone = align_by_eta(
                        self.events["JetMuonSubtr"], jets_notNone
                    )

                    # for the raw variables, take the ones from the standard jets
                    # because the pt_raw of the regressed jets is actually the regressed pt
                    # before the correction
                    for field in [f for f in saved_fields if "raw" in f]:
                        jets_notNone = ak.with_field(
                            jets_notNone,
                            ak.values_astype(
                                self.events["JetMuonSubtr"][field], "float32"
                            ),
                            field,
                        )

                elif (
                    self.jet_regressed_option == "option_3"
                    or self.jet_regressed_option == "option_4"
                ):
                    # remove the jets without regressed pt from the Jet collection
                    jets_None = align_by_eta(
                        self.events["JetMuonSubtr"], jets_notNone, put_none=True
                    )
                    mask_None = ak.is_none(jets_None, axis=1)

                    # for the raw variables, take the ones from the standard jets
                    # because the pt_raw of the regressed jets is actually the regressed pt
                    # before the correction
                    for field in [f for f in saved_fields if "raw" in f]:
                        jets_None = ak.with_field(
                            jets_None,
                            ak.values_astype(
                                self.events["JetMuonSubtr"][field], "float32"
                            ),
                            field,
                        )

                    jets_notNone = jets_None[~mask_None]

                else:
                    raise ValueError(
                        f"Unknown jet_regressed_option {self.jet_regressed_option}"
                    )

            jet_name_corr = jet_name.replace("MuonSubtr", "CorrMET")

            if (
                (self.jet_regressed_option == "option_4" and "PNet" in jet_name)
                or (self.jet_regressed_option == "option_5" and "PNet" in jet_name)
                or (not self.add_low_pt_jets and "PNet" not in jet_name)
            ):
                # Do not add the low pt jets to the collection
                self.events[jet_name_corr] = jets_notNone
            else:
                # Add the low pt jets to the collection
                self.events[jet_name_corr] = ak.concatenate(
                    [jets_notNone, self.events["JetLowPtMuonSubtr"]],
                    axis=1,
                )

            jet_good_name_corr = jet_name_corr.replace("Jet", "JetGood")
            self.events[jet_good_name_corr] = jet_type1_selection(
                self.events, jet_name_corr, self.params
            )
            self.events[jet_good_name_corr] = add_fields(
                self.events[jet_good_name_corr], saved_fields, four_vec="Momentum4D"
            )
            self.jet_good_list.append(jet_good_name_corr)

        self.met_branches = ["RawPuppiMET", "PuppiMET"]

        for jet_coll_name in self.jet_good_list:
            # Create the raw pt collection
            if self.jet_regressed_option == "option_1" and "PNet" in jet_coll_name:
                jet_raw_coll_name = "JetGoodCorrMET"
            else:
                jet_raw_coll_name = jet_coll_name

            jets_raw = ak.zip(
                {
                    "pt": self.events[jet_raw_coll_name].pt_raw,
                    "eta": self.events[jet_raw_coll_name].eta,
                    "phi": self.events[jet_raw_coll_name].phi,
                    "mass": self.events[jet_raw_coll_name].mass_raw,
                },
                with_name="Momentum4D",
            )

            if self.rescale_MET_with_regressed_pT:
                for met_branch, jet_coll_to_remove in zip(
                    ["RawPuppiMET", "PuppiMET"],
                    [jets_raw, self.events.JetGood],
                ):

                    new_MET = met_correction_after_jec(
                        self.events,
                        met_branch,
                        jet_coll_to_remove,
                        self.events[jet_coll_name],
                    )
                    jet_coll_suffix = jet_coll_name.split("JetGood")[-1]
                    new_met_branch = f"{met_branch}-Type1{jet_coll_suffix}"

                    self.events[new_met_branch] = ak.zip(
                        {
                            "pt": new_MET["pt"],
                            "phi": new_MET["phi"],
                        },
                    )
                    self.met_branches.append(new_met_branch)

        self.events["MuonGood"] = muon_selection_custom(self.events, self.params)
        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params
        )

        # di lepton is composed by the 2 leading muons
        self.events["ll"] = get_dilepton(None, self.events.MuonGood)

    def get_hadronic_recoil(self, qT, MET_coll):
        met = self.events[MET_coll]

        hadronic_recoil_px = -(met.pt * np.cos(met.phi) + qT.x)
        hadronic_recoil_py = -(met.pt * np.sin(met.phi) + qT.y)

        hadronic_recoil_pt = np.sqrt(hadronic_recoil_px**2 + hadronic_recoil_py**2)
        hadronic_recoil_phi = np.arctan2(hadronic_recoil_py, hadronic_recoil_px)

        hadronic_recoil = ak.zip(
            {
                "px": hadronic_recoil_px,
                "py": hadronic_recoil_py,
                "pt": hadronic_recoil_pt,
                "phi": hadronic_recoil_phi,
            },
            with_name="Momentum2D",
        )

        return hadronic_recoil

    def compute_projections_qT(self, hadronic_recoil_coll):
        v_qT = self.events["qT"]

        # hadronic recoil
        u_vec = self.events[hadronic_recoil_coll]
        # compute dot product of v_qT and MET
        response = v_qT.dot(-u_vec) / v_qT.dot(v_qT)

        v_paral_predict = v_qT.dot(u_vec) / v_qT.rho

        # subtract the module of qT because the projection
        # of u is in the direction opposite of qT
        u_paral_predict = v_paral_predict + v_qT.rho

        # vector 90 degrees to the left of v_qT
        v_qT_perp = ak.zip(
            {
                "x": -v_qT.y,
                "y": v_qT.x,
            },
            with_name="Vector2D",
        )

        u_perp_predict = v_qT_perp.dot(u_vec) / v_qT_perp.rho

        return u_perp_predict, u_paral_predict, response

    def process_extra_after_presel(self, variation) -> ak.Array:
        self.events["qT"] = ak.zip(
            {"x": self.events["ll"].px, "y": self.events["ll"].py},
            with_name="Vector2D",
        )
        for MET_coll in self.met_branches:
            # substract leptons from MET
            self.events[f"u{MET_coll}"] = self.get_hadronic_recoil(
                self.events["qT"], MET_coll
            )

            u_perp_predict, u_paral_predict, response = self.compute_projections_qT(
                f"u{MET_coll}"
            )
            self.events[f"u{MET_coll}"] = ak.with_field(
                self.events[f"u{MET_coll}"],
                u_perp_predict,
                "u_perp_predict",
            )
            self.events[f"u{MET_coll}"] = ak.with_field(
                self.events[f"u{MET_coll}"],
                u_paral_predict,
                "u_paral_predict",
            )
            self.events[f"u{MET_coll}"] = ak.with_field(
                self.events[f"u{MET_coll}"], response, "response"
            )

    def count_objects(self, variation):
        self.events["nMuonGood"] = ak.num(self.events.MuonGood)
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood)
        self.events["nJetGood"] = ak.num(self.events.JetGood)
