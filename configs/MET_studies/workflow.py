import awkward as ak
import os
import cachetools
import numpy as np
import vector

vector.register_awkward()

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.jets import jet_correction, met_correction_after_jec
from pocket_coffea.lib.leptons import lepton_selection, get_dilepton
from pocket_coffea.lib.deltaR_matching import object_matching, deltaR_matching_nonunique

from configs.jme.workflow import QCDBaseProcessor
from configs.jme.custom_cut_functions import jet_selection_nopu


# def getdot(vx, vy):
#     return np.einsum("bi,bi->b", vx, vy)


# def getscale(vx):
#     return np.sqrt(self.getdot(vx, vx))


# def scalermul(a, v):
#     return np.einsum("b,bi->bi", a, v)


class METProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)
        self.only_physical_jet = self.workflow_options["only_physical_jet"]
        self.rescale_MET_with_regressed_pT = self.workflow_options[
            "rescale_MET_with_regressed_pT"
        ]
        self.jec_pt_threshold = self.workflow_options["jec_pt_threshold"]

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
        # print("GenMet", self.events["GenMET"].pt)
        # print("GenMetPlusNeutrino", self.events["GenMETPlusNeutrino"].pt)

    def apply_object_preselection(self, variation):
        # super().apply_object_preselection(variation)
        # print("jet pt raw", self.events["Jet"].pt_raw)
        if "pt_raw" not in self.events["Jet"].fields:
            # print("Compute raw pt and mass for Jet")
            self.events["Jet"] = ak.with_field(
                self.events["Jet"],
                self.events["Jet"].pt
                * (1 - self.events["Jet"].rawFactor),
                "pt_raw",
            )
            self.events["Jet"] = ak.with_field(
                self.events["Jet"],
                self.events["Jet"].mass
                * (1 - self.events["Jet"].rawFactor),
                "mass_raw",
            )
        # print("Jet pt raw", self.events["Jet"].pt_raw)

        # keep only jets with pt_raw > 15 GeV and |eta| < 4.7
        self.events["JetPuppiMET"] = jet_selection_nopu(
            self.events, "Jet", self.params, "pt_raw"
        )

        self.events["GenJetGood"] = self.events.GenJet[
            self.events.GenJet.pt > self.params.object_preselection["GenJet"]["pt"]
        ]

        if self.only_physical_jet:
            physisical_jet_mask = (
                self.events.JetPuppiMET.pt_raw * np.cosh(self.events.JetPuppiMET.eta)
                < (13.6 * 1000) / 2
            )
            self.events["JetPuppiMET"] = self.events.JetPuppiMET[physisical_jet_mask]

        jets_raw = ak.zip(
            {
                "pt": self.events.JetPuppiMET.pt_raw,
                "eta": self.events.JetPuppiMET.eta,
                "phi": self.events.JetPuppiMET.phi,
                "mass": self.events.JetPuppiMET.mass_raw,
            },
            with_name="Momentum4D",
        )

        cache = cachetools.Cache(np.inf)
        jets_dict = {}

        jet_calib_params = self.params.jets_calibration
        for jet_type, jet_coll_name in jet_calib_params.collection[self._year].items():
            # if "chs" in jet_type or "Puppi" in jet_type:
            #     continue
            
            if "PNet" in jet_coll_name:
                # For regression
                # define the pnet reg jet colleciton
                jets_dict[jet_coll_name] = ak.with_field(
                    self.events["JetPuppiMET"],
                    self.events.JetPuppiMET.pt_raw
                    * self.events.JetPuppiMET.PNetRegPtRawCorr
                    * (
                        self.events.JetPuppiMET.PNetRegPtRawCorrNeutrino
                        if "Neutrino" in jet_coll_name
                        else 1
                    ),
                    # ak.where(
                    #     self.events.JetPuppiMET.PNetRegPtRawCorr > 0,
                    #     self.events.JetPuppiMET.pt_raw
                    #     * self.events.JetPuppiMET.PNetRegPtRawCorr
                    #     * (
                    #         self.events.JetPuppiMET.PNetRegPtRawCorrNeutrino
                    #         if "Neutrino" in jet_coll_name
                    #         else 1
                    #     ),
                    #     self.events.JetPuppiMET.pt,
                    # ),
                    "pt",
                )
                jets_dict[jet_coll_name] = ak.with_field(
                    jets_dict[jet_coll_name],
                    jets_dict[jet_coll_name].pt,
                    "pt_raw",
                )
                jets_dict[jet_coll_name] = ak.with_field(
                    jets_dict[jet_coll_name],
                    ak.zeros_like(jets_dict[jet_coll_name].pt),
                    "rawFactor",
                )

                jets_dict[jet_coll_name] = ak.with_field(
                    jets_dict[jet_coll_name],
                    jets_dict[jet_coll_name].mass_raw
                    * jets_dict[jet_coll_name].PNetRegPtRawCorr
                    * (
                        jets_dict[jet_coll_name].PNetRegPtRawCorrNeutrino
                        if "Neutrino" in jet_coll_name
                        else 1
                    ),
                    # ak.where(
                    #     jets_dict[jet_coll_name].PNetRegPtRawCorr > 0,
                    #     jets_dict[jet_coll_name].mass_raw
                    #     * jets_dict[jet_coll_name].PNetRegPtRawCorr
                    #     * (
                    #         jets_dict[jet_coll_name].PNetRegPtRawCorrNeutrino
                    #         if "Neutrino" in jet_coll_name
                    #         else 1
                    #     ),
                    #     jets_dict[jet_coll_name].mass,
                    # ),
                    "mass",
                )
                jets_dict[jet_coll_name] = ak.with_field(
                    jets_dict[jet_coll_name],
                    jets_dict[jet_coll_name].mass,
                    "mass_raw",
                )
            else:
                # For nominal jets
                jets_dict[jet_coll_name] = ak.copy(self.events["JetPuppiMET"])

            # Calibrate the jets
            jets_calib_dict = {}
            # always apply the JEC to the regressed pt
            if True or jet_calib_params.apply_jec_nominal[self._year]:
                # print(jet_type, jet_coll_name)
                # print(
                #     "PNETReg before correction",
                #     jets_dict[jet_coll_name].pt,
                #     jets_dict[jet_coll_name].mass,
                #     jets_dict[jet_coll_name].eta,
                # )

                jets_calib_dict[jet_coll_name] = jet_correction(
                    params=self.params,
                    events=self.events,
                    jets=jets_dict[jet_coll_name],
                    factory=self.jmefactory,
                    jet_type=jet_type,
                    chunk_metadata={
                        "year": self._year,
                        "isMC": self._isMC,
                        "era": self._era,
                    },
                    cache=cache,
                )
                # print(
                #     "PNETReg after correction",
                #     jets_calib_dict[jet_coll_name].pt,
                #     jets_calib_dict[jet_coll_name].mass,
                #     jets_calib_dict[jet_coll_name].eta,
                # )
            else:
                jets_calib_dict[jet_coll_name] = jets_dict[jet_coll_name]

            jet_coll_suffix = jet_coll_name.split("Jet")[-1]
            # print("jet_coll_suffix", jet_coll_suffix)

            # Correct MET with MC Truth only jets with pt reg > 15
            corr_reg_pt_mask = jets_calib_dict[jet_coll_name].pt_raw > self.jec_pt_threshold
            # compute pt
            self.events[f"JetPuppiMET{jet_coll_suffix}"] = ak.with_field(
                jets_calib_dict[jet_coll_name],
                ak.where(
                    corr_reg_pt_mask,
                    jets_calib_dict[jet_coll_name].pt,
                    jets_calib_dict[jet_coll_name].pt_raw,
                ),
                "pt",
            )

            # print("jets_dict[jet_coll_name] pt", jets_dict[jet_coll_name].pt)
            # print(f"JetPuppiMET{jet_coll_suffix} pt ", self.events[f"JetPuppiMET{jet_coll_suffix}"].pt)

            # corr_reg_pt_mask_flat = ak.flatten(corr_reg_pt_mask, axis=1)
            # jets_flat=ak.flatten(self.events.JetPuppiMET, axis=1)[~corr_reg_pt_mask_flat]
            # jets_reg_flat = ak.flatten(jets_dict[jet_coll_name], axis=1)[~corr_reg_pt_mask_flat]
            # jets_reg_corr_flat = ak.flatten(
            #     self.events[f"JetPuppiMET{jet_coll_suffix}"], axis=1
            # )[~corr_reg_pt_mask_flat]
            # print("jets_flat", jets_flat.pt, jets_flat.eta)
            # print("jets_reg_flat", jets_reg_flat.pt, jets_reg_flat.eta)
            # print(
            #     f"jets_reg_corr_flat", jets_reg_corr_flat.pt, jets_reg_corr_flat.eta
            # )

            # compute px and py
            self.events[f"JetPuppiMET{jet_coll_suffix}"] = ak.with_field(
                self.events[f"JetPuppiMET{jet_coll_suffix}"],
                self.events[f"JetPuppiMET{jet_coll_suffix}"].pt
                * np.cos(self.events[f"JetPuppiMET{jet_coll_suffix}"].phi),
                "px",
            )
            self.events[f"JetPuppiMET{jet_coll_suffix}"] = ak.with_field(
                self.events[f"JetPuppiMET{jet_coll_suffix}"],
                self.events[f"JetPuppiMET{jet_coll_suffix}"].pt
                * np.sin(self.events[f"JetPuppiMET{jet_coll_suffix}"].phi),
                "py",
            )
            # compute mass
            self.events[f"JetPuppiMET{jet_coll_suffix}"] = ak.with_field(
                self.events[f"JetPuppiMET{jet_coll_suffix}"],
                ak.where(
                    corr_reg_pt_mask,
                    self.events[f"JetPuppiMET{jet_coll_suffix}"][jet_coll_name].mass,
                    self.events[f"JetPuppiMET{jet_coll_suffix}"][jet_coll_name].mass_raw,
                ),
                "mass",
            )

            # print(f"JetPuppiMET", self.events[f"JetPuppiMET"].pt)
            # print(
            #     f"JetPuppiMET{jet_coll_suffix}",
            #     self.events[f"JetPuppiMET{jet_coll_suffix}"].pt,
            # )

            # print("jetpuppimet pt", self.events.JetPuppiMET.pt)
            # print("jetpuppimetPNet pt", self.events[f"JetPuppiMET{jet_coll_suffix}"].pt)

            # print("jetpuppimet mass", self.events.JetPuppiMET.mass)
            # print("jetpuppimetPNet mass", self.events[f"JetPuppiMET{jet_coll_suffix}"].mass)

            if self.rescale_MET_with_regressed_pT:
                # met_branch = jet_calib_params.rescale_MET_branch[self._year]

                raw_met_branch = "RawPuppiMET"
                # print(
                #     met_branch, self.events[met_branch].pt, self.events[met_branch].phi
                # )
                # print(
                #     "JetPuppiMET px",
                #     self.events["JetPuppiMET"].px,
                #     "py",
                #     self.events["JetPuppiMET"].py,
                # )
                # print(
                #     "jet_dict px",
                #     self.events[f"JetPuppiMET{jet_coll_suffix}"].px,
                #     "py",
                #     self.events[f"JetPuppiMET{jet_coll_suffix}"].py,
                # )

                new_MET = met_correction_after_jec(
                    self.events,
                    raw_met_branch,
                    jets_raw,
                    self.events[f"JetPuppiMET{jet_coll_suffix}"],
                )
                
                met_branch=raw_met_branch.replace("Raw", "")
                self.events[f"{met_branch}{jet_coll_suffix}"] = ak.zip(
                    {
                        "pt": new_MET["pt"],
                        "phi": new_MET["phi"],
                    },
                )

                # print(
                #     f"{met_branch}{jet_coll_suffix}",
                #     self.events[f"{met_branch}{jet_coll_suffix}"].pt,
                #     self.events[f"{met_branch}{jet_coll_suffix}"].phi,
                # )
            # breakpoint()

        # self.add_GenMET_plus_neutrino()

        self.events["MuonGood"] = lepton_selection(self.events, "Muon", self.params)
        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params
        )
        self.events["ll"] = get_dilepton(None, self.events.MuonGood)

    def subtract_leptons_from_MET(self, lepton_coll, MET_coll):
        met = self.events[MET_coll]
        leptons_px = ak.sum(
            self.events[lepton_coll].pt * np.cos(self.events[lepton_coll].phi), axis=1
        )
        leptons_py = ak.sum(
            self.events[lepton_coll].pt * np.sin(self.events[lepton_coll].phi), axis=1
        )
        new_met_px = met.pt * np.cos(met.phi) + leptons_px
        new_met_py = met.pt * np.sin(met.phi) + leptons_py

        new_met_pt = np.sqrt(new_met_px**2 + new_met_py**2)
        new_met_phi = np.arctan2(new_met_py, new_met_px)

        # print(MET_coll)
        # print("old_met", met.pt * np.cos(met.phi), met.pt * np.sin(met.phi))
        # print("leptons", leptons_px, leptons_py)
        # print("new_met", new_met_px, new_met_py)

        return ak.zip(
            {"px": new_met_px, "py": new_met_py, "pt": new_met_pt, "phi": new_met_phi},
            with_name="Momentum2D",
        )

    def compute_projections_qT(self, MET_coll):
        # print("MET_coll", MET_coll)
        v_qT = self.events["qT"]
        # print("v_qT", v_qT.px, v_qT.py, v_qT.rho)
        # print("MET", self.events[MET_coll].px, self.events[MET_coll].py)

        # hadronic recoil
        u_vec = -self.events[MET_coll]
        # compute dot product of v_qT and MET
        response = v_qT.dot(-u_vec) / v_qT.dot(v_qT)

        # print("response", response)

        v_paral_predict = v_qT.dot(u_vec) / v_qT.rho

        # print("v_paral_predict", v_paral_predict.px, v_paral_predict.py)

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

        # print("u_paral_predict", u_paral_predict)
        # print("u_perp_predict", u_perp_predict)

        return u_perp_predict, u_paral_predict, response

        # response_np = getdot(
        #     self.events[MET_coll], v_qT
        # ) / getdot(v_qT, v_qT)

        # print("response_np",response_np)

    def process_extra_after_presel(self, variation) -> ak.Array:
        self.events["qT"] = ak.zip(
            {"x": self.events["ll"].px, "y": self.events["ll"].py},
            with_name="Vector2D",
        )
        # substract leptons from MET
        for MET_coll in [
            "PuppiMET",
            "RawPuppiMET",
            "PuppiMETType1",
            "PuppiMETPNet",
            "PuppiMETPNetPlusNeutrino",
            # "GenMET",
            # "GenMETPlusNeutrino",
        ]:
            lepton_coll = "MuonGood"
            self.events[f"{MET_coll}_{lepton_coll}"] = self.subtract_leptons_from_MET(
                lepton_coll, MET_coll
            )
            # print(f"{MET_coll}", self.events[f"{MET_coll}"].pt)
            # print(
            #     f"{MET_coll}_{lepton_coll}", self.events[f"{MET_coll}_{lepton_coll}"].pt
            # )

            u_perp_predict, u_paral_predict, response = self.compute_projections_qT(
                f"{MET_coll}_{lepton_coll}"
            )
            self.events[f"{MET_coll}_{lepton_coll}"] = ak.with_field(
                self.events[f"{MET_coll}_{lepton_coll}"],
                u_perp_predict,
                "u_perp_predict",
            )
            self.events[f"{MET_coll}_{lepton_coll}"] = ak.with_field(
                self.events[f"{MET_coll}_{lepton_coll}"],
                u_paral_predict,
                "u_paral_predict",
            )
            self.events[f"{MET_coll}_{lepton_coll}"] = ak.with_field(
                self.events[f"{MET_coll}_{lepton_coll}"], response, "response"
            )

    def count_objects(self, variation):
        self.events["nMuonGood"] = ak.num(self.events.MuonGood)
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood)
        self.events["nJetPuppiMET"] = ak.num(self.events.JetPuppiMET)
