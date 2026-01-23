import logging

import awkward as ak
import numpy as np
import vector
import copy
from pocket_coffea.lib.deltaR_matching import object_matching
from pocket_coffea.workflows.base import BaseProcessorABC

from utils.basic_functions import add_fields, align_by_eta
from utils.dnn_evaluation_functions import get_dnn_prediction

# from utils.inference_session_onnx_slurm import get_model_session
from utils.inference_session_onnx import get_model_session
from utils.parton_matching_function import get_parton_last_copy
from utils.reconstruct_higgs_candidates import (
    get_jets_no_higgs_from_idx,
    reconstruct_higgs_from_idx,
    reconstruct_higgs_from_provenance,
    run2_matching_algorithm,
)
from utils.spanet_evaluation_functions import get_best_pairings, get_pairing_information

from .custom_object_preselection_common import (
    lepton_selection,
)
from utils.custom_cut_functions import custom_jet_selection

vector.register_awkward()

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger()

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


class HH4bCommonProcessor(BaseProcessorABC):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)

        for key, value in self.workflow_options.items():
            setattr(self, key, value)

    def process_extra_after_skim(self):
        self.events["JetDefault"] = copy.copy(self.events["Jet"])
        self.events["JetPNet"] = copy.copy(self.events["Jet"])
        self.events["JetPNetPlusNeutrino"] = copy.copy(self.events["Jet"])

    def apply_object_preselection(self, variation):
        # Use the regressed pt from PNet+Neutrino collection if available,
        # otherwise use the JEC corrected pt collection
        # This way we consider correctly all fields which change depending on
        # the pt definition, namely the pt, mass and the associated systematic variations
        self.events["Jet"] = ak.where(
            ak.nan_to_num(self.events["JetPNetPlusNeutrino"].pt, nan=-1) > 0,
            self.events["JetPNetPlusNeutrino"],
            self.events.JetDefault,
        )
        # save also the different pt definitions for bookkeeping
        # we anyway miss the different mass definitions and the various variations
        self.events["Jet"] = ak.with_field(
            self.events.Jet,
            self.events.JetDefault.pt,
            "pt_default",
        )
        self.events["Jet"] = ak.with_field(
            self.events.Jet,
            self.events.JetPNetPlusNeutrino.pt,
            "pt_regressed",
        )

        if self.add_jet_spanet:
            # reorder the jets by pt regressed
            self.events["Jet"] = self.events["Jet"][
                ak.argsort(self.events["Jet"].pt, axis=1, ascending=False)
            ]

        # get index after reordering in pt
        self.events["Jet"] = ak.with_field(
            self.events.Jet, ak.local_index(self.events.Jet, axis=1), "index"
        )

        if (
            self.tight_cuts
            and "pt_tight" in self.params.object_preselection["Jet"].keys()
        ):
            self.pt_cut_name = "pt_tight"
        else:
            self.pt_cut_name = "pt"

        # Cut on the JEC pt (w/o regression)
        self.events["JetGood"], _ = custom_jet_selection(
            self.events,
            "Jet",
            self.params,
            year=self._year,
            pt_type="pt_default",
            pt_cut_name=self.pt_cut_name,
        )

        # Add btag WP
        self.events["JetGood"] = self.generate_btag_workingpoints(
            self.events["JetGood"], 5
        )
        self.events["JetGood"] = self.generate_btag_workingpoints(
            self.events["JetGood"], 3
        )

        self.events["Electron"] = ak.with_field(
            self.events.Electron,
            self.events.Electron.eta + self.events.Electron.deltaEtaSC,
            "etaSC",
        )

        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params
        )
        self.events["MuonGood"] = lepton_selection(self.events, "Muon", self.params)

        # order jet by btag score
        self.events["JetGood"] = self.events.JetGood[
            ak.argsort(self.events.JetGood.btagPNetB, axis=1, ascending=False)
        ]
        # keep only the first 4 jets for the Higgs candidates reconstruction
        self.events["JetGoodHiggs"] = self.events.JetGood[:, :4]

        # Trying to reshuffle jets 4 and above by pt instead of b-tag score
        if self.fifth_jet == "pt":
            jets5plus = self.events["JetGood"][:, 4:]
            jets5plus_pt = jets5plus[ak.argsort(jets5plus.pt, axis=1, ascending=False)]
            self.events["JetGood"] = ak.concatenate(
                (self.events["JetGoodHiggs"], jets5plus_pt), axis=1
            )

    # def apply_preselection(self, variation):
    #     """
    #     Workaround to have the possibility for preselections depending on samples
    #     Needs correct implementation in the config file to create a dict of all required samples.
    #     The keys can then be set as samples and the values are the cuts for the respective sample
    #     """
    #     self._preselections_temp = self._preselections
    #     if isinstance(self._preselections, dict):
    #         self._preselections = self._preselections_temp[self._sample]
    #     super().apply_preselection(self, variation)
    #     self._preselections = self._preselections_temp

    def generate_btag_workingpoints(self, jets, num_wp):
        # L, M, T, XT, XXT
        # Right now hardcoded particleNet postEE
        wps = self.params["btagging"]["working_point"][self._year]["btagging_WP"][
            "btagPNetB"
        ]
        btag_wp = ak.zeros_like(jets.btagPNetB, dtype=np.int32) - (
            1 if self.old_wp_def else 0
        )
        for i, thr in enumerate(sorted(wps.values())):
            if i >= num_wp:
                break
            btag_wp = ak.where(
                jets.btagPNetB > thr, i + 1 - (1 if self.old_wp_def else 0), btag_wp
            )  # NOTE: the -1 is to use the old configuration

        # raise ValueError("WARNING: change the definition")
        return ak.with_field(jets, btag_wp, f"btagPNetB_{num_wp}wp")

    def generate_btag_delta_workingpoints(self, jets, num_wp):
        wp_array = jets[f"btagPNetB_{num_wp}wp"]
        num_jets = ak.num(wp_array)
        deltaWP = ak.where(
            # if 4 jets
            num_jets == 4,
            ak.concatenate(
                [
                    (wp_array[:, 0] - wp_array[:, 1])[..., None],
                    (wp_array[:, 1] - wp_array[:, 0])[..., None],
                    (wp_array[:, 2] - wp_array[:, 3])[..., None],
                    (wp_array[:, 3] - wp_array[:, 2])[..., None],
                    wp_array[:, 4:],
                ],
                axis=1,
            ),
            # if more than 4 jets
            ak.concatenate(
                [
                    (wp_array[:, 0] - wp_array[:, 1])[..., None],
                    (wp_array[:, 1] - wp_array[:, 0])[..., None],
                    (wp_array[:, 2] - wp_array[:, 3])[..., None],
                    (wp_array[:, 3] - wp_array[:, 2])[..., None],
                    (ak.pad_none(wp_array, 5)[:, 4] - wp_array[:, 3])[..., None],
                    # (wp_array[:, 4] - wp_array[:, 3])[..., None],
                    wp_array[:, 5:],
                ],
                axis=1,
            ),
        )
        return ak.with_field(jets, deltaWP, f"btagPNetB_delta{num_wp}wp")

    def get_jet_higgs_provenance(self, which_bquark):  # -> ak.Array:
        # Select b-quarks at Gen level, coming from H->bb decay
        self.events["GenPart"] = ak.with_field(
            self.events.GenPart, ak.local_index(self.events.GenPart, axis=1), "index"
        )
        genpart = self.events.GenPart

        isHiggs = genpart.pdgId == 25
        isB = abs(genpart.pdgId) == 5
        isLast = genpart.hasFlags(["isLastCopy"])
        isFirst = genpart.hasFlags(["isFirstCopy"])
        isHard = genpart.hasFlags(["fromHardProcess"])

        higgs = genpart[isHiggs & isLast & isHard]
        higgs = higgs[ak.num(higgs.childrenIdxG, axis=2) == 2]
        higgs = higgs[ak.argsort(higgs.pt, ascending=False)]

        if which_bquark == "last_numba":
            bquarks_first = genpart[isB & isHard & isFirst]
            mother_bquarks = genpart[bquarks_first.genPartIdxMother]
            bquarks_from_higgs = bquarks_first[mother_bquarks.pdgId == 25]
            provenance = ak.where(
                bquarks_from_higgs.genPartIdxMother == higgs.index[:, 0], 1, 2
            )

            # define variables to get the last copy
            children_idxG = ak.without_parameters(genpart.childrenIdxG, behavior={})
            children_idxG_flat = ak.flatten(children_idxG, axis=1)
            genpart_pdgId_flat = ak.flatten(
                ak.without_parameters(genpart.pdgId, behavior={}), axis=1
            )
            genpart_LastCopy_flat = ak.flatten(
                ak.without_parameters(genpart.hasFlags(["isLastCopy"]), behavior={}),
                axis=1,
            )
            genpart_pt_flat = ak.flatten(
                ak.without_parameters(genpart.pt, behavior={}), axis=1
            )
            genparts_flat = ak.flatten(genpart)
            genpart_offsets = np.concatenate(
                [
                    [0],
                    np.cumsum(ak.to_numpy(ak.num(genpart, axis=1), allow_missing=True)),
                ]
            )
            b_quark_idx = ak.to_numpy(
                bquarks_from_higgs.index + genpart_offsets[:-1], allow_missing=False
            )
            b_quarks_pdgId = ak.to_numpy(bquarks_from_higgs.pdgId, allow_missing=False)
            nevents = b_quark_idx.shape[0]
            firstgenpart_idxG = ak.firsts(genpart[:, 0].children).genPartIdxMotherG
            firstgenpart_idxG_numpy = ak.to_numpy(
                firstgenpart_idxG, allow_missing=False
            )

            b_quark_last_idx = get_parton_last_copy(
                b_quark_idx,
                b_quarks_pdgId,
                children_idxG_flat,
                genpart_pdgId_flat,
                genpart_offsets,
                genpart_LastCopy_flat,
                genpart_pt_flat,
                nevents,
                firstgenpart_idxG_numpy,
            )
            bquarks = genparts_flat[b_quark_last_idx]

        elif which_bquark == "last":
            bquarks = genpart[isB & isLast & isHard]
            bquarks_first = bquarks
            while True:
                b_mother = genpart[bquarks_first.genPartIdxMother]
                mask_mother = (abs(b_mother.pdgId) == 5) | ((b_mother.pdgId) == 25)
                bquarks = bquarks[mask_mother]
                bquarks_first = bquarks_first[mask_mother]
                b_mother = b_mother[mask_mother]
                if ak.all((b_mother.pdgId) == 25):
                    break
                bquarks_first = ak.where(
                    abs(b_mother.pdgId) == 5, b_mother, bquarks_first
                )
            provenance = ak.where(
                bquarks_first.genPartIdxMother == higgs.index[:, 0], 1, 2
            )
        elif which_bquark == "first":
            bquarks = ak.flatten(higgs.children, axis=2)
            provenance = ak.where(bquarks.genPartIdxMother == higgs.index[:, 0], 1, 2)
        else:
            raise ValueError(
                "which_bquark for the parton matching must be 'first' or 'last' or 'last_numba'"
            )

        # Adding the provenance to the quark object
        bquarks = ak.with_field(bquarks, provenance, "provenance")
        self.events["bQuark"] = bquarks

        # Calling our general object_matching function.
        # The output is an awkward array with the shape of the second argument and None where there is no matching.
        # So, calling like this, we will get out an array of matched_quarks with the dimension of the JetGood.
        matched_bquarks_higgs, matched_jets_higgs, deltaR_matched_higgs = (
            object_matching(
                bquarks,
                self.events.JetGoodHiggs,
                dr_min=self.parton_jet_min_dR,
            )
        )
        # matched all jetgood
        matched_bquarks, matched_jets, deltaR_matched = object_matching(
            bquarks,
            self.events.JetGood,
            dr_min=self.parton_jet_min_dR,
        )

        # add provenance
        matched_jets_higgs = ak.with_field(
            matched_jets_higgs, matched_bquarks_higgs.provenance, "provenance"
        )
        self.events["JetGoodHiggs"] = ak.with_field(
            self.events.JetGoodHiggs, matched_bquarks_higgs.provenance, "provenance"
        )
        matched_jets = ak.with_field(
            matched_jets, matched_bquarks.provenance, "provenance"
        )
        self.events["JetGood"] = ak.with_field(
            self.events.JetGood, matched_bquarks.provenance, "provenance"
        )

        # add deltaR information
        self.events["bQuarkHiggsMatched"] = ak.with_field(
            matched_bquarks_higgs, deltaR_matched_higgs, "dRMatchedJet"
        )
        self.events["JetGoodHiggsMatched"] = ak.with_field(
            matched_jets_higgs, deltaR_matched_higgs, "dRMatchedJet"
        )
        self.events["bQuarkMatched"] = ak.with_field(
            matched_bquarks, deltaR_matched, "dRMatchedJet"
        )
        self.events["JetGoodMatched"] = ak.with_field(
            matched_jets, deltaR_matched, "dRMatchedJet"
        )

        # add pdgId information
        self.events["JetGoodHiggsMatched"] = ak.with_field(
            self.events.JetGoodHiggsMatched,
            self.events.bQuarkHiggsMatched.pdgId,
            "pdgId",
        )
        self.events["JetGoodMatched"] = ak.with_field(
            self.events.JetGoodMatched,
            self.events.bQuarkMatched.pdgId,
            "pdgId",
        )

    def do_vbf_parton_matching(self):  # -> ak.Array:
        # Select vbf quarks
        self.events.GenPart = ak.with_field(
            self.events.GenPart, ak.local_index(self.events.GenPart, axis=1), "index"
        )
        genpart = self.events.GenPart

        isQuark = abs(genpart.pdgId) < 7
        isHard = genpart.hasFlags(["fromHardProcess"])

        quarks = genpart[isQuark & isHard]
        quarks = quarks[quarks.genPartIdxMother != -1]

        quarks_mother = genpart[quarks.genPartIdxMother]
        quarks_mother_children = quarks_mother.children
        quarks_mother_children_isH = (
            ak.sum((quarks_mother_children.pdgId == 25), axis=-1) == 2
        )
        vbf_quarks = quarks[quarks_mother_children_isH]

        # define variables to get the last copy
        children_idxG = ak.without_parameters(genpart.childrenIdxG, behavior={})
        children_idxG_flat = ak.flatten(children_idxG, axis=1)
        genpart_pdgId_flat = ak.flatten(
            ak.without_parameters(genpart.pdgId, behavior={}), axis=1
        )
        genpart_LastCopy_flat = ak.flatten(
            ak.without_parameters(genpart.hasFlags(["isLastCopy"]), behavior={}), axis=1
        )
        genpart_pt_flat = ak.flatten(
            ak.without_parameters(genpart.pt, behavior={}), axis=1
        )
        genparts_flat = ak.flatten(genpart)
        genpart_offsets = np.concatenate(
            [[0], np.cumsum(ak.to_numpy(ak.num(genpart, axis=1), allow_missing=True))]
        )
        vbf_quark_idx = ak.to_numpy(
            vbf_quarks.index + genpart_offsets[:-1], allow_missing=False
        )
        vbf_quarks_pdgId = ak.to_numpy(vbf_quarks.pdgId, allow_missing=False)
        nevents = vbf_quark_idx.shape[0]
        firstgenpart_idxG = ak.firsts(genpart[:, 0].children).genPartIdxMotherG
        firstgenpart_idxG_numpy = ak.to_numpy(firstgenpart_idxG, allow_missing=False)

        # find the quark last copy
        vbf_quark_last_idx = get_parton_last_copy(
            vbf_quark_idx,
            vbf_quarks_pdgId,
            children_idxG_flat,
            genpart_pdgId_flat,
            genpart_offsets,
            genpart_LastCopy_flat,
            genpart_pt_flat,
            nevents,
            firstgenpart_idxG_numpy,
        )
        vbf_quarks_last = genparts_flat[vbf_quark_last_idx]

        # match quarks with the VBF jet candidates
        matched_vbf_quarks, matched_vbf_jets, deltaR_matched_vbf = object_matching(
            vbf_quarks_last,
            self.events.JetGoodVBF,
            dr_min=self.parton_jet_min_dR,
        )

        matched_vbf_jets = ak.with_field(
            matched_vbf_jets, ak.ones_like(matched_vbf_jets.pt) * 3, "provenance"
        )
        matched_vbf_quarks = ak.with_field(
            matched_vbf_quarks, matched_vbf_jets.provenance, "provenance"
        )

        self.events["JetGoodVBF_matched"] = matched_vbf_jets
        self.events["JetGoodVBF"] = ak.with_field(
            self.events.JetGoodVBF,
            self.events["JetGoodVBF_matched"].provenance,
            "provenance",
        )

        self.events["quarkVBF_matched"] = matched_vbf_quarks

        self.events["quarkVBF"] = ak.with_field(
            vbf_quarks_last, matched_vbf_quarks.provenance, "provenance"
        )

        # breakpoint()

    def dummy_provenance(self):
        self.events["JetGoodHiggs"] = ak.with_field(
            self.events.JetGoodHiggs,
            ak.ones_like(self.events.JetGoodHiggs.pt) * -1,
            "provenance",
        )
        self.events["JetGoodHiggsMatched"] = self.events.JetGoodHiggs

        self.events["JetGood"] = ak.with_field(
            self.events.JetGood, ak.ones_like(self.events.JetGood.pt) * -1, "provenance"
        )
        self.events["JetGoodMatched"] = self.events.JetGood

    def count_objects(self, variation):
        # NOTE:  ak.num counts even the None values, while ak.count counts only the non-None values
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood, axis=1)
        self.events["nMuonGood"] = ak.num(self.events.MuonGood, axis=1)
        self.events["nJetGood"] = ak.num(self.events.JetGood, axis=1)
        self.events["nJetGoodHiggs"] = ak.num(self.events.JetGoodHiggs, axis=1)

    def HelicityCosTheta(self, higgs, jet):
        higgs = add_fields(higgs, four_vec="Momentum4D")
        higgs_velocity = higgs.to_beta3()
        jet = add_fields(jet, four_vec="Momentum4D")
        jet = jet.boost_beta3(-higgs_velocity)
        return np.cos(jet.theta)

    def Costhetastar_CS(self, higgs1_vec, hh_vec):
        hh_vec = add_fields(hh_vec, four_vec="Momentum4D")
        hh_velocity = hh_vec.to_beta3()
        higgs1_vec = add_fields(higgs1_vec, four_vec="Momentum4D")
        higgs1_vec = higgs1_vec.boost_beta3(-hh_velocity)
        return abs(np.cos(higgs1_vec.theta))

    def get_sigma_mbb(self, jet1, jet2):
        jet1 = add_fields(jet1)
        jet2 = add_fields(jet2)

        # PNetRegPtRawRes is the resolution of the jet pt estimated by PNet
        jet1_up = jet1 * (1 + jet1.PNetRegPtRawRes)
        jet2_up = jet2 * (1 + jet2.PNetRegPtRawRes)

        jet1_down = jet1 * (1 - jet1.PNetRegPtRawRes)
        jet2_down = jet2 * (1 - jet2.PNetRegPtRawRes)

        jet1_up_sigma = ak.singletons(abs((jet1 + jet2).mass - (jet1_up + jet2).mass))
        jet1_down_sigma = ak.singletons(
            abs((jet1 + jet2).mass - (jet1_down + jet2).mass)
        )
        jet1_sigma_conc = ak.concatenate((jet1_up_sigma, jet1_down_sigma), axis=1)
        sigma_hbbCand_A = ak.max(jet1_sigma_conc, axis=1)

        jet2_up_sigma = ak.singletons(abs((jet1 + jet2).mass - (jet1 + jet2_up).mass))
        jet2_down_sigma = ak.singletons(
            abs((jet1 + jet2).mass - (jet1 + jet2_down).mass)
        )
        jet2_sigma_conc = ak.concatenate((jet2_up_sigma, jet2_down_sigma), axis=1)
        sigma_hbbCand_B = ak.max(jet2_sigma_conc, axis=1)

        return ak.flatten(np.sqrt(sigma_hbbCand_A**2 + sigma_hbbCand_B**2), axis=None)

    def get_jets_no_higgs(self, jet_higgs_idx_per_event):
        jet_offsets = np.concatenate(
            [
                [0],
                np.cumsum(
                    ak.to_numpy(ak.num(self.events.Jet, axis=1), allow_missing=True)
                ),
            ]
        )
        local_index_all = ak.local_index(self.events.Jet, axis=1)
        jets_index_all = ak.to_numpy(
            ak.flatten(local_index_all + jet_offsets[:-1]), allow_missing=True
        )
        jets_from_higgs_idx = ak.to_numpy(
            ak.flatten(jet_higgs_idx_per_event + jet_offsets[:-1]),
            allow_missing=False,
        )
        jets_no_higgs_idx = get_jets_no_higgs_from_idx(
            jets_index_all, jets_from_higgs_idx
        )
        jets_no_higgs_idx_unflat = (
            ak.unflatten(jets_no_higgs_idx, ak.num(self.events.Jet, axis=1))
            - jet_offsets[:-1]
        )
        jets_not_from_higgs = self.events.Jet[jets_no_higgs_idx_unflat >= 0]
        return jets_not_from_higgs

    def define_dnn_variables(
        self, higgs1, higgs2, jets_from_higgs, jet_higgs_idx_per_event, sb_variables
    ):
        ########################
        # ADDITIONAL VARIABLES #
        ########################

        # HT : scalar sum of all jets with pT > 25 GeV inside | η | < 2.5
        self.events["HT"] = ak.sum(self.events.JetGood.pt, axis=1)

        self.events["era"] = ak.full_like(
            self.events.HT, era_dict[f"{self._year}_{self._era}"]
        )
        self.events["year"] = ak.full_like(self.events.HT, year_dict[f"{self._year}"])

        self.events["JetNotFromHiggs"] = self.get_jets_no_higgs(jet_higgs_idx_per_event)

        self.params.object_preselection.update(
            {"JetNotFromHiggs": self.params.object_preselection["Jet"]}
        )

        # Cut on the JEC pt (w/o regression)
        self.events["JetNotFromHiggs"], _ = custom_jet_selection(
            self.events,
            "JetNotFromHiggs",
            self.params,
            year=self._year,
            pt_type="pt_default",
            pt_cut_name=self.pt_cut_name,
        )

        if self.add_jet_spanet:
            if self.fifth_jet == "pt":
                # order the self.events["JetNotFromHiggs"] according to btag or to pt
                # depending on wether the pairing chose the 5th jet or not
                self.events["JetNotFromHiggs"] = ak.where(
                    self.events["btag_order_add_jet"],
                    self.events["JetNotFromHiggs"][
                        ak.argsort(
                            self.events["JetNotFromHiggs"].btagPNetB,
                            axis=1,
                            ascending=False,
                        )
                    ],
                    self.events["JetNotFromHiggs"][
                        ak.argsort(
                            self.events["JetNotFromHiggs"].pt, axis=1, ascending=False
                        )
                    ],
                )
            else:
                self.events["JetNotFromHiggs"] = self.events["JetNotFromHiggs"][
                    ak.argsort(
                        self.events["JetNotFromHiggs"].btagPNetB,
                        axis=1,
                        ascending=False,
                    )
                ]

        add_jet1pt = ak.pad_none(self.events.JetNotFromHiggs, 1, clip=True)[:, 0]

        # Minimum ∆R ( jj ) among all possible pairings of the leading b-tagged jets
        # Maximum ∆R( jj ) among all possible pairings of the leading b-tagged jets
        _, JetGood2 = ak.unzip(
            ak.cartesian(
                [
                    self.events.JetGood[:, :4],
                    self.events.JetGood[:, :4],
                ],
                nested=True,
            )
        )
        dR = self.events.JetGood[:, :4].delta_r(JetGood2)
        # remove dR between the same jets
        dR = ak.mask(dR, dR > 0)
        # flatten the last 2 dimension of the dR array  to get an array for each event
        dR = ak.flatten(dR, axis=2)
        self.events["dR_min"] = ak.min(dR, axis=1)
        self.events["dR_max"] = ak.max(dR, axis=1)

        sigma_over_higgs1_reco_mass = (
            self.get_sigma_mbb(
                jets_from_higgs[:, 0],
                jets_from_higgs[:, 1],
            )
            / higgs1.mass
        )
        sigma_over_higgs2_reco_mass = (
            self.get_sigma_mbb(
                jets_from_higgs[:, 2],
                jets_from_higgs[:, 3],
            )
            / higgs2.mass
        )

        # Leading-pT H candidate pT , η, φ, and mass
        # Subleading-pT H candidate pT , η, φ, and mass
        # Angular separation (∆R) between b jets for each H candidate
        higgs1 = ak.with_field(
            higgs1,
            jets_from_higgs[:, 0].delta_r(jets_from_higgs[:, 1]),
            "dR",
        )
        higgs2 = ak.with_field(
            higgs2,
            jets_from_higgs[:, 2].delta_r(jets_from_higgs[:, 3]),
            "dR",
        )

        # helicity | cos θ | for each H candidate
        higgs1 = ak.with_field(
            higgs1,
            abs(self.HelicityCosTheta(higgs1, jets_from_higgs[:, 0])),
            "helicityCosTheta",
        )
        higgs2 = ak.with_field(
            higgs2,
            abs(
                self.HelicityCosTheta(
                    higgs2,
                    jets_from_higgs[:, 2],
                )
            ),
            "helicityCosTheta",
        )

        # di-Higgs system
        # pT , η, and mass of HH system
        hh = add_fields(higgs1 + higgs2)

        # | cos θ ∗ | of HH system
        hh = ak.with_field(
            hh,
            self.Costhetastar_CS(higgs1, hh),
            "Costhetastar_CS",
        )

        # Angular separation (∆R, ∆η, ∆φ) between H candidates
        hh = ak.with_field(
            hh,
            higgs1.delta_r(higgs2),
            "dR",
        )
        hh = ak.with_field(
            hh,
            abs(higgs1.eta - higgs2.eta),
            "dEta",
        )
        hh = ak.with_field(
            hh,
            higgs1.delta_phi(higgs2),
            "dPhi",
        )

        if sb_variables:
            # dPhi
            higgs1 = ak.with_field(
                higgs1,
                jets_from_higgs[:, 0].delta_phi(jets_from_higgs[:, 1]),
                "dPhi",
            )
            higgs2 = ak.with_field(
                higgs2,
                jets_from_higgs[:, 2].delta_phi(jets_from_higgs[:, 3]),
                "dPhi",
            )

            # dEta
            higgs1 = ak.with_field(
                higgs1,
                abs(jets_from_higgs[:, 0].eta - jets_from_higgs[:, 1].eta),
                "dEta",
            )
            higgs2 = ak.with_field(
                higgs2,
                abs(jets_from_higgs[:, 2].eta - jets_from_higgs[:, 3].eta),
                "dEta",
            )

            # add jet and higgs1
            add_jet1pt = ak.with_field(
                add_jet1pt,
                abs(add_jet1pt.eta - higgs1.eta),
                "LeadingHiggs_dEta",
            )
            add_jet1pt = ak.with_field(
                add_jet1pt,
                add_jet1pt.delta_phi(higgs1),
                "LeadingHiggs_dPhi",
            )
            add_jet1pt = ak.with_field(
                add_jet1pt,
                (add_jet1pt + higgs1).mass,
                "LeadingHiggs_mass",
            )

            # add jet and higgs2
            add_jet1pt = ak.with_field(
                add_jet1pt,
                abs(add_jet1pt.eta - higgs2.eta),
                "SubLeadingHiggs_dEta",
            )
            add_jet1pt = ak.with_field(
                add_jet1pt,
                add_jet1pt.delta_phi(higgs2),
                "SubLeadingHiggs_dPhi",
            )
            add_jet1pt = ak.with_field(
                add_jet1pt,
                (add_jet1pt + higgs2).mass,
                "SubLeadingHiggs_mass",
            )

        return (
            higgs1,
            higgs2,
            hh,
            add_jet1pt,
            sigma_over_higgs1_reco_mass,
            sigma_over_higgs2_reco_mass,
        )

    def get_true_pairing_and_compare(
        self, suffix="", pairing_predictions=None, pairing_suffix=""
    ):
        # do truth matching to get b-jet from Higgs
        self.get_jet_higgs_provenance(which_bquark=self.which_bquark)
        self.events["nbQuarkHiggsMatched"] = ak.num(
            self.events.bQuarkHiggsMatched, axis=1
        )
        self.events["nbQuarkMatched"] = ak.num(self.events.bQuarkMatched, axis=1)

        # reconstruct the higgs candidates
        (
            self.events[f"HiggsLeading{suffix}"],
            self.events[f"HiggsSubLeading{suffix}"],
            self.events[f"JetGoodFromHiggsOrdered{suffix}"],
            pairing_true,
        ) = reconstruct_higgs_from_provenance(self.events.JetGoodMatched)

        matched_jet_higgs_idx_not_none = self.events.JetGoodMatched.index[
            ~ak.is_none(self.events.JetGoodMatched.index, axis=1)
        ]

        # Define distance parameter for selection:
        self.events[f"Rhh{suffix}"] = np.sqrt(
            (self.events[f"HiggsLeading{suffix}"].mass - 125) ** 2
            + (self.events[f"HiggsSubLeading{suffix}"].mass - 120) ** 2
        )

        if pairing_predictions is not None:
            # Masking and calculating efficiency
            # Need to sort the double pairs in innermost dimension for easier evaluation
            pairing_predictions = ak.sort(ak.Array(pairing_predictions), axis=-1)
            if pairing_suffix == "Run2":
                # keep up to 4 jets
                pairing_true = ak.where(
                    pairing_true < 4,
                    pairing_true,
                    -1,
                )
            else:
                # keep up to 5 jets
                pairing_true = ak.where(
                    pairing_true < self.max_num_jets,
                    pairing_true,
                    -1,
                )

            mask_fully_matched = ak.all(ak.flatten(pairing_true, axis=2) >= 0, axis=1)
            pairing_true = ak.sort(pairing_true, axis=-1)

            match_0 = ak.all(
                pairing_true[:, 0] == pairing_predictions[:, 0], axis=-1
            )  # shape: (N_events, 2)
            match_1 = ak.all(
                pairing_true[:, 0] == pairing_predictions[:, 1], axis=-1
            )  # shape: (N_events, 2)
            match_2 = ak.all(
                pairing_true[:, 1] == pairing_predictions[:, 0], axis=-1
            )  # shape: (N_events, 2)
            match_3 = ak.all(
                pairing_true[:, 1] == pairing_predictions[:, 1], axis=-1
            )  # shape: (N_events, 2)
            correct_prediction = (match_0 | match_1) & (match_2 | match_3)

            self.events[f"correct_prediction{pairing_suffix}"] = correct_prediction
            self.events[f"correct_prediction_fully_matched{pairing_suffix}"] = ak.mask(
                correct_prediction, mask_fully_matched
            )
        else:
            mask_fully_matched = ak.all(ak.flatten(pairing_true, axis=2) >= 0, axis=1)

        self.events["mask_fully_matched"] = mask_fully_matched

        return matched_jet_higgs_idx_not_none

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        # Define the Delta WP
        self.events["JetGood"] = self.generate_btag_delta_workingpoints(
            self.events["JetGood"], 5
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
                self.max_num_jets,
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

            # apply logit transformation
            # self.events["Logit_Delta_pairing_probabilities"] = np.log(
            #     self.events["Delta_pairing_probabilities"]
            #     / (1 - self.events["Delta_pairing_probabilities"])
            # )

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

        # reconstruct the higgs candidates for Run2 method
        if self.run2:
            (
                pairing_predictions,
                self.events["delta_dhh"],
                self.events["HiggsLeadingRun2"],
                self.events["HiggsSubLeadingRun2"],
                self.events["JetGoodFromHiggsOrderedRun2"],
            ) = run2_matching_algorithm(self.events["JetGoodHiggs"])

            matched_jet_higgs_idx_not_noneRun2 = (
                self.events.JetGoodFromHiggsOrderedRun2.index
            )
            self.events["Rhh_Run2"] = np.sqrt(
                (self.events.HiggsLeadingRun2.mass - 125) ** 2
                + (self.events.HiggsSubLeadingRun2.mass - 120) ** 2
            )
            if self._isMC:
                matched_jet_higgs_idx_not_noneTrue = self.get_true_pairing_and_compare(
                    suffix="True",
                    pairing_predictions=pairing_predictions,
                    pairing_suffix="Run2",
                )

            # if the 5th jet is matched, then the add jet should be order by btag
            # because we want to consider the leading in btag which the pairing discarded
            # (useless for Run2 pairing because it's always 4 jets)
            self.events["btag_order_add_jet"] = ak.any(
                ak.flatten(pairing_predictions, axis=-1) > 3, axis=-1
            )

        if not (self._isMC and not self.spanet):
            self.dummy_provenance()

        self.events["nJetGoodHiggsMatched"] = ak.num(
            self.events.JetGoodHiggsMatched, axis=1
        )
        self.events["nJetGoodMatched"] = ak.num(self.events.JetGoodMatched, axis=1)

        if self.vbf_ggf_dnn:
            (
                model_session_vbf_ggf_dnn,
                input_name_vbf_ggf_dnn,
                output_name_vbf_ggf_dnn,
            ) = get_model_session(self.vbf_ggf_dnn, "vbf_ggf_dnn")
            del model_session_vbf_ggf_dnn
            del input_name_vbf_ggf_dnn
            del output_name_vbf_ggf_dnn

        if self.dnn_variables and self.spanet:
            (
                self.events["HiggsLeading"],
                self.events["HiggsSubLeading"],
                self.events["HH"],
                self.events["add_jet1pt"],
                self.events["sigma_over_higgs1_reco_mass"],
                self.events["sigma_over_higgs2_reco_mass"],
            ) = self.define_dnn_variables(
                self.events.HiggsLeading,
                self.events.HiggsSubLeading,
                self.events.JetGoodFromHiggsOrdered,
                matched_jet_higgs_idx_not_none,
                sb_variables=True,  # if self.SIG_BKG_DNN else False,
            )
        if self.dnn_variables and self.run2:
            (
                self.events["HiggsLeadingRun2"],
                self.events["HiggsSubLeadingRun2"],
                self.events["HHRun2"],
                self.events["add_jet1ptRun2"],
                self.events["sigma_over_higgs1_reco_massRun2"],
                self.events["sigma_over_higgs2_reco_massRun2"],
            ) = self.define_dnn_variables(
                self.events.HiggsLeadingRun2,
                self.events.HiggsSubLeadingRun2,
                self.events.JetGoodFromHiggsOrderedRun2,
                matched_jet_higgs_idx_not_noneRun2,
                sb_variables=True,  # if self.sig_bkg_dnn else False,
            )

        if self.bkg_morphing_dnn and not self._isMC:
            (
                model_session_bkg_morphing_dnn,
                input_name_bkg_morphing_dnn,
                output_name_bkg_morphing_dnn,
            ) = get_model_session(self.bkg_morphing_dnn, "bkg_morphing_dnn")

            if self.spanet:
                self.events["bkg_morphing_dnn_weight"] = ak.flatten(
                    get_dnn_prediction(
                        model_session_bkg_morphing_dnn,
                        input_name_bkg_morphing_dnn,
                        output_name_bkg_morphing_dnn,
                        self.events,
                        self.bkg_morphing_dnn_input_variables,
                        pad_value=self.pad_value,
                    )[0],
                    axis=None,
                )

            if self.run2:
                self.events["bkg_morphing_dnn_weightRun2"] = ak.flatten(
                    get_dnn_prediction(
                        model_session_bkg_morphing_dnn,
                        input_name_bkg_morphing_dnn,
                        output_name_bkg_morphing_dnn,
                        self.events,
                        self.bkg_morphing_dnn_input_variables,
                        pad_value=self.pad_value,
                        run2=True,
                    )[0],
                    axis=None,
                )
            del model_session_bkg_morphing_dnn
            del input_name_bkg_morphing_dnn
            del output_name_bkg_morphing_dnn

        if self.bkg_morphing_spread_dnn and not self._isMC:
            (
                model_session_bkg_morphing_spread_dnn,
                input_name_bkg_morphing_spread_dnn,
                output_name_bkg_morphing_spread_dnn,
            ) = get_model_session(
                self.bkg_morphing_spread_dnn, "bkg_morphing_spread_dnn"
            )

            if self.spanet:
                self.events["bkg_morphing_spread_dnn_weights"] = np.transpose(
                    get_dnn_prediction(
                        model_session_bkg_morphing_spread_dnn,
                        input_name_bkg_morphing_spread_dnn,
                        output_name_bkg_morphing_spread_dnn,
                        self.events,
                        self.bkg_morphing_dnn_input_variables,
                        pad_value=self.pad_value,
                    )
                )

            if self.run2:
                self.events["bkg_morphing_spread_dnn_weightsRun2"] = np.transpose(
                    get_dnn_prediction(
                        model_session_bkg_morphing_spread_dnn,
                        input_name_bkg_morphing_spread_dnn,
                        output_name_bkg_morphing_spread_dnn,
                        self.events,
                        self.bkg_morphing_dnn_input_variables,
                        pad_value=self.pad_value,
                        run2=True,
                    )
                )
            del model_session_bkg_morphing_spread_dnn
            del input_name_bkg_morphing_spread_dnn
            del output_name_bkg_morphing_spread_dnn

        if self.sig_bkg_dnn:
            (
                model_session_SIG_BKG_DNN,
                input_name_SIG_BKG_DNN,
                output_name_SIG_BKG_DNN,
            ) = get_model_session(self.sig_bkg_dnn, "sig_bkg_dnn")

            if self.spanet:
                sig_bkg_dnn_score = get_dnn_prediction(
                    model_session_SIG_BKG_DNN,
                    input_name_SIG_BKG_DNN,
                    output_name_SIG_BKG_DNN,
                    self.events,
                    self.sig_bkg_dnn_input_variables,
                    pad_value=self.pad_value,
                )[0]
                # if array is 1 dim just take it
                if sig_bkg_dnn_score.ndim == 1:
                    self.events["sig_bkg_dnn_score"] = sig_bkg_dnn_score
                else:
                    # if array is 2 dim take the last column
                    self.events["sig_bkg_dnn_score"] = sig_bkg_dnn_score[:, -1]

            if self.run2:
                sig_bkg_dnn_score = get_dnn_prediction(
                    model_session_SIG_BKG_DNN,
                    input_name_SIG_BKG_DNN,
                    output_name_SIG_BKG_DNN,
                    self.events,
                    self.sig_bkg_dnn_input_variables,
                    pad_value=self.pad_value,
                    run2=True,
                )[0]
                # if array is 1 dim just take it
                if sig_bkg_dnn_score.ndim == 1:
                    self.events["sig_bkg_dnn_scoreRun2"] = sig_bkg_dnn_score
                else:
                    # if array is 2 dim take the last column
                    self.events["sig_bkg_dnn_scoreRun2"] = sig_bkg_dnn_score[:, -1]

                del model_session_SIG_BKG_DNN
                del input_name_SIG_BKG_DNN
                del output_name_SIG_BKG_DNN
