import awkward as ak
import copy
import numpy as np

from utils.custom_cut_functions import custom_jet_selection
from utils.basic_functions import add_fields
from configs.HH4b_common.workflow_common import HH4bCommonProcessor
from configs.HH4b_common.custom_object_preselection_common import forward_jet_veto


class VBFHH4bProcessor(HH4bCommonProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.vbf_parton_matching = self.workflow_options["vbf_parton_matching"]
        self.vbf_analysis = self.workflow_options["vbf_analysis"]

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation=variation)
    
    def get_vbf_jet_candidates(self):
        if self.vbf_analysis:

            self.events["JetGoodClip"] = copy.copy(
                self.events.JetGood[:, : self.max_num_jets]
            )
            jet_goodhiggs_idx_not_none = self.events.JetGoodClip.index

            self.events["JetVBF"] = self.get_jets_no_higgs(jet_goodhiggs_idx_not_none)
            self.events["JetGoodVBF"], _ = custom_jet_selection(
                self.events,
                "JetVBF",
                self.params,
                year=self._year,
                pt_type="pt_default",
                pt_cut_name=self.pt_cut_name,
            )
            # order in pt
            self.events["JetGoodVBF"] = self.events.JetGoodVBF[
                ak.argsort(self.events.JetGoodVBF.pt, axis=1, ascending=False)
            ]

            # apply forward jet veto
            self.events["JetGoodVBF"] = forward_jet_veto(
                self.events, "JetGoodVBF", pt_type="pt_default"
            )

    # def count_objects(self, variation):
    #     super().count_objects(variation=variation)
    #     if self.vbf_analysis:
    #         self.events["nJetGoodVBF"] = ak.num(self.events.JetGoodVBF, axis=1)

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        super().process_extra_after_presel(variation=variation)
        if self.vbf_analysis:
            self.get_vbf_jet_candidates()
            
            if self._isMC and self.vbf_parton_matching:
                self.do_vbf_parton_matching()

                self.events["nJetGoodVBF_matched"] = ak.num(
                    self.events.JetGoodVBF_matched, axis=1
                )

            # choose vbf jets as the two jets with the highest pt that are not from higgs decay
            self.events["JetVBFLeadingPtNotFromHiggs"] = self.events.JetGoodVBF[:, :2]

            # Adds none jets to events that have less than 2 jets
            self.events["JetGoodVBF"] = ak.pad_none(self.events.JetGoodVBF, 2)

            # choose vbf jet candidates as the ones with the highest mjj that are not from higgs decay
            jet_combinations = ak.combinations(self.events.JetGoodVBF, 2)
            jet_combinations_mass = (jet_combinations["0"] + jet_combinations["1"]).mass
            jet_combinations_mass_max_idx = ak.to_numpy(
                ak.argsort(jet_combinations_mass, axis=1, ascending=False)[:, 0]
            )
            jets_max_mass = jet_combinations[
                ak.local_index(jet_combinations, axis=0), jet_combinations_mass_max_idx
            ]

            mask_is_not_none = ~ak.is_none(jets_max_mass["0"])
            vbf_good_jets_max_mass_0 = ak.unflatten(
                self.events.Jet[
                    ak.local_index(self.events.Jet, axis=0),
                    ak.fill_none(jets_max_mass["0"].index, -1),
                ],
                1,
            )
            vbf_good_jets_max_mass_1 = ak.unflatten(
                self.events.Jet[
                    ak.local_index(self.events.Jet, axis=0),
                    ak.fill_none(jets_max_mass["1"].index, -1),
                ],
                1,
            )

            vbf_good_jet_leading_mjj = ak.with_name(
                ak.concatenate(
                    [vbf_good_jets_max_mass_0, vbf_good_jets_max_mass_1], axis=1
                ),
                name="PtEtaPhiMCandidate",
            )

            vbf_good_jet_leading_mjj_fields_dict = {
                field: getattr(vbf_good_jet_leading_mjj, field)
                for field in vbf_good_jet_leading_mjj.fields
                if ("muon" not in field and "electron" not in field)
            }
            self.events["JetGoodVBFLeadingMjj"] = add_fields(
                vbf_good_jet_leading_mjj, vbf_good_jet_leading_mjj_fields_dict
            )

            # get additional VBF jets
            jet_vbf_leading_mjj_idx_not_none = self.events["JetGoodVBFLeadingMjj"].index
            jet_good_vbf_leading_mjj_idx_not_none = ak.concatenate(
                [
                    self.events.JetGoodClip.index,
                    jet_vbf_leading_mjj_idx_not_none,
                ],
                axis=1,
            )

            self.events["JetAdditionalVBF"] = self.get_jets_no_higgs(
                jet_good_vbf_leading_mjj_idx_not_none
            )
            self.params.object_preselection.update(
                {"JetAdditionalVBF": self.params.object_preselection["JetVBF"]}
            )

            self.events["JetAdditionalGoodVBF"], _ = custom_jet_selection(
                self.events,
                "JetAdditionalVBF",
                self.params,
                year=self._year,
                pt_type="pt_default",
                pt_cut_name=self.pt_cut_name,
            )

            # order in energy
            self.events["JetAdditionalGoodVBF"] = self.events["JetAdditionalGoodVBF"][
                ak.argsort(self.events.JetAdditionalGoodVBF.energy, axis=1, ascending=False)
            ][:, : self.max_num_add_vbf_jets]

            jet_good_padded = ak.pad_none(self.events.JetGoodClip, self.max_num_jets)
            jet_good_vbf_masked = ak.mask(self.events.JetGoodVBFLeadingMjj, mask_is_not_none)
            jet_add_good_vbf_padded = ak.pad_none(
                self.events.JetAdditionalGoodVBF, self.max_num_add_vbf_jets
            )

            # merge the 3 jet collection to feed to spanet training
            self.events["JetGoodTotalSPANet"] = ak.concatenate(
                [
                    jet_good_padded,
                    jet_good_vbf_masked,
                    jet_add_good_vbf_padded,
                ],
                axis=1,
            )

        self.flatten_pt(variation)
