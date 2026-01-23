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
        if self.vbf_analysis:
                
            jet_goodhiggs_idx_not_none = self.events.JetGoodHiggs.index
            
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
            self.events["JetGoodVBF"]=forward_jet_veto(self.events, "JetGoodVBF", pt_type="pt_default")
            
    def count_objects(self, variation):
        super().count_objects(variation=variation)
        if self.vbf_analysis:
            self.events["nJetGoodVBF"] = ak.num(self.events.JetGoodVBF, axis=1)

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        super().process_extra_after_presel(variation=variation)
        if self.vbf_analysis:            

            if self._isMC and self.vbf_parton_matching:
                self.do_vbf_parton_matching()

                self.events["nJetGoodVBF_matched"] = ak.num(
                    self.events.JetGoodVBF_matched, axis=1
                )

                # Create new variable delta eta and invariant mass of the jets
                JetGoodVBF_matched_padded = ak.pad_none(
                    self.events.JetGoodVBF_matched, 2
                )  # Adds none jets to events that have less than 2 jets

                self.events["deltaEta_matched"] = abs(
                    JetGoodVBF_matched_padded.eta[:, 0] - JetGoodVBF_matched_padded.eta[:, 1]
                )

                self.events["jj_mass_matched"] = (
                    JetGoodVBF_matched_padded[:, 0] + JetGoodVBF_matched_padded[:, 1]
                ).mass

                # This product will give only -1 or 1 values, as it's needed to see if the two jets are in the same side or not
                self.events["etaProduct"] = (
                    JetGoodVBF_matched_padded.eta[:, 0] * JetGoodVBF_matched_padded.eta[:, 1]
                ) / abs(
                    JetGoodVBF_matched_padded.eta[:, 0] * JetGoodVBF_matched_padded.eta[:, 1]
                )

            # choose vbf jets as the two jets with the highest pt that are not from higgs decay
            self.events["JetVBFLeadingPtNotFromHiggs"] = self.events.JetGoodVBF[
                :, :2
            ]

            # choose higgs jets as the two jets with the highest mjj that are not from higgs decay
            jet_combinations = ak.combinations(self.events.JetGoodVBF, 2)
            jet_combinations_mass = (jet_combinations["0"] + jet_combinations["1"]).mass
            jet_combinations_mass_max_idx = ak.to_numpy(
                ak.argsort(jet_combinations_mass, axis=1, ascending=False)[:, 0]
            )

            jets_max_mass = jet_combinations[
                ak.local_index(jet_combinations, axis=0), jet_combinations_mass_max_idx
            ]
            vbf_jets_max_mass_0 = ak.unflatten(
                self.events.Jet[
                    ak.local_index(self.events.Jet, axis=0),
                    ak.to_numpy(jets_max_mass["0"].index),
                ],
                1,
            )
            vbf_jets_max_mass_1 = ak.unflatten(
                self.events.Jet[
                    ak.local_index(self.events.Jet, axis=0),
                    ak.to_numpy(jets_max_mass["1"].index),
                ],
                1,
            )

            vbf_jet_leading_mjj = ak.with_name(
                ak.concatenate([vbf_jets_max_mass_0, vbf_jets_max_mass_1], axis=1),
                name="PtEtaPhiMCandidate",
            )

            vbf_jet_leading_mjj_fields_dict = {
                field: getattr(vbf_jet_leading_mjj, field)
                for field in vbf_jet_leading_mjj.fields
                if ("muon" not in field and "electron" not in field)
            }
            self.events["JetVBFLeadingMjjNotFromHiggs"] = add_fields(
                vbf_jet_leading_mjj, vbf_jet_leading_mjj_fields_dict
            )

            self.events["JetVBFLeadingPtNotFromHiggs_deltaEta"] = abs(
                self.events.JetVBFLeadingPtNotFromHiggs.eta[:, 0]
                - self.events.JetVBFLeadingPtNotFromHiggs.eta[:, 1]
            )

            self.events["JetVBFLeadingMjjNotFromHiggs_deltaEta"] = abs(
                self.events.JetVBFLeadingMjjNotFromHiggs.eta[:, 0]
                - self.events.JetVBFLeadingMjjNotFromHiggs.eta[:, 1]
            )

            self.events["JetVBFLeadingPtNotFromHiggs_jjMass"] = (
                self.events.JetVBFLeadingPtNotFromHiggs[:, 0]
                + self.events.JetVBFLeadingPtNotFromHiggs[:, 1]
            ).mass

            self.events["JetVBFLeadingMjjNotFromHiggs_jjMass"] = (
                self.events.JetVBFLeadingMjjNotFromHiggs[:, 0]
                + self.events.JetVBFLeadingMjjNotFromHiggs[:, 1]
            ).mass

            self.events["HH_deltaR"] = self.events.HiggsLeading.delta_r(
                self.events.HiggsSubLeading
            )

            self.events["jj_deltaR"] = self.events.JetVBFLeadingPtNotFromHiggs[
                :, 0
            ].delta_r(self.events.JetVBFLeadingPtNotFromHiggs[:, 1])

            self.events["H1j1_deltaR"] = self.events.HiggsLeading.delta_r(
                self.events.JetVBFLeadingPtNotFromHiggs[:, 0]
            )

            self.events["H1j2_deltaR"] = self.events.HiggsLeading.delta_r(
                self.events.JetVBFLeadingPtNotFromHiggs[:, 1]
            )

            self.events["H2j1_deltaR"] = self.events.HiggsSubLeading.delta_r(
                self.events.JetVBFLeadingPtNotFromHiggs[:, 0]
            )

            self.events["H2j2_deltaR"] = self.events.HiggsSubLeading.delta_r(
                self.events.JetVBFLeadingPtNotFromHiggs[:, 1]
            )

            JetVBFLeadingPtNotFromHiggs_etaAverage = (
                self.events.JetVBFLeadingPtNotFromHiggs.eta[:, 0]
                + self.events.JetVBFLeadingPtNotFromHiggs.eta[:, 1]
            ) / 2

            self.events["HH_centrality"] = np.exp(
                (
                    -(
                        (
                            self.events.HiggsLeading.eta
                            - JetVBFLeadingPtNotFromHiggs_etaAverage
                        )
                        ** 2
                    )
                    - (
                        self.events.HiggsSubLeading.eta
                        - JetVBFLeadingPtNotFromHiggs_etaAverage
                    )
                    ** 2
                )
                / (self.events.JetVBFLeadingPtNotFromHiggs_deltaEta) ** 2
            )
