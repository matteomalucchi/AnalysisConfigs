import awkward as ak

from pocket_coffea.workflows.base import BaseProcessorABC


class HH4bbQuarkMatchingProcessor(BaseProcessorABC):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.dr_min = self.workflow_options["parton_jet_min_dR"]
        self.max_num_jets = self.workflow_options["max_num_jets"]
        self.which_bquark = self.workflow_options["which_bquark"]
        self.fifth_jet = self.workflow_options["fifth_jet"]
        self.tight_cuts = self.workflow_options["tight_cuts"]
        self.classification = self.workflow_options["classification"]
        self.spanet_model = self.workflow_options["spanet_model"]
        self.random_pt = self.workflow_options["random_pt"]

    def apply_object_preselection(self, variation):
        variation = variation


    def get_jet_higgs_provenance(self, which_bquark):  # -> ak.Array:
        which_bquark = which_bquark

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
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood, axis=1)
        self.events["nMuonGood"] = ak.num(self.events.MuonGood, axis=1)
        self.events["nJetGood"] = ak.num(self.events.JetGood, axis=1)
        self.events["nJetGoodHiggs"] = ak.num(self.events.JetGoodHiggs, axis=1)

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        variation = variation
