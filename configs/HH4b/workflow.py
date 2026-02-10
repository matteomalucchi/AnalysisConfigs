import awkward as ak
import sys
import numpy as np

from configs.HH4b_common.workflow_common import HH4bCommonProcessor

class HH4bbQuarkMatchingProcessor(HH4bCommonProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)

    def process_extra_after_skim(self):
        super().process_extra_after_skim()
        self.def_provenance_field()
        self.define_jet_collections()
        
        
    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation)

        # create the provenance field
        for jet_coll in ["JetGood", "JetGoodHiggs", "JetGoodMatched", "JetGoodHiggsMatched"]:
            self.events[jet_coll] = ak.with_field(
                self.events[jet_coll],
                self.events[jet_coll].provenance_higgs,
                "provenance",
            )
            
    def process_extra_after_presel(self, variation):  # -> ak.Array
        self.flatten_pt(variation)
        super().process_extra_after_presel(variation)
        
            
        
