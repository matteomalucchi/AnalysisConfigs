import awkward as ak
import sys
import numpy as np

from configs.HH4b_common.workflow_common import HH4bCommonProcessor

class HH4bbQuarkMatchingProcessor(HH4bCommonProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation)

    def process_extra_after_presel(self, variation):  # -> ak.Array
        self.flatten_pt(variation)
        super().process_extra_after_presel(variation)
