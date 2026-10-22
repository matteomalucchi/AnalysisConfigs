import awkward as ak

from configs.VBF_HH4b.workflow import VBFHH4bProcessor
from utils_configs.custom_cut_functions import get_custom_JetVetoMap_Mask


class SyncHH4bProcessor(VBFHH4bProcessor):
    """VBF HH4b workflow of the synchronization (HH4b_synchronization.py), which
    also computes the variables of the synchronization tree."""

    def process_extra_after_presel(self, variation):
        super().process_extra_after_presel(variation=variation)
        self.define_synchronization_variables()

    def define_synchronization_variables(self):
        """Variables of the synchronization tree: `JetSync` (`JetSync` object
        preselection), the standard-JEC raw pt/mass and JEC/JER factors of the
        jets (`*_default` fields), `passJetVetoMap` and the indices (the `index`
        field of the jets, -1 if missing) of the 4 jets of the HH pairing
        (`ak4_h{1,2}b{1,2}_index`)."""
        if self.add_jet_spanet:
            raise ValueError(
                "synchronization needs the Jet collection in the JetDefault order: "
                "it cannot be combined with add_jet_spanet"
            )
        # raw pt/mass and JEC/JER factors of the standard JEC (the regressed jets
        # carry the ones of the regression)
        for field in ("pt_raw", "mass_raw", "jec_factor", "jer_factor"):
            if field in self.events.JetDefault.fields:
                self.events["Jet"] = ak.with_field(
                    self.events.Jet, self.events.JetDefault[field], f"{field}_default"
                )
        # AK4 jets of the `JetSync` object preselection
        self.select_jets("JetSync", "JetSync")
        # jet veto maps
        self.events["passJetVetoMap"] = get_custom_JetVetoMap_Mask(
            self.events,
            {"jet_type": "Jet", "pt_type": "pt_default"},
            self._year,
            self.params,
            sample=self._sample,
            isMC=self._isMC,
        )
        # indices (`index` field of the jets) of the 4 jets of the HH pairing
        if "JetGoodFromHiggsOrdered" in self.events.fields:
            higgs_jet_idx = ak.pad_none(
                self.events.JetGoodFromHiggsOrdered.index, 4, clip=True
            )
        else:
            higgs_jet_idx = ak.Array([[None] * 4] * len(self.events))
        for i, name in enumerate(("h1b1", "h1b2", "h2b1", "h2b2")):
            self.events[f"ak4_{name}_index"] = ak.fill_none(higgs_jet_idx[:, i], -1)
