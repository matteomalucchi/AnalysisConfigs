import copy

import awkward as ak
import numpy as np
from pocket_coffea.lib.correction_cache import load_correction_set
from omegaconf import OmegaConf
from pocket_coffea.lib.jets import (
    abs_eta_in_regions,
    compute_jetId,
    get_forward_jets_mitigation,
    get_jer_correction_set,
    get_jersmear_SFunc,
    safe_jersmear,
)

from configs.VBF_HH4b.workflow import VBFHH4bProcessor
from utils_configs.custom_cut_functions import get_custom_JetVetoMap_Mask


class SyncHH4bProcessor(VBFHH4bProcessor):
    """VBF HH4b workflow of the synchronization (HH4b_synchronization.py), which
    also builds the `JetCuts` collection used by the preselection and computes
    the variables of the synchronization tree.

    With the `jet_veto_map_regressed_pt` option the jet veto map variables use
    the pt of the Jet collection (the regressed pt where the regression is
    applied) instead of the standard JEC pt (`pt_default`)."""

    @property
    def jet_veto_map_pt(self):
        """Jet pt field on which the jet veto map is evaluated."""
        return "pt" if getattr(self, "jet_veto_map_regressed_pt", False) else "pt_default"

    def apply_object_preselection(self, variation):
        if self._isMC:
            # JER of another period (`jets_calibration.jer_override`), before the
            # regressed and standard jets are merged into Jet
            self.apply_jer_override()
        super().apply_object_preselection(variation=variation)
        # jets of the preselection (`JetCuts` object preselection)
        self.select_jets("JetCuts", "JetCuts")

    def apply_jer_override(self):
        """Replace the JER smearing of the calibrated jet collections with the
        JER of `jets_calibration.jer_override[year][jet_type]` (e.g. 2023
        postBPix for 2024), keeping the JEC of the year.

        For each jet type with an override, the collection it is calibrated into
        (`jets_calibration.collection`) is smeared again from its JEC-corrected pt
        (`pt_raw * jec_factor`), with the same gen matching (dR < R/2 in
        `pt_gen`, |pt - pt_gen| < 3 sigma), random seed inputs and
        `jer_genmatched_only` forward jets mitigation as the jet calibration.
        Only the nominal JER is replaced.
        """
        overrides = OmegaConf.select(
            self.params, f"jets_calibration.jer_override.{self._year}"
        )
        if not overrides:
            return
        collections = self.params.jets_calibration.collection[self._year]
        mitigation_cfg = OmegaConf.select(
            self.params, "jets_calibration.forward_jets_mitigation"
        )
        for jet_type, jer_cfg in overrides.items():
            coll = collections.get(jet_type)
            if coll is None or coll not in self.events.fields:
                continue
            self.events[coll] = self.smear_jets_with_jer(
                self.events[coll], jet_type, jer_cfg, mitigation_cfg
            )

    def smear_jets_with_jer(self, jets, jet_type, jer_cfg, mitigation_cfg):
        """Smear `jets` (calibrated with JEC, with `pt_raw`, `mass_raw`,
        `jec_factor`, `pt_gen`, `event_rho`) with the JER of `jer_cfg`."""
        counts = ak.num(jets)
        flat = ak.flatten(jets)
        pt_jec = flat.pt_raw * flat.jec_factor
        mass_jec = flat.mass_raw * flat.jec_factor

        jer_ptres_tag = f"{jer_cfg.jer}_PtResolution_{jer_cfg.jer_jet_type}"
        jer_sf_tag = f"{jer_cfg.jer}_ScaleFactor_{jer_cfg.jer_jet_type}"
        ceval_jer = get_jer_correction_set(
            jer_cfg.json_path, (jer_ptres_tag, jer_sf_tag)
        )
        missing = [
            tag
            for tag in (jer_ptres_tag, jer_sf_tag)
            if tag not in list(ceval_jer.keys())
        ]
        if missing:
            available = sorted(
                name
                for name in load_correction_set(jer_cfg.json_path).keys()
                if "PtResolution" in name or "ScaleFactor" in name
            )
            raise ValueError(
                f"jer_override of {jet_type} ({self._year}): the JER corrections "
                f"{missing} are not in {jer_cfg.json_path}. Set `jer` and "
                f"`jer_jet_type` in jets_calibration.jer_override so that "
                f"'<jer>_PtResolution_<jer_jet_type>' and "
                f"'<jer>_ScaleFactor_<jer_jet_type>' are among the JER "
                f"corrections of the file: {available}"
            )
        eval_dict = {
            "JetPt": pt_jec,
            "JetEta": flat.eta,
            "Rho": flat.event_rho,
            "GenPt": flat.pt_gen,
            "EventID": ak.flatten(ak.broadcast_arrays(self.events.event, jets.pt)[0]),
        }
        eval_dict["JER"] = ceval_jer[jer_ptres_tag].evaluate(
            *[eval_dict[inp.name] for inp in ceval_jer[jer_ptres_tag].inputs]
        )
        # the gen match is used (scaling method) only if |pt - pt_gen| < 3 sigma
        eval_dict["GenPt"] = np.where(
            np.abs(eval_dict["JetPt"] - eval_dict["GenPt"])
            < 3 * eval_dict["JetPt"] * eval_dict["JER"],
            eval_dict["GenPt"],
            -1.0,
        )
        smear = safe_jersmear(get_jersmear_SFunc(eval_dict, ceval_jer, jer_sf_tag))

        # forward jets mitigation: no stochastic smearing in the regions
        genmatched_only = get_forward_jets_mitigation(
            mitigation_cfg, "jer_genmatched_only", self._year, jet_type
        )
        if genmatched_only is not None:
            no_smear = abs_eta_in_regions(
                flat.eta, genmatched_only["eta_regions"][str(self._year)]
            ) & (eval_dict["GenPt"] < 0)
            smear = ak.where(no_smear, 1.0, smear)

        new_fields = {
            "pt": pt_jec * smear,
            "mass": mass_jec * smear,
            "jer_factor": smear,
        }
        new_fields["pt_jer"] = new_fields["pt"]
        new_fields["mass_jer"] = new_fields["mass"]
        new_fields["rawFactor"] = ak.where(
            new_fields["pt"] != 0, 1 - flat.pt_raw / new_fields["pt"], 0
        )
        for field, values in new_fields.items():
            flat = ak.with_field(flat, values, field)
        return ak.unflatten(flat, counts)

    def process_extra_after_presel(self, variation):
        super().process_extra_after_presel(variation=variation)
        self.define_synchronization_variables()

    def jets_pass_jet_veto_map(self, jets):
        """Per-jet version of the jet veto map event cut
        (`get_JetVetoMap_Mask`, on the `jet_veto_map_pt` pt): False for the
        jets that make the event fail the veto, i.e. the jets with tight lepton
        veto ID, pt > 15 GeV, EM fraction < 0.9 inside a vetoed region."""
        # the jet ID is computed as AK4PFPuppi whatever the collection name
        params = copy.copy(self.params)
        params.jets_calibration.collection[self._year] = {"AK4PFPuppi": "Jet"}
        jet_id = compute_jetId(self.events, "Jet", params, self._year)
        del params
        considered = (
            (jet_id >= 6)
            & (abs(jets.eta) < 5.19)
            & (jets[self.jet_veto_map_pt] > 15.0)
            & ((jets.neEmEF + jets.chEmEF) < 0.9)
        )
        vetomap = self.params.jet_scale_factors.vetomaps[self._year]
        corr = load_correction_set(vetomap["file"])[vetomap["name"]]
        flat = ak.flatten(jets)
        in_vetoed_region = ak.unflatten(
            corr.evaluate(
                "jetvetomap",
                ak.to_numpy(flat.eta),
                np.clip(ak.to_numpy(flat.phi), -3.14159, 3.14159),
            )
            > 0,
            ak.num(jets),
        )
        return ~(considered & in_vetoed_region)

    def define_synchronization_variables(self):
        """Variables of the synchronization tree: `JetSync` (`JetSync` object
        preselection), the standard-JEC raw pt/mass and JEC/JER factors of the
        jets (`*_default` fields), the per-jet jet veto map flag
        (`passes_jetvetomap`), whether the jet is one of the 4 jets given to the
        D_HH pairing (`used_in_dHH`), `passJetVetoMap` and the indices (the
        `index` field of the jets, -1 if missing) of the 4 jets of the HH
        pairing (`ak4_h{1,2}b{1,2}_index`)."""
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
        # per-jet jet veto map flag
        self.events["Jet"] = ak.with_field(
            self.events.Jet,
            self.jets_pass_jet_veto_map(self.events.Jet),
            "passes_jetvetomap",
        )
        # jets given to the D_HH pairing (the 4 JetGood leading in b-tag score)
        pairs = ak.cartesian(
            [self.events.Jet.index, self.events.JetGoodHiggs.index], nested=True
        )
        self.events["Jet"] = ak.with_field(
            self.events.Jet,
            ak.fill_none(ak.any(pairs["0"] == pairs["1"], axis=2), False),
            "used_in_dHH",
        )
        # saved jet collections, ordered by pt and with the fields above:
        # `JetSync` (`JetSync` object preselection), `JetSyncNoPt` (the same
        # without pt threshold) and `JetCuts` (the jets of
        # the preselection, without pt threshold)
        self.select_jets("JetSync", "JetSync", order_by="pt")
        self.select_jets("JetSyncNoPt", "JetSyncNoPt", order_by="pt")
        self.select_jets("JetCuts", "JetCuts", order_by="pt")
        # jet veto maps
        self.events["passJetVetoMap"] = get_custom_JetVetoMap_Mask(
            self.events,
            {"jet_type": "Jet", "pt_type": self.jet_veto_map_pt},
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
