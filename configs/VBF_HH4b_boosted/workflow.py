import awkward as ak
import pandas as pd
import copy
import json
import numpy as np
import xgboost as xgb

from utils_configs.custom_cut_functions import custom_jet_selection
from configs.HH4b_common.workflow_common import HH4bCommonProcessor
from utils_configs.reconstruct_resonances import get_lead_mjj_jet_pair
from utils_configs.bdt_evaluation_functions import get_default_bdt_inputs, evaluate_bdt, disc_TXbb

from configs.HH4b_common.custom_object_preselection_common import clean_ak4_boosted
from configs.HH4b_common.custom_cuts_common import boosted_mass_window


class VBFHH4bProcessor(HH4bCommonProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        # TXbb cuts in the low-TXbb regions matching the TXbb working points of the high-TXbb regions (quantile map)
        self.txbb_cdf_wp_cuts = self._load_txbb_cdf_wp_cuts(getattr(self, "txbb_cdf_wp_cuts", ""))
        # The downstream cuts and the DNN input read this column as the index of the working points (offset 0)
        if self.txbb_cdf_wp_cuts and self.old_wp_def:
            raise ValueError("txbb_cdf_wp_cuts cannot be used with old_wp_def=True: the cuts on the working-point index assume the offset 0")
        if not self.txbb_cdf_wp_cuts:
            for option in ("bkg_morphing_dnn_input_variables", "sig_bkg_dnn_input_variables"):
                if any(v[1] == "btagBBTXbb_CDF_rescaled_WPs" for v in (getattr(self, option, None) or {}).values()):
                    raise ValueError(f"{option} uses btagBBTXbb_CDF_rescaled_WPs, set the option txbb_cdf_wp_cuts to define it")
        # Same idea for the leading Higgs (option txbb_cdf_wp_cuts_leading), conditional on the subleading working-point index. The leading
        # TXbb in A/B overlaps the range of C/D, so it is applied by region (decided by the subleading TXbb), see `define_cdf_rescaled_leading_wps`.
        self.txbb_cdf_wp_cuts_leading = self._load_txbb_cdf_wp_cuts_leading(getattr(self, "txbb_cdf_wp_cuts_leading", ""), self.txbb_cdf_wp_cuts)
        if self.txbb_cdf_wp_cuts_leading:
            if not self.txbb_cdf_wp_cuts:
                raise ValueError("txbb_cdf_wp_cuts_leading needs txbb_cdf_wp_cuts (the region of an event is decided by the subleading TXbb)")
            if self.old_wp_def:
                raise ValueError("txbb_cdf_wp_cuts_leading cannot be used with old_wp_def=True: the working-point index assumes the offset 0")
        else:
            # HiggsLeading inherits a field of the same name from FatJet (the by-value map of the subleading Higgs): it must not be used
            # as the leading working-point index, only the by-region column defined by txbb_cdf_wp_cuts_leading is valid
            for option in ("bkg_morphing_dnn_input_variables", "sig_bkg_dnn_input_variables"):
                if any(v[0] == "HiggsLeading" and v[1] == "btagBBTXbb_CDF_rescaled_WPs" for v in (getattr(self, option, None) or {}).values()):
                    raise ValueError(f"{option} uses HiggsLeading btagBBTXbb_CDF_rescaled_WPs, set the option txbb_cdf_wp_cuts_leading to define it")
        # Merging of the top working points and removal of the events with a leading index below the subleading one (A, B)
        self.txbb_cdf_max_wp = int(getattr(self, "txbb_cdf_max_wp", 0) or 0)
        self.cut_sub_above_lead = bool(getattr(self, "cut_sub_above_lead", False))
        if self.txbb_cdf_max_wp:
            if not 2 <= self.txbb_cdf_max_wp <= 5:
                raise ValueError(f"txbb_cdf_max_wp must be 0 (off) or between 2 and 5, got {self.txbb_cdf_max_wp}")
            if self.txbb_cdf_wp_cuts_leading:
                raise ValueError("txbb_cdf_max_wp cannot be combined with txbb_cdf_wp_cuts_leading: its rows are indexed by the unmerged subleading index")
        # Without txbb_cdf_wp_cuts these two options act on the LINEAR rescaling: the subleading index is `btagBBTXbb_morph_5wp`
        # (WP edges on the linearly stretched TXbb) instead of the CDF column.
        if (self.txbb_cdf_max_wp or self.cut_sub_above_lead) and self.old_wp_def:
            raise ValueError("txbb_cdf_max_wp and cut_sub_above_lead cannot be used with old_wp_def=True: the working-point indices assume the offset 0")
        if self.cut_sub_above_lead and self.txbb_cdf_wp_cuts_leading:
            raise ValueError("cut_sub_above_lead cannot be combined with txbb_cdf_wp_cuts_leading: it compares with the nominal leading index")

    @staticmethod
    def _load_txbb_cdf_wp_cuts(option):
        """Read the output of scripts/find_txbb_cdf_wp_cuts.py (path or dict). Returns the sorted cuts, or None if not requested."""
        if not option:
            return None
        if isinstance(option, str):
            with open(option) as f:
                option = json.load(f)
        if option.get("region_setting", "50GeV") != "50GeV":
            raise ValueError("txbb_cdf_wp_cuts is only supported for the 50GeV region setting")
        cuts = [float(c) for c in option["cuts_A"]]
        if len(cuts) != 4 or any(b < a for a, b in zip(cuts, cuts[1:])):
            raise ValueError(f"txbb_cdf_wp_cuts needs 4 increasing cuts_A (for the WPs 0.8, 0.95, 0.975, 0.99), got {cuts}")
        return cuts

    @staticmethod
    def _load_txbb_cdf_wp_cuts_leading(option, subleading_cuts):
        """Read the leading-Higgs map of scripts/find_txbb_cdf_wp_cuts.py --higgs leading (path or dict). Returns a table [5][4], or None.

        Row s - 1 holds the 4 cuts (for the WPs 0.8, 0.95, 0.975, 0.99) used in A/B when the subleading working-point index is s.
        - flat `cuts_A` (4 increasing values, the marginal map): the same row for every s;
        - nested `cuts_A` (5 rows, `conditional: true`, written with --conditional): one row per s. A cut of -1 (sentinel) is always exceeded
          and can only be a prefix of a row. The rows were derived with the subleading cuts stored in `subleading_cuts_A`, which must be the
          ones of the option txbb_cdf_wp_cuts (the subleading index s of an event in A/B is defined by them).
        """
        if not option:
            return None
        if isinstance(option, str):
            with open(option) as f:
                option = json.load(f)
        if option.get("region_setting", "50GeV") != "50GeV":
            raise ValueError("txbb_cdf_wp_cuts_leading is only supported for the 50GeV region setting")
        n_wp, n_sub, sentinel = 4, 5, -1.0
        cuts = option["cuts_A"]
        if len(cuts) == n_wp and not any(isinstance(c, (list, tuple)) for c in cuts):
            row = [float(c) for c in cuts]
            if any(b < a for a, b in zip(row, row[1:])):
                raise ValueError(f"txbb_cdf_wp_cuts_leading needs 4 increasing cuts_A (for the WPs 0.8, 0.95, 0.975, 0.99), got {row}")
            return [list(row) for _ in range(n_sub)]
        if not option.get("conditional", False):
            raise ValueError("txbb_cdf_wp_cuts_leading: a nested cuts_A needs 'conditional': true (output of --conditional)")
        table = [[float(c) for c in row] for row in cuts]
        if len(table) != n_sub or any(len(row) != n_wp for row in table):
            raise ValueError(f"txbb_cdf_wp_cuts_leading needs a {n_sub}x{n_wp} table cuts_A (rows: subleading index 1..{n_sub}), got {table}")
        for s_idx, row in enumerate(table, start=1):
            if any(b < a for a, b in zip(row, row[1:])):
                raise ValueError(f"txbb_cdf_wp_cuts_leading: the cuts of the row s={s_idx} are not increasing: {row}")
            if any(c != sentinel and c < 0 for c in row):
                raise ValueError(f"txbb_cdf_wp_cuts_leading: row s={s_idx} has a negative cut other than the sentinel {sentinel}: {row}")
        groups = option.get("subleading_groups", [[i] for i in range(1, n_sub + 1)])
        if [i for g in groups for i in g] != list(range(1, n_sub + 1)):
            raise ValueError(f"txbb_cdf_wp_cuts_leading: subleading_groups must be a contiguous partition of 1..{n_sub}, got {groups}")
        for g in groups:
            if any(table[i - 1] != table[g[0] - 1] for i in g):
                raise ValueError(f"txbb_cdf_wp_cuts_leading: the rows of the pooled subleading indices {g} differ")
        if subleading_cuts is not None:
            used = option.get("subleading_cuts_A")
            if used is None or len(used) != len(subleading_cuts) or any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(used, subleading_cuts)):
                raise ValueError(
                    "txbb_cdf_wp_cuts_leading was derived with other subleading cuts than the option txbb_cdf_wp_cuts "
                    f"({used} vs {list(subleading_cuts)}): rerun scripts/find_txbb_cdf_wp_cuts.py --higgs leading --conditional"
                )
        return table

    def define_cdf_rescaled_wps(self, fatjets, b_low, b_thresh):
        """TXbb working-point index with the low-TXbb jets (A, B) mapped to the WPs of the high-TXbb jets (C, D) by quantile.

        Jets above `b_thresh` (C, D): nominal WP index (number of the 5 WPs exceeded).
        Jets below `b_thresh` (A, B): 1 + number of the quantile-matched cuts exceeded, 0 below `b_low` (a jet exactly at `b_low` gets 1, like the baseline `>=`).
        Same scale as `btagBBTXbb_morph_5wp`. The lowest WP (0.65) is the lower edge of C, so it maps onto `b_low` and needs no cut.
        """
        fatjets = self.generate_btag_workingpoints(fatjets, 5, "btagBBTXbb")
        txbb = fatjets["btagBBTXbb"]
        offset = 1 if self.old_wp_def else 0
        # >= to match hh4b_boosted_baseline (TXbb >= b_low): a jet exactly at b_low (float32 ties occur in MC) must not get index 0
        low_wp = ak.where(txbb >= b_low, 1, 0) - offset
        for cut in self.txbb_cdf_wp_cuts:
            low_wp = low_wp + ak.where(txbb > cut, 1, 0)
        cdf_wp = ak.where(txbb > b_thresh, fatjets["btagBBTXbb_5wp"], low_wp)
        # NOTE: the FatJet column is never merged (`txbb_cdf_max_wp`): the leading jet inherits it, and the leading working-point
        # indices must keep all their bins. The merge is applied to the subleading Higgs only, see `merge_subleading_cdf_wps`.
        return ak.with_field(fatjets, cdf_wp, "btagBBTXbb_CDF_rescaled_WPs")

    def merge_subleading_cdf_wps(self, higgs_sublead):
        """Merge the working-point indices above `txbb_cdf_max_wp` into it, for the SUBLEADING Higgs only.

        Capping the cumulative index merges the working points (with 3: 0.95, 0.975, 0.99 -> index 3). The leading Higgs is not
        touched: neither its nominal index `btagBBTXbb_5wp` (0..5, used by `hh4b_boosted_lead_ge_sub`) nor its continuous TXbb.
        """
        # CDF column with txbb_cdf_wp_cuts, else the index of the linear rescaling (linear mode)
        column = "btagBBTXbb_CDF_rescaled_WPs" if self.txbb_cdf_wp_cuts else "btagBBTXbb_morph_5wp"
        cdf_wp = higgs_sublead[column]
        cdf_wp = ak.where(cdf_wp > self.txbb_cdf_max_wp, self.txbb_cdf_max_wp, cdf_wp)
        return ak.with_field(higgs_sublead, cdf_wp, column)

    def _txbb_region_edges(self):
        """(b_low, b_thresh) of the region setting: baseline TXbb cut and the threshold between the low-TXbb (A, B) and high-TXbb (C, D) regions."""
        if self.region_setting == "40GeV":
            return 0.3, 0.8
        if self.region_setting == "50GeV":
            return 0.05, 0.65
        raise ValueError(f"Unknown region_setting {self.region_setting}")

    def define_cdf_rescaled_leading_wps(self, higgs_lead, higgs_sublead, b_low, b_thresh):
        """TXbb working-point index of the leading Higgs, quantile-mapped in the low-TXbb regions (A, B) given the subleading index, BY REGION.

        Unlike the subleading Higgs, the leading TXbb in A and B overlaps the range of C and D (it is at least the subleading one and
        goes up to 1), so the map cannot be chosen by the value of the TXbb. The region is decided by the subleading TXbb, with the
        same predicate as `define_cdf_rescaled_wps` (subleading TXbb > b_thresh: C, D):
        - C, D: nominal WP index of the leading jet (number of the 5 WPs exceeded).
        - A, B: `(lead >= b_low)` + number of the cuts of the row s exceeded, where s is the subleading working-point index (field
          `btagBBTXbb_CDF_rescaled_WPs` of the subleading Higgs, 1..5 in A, B) and the row s holds the 4 leading cuts matched in A to the
          leading TXbb of C at the same s (a cut of -1 is always exceeded, so the index is at least s with the conditional table). A flat
          (marginal) option gives the same row for every s. A jet exactly at `b_low` gets 1, like the baseline `>=`.
        Same scale as `btagBBTXbb_CDF_rescaled_WPs` of the subleading Higgs. Returns `higgs_lead` with the field `btagBBTXbb_CDF_rescaled_WPs`,
        which replaces the by-value field inherited from the FatJet.
        """
        txbb = higgs_lead["btagBBTXbb"]
        low_region = ak.fill_none(higgs_sublead["btagBBTXbb"] <= b_thresh, False)
        sub_index = ak.fill_none(higgs_sublead["btagBBTXbb_CDF_rescaled_WPs"], 0)
        low_wp = ak.where(txbb >= b_low, 1, 0)
        for row_number, row in enumerate(self.txbb_cdf_wp_cuts_leading, start=1):
            row_wp = ak.zeros_like(low_wp)
            for cut in row:
                row_wp = row_wp + ak.where(txbb > cut, 1, 0)
            low_wp = low_wp + ak.where(sub_index == row_number, row_wp, 0)
        # nominal index of the leading jet, computed from its own TXbb (not read from the inherited FatJet fields)
        nominal_wp = ak.zeros_like(txbb, dtype=np.int32)
        for wp in (0.65, 0.8, 0.95, 0.975, 0.99):
            nominal_wp = nominal_wp + ak.where(txbb > wp, 1, 0)
        return ak.with_field(higgs_lead, ak.where(low_region, low_wp, nominal_wp), "btagBBTXbb_CDF_rescaled_WPs")

    def process_extra_after_skim(self):
        super().process_extra_after_skim()

        if (
            self.vbf_parton_matching
            and self._isMC
            and "VBFHHto4B" in self.events.metadata["dataset"]
        ):
            # do truth matching to get VBF-jets
            self.do_vbf_parton_matching(
                which_vbf_quark=self.which_vbf_quark, jet_collection="Jet"
            )

        self.def_provenance_field()
        self.define_jet_collections()

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation=variation)
        if self.boosted:
            # Uncorrected softdrop mass for the comparison with the MiniAOD ntuples: the msoftdrop
            # calibrator overwrites msoftdrop in place, so rebuild it from the raw subjets
            subjets = self.events["FatJet"].subjets
            msoftdrop_raw = (subjets * (1 - subjets.rawFactor)).sum(axis=-1).mass
            self.events["FatJet"] = ak.with_field(
                self.events["FatJet"],
                ak.where(self.events["FatJet"].msoftdrop > 0, msoftdrop_raw, self.events["FatJet"].msoftdrop),
                "msoftdrop_raw",
            )

            # ===== BB-tagging =====
            # here we propagate the btagging scores to the FatJet collection as is done in the pocket coffea jet_selection
            # if we're interested in other taggers, we need to add them here or swap to the mass correlated ones ("particleNetWithMass_HbbvsQCD", "particleNetWithMass_HccvsQCD")

            if "2024" in self._year:
                self.events["FatJet"] = ak.with_field(
                    self.events["FatJet"],
                    (self.events["FatJet"]["globalParT3_Xbb"] / (self.events["FatJet"]["globalParT3_Xbb"] + self.events["FatJet"]["globalParT3_QCD"])),
                    "btagBBTXbb",
                )
                b_low, b_thresh = self._txbb_region_edges()
                self.events["FatJet"] = ak.with_field(self.events["FatJet"], ak.where(self.events["FatJet"]["btagBBTXbb"] > b_thresh, (self.events["FatJet"]["btagBBTXbb"]), (self.events["FatJet"]["btagBBTXbb"] - b_low) * (1-b_thresh) / (b_thresh - b_low) + b_thresh), "btagBBTXbb_morph")
                # self.events["FatJet"] = ak.with_field(
                #     self.events["FatJet"],
                #     (self.events["FatJet"]["particleNetLegacy_Xbb"] / (self.events["FatJet"]["particleNetLegacy_Xbb"] + self.events["FatJet"]["particleNetLegacy_QCD"])),
                #     "btagBBPNetLegacy",
                # )
                self.events["FatJet"] = ak.with_field(
                    self.events["FatJet"],
                    self.events["FatJet"]["globalParT3_Xbb"],
                    "btagBB",
                )
            else:
                self.events["FatJet"] = ak.with_field(
                    self.events["FatJet"],
                    self.events["FatJet"]["particleNet_XbbVsQCD"],
                    "btagBB",
                )
            self.events["FatJet"] = ak.with_field(
                self.events["FatJet"],
                self.events["FatJet"]["particleNet_XccVsQCD"],
                "btagCC",
            )
            # Creating btag working points for both the nominal and the morphed/scaled btag values
            self.events["FatJet"] = self.generate_btag_workingpoints(self.events["FatJet"], 5, "btagBBTXbb_morph")
            self.events["FatJet"] = self.generate_btag_workingpoints(self.events["FatJet"], 5, "btagBBTXbb")
            if "2024" in self._year and self.txbb_cdf_wp_cuts:
                self.events["FatJet"] = self.define_cdf_rescaled_wps(self.events["FatJet"], b_low, b_thresh)
            # jet ordered in btagging score
            if not self.TXbb_order:
                self.events["FatJet"] = self.events["FatJet"][
                    ak.argsort(self.events["FatJet"]["btagBB"], axis=1, ascending=False)
                ]
            else:
                self.events["FatJet"] = self.events["FatJet"][
                    ak.argsort(self.events["FatJet"]["btagBBTXbb"], axis=1, ascending=False)
                ]
            # ===== Regressions (needed for additional cuts) ===== 
            if self._year == "2024":
                self.events["FatJet"] = ak.with_field(self.events["FatJet"], self.events["FatJet"]["mass"], "mass_orig")
                self.events["FatJet"] = ak.with_field(
                    self.events["FatJet"],
                    self.events["FatJet"].mass_raw * self.events["FatJet"].globalParT3_massCorrGeneric,  # Check, if we should use globalParT3_massCorrGeneric
                    "mass"
                )
                self.events["FatJet"] = ak.with_field(
                    self.events["FatJet"],
                    self.events["FatJet"].mass_raw * self.events["FatJet"].globalParT3_massCorrX2p,  # Check, if we should use globalParT3_massCorrGeneric
                    "mass_x2p"
                )
            else:
                self.events["FatJet"] = ak.with_field(self.events["FatJet"], self.events["FatJet"]["mass"], "mass_orig")
                self.events["FatJet"] = ak.with_field(
                    self.events["FatJet"],
                    self.events["FatJet"].mass * self.events["FatJet"].particleNet_massCorr,
                    "mass"
                )
                # self.events["FatJetGood"] = ak.with_field(
                #     self.events["FatJetGood"],
                #     self.events["FatJetGood"].pt_raw * self.events["FatJetGood"].particleNet_massCorr,
                #     "pt_regr"
                # )

            # Select FatJets for the boosted category
            self.events["FatJetGood"] = copy.copy(self.events.FatJet)

            # We do only take the masks from the leading and subleading jets. Then we apply the masks to FatJetGood
            # This does require the additional fields we add to the FatJet collection inside the function to be also added to the final collection
            _, mask_fat_lead = custom_jet_selection(
                self.events,
                jet_type="FatJetGood",
                jet_type_obj_presel="FatJetLeading",
                params=self.params,
                year=self._year,
                jet_tagger="PNet",
                pt_type="pt",
                pt_cut_name="pt",
            )
            _, mask_fat_sublead = custom_jet_selection(
                self.events,
                jet_type="FatJetGood",
                jet_type_obj_presel="FatJetSubLeading",
                params=self.params,
                year=self._year,
                jet_tagger="PNet",
                pt_type="pt",
                pt_cut_name="pt",
            )
            fatjet_obj_presel_lead = self.params.object_preselection["FatJetLeading"]
            fatjet_obj_presel_sublead = self.params.object_preselection["FatJetSubLeading"]
            if "mass_regr_min" in fatjet_obj_presel_lead.keys() and "mass_regr_max" in fatjet_obj_presel_lead.keys():
                mask_mass_regr = (
                    (self.events["FatJetGood"]["mass"] >= fatjet_obj_presel_lead["mass_regr_min"]) &
                    (self.events["FatJetGood"]["mass"] <= fatjet_obj_presel_lead["mass_regr_max"])
                )
                mask_fat_lead = mask_fat_lead & mask_mass_regr
            if "mass_regr_min" in fatjet_obj_presel_sublead.keys() and "mass_regr_max" in fatjet_obj_presel_sublead.keys():
                mask_mass_regr = (
                    (self.events["FatJetGood"]["mass"] >= fatjet_obj_presel_sublead["mass_regr_min"]) &
                    (self.events["FatJetGood"]["mass"] <= fatjet_obj_presel_sublead["mass_regr_max"])
                )
                mask_fat_sublead = mask_fat_sublead & mask_mass_regr
            # == Cut on b-tag
            # mask_fat_lead = mask_fat_lead & (self.events["FatJetGood"]["btagBB"] > 0.65)
            # mask_fat_sublead = mask_fat_sublead & (self.events["FatJetGood"]["btagBB"] > 0.00) # used to be 0.05

            # ===== Cutting and combining the two fatjets =====
            self.events["FatJetGoodLeading"] = self.events["FatJetGood"][mask_fat_lead][:, :1]
            lead_idx = ak.local_index(self.events["FatJetGood"], axis=1)[mask_fat_lead][:, :1]
            all_idx = ak.local_index(self.events["FatJetGood"], axis=1)
            not_lead_mask = all_idx != ak.firsts(lead_idx)
            mask_fat_sublead_excl = mask_fat_sublead & not_lead_mask
            self.events["FatJetGoodSubLeading"] = self.events["FatJetGood"][mask_fat_sublead_excl][:, :1]

            self.events["FatJetGoodSelected"] = ak.concatenate([self.events["FatJetGoodLeading"], self.events["FatJetGoodSubLeading"]], axis=1)
            self.events["nFatJetGoodSelected"] = ak.num(self.events["FatJetGoodSelected"], axis=1)
            self.events["nFatJet"] = ak.num(self.events["FatJet"], axis=1)
            self.events["nFatJetGood"] = ak.num(self.events["FatJetGood"], axis=1)

            # mask_pt = self.events["FatJetGood"]["pt"] > 250
            # mask_eta = abs(self.events["FatJetGood"]["eta"]) < 2.5
            # mask_msd = self.events["FatJetGood"]["msoftdrop"] > 50

            # jetidtight = (
            #     (
            #         (np.abs(self.events["FatJetGood"].eta) <= 2.6)
            #         & (self.events["FatJetGood"].neHEF < 0.99)
            #         & (self.events["FatJetGood"].neEmEF < 0.9)
            #         & ((self.events["FatJetGood"].chMultiplicity + self.events["FatJetGood"].neMultiplicity) > 1)
            #         & (self.events["FatJetGood"].chHEF > 0.01)
            #         & (self.events["FatJetGood"].chMultiplicity > 0)
            #     )
            # | (
            # ((np.abs(self.events["FatJetGood"].eta) > 2.6) & (np.abs(self.events["FatJetGood"].eta) <= 2.7))
            # & (self.events["FatJetGood"].neHEF < 0.90)
            # & (self.events["FatJetGood"].neEmEF < 0.99)
            # )
            # | (((np.abs(self.events["FatJetGood"].eta) > 2.7) & (np.abs(self.events["FatJetGood"].eta) <= 3.0)) & (self.events["FatJetGood"].neHEF < 0.99))
            # | ((np.abs(self.events["FatJetGood"].eta) > 3.0) & (self.events["FatJetGood"].neMultiplicity >= 2) & (self.events["FatJetGood"].neEmEF < 0.4))
            # )
            # jetidtightlepveto = (
            # (np.abs(self.events["FatJetGood"].eta) <= 2.7) & jetidtight & (self.events["FatJetGood"].muEF < 0.8) & (self.events["FatJetGood"].chEmEF < 0.8)
            # ) | ((np.abs(self.events["FatJetGood"].eta) > 2.7) & jetidtight)

            # self.events["FatJetGoodTest"] = self.events["FatJetGood"][jetidtight & jetidtightlepveto & mask_pt & mask_eta & mask_msd]


        if self.vbf_analysis:
            if not self.boosted:
                self.define_vbf_jet_collections()

            if self.boosted:
                self.events["JetGood"], mask_jet_vbf = custom_jet_selection(
                    self.events,
                    "Jet",
                    "JetBoosted",
                    self.params,
                    year=self._year,
                    pt_type="pt",
                    pt_cut_name=self.pt_cut_name,
                    forward_jet_veto=False,
                )
                self.events["JetGoodVBF"], mask_jet_vbf = custom_jet_selection(
                    self.events,
                    "Jet",
                    "JetVBF",
                    self.params,
                    year=self._year,
                    pt_type="pt",
                    pt_cut_name=self.pt_cut_name,
                    forward_jet_veto=False,
                )
                self.events["JetGoodVBF"] = self.events.JetGoodVBF[
                    ak.argsort(self.events.JetGoodVBF.pt, axis=1, ascending=False)
                ]
                # alias needed by count_objects of the common workflow
                self.events["JetGoodVBFCandidates"] = self.events["JetGoodVBF"]
                self.events["JetVBF"] = copy.copy(self.events.Jet)
                self.events["HT_jetJetGoodVBF"] = ak.sum(self.events.JetGoodVBF.pt, axis=1)
                self.events["HT_jetJetGood"] = ak.sum(self.events.JetGood.pt, axis=1)
                self.events["HT_jetJet"] = ak.sum(self.events.Jet.pt, axis=1)
                self.events["HT"] = ak.sum(self.events.Jet.pt, axis=1)

                self.events["JetGoodCloseToFatJet"], mask_jet_close_to_fatjet = custom_jet_selection(
                    self.events,
                    "Jet",
                    "JetNearFatJet",
                    self.params,
                    year=self._year,
                    pt_type="pt",
                    pt_cut_name=self.pt_cut_name,
                    forward_jet_veto=False,
                )
                # order in pt
                # Clean From AK8
                # self.events["JetGoodVBFCandidates"] = clean_ak4_boosted(
                #         self.events["JetGoodBoosted"],
                #         self.events["FatJetGoodSelected"],
                #         self.events["Muon"][self.events["Muon"]["pt"] > 7],
                #         self.events["Electron"][self.events["Electron"]["pt"] > 5],
                #         dr_jets=1.2, dr_lep=0.4
                #         )
                # self.events["JetGoodCloseToFatJet"] = clean_ak4_boosted(
                #         self.events["JetGoodBoosted"][np.abs(self.events["JetGoodBoosted"]["eta"]) < 2.5],
                #         self.events["FatJetGoodSelected"],
                #         self.events["Muon"][self.events["Muon"]["pt"] > 7],
                #         self.events["Electron"][self.events["Electron"]["pt"] > 5],
                #         dr_jets=0.9, dr_lep=0.4
                #         )

                # The equivalent of this for the not-boosted is in the main workflow_common. But there it is after the preselection. So I am not sure, how to merge the two.
                # vbf_pool = self.events["JetGoodVBF"]

                # # Shortcut to the VBF jet preselection values
                # jetvbf_obj_presel = self.params.object_preselection["JetVBF"]
                # # looser VBF cuts
                # mask_pt_vbf = ak.fill_none(vbf_pool.pt > jetvbf_obj_presel["pt"], False)
                # # additional cuts for the region 2.5 < |eta| < 3.0
                # central_or_forward = (np.abs(vbf_pool.eta) < jetvbf_obj_presel["gap_eta_min"]) | (np.abs(vbf_pool.eta) > jetvbf_obj_presel["gap_eta_max"])
                # gap_higher_pt = (np.abs(vbf_pool.eta) >= jetvbf_obj_presel["gap_eta_min"]) & (np.abs(vbf_pool.eta) <= jetvbf_obj_presel["gap_eta_max"]) & (vbf_pool.pt > jetvbf_obj_presel["gap_pt"])
                # within_max_eta = np.abs(vbf_pool.eta) < jetvbf_obj_presel["eta"]

                # mask_eta_vbf = ak.fill_none(
                #     (central_or_forward | gap_higher_pt) & within_max_eta,
                #     False,
                # )
                # self.events["JetGoodVBF"] = ak.pad_none(vbf_pool[mask_pt_vbf & mask_eta_vbf], 2, clip=True)

                self.events["JetGoodVBFEnergyOrderedPadded"] = get_lead_mjj_jet_pair(
                    self.events, "JetGoodVBF"
                )
                self.events["JetGoodVBFGood"] = self.events["JetGoodVBFEnergyOrderedPadded"][~ak.is_none(self.events["JetGoodVBFEnergyOrderedPadded"]["pt"], axis=1)]
                for jet_coll in ["JetGood", "JetGoodVBF", "JetGoodVBFGood"]:
                    padded = ak.pad_none(self.events[jet_coll], 2, axis=1)
                    vbf_mjj = (
                        padded[:, 0]
                        + padded[:, 1]
                    ).mass
                    vbf_deta = abs(
                        padded[:, 0].eta
                        - padded[:, 1].eta
                    )

                    self.events[f"mjj{jet_coll}"] = ak.fill_none(vbf_mjj, -999.0)
                    self.events[f"deta{jet_coll}"] = ak.fill_none(vbf_deta, -999.0)

                # # build dijets for veto
                # dijets = ak.combinations(self.events["JetGoodVBFCandidates"], 2, fields=["j_lead", "j_sublead"])
                # dijets = ak.fill_none(dijets, [])
                # d4 = dijets.j_lead + dijets.j_sublead
                # for param in ["mass", "pt", "eta", "phi"]:
                #     dijets = ak.with_field(dijets, getattr(d4, param), param)
                # dijets = ak.with_field(dijets, np.abs(dijets.j_lead.eta - dijets.j_sublead.eta), "dEta")

                # self.events["DiJetVBF"] = ak.pad_none(dijets,1,clip=True)[:,0]

    def count_objects(self, variation):
        super().count_objects(variation=variation)
        if self.vbf_analysis:
            self.events["nJetGoodVBF"] = ak.num(self.events.JetGoodVBF, axis=1)
            self.events["nJetGoodCloseToFatJet"] = ak.num(self.events.JetGoodCloseToFatJet, axis=1)
            self.events["nJetGoodVBFEnergyOrderedPadded"] = ak.sum(~ak.is_none(self.events.JetGoodVBFEnergyOrderedPadded, axis=1), axis=1)
            self.events["nJetGoodVBFGood"] = ak.num(self.events.JetGoodVBF, axis=1)
            self.events["nJet"] = ak.num(self.events.JetGoodVBF, axis=1)

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        if self.vbf_analysis and not self.boosted:

            self.define_vbf_pair_collections()

            # Define mjj, delta eta and centrality of leading mjj vbf jet candidates
            # the Higgs candidates are only built after the common workflow runs,
            # so the centrality cannot be computed here
            self.define_vbf_kinematics(
                ["JetGoodVBFMergedProvVBFPadded"], [0], centrality=False
            )

        super().process_extra_after_presel(variation=variation)

        if self.bdt_model:
            bdt_events = get_default_bdt_inputs(self.events)
            self.events["boosted_bdt_score"], self.events["boosted_bdt_vbf_score"] = evaluate_bdt(self.bdt_model, bdt_events)

        if self.boosted and getattr(self, "boosted_fit_sidebands", False):
            self.eval_sideband_scores()

    def get_sideband_mass_shifts(self):
        """Shifts of the (leading, subleading) Higgs candidate mass that move the fit sidebands into the signal window.

        - "_qcdsb": QCD sideband (region C / A): the Higgs mass defining the ABCD plane (`lead_sublead`) is moved
          by +sideband_size if it is below the window (-sideband_size if above, for completeness).
        - "_ttsb": ttbar sideband (regions E / F): only the leading Higgs mass is moved down by sideband_size, the
          subleading one is NOT shifted (the variables built from the leading mass follow: see `build_mass_shifted_events`).
        """
        window = boosted_mass_window(self.region_setting)
        size = float(window["sideband_size"])
        abcd_higgs = "HiggsLeading" if self.lead_sublead == "lead" else "HiggsSubLeading"
        abcd_mass = self.events[abcd_higgs].mass
        shift_abcd = ak.fill_none(
            ak.where(abcd_mass < window["lower"], size, ak.where(abcd_mass > window["upper"], -size, 0.0)),
            0.0,
        )
        no_shift = ak.zeros_like(shift_abcd)
        qcd_shift = (shift_abcd, no_shift) if self.lead_sublead == "lead" else (no_shift, shift_abcd)
        return {
            "_qcdsb": qcd_shift,
            "_ttsb": (no_shift - size, no_shift),
        }

    def build_mass_shifted_events(self, shift_lead, shift_sublead):
        """Copy of the events with the masses of the two selected fat jets shifted and all the Higgs/HH variables
        that depend on them (HH mass, m_Hjet_min_dR, massclosestVBF, ...) rebuilt. `self.events` is not modified."""
        events_nominal = self.events
        self.events = copy.copy(events_nominal)
        try:
            fatjets = self.events["FatJetGoodSelected"]
            jet_idx = ak.local_index(fatjets, axis=1)
            shift = ak.where(jet_idx == 0, shift_lead, shift_sublead)
            fatjets = ak.with_field(fatjets, fatjets.mass + shift, "mass")
            self.events["FatJetGoodSelected"] = fatjets
            (
                self.events["HiggsLeading"],
                self.events["HiggsSubLeading"],
                self.events["HH"],
                self.events["LeadingVBFJet"],
                self.events["SubLeadingVBFJet"],
            ) = self.define_boosted_dnn_variables(
                fatjets[:, 0],
                fatjets[:, 1],
                self.events.JetGoodVBFGood,
                vbf_variables=True,
            )
            self.build_boosted_variables()
            events_shifted = self.events
        finally:
            self.events = events_nominal
        return events_shifted

    def eval_sideband_scores(self):
        """Evaluate the category scores (BDT and sig-vs-bkg DNN) with the Higgs masses shifted into the signal
        window, as it is done for the ttbar/QCD sidebands of the fit. Saved as `<score><suffix>`.
        The background morphing weight is NOT shifted: it is always evaluated with the true masses."""
        for suffix, (shift_lead, shift_sublead) in self.get_sideband_mass_shifts().items():
            events_shifted = self.build_mass_shifted_events(shift_lead, shift_sublead)
            if self.bdt_model:
                bdt_score, bdt_vbf_score = evaluate_bdt(self.bdt_model, get_default_bdt_inputs(events_shifted))
                self.events[f"boosted_bdt_score{suffix}"] = bdt_score
                self.events[f"boosted_bdt_vbf_score{suffix}"] = bdt_vbf_score
            if self.sig_bkg_dnn:
                for name, score in self.eval_sig_bkg_dnn(events_shifted).items():
                    self.events[f"{name}{suffix}"] = score
            del events_shifted

    def build_boosted_variables(self):
        """Define the Higgs variable used for a boosted analysis."""
        self.events["HiggsLeading"] = ak.with_field(
            self.events["HiggsLeading"],
            ak.fill_none(self.events.HiggsLeading.tau3 / self.events.HiggsLeading.tau2, -999),
            "Tau3OverTau2"
        )
        self.events["HiggsSubLeading"] = ak.with_field(
            self.events["HiggsSubLeading"],
            ak.fill_none(self.events.HiggsSubLeading.tau3 / self.events.HiggsSubLeading.tau2, -999),
            "Tau3OverTau2"
        )
        self.events["HiggsLeading"] = ak.with_field(
            self.events["HiggsLeading"],
            ak.fill_none(self.events.HiggsLeading.pt / self.events.HH.mass, -999),
            "divHHmass"
        )
        self.events["HiggsSubLeading"] = ak.with_field(
            self.events["HiggsSubLeading"],
            ak.fill_none(self.events.HiggsSubLeading.pt / self.events.HH.mass, -999),
            "divHHmass"
        )
        self.events["JetGoodVBFNearHiggsLeading"] = ak.firsts(self.events["JetGoodCloseToFatJet"][ak.argsort(self.events["JetGoodCloseToFatJet"].delta_r(self.events["HiggsLeading"]), ascending=True)])
        self.events["JetGoodVBFNearHiggsSubLeading"] = ak.firsts(self.events["JetGoodCloseToFatJet"][ak.argsort(self.events["JetGoodCloseToFatJet"].delta_r(self.events["HiggsSubLeading"]), ascending=True)])
        self.events["HiggsLeadingByHiggsSubLeadingPt"] = self.events.HiggsLeading.pt / self.events.HiggsSubLeading.pt
        self.events["HiggsLeading"] = ak.with_field(
            self.events["HiggsLeading"],
            ak.fill_none(self.events.HiggsLeading.delta_r(self.events.JetGoodVBFNearHiggsLeading), -999),
            "dRclosestVBF"
        )
        self.events["HiggsSubLeading"] = ak.with_field(
            self.events["HiggsSubLeading"],
            ak.fill_none(self.events.HiggsSubLeading.delta_r(self.events.JetGoodVBFNearHiggsSubLeading), -999),
            "dRclosestVBF"
        )
        self.events["HiggsLeading"] = ak.with_field(
            self.events["HiggsLeading"],
            ak.fill_none((self.events.HiggsLeading + self.events.JetGoodVBFNearHiggsLeading).mass, -999),
            "massclosestVBF",
        )
        self.events["HiggsSubLeading"] = ak.with_field(
            self.events["HiggsSubLeading"],
            ak.fill_none((self.events.HiggsSubLeading + self.events.JetGoodVBFNearHiggsSubLeading).mass, -999),
            "massclosestVBF",
        )
        self.events["HiggsLeading"] = ak.with_field(
            self.events["HiggsLeading"],
            ak.fill_none(disc_TXbb(self.events.HiggsLeading.btagBBTXbb), -999),
            "btagBBTXbb_dig",
            )
        self.events["HiggsLeading"] = ak.with_field(
            self.events["HiggsLeading"],
            ak.fill_none(disc_TXbb(self.events.HiggsLeading.btagBBTXbb, "old"), -999),
            "btagBBTXbb_dig_old",
            )
        if self.txbb_cdf_wp_cuts_leading and "2024" in self._year:
            # quantile-mapped working-point index of the leading Higgs, by region (needs the subleading TXbb)
            b_low, b_thresh = self._txbb_region_edges()
            self.events["HiggsLeading"] = self.define_cdf_rescaled_leading_wps(
                self.events["HiggsLeading"], self.events["HiggsSubLeading"], b_low, b_thresh
            )
        if self.txbb_cdf_max_wp and "2024" in self._year:
            # merged top working points of the subleading Higgs (the leading Higgs keeps all its bins)
            self.events["HiggsSubLeading"] = self.merge_subleading_cdf_wps(self.events["HiggsSubLeading"])
