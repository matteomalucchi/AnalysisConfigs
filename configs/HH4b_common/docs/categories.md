# Categories

The categories (PocketCoffea `categories`, in which the histograms and the columns are filled)
are built by `define_categories` in
[`config_files/configurator_tools.py`](../config_files/configurator_tools.py), from the
configuration options (see [config_options.md](config_options.md#categories-and-regions)).
The approach changes the categories completely: the first approach uses the 4b/2b split and a
circular region in the Higgs mass plane, the second approach the b-tag categories and the
SR/CR of HIG-24-010 (see [approaches.md](approaches.md)). The VBF and ggF categories are built
in the same way on top of both.

On data, the categories whose name contains `postW` get the background morphing weight
`bkg_morphing_dnn_weight` (the `preW` ones are the same selection without it), and, with
`bkg_morphing_spread_dnn`, also save `bkg_morphing_spread_dnn_weights`.

## First approach

### Building blocks

| Cut | Definition |
| --- | --- |
| `4b` (`hh4b_4b_region`) | the 3rd and 4th `JetGoodHiggs` jets (b-tag ordered) have `btagB` > 0.2605 |
| `2b` (`hh4b_2b_region`) | the 3rd and 4th `JetGoodHiggs` jets have `btagB` < 0.2605 (the 2 leading b-tag jets are constrained only by the b-tag cut of the event preselection) |
| `signal_region` (`hh4b_signal_region`) | $R_{HH} = \sqrt{(M_{H1}-125)^2 + (M_{H2}-120)^2}$ < 30 GeV |
| `control_region` (`hh4b_control_region`) | 30 ≤ $R_{HH}$ < 55 GeV; 30 ≤ $R_{HH}$ < 80 GeV with `expandCR` (`hh4b_control_region_wide`, only for the `preW`/`postW` categories) |
| `VR1_signal_region`, `VR1_control_region` | same as the signal and control regions, centred at $(M_{H1}, M_{H2})$ = (185, 180) GeV |
| `blind` (`blinded`) | `sig_bkg_dnn_score` < 0.9 |
| `high_score` (`hh4b_sig_bkg_score_cut`) | `sig_bkg_dnn_score` ≥ 0.5 |
| `preW` / `postW` | no cut: the same selection without / with the background morphing weight on data |
| `sf_btag` | no cut: copy of the category used for the b-tag SF computation |

The category name is the concatenation of the building blocks (`define_single_category`), e.g.
`2b_signal_region_postW_blind` = 2b + signal region + blinding cut, with the morphing weight.

### Categories

| Category | Created when |
| --- | --- |
| `4b_region` | always (4b, no mass-plane cut), except with `vr1`, where it is added only without `spanet` and `run2` |
| `4b_signal_region`, `4b_control_region` | always |
| `2b_signal_region_preW`, `2b_control_region_preW` | always (`4b_*_preW` with `mixeddata`) |
| `2b_signal_region_postW`, `2b_control_region_postW`, `2b_region_postW` | `bkg_morphing_dnn` (`4b_signal_region_postW`, `4b_control_region_postW` with `mixeddata`) |
| `4b_signal_region_blind`, `2b_signal_region_preW_blind`, `2b_signal_region_postW_blind` | `blind` (`4b_*` instead of `2b_*` with `mixeddata`) |
| `4b_signal_region_high_score`, `2b_signal_region_preW_high_score`, `2b_signal_region_postW_high_score` (and their `_blind` versions with `blind`) | `sig_bkg_dnn` (`high_score_reg` in `HH4b_parton_matching_config.py`; `4b_*` instead of `2b_*` with `mixeddata`) |
| `4b_VR1_signal_region`, `4b_VR1_control_region`, `2b_VR1_signal_region_preW`, `2b_VR1_control_region_preW` | `vr1`, instead of all the above |
| `2b_VR1_signal_region_postW`, `2b_VR1_control_region_postW` | `vr1` and `bkg_morphing_dnn` |
| `<category>_sf_btag` | `btag_sf_comp` (b-tag SF configs of `configs/HH4b_btagging`): a copy of every category |
| VBF and ggF categories | `vbf_analysis`, see [below](#vbf-and-ggf-categories) |

## Second approach

### Building blocks

- Mass plane (`hh4b_mass_plane_region`, `SECOND_APPROACH_MASS_PLANE` in
  [`custom_cuts_common.py`](../custom_cuts_common.py)), with
  $X_{HH} = \sqrt{(M_{H1}-125)^2 + (M_{H2}-120)^2}$ and
  $X_{ab} = \sqrt{((M_{H1}-m_a)/(0.1 M_{H1}))^2 + ((M_{H2}-m_b)/(0.1 M_{H2}))^2}$,
  $m_H$ = 125 (H1) / 120 (H2) GeV, $m_Z$ = 91 (H1) / 89 (H2) GeV:
  - `SR` = $X_{HH}$ < 30 GeV ∪ $X_{ZZ}$ < 2.6 ∪ $X_{ZH}$ < 1.9 ∪ $X_{HZ}$ < 1.9;
  - `CR` = (52 < $M_{H1}$ < 180 GeV) ∩ (50 < $M_{H2}$ < 173 GeV), outside the SR.
- b-tag categories (`hh4b_btag_category`) of the 4 HH jets (`JetGoodFromHiggsOrdered`), with
  the working points of the year's tagger and $s_0 \ge s_1 \ge s_2 \ge s_3$ their b-tag scores:

  | Category | b-tag requirement | Use |
  | --- | --- | --- |
  | `3T1M` | $s_2$ ≥ T, $s_3$ ≥ M (≥ 3 T, 4th jet at least M) | signal region; its CR is the target of the CR2M → CR3T1M background training |
  | `3T1L` | $s_2$ ≥ T, L ≤ $s_3$ < M (exactly 3 T, 4th jet L but not M) | second signal region |
  | `2M` | $s_1$ ≥ M, $s_2$ < M (exactly 2 M) | background estimate: training in CR2M, weights applied on data in SR2M |
  | `3MnT` | $s_2$ ≥ M, $s_0$ < T (the 3 leading jets pass M but fail T) | validation, background training of SR3T1L |
  | `3M` | $s_2$ ≥ M, $s_3$ < M (exactly 3 M) | 2024 and 2025 only |
  | `4M` | $s_3$ ≥ M | 2024 and 2025 only |

### Categories

| Category | Created when |
| --- | --- |
| `SR3T1M`, `CR3T1M`, `SR3T1L`, `CR3T1L`, `SR2M`, `CR2M`, `SR3MnT` | always |
| `SR3M`, `CR3M`, `SR4M`, `CR4M` | always (empty in 2022-2023) |
| `SR2M_postW`, `CR2M_postW` | `bkg_morphing_dnn` |
| `SR3T1M_blind`, `SR3T1L_blind` | `blind` |
| VBF and ggF categories | `vbf_analysis`, see [below](#vbf-and-ggf-categories) |

## VBF and ggF categories

With `vbf_analysis`, the VBF and ggF categories are built in the same way in the
first and in the second approach (`define_vbf_ggf_categories`): for each
inclusive category `<name>`, defined by the b-tag and mass-plane cuts of the
approach, two orthogonal categories covering `<name>` are created:

| Category | Selection |
| --- | --- |
| `vbf_<name>` | ≥ 2 VBF jets (first two jets of the VBF jet collection, see below), with $m_{jj}$ > 400 GeV and $\Delta\eta$ > 3.5 in the first approach only (`vbf_kin_cuts`), and, when a `vbf_discriminator` is set, `VBF_ggF_score` ≥ `ggf_vbf_threshold` |
| `ggf_<name>` | events failing any of the cuts of `vbf_<name>` (`hh4b_vbf_selection_region(..., pass_vbf=False)`) |

The VBF jet collection the cuts are applied on (`get_vbf_jet_collection` in
`configs/HH4b_common/vbf_jet_collections.py`) depends on what the workflow does:

- when creating the SPANet training inputs (no `spanet` and no `run2`):
  `JetGoodVBFMergedProvVBFPadded`, whose first two jets are the pair fed to
  SPANet (`spanet_vbf_candidates`);
- when a pairing is applied (`spanet` or `run2`): `JetGoodVBFSelectedPair`, the
  VBF pair assigned after the Higgs pairing, ordered by pt;
- boosted analysis: `JetGoodVBFEnergyOrdered` (pair leading in $m_{jj}$).

The same collection is the one saved in the output and the one to use for any
VBF study outside the workflow. `JetGoodVBFSelectedPair` is the one predicted by SPANet when the
model predicts the VBF jets (the ggF/VBF discriminator with
`vbf_pairing_from_vbf_discriminator`, or the Higgs pairing model). Otherwise it
is chosen among the jets left over by the Higgs pairing with the same
definition as the pair fed to the models (`spanet_vbf_candidates`, automatic by
default): in the first approach the pair leading in $m_{jj}$; in the second
approach the leading-pt jet and, if present, the leading-pt jet with opposite
$\eta$ ($\eta_1 \eta_2 < 0$) (see [jets.md](jets.md#vbf-jet-collections)).

The approaches differ in the definition of the VBF jets, in the $m_{jj}$/$\Delta\eta$ cuts
(first approach only) and in the inclusive categories `<name>`:

| Approach | Function | $m_{jj}$, $\Delta\eta$ cuts | `<name>` |
| --- | --- | --- | --- |
| `first` | `define_first_approach_vbf_categories` | yes | `4b_signal_region`, `4b_control_region`; with `bkg_morphing_dnn` also `2b_signal_region_postW`, `2b_control_region_postW` (`4b_*_postW` with `mixeddata`) |
| `second` | `define_second_approach_vbf_categories` | no | `SR3T1M`, `CR3T1M`, `SR3T1L`, `CR3T1L` (2022-2023, `SECOND_APPROACH_VBF_BTAG_CATEGORIES`), `SR4M`, `CR4M` (2024-2025), `SR2M`, `CR2M`; with `bkg_morphing_dnn` also `SR2M_postW`, `CR2M_postW` |

e.g. `vbf_4b_signal_region` / `ggf_4b_signal_region` (first) and `vbf_SR3T1M` /
`ggf_SR3T1M` (second). In the first approach, when a `vbf_discriminator` is set,
only the VBF and ggF categories are defined (the inclusive ones are dropped).

## Boosted analysis

With `boosted`, the categories are built on the two selected FatJets (`FatJetGoodSelected`,
leading and subleading). All the names get `_vbf` after `boosted` with `vbf_analysis`, which adds
≥ 2 VBF jets with $m_{jj}$ > 300 GeV and $\Delta\eta$ > 3.5 (`hh4b_boosted_vbf_region`).

| Cut | Definition |
| --- | --- |
| `signal` | subleading FatJet `btagBB` > 0.60 (the cut on the leading regressed mass, 100-150 GeV, is defined but not applied) |
| `boosted_group_signal` | leading FatJet `btagBBTXbb` ≥ 0.30 (selection of the other boosted group) |
| `ttbar` | both FatJets with 150 < `mass_regr` < 200 GeV |
| `qcd_A` | leading 40 < `mass_regr` < 100 GeV, subleading 40 < `mass_regr` < 200 GeV and `btagBB` < 0.60 |
| `qcd_B` | as `qcd_A` with subleading `btagBB` ≥ 0.60 |
| `qcd_C` | leading 100 < `mass_regr` < 150 GeV, subleading 40 < `mass_regr` < 200 GeV and `btagBB` < 0.60 |
| `qcd` | `cuts.hh4b_boosted_qcd_control_region`, which is not defined in `custom_cuts_common.py` (only `hh4b_boosted_qcd_control_region_tot` is): `split_qcd: False` currently fails |
| `pass`, `fail` | `VBF_ggF_score` ≥ / < `ggf_vbf_threshold` |

| Category | Created when |
| --- | --- |
| `boosted_boosted_group_signal_region` | `approach: "boosted"` (`other_group` in `VBF_HH4b_boosted_config.py`) |
| `boosted_incl_region`, `boosted_incl_signal_region`, `boosted_incl_qcd_{A,B,C}_region` | no `vbf_discriminator` (no ggF/VBF score cut) |
| `boosted_incl_qcd_A_region_postW`, `boosted_incl_qcd_C_region_postW` (`..._postW_blind` with `blind`) | no `vbf_discriminator`, `bkg_morphing_dnn` |
| `boosted_signal_region`, `boosted_ttbar_region`, `boosted_pass_region`, `boosted_fail_region` | `vbf_discriminator`; all but `fail` require `VBF_ggF_score` ≥ `ggf_vbf_threshold` |
| `boosted_qcd_{A,B,C}_region` (`split_qcd`) or `boosted_qcd_region` | `vbf_discriminator`; they also require `VBF_ggF_score` ≥ `ggf_vbf_threshold` |
| `boosted_qcd_A_region_postW`, `boosted_qcd_C_region_postW` (`..._postW_blind` with `blind`) | `vbf_discriminator`, `split_qcd`, `bkg_morphing_dnn` |

## Categories set by the config templates

Some templates replace the categories of `define_categories` for special productions:

| Template | Condition | Categories |
| --- | --- | --- |
| `HH4b_parton_matching_config.py` | no ONNX model, not `boosted`, and not the second approach with `run2` | `4b_region` only (SPANet training samples) |
| `VBF_HH4b_config.py` | always | `4b_region` is removed |
| `VBF_HH4b_config.py` | `BASELINE` | `baseline` (no cut) |
| `VBF_HH4b_config.py` | `SPANET_TRAINING` | `4b_region`; `4b_signal_region` with `SIGNAL_REGION`; with `VBF_REGIONS` also `vbf_best_candidates_6_jets_4b_region` ($m_{jj}$ > 400 GeV, $\Delta\eta$ > 3.5) and `vbf_best_candidates_6_jets_nokincut_4b_region` (no kinematic cut), and their `_signal_region` versions with `SIGNAL_REGION` |
| `VBF_HH4b_config.py` | `TRAIN_BKG_MORPHING` | `4b_control_region`, `2b_control_region_preW` (background morphing training) |
