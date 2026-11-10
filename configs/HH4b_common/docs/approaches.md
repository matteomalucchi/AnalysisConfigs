# First and second approach

The resolved HH4b analysis can be run with two different selections, following the two
approaches of HIG-24-010. The approach is chosen with the `approach` option (`"first"`, default,
or `"second"`; `"boosted"` is used for the boosted analysis, see
[config_options.md](config_options.md#analysis-flavour)). This page lists what the two approaches
share and where they differ. The objects are described in [jets.md](jets.md), the categories in
[categories.md](categories.md).

## Common points

- **Workflow and calibration**: same workflows (`HH4bCommonProcessor` and its VBF subclass),
  same jet calibration (`params/jets_calibration_regression_json.yaml`, with the
  `JetDefault`, `JetPtRegressed` and `JetPtRegressedPlusNeutrino` collections merged into `Jet`),
  same Run3 jets mitigations (`pt_cut_HE`, `jer_genmatched_only`, `raw_factor_cut`, see
  [jets.md](jets.md#jets-mitigations)) and same year-dependent b-tagging and pT-regression
  algorithms (ParticleNet in 2022-2023, UParTAK4 in 2024-2025, read through the generic `btagB`
  field).
- **Jet ordering**: `JetGood` ordered by b-tag score, `JetGoodHiggs` = the 4 jets leading in
  b-tag score.
- **Skim**: noise filters, golden JSON, ≥ 1 good primary vertex and the HLT paths of
  `params/triggers.yaml`, the same in both approaches:

  | Year | HLT path |
  | --- | --- |
  | 2022 | `QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65` |
  | 2023 preBPix | `QuadPFJet70_50_40_35_PNet2BTagMean0p65` (JetMET) or `PFHT280_QuadPFJet30_PNet2BTagMean0p55` (ParkingHH) |
  | 2023 postBPix | `PFHT280_QuadPFJet30_PNet2BTagMean0p55` |
  | 2024, 2025 | `PFHT250_QuadPFJet25_PNet2BTagMean0p55` |

- **Event preselection**: lepton veto, ≥ 4 jets with year-dependent pt thresholds and a
  b-tag requirement, and the jet veto maps (on `pt_default`, or on the pt of `Jet` with
  `jet_veto_map_regressed_pt`).
- **Higgs pairing**: SPANet (`spanet`) or the Run-2 $D_{HH}$ algorithm (`run2`) on
  `JetGoodHiggs`, with the same code (`run2_matching_algorithm`): in each pairing the leading
  Higgs candidate is the one with the higher pt, and among the pairings with $D_{HH}$ within 30
  GeV of the best one the tie-break chooses the highest pt (the variable differs, see below).
- **2025**: treated as 2024 in both approaches: same HLT paths (and L1 seeds in the first
  approach), same object and event preselection, same b-tag categories and the same `pt_cut_HE`
  mitigation (extended to 2025 in `params/jets_calibration_regression_json.yaml`).
- **Models and options**: all the models (SPANet, ggF/VBF discriminator, background morphing,
  signal/background DNN) and the other options can be used with both approaches.
- **VBF and ggF categories**: the same split of every inclusive category into `vbf_<name>` and
  `ggf_<name>` (`define_vbf_ggf_categories`, see
  [categories.md](categories.md#vbf-and-ggf-categories)), with the same VBF pair from SPANet
  when the model predicts it.
- **Background estimate**: a DNN reweighting a lower-b-tag region to the signal b-tag region,
  trained in the control region and applied in the signal region (`preW`/`postW` categories);
  only the regions change.

## Differences

| | First approach (`"first"`) | Second approach (`"second"`) |
| --- | --- | --- |
| Object preselection | `params/object_preselection_first_approach.yaml` | `params/object_preselection_second_approach.yaml` |
| `JetGood` | pt > 25 GeV (35 GeV with `tight_cuts`), \|η\| < 2.5, tight ID with lepton veto | pt > 30 GeV (25 GeV in 2023, given per year as `{default: ..., <year>: ...}`), \|η\| < 2.5, tight ID |
| Leptons (veto) | electrons pt > 10 GeV, \|η\| < 2.5, cut-based loose ID, \|dxy\| < 0.1, \|dz\| < 0.2; muons pt > 10 GeV, \|η\| < 2.4, `looseId`, loose PF isolation, \|dxy\| < 0.2, \|dz\| < 0.5 | electrons pt > 15 GeV, \|η\| < 2.5, `mvaIso_WP90`, `pfRelIso03_all` < 0.15; muons pt > 10 GeV, \|η\| < 2.4, `looseId`, `pfRelIso04_all` < 0.15; both \|dz\| < 0.1, \|dxy\| < 0.05 for \|η\| < 1.479 and \|dz\| < 0.2, \|dxy\| < 0.1 above |
| pT regression | +neutrino regression for the jets passing the loose WP, regression without neutrinos for the others (`separate_regression_by_btag`) | +neutrino regression for the jets passing the loose WP, standard JEC for the others |
| Skim | HLT and L1 seeds (`get_L1sel`, unless `noL1`) | HLT only, no L1 seed requirement |
| Event preselection | `hh4b_presel_era_dependent` (default, `era_dependent_presel`): **A** (2022, 2023 preBPix JetMET data) pt > 80, 60, 45, 35 GeV and mean b-tag score of the 2 leading b-tag jets > 0.65; **B** (2023 preBPix ParkingHH and MC, 2023 postBPix) pt > 35, 35, 35, 30 GeV and mean > 0.55; **C** (2024, 2025) pt > 30 GeV and ≥ 2 jets passing the UParT medium WP. Without `era_dependent_presel`: A in every year (`hh4b_presel`) | `hh4b_presel_second_approach`: ≥ 4 `JetGood` with pt > 70, 50, 40, 35 GeV (2022) or 30 GeV (2023-2025), ≥ 2 `JetGood` with pt > 30 GeV passing the medium WP of the year's tagger, mean of the 2 highest b-tag scores > 0.65 (2022) / 0.55 (2023), no cut in 2024-2025 |
| HT | sum of all the `JetGood` pt | sum of the `JetGood` pt above 30 GeV (`SECOND_APPROACH_HT_MIN_PT`) |
| $D_{HH}$ tie-break | highest lab-frame pt of the two Higgs candidates | pt of the Higgs candidates in the HH rest frame (as in HIG-20-005) |
| VBF candidates (`JetVBF`) | pt > 25 GeV, \|η\| < 4.7, tight ID with lepton veto, no b-tag requirement | pt > 30 GeV, \|η\| < 4.7, tight ID, not passing the medium WP of the year's tagger (`max_btag_wp: M`) |
| VBF pair (without a SPANet prediction) | the pair leading in $m_{jj}$ | the leading-pt jet and the leading-pt jet with $\eta_1 \eta_2 < 0$, if present |
| VBF selection | ≥ 2 VBF jets, $m_{jj}$ > 400 GeV, $\Delta\eta$ > 3.5 | ≥ 2 VBF jets, no $m_{jj}$/$\Delta\eta$ cut |
| Mass plane | circular: SR $R_{HH}$ < 30 GeV, CR 30 ≤ $R_{HH}$ < 55 GeV (80 GeV with `expandCR`), around (125, 120) GeV | SR = $X_{HH}$ < 30 GeV ∪ $X_{ZZ}$ < 2.6 ∪ $X_{ZH}$ < 1.9 ∪ $X_{HZ}$ < 1.9; CR = (52 < $M_{H1}$ < 180 GeV) ∩ (50 < $M_{H2}$ < 173 GeV) outside the SR |
| b-tag categories | `4b` / `2b`: 3rd and 4th b-tag jets above / below 0.2605 | `3T1M`, `3T1L`, `2M`, `3MnT`, and `3M`, `4M` in 2024-2025, with the WPs of the year's tagger |
| Signal regions | `4b_signal_region` | `SR3T1M`, `SR3T1L` (2022-2023), `SR4M` (2024-2025) |
| Background estimate | 2b → 4b: `2b_*_preW` / `2b_*_postW` | 2M → 3T1M: training CR2M → CR3T1M, weights applied in `SR2M_postW` / `CR2M_postW`; 3MnT for the training of SR3T1L |
| VBF and ggF inclusive categories | `4b_signal_region`, `4b_control_region` (+ `2b_*_postW`) | `SR/CR3T1M`, `SR/CR3T1L` (2022-2023), `SR/CR4M` (2024-2025), `SR/CR2M` (+ `_postW`) |
| Validation regions | `vr1` (mass plane centred at (185, 180) GeV) | `SR3MnT` |

Not implemented in the second approach: the offline-online trigger object matching (2022-2023)
and the b-tag SFs of the second approach. The JER smearing factor of the standard-JEC jets applied on top of the
regressed jets is the `regressed_jets_default_jer` option (off by default: the regressed
collections are then smeared with their own JER).

The synchronization config ([`configs/HH4b_synchronization`](../../HH4b_synchronization/README.md))
runs the second approach with the $D_{HH}$ pairing and the VBF analysis, without models.
