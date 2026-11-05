# Jets and other physics objects

This page describes how the jets (and the other objects) used by the HH4b workflows are built,
from the calibrated NanoAOD collections to the Higgs and VBF candidates. The options mentioned
here are described in [config_options.md](config_options.md), the differences between the
approaches in [approaches.md](approaches.md).

## The `Jet` collection and the pT regression

The jet calibration (`params/jets_calibration_regression_json.yaml`) calibrates three copies of
the NanoAOD `Jet` collection, created by `define_jet_collections`:

| Collection | Jet type of the calibration | Content |
| --- | --- | --- |
| `JetDefault` | `AK4PFPuppi` | standard JEC (and JER on MC), no regression |
| `JetPtRegressed` | `AK4PFPuppiPNetRegression` (2022-2023), `AK4PFPuppiUParTRegression` (2024-2025) | pT regression without neutrinos, then JEC and JER |
| `JetPtRegressedPlusNeutrino` | `AK4PFPuppiPNetRegressionPlusNeutrino` (2022-2023), `AK4PFPuppiUParTRegressionPlusNeutrino` (2024-2025) | pT regression with neutrinos, then JEC and JER |

`apply_object_preselection` then rebuilds `Jet` jet by jet from these collections with
`merge_regressed_jets` (PocketCoffea). Whole jets are taken from one collection, so that every
pT-dependent field (pt, mass, systematic variations) stays consistent. A regressed jet with an
invalid regression (pt ≤ 0) falls back to the next collection of the chain, ending on
`JetDefault`:

| `approach` | Jets passing the loose b-tag WP | Other jets |
| --- | --- | --- |
| `first`, `boosted` (`separate_regression_by_btag: True`, default) | `JetPtRegressedPlusNeutrino` | `JetPtRegressed` |
| `first`, `boosted` (`separate_regression_by_btag: False`) | `JetPtRegressedPlusNeutrino` | `JetPtRegressedPlusNeutrino` |
| `second` | `JetPtRegressedPlusNeutrino` | `JetDefault` (no regression) |

Two options modify the regressed jets before the merge:

- `regression_min_pt_raw` (e.g. `15`): no regression for the jets with `JetDefault.pt_raw` (raw
  NanoAOD pt) not above the value. Their regressed pt is set to -1 in the collections passed to
  `merge_regressed_jets`, which then takes `JetDefault` for them.
- `regressed_jets_default_jer` (MC only): the regressed jets are smeared with the JER factor of
  `JetDefault` instead of their own one (also for the JER up/down variations). The JER scaling
  method compares the jet pt with the GenJet pt, which has no neutrinos, so on the +neutrino
  regressed pt it smears the jets upwards.

`Jet` also keeps the other pt definitions for bookkeeping: `pt_default` (`JetDefault.pt`),
`pt_regressed` (`JetPtRegressedPlusNeutrino.pt`, also for the jets that do not use it) and
`pt_raw_res` (the pT resolution estimated by the regression of the year).

## Year-dependent jet algorithms

The b-tagging and the pT-regression algorithms are not the same in every year
(ParticleNet up to 2023, UParTAK4 from 2024), so the workflows never read a
NanoAOD branch of a specific algorithm. They read which algorithm the year uses
from the parameters and copy its discriminants onto algorithm-independent
fields, defined in `utils_configs/jet_algorithms.py`:

| Field | Content |
| --- | --- |
| `btagB` | b-tag discriminant of `btagging.working_point.<year>.btagging_algorithm`, the algorithm used to order the jets, to build the working points (`btagB_3wp`, `btagB_5wp`, `btagB_delta5wp`) and to split the pT regression |
| `btagCvL`, `btagCvB`, `btagQvG` | the other discriminants of the same algorithm, when the input has them |
| `pt_raw_res` | pT resolution estimated by the regression that calibrates `JetPtRegressed` in that year |

These are the names to use in the columns and in the model inputs: the same
configuration then runs on every year, and the working points are read from
`btagging.working_point.<year>.btagging_WP` instead of being hardcoded.

## Selected jets and Higgs candidates

| Collection | Content |
| --- | --- |
| `JetGood` | `Jet` passing the `Jet` object preselection (`params/object_preselection_<approach>_approach.yaml`; pt cut on `preselection_pt_field`, `pt_tight` with `tight_cuts`), ordered by `btagB`. With `fifth_jet: "pt"` the jets after the 4th are re-ordered by pt. |
| `JetGoodHiggs` | The 4 `JetGood` jets leading in b-tag score: the input of the $D_{HH}$ pairing. |
| `JetGoodMatched`, `JetGoodHiggsMatched` | MC: `JetGood` / `JetGoodHiggs` with the jets not matched to a Higgs (or Z) b-quark masked (`provenance_X`). |
| `HiggsLeading`, `HiggsSubLeading` | Higgs candidates of the pairing (SPANet or $D_{HH}$), with the DNN variables (`dEta`, `dPhi`, ...). The leading candidate is the one with the higher pt. |
| `JetGoodFromHiggsOrdered` | The 4 jets of the pairing: the 2 jets of `HiggsLeading`, then the 2 of `HiggsSubLeading` (also split into `JetGoodFromHiggsOrderedLeading` / `...SubLeading`). |
| `HH` | `HiggsLeading + HiggsSubLeading`. |
| `JetNotFromHiggs` | The `JetGood` jets not taken by the Higgs pairing (nor by the VBF pairing, in the VBF analysis). |
| `add_jet1pt` | Leading additional jet (from `JetNotFromHiggs`), ordered by b-tag score if the pairings left over one of the 4 jets leading in b-tag score (`btag_order_add_jet`), by pt otherwise. `JetGoodFromHiggsOrdered5Jets` is `JetGoodFromHiggsOrdered` followed by it. |
| `ElectronGood`, `MuonGood` | Leptons passing the `Electron` / `Muon` object preselection, used for the lepton veto of the event preselection and for the cleaning of the jets. |

Event variables built from these collections: `HT` (scalar sum of the `JetGood` pt, only of the
jets above 30 GeV in the second approach, `SECOND_APPROACH_HT_MIN_PT`), `Rhh` (distance of
($M_{H1}$, $M_{H2}$) from (125, 120) GeV, used by the first approach regions), `delta_dhh`
($D_{HH}$ pairing), `dR_min` / `dR_max`, `sigma_over_higgs{1,2}_reco_mass`.

## VBF jet collections

Built with `vbf_analysis`. Which collection holds "the" VBF jets (used by the
VBF categories, saved in the output and to be used in any study outside the
workflow) depends on what the workflow does (`get_vbf_jet_collection` in
`configs/HH4b_common/vbf_jet_collections.py`):

| Mode | VBF jets |
| --- | --- |
| Creating the SPANet training inputs (no `spanet`, no `run2`) | the first two jets of `JetGoodVBFMergedProvVBFPadded` (with `mjjJetGoodVBFMergedProvVBFPadded`, ...) |
| Applying a pairing (`spanet` or `run2`) | `JetGoodVBFSelectedPair` (with `mjjJetGoodVBFSelectedPair`, ...) |
| Boosted analysis | `JetGoodVBFEnergyOrdered` |

| Collection | Content |
| --- | --- |
| `JetGoodClip`, `JetGoodPadded` | The `max_num_jets_good` jets of `JetGood` leading in b-tag score (`JetGoodPadded`: padded to that number). |
| `JetGoodVBFCandidates` | Jets passing the `JetVBF` object preselection which are not among the `max_num_jets_good` jets leading in b-tag score (or, when a pairing is applied, not taken by the Higgs pairing), ordered by pt. |
| `JetGoodVBFLeadingMjj` | Pair of candidates leading in $m_{jj}$, ordered by energy. Built only when it is the pair of `spanet_vbf_candidates` (automatic: first approach). |
| `JetGoodVBFLeadingPtOppositeEta` | Leading-pt candidate and leading-pt candidate with $\eta_1 \eta_2 < 0$ (padded with `None`), ordered by pt. Built only when it is the pair of `spanet_vbf_candidates` (automatic: second approach). |
| `JetAdditionalGoodVBF` | Candidates left over by the pair chosen with `spanet_vbf_candidates`, ordered by `jets_add_vbf_order` and padded to `max_num_jets_add_vbf`. |
| `JetGoodVBFMergedProvVBFPadded` | That pair followed by `JetAdditionalGoodVBF`, with `provenance = provenance_vbf`: the VBF part of the SPANet inputs and of the ggF/VBF discriminator inputs. When a pairing is applied it is rebuilt out of the candidates left over by the Higgs pairing. |
| `JetGoodProvHiggsPadded` | The Higgs jets with `provenance = provenance_X` (the `max_num_jets_good` jets leading in b-tag score, or the jets of the Higgs pairing when one is applied). |
| `JetTotalSPANetPadded` | `JetGoodPadded` (the `max_num_jets_good` jets leading in b-tag score), followed by the pair chosen with `spanet_vbf_candidates` and by `JetAdditionalGoodVBF`: the SPANet input. |
| `JetGoodVBFSelectedPair` | Only when a pairing (`spanet` or `run2`) is applied: the VBF pair, ordered by pt. It is the one predicted by the ggF/VBF discriminator when it predicts the VBF jets (`vbf_pairing_from_vbf_discriminator`), otherwise the one predicted by the Higgs pairing model when it predicts the VBF jets, otherwise, out of the candidates left over by the Higgs pairing, the pair of `spanet_vbf_candidates` (automatic: `leading_pt_opposite_eta` in the second approach, `leading_mjj` otherwise), i.e. the same definition as the pair fed to the models. Saved with `VBF_SELECTED_PAIR_COLUMNS`. |
| `JetGoodVBFEnergyOrdered` | Boosted analysis only: the pair of candidates leading in $m_{jj}$, ordered by energy. |

The VBF variables (`mjj`, `deta`, centrality of the Higgs candidates) are computed on the VBF
jets of the mode (`mjj<collection>`, `deta<collection>`, `centralityHiggs{Leading,SubLeading}<collection>`).

## Jets mitigations

`params/jets_calibration_regression_json.yaml` switches on the PocketCoffea Run3 jets
mitigations (`jets_calibration.jets_mitigation`):

- selection mitigations, applied by PocketCoffea to every AK4 jet selection (also the ones made
  with `custom_jet_selection`, on its `pt_type` pt), never to the FatJets: `pt_cut_HE`
  (pt > 50 GeV in 2.5 < \|η\| < 3.0, also in 2025) and `raw_factor_cut`;
- calibration mitigation `jer_genmatched_only`, applied to the jet types of `jet_types`:
  `AK4PFPuppi` (also the `CorrT1METJet` jets, aliased to it) and the pt-regressed jet types
  (`AK4PFPuppiPNetRegression[PlusNeutrino]`, `AK4PFPuppiUParTRegression[PlusNeutrino]`), i.e.
  the calibration of `JetDefault`, `JetPtRegressed` and `JetPtRegressedPlusNeutrino`.

The synchronization config uses its own copy of the file, with `pt_cut_HE` switched off.

## Jet veto map

The jet veto maps are applied in the event preselection (`hh4b_JetVetoMap`), after the
calibration, on the standard JEC pt `pt_default`, or, with `jet_veto_map_regressed_pt`, on the
pt of `Jet` (`hh4b_JetVetoMap_regressed`, the regressed pt where the regression is applied).
They are not applied with `boosted_presel` and `mixeddata`.

## FatJets (boosted analysis)

With `boosted`, `FatJetGood` is the `FatJet` collection with the generic b-tag fields (`btagBB`,
`btagBBTXbb`, ...), ordered by `btagBB` (`btagBBTXbb` with `TXbb_order`), with the regressed
mass and pt (`mass_regr`, `pt_regr`, from `particleNet_massCorr`). `FatJetGoodSelected` is the
leading `FatJetGood` jet passing the `FatJetLeading` object preselection followed by the leading
other jet passing `FatJetSubLeading` (pt, \|η\|, jet ID, soft-drop and regressed mass). The jets
mitigations are never applied to the FatJets.
