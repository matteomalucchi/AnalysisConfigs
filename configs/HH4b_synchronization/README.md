# HH4b synchronization

Configuration of the synchronization of the resolved HH4b analysis (second
approach of HIG-24-010, D_HH pairing, no models).

| File | Content |
| --- | --- |
| `HH4b_synchronization.py` | Configuration: parameters, preselection, columns of the synchronization tree. |
| `config_options.py` | Workflow options (`approach: "second"`, `run2`, `vbf_analysis`, ...), on top of `configs/HH4b_common/config_files/default_config.py`. |
| `workflow.py` | `SyncHH4bProcessor`: the VBF HH4b workflow, which also builds the `JetCuts` collection used by the preselection and computes the variables of the synchronization tree (`*_default` fields, `passes_jetvetomap`, `used_in_dHH` (jet among the 4 `JetGoodFromHiggsOrdered` jets of the D_HH pairing), `passJetVetoMap`, `ak4_h{1,2}b{1,2}_index`). The saved jet collections `JetSync`, `JetSyncNoPt` (as `JetSync` without pt threshold) and `JetCuts` (the jets of the preselection, without pt threshold) are ordered by pt and have the same columns (`SYNC_JET_COLUMNS` in `HH4b_synchronization.py`, including the MC truth `hadronFlavour`, `partonFlavour`, `genJetIdx` (index of the matched GenJet, -1 if none) and `provenance_higgs`). |
| `custom_cuts.py`, `custom_cut_functions.py` | Preselection: `hh4b_4jets_presel` (>= 4 `JetCuts` jets with \|η\| < 2.5) and `hh4b_4jetgood_presel` (>= 4 `JetGood` jets). |
| `params/object_preselection_synchronization.yaml` | Object preselection: `Jet` (`JetGood`, pt > 15 GeV), `JetVBF` (no pt threshold), `JetCuts` (preselection jets), `JetSync` and `JetSyncNoPt` (saved jets, with and without pt > 15 GeV). |
| `params/jets_calibration_regression_json_synchronization.yaml` | Jet calibration. |
| `datasets/signal_ggF_HH4b_synchronization.json` | Synchronization sample. |

**JER of 2023 postBPix for 2024.** `jets_calibration.jer_override` in
`params/jets_calibration_regression_json_synchronization.yaml` gives, per year and
jet type, the JER to use instead of the one of the year: for 2024 the 2023
postBPix JER for the standard jets (`AK4PFPuppi`, `JetDefault`) and for the
UParT regressed jets (`AK4PFPuppiUParTRegression` and `...PlusNeutrino`, with
their own JER from the 2023 postBPix regression file), all with the
`Summer23BPixPrompt23_RunD_JRV3_MC` JER. The synchronization workflow
(`SyncHH4bProcessor.apply_jer_override`, MC only) smears again these collections
from their JEC-corrected pt (`pt_raw * jec_factor`, the 2024 JEC), with the same
gen matching and `jer_genmatched_only` mitigation as the jet calibration, before
they are merged into `Jet`. Only the nominal JER is replaced; the normal workflow
is unchanged.

The jet veto map variables use the standard JEC pt (`pt_default`), or the pt of
the `Jet` collection (the regressed pt where the regression is applied) with
`jet_veto_map_regressed_pt: True` in `config_options.py`.

## Run

The configuration does not use the `__config_file__` placeholder, so it can be
run directly:

```bash
pocket-coffea run --cfg configs/HH4b_synchronization/HH4b_synchronization.py \
    --custom-run-options params/t3_run_options_xxsmall.yaml \
    -o <output_dir> --test
```
