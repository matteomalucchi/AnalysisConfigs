# HH4b synchronization

Configuration of the synchronization of the resolved HH4b analysis (second
approach of HIG-24-010, D_HH pairing, no models).

| File | Content |
| --- | --- |
| `HH4b_synchronization.py` | Configuration: parameters, preselection, columns of the synchronization tree. |
| `config_options.py` | Workflow options (`approach: "second"`, `run2`, `vbf_analysis`, ...), on top of `configs/HH4b_common/config_files/default_config.py`. |
| `workflow.py` | `SyncHH4bProcessor`: the VBF HH4b workflow, which also builds the `JetCuts` collection used by the preselection and computes the variables of the synchronization tree (`*_default` fields, `passes_jetvetomap`, `used_in_dHH`, `passJetVetoMap`, `ak4_h{1,2}b{1,2}_index`). The saved jet collections `JetSync` and `JetCuts` (the jets of the preselection, without pt threshold) are ordered by pt and have the same columns (`SYNC_JET_COLUMNS` in `HH4b_synchronization.py`). |
| `custom_cuts.py`, `custom_cut_functions.py` | Preselection: `hh4b_4jets_presel` (>= 4 `JetCuts` jets with \|η\| < 2.5) and `hh4b_4jetgood_presel` (>= 4 `JetGood` jets). |
| `params/object_preselection_synchronization.yaml` | Object preselection: `Jet` (`JetGood`, pt > 15 GeV), `JetVBF` (no pt threshold), `JetCuts` (preselection jets), `JetSync` (saved jets). |
| `params/jets_calibration_regression_json_synchronization.yaml` | Jet calibration. |
| `datasets/signal_ggF_HH4b_synchronization.json` | Synchronization sample. |

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
