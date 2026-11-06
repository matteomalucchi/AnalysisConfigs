# Configuration options

The analysis is steered by two dictionaries, `onnx_model_dict` and
`config_options_dict`, defined in a small config file inside
[`config_files/`](../config_files/) (e.g. [`spanet_ptflat.py`](../config_files/spanet_ptflat.py)).
Every such file starts from the defaults in
[`config_files/default_config.py`](../config_files/default_config.py) and overrides
only the entries it needs:

```python
import configs.HH4b_common.dnn_input_variables as dnn_vars
from configs.HH4b_common.config_files.default_config import (
    default_onnx_model_dict as onnx_model_dict,
    default_config_options_dict as config_options_dict,
)

onnx_model_dict |= {
    "spanet": "/path/to/spanet_model.onnx",
    "bkg_morphing_dnn": "/path/to/morphing_model.onnx",
}

config_options_dict |= {
    "run2": False,
    "spanet_input_name": dnn_vars.pairing_spanet_btagWP5,
    # ...
} | onnx_model_dict          # <-- always re-merge the models at the end
```

> [!IMPORTANT]
> The `| onnx_model_dict` at the end is mandatory: the model paths are part of
> `config_options_dict` and are read by the workflow as `self.spanet`,
> `self.bkg_morphing_dnn`, etc.

The name of the config file is the first argument of `run_pocket_coffea`:

```bash
run_pocket_coffea <config_name> <config_file> <run_options> <output_dir>
# e.g.
run_pocket_coffea spanet_ptflat HH4b_parton_matching_config.py params/t3_run_options.yaml /work/out_hh4b/test
```

The wrapper replaces the `__config_file__` placeholder in the config template
(`configs/HH4b/HH4b_parton_matching_config.py`, `configs/VBF_HH4b/VBF_HH4b_config.py`, ...)
with `<config_name>`, so the same template can be run with many different option sets.
Its additional options (`--test`, `--copy-to`, `--merge-parquet`, `--merge-parquet-update-config`, ...)
are listed in the [README](../README.md#run-analysis).

`config_options_dict` is passed to the PocketCoffea `Configurator` as
`workflow_options`. `HH4bCommonProcessor.__init__` turns every key into an
attribute of the processor, so an option `"foo"` is read inside the workflows as
`self.foo`. The same dictionary is also read directly by the config templates to
build the categories, the columns, the skim and the preselection.

The tables below describe all the options available in
[`default_config.py`](../config_files/default_config.py).

## ONNX models (`onnx_model_dict`)

All models are optional: an empty string means "do not run this model".
The presence or absence of a model also drives which categories and columns are
created — if no model at all is given (and `boosted` is `False`), only the
`4b_region` used to produce the SPANet training samples is defined.

| Option | Default | Description |
| --- | --- | --- |
| `spanet` | `""` | Jet-to-Higgs pairing model. If set, the pairing comes from SPANet instead of the Run-2 $D_{HH}$ algorithm or the gen-level truth matching. It also fills the pairing-probability variables (`Delta_pairing_probabilities`, `Arctanh_Delta_pairing_probabilities`, ...). |
| `vbf_discriminator` | `""` | ggF vs VBF classifier, producing `VBF_ggF_score`. It can be the *same file* as `spanet` (the class-probability output of the pairing model is used) or a standalone model, in which case `vbf_discriminator_input_variables` and `max_num_jets_vbf_discriminator` must also be set. Requires `vbf_analysis=True`. |
| `bkg_morphing_dnn` | `""` | DNN reweighting the 2b (or mixed) data to the 4b data. Produces the per-event `bkg_morphing_dnn_weight`, applied as a weight in the `*_postW` categories, and enables the creation of those categories. |
| `sig_bkg_dnn` | `""` | Signal vs background classifier, producing `sig_bkg_dnn_score`. It enables the score histograms, the `high_score` categories and the blinding cut. |
| `bkg_morphing_spread_dnn` | `""` | Ensemble of morphing models used to estimate the spread (systematic uncertainty) of the morphing weights. Saves `bkg_morphing_spread_dnn_weights` for data in the `postW` categories. Not implemented for the boosted analysis. |

## Analysis flavour

What `approach` changes in the selection, the pairing and the categories is described in [approaches.md](approaches.md).

| Option | Default | Description |
| --- | --- | --- |
| `approach` | `"first"` | Jet pt definition and object preselection. Selects the `params/object_preselection_<approach>_approach.yaml` file and how the `Jet` collection is built in `apply_object_preselection`. `"first"`: HIG-24-010 first approach, with the regressed pt split by b-tag (`separate_regression_by_btag`). `"second"`: the second approach (HIG-24-010), see [First and second approach](approaches.md): regressed pt for the jets passing the loose b-tag WP, its own object and event preselection, no L1 seeds in the skim. `"boosted"`: skip the resolved jet-pt handling and the b-tag WP definition, and drop the HLT selection from the skim. |
| `boosted` | `False` | Run the boosted (AK8 / `FatJet`) analysis: builds the `FatJetGood` collection, the boosted categories, the boosted DNN variables and the `FatJetGood` columns. |
| `run2` | `False` | Use the Run-2 $D_{HH}$ pairing algorithm instead of SPANet. Cannot be combined with `random_pt` (there is an explicit `assert` in the config templates). |
| `vbf_analysis` | `False` | Run the VBF part of the analysis: builds the VBF jet collections (`JetGoodVBF*`, `JetAdditionalGoodVBF`, the merged `JetTotalSPANet*` collections), the VBF variables (`mjj`, `deta`, centrality) and the VBF categories. The VBF jets used by the categories and saved in the output are `JetGoodVBFMergedProvVBFPadded` (first two jets) when creating the SPANet training inputs and `JetGoodVBFSelectedPair` when `spanet` or `run2` is set (see [jets.md](jets.md#vbf-jet-collections)). |
| `mixeddata` | `False` | Run on the mixed-data samples (data-driven background model) instead of the 2b data. The `2b_*_preW`/`postW` categories become `4b_*_preW`/`postW`, and the HLT/L1 skim cuts and the jet calibration are switched off. |

## Skim and preselection

| Option | Default | Description |
| --- | --- | --- |
| `tight_cuts` | `False` | Apply the tight jet pt cuts (`pt_tight` in the object preselection) and require the pt preselection on the 4 b-tag-ordered Higgs candidate jets (`JetGoodHiggs`) instead of all `JetGood`. |
| `vbf_presel` | `False` | Use the VBF preselection. **Not supported anymore**: `define_preselection` raises a `ValueError` if this is `True`, because the cut acts on the wrong jet collection. |
| `boosted_presel` | `False` | Use the boosted preselection (at least 2 FatJets) instead of the resolved one. It also disables the jet veto map cut. |
| `preselection_pt_field` | `"pt"` | Jet pt field used by the jet object preselection (`select_jets`) and by the jet pt cuts of the resolved event preselection. `"pt"` is the pt of the `Jet` collection, i.e. the regressed pt when the pt regression is applied (see `approach`); `"pt_default"` is the standard JEC pt without regression. The config files written before this option existed set it to `"pt_default"` to keep their old selection. The jet veto map always uses `"pt_default"`. |
| `era_dependent_presel` | `True` | Choose the resolved preselection (`hh4b_presel_era_dependent`) per chunk from its year and primary dataset, so that several years can be processed in one go. **A** (2022 MC and data, 2023 preBPix JetMET data): jet $p_T$ > 80, 60, 45, 35 GeV and mean PNet score of the 2 highest-score jets > 0.65. **B** (2023 preBPix ParkingHH data and MC, 2023 postBPix MC and data): jet $p_T$ > 35, 35, 35, 30 GeV and mean PNet score > 0.55. **C** (2024 and 2025 MC and data): jet $p_T$ > 30, 30, 30, 30 GeV and at least 2 jets passing the UParT medium WP. Data samples are treated as ParkingHH when their name contains `ParkingHH`, as JetMET otherwise. The $\|\eta\|$ < 2.5 cut is in the object preselection. Ignored if `no_btag` or `boosted_presel` is set. The config files written before this option existed set it to `False`, so they keep the old preselection (`hh4b_presel`, i.e. A for all the years) and stay reproducible. |
| `no_btag` | `False` | Drop the b-tag requirement from the preselection (`hh4b_presel_nobtag`). Needed to measure the b-tag WP efficiencies, which has to be done in a region where no cut on the b-tag score is applied. The configs in `configs/HH4b_btagging` turn it on at the call site with `define_preselection(config_options_dict | {"no_btag": True})`. |
| `semi_tight_vbf` | `True` | Legacy flag for the semi-tight VBF jet selection. It is only accepted as an argument of `jet_selection_nopu` and is currently not used by any workflow. |
| `noL1` | `False` | Drop the L1 seed requirement (`get_L1sel`) from the skim. Needed for the samples/eras for which the L1 emulation is not available. |
| `jet_veto_map_regressed_pt` | `False` | Apply the jet veto map on the pt of the `Jet` collection, i.e. the regressed pt where the pt regression is applied (see `approach`), instead of the standard JEC pt `pt_default` (`hh4b_JetVetoMap_regressed` instead of `hh4b_JetVetoMap`). The jet veto map function of PocketCoffea is unchanged: the wrapper `get_custom_JetVetoMap_Mask` only replaces the pt of the jets. |
| `xx4b_presel` | `False` | Apply a gen-level cut selecting $X(\to b\bar{b})X(\to b\bar{b})$ decays (X = H or Z), adding `cuts.XX4b_presel` to the preselection. Needed for the private ZZ/ZH samples, which are inclusive in the decay channel. Off by default because it changes the yields of every sample whose name contains "HH", "ZH" or "ZZ". |

## Truth matching (MC only)

| Option | Default | Description |
| --- | --- | --- |
| `which_bquark` | `"last"` | Which copy of the b-quarks from $H\to b\bar{b}$ is matched to the jets. `"first"`: the direct children of the Higgs. `"last"`: the last copy, found by walking up the decay chain. `"last_numba"`: the last copy, found with the numba helper `get_parton_last_copy`. `"last_numba_with_status"`: same, but the b-quarks are selected requiring `status == 23`. |
| `parton_jet_min_dR` | `0.4` | Maximum $\Delta R$ between a parton and a jet for the matching to succeed, both for the Higgs b-quarks and for the VBF quarks. |

## VBF

The VBF jet collections are described in [jets.md](jets.md#vbf-jet-collections), the VBF and ggF categories in [categories.md](categories.md#vbf-and-ggf-categories).

| Option | Default | Description |
| --- | --- | --- |
| `vbf_parton_matching` | `False` | Run the gen-level matching of the VBF quarks to the jets (VBF MC only) and fill the `provenance_vbf` field. When `False`, a dummy (all-`None`) `provenance_vbf` is created. |
| `which_vbf_quark` | `"with_mothers_children"` | How the VBF quarks are identified at gen level. `"with_mothers_children"`: hard-process quarks whose mother also has two Higgs children. `"with_status"`: outgoing (`status == 23`) non-b partons. |
| `max_num_jets_add_vbf` | `2` | Number of *additional* VBF jet candidates (on top of the pair chosen with `spanet_vbf_candidates`) kept in `JetAdditionalGoodVBF` and merged into the SPANet input collections. |
| `jets_add_vbf_order` | `"energy"` | Field used to order the additional VBF jets, e.g. `"energy"` or `"pt"`. |
| `spanet_vbf_candidates` | `None` | VBF pair fed to SPANet (`JetTotalSPANetPadded`) and to the ggF/VBF discriminator (`JetGoodVBFMergedProvVBFPadded`) before the additional VBF jets; also the VBF pair of the categories (`JetGoodVBFSelectedPair`) when no model predicts it, so the two always agree. `None`: automatic, `"leading_pt_opposite_eta"` in the second approach and `"leading_mjj"` otherwise. `"leading_mjj"`: the pair leading in $m_{jj}$. `"leading_pt_opposite_eta"`: the leading-pt jet and the leading-pt jet with $\eta_1 \eta_2 < 0$. Set it explicitly only to apply a model trained with the other definition. |
| `vbf_matching_after_higgs_pairing` | `False` | Run the SPANet Higgs pairing first and define the VBF candidates from the jets left over by the pairing, instead of from the b-tag-ordered `JetGoodClip` collection. Requires `spanet`. |
| `vbf_pairing_from_vbf_discriminator` | `True` | Take the VBF pair (`JetGoodVBFSelectedPair`, only built when `spanet` or `run2` is set) from the SPANet model given in `vbf_discriminator` instead of using the leading-mjj pair. Requires a standalone model (different file from `spanet`) predicting the VBF jet assignment (2 jets) on top of the ggF/VBF classification, with its sequential inputs read from the VBF candidate collection. The model is run only once and also provides `VBF_ggF_score`. |
| `ggf_vbf_threshold` | `0.95` | Threshold on the ggF-vs-VBF discriminator score (`VBF_ggF_score`) used in the VBF categories. Only relevant if `vbf_discriminator` is set. See [VBF and ggF categories](categories.md#vbf-and-ggf-categories). |
| `vbf_selection` | `None` | Boosted VBF only: overrides `vbf_analysis` when the categories are defined in [`VBF_HH4b_boosted_config.py`](../../VBF_HH4b_boosted/VBF_HH4b_boosted_config.py), so that the VBF regions can be built without switching on the full VBF jet reconstruction in the workflow. `None` means "not set", i.e. `vbf_analysis` is used instead. |

## Fox-Wolfram momenta

| Option | Default | Description |
| --- | --- | --- |
| `max_order_FW` | `0` | Maximum order $l$ of the Fox-Wolfram moments computed on `JetGood`. `0` disables the computation; otherwise the columns `FW_H{i}_{norm}` and `FW_R{i}_{norm}` are added to `events` for `i in range(max_order_FW)`. |
| `FW_momenta_norms` | `["W_T"]` | List of normalisation schemes used for the Fox-Wolfram moments. One set of columns is produced per scheme. |

## Jet collections and pairing

The jet collections are described in [jets.md](jets.md).

| Option | Default | Description |
| --- | --- | --- |
| `max_num_jets_good` | `5` | Number of `JetGood` jets kept for the VBF analysis (`JetGoodClip`/`JetGoodPadded`). It is also the offset at which the VBF pair starts inside the merged SPANet collections. |
| `max_num_jets_higgs_pairing` | `5` | Number of jets given to the SPANet pairing model. The true pairing is truncated to this number when computing the pairing efficiency. |
| `max_num_jets_vbf_discriminator` | `None` | Number of jets given to a standalone ggF/VBF discriminator model. `None` means "use all the jets of the input collection". |
| `max_num_jets_spanet_class` | `4` | Number of jets given to a SPANet-format signal-vs-background classifier. Only used when `sig_bkg_dnn` is a SPANet-format model. |
| `fifth_jet` | `"pt"` | Ordering of the jets beyond the 4 Higgs candidates. `"pt"` re-sorts the 5th and following jets by pt (the first 4 stay b-tag ordered); any other value keeps the pure b-tag ordering. |
| `add_jet_spanet` | `False` | Sort the `Jet` collection by regressed pt before the good-jet selection, and order the additional jet collection (`JetNotFromHiggs`) by b-tag score or pt depending on whether the Higgs pairing (and, when `vbf_analysis` is on, the VBF pairing) took all the 4 jets leading in b-tag score. |
| `old_wp_def` | `False` | Use the old b-tag working-point convention, where the WP index starts at `-1` (no WP passed) instead of `0`. It must match the convention used to train the SPANet/DNN models. |
| `TXbb_order` | `False` | Boosted only: order the `FatJetGood` collection by `btagBBTXbb` instead of `btagBB`. |
| `only5jetsbSF` | `False` | b-tag SF studies only (`configs/HH4b_btagging`): compute the b-tag scale factors using only the 5 leading jets instead of all the `JetGood`. |
| `separate_regression_by_btag` | `True` | First and boosted approaches: split the pT regression by b-tag, with the +neutrino regression for the jets passing the loose WP and the regression without neutrinos for the others, each falling back to the standard JEC jets where the regression is not valid. `False` is the old behaviour: the +neutrino regression wherever valid, else the standard JEC jets. Not used in the second approach (see [jets.md](jets.md#the-jet-collection-and-the-pt-regression)). |
| `regressed_jets_default_jer` | `False` | MC only: smear the pT-regressed jets (`JetPtRegressed`, `JetPtRegressedPlusNeutrino`) with the JER factor of the standard JEC jets (`JetDefault`) instead of their own one (also for the JER up/down variations). The JER scaling method compares the jet pt with the GenJet pt, which has no neutrinos, so on the +neutrino regressed pt it smears the jets upwards. |
| `regression_min_pt_raw` | `None` | Apply the pT regression only to the jets with `pt_raw` (raw pt of `JetDefault`) above this value in GeV, e.g. `15`, for every `approach`. The regressed pt (`JetPtRegressed`, `JetPtRegressedPlusNeutrino`) of the other jets is set to -1 (failed regression, pt <= 0), so `merge_regressed_jets` takes the standard JEC jets for them. `None` applies the regression to every jet. |

## Model inputs and padding

| Option | Default | Description |
| --- | --- | --- |
| `spanet_input_name` | `dnn_vars.pairing_spanet_btag` | Input features of the SPANet pairing model, as an `OrderedDict` with a `"sequential"` (per-jet) and a `"global"` (per-event) block. It must match the event file used for the SPANet training. The available sets are defined in [`dnn_input_variables.py`](../dnn_input_variables.py) (`pairing_spanet_nobtag`, `pairing_spanet_btag`, `pairing_spanet_btagWP5`, `pairing_spanet_btagWP3`, `pairing_spanet_btagDeltaWP5`, `pairing_spanet_vbf_JetVBF_DNNVars_JetHiggsGlobal`, ...). The first entry of the `"sequential"` block also defines the jet collection used for the pairing. |
| `spanet_input_name_list` | `None` | Flat list of the SPANet sequential input names. Only its last entry is inspected by the config templates, to decide whether the b-tag working-point columns have to be saved for the SPANet training. `None` means "derive it from `spanet_input_name`", which is what the templates do; set it explicitly only to override that. |
| `sig_bkg_dnn_input_variables` | `dnn_vars.sig_bkg_dnn_input_variables` | Input features of the signal-vs-background DNN. It is also used, together with the morphing variables, to build the list of columns saved when `dnn_variables` is `True`. |
| `bkg_morphing_dnn_input_variables` | `dnn_vars.bkg_morphing_dnn_input_variables` | Input features of the background morphing DNN (and of the spread model). |
| `vbf_discriminator_input_variables` | `None` | Input features of a standalone ggF/VBF discriminator model. Only needed when `vbf_discriminator` is a different file from `spanet`. |
| `pad_value` | `-999.0` | Value used to fill the missing entries (padding, unmatched jets, ...) of the DNN inputs, and to replace the out-of-range values in `Padded_Arctanh_Delta_pairing_probabilities`. |
| `pad_value_spanet` | `9999.0` | Value used to pad the jet arrays fed to the SPANet models. It is kept separate from `pad_value` because SPANet was trained with a different padding convention. |
| `spanet_global_mask` | `"ones"` | Mask of the SPANet global inputs: `"ones"` keeps the mask always `True`, `"padded"` masks away the global inputs whose features are all equal to `pad_value_spanet` (e.g. a missing VBF jet). |

## Pairing probability variables (SPANet only)

| Option | Default | Description |
| --- | --- | --- |
| `arctanh_delta_prob_bin_edge` | `2.44` | Bin edge used to build `Binned_Arctanh_Delta_pairing_probabilities`, the binned version of $\mathrm{arctanh}(p_\mathrm{best} - p_\mathrm{second\,best})$. |
| `arctanh_delta_prob_pad_limit` | `2.0` | Upper limit above which `Arctanh_Delta_pairing_probabilities` is replaced by `pad_value` in `Padded_Arctanh_Delta_pairing_probabilities`. |

## Categories and regions

The categories built from these options are described in [categories.md](categories.md).

| Option | Default | Description |
| --- | --- | --- |
| `vr1` | `False` | Use the VR1 validation regions (Higgs mass planes centred at $(185, 180)$ GeV) instead of the nominal signal/control regions. |
| `expandCR` | `False` | Use the wide control region ($30 < R_{HH} < 80$ instead of $30 < R_{HH} < 55$) for the morphing `preW`/`postW` categories. |
| `blind` | `False` | Add the blinded copies of the signal regions, keeping only the events with `sig_bkg_dnn_score` below the blinding threshold (0.9). |
| `split_qcd` | `True` | Boosted only: split the QCD control region into the `qcd_A`/`qcd_B`/`qcd_C` sub-regions instead of defining a single `qcd` region. Ignored when `boosted` is `False`. |
| `qt_postEE` | `None` | Path to the pickled quantile transformer used to define variable-width bins of `sig_bkg_dnn_score` (constant SM signal yield per bin) for the 2022_postEE datacards. `None` (or `""`) falls back to the uniform binning. See [Quantile transformer to obtain constant signal binning](../README.md#quantile-transformer-to-obtain-constant-signal-binning). |
| `qt_preEE` | `None` | Same as `qt_postEE`, for the 2022_preEE datacards. |

## SPANet training sample production

| Option | Default | Description |
| --- | --- | --- |
| `random_pt` | `False` | Randomly scale the jet pt (and mass) to flatten the pt spectrum of the SPANet training sample. It saves the `*PtFlatten*` jet collections and the `random_pt_weights` column. Cannot be combined with `run2`. |
| `rand_type` | `0.3` | Range of the random pt scale factor applied by `random_pt`: `0.5` → $[0.5, 1.5]$, `0.3` → $[0.3, 1.7]$, `0.1` → $[0.1, 10.0]$. Any other value raises a `ValueError`. |

## Output

| Option | Default | Description |
| --- | --- | --- |
| `dnn_variables` | `True` | Compute and save the DNN input variables (Higgs and HH kinematics, additional jet, `sigma_over_higgs*_reco_mass`, ...). If `False`, only the SPANet training columns (or the default jet columns) are saved. |
| `save_chunk` | `False` | Dump the columns as `parquet` files per chunk instead of accumulating them in the coffea output (it sets `dump_columns_as_arrays_per_chunk` and disables the flattening of the columns). The output path is stored in the `config.json` of the run. The chunks can be merged into one file per dataset/region/variation with `run_pocket_coffea ... --merge-parquet` (see [Merge the parquet chunks](../README.md#merge-the-parquet-chunks)). |
| `donotscale_sumgenweights` | `False` | Do not normalise the MC by the sum of the generator weights. Kept for backward compatibility with the PocketCoffea `Configurator` option; it is currently not forwarded by any of the config templates. |

## Boosted BDT

| Option | Default | Description |
| --- | --- | --- |
| `bdt_model` | `""` | Path to the XGBoost model of the other-group boosted analysis. It produces `boosted_bdt_score` and `boosted_bdt_vbf_score`. An empty string disables it. |

## Keys added by the config templates

These keys are not meant to be set by hand in a config file: the templates add
them to `config_options_dict` while building the `Configurator`.

| Key | Added by | Description |
| --- | --- | --- |
| `dump_columns_as_arrays_per_chunk` | all templates | Set from `save_chunk`, and read by PocketCoffea to write the columns as `parquet` files per chunk. In the `config.json` of the output it is overwritten (together with `save_chunk`) with the path of the merged files by `merge_parquet.py --update-config` (`run_pocket_coffea ... --merge-parquet-update-config`), the original being kept as `dump_columns_as_arrays_per_chunk_unmerged`. |
| `num_bins` | [`config_compute_befficiency_HH4b.py`](../../HH4b_btagging/config_compute_befficiency_HH4b.py) | Leftover of the b-tag efficiency studies; currently not read anywhere. |
