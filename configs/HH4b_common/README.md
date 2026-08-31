# HH4b analysis

> [!IMPORTANT]
> Work in Progress

This folder contains the configuration files and customization code for the HH4b analysis.

## Configuration options

The analysis is steered by two dictionaries, `onnx_model_dict` and
`config_options_dict`, defined in a small config file inside
[`config_files/`](./config_files/) (e.g. [`spanet_ptflat.py`](./config_files/spanet_ptflat.py)).
Every such file starts from the defaults in
[`config_files/default_config.py`](./config_files/default_config.py) and overrides
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

`config_options_dict` is passed to the PocketCoffea `Configurator` as
`workflow_options`. `HH4bCommonProcessor.__init__` turns every key into an
attribute of the processor, so an option `"foo"` is read inside the workflows as
`self.foo`. The same dictionary is also read directly by the config templates to
build the categories, the columns, the skim and the preselection.

The tables below describe all the options available in
[`default_config.py`](./config_files/default_config.py).

### ONNX models (`onnx_model_dict`)

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

### Analysis flavour

| Option | Default | Description |
| --- | --- | --- |
| `approach` | `"first"` | Jet pt definition and object preselection. Selects the `params/object_preselection_<approach>_approach.yaml` file and how the `Jet` collection is built in `apply_object_preselection`. `"first"`: use the PNet+Neutrino regressed pt when available (HIG-24-010 first approach). `"second"`: use the regressed pt also when the jet passes the loose b-tag WP (HIG-24-010 second approach). `"boosted"`: skip the resolved jet-pt handling and the b-tag WP definition, and drop the HLT selection from the skim. |
| `boosted` | `False` | Run the boosted (AK8 / `FatJet`) analysis: builds the `FatJetGood` collection, the boosted categories, the boosted DNN variables and the `FatJetGood` columns. |
| `run2` | `False` | Use the Run-2 $D_{HH}$ pairing algorithm instead of SPANet. Cannot be combined with `random_pt` (there is an explicit `assert` in the config templates). |
| `vbf_analysis` | `False` | Run the VBF part of the analysis: builds the VBF jet collections (`JetGoodVBF*`, `JetAdditionalGoodVBF`, the merged `JetTotalSPANet*` collections), the VBF variables (`mjj`, `deta`, centrality) and the VBF categories. |
| `mixeddata` | `False` | Run on the mixed-data samples (data-driven background model) instead of the 2b data. The `2b_*_preW`/`postW` categories become `4b_*_preW`/`postW`, and the HLT/L1 skim cuts and the jet calibration are switched off. |

### Skim and preselection

| Option | Default | Description |
| --- | --- | --- |
| `tight_cuts` | `False` | Apply the tight jet pt cuts (`pt_tight` in the object preselection) and require the pt preselection on the 4 b-tag-ordered Higgs candidate jets (`JetGoodHiggs`) instead of all `JetGood`. |
| `vbf_presel` | `False` | Use the VBF preselection. **Not supported anymore**: `define_preselection` raises a `ValueError` if this is `True`, because the cut acts on the wrong jet collection. |
| `boosted_presel` | `False` | Use the boosted preselection (at least 2 FatJets) instead of the resolved one. It also disables the jet veto map cut. |
| `semi_tight_vbf` | `True` | Legacy flag for the semi-tight VBF jet selection. It is only accepted as an argument of `jet_selection_nopu` and is currently not used by any workflow. |
| `noL1` | `False` | Drop the L1 seed requirement (`get_L1sel`) from the skim. Needed for the samples/eras for which the L1 emulation is not available. |

### Truth matching (MC only)

| Option | Default | Description |
| --- | --- | --- |
| `which_bquark` | `"last"` | Which copy of the b-quarks from $H\to b\bar{b}$ is matched to the jets. `"first"`: the direct children of the Higgs. `"last"`: the last copy, found by walking up the decay chain. `"last_numba"`: the last copy, found with the numba helper `get_parton_last_copy`. `"last_numba_with_status"`: same, but the b-quarks are selected requiring `status == 23`. |
| `parton_jet_min_dR` | `0.4` | Maximum $\Delta R$ between a parton and a jet for the matching to succeed, both for the Higgs b-quarks and for the VBF quarks. |

### VBF

| Option | Default | Description |
| --- | --- | --- |
| `vbf_parton_matching` | `False` | Run the gen-level matching of the VBF quarks to the jets (VBF MC only) and fill the `provenance_vbf` field. When `False`, a dummy (all-`None`) `provenance_vbf` is created. |
| `which_vbf_quark` | `"with_mothers_children"` | How the VBF quarks are identified at gen level. `"with_mothers_children"`: hard-process quarks whose mother also has two Higgs children. `"with_status"`: outgoing (`status == 23`) non-b partons. |
| `max_num_jets_add_vbf` | `2` | Number of *additional* VBF jet candidates (on top of the leading-$m_{jj}$ pair) kept in `JetAdditionalGoodVBF` and merged into the SPANet input collections. |
| `jets_add_vbf_order` | `"energy"` | Field used to order the additional VBF jets, e.g. `"energy"` or `"pt"`. |
| `vbf_matching_after_higgs_pairing` | `False` | Run the SPANet Higgs pairing first and define the VBF candidates from the jets left over by the pairing, instead of from the b-tag-ordered `JetGoodClip` collection. Requires `spanet`. |
| `ggf_vbf_threshold` | `0.95` | Threshold on the ggF-vs-VBF discriminator score (`VBF_ggF_score`) used to split the pass/fail VBF categories. Only relevant if `vbf_discriminator` is set. |

### Fox-Wolfram momenta

| Option | Default | Description |
| --- | --- | --- |
| `max_order_FW` | `0` | Maximum order $l$ of the Fox-Wolfram moments computed on `JetGood`. `0` disables the computation; otherwise the columns `FW_H{i}_{norm}` and `FW_R{i}_{norm}` are added to `events` for `i in range(max_order_FW)`. |
| `FW_momenta_norms` | `["W_T"]` | List of normalisation schemes used for the Fox-Wolfram moments. One set of columns is produced per scheme. |

### Jet collections and pairing

| Option | Default | Description |
| --- | --- | --- |
| `max_num_jets_good` | `5` | Number of `JetGood` jets kept for the VBF analysis (`JetGoodClip`/`JetGoodPadded`). It is also the offset at which the VBF pair starts inside the merged SPANet collections. |
| `max_num_jets_higgs_pairing` | `5` | Number of jets given to the SPANet pairing model. The true pairing is truncated to this number when computing the pairing efficiency. |
| `max_num_jets_vbf_discriminator` | `None` | Number of jets given to a standalone ggF/VBF discriminator model. `None` means "use all the jets of the input collection". |
| `max_num_jets_spanet_class` | `4` | Number of jets given to a SPANet-format signal-vs-background classifier. Only used when `sig_bkg_dnn` is a SPANet-format model. |
| `fifth_jet` | `"pt"` | Ordering of the jets beyond the 4 Higgs candidates. `"pt"` re-sorts the 5th and following jets by pt (the first 4 stay b-tag ordered); any other value keeps the pure b-tag ordering. |
| `add_jet_spanet` | `False` | Sort the `Jet` collection by regressed pt before the good-jet selection, and order the additional jet collection (`JetNotFromHiggs`) by b-tag score or pt depending on whether the pairing picked the 5th jet. |
| `old_wp_def` | `False` | Use the old b-tag working-point convention, where the WP index starts at `-1` (no WP passed) instead of `0`. It must match the convention used to train the SPANet/DNN models. |
| `TXbb_order` | `False` | Boosted only: order the `FatJetGood` collection by `btagBBTXbb` instead of `btagBB`. |
| `only5jetsbSF` | `False` | b-tag SF studies only (`configs/HH4b_btagging`): compute the b-tag scale factors using only the 5 leading jets instead of all the `JetGood`. |

### Model inputs and padding

| Option | Default | Description |
| --- | --- | --- |
| `spanet_input_name` | `dnn_vars.pairing_spanet_btag` | Input features of the SPANet pairing model, as an `OrderedDict` with a `"sequential"` (per-jet) and a `"global"` (per-event) block. It must match the event file used for the SPANet training. The available sets are defined in [`dnn_input_variables.py`](./dnn_input_variables.py) (`pairing_spanet_nobtag`, `pairing_spanet_btag`, `pairing_spanet_btagWP5`, `pairing_spanet_btagWP3`, `pairing_spanet_btagDeltaWP5`, ...). The first entry of the `"sequential"` block also defines the jet collection used for the pairing. |
| `sig_bkg_dnn_input_variables` | `dnn_vars.sig_bkg_dnn_input_variables` | Input features of the signal-vs-background DNN. It is also used, together with the morphing variables, to build the list of columns saved when `dnn_variables` is `True`. |
| `bkg_morphing_dnn_input_variables` | `dnn_vars.bkg_morphing_dnn_input_variables` | Input features of the background morphing DNN (and of the spread model). |
| `vbf_discriminator_input_variables` | `None` | Input features of a standalone ggF/VBF discriminator model. Only needed when `vbf_discriminator` is a different file from `spanet`. |
| `pad_value` | `-999.0` | Value used to fill the missing entries (padding, unmatched jets, ...) of the DNN inputs, and to replace the out-of-range values in `Padded_Arctanh_Delta_pairing_probabilities`. |
| `pad_value_spanet` | `9999.0` | Value used to pad the jet arrays fed to the SPANet models. It is kept separate from `pad_value` because SPANet was trained with a different padding convention. |

### Pairing probability variables (SPANet only)

| Option | Default | Description |
| --- | --- | --- |
| `arctanh_delta_prob_bin_edge` | `2.44` | Bin edge used to build `Binned_Arctanh_Delta_pairing_probabilities`, the binned version of $\mathrm{arctanh}(p_\mathrm{best} - p_\mathrm{second\,best})$. |
| `arctanh_delta_prob_pad_limit` | `2.0` | Upper limit above which `Arctanh_Delta_pairing_probabilities` is replaced by `pad_value` in `Padded_Arctanh_Delta_pairing_probabilities`. |

### Categories and regions

| Option | Default | Description |
| --- | --- | --- |
| `vr1` | `False` | Use the VR1 validation regions (Higgs mass planes centred at $(185, 180)$ GeV) instead of the nominal signal/control regions. |
| `expandCR` | `False` | Use the wide control region ($30 < R_{HH} < 80$ instead of $30 < R_{HH} < 55$) for the morphing `preW`/`postW` categories. |
| `blind` | `False` | Add the blinded copies of the signal regions, keeping only the events with `sig_bkg_dnn_score` below the blinding threshold (0.9). |
| `qt_postEE` | `None` | Path to the pickled quantile transformer used to define variable-width bins of `sig_bkg_dnn_score` (constant SM signal yield per bin) for the 2022_postEE datacards. `None` (or `""`) falls back to the uniform binning. See [Quantile transformer to obtain constant signal binning](#quantile-transformer-to-obtain-constant-signal-binning). |
| `qt_preEE` | `None` | Same as `qt_postEE`, for the 2022_preEE datacards. |

### SPANet training sample production

| Option | Default | Description |
| --- | --- | --- |
| `random_pt` | `False` | Randomly scale the jet pt (and mass) to flatten the pt spectrum of the SPANet training sample. It saves the `*PtFlatten*` jet collections and the `random_pt_weights` column. Cannot be combined with `run2`. |
| `rand_type` | `0.3` | Range of the random pt scale factor applied by `random_pt`: `0.5` → $[0.5, 1.5]$, `0.3` → $[0.3, 1.7]$, `0.1` → $[0.1, 10.0]$. Any other value raises a `ValueError`. |

### Output

| Option | Default | Description |
| --- | --- | --- |
| `dnn_variables` | `True` | Compute and save the DNN input variables (Higgs and HH kinematics, additional jet, `sigma_over_higgs*_reco_mass`, ...). If `False`, only the SPANet training columns (or the default jet columns) are saved. |
| `save_chunk` | `False` | Dump the columns as `parquet` files per chunk instead of accumulating them in the coffea output (it sets `dump_columns_as_arrays_per_chunk` and disables the flattening of the columns). The output path is stored in the `config.json` of the run. |
| `donotscale_sumgenweights` | `False` | Do not normalise the MC by the sum of the generator weights. Kept for backward compatibility with the PocketCoffea `Configurator` option; it is currently not forwarded by any of the config templates. |

### Boosted BDT

| Option | Default | Description |
| --- | --- | --- |
| `bdt_model` | `""` | Path to the XGBoost model of the other-group boosted analysis. It produces `boosted_bdt_score` and `boosted_bdt_vbf_score`. An empty string disables it. |

### Options not present in the defaults

A few configs add extra keys on top of `default_config.py`:

| Option | Used by | Description |
| --- | --- | --- |
| `split_qcd` | boosted configs | Split the boosted QCD control region into the `qcd_A`/`qcd_B`/`qcd_C` sub-regions instead of a single `qcd` region. |
| `vbf_selection` | boosted configs | Takes precedence over `vbf_analysis` in [`VBF_HH4b_boosted_config.py`](../VBF_HH4b_boosted/VBF_HH4b_boosted_config.py) when defining the categories. |
| `no_btag` | `configs/HH4b_btagging` | Passed to `define_preselection` to remove the b-tag requirement from the preselection, needed to measure the b-tag WP efficiencies. |
| `spanet_input_name_list` | `HH4b_parton_matching_config.py`, `HH4b_boosted_config.py` | Flat list of the SPANet input names; only its last entry is inspected, to decide whether the b-tag working-point columns have to be saved. |

## Full analysis workflow

The full analysis workflow is composed by multiple steps, which are spread in different repositories:

- Configurations for [Pocket Coffea](https://github.com/matteomalucchi/PocketCoffea)
  - <https://github.com/matteomalucchi/AnalysisConfigs>
- Configurations for [SPANet](https://github.com/matteomalucchi/SPANet)
  - <https://github.com/matteomalucchi/HH4b_SPANet>
- DNN training
  - <https://github.com/matteomalucchi/ML_pytorch>

### Build datasets

> [!TIP]
> @Tier-3/AnalysisConfigs &rarr;  [README](https://github.com/matteomalucchi/AnalysisConfigs/blob/main/README.md)

To build the datasets needed for the Analysis, run the following command on `tier-3`:

```bash
"build-datasets --cfg datasets/datasets_definitions.json -o -rs 'T[123]_(FR|IT|BE|CH|DE|US)_\w+'"
```

### Skimming

> [!TIP]
> @Tier-3/AnalysisConfigs &rarr; [PocketCoffea skimming docs](https://pocketcoffea.readthedocs.io/en/stable/recipes.html#skimming-events)

Skimming reduces large NanoAOD files to smaller ones containing only events passing trigger/skim cuts, which significantly speeds up the main analysis. The general workflow is:

1. **Set the output path** in the skim config by editing `save_skimmed_files` in the `Configurator`. For the resolved analysis this is in [`configs/HH4b/HH4b_save_skimmed.py`](../HH4b/HH4b_save_skimmed.py); for the boosted analysis it is in [`configs/HH4b/HH4b_boosted_save_skimmed.py`](../HH4b/HH4b_boosted_save_skimmed.py).

   ```python
   cfg = Configurator(
       save_skimmed_files="root://<xrootd-endpoint>//<output_path>/<skim_dir_name>",
       ...
   )
   ```

2. **Run the skim** on Tier-3:

   ```bash
   cd configs/HH4b
   run_pocket_coffea skim <skim_config.py> <run_options.yaml> <output_dir>
   ```

   > Example (resolved analysis):
   > ```bash
   > run_pocket_coffea skim HH4b_save_skimmed.py params/t3_run_options_skim_resolved.yaml /work/mmalucch/out_hh4b/JetMET_skim/
   > ```
   > Example (boosted analysis):
   > ```bash
   > run_pocket_coffea skim HH4b_boosted_save_skimmed.py params/t3_run_options_skim_resolved.yaml /work/mmalucch/out_hh4b/boosted_skim/
   > ```

3. **Merge coffea outputs** — mandatory to collect the cutflow and `sum_genweights` from all jobs (including chunks that skimmed 0 events), needed for correct cross-section normalization:

   ```bash
   cd <output_dir>
   pocket-coffea merge-outputs -o output_all.coffea *.coffea -f
   ```

   > Example:
   > ```bash
   > cd /work/mmalucch/out_hh4b/JetMET_skim
   > pocket-coffea merge-outputs -o output_all.coffea *.coffea -f
   > ```

4. **Hadd skimmed ROOT files** — groups small per-chunk ROOT files into larger ones. The `--dry` flag previews the job splitting; remove it to write the hadd scripts. The `-e` option sets the target events per output file and `-s` the number of parallel hadd processes per job:

   ```bash
   pocket-coffea hadd-skimmed-files \
     -fl output_all.coffea \
     -o root://<xrootd-endpoint>//<output_path>/<hadd_skim_dir_name> \
     -e <events_per_file> --dry -s <n_parallel>
   ```

   > Example:
   > ```bash
   > pocket-coffea hadd-skimmed-files \
   >   -fl output_all.coffea \
   >   -o root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/mmalucch/HH4b/skimmed_files/DATA_JetMET_resolved_skimmed/JetMET_hadd_skim \
   >   -e 100000 --dry -s 6
   > ```

   To re-run only files that failed the hadd, add `--check` (skips already-produced output files):

   ```bash
   pocket-coffea hadd-skimmed-files \
     -fl output_all.coffea \
     -o root://<xrootd-endpoint>//<output_path>/<hadd_skim_dir_name> \
     -e <events_per_file> --dry -s <n_parallel> --check
   ```

5. **Submit hadd jobs** via Slurm (requires a ROOT-enabled environment):

   ```bash
   micromamba activate root-env
   sbatch -p standard --account=t3 --mem=<mem_mb> --wrap "python do_hadd.py" -c <n_cores> --output do_hadd_pipe.txt
   ```

   > Example:
   > ```bash
   > micromamba activate root-env
   > sbatch -p standard --account=t3 --mem=15000 --wrap "python do_hadd.py" -c 6 --output do_hadd_pipe.txt
   > ```

### Produce SPANet input files

On `tier-3`, run the following commands to produce the input files for SPANet training.

#### Run pocket-coffea to produce coffea files

> [!TIP]
> @Tier-3/AnalysisConfigs &rarr;  [README](https://github.com/matteomalucchi/AnalysisConfigs/blob/main/README.md)

```bash
cd AnalysisConfigs/configs/HH4b
# SPANet training with normal pT spectrum
run_pocket_coffea no_model HH4b_spanet_input.py <t3_run_options> <output_dir>
# SPANet training with flat pT spectrum
run_pocket_coffea pt_vary HH4b_spanet_input.py <t3_run_options> <output_dir>

# e.g.
run_pocket_coffea no_model HH4b_spanet_input.py params/t3_run_options.yaml ../../../sample_spanet/loose_MC_postEE_btagWP
```

> [!NOTE]
> It does not matter, if the config file (e.g. `no_model`) is passed with or without the `.py` ending. The script handles this automatically.

> [!NOTE]
> To run a test on a small number of files, add the `--test` flag at the **end** of the command.

#### Convert coffea files to h5 files

> [!TIP]
> @Tier-3/HH4b_SPANet &rarr;  [README](https://github.com/matteomalucchi/HH4b_SPANet/blob/main/README.md)

```bash
cd HH4b_SPANet/utils/dataset
python3 coffea_to_parquet.py -i <input_coffea_file> -o <output_dir> -c 4b_region
python3 parquet_to_h5.py -i <input_parquet_files> -o <output_dir> -f 0.8

# e.g. 
python3 /work/tharte/HH4b_SPANet/utils/dataset/coffea_to_parquet.py -i .//output_all.coffea -o . -c 4b_region
python3 /work/tharte/HH4b_SPANet/utils/dataset/parquet_to_h5.py -i ./*.parquet -o /scratch/tharte/166814/ -f 0.8
```

#### Copy h5 files to `lxplus`

> [!TIP]
> @Tier-3

```bash
scp -r <dir> <user>@lxplus.cern.ch:<dir>

# e.g.
scp -r loose_MC_postEE_btagWP tharte@lxplus.cern.ch:/eos/user/t/tharte/Analysis_data/spanet_samples
```

### Train and evaluate SPANet model

> [!TIP]
> @lxplus/HH4b_SPANet &rarr; [README](https://github.com/matteomalucchi/HH4b_SPANet/blob/main/README.md)

Edit the option_file accordingly to the training you want to perform.

- Event info file (input parameters to SPANet):
  - Inside the folder `HH4b_SPANet/event_files/HH4b/`
- Option file:
  - Inside the folder `HH4b_SPANet/Option_files/HH4b`
  - Lines to edit:

    - ```json
      "event_info_file": "...",                
      "training_file": "...",  
      ```

    - e.g.

    ```json
      "event_info_file": "/afs/cern.ch/user/t/tharte/public/Software/HH4b_SPANet/event_files/HH4b/hh4b_5jet_btag_wp.yaml",                
      "training_file": "/eos/user/t/tharte/Analysis_data/spanet_samples/loose_MC_postEE_btagWP/output_JetGood_train.h5",  
      ```

Then run the following command on `lxplus`:

```bash
cd HH4b_SPANet/
python jobs/submit_jobs_seed.py -o <options_files/option_file.json> -c <jobs/config/config.yaml> -s <start_seed>:<end_seed> -a <"additional arguments to pass to spanet.train"> --suffix <directory_suffix> -out <output_dir>

# e.g. 
python3 ~/public/Software/HH4b_SPANet/jobs/submit_jobs_seed.py -c ~/public/Software/HH4b_SPANet/jobs/config/jet_assignment_deep_network_3d.yaml -s 100:101 -o options_files/HH4b/hh4b_5jets_ptreg_loose_300_btag_wp.json -out /eos/user/t/tharte/Analysis_data/spanet_output
```

#### Compute SPANet predictions

> [!TIP]
> @lxplus/HH4b_SPANet &rarr; [README](https://github.com/matteomalucchi/HH4b_SPANet/blob/main/README.md)

Once the model is trained, compute the predictions on the input h5 files using the following command on `lxplus`:

```bash
python -m spanet.predict <path_to_spanet_output/out_seed_training_yyy/version_z> <output/file.h5> -tf </true/file/with/inputs> --gpu

app_spanet
. ~/.bashrc
env_spanet
python -m spanet.predict ./spanet_output/out_spanet_outputs/out_hh4b_5jets_ptreg_loose_300_btag_wp/out_seed_trainings_100/version_0/ predictions/spanet_hh4b_5jets_300_ptreg_loose_s100_btag_wp.h5 -tf spanet_samples/loose_MC_postEE_btagWP/output_JetGood_test.h5 --gpu 
# In case of checking the mass sculpting with data, choose a data file as -tf argument
```

#### Plot pairing efficiency and mass sculpting

> [!TIP]
> @lxplus/HH4b_SPANet &rarr; [README](https://github.com/matteomalucchi/HH4b_SPANet/blob/main/README.md)

Next step is to fill in an entry in the efficiency script `HH4b_SPANet/utils/performance/efficiency_configurations.py`:

The `efficiency_configuration` script contains two dictionaries: `spanet_dict` and `true_dict`. These have to be completed with the new models:
```python
spanet_dict = {
    ...
    '<unique_identifier>': {                                                                                            
        'file': f'</path/to/file>.h5',                                                     
        'true': '<unique_identifier_of_true_file>',                                                                              
        'label': '<label_for_plot>',                                                                                                
        'color': '<color_in_plot>'},
	...
	#e.g.
    '5_jets_ptvary_btag_wp_3V00e_allklambda': {                                                                                     
 	    'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp.h5',                                                 
 	    'true': '5_jets_pt_true_wp_allklambda',                                                                                    
 	    'label': 'SPANet btag 5 WP - Flattened pt [0.3,1.7]',                                                                      
 	    'color': 'orangered'},
}

true_dict = {
    ...
	'<unique_identifier>': {'name': '</path/to/truefile>', 'klambda': <'preEE'/'postEE'>}  # klambda settings define, which klambdas are in the file (preEE has less klambda than postEE)
	...
    #e.g.
	'5_jets_pt_true_wp_allklambda': {'name': f"{true_dir_thierry}../spanet_samples/loose_MC_postEE_btagWP/output_JetGood_test.h5", 'klambda': 'postEE'}, 
}
```

And then run the efficiency script in an empty folder (A different environment is needed without the apptainer image):

```bash
python3 <path/to/HH4b_SPANet>/utils/performance/efficiency_studies.py -pd <output/dir/for/plots> <-k> <-d>  # -k is to separate klambdas, -d is to run on datafiles

# e.g.
env_utils 
python3 ~/public/Software/HH4b_SPANet/utils/performance/efficiency_studies.py -pd . -k # To run the mass shapes using data inputs, replace -k with -d
```

#### Convert SPANet model to ONNX

> [!TIP]
> @lxplus/HH4b_SPANet &rarr; [README](https://github.com/matteomalucchi/HH4b_SPANet/blob/main/README.md)

Converting the file to `onnx` to use it in PocketCoffea (You need again the SPANet environment from before for the prediction):

``` bash
python -m spanet.export <path_to_spanet_output/out_seed_training_yyy/version_z> <onnx_output_name.onnx> --gpu 

#e.g.
python -m spanet.export out_spanet_outputs/out_hh4b_5jets_ptvary_loose_300_btag_wp/out_seed_trainings_100/version_0/ spanet_hh4b_5jets_ptvary_loose_300_btag_5wp_s100.onnx --gpu
```

#### Copy ONNX model to `tier-3`

> [!TIP]
> @lxplus

Finally, copy the model to `tier-3` to use it in PocketCoffea:

```bash
scp <model.onnx> <user>@t3ui07.psi.ch:<dir>
```

### Apply SPANet model to data for background morphing

> [!TIP]
> @Tier-3/AnalysisConfigs

Create a config in [`AnalysisConfigs/configs/HH4b_common/config_files`](./config_files/) and set the `"spanet"` entry of the `onnx_model_dict` to the path of the ONNX model you copied to `tier-3`.

Then run PocketCoffea with that config to produce coffea files with SPANet predictions on data files to be used for background morphing using the following command:

```bash
run_pocket_coffea <config_name> <config_file> <t3_run_options> <output_dir>
```

> [!NOTE]
> If the columns are saved as `parquet` in a different folder (using the `save_chunks` setting), the path to the files is stored in the `config.json`.
> If one of the scripts using the columns does not find the columns, the problem could be, that this file was overwritten/is missing.

### Train DNN model for background morphing

> [!TIP]
> @Tier-3/ML_pytorch &rarr; [README](https://github.com/matteomalucchi/ML_pytorch/blob/main/README.md)

Create a config in `ML_pytorch/configs/bkg_reweighting/` and set the `data_dirs` entry to the path of the coffea files you produced in the previous step.

Then run the training using the following command:

```bash
sbatch run_20_trainings_in_4_parallel.sh <config_file> <output_folder>

# when this has finished, you can merge the results with:
cd <output_folder>
ml_onnx -i best_models -o best_models -ar -v bkg_morphing_dnn_DeltaProb_input_variables

```

The training will produce the ONNX model to be used in PocketCoffea for background morphing, as well as plots with the training history, the ROC curve and an overtraining check.

### Apply background morphing DNN to data and produce MC signal files

> [!TIP]
> @Tier-3/AnalysisConfigs

Update the config created before  in [`AnalysisConfigs/configs/HH4b_common/config_files`](./config_files/) and set the `"bkg_morphing_dnn"` entry of the `onnx_model_dict` to the path of the ONNX model you produced in the previous step.

Then run PocketCoffea with that config to produce coffea files with the background prediction and signal samples using the following command:

```bash
run_pocket_coffea <config_name> <config_file> <t3_run_options> <output_dir>
```

#### Plot morphed 2b vs 4b

> [!TIP]
> @Tier-3/AnalysisConfigs

To compare the morphed 2b data with the 4b data in CR and SR, run the following command:

```bash
sbatch -p short --account=t3 --time=00:05:00 --mem 25gb --cpus-per-task=8 --wrap="python AnalysisConfigs/scripts/plot_2bMorphedvs4b.py -i <input_directory> -o <output_directory> <--novars> <-r2>"
```

### Train DNN for signal / background classification

> [!TIP]
> @Tier-3/ML_pytorch &rarr; [README](https://github.com/matteomalucchi/ML_pytorch/blob/main/README.md)

Create a config in `ML_pytorch/configs/ggF_bkg_classifier/` and set the `data_dirs` entry to the path of the coffea files you produced in the previous step.

Then run the training using the following command:

```bash
sbatch run_sig_bkg_classifier.sh <config_file> <output_folder>
```

### Apply DNN to data and MC signal files

> [!TIP]
> @Tier-3/AnalysisConfigs

> [!NOTE]
> If the goal is to produce Datacards, the `quantile_transformer` has to be run before this step (See in section [Quantile transformer to obtain constant signal binning](#Quantile-transformer-to-obtain-constant-signal-binning)).

Update the config created before  in [`AnalysisConfigs/configs/HH4b_common/config_files`](./config_files/) and set the `"sig_bkg_dnn"` entry of the `onnx_model_dict` to the path of the ONNX model you produced in the previous step.

Then run PocketCoffea with that config to produce coffea files with the background prediction and signal samples using the following command:

```bash
run_pocket_coffea <config_name> <config_file> <t3_run_options> <output_dir>
```

#### Plot DNN score

> [!TIP]
> @Tier-3/AnalysisConfigs


```bash
sbatch -p short --account=t3 --time=00:05:00 --mem 25gb --cpus-per-task=8 --wrap="python AnalysisConfigs/scripts/plot_DNN_score.py -i <input_directory> -im <input_signal_file> -o <output_directory> <--novars>  <-r2>"
```

### Datacard production

This section describes how to produce the datacards for the final statistical analysis.

> [!IMPORTANT]
> Work in Progress

#### Quantile transformer to obtain constant signal binning

> [!TIP]
> @Tier-3/AnalysisConfigs

The quantile transformer is mainly needed for Datacard production. The idea is to compute the bin widths for the `sig_bkg_score` variables in a way, that each bin contains the same amount of MC SM signal. This can be done in two ways:

The first option is to train the DNN model for signal / background classification and then apply this model on a previously created PocketCoffea file that used the same SPANet model for the pairing. This is the recommended way:
```bash
python </path/to/script>/extract_quantile_transformer.py -i </path/to/coffeafiles>/output_GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE.coffea \
	--onnx-model </path/to/model/modelname>.onnx \
	--input-variables <sig_bkg_input_variable_list_name> <--novars>
	-o <output_directory (default is ./quantile_transformer)****>
```

The second option is to use the score variables that are already in the `.coffea` files. In this case, the last PocketCoffea command has to be rerun after defining the bins to get the variables for the datacards.

Due to the need to rerun PocketCoffea, this second option is Not recommended:

```bash
python scripts/extract_quantile_transformer.py -i <input_signal_file> <--novars>
```

Set the `qt_postEE`/`qt_preEE` entry in the config created before  in [`AnalysisConfigs/configs/HH4b_common/config_files`](./config_files/) to the path of the quantile transformer you produced in the previous step.

If the first option was chosen, continue with the section [Apply DNN to data and MC signal files](#Apply-DNN-to-data-and-MC-signal-files).

Otherwise, run PocketCoffea with that config to produce coffea files using the following command:

```bash
run_pocket_coffea <config_name> <config_file> <t3_run_options> <output_dir>
```

#### Produce datacards

> [!TIP]
> @Tier-3/AnalysisConfigs


To produce the datacards, we need a single `output_all.coffea` file made from all the relevant coffea outputs from the last `PocketCoffea` run:

```bash
# Inside output folder:
pocket-coffea merge-outputs -o output_all.coffea *.coffea -f
```

Then the `build_datacards.py` script can just be run like this:

```bash
python </path/to/AnalysisConfigs>/scripts/build_datacards.py -i <input_folder> -o <desired output folder>
```

#### Compute b-tag WP efficiencies

> [!TIP]
> @Tier-3/AnalysisConfigs

We need to compute the b-tag efficiencies within the phase-space where we perform our analysis. This has to be done in a region in which we do Not cut at all on b-tags. This requires to run a different config file:

```bash
cd AnalysisConfig/configs/HH4b_btagging
pocket-coffea run --cfg config_compute_befficiency_HH4b.py -e dask@T3_CH_PSI --custom-run-options <run_option_file> -o <outputfolder>
```

Using the output from that, we can then run the scrip `produceBtagEff.py`. This file needs an input file of type `output_all.coffea`. It still needs some improvement. But the core works:
Different sample groups can be defined, that are combined and use the same efficiencies. This is expected in `YAML` fromat and can be given as input parameter `-g`. The idea is, that the same file is also used to load the groups into the params for `PocketCoffea`. An example with `ttHbb` and `HH4b` groups is given in `AnalysisConfig/configs/HH4b_common/params/btagging_sampleGroups.yaml`.

```bash
python </path/to/AnalysisConfigs>/configs/HH4b_btagging/produceBtagEff.py -i <input_file> -o <desired output folder> -g <sampleGroup file (default works if script is not moved)>
```

The output is then the b-tag WP efficiency files:
```bash
btag_efficiencies_btagDeepFlavB_2022_postEE.json
btag_efficiencies_btagPNetB_2022_postEE.json
btag_efficiencies_btagRobustParTAK4B_2022_postEE.json
```
and they have to be copied to the folder:
```bash
cp btag_efficiencies*.json AnalysisConfig/configs/HH4b_common/params/btag_efficiencies_multipleWP/
```

To then validate the procedure, we need to run on the same region a corrected and an uncorrected set.
Both should then have the same normalisation.
This part is still subject to changes and might be still bugged.
```bash
# Still inside HH4b_btagging
run_pocket_coffea no_model HH4b_parton_matching_config_btagWPsf.py ../HH4b/params/t3_run_options.yaml ../../../samples_no_model_input_for_spanet/no_model_sf_btag_comparison
```

The output from that will save histograms of different kinematic variables. This could be expanded, but should show the differences well enough.
There will be two regions saved. One is called `inclusive`, which only contains standard variations and weights and No b-tag sf. Then there is a `inclusive_btag_sf`. This contains also the b-tag sf. The histograms from both regions can be compared and should more or less fit. All histograms should have the same summed up values within each region if considering over-/underflow bins.

> **TODO.** Write file for comparison of both regions (Notebook Matteo)

## Example commands

### Run analysis

```python
run_pocket_coffea <config_name> <config_file> <run_options> <output_dir> <--test>

# e.g.
run_pocket_coffea spanet_ptflat_rerun_matteo_transform VBF_HH4b_config.py params/t3_run_options_spanet_predict_10Gb.yaml /work/mmalucch/out_hh4b/out_transformed_DNN_score
```

### Run ggF HH4b analysis

```python
cd configs/HH4b
pocket-coffea run --cfg HH4b_parton_matching_config.py -e dask> [!TIP] @T3_CH_PSI --custom-run-options params/t3_run_options_spanet_predict.yaml -o /work/mmalucch/out_test --executor-custom-setup onnx_executor.py
```

### Run VBF HH4b analysis

```python
cd configs/VBF_HH4b
pocket-coffea run --cfg VBF_HH4b_test_config.py -e dask> [!TIP] @T3_CH_PSI --custom-run-options params/t3_run_options_spanet_predict.yaml -o /work/mmalucch/out_hh4b/out_vbf_jets_candidates/  --executor-custom-setup onnx_executor.py
```

### Run plot 2bvs4b

```python
sbatch -p short --account=t3 --time=00:05:00 --mem 25gb --cpus-per-task=8 --wrap="python plot_2bMorphedvs4b.py -i <input_directory> -o <output_directory>"
```

### Run plot DNN_score

```python
sbatch -p short --account=t3 --time=00:10:00 --mem 40gb --cpus-per-task=1 --wrap="python ~/AnalysisConfigs/scripts/plot_DNN_score.py -id ./  -im output_GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE.coffea -r2 -om /work/mmalucch/out_ML_pytorch/DNN_DHH_method_class_weights_e5drop75_postEE_allklambda_matteo/state_dict/model_best_epoch_19.onnx"
```

### Plot variables before datacard
```bash
pocket-coffea make-plots -i output_all.coffea --cfg parameters_dump.yaml -o plots
```

### Run Datacard creation
```bash
pocket-coffea merge-outputs -o output_all.coffea *.coffea -f
python /work/tharte/datasets/AnalysisConfigs_develop/scripts/build_datacards.py -i ./ -o datacards
```
