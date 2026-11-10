# Boosted HH4b analysis

> [!IMPORTANT]
> Work in Progress

This folder contains the config templates and the workflow of the boosted (AK8 / `FatJet`) HH4b analysis:

- [`VBF_HH4b_boosted_config.py`](./VBF_HH4b_boosted_config.py): nominal template, no systematic variations.
- [`VBF_HH4b_boosted_config_with_syst.py`](./VBF_HH4b_boosted_config_with_syst.py): same template with the weight and
  shape variations switched on, needed to produce the datacards.
- [`workflow.py`](./workflow.py): the boosted processor.

Many steps are the same as in the resolved analysis. The configuration options (`config_options_dict`,
`onnx_model_dict`, `run_pocket_coffea`, ...) and the common steps (building the datasets, skimming, ...) are described
in the [HH4b_common README](../HH4b_common/README.md). This README only covers what is different for the boosted
analysis and lists the full chain in order.

The workflow uses three repositories:

- [AnalysisConfigs](https://github.com/matteomalucchi/AnalysisConfigs) (this repository): PocketCoffea configs, plots,
  category optimisation, datacards
- [ML_pytorch](https://github.com/matteomalucchi/ML_pytorch): background-morphing DNN and signal-vs-background DNN
- [CMSSW](https://github.com/cms-sw/cmssw) +
  [Combine](https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit): statistical analysis

> [!NOTE]
> All the paths below are examples from `/work/tharte`, and all the config names are examples too. Adapt them to your
> own setup. The config files themselves live in
> [`../HH4b_common/config_files/`](../HH4b_common/config_files/) (e.g.
> [`boosted_other_group_50GeV_lower_sideband.py`](../HH4b_common/config_files/boosted_other_group_50GeV_lower_sideband.py)).

## Overview

| Step | Where | What it produces |
| --- | --- | --- |
| [0. Datasets and skimming](#0-datasets-and-skimming) | AnalysisConfigs | skimmed NanoAOD |
| [1. ABCD regions](#1-run-the-abcd-regions) | AnalysisConfigs | columns in the 4 ABCD regions |
| [2. Background-morphing DNN](#2-train-the-background-morphing-dnn) | ML_pytorch | `average_model_from_onnx.onnx` |
| [3. Apply the morphing and check the closure](#3-apply-the-morphing-and-check-the-closure) | AnalysisConfigs | morphed `*_postW` regions, closure plots |
| [4. Signal-vs-background DNN](#4-train-the-signal-vs-background-dnn) | ML_pytorch | multiclass sig/bkg ONNX model |
| [5. Optimise the categories](#5-optimise-the-signal-categories) | AnalysisConfigs | `optimised_signal_cuts` |
| [6. Final run with systematics](#6-final-run-with-the-optimised-categories-and-systematics) | AnalysisConfigs | `output_all.coffea` with all variations |
| [7. Signal-region plots](#7-signal-region-plots) | AnalysisConfigs | plots of `cat1`, `cat2`, `cat3`, `catvbf` |
| [8. Datacards](#8-datacards) | AnalysisConfigs | Combine datacards and shapes |
| [9. Limits with Combine](#9-limits-with-combine) | CMSSW | expected limits |

## Boosted config options

These options only matter for the boosted analysis (see also the tables in the
[HH4b_common README](../HH4b_common/README.md#configuration-options)):

| Option | Description |
| --- | --- |
| `boosted`, `boosted_presel`, `approach: "boosted"` | Switch on the boosted analysis (FatJet collections, boosted preselection and categories). |
| `region_setting` | Size of the signal and control regions (`"40GeV"` or `"50GeV"`). |
| `lead_sublead` | Higgs candidate whose mass defines the mass signal region and sideband (`"lead"` or `"sublead"`). |
| `boosted_full_sideband` | Use the full mass sideband instead of only the lower one. |
| `bkg_morphing_direction` | Direction of the background morphing in the ABCD plane, `"btag"` (default) or `"mass"`. See [ABCD regions and morphing direction](#abcd-regions-and-morphing-direction). |
| `TXbb_order` | Order the `FatJetGood` collection by `btagBBTXbb`. |
| `bdt_model` | XGBoost model of the other-group boosted analysis, producing `boosted_bdt_score` and `boosted_bdt_vbf_score`. If empty, the BDT is not evaluated and its columns and histograms are not saved. |
| `optimised_signal_cuts` | Dictionary with the category cuts on the signal-vs-background DNN scores (output of [step 5](#5-optimise-the-signal-categories)). `False` by default. |

### Categories

Which categories are defined depends on which models are in the config:

| Config contains | Categories |
| --- | --- |
| no `bkg_morphing_dnn` | the 4 ABCD regions only: `boosted_signal_region_D`, `boosted_signal_sideband_region_C`, `boosted_control_region_B`, `boosted_control_sideband_region_A` |
| `bkg_morphing_dnn` | the ABCD regions and the morphed `*_postW` regions (see [below](#abcd-regions-and-morphing-direction)) |
| `bkg_morphing_dnn` and `optimised_signal_cuts` | the regions above and the signal categories `cat1`, `cat2`, `cat3`, `catvbf`, cutting on the **DNN** scores |
| `bkg_morphing_dnn` and `bdt_model` (no `optimised_signal_cuts`) | the regions above and the signal categories, cutting on the **BDT** scores with the default cuts of `_get_params` in [`custom_cuts_common.py`](../HH4b_common/custom_cuts_common.py) |

### ABCD regions and morphing direction

The ABCD plane is defined by the TXbb of the subleading Higgs candidate and by the mass of the Higgs candidate
chosen with `lead_sublead` (thresholds for `region_setting` `"40GeV"` / `"50GeV"`):

|  | mass sideband (40 / 50 GeV below the window) | mass window (110–155 / 100–150 GeV) |
| --- | --- | --- |
| **TXbb > 0.8 / 0.65** | C: `boosted_signal_sideband_region_C` | D: `boosted_signal_region_D` (signal region) |
| **TXbb < 0.8 / 0.65** | A: `boosted_control_sideband_region_A` | B: `boosted_control_region_B` |

With `boosted_full_sideband`, the sideband covers both sides of the mass window.

The morphing DNN is trained from A to a neighbouring region and applied to the region next to D. The direction
is set by the `bkg_morphing_direction` option and has to match how the model was trained:

| `bkg_morphing_direction` | Training | Applied to | Morphed category |
| --- | --- | --- | --- |
| `"btag"` (default) | A &rarr; C (along TXbb) | B | `boosted_control_region_B_postW` |
| `"mass"` | A &rarr; B (along mass) | C | `boosted_signal_sideband_region_C_postW` |

`boosted_control_sideband_region_A_postW` is always created, to check the training closure.

### Quantile-mapped TXbb working points (optional, 50GeV)

The TXbb shape is not used in the morphing, only its working-point (WP) index. The nominal rescaling of the low-TXbb
regions (A, B) is the linear `btagBBTXbb_morph`. As an alternative, `btagBBTXbb_CDF_rescaled_WPs` (subleading Higgs
only) maps the WPs of C onto A by quantile: the cuts in A are the TXbb values at the percentiles of the WPs
(0.8, 0.95, 0.975, 0.99) in C, so the fraction of A in each WP bin equals the one of C. The lowest WP (0.65) is the
lower edge of C and needs no cut. The column has the same 0&ndash;5 scale as `btagBBTXbb_morph_5wp`: in C and D it is the
nominal WP index, in A and B it is `1 + number of cuts exceeded`.

It takes two passes, because the cuts are derived from the data in A and C (the nominal TXbb in these regions does not
depend on the option):

1. Run the ABCD regions without the option (step 1).
2. Find the cuts:

   ```bash
   python scripts/find_txbb_cdf_wp_cuts.py -i <output_dir> -o <cuts_dir>
   ```

   This writes `txbb_cdf_wp_cuts.json` (with the percentiles and cuts), a CDF plot, and logs the fraction of A and C
   in each WP bin, for the quantile map and for the linear morph.
3. Add `"txbb_cdf_wp_cuts": "<cuts_dir>/txbb_cdf_wp_cuts.json"` to the config options and rerun. The column
   `HiggsSubLeading_btagBBTXbb_CDF_rescaled_WPs` is then saved next to `btagBBTXbb_morph_5wp`.

Setting the option also switches everything downstream of the subleading-Higgs TXbb to this column, in all four regions
(A, B, C, D), so that neither the linear morph nor the TXbb shape is used:

- the `txbbwp<i>` bins (`hh4b_boosted_TXbb_bin`) and the signal/background categories (`hh4b_boosted_category_*`,
  `hh4b_boosted_background_*`) cut on the WP index. A cut `txbb > WP_k` becomes `index > k`, `txbb < WP_k` becomes
  `index <= k`, and the bin `(EDGES[i], EDGES[i+1]]` becomes `index == i + 1`. The TXbb cuts of the categories
  (`txbb_c1`, `txbb_vbf`, `txbb_c2a` of `optimised_signal_cuts`) must therefore be working points
  (0.8, 0.95, 0.975, 0.99), otherwise building the config raises an error. The default cuts of `_get_params` are not
  working points, so `optimised_signal_cuts` is required. `old_wp_def` must be `False`.
- the morphing DNN has to be trained with the new variable: use `bkg_morphing_boosted_dnn_input_variables_other_group_txbb_cdf_wp`
  (it replaces the subleading `btagBBTXbb_morph_5wp` by the CDF column and the leading `btagBBTXbb_morph_5wp` by the nominal
  `btagBBTXbb`, neither digitised nor morphed). In ML_pytorch
  the input file is `bkg_morphing_dnn_boosted_bdt_inputs_50GeV_wps_cdf` and the training yml
  `configs/hh4b_bkg_reweighting/boosted_vbf/..._lower_band_txbbwps_cdf.yml`, with the outputs of this step as `data_dirs`.
  A model trained on `btagBBTXbb_morph_5wp` cannot be used: its input has a different meaning in A.
- `scripts/optimise_boosted_categories.py --txbb-cdf` uses the CDF column for the QCD estimate (B_postW).

The config `boosted_other_group_50GeV_lower_sideband_txbbwps_cdf_model.py` is the second pass: it needs the
retrained morphing model (without it, only the ABCD regions are defined). Without the option, nothing changes.

## 0. Datasets and skimming

The datasets are built in the same way as in the resolved analysis, see
[Build datasets](../HH4b_common/README.md#build-datasets).

The skimming is also the same as in [Skimming](../HH4b_common/README.md#skimming); just use the boosted skim
config [`configs/HH4b/HH4b_boosted_save_skimmed.py`](../HH4b/HH4b_boosted_save_skimmed.py):

```bash
cd configs/HH4b
run_pocket_coffea skim HH4b_boosted_save_skimmed.py params/t3_run_options_skim_resolved.yaml <output_dir>
```

Then merge the outputs, hadd the skimmed files and point the dataset JSONs to them as described in the common README.

## 1. Run the ABCD regions

> [!TIP]
> @Tier-3/AnalysisConfigs

Unlike the resolved analysis, the boosted analysis needs no SPANet pairing step. Start with a config without any
model (the BDT can already be on) and run it with the boosted template. Without a morphing model only the 4 ABCD
regions are produced (see [Categories](#categories)):

```bash
cd configs/VBF_HH4b_boosted
run_pocket_coffea <config_name> VBF_HH4b_boosted_config.py params/t3_run_options.yaml <output_dir>

# e.g.
run_pocket_coffea boosted_other_group_50GeV_lower_sideband VBF_HH4b_boosted_config.py params/t3_run_options.yaml \
  /work/tharte/datasets/samples_vbf_boosted/other_group/full_2024_50GeV_lower_sideband
```

The output columns are the input for the morphing DNN training.

## 2. Train the background-morphing DNN

> [!TIP]
> @Tier-3/ML_pytorch &rarr; [README](https://github.com/matteomalucchi/ML_pytorch/blob/main/README.md)

> [!IMPORTANT]
> The boosted morphing DNN is a **multiclass** model (QCD and ttbar classes in the numerator and denominator
> regions), so that the ttbar contribution can be subtracted from the data. Until it is merged, use the branch
> [`claude/add-multiclass-support-mgeOZ`](https://github.com/thierryharte/ML_pytorch/tree/claude/add-multiclass-support-mgeOZ)
> of ML_pytorch.

The class layout, the alpha computation and the merging are also described in the "Boosted HH4b" sections of the
ML_pytorch README.

Create a config in `ML_pytorch/configs/hh4b_bkg_reweighting/boosted_vbf/` that points `data_dirs` to the output of
[step 1](#1-run-the-abcd-regions), and run the 20 trainings as in the resolved analysis:

```bash
sbatch run_20_trainings_in_4_parallel.sh <config_file> <output_folder>
```

### Compute the ttbar fraction alpha

The classifier is trained with all classes balanced to the same total weight, so the ttbar output is normalised as if
ttbar were all of the data. To subtract it correctly, the morphing weight needs the physical ttbar fraction
$\alpha = N_{t\bar{t}} / N_\mathrm{data}$. Compute it in the folder with the coffea files that the DNN was trained on:

```bash
cd <training_input_dir>   # e.g. /work/tharte/datasets/samples_vbf_boosted/other_group/full_2024_50GeV_lower_sideband
python /work/tharte/datasets/ML_pytorch_matteo/ml_pytorch/scripts/compute_alpha.py
```

The output looks like this:

```text
region                            N_data      N_ttbar     alpha    MCstat
boosted_signal_region_D             2361        661.3    0.2801    0.0040
boosted_signal_sideband_region_C   21744       2343.8    0.1078    0.0008
boosted_control_region_B           46741      19818.3    0.4240    0.0011
boosted_control_sideband_region_A 545355      60568.0    0.1111    0.0002
```

The value needed is the `alpha` of the numerator region of the morphing. For the default `"btag"` direction that
is **region C** (`boosted_signal_sideband_region_C`, here `0.1078`); for the `"mass"` direction it is region B.

### Merge the 20 models

Merge the 20 models into one ONNX model and pass alpha. Run this inside the `best_models` folder of the training
output:

```bash
cd <output_folder>/best_models
ml_onnx -i ./ -o ./ -ar -v <input_variables> -mc -al <alpha_C> -ald 0.0

# e.g.
ml_onnx -i ./ -o ./ -ar -v bkg_morphing_dnn_boosted_bdt_inputs_50GeV -mc -al 0.1078 -ald 0.0
```

- `-ar`: average the ratios of the outputs of all the models in the folder
- `-v`: name of the input-variables module, the same one used for the training
- `-mc`: the models are multiclass
- `-al`: alpha of the numerator region (region C for the `"btag"` direction)
- `-ald`: alpha of the denominator region A (`0.0` switches off the ttbar subtraction in the denominator)

This produces `average_model_from_onnx.onnx`, the morphing model used in PocketCoffea.

## 3. Apply the morphing and check the closure

> [!TIP]
> @Tier-3/AnalysisConfigs

As in the resolved analysis, put the merged model in the `bkg_morphing_dnn` entry of the `onnx_model_dict` of a
(new) config, e.g.
[`boosted_other_group_50GeV_lower_sideband_btag_direction.py`](../HH4b_common/config_files/boosted_other_group_50GeV_lower_sideband_btag_direction.py):

```python
onnx_model_dict |= {
    "bkg_morphing_dnn": "/work/tharte/datasets/ML_pytorch_matteo/out/bkg_reweighting/boosted_vbf/<training>_btagging_<...>/best_models/average_model_from_onnx.onnx",
}
```

For a model trained along the mass, also set `"bkg_morphing_direction": "mass"` in `config_options_dict` (see
[ABCD regions and morphing direction](#abcd-regions-and-morphing-direction)). Then run the same template again:

```bash
run_pocket_coffea boosted_other_group_50GeV_lower_sideband_btag_direction VBF_HH4b_boosted_config.py params/t3_run_options.yaml \
  /work/tharte/datasets/samples_vbf_boosted/other_group/full_2024_50GeV_lower_sideband_btag_direction
```

Without `optimised_signal_cuts` (and without a BDT) this gives only the ABCD regions plus the `*_postW` regions.

### Closure test

Compare the morphed background with the data/MC with
[`plot_2bMorphedvs4b.py`](../../scripts/plot_2bMorphedvs4b.py) in boosted mode. Run it in the output folder:

```bash
sbatch -p standard --account=t3 --time=02:00:00 --mem 64gb --cpus-per-task=8 \
  --wrap="python /work/tharte/datasets/AnalysisConfigs_bossted_dev/scripts/plot_2bMorphedvs4b.py -i ./ --boosted \
  --only-vars /work/tharte/datasets/ML_pytorch_matteo/ml_pytorch/defaults/bkg_morphing_dnn_boosted_bdt_inputs_for_plots.py"
```

Like the config, the script assumes a morphing along the TXbb by default (`--bbtag-direction`); for a model trained
along the mass, add `--mass-direction`. `--only-vars` limits the plots to the variables listed in the given file.

## 4. Train the signal-vs-background DNN

> [!TIP]
> @Tier-3/ML_pytorch &rarr; [README](https://github.com/matteomalucchi/ML_pytorch/blob/main/README.md)

The boosted signal-vs-background DNN is also multiclass (ggF, VBF, ttbar, QCD), and the workflow saves its four
class scores and the `sig_bkg_dnn_score_ggFvsBKG` / `sig_bkg_dnn_score_VBFvsBKG` discriminants. Set up a config
pointing to the output of [step 3](#3-apply-the-morphing-and-check-the-closure) and train a single model:

```bash
sbatch run_sig_bkg_classifier.sh <config_file>

# e.g.
sbatch run_sig_bkg_classifier.sh boosted/1e-3_e20drop95_minDelta1em5_boosted_signal_background_classifier_50GeV.yml
```

> [!NOTE]
> The morphing trainings and the signal-vs-background trainings live in different folders, both for the configs and
> for the outputs (`out/bkg_reweighting/...` vs `out/sig_bkg_classifier/...`).

The ONNX model is saved in `state_dict/` of the training output (e.g. `state_dict/model_best_epoch_<N>.onnx`).
Put it in the `sig_bkg_dnn` entry of the `onnx_model_dict` of the config, together with the matching
`sig_bkg_dnn_input_variables`, and rerun PocketCoffea as in
[step 3](#3-apply-the-morphing-and-check-the-closure), so that the DNN scores are saved for the optimisation.

## 5. Optimise the signal categories

> [!TIP]
> @Tier-3/AnalysisConfigs

> [!DANGER]
> This is very much work in progress and has to be updated.


[`optimise_boosted_categories.py`](../../scripts/optimise_boosted_categories.py) optimises the cuts that define
`cat1`, `catvbf`, `cat2` and `cat3` on the ggF and VBF DNN discriminants and on the subleading TXbb, using the expected
limit from the Asimov significance as figure of merit. Run it in the output folder of the run with the
signal-vs-background DNN:

```bash
python /work/tharte/datasets/AnalysisConfigs_bossted_dev/scripts/optimise_boosted_categories.py -i ./ -o category_optimisation --holdout
```

`--holdout` optimises on the even event numbers and reports the result on the odd ones, to check for overtuning. See
`--help` for the other options (figure of merit, VBF weight, grids, minimum background, ...).

The optimised cuts are printed and saved in `category_optimisation/optimised_cuts_<discriminant>_<metric>.json`
(under `"optimised"` &rarr; `"cuts"`). The keys are the ones the config expects, so copy them directly into the
`optimised_signal_cuts` option of the config, e.g.
[`boosted_other_group_50GeV_lower_sideband_btag_direction_wps_different_model.py`](../HH4b_common/config_files/boosted_other_group_50GeV_lower_sideband_btag_direction_wps_different_model.py):

```python
config_options_dict |= {
    # ...
    "optimised_signal_cuts": {"txbb_c1": 0.875, "bdt_ggf_c1": 0.9904, "txbb_vbf": 0.875, "bdt_vbf": 0.98883,
                              "txbb_c2a": 0.715, "bdt_ggf_c2b": 0.92265, "bdt_ggf_bkg": 0.053765},
}
```

> [!NOTE]
> The keys keep the `bdt_` names of the BDT cuts (`_get_params` in
> [`custom_cuts_common.py`](../HH4b_common/custom_cuts_common.py)), but with `optimised_signal_cuts` they are applied
> to the DNN scores `sig_bkg_dnn_score_ggFvsBKG` and `sig_bkg_dnn_score_VBFvsBKG`.

## 6. Final run with the optimised categories and systematics

> [!TIP]
> @Tier-3/AnalysisConfigs

Run with the template with systematics. It is needed for the datacards, and the final workflow uses it too:

```bash
run_pocket_coffea <config_name> VBF_HH4b_boosted_config_with_syst.py params/t3_run_options.yaml <output_dir>

# e.g.
run_pocket_coffea boosted_other_group_50GeV_lower_sideband_btag_direction_wps_different_model VBF_HH4b_boosted_config_with_syst.py \
  params/t3_run_options.yaml \
  /work/tharte/datasets/samples_vbf_boosted/other_group/full_2024_50GeV_lower_sideband_btag_direction_different_model_optimised_cuts_with_systs
```

Then merge the outputs:

```bash
cd <output_dir>
pocket-coffea merge-outputs -o output_all.coffea *.coffea -f
```

## 7. Signal-region plots

> [!TIP]
> @Tier-3/AnalysisConfigs

[`plot_boosted_signal_regions.py`](../../scripts/plot_boosted_signal_regions.py) plots the signal categories. Data is
blinded in every category except `cat3` (change this with `-u/--unblind`):

```bash
python /work/tharte/datasets/AnalysisConfigs_bossted_dev/scripts/plot_boosted_signal_regions.py -i ./ -o SR_plots
```

## 8. Datacards

> [!TIP]
> @Tier-3/AnalysisConfigs

Merge the outputs into a single file (watch out, that if you do this, you need to be sure, that there is not already an `input_all.coffea` in the folder. Otherwise, you might duplicate events)
```bash
pocket-coffea merge-outputs -o output_all.coffea output_*.coffea -f
```

Build the datacards from the merged output with [`build_datacards.py`](../../scripts/build_datacards.py) in boosted
mode (`-b`). The binning options depend on the fit variable and are worth tuning:

```bash
python /work/tharte/datasets/AnalysisConfigs_bossted_dev/scripts/build_datacards.py -b -i ./output_all.coffea -o datacards_03unc -mu 0.3 -my 1
```

- `-mu`: merge adjacent bins until the relative statistical uncertainty of the total background in every bin is at
  most this value
- `-my`: also require at least this total background yield per bin
- `-nb`, `-sb`, `-fr`: fixed rebinning, single bin, fit range (see `--help`)

The output has one folder per fit variable (e.g. `SubLeadingHiggsMass/`), each with 4 datacards with the different regions along with the shapes in root files. We will have to merge them, but first we need combine for that.

## 9. Limits with Combine

> [!TIP]
> @Tier-3/CMSSW

### Installation (only once)

```bash
cd /work/<user>/software    # cmssw-el7 has to be started from inside /work
cmssw-el7
cmsrel CMSSW_11_3_4
cd CMSSW_11_3_4/src
cmsenv
git -c advice.detachedHead=false clone --depth 1 --branch v9.2.1 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
cd CombineHarvester
git checkout v2.1.0
cd ..
scramv1 b clean; scramv1 b # always make a clean build
```

### Environment (every session)

```bash
cd /work/<user>
cmssw-el7
cd software/CMSSW_11_3_4/src
cmsenv
# back to the folder with the datacard
cd <output_dir>/datacards/<fit_variable>
```

### Merge the datacards
```bash
combineCards.py catvbf=HH4b_merged_catvbf_run3.txt cat1=HH4b_merged_cat1_run3.txt cat2=HH4b_merged_cat2_run3.txt cat3=HH4b_merged_cat3_run3.txt > datacard_combined_HH4b_merged.txt
```


### Build the workspace

```bash
text2workspace.py VBFHH_run3.txt -m 125 -P HiggsAnalysis.CombinedLimit.hh_model_13p6TeV:model_default -o datacard_boosted.root \
  --physics-option doNNLOscaling=True --physics-option doklDependentUnc=True \
  --physics-option doBRscaling=True --physics-option doHscaling=True
```

### Expected limits

The commands below are the **stat-only** expected limits. For the version with all systematics, remove
`,allConstrainedNuisances` from `--freezeParameters` and drop the `--freezeNuisanceGroups autoMCStats` option.

```bash
# SM VBF (kappa_2V = 1), stat-only expected limit
combine -M AsymptoticLimits -d datacard_boosted.root -m 125 --run blind \
  --redefineSignalPOIs r_qqhh \
  --setParameters r=1,r_gghh=1,kl=1,kt=1,CV=1,C2V=1 \
  --freezeParameters r,r_gghh,kl,kt,CV,C2V,allConstrainedNuisances \
  --freezeNuisanceGroups autoMCStats -n .C2V1_statonly

# kappa_2V = 0, stat-only expected limit
combine -M AsymptoticLimits -d datacard_boosted.root -m 125 --run blind \
  --redefineSignalPOIs r_qqhh \
  --setParameters r=1,r_gghh=1,kl=1,kt=1,CV=1,C2V=0 \
  --freezeParameters r,r_gghh,kl,kt,CV,C2V,allConstrainedNuisances \
  --freezeNuisanceGroups autoMCStats -n .C2V0_statonly

# SM ggF, stat-only expected limit
combine -M AsymptoticLimits -d datacard_boosted.root -m 125 --run blind \
  --redefineSignalPOIs r_gghh \
  --setParameters r=1,r_qqhh=1,kl=1,kt=1,CV=1,C2V=1 \
  --freezeParameters r,r_qqhh,kl,kt,CV,C2V,allConstrainedNuisances \
  --freezeNuisanceGroups autoMCStats -n .kl1_statonly

# SM inclusive signal strength mu, stat-only expected limit
combine -M AsymptoticLimits -d datacard_boosted.root -m 125 --run blind \
  --redefineSignalPOIs r \
  --setParameters r_gghh=1,r_qqhh=1,kl=1,kt=1,CV=1,C2V=1 \
  --freezeParameters r_gghh,r_qqhh,kl,kt,CV,C2V,allConstrainedNuisances \
  --freezeNuisanceGroups autoMCStats -n .mu_statonly
```
