# AnalysisConfigs

Repository containing analysis configurations for PocketCoffea

## Setup

### Fast installation with micromamba

```bash
# Clone PocketCoffea
git clone git@github.com:PocketCoffea/PocketCoffea.git

# Clone AnalysisConfigs
git git@github.com:matteomalucchi/AnalysisConfigs.git

# Create a local environment and install the packages
cd AnalysisConfigs
micromamba env create -f pocket-coffea-environment.yml
micromamba activate pocket-coffea-env
```

### General installation

The first step is installing the main `PocketCoffea` package in your python environment.

Please have a look at the [Installation guide](https://pocketcoffea.readthedocs.io/en/latest/installation.html).

The `configs` package has been created to separate the core of the framework from all the necessary configuration files
and customization code needed the different analyses. The configuration is structured as a python package to make easier
the import of customization code into the framework configuration and also to make the sharing of analysis code easier.

Once you have a `PocketCoffea` local installation, you can install the `configs` and `utils` package with:

```bash
cd AnalysisConfigs
pip install -e .
```

This will install the `configs` package in editable mode.

## HH4b analysis

The instruction to run the HH4b analysis are in the [HH4b README](./configs/HH4b_common/README.md).

## Scripts

* [`scripts/postprocess_parquet.py`](./scripts/postprocess_parquet.py) (`postprocess_parquet` once installed): post-process the columns of a PocketCoffea output: merge the parquet chunks into one `merged.parquet` per dataset/region/variation and/or apply ONNX models (DNN or SPANet) to the columns (coffea or parquet). Main options: `-c <yaml>` (all the options and the models, see [`scripts/postprocess_parquet_example.yaml`](./scripts/postprocess_parquet_example.yaml); the arguments override it), `-i`, `-o`, `-j`, `--merge/--no-merge`, `--update-config` (point the `config.json` to the output), `--overwrite`, `--tmp-dir`, `--stall-timeout` (exit if nothing progresses for N s, default 3600; failed files are reported at the end and a restart reuses what was already produced), `--read-timeout` (read with `xrdcp` the `/pnfs` inputs whose read through the mount hangs for more than N s, default 300). Details in [docs/postprocess_parquet.md](./configs/HH4b_common/docs/postprocess_parquet.md).

* [`scripts/plot_2bMorphedvs4b.py`](./scripts/plot_2bMorphedvs4b.py): compare the 4b region to the 2b region before and after the background morphing (pre/post weights) using the columns of the coffea outputs. Example: `python scripts/plot_2bMorphedvs4b.py -i <dir>/output_DATA_*.coffea -o <plot_dir> --vbf -w 10 --weight-column events_morphing_weight`. Main options (see [`utils_configs/plot/args_plot.py`](./utils_configs/plot/args_plot.py)): `-i`/`-im` (data/MC inputs), `-o`, `-n` (normalisation: `num_events`, `sum_weights`, `density`), `-w` (workers), `-r` (region suffix), `--vbf`/`--boosted`/`-r2` (region set), `-om` (ONNX DNN to compute the score), `-wc` (weight column), `-mw` (mask large weights), `-t` (test). Before plotting, it prints a table with the number of events and the sum of the weights in each region and saves it to `<output>_<normalisation>/event_counts_DATA.txt` (and `event_counts_MC.txt` when MC is given). The pull plot shows only the postW (morphed) histograms, not the preW ones. The χ² of every distribution w.r.t. the reference is saved to `<output>_<normalisation>/chi2_values.json` and summarised for each category in a bar plot `chi2_pvalue_<category>` (p-value of the χ² of preW in red and postW in blue for all the variables, with a threshold line at `PVALUE_THRESHOLD`=0.05; the values printed on the bars are green above the threshold and black below; p-values below `PVALUE_FLOOR`=1e-10 are drawn at the floor), next to the weight plots.

* [`scripts/plot_DNN_score.py`](./scripts/plot_DNN_score.py): plot the DNN score (and the other columns) of data, reweighted 2b background and MC signal, and save the yields to `yields.txt`. Example: `python scripts/plot_DNN_score.py -i <data.coffea> -im <mc.coffea> -o <plot_dir> [--mixed -ir <rew.coffea>]`. Same options as `plot_2bMorphedvs4b.py` (`utils_configs/plot/args_plot.py`). It also prints and saves the number of events and sum of weights per region to `<output>_<normalisation>/event_counts_{DATA,MC}.txt` (`event_counts_BKG.txt` too with `--mixed`).

The shared plotting helpers (`plot_regions_names`, `print_event_counts`) are in [`utils_configs/plot/plot_utils.py`](./utils_configs/plot/plot_utils.py).

The HH→4b jet pairing and resonance reconstruction helpers are in [`utils_configs/reconstruct_resonances.py`](./utils_configs/reconstruct_resonances.py). `run2_matching_algorithm(jets, pt_in_hh_frame=False)` is the Run 2 pairing: it picks the pairing with the smallest D_HH (k = 1.04). If the two smallest D_HH differ by less than 30 GeV, it picks whichever of those two pairings has the higher Higgs-candidate pt. D_HH always uses the higher-pt candidate as the leading one. With `pt_in_hh_frame=True` (as in HIG-20-005), the tie-break uses the candidate pt in the HH rest frame. With the default `False`, it uses the highest lab-frame pt.

## MC Truth corrections for pT regression derivation

The instruction to run the derivation of the MC Truth corrections for pT regression are in the [jme README](./configs/jme/README.md).

## MET Type-1 correction analysis

The instruction to run the analysis for the MET Type-1 correction are in the [MET_studies README](./configs/MET_studies/README.md).