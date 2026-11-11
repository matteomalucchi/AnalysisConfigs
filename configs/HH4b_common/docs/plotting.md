# Plotting scripts

Two scripts in [`AnalysisConfigs/scripts/`](../../../scripts/) make the plots of the background
morphing and of the signal/background DNN, starting from the coffea (and parquet) files produced by
`run_pocket_coffea`:

| Script | What it plots |
| --- | --- |
| [`plot_2bMorphedvs4b.py`](../../../scripts/plot_2bMorphedvs4b.py) | The 4b data compared with the 2b data before (`preW`) and after (`postW`) the background morphing, in the CR and SR, for all the saved columns, plus the morphing weights. |
| [`plot_DNN_score.py`](../../../scripts/plot_DNN_score.py) | The signal/background DNN score: 4b data (blinded in SR) vs the morphed 2b background stacked with the MC signal, plus the bin-by-bin $s/\sqrt{b}$. |

Both scripts share the same argument parser
([`utils_configs/plot/args_plot.py`](../../../utils_configs/plot/args_plot.py)), but not every
argument is used by both: the tables below list only the arguments each script uses.

The categories are described in [categories.md](categories.md).

## Running

The scripts are run from the Tier-3 home directory (where `AnalysisConfigs` is cloned) in the
`pocket-coffea-env` environment. Reading the columns needs a lot of memory, so submit them to Slurm:

```bash
sbatch -p short --account=t3 --time=00:05:00 --mem 25gb --cpus-per-task=8 \
  --wrap="python AnalysisConfigs/scripts/<script>.py <arguments>"
```

Increase `--time` (e.g. `standard` partition, `--time=01:00:00`) when plotting all the columns of
large datasets. Match `--cpus-per-task` with `-w/--workers`.

The output directory is always `<output>_<normalisation>` (e.g. `-o plots -n sum_weights` writes to
`plots_sum_weights/`), with one subdirectory `<region>_columns/` per plotted region.

> [!NOTE]
> When `-i` is a directory, pass it **with a trailing slash** (`-i /path/to/output/`): the input
> directory is taken as `os.path.dirname` of the argument, so without the slash the parent
> directory would be used.

## Plot morphed 2b vs 4b: `plot_2bMorphedvs4b.py`

For each region (CR, SR) it plots every column saved in the files, comparing:

- the 4b data (black points, reference of the ratio);
- the 2b data after the morphing, `postW` (blue filled histogram);
- the 2b data before the morphing, `preW` (red line). If a `preW` category is missing in the files,
  it is built from the `postW` one with weight 1.

The ratio panel shows Data/Pred. and a $\chi^2$ is added to the plots. Additionally it produces:

- `weights_<category>`: the distribution of the morphing weights for each `postW` category;
- `N_<collection>`: the multiplicity of each jagged collection (e.g. the jets), counting the
  non-padded objects;
- `<score>_TRANSFORM`: the score variables drawn with equal-width bins.

Binning: 20 bins in the range between the 0.1% and 99.9% quantiles of the 4b distribution (NaN
values are dropped; if the range is degenerate, e.g. a constant column, it is widened to ±0.5 around
the value); integer variables (any integer dtype) get one bin per value. If `-im` is a single coffea file containing a score column, the
score variables use 20 bins with constant signal yield (computed on the MC signal).

### Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `-i`, `--input-data` | *required* | Coffea files of the data, or a directory: in this case all the `.coffea` files in it whose name contains `DATA` or `Mixed` are used. |
| `-im`, `--input-mc` | `None` | MC signal coffea files (or a directory: all the `.coffea` files without `DATA` in the data directory). Used for the constant-signal binning of the score and for `--comparison`. |
| `-o`, `--output` | `plots_2bVS4b` (`test_2bVS4b` with `--test`) | Output directory (the suffix `_<normalisation>` is appended). |
| `-n`, `--normalisation` | `sum_weights` | `sum_weights`: scale the 2b histograms to the sum of weights of the 4b one. `num_events`: scale them to the number of 4b events. `density`: normalise all histograms to unit area. |
| `-r`, `--region-suffix` | `""` | Suffix added to the region names (e.g. `-r _VR1` reads `4b_VR1_control_region`, ...). |
| `-r2`, `--run2` | `False` | Use the Run2 (`DHH` pairing) categories, ending in `Run2`. |
| `--vbf` | `False` | VBF analysis: plot the `vbf_` and `ggf_` CR and SR separately, and their sum (`CR_MERGED`, `SR_MERGED`). |
| `--boosted` | `False` | Boosted analysis: plot the boosted ABCD regions (`boosted_vbf_incl_*`). |
| `--mixed` | `False` | Mixed datasets: compare `4b_*_region` with `4b_*_region_postW/preW` from the `Mixed` files. |
| `-s`, `--spread` | `False` | Plot only the `SPREAD` regions: the score of the 2b morphed data with the mean weight per event compared with each k-fold weight (needs the `events_bkg_morphing_spread_dnn_weights` column), with the median and the 16%/84% quantiles of the k-folds in the ratio. Available in the default and `--run2` modes. |
| `-c`, `--comparison` | `False` | Add the `DATAMC` regions comparing the 2b morphed data and 4b data with the MC signal in SR (needs `-im`). |
| `-om`, `--onnx-model` | `""` | Path to an ONNX model of the signal/background DNN: the score `events_sig_bkg_dnn_score` is recomputed from the saved columns and plotted. |
| `-v`, `--input-variables` | `sig_bkg_dnn_input_variables` | Name of the list in [`dnn_input_variables.py`](../dnn_input_variables.py) with the inputs of the `-om` model. |
| `-wc`, `--weight-column` | `weight` | Column used as weight when filling the histograms (e.g. a different morphing weight). |
| `--novars` | `False` | Old file format, without the saved variations. |
| `-w`, `--workers` | `8` | Number of parallel processes (one variable per process). |
| `-l`, `--linear` | `False` | Linear y axis (default is log). |
| `-t`, `--test` | `False` | Quick test: only the variables containing `score`, `HH_mass`, `HiggsLeading` or the weight, at most 10000 events and 1 parquet file per category. |
| `-cms`, `--cmstext` | `Private` | Text next to the CMS label. |
| `-mw`, `--mask-large-weights` | `True` | Remove the events with NaN or weight > 100 from the predicted histograms (always on). |

Only one of `--run2`, `--boosted`, `--vbf`, `--mixed` is used, in this order of priority; without
any of them the default (first approach, ggF) regions are plotted.

### Examples

```bash
# Default regions, all the columns
python AnalysisConfigs/scripts/plot_2bMorphedvs4b.py -i /work/$USER/HH4b/out_morphing/ -o plots_2bVS4b

# VBF analysis (vbf_/ggf_ and merged CR/SR), linear scale, 16 workers
python AnalysisConfigs/scripts/plot_2bMorphedvs4b.py -i /work/$USER/HH4b/out_vbf/ -o plots_vbf --vbf -l -w 16

# Run2 pairing, constant-signal binning of the score from one MC signal file
python AnalysisConfigs/scripts/plot_2bMorphedvs4b.py -i /work/$USER/HH4b/out/ -r2 \
  -im /work/$USER/HH4b/out/output_GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_2022_postEE.coffea

# Spread of the k-fold morphing weights on the score
python AnalysisConfigs/scripts/plot_2bMorphedvs4b.py -i /work/$USER/HH4b/out/ -s -o plots_spread

# Quick test with a different weight column and unit-area normalisation
python AnalysisConfigs/scripts/plot_2bMorphedvs4b.py -i /work/$USER/HH4b/out/ -t -n density -wc weight_alt
```

## Plot DNN score: `plot_DNN_score.py`

For each region (CR, SR) and each score column it plots:

- the 4b data (black points, reference of the ratio). In the SR the last 20% of the bins (the
  highest score) are blinded;
- stacked: the 2b morphed data (`postW`, the background) and the MC signal.

The 2b morphed background is scaled by the 4b/2b ratio computed in the CR (see `-n` and `-sn`).
The MC signal weights are rescaled by `lumi / 26.27` (the 2022 luminosity is hardcoded), with
`lumi` taken from the data files.

Each score is plotted three times:

- `<score>`: 20 bins with constant signal yield, computed on the MC signal;
- `<score>_TRANSFORM`: the same bins drawn with equal width;
- `<score>_UNIFORM`: 20 uniform bins between 0 and 1.

For each of them also `<score>_sob`, `<score>_sig` and `<score>_bkg` are produced, with the
bin-by-bin $s/\sqrt{b}$, signal and background yields; the combined $s/\sqrt{b}$ is shown on the
plots. The yields are written to `yields.txt` in the output directory.

### Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `-i`, `--input-data` | *required* | Coffea files of the data, or a directory: in this case all the `.coffea` files in it whose name contains `DATA` are used. |
| `-im`, `--input-mc` | *required* | MC signal coffea files (a list of files, not a directory). The value of $\kappa_\lambda$ in the legend is read from the first file name (`kl-<value>`). |
| `-ir`, `--input-rew` | `None` | Coffea files with the morphed background; used only with `--mixed` (otherwise the background is taken from the `-i` files). |
| `-o`, `--output` | `plots_DNN_data_and_mc` (`test_DNN_data_and_mc` with `--test`) | Output directory (the suffix `_<normalisation>` is appended). |
| `-n`, `--normalisation` | `sum_weights` | Ratio used to scale the 2b morphed background to the 4b data: `sum_weights` (ratio of the sums of weights), `num_events` (ratio of the number of events) or `density` (no scaling, every histogram normalised to unit area). |
| `-sn`, `--separate-normalisation` | `False` | Compute the 4b/2b ratio separately in the SR and apply it there. By default the CR ratio is used everywhere. |
| `-r`, `--region-suffix` | `""` | Suffix added to the region names. |
| `-r2`, `--run2` | `False` | Use the Run2 (`DHH` pairing) categories and the `Run2` score columns. |
| `--mixed` | `False` | Mixed datasets: the background is `4b_*_region_postW` from the `-ir` files. |
| `-om`, `--onnx-model` | `""` | Path to an ONNX model of the signal/background DNN: the score is recomputed from the saved columns (overwriting the saved one). Without it only the weight and score columns are read. |
| `-v`, `--input-variables` | `sig_bkg_dnn_input_variables` | Name of the list in [`dnn_input_variables.py`](../dnn_input_variables.py) with the inputs of the `-om` model. |
| `--novars` | `False` | Old file format, without the saved variations. |
| `-w`, `--workers` | `8` | Number of parallel processes (one variable per process). |
| `-l`, `--linear` | `False` | Linear y axis (default is log). |
| `-t`, `--test` | `False` | Quick test: only the SR is plotted and the CR ratio is set to 1. |
| `-cms`, `--cmstext` | `Private` | Text next to the CMS label. |
| `-mw`, `--mask-large-weights` | `True` | Remove the background events with weight ≥ 100 (always on). |

### Examples

```bash
# Score saved in the files, normalisation from the CR
python AnalysisConfigs/scripts/plot_DNN_score.py -i /work/$USER/HH4b/out_dnn/ \
  -im /work/$USER/HH4b/out_dnn/output_GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_2022_postEE.coffea \
  -o plots_DNN

# Run2 pairing, SR normalised separately, old file format
python AnalysisConfigs/scripts/plot_DNN_score.py -i /work/$USER/HH4b/out_dnn/ -im <signal_file>.coffea \
  -r2 -sn --novars

# Recompute the score with a new ONNX model
python AnalysisConfigs/scripts/plot_DNN_score.py -i /work/$USER/HH4b/out_dnn/ -im <signal_file>.coffea \
  -om /work/$USER/ML_pytorch/out/best_models/model.onnx -v sig_bkg_dnn_input_variables
```
