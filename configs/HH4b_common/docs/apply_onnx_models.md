# Apply ONNX models to the saved columns

[`scripts/apply_onnx_models.py`](../../../scripts/apply_onnx_models.py) evaluates ONNX models on the
columns saved by a PocketCoffea run, without rerunning the full workflow (e.g. to add the score of a
newly trained model to an existing output).

```bash
python scripts/apply_onnx_models.py -c scripts/apply_onnx_models_example.yaml
# input, output, number of processes, chunk size and temporary directory can be overridden
python scripts/apply_onnx_models.py -c my_config.yaml -i <coffea_dir> -o <output_dir> -j 8 --chunk-size 50000
```

All the options are documented in
[`scripts/apply_onnx_models_example.yaml`](../../../scripts/apply_onnx_models_example.yaml).

## How it works

- **Input**: the directory with the coffea files of the run (or a list of coffea files), as for the
  other scripts. The columns are read from the coffea files if they are saved there, otherwise from
  the parquet files written with `save_chunk` (the directory is taken from
  `dump_columns_as_arrays_per_chunk` in the `config.json` of the run, or from `parquet_input`).
- **Inputs of the models**: each model takes the same input dictionary as the standard run, loaded
  from `dnn_input_variables.py` with `input_variables: <module>:<name>` or
  `input_variables: <path/to/file.py>:<name>`. The `events` record is rebuilt from the columns
  (`<Collection>_<feature>`, unflattened with `<Collection>_N` for the coffea columns) and the model is
  evaluated with `get_onnx_prediction`, so DNN and SPANet models are handled exactly as in the
  workflow. All the columns used by the input dictionary must be saved in the output of the run,
  except constant per-dataset inputs (e.g. `events_era`), which can be given with `extra_columns`
  (`{column: {dataset pattern: value}}`, used only as model inputs and not saved).
- **Selection**: each model is applied only to the `regions` and `datasets` listed in the config (names
  or glob patterns, all if not given), and only the selected `variations` are processed.
- **Output**: all the original columns plus the scores, saved with the `column_name` of each model
  (`score_index` sets which output columns are saved):
  - columns in the coffea files: copies of the coffea files with the new columns;
  - columns in parquet files: `<output>/parquet_files/<dataset>/<category>/<variation>/*.parquet`
    (copies of all the input files, with the new columns where a model is applied), the coffea files
    and a `config.json` pointing to the new parquet directory, so the plotting scripts can read the
    output as a normal run.
  The yaml used is copied as `apply_onnx_models_config.yaml`.
- **/pnfs**: since the slurm jobs cannot write on /pnfs, if the output is on /pnfs (or an xrootd url)
  the files are written in a local temporary directory (`tmp_dir`, `$TMPDIR` by default) and copied
  with `xrdcp` through the `redirector`. The parquet files are copied as soon as they are written.
  `xrdcp` needs a grid proxy: the script stops at the start if `$X509_USER_PROXY` is not valid for at
  least 2 h (`voms-proxy-init -voms cms -rfc --valid 168:0`). The script can be run directly in a
  slurm job (e.g. `sbatch -c 4 --mem-per-cpu 8000 --wrap "python scripts/apply_onnx_models.py -c <cfg>"`):
  the proxy must be on a shared file system (e.g. `/t3home`) and, since `/tmp` of the workers is only
  2 GB, `tmp_dir` should point to `/scratch/<user>`. If the script fails, the temporary directory is
  removed.
- **Memory**: the parquet files where a model is applied are processed one row group at a time, so
  the memory does not scale with the file size (the merged files can be several GB); the files where
  no model is applied are copied without being read.
- **Parallelism**: `workers: 1` runs everything in a single process. With more workers, the chunks of
  `chunk_size` events (coffea columns) or the parquet files are processed in parallel processes; each
  process creates its ONNX sessions once.
