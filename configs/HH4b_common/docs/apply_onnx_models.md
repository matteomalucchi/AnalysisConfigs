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
  workflow. All the columns used by the input dictionary must be saved in the output of the run.
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
- **Parallelism**: `workers: 1` runs everything in a single process. With more workers, the chunks of
  `chunk_size` events (coffea columns) or the parquet files are processed in parallel processes; each
  process creates its ONNX sessions once.
