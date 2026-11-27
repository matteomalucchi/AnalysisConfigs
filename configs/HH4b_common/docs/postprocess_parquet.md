# Post-process the saved columns: merge the parquet chunks and apply ONNX models

[`scripts/postprocess_parquet.py`](../../../scripts/postprocess_parquet.py) (`postprocess_parquet` once
`AnalysisConfigs` is installed with `pip install -e .`) post-processes the columns saved by a
PocketCoffea run, without rerunning the workflow. It can:

- **merge** the thousands of small `parquet` chunks written with `save_chunk` into one
  `merged.parquet` per `<dataset>/<category>/<variation>`, keeping the directory structure and the
  events order, so that the merged directory can be used in place of the original one (reading many
  small files from dCache, `/pnfs`, is dominated by the per-file latency);
- **apply ONNX models** (DNN or SPANet) to the columns (coffea or parquet), e.g. to add the score of a
  newly trained model to an existing output, optionally merging the result.

```bash
# merge only: into <...>/merged_parquet_files next to the parquet directory,
# and point the config.json of the output to the merged files
python scripts/postprocess_parquet.py -i /work/$USER/out_hh4b/<name> --update-config

# apply the ONNX models of the config
python scripts/postprocess_parquet.py -c scripts/postprocess_parquet_example.yaml
# apply them, merge the result and point the config.json to it
python scripts/postprocess_parquet.py -c my_config.yaml -i <output_dir> -j 8 --merge --update-config
```

Merging is also run automatically at the end of `run_pocket_coffea` with `--merge-parquet`, see
[the README](../README.md#merge-the-parquet-chunks).

## Options

Only the main options are command line arguments; they override the yaml config (`-c`), which holds
all the options and the ONNX models, documented in
[`scripts/postprocess_parquet_example.yaml`](../../../scripts/postprocess_parquet_example.yaml).

| Argument | yaml key | Description |
| --- | --- | --- |
| `-c/--config` | | yaml config. Not needed to merge only. |
| `-i/--input` | `input` | PocketCoffea output directory (with `config.json`), a coffea file, or a parquet files directory. |
| `-o/--output` | `output` | Output directory. Default for parquet columns: `merged_parquet_files` (merge only), `onnx_parquet_files` or `onnx_merged_parquet_files` (with models) next to the parquet directory. Needed for the coffea columns. |
| `-j/--workers` | `workers` | Parallel processes for the ONNX evaluation (default 1). Without models, threads reading the files to merge (`merge_threads`, default 32). |
| `--merge/--no-merge` | `merge` | Merge the parquet files into one `merged.parquet` per directory (default: on without models, off with models). |
| `--update-config` | `update_config` | Overwrite `dump_columns_as_arrays_per_chunk` and `save_chunk` in the `config.json` of the input with the output directory (keeping the same `root://` redirector, if any), so that the scripts reading the columns through the `config.json` (e.g. ML_pytorch) use it directly. The original paths are kept as `<key>_unmerged`. With models it also sets `copy_unprocessed: true`, so that the output contains all the columns. |
| `--overwrite` | `overwrite` | Produce again the output files that already exist (skipped by default). |
| `--remove-original` | `remove_original` | Merge only: after all the merges, check for each `<dataset>/<category>/<variation>` that the merged file has the same number of events as the original files (counted while merging) and that the original files did not change, then remove the original files and the directories left empty. If any check fails nothing is removed and the script fails. |
| `--tmp-dir` | `tmp_dir` | Local directory where the files are produced before being copied to `/pnfs` (see below). |
| `--read-timeout` | `read_timeout` | Seconds after which a read of an input file through the `/pnfs` mount is considered hanging and the file is read with `xrdcp` instead (default 300, 0 = always through the mount, see below). |
| `--stall-timeout` | `stall_timeout` | Exit with an error (code 3) if nothing progresses for this many seconds (default 3600, 0 = never), e.g. a read hanging on the `/pnfs` mount of a batch node (see below). |
| `--debug` | | Debug printout. |

Other yaml options: `redirector`, `merge_threads`, `files_per_step` (files kept in memory while
merging, default 256), `variations`, `parquet_input`, `parquet_subdir`, `max_parquet_files` and the
ONNX options `models`, `chunk_size`, `overwrite_columns`, `copy_unprocessed`, `process_output_all`,
`extra_columns`. Unknown keys are an error.

## How it works

- **Input**: with a PocketCoffea output directory, the parquet directory is read from
  `dump_columns_as_arrays_per_chunk` in its `config.json` (or from `parquet_input`). When merging
  only, the original directory (`<key>_unmerged`) is used if the `config.json` was already updated;
  with models, the current one (e.g. the merged files). With models, if the coffea files contain the
  columns, those are used instead.
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
- **Output with models**: all the original columns plus the scores, saved with the `column_name` of
  each model (`score_index` sets which output columns are saved):
  - columns in the coffea files: copies of the coffea files with the new columns, plus the
    `config.json` and metadata of the run and the yaml used (`postprocess_parquet_config.yaml`);
  - columns in parquet files: `<output>/<dataset>/<category>/<variation>/*.parquet`
    (`parquet_subdir` adds a subdirectory), copies of the input files with the new columns. With
    `--merge`, the new files of each directory are merged into `merged.parquet` as soon as all of them
    are done (they are written in `<output or tmp_dir>/_unmerged_chunks` and removed after the merge).
    The merges and the copies to the output run in a background thread, so they never block the
    collection of the results of the workers.
    The files of the datasets/regions where no model is applied are written only with
    `copy_unprocessed: true` (merged directly from the input with `--merge`).
- **/pnfs**: the files cannot be written directly on `/pnfs`, so they are produced in `tmp_dir`
  (default `<output name>_tmp` in `/scratch/<user>`, the local disk of the node, or in `$TMPDIR` or
  `/tmp` if there is no `/scratch`) and each one is copied to the output in the background while the next one is
  produced (at most two files wait in `tmp_dir`). The copy is done with `rsync` if `/pnfs` is writable
  through the mount (T3 login nodes), otherwise (batch nodes) with `xrdcp` through the `redirector`,
  which needs a grid proxy: the script stops at the start if `$X509_USER_PROXY` is not valid for at
  least 2 h (`voms-proxy-init -voms cms -rfc --valid 168:0`). Each copy is checked (size, adler32
  checksum, number of events) and removed from `tmp_dir`; if a check fails the copy is removed, the
  local file is kept and the script stops.
- **Failures**: with models, a parquet file that fails is logged immediately and the others go on;
  at the end the failed files (and the directories not merged because of them) are listed and the
  script fails. If a worker process dies (e.g. out of memory) the script stops immediately.
- **Hanging /pnfs reads**: a read through the `/pnfs` mount can hang forever on the batch nodes. Each
  input file on `/pnfs` is read in a separate thread (the worker processes first copy it locally): if
  this takes more than `read_timeout` seconds, the thread is abandoned and the file is copied with
  `xrdcp` through the `redirector` (3 attempts with the same timeout, needs a grid proxy).
- **Empty files**: the input files without events are copied as they are (no scores to add); when
  merging, the empty files are skipped.
- **Stalls**: if nothing progresses (no file processed, merge step or copy done) for `stall_timeout`
  seconds, the script logs the last step done and exits with code 3, killing its workers and keeping
  the temporary files. This avoids jobs hanging until their time limit, e.g. when a read hangs
  forever on the `/pnfs` mount of a batch node.
- **Restart**: the output files that already exist are skipped (unless `--overwrite`) and the files
  left in `tmp_dir` by a failed run are reused, so a failed run can simply be restarted. With models
  and `--merge`, the unmerged chunks left in `<tmp_dir>/_unmerged_chunks` are reused too (unless
  `--overwrite`), so only the missing files are processed: do not change the models between a failed
  run and its restart, or remove that directory. On the T3, `/scratch` is removed at the end of each
  slurm job: to be able to restart a failed job, set `tmp_dir` to `/work/<user>/...`.
  In a slurm job (e.g. `sbatch -c 4 --mem-per-cpu 8000 --wrap "python scripts/postprocess_parquet.py -c <cfg>"`)
  the proxy must be on a shared file system (e.g. `/t3home`) and, since `/tmp` of the workers is only
  2 GB, `tmp_dir` should point to `/scratch/<user>` or `/work/<user>`.
- **Memory**: the parquet files where a model is applied are processed one row group at a time, so
  the memory does not scale with the file size; the files where no model is applied are copied
  without being read. While merging, `files_per_step` files are kept in memory at the same time.
- **Parallelism**: `workers: 1` runs everything in a single process. With more workers, the chunks of
  `chunk_size` events (coffea columns) or the parquet files are processed in parallel processes; each
  process creates its ONNX sessions once.
