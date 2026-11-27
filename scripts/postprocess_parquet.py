"""
Post-process the columns saved by a PocketCoffea run: merge the parquet chunks
and/or apply ONNX models (DNN or SPANet) to the columns.

Merge (--merge, default when no `models` are given in the config)
    With `save_chunk`, PocketCoffea saves the columns as thousands of small
    parquet chunks, whose reading from dCache (/pnfs) is dominated by the
    per-file latency. The chunks of each leaf directory
    (<dataset>/<category>/<variation>) are merged into a single
    `merged.parquet`, keeping the directory structure and the events order, so
    that the output can be used in place of the original parquet directory.

Apply ONNX models (`models` in the yaml config)
    The `events` record seen by the workflow is rebuilt from the saved columns
    (coffea or parquet) and the models are evaluated with
    `get_onnx_prediction` (utils_configs/dnn_evaluation_functions.py), i.e. with
    the same input definitions (dnn_input_variables.py) and logic as in the
    PocketCoffea run. The output has all the original columns plus the scores:
    - columns in the coffea files: a copy of the coffea files with the scores;
    - columns in parquet files: a copy of the parquet files with the scores, in
      the same <dataset>/<category>/<variation> structure, merged into one
      `merged.parquet` per directory with --merge. The files of the
      datasets/regions where no model is applied are copied (and merged) only
      with `copy_unprocessed: true` (forced by --update-config).

Output on /pnfs: the files are produced in a local temporary directory
(--tmp-dir) and each one is copied to the output while the next one is
produced, with rsync if /pnfs is writable through the mount (T3 login nodes),
otherwise (batch nodes) with xrdcp through the xrootd `redirector` (needs a
valid grid proxy). Each copy is checked (size, adler32, number of events) and
the temporary file is removed. The output files that already exist are
skipped (unless --overwrite) and the files left in the temporary directory by
a failed run (also the unmerged chunks with models and --merge) are reused, so
a failed run can simply be restarted.

With models, a parquet file that fails is logged immediately and the others
go on; the failures are reported (and the script fails) at the end. The
merges and copies run in a background thread, so they never block the
collection of the results of the workers. If nothing progresses for
--stall-timeout seconds (e.g. a read hanging forever on the /pnfs mount of a
batch node), the script exits with an error, keeping the temporary files.
The input files on /pnfs whose read through the mount does not finish within
--read-timeout seconds are read with xrdcp instead. The input files without
events are copied as they are (no scores to add).

With --update-config, the parquet path in the config.json of the PocketCoffea
output (`dump_columns_as_arrays_per_chunk` and `save_chunk`) is overwritten
with the output directory (the original values are kept as `<key>_unmerged`).

With --remove-original (merge only), the original parquet files are removed
once all the merges are done and checked: for each
<dataset>/<category>/<variation>, the number of events in the merged file
must be the same as in the original files (counted while merging, so that
the files are read only once; from the parquet metadata for the directories
merged by a previous run) and the original files must not have changed. If
any check fails, the errors are printed, nothing is removed and the script
fails. The directories left empty are removed too.

The default temporary directory (output on /pnfs) is /scratch/<user>
(/scratch is local on the T3 nodes), or $TMPDIR (or /tmp) if there is no
/scratch.

Usage:
    # merge into <...>/merged_parquet_files next to the parquet directory
    python scripts/postprocess_parquet.py -i <PocketCoffea output dir> [--update-config] \
        [--remove-original]
    # apply the ONNX models of the config (and merge)
    python scripts/postprocess_parquet.py -c scripts/postprocess_parquet_example.yaml \
        [-i <dir>] [-o <dir>] [-j 8] [--merge] [--update-config]

The command line arguments override the yaml config. All the options are
documented in scripts/postprocess_parquet_example.yaml.
"""

import argparse
import fnmatch
import getpass
import glob
import importlib
import importlib.util
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from urllib.parse import urlsplit

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# allow running the script from any directory
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("postprocess_parquet")

# Same defaults as configs/HH4b_common/config_files/default_config.py
DEFAULT_PAD_VALUE = -999.0
DEFAULT_PAD_VALUE_SPANET = 9999.0
# xrootd door used to write on /pnfs if the config.json has no root:// path
DEFAULT_REDIRECTOR = "root://t3dcachedb03.psi.ch:1094"

# SPANet sequential features computed on the fly from `btagB`, they are not
# columns of the output
SPANET_DERIVED_FEATURES = {"btag12_ratioSubLead", "btag_ratioAll"}

# keys of config["workflow"]["workflow_options"] with the parquet directory
CONFIG_KEYS = ("dump_columns_as_arrays_per_chunk", "save_chunk")
# suffix of the keys keeping the original parquet directory (--update-config)
UNMERGED_SUFFIX = "_unmerged"
MERGED_NAME = "merged.parquet"
# With models and --merge, the new parquet files are written in this
# subdirectory of the (temporary) output, merged into MERGED_NAME and removed
UNMERGED_SUBDIR = "_unmerged_chunks"

# All the options of the yaml config with their defaults
DEFAULTS = {
    # also from the command line
    "input": None,
    "output": None,
    "tmp_dir": None,
    "workers": 1,
    "merge": None,  # default: true without models, false with models
    "update_config": False,
    "overwrite": False,
    "stall_timeout": 3600,  # seconds without progress before exiting (0 = never)
    "read_timeout": 300,  # seconds before reading a /pnfs input with xrdcp (0 = never)
    "remove_original": False,  # merge only: remove the original files after the checks
    # yaml only
    "redirector": None,
    "merge_threads": 32,
    "files_per_step": 256,
    "variations": None,
    "parquet_input": None,
    "parquet_subdir": None,
    "max_parquet_files": None,
    "models": None,
    "chunk_size": 100000,
    "overwrite_columns": False,
    "copy_unprocessed": False,
    "process_output_all": False,
    "extra_columns": None,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def root_to_local(path_or_url):
    """Turn 'root://host:port//abs/path' into '/abs/path'. Leave local paths unchanged."""
    if isinstance(path_or_url, str) and path_or_url.startswith("root://"):
        p = urlsplit(path_or_url).path
        while p.startswith("//"):
            p = p[1:]
        if not p.startswith("/"):
            p = "/" + p
        return p
    return path_or_url


def redirector_of(url):
    """'root://host:port' of a root:// url, None for a local path."""
    if isinstance(url, str) and url.startswith("root://"):
        u = urlsplit(url)
        return f"{u.scheme}://{u.netloc}"
    return None


def local_like(original, local_path):
    """local_path written in the same form of original (with the same
    root://host:port/ redirector if original has one)."""
    redirector = redirector_of(original)
    if redirector:
        return f"{redirector}/{os.path.abspath(local_path)}"
    return os.path.abspath(local_path)


def load_object(spec, base_dir=None):
    """
    Load a python object from a string of the form `<module>:<attribute>` or
    `<file.py>:<attribute>`, e.g.
    `configs.HH4b_common.dnn_input_variables:sig_bkg_dnn_input_variables` or
    `configs/HH4b_common/dnn_input_variables.py:sig_bkg_dnn_input_variables`.
    A dotted `<module>.<attribute>` is also accepted.
    """
    if isinstance(spec, dict):
        # the dictionary is given directly in the yaml
        return spec
    if ":" in spec:
        module_spec, attr = spec.rsplit(":", 1)
    else:
        module_spec, attr = spec.rsplit(".", 1)

    if module_spec.endswith(".py"):
        path = module_spec
        if not os.path.isabs(path):
            candidates = [os.path.abspath(path), os.path.join(REPO_DIR, path)]
            if base_dir:
                candidates.insert(0, os.path.join(base_dir, path))
            path = next((p for p in candidates if os.path.exists(p)), path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find the python file {module_spec}")
        name = f"_onnx_inputs_{abs(hash(path))}"
        mod_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)

    try:
        return getattr(module, attr)
    except AttributeError:
        raise AttributeError(f"{attr} not found in {module_spec}")


def matches(name, patterns):
    """True if `patterns` is empty/None or `name` matches one of the (glob) patterns."""
    if not patterns:
        return True
    if isinstance(patterns, str):
        patterns = [patterns]
    return any(name == p or fnmatch.fnmatchcase(name, p) for p in patterns)


def parse_config(args):
    cfg, base_dir = {}, os.getcwd()
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        base_dir = os.path.dirname(os.path.abspath(args.config))
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown options in {args.config}: {sorted(unknown)}")
    for key, value in DEFAULTS.items():
        cfg.setdefault(key, value)

    # command line arguments override the yaml
    for key in [
        "input",
        "output",
        "tmp_dir",
        "workers",
        "merge",
        "update_config",
        "overwrite",
        "stall_timeout",
        "read_timeout",
        "remove_original",
    ]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if not cfg["input"]:
        raise ValueError("'input' must be given (-i or in the config)")
    if not cfg["models"] and args.workers is not None:
        # merge only: -j is the number of reading threads
        cfg["merge_threads"] = args.workers
    if cfg["merge"] is None:
        cfg["merge"] = not cfg["models"]
    if not cfg["models"] and not cfg["merge"]:
        raise ValueError("Nothing to do: no models in the config and merging disabled")
    if cfg["remove_original"]:
        if cfg["models"]:
            raise ValueError("--remove-original is only possible when merging only (no models)")
        if cfg["max_parquet_files"]:
            raise ValueError(
                "--remove-original is not possible with max_parquet_files (not all the "
                "files are merged)"
            )
    if cfg["extra_columns"] is not None and not all(
        isinstance(v, dict) for v in cfg["extra_columns"].values()
    ):
        raise ValueError(
            "'extra_columns' must be a dictionary {column: {dataset pattern: value}}"
        )
    cfg["models"] = parse_models(cfg["models"] or [], base_dir)
    return cfg


def parse_models(models_cfg, base_dir):
    models = []
    column_names = set()
    for i, m in enumerate(models_cfg):
        m = dict(m)
        if "name" not in m:
            m["name"] = f"model_{i}"
        if "model" not in m:
            raise ValueError(f"Model {m['name']}: the path of the onnx file ('model') is missing")
        if "input_variables" not in m:
            raise ValueError(f"Model {m['name']}: 'input_variables' is missing")
        model_path = m["model"]
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            if os.path.exists(os.path.join(base_dir, model_path)):
                model_path = os.path.join(base_dir, model_path)
        m["model"] = os.path.abspath(model_path)
        if not os.path.exists(m["model"]):
            raise FileNotFoundError(f"Model {m['name']}: {m['model']} not found")

        m["input_variables_spec"] = m["input_variables"]
        m["input_variables"] = load_object(m["input_variables"], base_dir)

        m.setdefault("column_name", f"events_{m['name']}")
        m.setdefault("onnx_output", None)
        m.setdefault("spanet_output", "class_prob")
        m.setdefault("spanet_output_index", 0)
        m.setdefault("score_index", "auto")
        m.setdefault("pad_value", DEFAULT_PAD_VALUE)
        m.setdefault("pad_value_spanet", DEFAULT_PAD_VALUE_SPANET)
        m.setdefault("max_num_jets", None)
        m.setdefault("spanet_global_mask", "ones")
        m.setdefault("regions", None)
        m.setdefault("datasets", None)
        m.setdefault("exclude_datasets", None)
        m.setdefault("exclude_regions", None)
        if m["column_name"] in column_names:
            raise ValueError(f"The column name {m['column_name']} is used by more than one model")
        column_names.add(m["column_name"])
        models.append(m)
    return models


def models_for(models, dataset, category):
    """Models to be applied to the given dataset and region."""
    return [
        m
        for m in models
        if matches(dataset, m["datasets"])
        and matches(category, m["regions"])
        and not (m["exclude_datasets"] and matches(dataset, m["exclude_datasets"]))
        and not (m["exclude_regions"] and matches(category, m["exclude_regions"]))
    ]


# ---------------------------------------------------------------------------
# config.json of the PocketCoffea output
# ---------------------------------------------------------------------------


def read_config(config_file):
    with open(config_file) as f:
        return json.load(f)


def parquet_dir_from_config(config_file, original):
    """
    Parquet directory saved in the config.json and its xrootd redirector (None
    if not a root:// url). With `original`, the directory written by
    PocketCoffea (kept as <key>_unmerged by --update-config) is used instead of
    the current one.
    """
    options = read_config(config_file)["workflow"]["workflow_options"]
    for key in CONFIG_KEYS:
        value = (original and options.get(key + UNMERGED_SUFFIX)) or options.get(key)
        if value:
            return root_to_local(value), redirector_of(value)
    return None, None


def update_config(config_file, new_dir):
    """Overwrite the parquet directory in config_file with new_dir, keeping
    the original value in <key>_unmerged."""
    config = read_config(config_file)
    options = config["workflow"]["workflow_options"]
    for key in CONFIG_KEYS:
        original = options.get(key + UNMERGED_SUFFIX) or options.get(key)
        if not original:
            continue
        options[key + UNMERGED_SUFFIX] = original
        options[key] = local_like(original, new_dir)
        logger.info(f"config.json: {key} = {options[key]} (was {original})")
    tmp_file = config_file + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(config, f, indent=2)
    if os.path.exists(config_file):
        # files on dCache cannot be overwritten, they have to be removed first
        os.remove(config_file)
    os.replace(tmp_file, config_file)
    logger.info(f"Updated {config_file}")


# ---------------------------------------------------------------------------
# Columns needed by the models and reconstruction of the events record
# ---------------------------------------------------------------------------


def required_fields(models):
    """
    Return, for each collection, the needed features and how the collection is
    accessed: `flat` (one object per event, e.g. `HiggsLeading`) or `jagged`
    (`Collection:N` and the SPANet sequential inputs).
    """
    from utils_configs.dnn_evaluation_functions import get_collections

    fields = defaultdict(set)
    flat, jagged = set(), set()

    for m in models:
        variables = m["input_variables"]
        if "sequential" in variables:
            for coll, feats in get_collections(variables["sequential"]).items():
                jagged.add(coll)
                for feat in feats:
                    feat = feat.split(":")[0]
                    if feat in SPANET_DERIVED_FEATURES:
                        # the b-tag ratios are computed from btagB
                        fields[coll].add("btagB")
                    else:
                        fields[coll].add(feat)
                # pt is used to build the mask; btagB is always accessed by
                # define_spanet_sequential_inputs, but only really used for
                # the b-tag ratios, so it is optional otherwise
                fields[coll].add("pt")
                fields[coll].add("btagB?")
            global_vars = variables["global"]
        else:
            global_vars = variables
        for attributes in global_vars.values():
            collection, feature = attributes[0], attributes[1]
            if ":" in collection:
                coll = collection.split(":")[0]
                jagged.add(coll)
            else:
                coll = collection
                if coll != "events":
                    flat.add(coll)
            fields[coll].add(feature)
    return fields, flat, jagged


def column_name(collection, feature):
    return f"{collection}_{feature}"


def to_awkward(array, counts=None, n_events=None):
    """
    Convert a saved column to awkward. Columns saved per object are
    returned jagged:
    - parquet columns are already jagged (or regular 2D arrays);
    - coffea columns of collections are flattened and are unflattened with
      the number of objects per event saved in `<Collection>_N`.
    """
    arr = array if isinstance(array, ak.Array) else ak.Array(np.asarray(array))
    if arr.ndim > 1:
        # no-op if already jagged (also works with awkward 1, where the
        # ArrayType has no `content`)
        return ak.from_regular(arr, axis=1)
    if counts is not None:
        counts = np.asarray(counts, dtype=np.int64)
        if len(arr) == counts.sum():
            return ak.unflatten(arr, counts)
        if len(arr) != n_events:
            raise ValueError(
                f"{len(arr)} entries, neither the number of events ({n_events}) "
                f"nor the number of objects ({counts.sum()})"
            )
    return arr


def build_events(columns, models, n_events, unit_name=""):
    """
    Build the `events` record expected by `get_onnx_prediction` from the
    saved columns.

    `columns` is a mapping {column_name: array}. Jagged collections are
    stored either already jagged (parquet) or flattened with the number of
    objects in `<Collection>_N` (coffea).
    """
    fields, flat, jagged = required_fields(models)
    both = flat & jagged
    if both:
        raise ValueError(
            f"The collections {both} are used both as single objects and as "
            "lists of objects by the models applied together"
        )

    event_fields = {}
    for coll, feats in fields.items():
        coll_fields = {}
        counts = columns.get(f"{coll}_N", None) if coll != "events" else None
        for feat in sorted(feats):
            optional = feat.endswith("?")
            feat = feat.rstrip("?")
            name = column_name(coll, feat)
            if name not in columns:
                if optional:
                    continue
                raise KeyError(
                    f"{unit_name}: column {name} needed by the ONNX models "
                    "is not saved in the input files"
                )
            arr = to_awkward(columns[name], counts, n_events)
            if len(arr) != n_events:
                raise ValueError(
                    f"{unit_name}: column {name} has {len(arr)} entries, "
                    f"expected {n_events} events"
                )
            coll_fields[feat] = arr

        if coll == "events":
            event_fields.update(coll_fields)
            continue

        is_jagged = {f: arr.ndim > 1 for f, arr in coll_fields.items()}
        if coll in jagged:
            if not all(is_jagged.values()):
                raise ValueError(
                    f"{unit_name}: collection {coll} is used as a list of objects "
                    "but it is saved as a single object per event"
                )
            if "btagB" not in coll_fields:
                # only needed for the b-tag ratios of SPANet
                coll_fields["btagB"] = ak.ones_like(next(iter(coll_fields.values())))
        else:
            for f, arr in coll_fields.items():
                if is_jagged[f]:
                    if ak.max(ak.num(arr, axis=1), axis=None) > 1:
                        raise ValueError(
                            f"{unit_name}: collection {coll} is used as a single "
                            "object, but it has more than one object per event"
                        )
                    coll_fields[f] = ak.firsts(arr, axis=1)
        event_fields[coll] = ak.zip(coll_fields)

    return ak.zip(event_fields, depth_limit=1)


def extra_columns_for(cfg, dataset):
    """
    Constant input columns of the given dataset from `extra_columns`
    ({column: {dataset pattern: value}}): the value of the first pattern
    matching the dataset, if any.
    """
    extra = {}
    for col, values in (cfg.get("extra_columns") or {}).items():
        for pattern, value in values.items():
            if matches(dataset, pattern):
                extra[col] = value
                break
    return extra


def add_extra_columns(columns, extra, needed, n_events):
    """Add the constant columns `extra` that are needed and not already saved."""
    for col, value in extra.items():
        if col in needed and col not in columns:
            columns[col] = np.full(n_events, value, dtype=np.float64)
    return columns


def columns_needed(models):
    """Names of the columns (and counts) to read from the input."""
    fields, _, _ = required_fields(models)
    names = set()
    for coll, feats in fields.items():
        for feat in feats:
            names.add(column_name(coll, feat.rstrip("?")))
        if coll != "events":
            names.add(f"{coll}_N")
    return names


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_SESSIONS = {}


def get_session(model):
    """ONNX session cached per process (the same options as in the workflow)."""
    key = model["model"]
    if key not in _SESSIONS:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            model["model"], sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        input_name = [i.name for i in session.get_inputs()]
        output_name = [o.name for o in session.get_outputs()]
        _SESSIONS[key] = (session, input_name, output_name)
    return _SESSIONS[key]


def select_output(model, onnx_output, out_type, output_name):
    """Pick the array to be saved from the output of `get_onnx_prediction`."""
    if out_type == "spanet":
        group = model["spanet_output"]
        if group not in onnx_output:
            raise ValueError(
                f"Model {model['name']}: spanet_output must be one of {list(onnx_output.keys())}"
            )
        arrays = onnx_output[group]
        if not arrays:
            raise ValueError(f"Model {model['name']}: no SPANet output of type {group}")
        return np.asarray(arrays[model["spanet_output_index"]])
    if isinstance(onnx_output, list):
        return np.asarray(onnx_output[0])
    return np.asarray(onnx_output)


def score_columns(model, array):
    """
    Convert the model output to the columns to be saved.

    `score_index`:
    - "auto" (default, same as `sig_bkg_dnn_score` in the workflow): 1D outputs
      are saved as they are, outputs with <= 2 columns save the last one, and
      outputs with more columns save one column per class
      (`<column_name>_<i>`);
    - an integer: save only that column;
    - a list of integers: save those columns as `<column_name>_<i>`;
    - "all": save every column as `<column_name>_<i>`.
    """
    name = model["column_name"]
    idx = model["score_index"]
    if array.ndim == 1:
        return {name: array}
    array = array.reshape(array.shape[0], -1)
    n = array.shape[1]
    if idx == "auto":
        if n <= 2:
            return {name: array[:, -1]}
        return {f"{name}_{i}": array[:, i] for i in range(n)}
    if idx == "all":
        return {f"{name}_{i}": array[:, i] for i in range(n)}
    if isinstance(idx, int):
        return {name: array[:, idx]}
    if isinstance(idx, (list, tuple)):
        return {f"{name}_{i}": array[:, i] for i in idx}
    raise ValueError(f"Model {model['name']}: invalid score_index {idx}")


def evaluate_model(model, events):
    from utils_configs.dnn_evaluation_functions import get_onnx_prediction

    session, input_name, output_name = get_session(model)
    if model["onnx_output"] is not None:
        requested = model["onnx_output"]
        requested = [requested] if isinstance(requested, str) else list(requested)
        missing = [o for o in requested if o not in output_name]
        if missing:
            raise ValueError(
                f"Model {model['name']}: outputs {missing} not in the ONNX outputs {output_name}"
            )
        if "sequential" not in model["input_variables"]:
            output_name = requested
    onnx_output, out_type = get_onnx_prediction(
        session,
        list(input_name),
        list(output_name),
        events,
        model["input_variables"],
        model["pad_value"],
        model["pad_value_spanet"],
        model["max_num_jets"],
        spanet_global_mask=model["spanet_global_mask"],
    )
    array = select_output(model, onnx_output, out_type, output_name)
    if len(array) != len(events):
        raise ValueError(
            f"Model {model['name']}: {len(array)} outputs for {len(events)} events"
        )
    return {k: np.asarray(v, dtype=np.float32) for k, v in score_columns(model, array).items()}


def evaluate_in_chunks(models, events, chunk_size):
    """Evaluate sequentially in chunks of `chunk_size` events and concatenate."""
    n = len(events)
    if n == 0:
        # Without events the number of output columns of multi-class models
        # cannot be known: only the base column name is created
        return {m["column_name"]: np.zeros(0, dtype=np.float32) for m in models}
    pieces = defaultdict(list)
    for start in range(0, n, chunk_size):
        chunk = events[start : start + chunk_size]
        for model in models:
            for k, v in evaluate_model(model, chunk).items():
                pieces[k].append(v)
    return {k: np.concatenate(v) for k, v in pieces.items()}


# ---------------------------------------------------------------------------
# Output (local, or /pnfs through a temporary directory)
# ---------------------------------------------------------------------------


# How the input files on /pnfs are read (see read_with_timeout): set by
# configure_reads in the main process and in the worker processes
READS = {"timeout": 0, "redirector": DEFAULT_REDIRECTOR, "tmp_dir": None}


def configure_reads(timeout, redirector, tmp_dir):
    if tmp_dir is not None:
        os.makedirs(tmp_dir, exist_ok=True)
    READS.update(timeout=timeout, redirector=redirector.rstrip("/"), tmp_dir=tmp_dir)


def _call_with_timeout(fn, timeout):
    """
    Run fn() in a daemon thread and return (True, result), or (False, None) if
    it does not finish within timeout seconds. A hanging thread is abandoned:
    being a daemon, it does not block the exit of the process.
    """
    box = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as err:
            box["error"] = err

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return False, None
    if "error" in box:
        raise box["error"]
    return True, box["result"]


def read_with_timeout(fn, path):
    """
    Return fn(path) for an input file. A read through the /pnfs mount can hang
    forever (seen on the batch nodes): if fn does not finish within
    `read_timeout` seconds, the file is copied to a local temporary file with
    xrdcp through the xrootd redirector (same timeout, 3 attempts) and fn is
    called on the copy.
    """
    timeout = READS["timeout"]
    if not timeout or not path.startswith("/pnfs"):
        return fn(path)
    done, result = _call_with_timeout(lambda: fn(path), timeout)
    if done:
        return result
    logger.warning(f"Read of {path} through the mount hanging for {timeout} s: using xrdcp")
    fd, local = tempfile.mkstemp(suffix=".parquet", dir=READS["tmp_dir"])
    os.close(fd)
    try:
        for attempt in range(1, 4):
            try:
                subprocess.run(
                    ["xrdcp", "--nopbar", "--force", f"{READS['redirector']}/{path}", local],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as err:
                msg = err.stderr.strip() if getattr(err, "stderr", None) else str(err)
                if attempt == 3:
                    raise RuntimeError(f"xrdcp of {path} failed 3 times: {msg}") from err
                logger.warning(f"xrdcp of {path} failed (attempt {attempt}): {msg}")
        return fn(local)
    finally:
        os.remove(local)


def fetch_input(src, dst):
    """
    Copy the input file src to dst (see read_with_timeout). Each attempt
    writes its own temporary file, so that a hanging copy abandoned by the
    timeout cannot overwrite dst when it finishes later.
    """

    def copy(path):
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst), suffix=".part")
        os.close(fd)
        shutil.copyfile(path, tmp)
        return tmp

    os.replace(read_with_timeout(copy, src), dst)


class Watchdog:
    """
    Exit the script if nothing progresses for `timeout` seconds (0 = never),
    e.g. a read hanging forever on the /pnfs mount of a batch node, which
    would otherwise keep the job running until its time limit. The exit is
    done with os._exit, without waiting for the hung threads and without
    cleaning the temporary directory, so that a restart reuses the files
    already produced.
    """

    def __init__(self, timeout=0):
        self.timeout = timeout
        self.last_time = time.monotonic()
        self.last_step = "start"

    def start(self):
        if self.timeout:
            threading.Thread(target=self._run, daemon=True, name="watchdog").start()

    def progress(self, step):
        self.last_time = time.monotonic()
        self.last_step = step

    def _run(self):
        while True:
            time.sleep(min(60, self.timeout))
            idle = time.monotonic() - self.last_time
            if idle > self.timeout:
                logger.error(
                    f"No progress for {idle:.0f} s (stall_timeout {self.timeout} s), "
                    f"last step: {self.last_step}. Probably a file access is hanging "
                    "(e.g. /pnfs mount of this node): exiting. The files in the "
                    "temporary directory are kept and reused by the next run (not on "
                    "/scratch in a slurm job, which is removed at the end of the job)."
                )
                # the worker processes would be left running (e.g. hanging)
                for child in multiprocessing.active_children():
                    child.kill()
                logging.shutdown()
                os._exit(3)


# replaced in main(), progress() is a no-op in the worker processes
WATCHDOG = Watchdog()


def progress(step):
    WATCHDOG.progress(step)


def adler32(path, block_size=16 * 1024 * 1024):
    value = 1
    with open(path, "rb") as f:
        while block := f.read(block_size):
            value = zlib.adler32(block, value)
    return value


def remove_file(path, redirector=None):
    """Remove path, through xrootd (xrdfs rm) if redirector is given."""
    if redirector is None:
        os.remove(path)
    else:
        subprocess.run(["xrdfs", redirector, "rm", path], check=True)


def is_writable(path):
    """True if path (or its closest existing parent) can be written through the
    mount: /pnfs is writable on the T3 login nodes, read-only on the batch nodes."""
    path = os.path.abspath(path)
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return os.access(path, os.W_OK)


def remove_empty_dirs(path):
    """Remove the empty directories inside path (path itself too, if left empty)."""
    if not os.path.isdir(path):
        return
    for root, _, _ in os.walk(path, topdown=False):
        if not os.listdir(root):
            os.rmdir(root)


def check_and_remove(src_file, dst_file, redirector=None, cksum_done=False):
    """Check the copy dst_file of src_file (size, adler32 unless already checked
    by xrdcp, number of events for parquet files) and remove src_file.

    If the copy is not correct, dst_file is removed (through xrootd if
    redirector is given), src_file is kept and an error is raised."""
    errors = []
    src_size, dst_size = os.path.getsize(src_file), os.path.getsize(dst_file)
    if src_size != dst_size:
        errors.append(f"size {dst_size} != {src_size}")
    else:
        if not cksum_done:
            src_sum, dst_sum = adler32(src_file), adler32(dst_file)
            if src_sum != dst_sum:
                errors.append(f"adler32 {dst_sum:08x} != {src_sum:08x}")
        if src_file.endswith(".parquet"):
            src_rows = pq.ParquetFile(src_file).metadata.num_rows
            dst_rows = pq.ParquetFile(dst_file).metadata.num_rows
            if src_rows != dst_rows:
                errors.append(f"events {dst_rows} != {src_rows}")
    if errors:
        remove_file(dst_file, redirector)
        raise RuntimeError(
            f"Copy of {src_file} to {dst_file} failed ({', '.join(errors)}): "
            f"the copy was removed and the original file kept"
        )
    logger.info(f"Copy of {dst_file} checked, removing {src_file}")
    os.remove(src_file)
    progress(f"copied {dst_file}")


def copy_and_remove(src_file, dst_file, redirector=None):
    """Copy src_file to dst_file, check the copy and remove src_file.

    If redirector is None (output writable through the mount, e.g. /pnfs on
    the login nodes) the copy is done with rsync. Otherwise (output on /pnfs,
    read-only through the mount, e.g. on the batch nodes), the copy is done
    with xrdcp through it, creating the missing directories and checking the
    adler32 checksum. The copy is then read back through the mount to check
    it (see check_and_remove)."""
    # files on dCache cannot be overwritten, they have to be removed first
    if os.path.exists(dst_file):
        remove_file(dst_file, redirector)
    logger.info(f"Copying {src_file} to {dst_file}")
    if redirector is None:
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        subprocess.run(["rsync", "-a", src_file, dst_file], check=True)
    else:
        res = subprocess.run(
            [
                "xrdcp",
                "--nopbar",
                "--path",
                "--cksum",
                "adler32:source",
                src_file,
                f"{redirector}/{dst_file}",
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            if os.path.exists(dst_file):
                remove_file(dst_file, redirector)
            raise RuntimeError(
                f"xrdcp of {src_file} to {dst_file} failed ({res.stderr.strip()}): "
                "the original file was kept"
            )
    # with xrdcp the adler32 checksum is already checked by --cksum
    check_and_remove(src_file, dst_file, redirector, cksum_done=redirector is not None)


class Output:
    """
    Output directory. If `stage` is different from it (output on /pnfs), the
    files are produced in the local `stage` directory and each one is copied
    to the output in a background thread while the next one is produced (so
    at most two files wait in `stage`), checked and removed from `stage`.
    """

    def __init__(self, output, stage, redirector, overwrite):
        self.output = output
        self.stage = stage
        self.redirector = redirector
        self.overwrite = overwrite
        self.staged = stage != output
        self._copy_ex = ThreadPoolExecutor(1) if self.staged else None
        self._copy_future = None

    def final_path(self, rel):
        return os.path.join(self.output, rel)

    def stage_path(self, rel):
        """Path where the file `rel` is produced (its directory is created)."""
        path = os.path.join(self.stage, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def todo(self, rel):
        """
        What to do for the output file `rel`: "skip" if it already exists
        (unless overwrite), "copy" if it was left in `stage` by a failed run
        (only the copy is missing), "produce" otherwise.
        """
        final, staged = self.final_path(rel), os.path.join(self.stage, rel)
        if self.overwrite:
            return "produce"
        if os.path.exists(final):
            logger.info(f"Skipping existing {final}")
            if self.staged and os.path.exists(staged):
                # copied by a previous run stopped before removing it from stage
                check_and_remove(staged, final, self.redirector)
            return "skip"
        if self.staged and os.path.exists(staged):
            logger.info(f"Reusing existing {staged}")
            return "copy"
        return "produce"

    def publish(self, rel):
        """The file `rel` is ready in `stage`: copy it to the output."""
        if not self.staged:
            return
        # raises the error of the previous copy, if any
        self.wait()
        self._copy_future = self._copy_ex.submit(
            copy_and_remove, os.path.join(self.stage, rel), self.final_path(rel), self.redirector
        )

    def wait(self):
        if self._copy_future is not None:
            future, self._copy_future = self._copy_future, None
            future.result()

    def close(self):
        if not self.staged:
            return
        try:
            self.wait()
        finally:
            self._copy_ex.shutdown()
            remove_empty_dirs(self.stage)


def check_proxy(min_hours=2):
    """
    xrdcp to /pnfs needs a valid grid proxy: fail at the start, not after
    processing. The proxy (X509_USER_PROXY) must be on a shared file system
    (e.g. /t3home) to be seen by the slurm jobs.
    """
    proxy = os.environ.get("X509_USER_PROXY") or f"/tmp/x509up_u{os.getuid()}"
    res = subprocess.run(
        ["voms-proxy-info", "-exists", "-valid", f"{min_hours}:00", "-file", proxy],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"No grid proxy valid for at least {min_hours} h in {proxy}, needed to copy "
            "the output to /pnfs with xrdcp: run `voms-proxy-init -voms cms -rfc --valid 168:0`"
        )
    if proxy.startswith("/tmp"):
        logger.warning(
            f"The proxy {proxy} is in /tmp: set X509_USER_PROXY to a file on a shared "
            "file system (e.g. /t3home) to use it in the slurm jobs"
        )
    logger.info(f"Grid proxy: {proxy}")


# ---------------------------------------------------------------------------
# Parquet files
# ---------------------------------------------------------------------------


def parquet_files(directory, max_files=None):
    """Sorted parquet files directly inside directory (the order used by
    ML_pytorch to load them). As in pyarrow.dataset, the files starting with
    "." or "_" are ignored."""
    files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".parquet") and not f.startswith((".", "_"))
    )
    return files[:max_files] if max_files else files


def leaf_dirs(path, variations=None):
    """
    Directories directly containing parquet files (sorted), relative to path,
    e.g. <dataset>/<category>/<variation>. With `variations`, only the
    <variation> directories matching them are kept.
    """
    leaves = []
    for root, dirs, filenames in os.walk(path, followlinks=True):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_")))
        if not any(f.endswith(".parquet") and not f.startswith((".", "_")) for f in filenames):
            continue
        rel = os.path.relpath(root, path)
        parts = rel.split(os.sep)
        if variations and len(parts) > 2 and not matches(parts[2], variations):
            continue
        leaves.append(rel)
    return leaves


def merge_files(files, out_file, n_threads, files_per_step):
    """Merge the parquet files into out_file, keeping the events order.
    Return the number of events read from the files (= written)."""
    if not files:
        raise ValueError(f"No parquet files to merge into {out_file}")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    tmp_file = out_file + ".tmp"
    writer, first_schema, n_rows = None, None, 0
    with ThreadPoolExecutor(n_threads) as ex:
        # read the files in parallel in steps to limit the memory usage,
        # map keeps the order of the files
        for i in range(0, len(files), files_per_step):
            tables = list(
                ex.map(lambda f: read_with_timeout(pq.read_table, f), files[i : i + files_per_step])
            )
            n_read = min(i + files_per_step, len(files))
            progress(f"merging {out_file}: {n_read}/{len(files)} files read")
            if first_schema is None:
                first_schema = tables[0].schema
            # the empty files may lack columns (e.g. the ONNX scores)
            tables = [t for t in tables if t.num_rows > 0]
            if not tables:
                continue
            if writer is None:
                writer = pq.ParquetWriter(tmp_file, tables[0].schema)
            # one large row group per step instead of one per small chunk
            table = pa.concat_tables([t.cast(writer.schema) for t in tables])
            writer.write_table(table, row_group_size=table.num_rows)
            n_rows += table.num_rows
    if writer is None:
        # only empty files: empty file with the schema of the first one
        writer = pq.ParquetWriter(tmp_file, first_schema)
    writer.close()
    os.replace(tmp_file, out_file)
    logger.info(f"Merged {len(files)} files ({n_rows} events) into {out_file}")
    return n_rows


def num_rows(path):
    """Number of events of a parquet file, from its metadata (footer only)."""
    return read_with_timeout(lambda p: pq.ParquetFile(p).metadata.num_rows, path)


def count_rows(files, n_threads):
    """Total number of events of the parquet files, from their metadata."""
    with ThreadPoolExecutor(n_threads) as ex:
        return sum(ex.map(num_rows, files))


def run_merge(cfg, parquet_dir, out):
    """
    Merge only: one merged.parquet per leaf directory. With remove_original,
    return {leaf: {"files", "n_events", "merged"}}: the original files, their
    number of events (counted while merging, or from their metadata if the
    merged file was produced by a previous run) and the merged file, checked
    by check_merged before removing the original files.
    """
    merged = {}
    for leaf in leaf_dirs(parquet_dir, cfg["variations"]):
        rel = os.path.normpath(os.path.join(leaf, MERGED_NAME))
        todo = out.todo(rel)
        if todo == "skip" and not cfg["remove_original"]:
            continue
        files = parquet_files(os.path.join(parquet_dir, leaf), cfg["max_parquet_files"])
        if todo == "produce":
            n_events = merge_files(
                files, out.stage_path(rel), cfg["merge_threads"], cfg["files_per_step"]
            )
        else:
            # merged by a previous run: count the events of the original files
            n_events = count_rows(files, cfg["merge_threads"]) if cfg["remove_original"] else None
            progress(f"{leaf}: events counted")
        if cfg["remove_original"]:
            merged[leaf] = {"files": files, "n_events": n_events, "merged": out.final_path(rel)}
        if todo != "skip":
            out.publish(rel)
    return merged


def check_merged(parquet_dir, merged):
    """
    --remove-original: check, for each leaf directory, that the merged file
    exists and has the same number of events as the original files, and that
    the original files did not change in the meantime. All the failures are
    logged and an error is raised if any check fails.
    """
    errors = []
    for leaf, info in merged.items():
        try:
            current = parquet_files(os.path.join(parquet_dir, leaf))
            if current != info["files"]:
                errors.append(
                    f"{leaf}: the original files changed during the merge "
                    f"({len(info['files'])} files merged, {len(current)} files now)"
                )
                continue
            if not os.path.isfile(info["merged"]):
                errors.append(f"{leaf}: merged file {info['merged']} not found")
                continue
            n_merged = num_rows(info["merged"])
        except Exception as err:
            errors.append(f"{leaf}: check failed with {err!r}")
            continue
        progress(f"{leaf}: checked")
        if n_merged != info["n_events"]:
            errors.append(
                f"{leaf}: {n_merged} events in {info['merged']}, {info['n_events']} "
                f"in the {len(info['files'])} original files"
            )
        else:
            logger.info(
                f"Check OK {leaf}: {n_merged} events in the {len(info['files'])} original "
                "files and in the merged file"
            )
    if errors:
        for e in errors:
            logger.error(e)
        raise RuntimeError(
            f"--remove-original: {len(errors)}/{len(merged)} directories failed the check, "
            "NO original file was removed:\n" + "\n".join(f"  {e}" for e in errors)
        )
    logger.info(f"All the {len(merged)} merged directories passed the check")


def remove_original_files(parquet_dir, merged, redirector, n_threads):
    """
    --remove-original: remove the original parquet files of the merged leaf
    directories (through xrootd if redirector is given, i.e. /pnfs read-only
    through the mount) and then the directories of parquet_dir left empty.
    """
    files = [os.path.normpath(f) for info in merged.values() for f in info["files"]]
    # listed before the removal: through the mount, the removals done with
    # xrootd may not be seen immediately
    tree = list(os.walk(parquet_dir, topdown=False))
    logger.info(
        f"Removing the {len(files)} original parquet files of {len(merged)} directories "
        f"in {parquet_dir}" + (f" through {redirector}" if redirector else "")
    )

    def remove(path):
        remove_file(path, redirector)
        progress(f"removed {path}")

    with ThreadPoolExecutor(n_threads) as ex:
        for i, _ in enumerate(ex.map(remove, files), 1):
            if i % 1000 == 0:
                logger.info(f"{i}/{len(files)} original files removed")

    removed = set(files)
    n_dirs = 0
    for root, dirs, filenames in tree:
        root = os.path.normpath(root)
        if not all(os.path.join(root, n) in removed for n in dirs + filenames):
            continue
        try:
            if redirector is None:
                os.rmdir(root)
            else:
                subprocess.run(
                    ["xrdfs", redirector, "rmdir", root], check=True, capture_output=True
                )
        except (OSError, subprocess.CalledProcessError) as err:
            logger.warning(f"Could not remove the empty directory {root}: {err}")
            continue
        removed.add(root)
        n_dirs += 1
    logger.info(f"Removed {len(files)} original files and {n_dirs} empty directories")


def _process_parquet_file(task):
    """
    Read one parquet file, evaluate the models and write a copy with the new
    columns. Run in a separate process when workers > 1.
    """
    src, dst, models, chunk_size, unit_name, overwrite_columns, extra = task
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp_dst = dst + ".tmp"
    if not models:
        # nothing to add: plain copy, without loading the file in memory
        fetch_input(src, dst)
        return
    if READS["timeout"] and src.startswith("/pnfs"):
        # local copy first, so that a read hanging on the /pnfs mount is
        # caught by the timeout of read_with_timeout
        local_src = dst + ".input"
        fetch_input(src, local_src)
        try:
            _process_parquet_file((local_src, *task[1:]))
        finally:
            os.remove(local_src)
        return

    metadata = pq.ParquetFile(src).metadata
    n_row_groups = metadata.num_row_groups
    if metadata.num_rows == 0:
        # no events (awkward fails on the empty jagged columns): plain copy,
        # the missing scores are not a problem when merging
        shutil.copyfile(src, tmp_dst)
        os.replace(tmp_dst, dst)
        return

    needed = columns_needed(models)

    def row_groups_with_scores():
        # one row group at a time, so that the memory does not scale with the
        # size of the file (the merged files can be several GB)
        for i in range(n_row_groups):
            if metadata.row_group(i).num_rows == 0:
                continue
            # converted with awkward (same arrow schema as ak.to_parquet, used
            # by PocketCoffea), so that the files read back the same way
            array = ak.from_parquet(src, row_groups=[i])
            columns = {c: array[c] for c in array.fields if c in needed}
            add_extra_columns(columns, extra, needed, len(array))
            events = build_events(columns, models, len(array), unit_name)
            scores = evaluate_in_chunks(models, events, chunk_size)
            for name, arr in scores.items():
                if name in array.fields:
                    if not overwrite_columns:
                        raise ValueError(
                            f"{unit_name}: column {name} already exists in {src}, "
                            "set overwrite_columns: true to replace it"
                        )
                    logger.warning(f"{unit_name}: overwriting column {name} in {src}")
                array = ak.with_field(array, arr, name)
            yield ak.to_arrow_table(array)

    # one output row group per input row group
    writer = None
    try:
        for table in row_groups_with_scores():
            if writer is None:
                writer = pq.ParquetWriter(tmp_dst, table.schema)
            elif not table.schema.equals(writer.schema):
                table = table.cast(writer.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp_dst, dst)


def run_parquet_models(cfg, parquet_dir, out, pool):
    """Apply the models to the parquet columns (and merge them with --merge)."""
    merge = cfg["merge"]
    subdir = cfg["parquet_subdir"] or ""
    tasks = []  # (unit dir, output file relative path, task)
    files_left = {}  # merge: unit dir -> number of files not processed yet
    merge_only_units = []  # merge: (input files, merged file) of the units without models
    for leaf in leaf_dirs(parquet_dir, cfg["variations"]):
        parts = leaf.split(os.sep)
        dataset = parts[0]
        category = parts[1] if len(parts) > 1 else ""
        unit_models = models_for(cfg["models"], dataset, category)
        if not unit_models and not cfg["copy_unprocessed"]:
            logger.debug(f"{leaf}: no model to apply, skipped")
            continue
        files = parquet_files(os.path.join(parquet_dir, leaf), cfg["max_parquet_files"])
        unit_dir = os.path.normpath(os.path.join(subdir, leaf))
        if unit_models:
            logger.info(f"{leaf}: {len(files)} files, models {[m['name'] for m in unit_models]}")
        else:
            logger.info(f"{leaf}: {len(files)} files, no model to apply, copied")

        if merge:
            rel = os.path.join(unit_dir, MERGED_NAME)
            todo = out.todo(rel)
            if todo == "copy":
                out.publish(rel)
            if todo != "produce":
                continue
            if not unit_models:
                # merged directly from the input files
                merge_only_units.append((files, rel))
                continue
            out_rels = [os.path.join(UNMERGED_SUBDIR, unit_dir, os.path.basename(f)) for f in files]
            if not cfg["overwrite"]:
                # chunks left in the temporary directory by a failed run
                # (written with os.replace, so they are complete)
                todo = [
                    (f, r) for f, r in zip(files, out_rels)
                    if not os.path.exists(os.path.join(out.stage, r))
                ]
                if len(todo) < len(files):
                    logger.info(f"{leaf}: reusing {len(files) - len(todo)} existing chunks")
                files, out_rels = [f for f, _ in todo], [r for _, r in todo]
            files_left[unit_dir] = len(files)
        else:
            todo_files, out_rels = [], []
            for f in files:
                rel = os.path.join(unit_dir, os.path.basename(f))
                todo = out.todo(rel)
                if todo == "copy":
                    out.publish(rel)
                if todo == "produce":
                    todo_files.append(f)
                    out_rels.append(rel)
            files = todo_files

        extra = extra_columns_for(cfg, dataset)
        for f, rel in zip(files, out_rels):
            task = (
                f,
                out.stage_path(rel),
                unit_models,
                cfg["chunk_size"],
                leaf,
                cfg["overwrite_columns"],
                extra,
            )
            tasks.append((unit_dir, rel, task))

    logger.info(f"Processing {len(tasks)} parquet files")
    n_done = 0
    failed = []  # (input file, error) of the files that failed
    failed_units = set()

    # With --merge, the merges and the copies to the output run in one
    # background thread (the only one using `out`), so that a slow or hanging
    # merge never blocks the collection of the results of the workers
    merge_ex = ThreadPoolExecutor(1, thread_name_prefix="merge") if merge else None
    merge_jobs = {}  # future -> merged file

    def submit_merge(fn, rel, *args):
        fut = merge_ex.submit(fn, *args)
        merge_jobs[fut] = rel

        def log_error(f):
            if f.exception() is not None:
                logger.error(f"Merge of {rel} failed: {f.exception()!r}")

        fut.add_done_callback(log_error)

    def merge_and_publish(files, rel):
        merge_files(files, out.stage_path(rel), cfg["merge_threads"], cfg["files_per_step"])
        out.publish(rel)

    def file_done(unit_dir, rel):
        nonlocal n_done
        n_done += 1
        progress(f"{n_done}/{len(tasks)} parquet files done")
        if n_done % 50 == 0:
            logger.info(f"{n_done}/{len(tasks)} parquet files done")
        if not merge:
            out.publish(rel)
            return
        files_left[unit_dir] -= 1
        if files_left[unit_dir] == 0:
            # all the files of the unit are done: merge and copy them
            submit_merge(merge_unit, os.path.join(unit_dir, MERGED_NAME), cfg, out, unit_dir)

    def file_failed(unit_dir, src, err):
        if isinstance(err, BrokenProcessPool):
            # a worker died (e.g. out of memory): all the other tasks fail too
            raise err
        logger.error(f"{src} failed: {err!r}", exc_info=err)
        failed.append((src, err))
        failed_units.add(unit_dir)
        progress(f"{src} failed")

    completed = False
    try:
        futures = {}
        if pool is not None:
            futures = {
                pool.submit(_process_parquet_file, task): (u, r, task[0]) for u, r, task in tasks
            }
        if merge:
            # in the meantime, merge the units without models
            for files, rel in merge_only_units:
                submit_merge(merge_and_publish, rel, files, rel)
            # units whose chunks were all left by a previous run
            for unit_dir, n_left in files_left.items():
                if n_left == 0:
                    submit_merge(
                        merge_unit, os.path.join(unit_dir, MERGED_NAME), cfg, out, unit_dir
                    )
        if pool is None:
            for unit_dir, rel, task in tasks:
                try:
                    _process_parquet_file(task)
                except Exception as err:
                    file_failed(unit_dir, task[0], err)
                    continue
                file_done(unit_dir, rel)
        else:
            for fut in as_completed(futures):
                unit_dir, rel, src = futures[fut]
                err = fut.exception()
                if err is not None:
                    file_failed(unit_dir, src, err)
                    continue
                file_done(unit_dir, rel)
        completed = True
    finally:
        if merge_ex is not None:
            if not completed:
                logger.info("Waiting for the running merge before stopping")
            merge_ex.shutdown(wait=True, cancel_futures=not completed)

    merge_failed = [rel for fut, rel in merge_jobs.items() if not fut.cancelled() and fut.exception()]
    if failed or merge_failed:
        lines = [f"  {src}: {err!r}" for src, err in failed]
        lines += [f"  merge of {rel}" for rel in merge_failed]
        if failed_units:
            lines.append(
                f"Not merged (some files failed): {sorted(failed_units)}"
                if merge
                else f"Directories with failed files: {sorted(failed_units)}"
            )
        raise RuntimeError(
            f"{len(failed)} parquet files and {len(merge_failed)} merges failed "
            "(restart to process only what is missing):\n" + "\n".join(lines)
        )


def merge_unit(cfg, out, unit_dir):
    """
    Merge the new parquet files of one unit (written in UNMERGED_SUBDIR/<unit_dir>)
    into <unit_dir>/merged.parquet, keeping the events order, copy it to the
    output and remove the unmerged files.
    """
    chunks_dir = os.path.join(out.stage, UNMERGED_SUBDIR, unit_dir)
    rel = os.path.join(unit_dir, MERGED_NAME)
    files = parquet_files(chunks_dir)
    if len(files) == 1:
        # e.g. input already merged: no need to read it again
        os.replace(files[0], out.stage_path(rel))
    else:
        merge_files(files, out.stage_path(rel), cfg["merge_threads"], cfg["files_per_step"])
    shutil.rmtree(chunks_dir, ignore_errors=True)
    out.publish(rel)


# ---------------------------------------------------------------------------
# Coffea columns
# ---------------------------------------------------------------------------


def find_coffea_files(inp, process_output_all):
    if inp.endswith(".coffea"):
        return [inp]
    files = sorted(glob.glob(os.path.join(inp, "*.coffea")))
    others = [f for f in files if os.path.basename(f) != "output_all.coffea"]
    if others and not process_output_all:
        if len(others) != len(files):
            logger.info(
                "Skipping output_all.coffea (set process_output_all: true to "
                "include it), the per-dataset files are used"
            )
        files = others
    return files


def is_columns_leaf(d):
    """True if the dict maps column names to column accumulators/arrays."""
    return bool(d) and all(hasattr(v, "value") or isinstance(v, np.ndarray) for v in d.values())


def iter_column_units(columns, variations):
    """
    Iterate over accumulator["columns"]: yield
    (sample, dataset, category, variation, column_dict). `variation` is None
    if the columns are saved without variations.
    """
    for sample, d_sample in columns.items():
        for dataset, d_dataset in d_sample.items():
            for category, d_cat in d_dataset.items():
                if is_columns_leaf(d_cat):
                    yield sample, dataset, category, None, d_cat
                    continue
                for variation, d_var in d_cat.items():
                    if variations and not matches(variation, variations):
                        continue
                    yield sample, dataset, category, variation, d_var


def _value(col):
    return col.value if hasattr(col, "value") else col


def n_events_from_columns(columns):
    """Number of events of a unit of flattened coffea columns."""
    for name in ["weight"] + [c for c in columns if c.endswith("_N")] + [
        c for c in columns if c.startswith("events_")
    ]:
        if name in columns:
            return len(_value(columns[name]))
    raise ValueError("Cannot determine the number of events from the columns")


def run_coffea_models(cfg, coffea_files, out, pool):
    from coffea.processor import column_accumulator
    from coffea.util import load, save

    for coffea_file in coffea_files:
        rel = os.path.basename(coffea_file)
        todo = out.todo(rel)
        if todo == "copy":
            out.publish(rel)
        if todo != "produce":
            continue
        logger.info(f"Loading {coffea_file}")
        acc = load(coffea_file)
        jobs = []
        for sample, dataset, category, variation, cols in iter_column_units(
            acc["columns"], cfg["variations"]
        ):
            unit_models = models_for(cfg["models"], dataset, category)
            unit_name = f"{sample}/{dataset}/{category}/{variation}"
            if not unit_models:
                logger.debug(f"No model for {unit_name}")
                continue
            n_events = n_events_from_columns(cols)
            needed = columns_needed(unit_models)
            columns = {k: _value(v) for k, v in cols.items() if k in needed}
            add_extra_columns(columns, extra_columns_for(cfg, dataset), needed, n_events)
            events = build_events(columns, unit_models, n_events, unit_name)
            logger.info(
                f"{unit_name}: {n_events} events, models {[m['name'] for m in unit_models]}"
            )
            futures = []
            for start in range(0, max(n_events, 1), cfg["chunk_size"]):
                chunk = events[start : start + cfg["chunk_size"]]
                if pool is None:
                    futures.append(evaluate_in_chunks(unit_models, chunk, cfg["chunk_size"]))
                else:
                    futures.append(
                        pool.submit(evaluate_in_chunks, unit_models, chunk, cfg["chunk_size"])
                    )
            jobs.append((unit_name, cols, futures))

        for unit_name, cols, futures in jobs:
            results = [f.result() if pool is not None else f for f in futures]
            scores = {k: np.concatenate([r[k] for r in results]) for k in results[0]}
            for name, arr in scores.items():
                if name in cols:
                    if not cfg["overwrite_columns"]:
                        raise ValueError(
                            f"{unit_name}: column {name} already exists, "
                            "set overwrite_columns: true to replace it"
                        )
                    logger.warning(f"{unit_name}: overwriting column {name}")
                cols[name] = column_accumulator(arr)
            logger.info(f"{unit_name}: added columns {list(scores)}")
            progress(f"{unit_name}: added columns")

        path = out.stage_path(rel)
        save(acc, path + ".tmp")
        os.replace(path + ".tmp", path)
        out.publish(rel)
        del acc


def copy_run_metadata(run_dir, out, config_path):
    """Copy the config.json and the other metadata (yaml/json) of the run, and
    the yaml config of this script, next to the new coffea files."""
    files = [
        f
        for f in sorted(glob.glob(os.path.join(run_dir, "*")))
        if os.path.isfile(f) and f.endswith((".yaml", ".json"))
    ]
    names = [os.path.basename(f) for f in files]
    if config_path:
        files.append(config_path)
        names.append("postprocess_parquet_config.yaml")
    for src, name in zip(files, names):
        shutil.copy(src, out.stage_path(name))
        out.publish(name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge the parquet chunks and/or apply ONNX models to the columns "
        "of a PocketCoffea output. The arguments override the yaml config.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="yaml config with all the options and the ONNX models "
        "(see scripts/postprocess_parquet_example.yaml)",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="PocketCoffea output directory (with config.json and the coffea files), "
        "a coffea file, or a parquet files directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (default for parquet columns: merged_parquet_files, "
        "onnx_parquet_files or onnx_merged_parquet_files next to the parquet directory)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel processes for the ONNX evaluation (default 1); without "
        "models, number of threads reading the files to merge (default 32)",
    )
    parser.add_argument(
        "--merge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Merge the parquet files into one merged.parquet per "
        "<dataset>/<category>/<variation> (default: true without models, false with models)",
    )
    parser.add_argument(
        "--update-config",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overwrite the parquet directory (dump_columns_as_arrays_per_chunk and "
        "save_chunk) in the config.json of --input with the output directory (the "
        "original values are kept as <key>_unmerged)",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Produce again the output files that already exist (skipped by default)",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Local directory where the files are produced before being copied to the "
        "output, needed if the output is on /pnfs (default: <output name>_tmp in "
        "/scratch/<user>, or in $TMPDIR or /tmp if there is no /scratch)",
    )
    parser.add_argument(
        "--remove-original",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Merge only: once all the merges are done, check for each "
        "<dataset>/<category>/<variation> that the merged file has the same number of "
        "events as the original parquet files (counted while merging) and then remove "
        "the original files (and the directories left empty). If any check fails, "
        "nothing is removed and the script fails",
    )
    parser.add_argument(
        "--stall-timeout",
        type=int,
        default=None,
        help="Exit with an error if nothing progresses for this many seconds, e.g. a read "
        "hanging on the /pnfs mount (default 3600, 0 = never); the temporary files are "
        "kept and reused by the next run",
    )
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=None,
        help="Seconds after which a read of an input file through the /pnfs mount is "
        "considered hanging and the file is read with xrdcp instead (default 300, "
        "0 = always read through the mount)",
    )
    parser.add_argument("--debug", action="store_true", help="Debug printout")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    cfg = parse_config(args)
    models = cfg["models"]

    # input: PocketCoffea output directory, coffea file or parquet directory
    inp = os.path.abspath(root_to_local(cfg["input"])).rstrip("/")
    run_dir = os.path.dirname(inp) if inp.endswith(".coffea") else inp
    config_file = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_file):
        run_dir, config_file = None, None
    if cfg["update_config"] and config_file is None:
        raise ValueError(f"--update-config needs a config.json in {inp}")

    # columns in the coffea files or in parquet files?
    coffea_files = find_coffea_files(inp, cfg["process_output_all"]) if run_dir else []
    columns_in_coffea = False
    if models and coffea_files:
        from coffea.util import load

        columns_in_coffea = bool(load(coffea_files[0]).get("columns"))

    parquet_dir, redirector = None, None
    if columns_in_coffea:
        logger.info(f"Columns saved in the coffea files: {coffea_files}")
        if cfg["merge"]:
            logger.warning("The columns are in the coffea files: nothing to merge")
        if cfg["update_config"]:
            logger.warning("The columns are in the coffea files: --update-config ignored")
            cfg["update_config"] = False
    else:
        if config_file:
            # merge only: always from the original chunks (also when the
            # config.json was updated by a previous run)
            parquet_dir, redirector = parquet_dir_from_config(config_file, original=not models)
        elif not inp.endswith(".coffea"):
            parquet_dir = inp
        parquet_dir = cfg["parquet_input"] or parquet_dir
        if not parquet_dir:
            raise ValueError(
                f"No parquet directory found ({', '.join(CONFIG_KEYS)} in the config.json "
                "of the input): set 'parquet_input' in the config"
            )
        parquet_dir = os.path.abspath(root_to_local(parquet_dir)).rstrip("/")
        if not os.path.isdir(parquet_dir):
            raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")
        logger.info(f"Columns saved as parquet files in {parquet_dir}")
    # xrootd redirector of the input, used if a read through /pnfs hangs
    input_redirector = redirector

    if models and cfg["update_config"] and not cfg["copy_unprocessed"]:
        logger.info("--update-config: copy_unprocessed set, to have all the columns in the output")
        cfg["copy_unprocessed"] = True
    if cfg["update_config"] and cfg["variations"]:
        logger.warning(
            f"--update-config: only the variations {cfg['variations']} are in the output"
        )

    # output and temporary directory
    output = cfg["output"]
    if output is None:
        if columns_in_coffea:
            raise ValueError("The output directory (-o) is needed for the coffea columns")
        if not models:
            name = "merged_parquet_files"
        else:
            name = "onnx_merged_parquet_files" if cfg["merge"] else "onnx_parquet_files"
        output = os.path.join(os.path.dirname(parquet_dir), name)
    redirector = cfg["redirector"] or redirector_of(output) or redirector
    output = os.path.abspath(root_to_local(output)).rstrip("/")
    if output in (parquet_dir, run_dir):
        raise ValueError("The output directory cannot be the input directory")

    if cfg["remove_original"]:
        for path in (output, os.path.abspath(cfg["tmp_dir"]) if cfg["tmp_dir"] else None):
            if path and (path == parquet_dir or path.startswith(parquet_dir + os.sep)):
                raise ValueError(
                    f"--remove-original: {path} is inside the parquet directory {parquet_dir}"
                )

    tmp_dir = cfg["tmp_dir"]
    if tmp_dir is None and output.startswith("/pnfs"):
        # local disk of the node (also in the slurm jobs), not /work or /tmp
        if os.path.isdir("/scratch"):
            base = os.path.join("/scratch", getpass.getuser())
        else:
            base = os.environ.get("TMPDIR") or "/tmp"
        tmp_dir = os.path.join(base, os.path.basename(output) + "_tmp")
    stage = os.path.abspath(tmp_dir) if tmp_dir else output
    # /pnfs read-only through the mount (batch nodes): write through xrootd,
    # otherwise (e.g. login nodes) copy with rsync
    if output.startswith("/pnfs") and not is_writable(output):
        redirector = (redirector or DEFAULT_REDIRECTOR).rstrip("/")
        check_proxy()
    else:
        redirector = None

    read_redirector = cfg["redirector"] or input_redirector or DEFAULT_REDIRECTOR
    reads = (cfg["read_timeout"], read_redirector, os.path.join(stage, "_input_tmp"))
    configure_reads(*reads)

    # --remove-original: /pnfs read-only through the mount (batch nodes):
    # remove through xrootd (checked now, not after merging)
    remove_redirector = None
    if cfg["remove_original"]:
        if parquet_dir.startswith("/pnfs") and not is_writable(parquet_dir):
            remove_redirector = read_redirector.rstrip("/")
            check_proxy()
        logger.info(
            f"--remove-original: the original files in {parquet_dir} are removed after "
            "the check of the merged files"
        )

    logger.info(f"Output: {output}")
    if stage != output:
        logger.info(f"Temporary directory: {stage}")
    if redirector is not None:
        logger.info(f"Copying to {output} with xrdcp through {redirector}")
    for m in models:
        logger.info(
            f"Model {m['name']}: {m['model']}, inputs {m['input_variables_spec']}, "
            f"column {m['column_name']}, regions {m['regions'] or 'all'}, "
            f"datasets {m['datasets'] or 'all'}"
        )

    pool = None
    if models and cfg["workers"] > 1:
        pool = ProcessPoolExecutor(
            max_workers=cfg["workers"], initializer=configure_reads, initargs=reads
        )
        # start all the processes now, before any other thread is created
        pool.submit(int).result()
        logger.info(f"Running with {cfg['workers']} parallel processes")

    # started after the worker processes, so that they do not inherit it
    global WATCHDOG
    WATCHDOG = Watchdog(cfg["stall_timeout"])
    WATCHDOG.start()

    out = Output(output, stage, redirector, cfg["overwrite"])
    completed = False
    merged = {}
    try:
        if not models:
            merged = run_merge(cfg, parquet_dir, out)
        elif columns_in_coffea:
            run_coffea_models(cfg, coffea_files, out, pool)
            copy_run_metadata(run_dir, out, args.config)
        else:
            run_parquet_models(cfg, parquet_dir, out, pool)
        completed = True
    finally:
        if pool is not None:
            # without waiting: a worker may have a thread hanging on /pnfs
            # (abandoned by read_with_timeout), all the results are collected
            pool.shutdown(wait=False, cancel_futures=True)
            for child in multiprocessing.active_children():
                child.kill()
        unmerged = os.path.join(stage, UNMERGED_SUBDIR)
        if completed:
            # do not leave GBs of unmerged files behind (e.g. a full /tmp)
            shutil.rmtree(unmerged, ignore_errors=True)
        elif os.path.isdir(unmerged):
            logger.warning(
                f"Unmerged files kept in {unmerged}: they are reused by the next run "
                "with the same config (remove them otherwise)"
            )
        out.close()

    if cfg["remove_original"]:
        # after out.close(): all the merged files are copied to the output.
        # Raises (nothing removed, config.json not updated) if a check fails
        check_merged(parquet_dir, merged)
    if cfg["update_config"]:
        update_config(config_file, os.path.join(output, cfg["parquet_subdir"] or ""))
    if cfg["remove_original"]:
        remove_original_files(parquet_dir, merged, remove_redirector, cfg["merge_threads"])
        if cfg["update_config"]:
            logger.info(
                f"The <key>{UNMERGED_SUFFIX} entries of {config_file} point to the removed "
                "original files"
            )
    logger.info(f"Done: output in {output}")


if __name__ == "__main__":
    main()
