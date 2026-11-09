"""
Apply ONNX models to the columns saved by a PocketCoffea run.

The columns can be stored either inside the coffea files (`columns`
accumulator) or as parquet files per chunk (`save_chunk` option, i.e.
`dump_columns_as_arrays_per_chunk`). In both cases the `events` record seen by
the workflow is rebuilt from the saved columns and the models are evaluated
with `get_onnx_prediction` from `utils_configs/dnn_evaluation_functions.py`,
i.e. with exactly the same input definitions (`dnn_input_variables.py`) and
the same logic as in the standard PocketCoffea run.

The output contains all the original columns plus the ONNX scores:
- coffea columns: a copy of the coffea files with the score columns added;
- parquet columns: a copy of every parquet file with the score columns added,
  the coffea files and a `config.json` pointing to the new parquet directory,
  so that the plotting scripts can read the new output directly.

If the output directory is on /pnfs (or is an xrootd url), the files are first
written in a local temporary directory and then copied with `xrdcp`, since
the slurm jobs cannot write on /pnfs directly.

Usage:
    python scripts/apply_onnx_models.py -c scripts/apply_onnx_models_example.yaml
    python scripts/apply_onnx_models.py -c config.yaml -i <coffea_dir> -o <out_dir> -j 8

See `scripts/apply_onnx_models_example.yaml` for all the configuration options.
"""

import argparse
import fnmatch
import glob
import importlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import awkward as ak
import numpy as np
import yaml

# allow running the script from any directory
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from utils_configs.dnn_evaluation_functions import (  # noqa: E402
    get_collections,
    get_onnx_prediction,
)

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("apply_onnx_models")

# Same defaults as configs/HH4b_common/config_files/default_config.py
DEFAULT_PAD_VALUE = -999.0
DEFAULT_PAD_VALUE_SPANET = 9999.0
DEFAULT_REDIRECTOR = "root://t3dcachedb03.psi.ch:1094/"

# SPANet sequential features computed on the fly from `btagB`, they are not
# columns of the output
SPANET_DERIVED_FEATURES = {"btag12_ratioSubLead", "btag_ratioAll"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


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


def parse_config(config_path, args):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    base_dir = os.path.dirname(os.path.abspath(config_path))

    # command line arguments override the yaml
    if args.input:
        cfg["input"] = args.input
    if args.output:
        cfg["output"] = args.output
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.chunk_size is not None:
        cfg["chunk_size"] = args.chunk_size
    if args.tmp_dir:
        cfg["tmp_dir"] = args.tmp_dir

    for key in ["input", "output", "models"]:
        if not cfg.get(key):
            raise ValueError(f"'{key}' must be specified in the config (or command line)")

    cfg.setdefault("workers", 1)
    cfg.setdefault("chunk_size", 100000)
    cfg.setdefault("variations", None)
    cfg.setdefault("redirector", DEFAULT_REDIRECTOR)
    cfg.setdefault("tmp_dir", None)
    cfg.setdefault("process_output_all", False)
    cfg.setdefault("max_parquet_files", None)
    cfg.setdefault("overwrite", False)
    if isinstance(cfg["input"], str):
        cfg["input"] = [cfg["input"]]

    models = []
    column_names = set()
    for i, m in enumerate(cfg["models"]):
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
    cfg["models"] = models
    return cfg


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
# Columns needed by the models and reconstruction of the events record
# ---------------------------------------------------------------------------


def required_fields(models):
    """
    Return, for each collection, the needed features and how the collection is
    accessed: `flat` (one object per event, e.g. `HiggsLeading`) or `jagged`
    (`Collection:N` and the SPANet sequential inputs).
    """
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
        if "var *" not in str(arr.type.content):
            arr = ak.from_regular(arr, axis=1)
        return arr
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
                coll_fields["btagB"] = ak.ones_like(coll_fields["pt"])
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


def evaluate_chunk(models, events):
    """Evaluate all the models on a chunk of events."""
    out = {}
    for model in models:
        out.update(evaluate_model(model, events))
    return out


def evaluate_in_chunks(models, events, chunk_size):
    """Evaluate sequentially in chunks of `chunk_size` events and concatenate."""
    n = len(events)
    if n == 0:
        return {name: np.zeros(0, dtype=np.float32) for name in empty_columns(models)}
    pieces = defaultdict(list)
    for start in range(0, n, chunk_size):
        for k, v in evaluate_chunk(models, events[start : start + chunk_size]).items():
            pieces[k].append(v)
    return {k: np.concatenate(v) for k, v in pieces.items()}


def empty_columns(models):
    # Without events the number of output columns of multi-class models
    # cannot be known: only the base column name is created
    return [m["column_name"] for m in models]


# ---------------------------------------------------------------------------
# Output handling (local, /pnfs via a temporary directory)
# ---------------------------------------------------------------------------


class OutputManager:
    """
    Write to the final output directory, or to a local temporary directory
    which is copied to /pnfs (or to an xrootd url) at the end, as done in
    `merge_parquet`, since the slurm jobs cannot write on /pnfs.
    """

    def __init__(self, output, redirector, tmp_dir=None):
        self.output = output.rstrip("/")
        self.redirector = redirector.rstrip("/") + "/"
        self.remote = output.startswith("/pnfs") or "://" in output
        if self.remote:
            base = tmp_dir or os.environ.get("TMPDIR") or tempfile.gettempdir()
            os.makedirs(base, exist_ok=True)
            self.local = tempfile.mkdtemp(prefix="apply_onnx_", dir=base)
            logger.info(
                f"Output on {output}: writing in the temporary directory {self.local} "
                "and copying at the end"
            )
        else:
            self.local = os.path.abspath(output)
            os.makedirs(self.local, exist_ok=True)

    def local_path(self, *relpath):
        path = os.path.join(self.local, *relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def final_url(self, relpath=""):
        """Path (with redirector if on /pnfs) of a file in the final output."""
        path = os.path.join(self.output, relpath) if relpath else self.output
        if path.startswith("/pnfs"):
            return self.redirector + path
        return path

    def final_local_path(self, relpath=""):
        """Path of a file in the final output as seen from the file system."""
        path = os.path.join(self.output, relpath) if relpath else self.output
        if "://" in path:
            path = "/" + path.split("://", 1)[-1].split("/", 1)[-1].lstrip("/")
        return path

    def copy_file(self, relpath):
        """Copy one file from the temporary directory to the final output."""
        if not self.remote:
            return
        src = os.path.join(self.local, relpath)
        dst = self.final_url(relpath)
        cmd = ["xrdcp", "-f", "-s", "-p", src, dst]
        for attempt in range(4):
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                os.remove(src)
                return
            logger.warning(f"xrdcp {src} -> {dst} failed (attempt {attempt + 1}): {res.stderr}")
            time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"Could not copy {src} to {dst}")

    def finalize(self):
        if not self.remote:
            return
        for root, _, files in os.walk(self.local):
            for f in files:
                self.copy_file(os.path.relpath(os.path.join(root, f), self.local))
        shutil.rmtree(self.local, ignore_errors=True)
        logger.info(f"Output copied to {self.output}")


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------


def find_coffea_files(inputs, process_output_all):
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            dir_files = sorted(glob.glob(os.path.join(inp, "*.coffea")))
            others = [f for f in dir_files if os.path.basename(f) != "output_all.coffea"]
            if others and not process_output_all:
                skipped = [f for f in dir_files if f not in others]
                if skipped:
                    logger.info(
                        "Skipping output_all.coffea (set process_output_all: true to "
                        "include it), the per-dataset files are used"
                    )
                dir_files = others
            files += dir_files
        elif inp.endswith(".coffea"):
            files.append(inp)
        else:
            raise ValueError(f"Input {inp} is neither a directory nor a coffea file")
    if not files:
        raise ValueError(f"No coffea files found in {inputs}")
    return files


def read_config_json(coffea_file):
    path = os.path.join(os.path.dirname(coffea_file), "config.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        return path, json.load(f)


def strip_redirector(path):
    if path is not None and "://" in path:
        path = "/" + path.split("://", 1)[-1].split("/", 1)[-1].lstrip("/")
    return path


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


# ---------------------------------------------------------------------------
# Coffea columns
# ---------------------------------------------------------------------------


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


def _evaluate_unit_task(models, events, chunk_size):
    return evaluate_in_chunks(models, events, chunk_size)


def process_coffea_columns(cfg, coffea_files, out, pool):
    from coffea.processor import column_accumulator
    from coffea.util import load, save

    for coffea_file in coffea_files:
        logger.info(f"Loading {coffea_file}")
        acc = load(coffea_file)
        # futures: (unit key) -> list of (start, future)
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
                        pool.submit(_evaluate_unit_task, unit_models, chunk, cfg["chunk_size"])
                    )
            jobs.append((unit_name, cols, futures))

        for unit_name, cols, futures in jobs:
            results = [f.result() if pool is not None else f for f in futures]
            scores = {k: np.concatenate([r[k] for r in results]) for k in results[0]}
            for name, arr in scores.items():
                if name in cols:
                    if not cfg["overwrite"]:
                        raise ValueError(
                            f"{unit_name}: column {name} already exists, "
                            "set overwrite: true to replace it"
                        )
                    logger.warning(f"{unit_name}: overwriting column {name}")
                cols[name] = column_accumulator(arr)
            logger.info(f"{unit_name}: added columns {list(scores)}")

        relpath = os.path.basename(coffea_file)
        save(acc, out.local_path(relpath))
        out.copy_file(relpath)
        del acc

    copy_config_json(coffea_files[0], out, parquet_dir=None)


def copy_config_json(coffea_file, out, parquet_dir):
    path, config = read_config_json(coffea_file)
    if config is None:
        return
    if parquet_dir is not None:
        config["workflow"]["workflow_options"]["dump_columns_as_arrays_per_chunk"] = parquet_dir
    with open(out.local_path("config.json"), "w") as f:
        json.dump(config, f, indent=4)
    out.copy_file("config.json")
    # keep the other metadata files of the run (e.g. parameters dump)
    for extra in glob.glob(os.path.join(os.path.dirname(coffea_file), "*")):
        name = os.path.basename(extra)
        if name == "config.json" or name.endswith(".coffea") or not os.path.isfile(extra):
            continue
        if name.endswith((".yaml", ".json")):
            shutil.copy(extra, out.local_path(name))
            out.copy_file(name)


# ---------------------------------------------------------------------------
# Parquet columns
# ---------------------------------------------------------------------------


def _process_parquet_file(task):
    """
    Read one parquet file, evaluate the models and write a copy with the new
    columns. Run in a separate process when workers > 1.
    """
    src, dst, models, chunk_size, unit_name, overwrite = task
    # read and written with awkward (as done by PocketCoffea), so that the
    # awkward metadata stored in the parquet file stays consistent
    array = ak.from_parquet(src)
    if models:
        needed = columns_needed(models)
        columns = {c: array[c] for c in array.fields if c in needed}
        events = build_events(columns, models, len(array), unit_name)
        scores = evaluate_in_chunks(models, events, chunk_size)
        for name, arr in scores.items():
            if name in array.fields:
                if not overwrite:
                    raise ValueError(
                        f"{unit_name}: column {name} already exists in {src}, "
                        "set overwrite: true to replace it"
                    )
                logger.warning(f"{unit_name}: overwriting column {name} in {src}")
            array = ak.with_field(array, arr, name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ak.to_parquet(array, dst)
    return dst, len(array)


def list_parquet_units(col_dir, datasets, variations):
    """
    Yield (dataset, category, variation, [parquet files]) from the directory
    tree written by PocketCoffea: <col_dir>/<dataset>/<category>[/<variation>]/*.parquet
    """
    for dataset in datasets:
        dpath = os.path.join(col_dir, dataset)
        if not os.path.isdir(dpath):
            logger.warning(f"No parquet directory for dataset {dataset} in {col_dir}")
            continue
        for category in sorted(os.listdir(dpath)):
            cpath = os.path.join(dpath, category)
            if not os.path.isdir(cpath):
                continue
            direct = sorted(glob.glob(os.path.join(cpath, "*.parquet")))
            if direct:
                yield dataset, category, None, direct
            for variation in sorted(os.listdir(cpath)):
                vpath = os.path.join(cpath, variation)
                if not os.path.isdir(vpath):
                    continue
                if variations and not matches(variation, variations):
                    continue
                yield dataset, category, variation, sorted(
                    glob.glob(os.path.join(vpath, "*.parquet"))
                )


def process_parquet_columns(cfg, coffea_files, col_dir, out, pool):
    # datasets to process: one per coffea file `output_<dataset>.coffea`,
    # or all the datasets in the parquet directory for output_all.coffea
    datasets = []
    for f in coffea_files:
        name = os.path.basename(f)
        if name == "output_all.coffea":
            datasets = sorted(
                d for d in os.listdir(col_dir) if os.path.isdir(os.path.join(col_dir, d))
            )
            break
        datasets.append(name[: -len(".coffea")].split("_", 1)[-1])
    datasets = list(dict.fromkeys(datasets))

    parquet_rel = cfg.get("parquet_subdir", "parquet_files")
    tasks = []
    for dataset, category, variation, files in list_parquet_units(
        col_dir, datasets, cfg["variations"]
    ):
        unit_models = models_for(cfg["models"], dataset, category)
        unit_name = f"{dataset}/{category}/{variation}"
        if cfg["max_parquet_files"]:
            files = files[: cfg["max_parquet_files"]]
        logger.info(
            f"{unit_name}: {len(files)} files, models {[m['name'] for m in unit_models]}"
        )
        for f in files:
            rel = os.path.relpath(f, col_dir)
            relpath = os.path.join(parquet_rel, rel)
            tasks.append(
                (
                    relpath,
                    (
                        f,
                        out.local_path(relpath),
                        unit_models,
                        cfg["chunk_size"],
                        unit_name,
                        cfg["overwrite"],
                    ),
                )
            )

    logger.info(f"Processing {len(tasks)} parquet files")
    n_done = 0
    if pool is None:
        for relpath, task in tasks:
            _process_parquet_file(task)
            out.copy_file(relpath)
            n_done += 1
            if n_done % 50 == 0:
                logger.info(f"{n_done}/{len(tasks)} parquet files done")
    else:
        futures = {pool.submit(_process_parquet_file, task): relpath for relpath, task in tasks}
        for fut in as_completed(futures):
            fut.result()
            # copied as soon as it is ready, to keep the temporary space small
            out.copy_file(futures[fut])
            n_done += 1
            if n_done % 50 == 0:
                logger.info(f"{n_done}/{len(tasks)} parquet files done")

    # the coffea files (without columns) and the config.json pointing to the
    # new parquet directory, so that the output can be read as the input
    for f in coffea_files:
        relpath = os.path.basename(f)
        shutil.copy(f, out.local_path(relpath))
        out.copy_file(relpath)
    if os.path.basename(coffea_files[0]) != "output_all.coffea":
        all_file = os.path.join(os.path.dirname(coffea_files[0]), "output_all.coffea")
        if os.path.exists(all_file) and all_file not in coffea_files:
            shutil.copy(all_file, out.local_path("output_all.coffea"))
            out.copy_file("output_all.coffea")
    copy_config_json(coffea_files[0], out, parquet_dir=out.final_url(parquet_rel))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Apply ONNX models to the columns (coffea or parquet) of a PocketCoffea output"
    )
    parser.add_argument("-c", "--config", required=True, help="yaml configuration file")
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        default=None,
        help="Input directory with the coffea files (or coffea files), overrides the config",
    )
    parser.add_argument("-o", "--output", default=None, help="Output directory, overrides the config")
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel processes (1 = single process), overrides the config",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="Number of events per chunk, overrides the config"
    )
    parser.add_argument("--tmp-dir", default=None, help="Local temporary directory for /pnfs outputs")
    parser.add_argument("--debug", action="store_true", help="Debug printout")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    cfg = parse_config(args.config, args)

    for m in cfg["models"]:
        logger.info(
            f"Model {m['name']}: {m['model']}, inputs {m['input_variables_spec']}, "
            f"column {m['column_name']}, regions {m['regions'] or 'all'}, "
            f"datasets {m['datasets'] or 'all'}"
        )

    coffea_files = find_coffea_files(cfg["input"], cfg["process_output_all"])
    logger.info(f"Input coffea files: {coffea_files}")

    # columns in the coffea files or in parquet files?
    from coffea.util import load

    first = load(coffea_files[0])
    columns_in_coffea = bool(first.get("columns"))
    del first

    col_dir = None
    if not columns_in_coffea:
        _, config = read_config_json(coffea_files[0])
        try:
            col_dir = strip_redirector(
                config["workflow"]["workflow_options"]["dump_columns_as_arrays_per_chunk"]
            )
        except (TypeError, KeyError):
            col_dir = None
        col_dir = cfg.get("parquet_input") or col_dir
        if not col_dir:
            raise ValueError(
                "The coffea files have no columns and the parquet directory could not "
                "be found in config.json: set 'parquet_input' in the config"
            )
        col_dir = strip_redirector(col_dir)
        logger.info(f"Columns saved as parquet files in {col_dir}")

    out = OutputManager(cfg["output"], cfg["redirector"], cfg["tmp_dir"])

    pool = None
    if cfg["workers"] > 1:
        pool = ProcessPoolExecutor(max_workers=cfg["workers"])
        logger.info(f"Running with {cfg['workers']} parallel processes")
    else:
        logger.info("Running in a single process")

    try:
        if columns_in_coffea:
            process_coffea_columns(cfg, coffea_files, out, pool)
        else:
            process_parquet_columns(cfg, coffea_files, col_dir, out, pool)
    finally:
        if pool is not None:
            pool.shutdown()

    # save the configuration used, for reference
    shutil.copy(args.config, out.local_path("apply_onnx_models_config.yaml"))
    out.copy_file("apply_onnx_models_config.yaml")
    out.finalize()
    logger.info("Done")


if __name__ == "__main__":
    main()
