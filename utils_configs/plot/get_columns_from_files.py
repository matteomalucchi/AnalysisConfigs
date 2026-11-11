import json
import logging
import os
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from coffea.util import load
import glob
import gc
from collections import defaultdict

PAD_VALUE = -999
# number of rows read at once from parquet (bounds the Arrow memory per thread)
BATCH_SIZE = 65_536

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger()


def get_columns_from_files(
    inputfiles, sel_var="nominal", filter_lambda=None, debug=False, novars=False, max_num_parquet_files=None, filter_mixed=False,
    categories=None, float32=False, max_events=None,
):
    """
    categories: optional iterable of category names to load from parquet
        (None = all). Skipping unused categories saves memory.
    float32: if True, float64 parquet columns (except weights) are converted
        to float32 to halve their memory.
    max_events: optional maximum number of events kept per category (None = all).
        From parquet only the needed rows are read; coffea columns are loaded
        and then truncated.
    """
    if not debug:
        logger.setLevel(level=logging.INFO)
    if novars:
        return get_columns_from_files_novars(inputfiles, filter_lambda, debug, max_num_parquet_files, filter_mixed, categories, float32, max_events)
    logger.info(f"Loading variations: {sel_var}")
    cat_col = {}
    total_datasets_list = []
    # get the columns
    for inputfile in inputfiles:
        accumulator = load(inputfile)
        samples = list(accumulator["columns"].keys())
        if accumulator["columns"] == {}:
            logger.info("Empty columns, trying to read from parquet files")
            return get_columns_from_parquet(inputfiles, sel_var, filter_lambda, debug, accumulator["sum_genweights"], max_num_parquet_files=max_num_parquet_files, filter_mixed=filter_mixed, categories=categories, float32=float32, max_events=max_events)
        if debug:
            logger.debug(f"inputfile {inputfile}")
        for sample in samples:
            if debug:
                logger.debug(f"sample {sample}")
            datasets = list(accumulator["columns"][sample].keys())
            for dataset in datasets:
                if dataset not in total_datasets_list:
                    total_datasets_list.append(dataset)
                if debug:
                    logger.debug(f"dataset {dataset}")
                categories = list(accumulator["columns"][sample][dataset].keys())
                for category in categories:
                    if debug:
                        logger.debug(f"category {category}")
                    if filter_mixed and "Mixed" in dataset and category not in ["4b_control_region_preW", "4b_control_region_postW", "4b_signal_region_preW", "4b_signal_region_postW"]:
                        logger.debug(f"skipping category {category} for dataset {dataset} due to MixedData skipping")
                        continue
                    if filter_mixed and "Mixed" not in dataset and category in ["4b_control_region_preW", "4b_control_region_postW", "4b_signal_region_preW", "4b_signal_region_postW"]:
                        logger.debug(f"skipping category {category} for dataset {dataset} due to MixedData skipping")
                        continue
                    if category not in cat_col:
                        cat_col[category] = {}
                    variations = list(
                        accumulator["columns"][sample][dataset][category].keys()
                    )
                    logger.debug(sel_var)
                    if sel_var.lower() == "all":
                        logger.debug("loading all variations")
                        for variation in variations:
                            if debug:
                                logger.debug(f"variation {variation}")
                            if variation not in cat_col[category]:
                                cat_col[category][variation] = {}
                            # cat_col[category][variation] = np.concatenate((cat_col[category][variation], fill_category(accumulator["columns"][sample][dataset][category], accumulator["sum_genweights"], dataset, filter_lambda, variation, debug)))
                            for i, column in enumerate(
                                list(
                                    accumulator["columns"][sample][dataset][category][
                                        variation
                                    ].keys()
                                )
                            ):
                                # filter with lamda function
                                if filter_lambda is not None:
                                    if not filter_lambda(column):
                                        if debug:
                                            logger.debug(
                                                f"Skipping column {column} due to filter"
                                            )
                                        continue
                                column_array = accumulator["columns"][sample][dataset][
                                    category
                                ][variation][column].value

                                if (
                                    column == "weight"
                                    and dataset in accumulator["sum_genweights"]
                                ):
                                    column_array = (
                                        column_array
                                        / accumulator["sum_genweights"][dataset]
                                    )

                                if column not in cat_col[category][variation]:
                                    cat_col[category][variation][column] = column_array
                                else:
                                    logger.debug("concatenating")
                                    cat_col[category][variation][column] = (
                                        np.concatenate(
                                            (
                                                cat_col[category][variation][column],
                                                column_array,
                                            )
                                        )
                                    )
                                if i == 0:
                                    if debug:
                                        logger.debug(
                                            f"column {column}",
                                            column_array.shape,
                                            cat_col[category][variation][column].shape,
                                        )
                    elif str(sel_var) in variations:
                        # cat_col[category] = np.concatenate((cat_col[category], fill_category(accumulator["columns"][sample][dataset][category], accumulator["sum_genweights"], dataset, filter_lambda, sel_var, debug)))
                        for i, column in enumerate(
                            list(
                                accumulator["columns"][sample][dataset][category][
                                    sel_var
                                ].keys()
                            )
                        ):
                            # filter with lamda function
                            if filter_lambda is not None:
                                if not filter_lambda(column):
                                    if debug:
                                        logger.debug(
                                            f"Skipping column {column} due to filter"
                                        )
                                    continue
                            column_array = accumulator["columns"][sample][dataset][
                                category
                            ][str(sel_var)][column].value

                            if (
                                column == "weight"
                                and dataset in accumulator["sum_genweights"]
                            ):
                                column_array = (
                                    column_array
                                    / accumulator["sum_genweights"][dataset]
                                )

                            if column not in cat_col[category]:
                                cat_col[category][column] = column_array
                            else:
                                logger.debug("concatenating")
                                cat_col[category][column] = np.concatenate(
                                    (cat_col[category][column], column_array)
                                )
                            if i == 0:
                                if debug:
                                    logger.debug(
                                        f"column {column}",
                                        column_array.shape,
                                        cat_col[category][column].shape,
                                    )
                    else:
                        raise ValueError(
                            f"Variation {sel_var} not found in variations {variations}"
                        )
    if max_events is not None:
        for category in cat_col:
            if sel_var.lower() == "all":
                for variation in cat_col[category]:
                    _truncate_columns(cat_col[category][variation], max_events)
            else:
                _truncate_columns(cat_col[category], max_events)
    return cat_col, total_datasets_list


def _truncate_columns(columns, max_events):
    """
    Keep only the first max_events events of a {column: array} dict (in place).
    Flattened jagged columns (<coll>_<var> of length sum(<coll>_N)) are cut
    at the objects of the kept events. The number of events is taken from "weight".
    """
    if "weight" not in columns or len(columns["weight"]) <= max_events:
        return
    n_events = len(columns["weight"])
    n_obj_kept = {
        c[:-2]: int(np.sum(columns[c][:max_events])) for c in columns if c.endswith("_N")
    }
    for column, arr in list(columns.items()):
        if len(arr) == n_events:
            columns[column] = arr[:max_events]
            continue
        coll = next((k for k in n_obj_kept if column.startswith(f"{k}_")), None)
        if coll is None:
            logger.warning(f"Cannot truncate column {column} (length {len(arr)} != {n_events} events)")
            continue
        columns[column] = arr[: n_obj_kept[coll]]


def fill_category(
    accumulator, sum_genweights, dataset, filter_lambda, variation, debug
):
    coldict = {}
    columns = list(accumulator[variation].keys())
    for i, column in enumerate(columns):
        # filter with lamda function
        if filter_lambda is not None:
            if not filter_lambda(column):
                if debug:
                    logger.debug(f"Skipping column {column} due to filter")
                continue
        column_array = accumulator[variation][column].value

        if column == "weight" and dataset in sum_genweights:
            column_array = column_array / sum_genweights[dataset]

        if column not in coldict:
            coldict[column] = column_array
        else:
            logger.debug("concatenating")
            coldict[column] = np.concatenate((coldict[column], column_array))
        if i == 0:
            if debug:
                logger.debug(
                    f"column {column}",
                    column_array.shape,
                    coldict[column].shape,
                )
    return coldict


def get_columns_from_parquet(
    input_files,
    sel_var="nominal",
    filter_lambda=None,
    debug=False,
    sum_genweights=None,
    max_num_parquet_files=None,
    num_workers=4,
    filter_mixed=False,
    categories=None,
    float32=False,
    max_events=None,
):
    """
    Parquet loader reading in batches of BATCH_SIZE rows, with
    ThreadPoolExecutor for parallel variation loading.
    If max_events is given, reading stops after max_events rows per
    (dataset, category, variation) and each category is truncated to max_events.

    Memory: only the requested `categories` are read, each Arrow batch is
    converted to numpy and released right away, list columns are converted
    to 2D arrays without passing through numpy object arrays, and the
    per-dataset arrays are concatenated once at the end. With float32=True, float64
    columns (except weights) are stored as float32.

    Returns:
        cat_col[category][(variation)][column]  -> numpy array   (sel_var="all")
        cat_col[category][column]               -> numpy array   (sel_var="nominal" or specific)
        total_datasets_list
    """
    if sum_genweights is None:
        sum_genweights = {}

    cat_col: dict = {}
    total_datasets_list: list = []
    dirs_datasets: dict = {}

    for input_file in input_files:
        result = get_parquet_save_directory(input_file)
        if result is None:
            logger.warning(f"Could not resolve parquet directory for {input_file}, skipping.")
            continue
        col_dir, dset = result
        dirs_datasets[dset] = col_dir

    load_all = sel_var.lower() == "all"

    # ------------------------------------------------------------------
    # Build a flat list of (variation_path, dataset, category, variation)
    # work units so we can parallelise over them.
    # ------------------------------------------------------------------
    work_units = []

    for sel_dataset, rootdir in dirs_datasets.items():
        logger.debug(f"Scanning {rootdir}")
        datasets = os.listdir(rootdir) if sel_dataset == "all" else [sel_dataset]

        for dataset in datasets:
            dataset_path = os.path.join(rootdir, dataset)
            if not os.path.isdir(dataset_path):
                continue

            if dataset not in total_datasets_list:
                total_datasets_list.append(dataset)

            for category in os.listdir(dataset_path):
                category_path = os.path.join(dataset_path, category)
                if not os.path.isdir(category_path):
                    continue
                if categories is not None and category not in categories:
                    logger.debug(f"skipping category {category} (not requested)")
                    continue
                if filter_mixed and "Mixed" in dataset and category not in ["4b_control_region_preW", "4b_control_region_postW", "4b_signal_region_preW", "4b_signal_region_postW"]:
                    logger.debug(f"skipping category {category} for dataset {dataset} due to MixedData skipping")
                    continue
                if filter_mixed and "Mixed" not in dataset and category in ["4b_control_region_preW", "4b_control_region_postW", "4b_signal_region_preW", "4b_signal_region_postW"]:
                    logger.debug(f"skipping category {category} for dataset {dataset} due to MixedData skipping")
                    continue

                if load_all:
                    variations = [
                        v for v in os.listdir(category_path)
                        if os.path.isdir(os.path.join(category_path, v))
                    ]
                elif not sel_var:
                    variations = [""]
                else:
                    variations = [sel_var]

                for variation in variations:
                    variation_path = os.path.join(category_path, variation)
                    if not os.path.isdir(variation_path):
                        continue
                    work_units.append((dataset, category, variation, variation_path))

    logger.info(f"Found {len(work_units)} (dataset, category, variation) units to load.")

    def _split_if_multidim(column: str, arr: np.ndarray) -> dict:
        """
        1D array -> {column: arr}
        2D array (n_events, n_objects) -> {column_0: ..., column_1: ...}
        The 2D array is transposed once so that every split column is a
        contiguous view of the same buffer (no extra copy).
        """
        if arr.ndim == 1:
            return {column: arr}
        if arr.ndim == 2:
            arr_t = np.ascontiguousarray(arr.T)
            return {f"{column}_{idx}": arr_t[idx] for idx in range(arr_t.shape[0])}
        raise ValueError(f"Column {column} has unsupported ndim={arr.ndim}, dtype={arr.dtype}")

    def _arrow_to_numpy(column: str, chunked) -> np.ndarray:
        """
        Convert an Arrow ChunkedArray to numpy. List columns are flattened in
        Arrow and reshaped to (n_events, n_objects), avoiding numpy object
        arrays (one Python object per event). Ragged lists are padded with
        PAD_VALUE.
        """
        typ = chunked.type
        if not (pa.types.is_list(typ) or pa.types.is_large_list(typ) or pa.types.is_fixed_size_list(typ)):
            return chunked.to_numpy()

        lengths = pc.fill_null(pc.list_value_length(chunked), 0).to_numpy()
        values = pc.list_flatten(chunked).to_numpy()
        n_events = len(lengths)
        n_objects = int(lengths.max()) if n_events else 0
        if n_events == 0 or (lengths == n_objects).all():
            return values.reshape(n_events, n_objects)

        logger.warning(
            f"Column {column} is ragged with lengths {np.unique(lengths)}; "
            f"padding to max length with PAD_VALUE"
        )
        padded = np.full((n_events, n_objects), PAD_VALUE, dtype=np.result_type(values.dtype, np.float64))
        padded[np.arange(n_objects) < lengths[:, None]] = values
        return padded

    # ------------------------------------------------------------------
    # Worker: load one variation_path -> returns column dict
    # ------------------------------------------------------------------
    def load_variation(dataset, category, variation, variation_path):
        files = sorted(glob.glob(os.path.join(variation_path, "*.parquet")))
        if not files:
            logger.warning(f"No parquet files found in {variation_path}")
            return dataset, category, variation, {}

        if max_num_parquet_files:
            files = files[:max_num_parquet_files]

        logger.debug(f"Loading {len(files)} files from {variation_path}")

        # Push column filter into Arrow — skips unneeded columns at file level
        schema_names = pq.read_schema(files[0]).names
        if filter_lambda is not None:
            schema_cols = [c for c in schema_names if filter_lambda(c)]
        else:
            schema_cols = schema_names

        # Read batch by batch (no pre-buffering) and convert each batch to
        # numpy right away, so that only one small Arrow batch is in memory at
        # a time (reading the full table at once more than doubles the peak)
        batches_dict = {c: [] for c in schema_cols}
        n_read = 0
        for file in files:
            if max_events is not None and n_read >= max_events:
                break
            parquet_file = pq.ParquetFile(file, pre_buffer=False)
            for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=schema_cols):
                if max_events is not None:
                    if n_read >= max_events:
                        break
                    batch = batch.slice(0, max_events - n_read)
                n_read += batch.num_rows
                for column in schema_cols:
                    arr = _arrow_to_numpy(column, batch.column(column))
                    if float32 and arr.dtype == np.float64 and "weight" not in column:
                        arr = arr.astype(np.float32)
                    batches_dict[column].append(arr)
                del batch
            del parquet_file

        coldict = {}
        for column in schema_cols:
            arrays = batches_dict.pop(column)
            if not arrays:
                continue
            # pad 2D arrays if the number of objects changes between batches
            if arrays[0].ndim == 2:
                n_objects = max(a.shape[1] for a in arrays)
                arrays = [
                    a if a.shape[1] == n_objects
                    else np.pad(a, ((0, 0), (0, n_objects - a.shape[1])), constant_values=PAD_VALUE)
                    for a in arrays
                ]
            arr = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
            del arrays
            arr = _apply_genweight(column, arr, dataset, sum_genweights)

            for split_column, split_arr in _split_if_multidim(column, arr).items():
                coldict[split_column] = split_arr
            del arr

        gc.collect()
        pa.default_memory_pool().release_unused()

        return dataset, category, variation, coldict

    # ------------------------------------------------------------------
    # Run workers in parallel (Arrow releases the GIL during I/O)
    # ------------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Arrays of the different datasets are collected in lists and concatenated
    # once at the end (repeated np.concatenate re-allocates the growing array)
    parts: dict = {}
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(load_variation, *unit): unit
            for unit in work_units
        }

        for future in as_completed(futures):
            dataset, category, variation, coldict = future.result()
            del futures[future]
            if not coldict:
                continue

            # sel_var="all"  ->  cat_col[category][variation][column]
            # sel_var=specific or "" -> cat_col[category][column]
            key = (category, variation if load_all else None)
            target = parts.setdefault(key, {})
            for column, arr in coldict.items():
                target.setdefault(column, []).append(arr)
            del coldict

    for (category, variation), columns in parts.items():
        cat_col.setdefault(category, {})
        target = cat_col[category].setdefault(variation, {}) if load_all else cat_col[category]
        for column in list(columns):
            arrays = columns.pop(column)
            target[column] = (arrays[0] if len(arrays) == 1 else np.concatenate(arrays))[:max_events]
            del arrays
    del parts
    gc.collect()

    logger.info("Parquet loading complete.")
    return cat_col, total_datasets_list


def _apply_genweight(column: str, arr: np.ndarray, dataset: str, sum_genweights: dict) -> np.ndarray:
    """Divide weight column by sum_genweights when applicable."""
    if column == "weight" and dataset in sum_genweights:
        denom = sum_genweights[dataset]
        if denom != 0:
            return arr / denom
    return arr


def get_parquet_save_directory(input_parquet):
    dataset = input_parquet.split("/")[-1].split(".")[0].split("_", 1)[-1]
    config_json_path = os.path.join(os.path.dirname(input_parquet), "config.json")
    try:
        with open(config_json_path, "r") as f:
            config = json.load(f)
        col_dir = config["workflow"]["workflow_options"][
            "dump_columns_as_arrays_per_chunk"
        ]
        # Strip the redirector (e.g. root://t3dcachedb03.psi.ch:1094/) from the path if it exists
        if col_dir is not None and "://" in col_dir:
            col_dir = col_dir.split("://")[-1].split("/", 1)[-1]
            col_dir = "/" + col_dir.split("/", 1)[-1]
        logger.debug(f"dump_columns_as_arrays_per_chunk: {col_dir}")
    except Exception as e:
        logger.debug(
            f"Could not determine save directory (probably bad config.json): {e}"
        )
        return None
    return col_dir, dataset


def get_columns_from_files_novars(inputfiles, filter_lambda=None, debug=False, max_num_parquet_files=None, filter_mixed=False, categories=None, float32=False, max_events=None):
    cat_col = {}
    total_datasets_list = []
    # get the columns
    for inputfile in inputfiles:
        accumulator = load(inputfile)
        samples = list(accumulator["columns"].keys())
        if debug:
            print(f"inputfile {inputfile}")
        if accumulator["columns"] == {}:
            logger.info("Empty columns, trying to read from parquet files")
            return get_columns_from_parquet(
                inputfiles, "", filter_lambda, debug, accumulator["sum_genweights"], max_num_parquet_files=max_num_parquet_files, filter_mixed=filter_mixed,
                categories=categories, float32=float32, max_events=max_events,
            )
        for sample in samples:
            if debug:
                print(f"sample {sample}")
            datasets = list(accumulator["columns"][sample].keys())
            for dataset in datasets:
                if dataset not in total_datasets_list:
                    total_datasets_list.append(dataset)
                if debug:
                    print(f"dataset {dataset}")
                categories = list(accumulator["columns"][sample][dataset].keys())
                for category in categories:
                    if debug:
                        print(f"category {category}")
                    if filter_mixed and "Mixed" in dataset and category not in ["4b_control_region_preW", "4b_control_region_postW", "4b_signal_region_preW", "4b_signal_region_postW"]:
                        logger.debug(f"skipping category {category} for dataset {dataset} due to MixedData skipping")
                        continue
                    if filter_mixed and "Mixed" not in dataset and category in ["4b_control_region_preW", "4b_control_region_postW", "4b_signal_region_preW", "4b_signal_region_postW"]:
                        logger.debug(f"skipping category {category} for dataset {dataset} due to MixedData skipping")
                        continue
                    if category not in cat_col:
                        cat_col[category] = {}
                    columns = list(
                        accumulator["columns"][sample][dataset][category].keys()
                    )
                    for i, column in enumerate(columns):
                        # filter with lamda function
                        if filter_lambda is not None:
                            if not filter_lambda(column):
                                if debug:
                                    print(f"Skipping column {column} due to filter")
                                continue
                        column_array = accumulator["columns"][sample][dataset][
                            category
                        ][column].value

                        if (
                            column == "weight"
                            and dataset in accumulator["sum_genweights"]
                        ):
                            column_array = (
                                column_array / accumulator["sum_genweights"][dataset]
                            )

                        if column not in cat_col[category]:
                            cat_col[category][column] = column_array
                        else:
                            cat_col[category][column] = np.concatenate(
                                (cat_col[category][column], column_array)
                            )

                        if i == 0:
                            if debug:
                                print(
                                    f"column {column}",
                                    column_array.shape,
                                    cat_col[category][column].shape,
                                )

    if max_events is not None:
        for category in cat_col:
            _truncate_columns(cat_col[category], max_events)
    return cat_col, total_datasets_list
