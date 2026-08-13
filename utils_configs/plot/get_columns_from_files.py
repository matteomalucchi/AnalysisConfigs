import json
import logging
import os
import numpy as np
import pyarrow.dataset as ds
from coffea.util import load
import glob
import gc
from collections import defaultdict

PAD_VALUE = -999

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger()


def get_columns_from_files(
    inputfiles, sel_var="nominal", filter_lambda=None, debug=False, novars=False, max_num_parquet_files=None, filter_mixed=False,
):
    if not debug:
        logger.setLevel(level=logging.INFO)
    if novars:
        return get_columns_from_files_novars(inputfiles, filter_lambda, debug, max_num_parquet_files)
    logger.info(f"Loading variations: {sel_var}")
    cat_col = {}
    total_datasets_list = []
    # get the columns
    for inputfile in inputfiles:
        accumulator = load(inputfile)
        samples = list(accumulator["columns"].keys())
        if accumulator["columns"] == {}:
            logger.info("Empty columns, trying to read from parquet files")
            return get_columns_from_parquet(inputfiles, sel_var, filter_lambda, debug, accumulator["sum_genweights"], max_num_parquet_files=max_num_parquet_files, filter_mixed=True)
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
    return cat_col, total_datasets_list


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
):
    """
    Fast Parquet loader using Arrow's to_table() (no batching overhead) and
    ThreadPoolExecutor for parallel variation loading.

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

    def _split_if_multidim(column: str, arr) -> dict:
        """
        Handles three cases:
          1. Plain 1D array -> {column: arr}
          2. Plain 2D array (n_events, n_objects) -> {column_0: ..., column_1: ...}
          3. Object-dtype array of variable-length lists (ragged) -> pad to fixed N
             using PAD_VALUE, then split, OR raise if ragged and no padding strategy.
        """
        arr = np.asarray(arr)

        if arr.ndim == 1 and arr.dtype != object:
            return {column: arr}

        if arr.ndim == 2:
            n_objects = arr.shape[1]
            return {f"{column}_{idx}": arr[:, idx] for idx in range(n_objects)}

        if arr.dtype == object:
            # Ragged array of lists/arrays per event
            lengths = {len(row) for row in arr}
            if len(lengths) != 1:
                logger.warning(
                    f"Column {column} is ragged with varying lengths {lengths}; "
                    f"padding to max length with PAD_VALUE"
                )
                n_objects = max(lengths)
                padded = np.full((len(arr), n_objects), PAD_VALUE, dtype=np.float64)
                for row_idx, row in enumerate(arr):
                    padded[row_idx, : len(row)] = row
                arr2d = padded
            else:
                n_objects = lengths.pop()
                arr2d = np.stack([np.asarray(row, dtype=np.float64) for row in arr])

            return {f"{column}_{idx}": arr2d[:, idx] for idx in range(n_objects)}

        raise ValueError(f"Column {column} has unsupported ndim={arr.ndim}, dtype={arr.dtype}")

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

        arrow_ds = ds.dataset(files, format="parquet")

        # Push column filter into Arrow — skips unneeded columns at file level
        if filter_lambda is not None:
            schema_cols = [c for c in arrow_ds.schema.names if filter_lambda(c)]
        else:
            schema_cols = None  # read all

        # Load everything in one shot — no batching overhead
        table = arrow_ds.to_table(columns=schema_cols)

        coldict = {}
        for column in table.schema.names:
            # Zero-copy where Arrow buffer is already C-contiguous, else a single copy
            arr = table.column(column).combine_chunks().to_numpy(zero_copy_only=False)
            arr = _apply_genweight(column, arr, dataset, sum_genweights)

            for split_column, split_arr in _split_if_multidim(column, arr).items():
                coldict[split_column] = split_arr

        del table, arrow_ds
        gc.collect()

        return dataset, category, variation, coldict

    # ------------------------------------------------------------------
    # Run workers in parallel (Arrow releases the GIL during I/O)
    # ------------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(load_variation, *unit): unit
            for unit in work_units
        }

        for future in as_completed(futures):
            dataset, category, variation, coldict = future.result()
            if not coldict:
                continue

            cat_col.setdefault(category, {})

            # sel_var="all"  ->  cat_col[category][variation][column]
            # sel_var=specific or "" -> cat_col[category][column]
            if load_all:
                target = cat_col[category].setdefault(variation, {})
            else:
                target = cat_col[category]

            for column, arr in coldict.items():
                _merge_into(target, column, arr)

    logger.info("Parquet loading complete.")
    return cat_col, total_datasets_list


def _apply_genweight(column: str, arr: np.ndarray, dataset: str, sum_genweights: dict) -> np.ndarray:
    """Divide weight column by sum_genweights when applicable."""
    if column == "weight" and dataset in sum_genweights:
        denom = sum_genweights[dataset]
        if denom != 0:
            return arr / denom
    return arr


def _merge_into(coldict: dict, column: str, arr: np.ndarray) -> None:
    """Concatenate arr into coldict[column], creating the key if absent."""
    if column not in coldict:
        coldict[column] = arr
    else:
        coldict[column] = np.concatenate((coldict[column], arr))


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


def get_columns_from_files_novars(inputfiles, filter_lambda=None, debug=False, max_num_parquet_files=None, filter_mixed=False):
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
                inputfiles, "", filter_lambda, debug, accumulator["sum_genweights"], max_num_parquet_files=max_num_parquet_files
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

    return cat_col, total_datasets_list
