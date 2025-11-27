import json
import logging
import os

import numpy as np
import pyarrow.dataset as ds
from coffea.util import load

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger()


def get_columns_from_files(inputfiles, sel_var="nominal", filter_lambda=None, debug=False, novars=False):
    if not debug:
        logger.setLevel(level=logging.INFO)
    if novars:
        return get_columns_from_files_novars(inputfiles, filter_lambda, debug)
    logger.info(f"Loading variations: {sel_var}")
    cat_col = {}
    total_datasets_list = []
    # get the columns
    for inputfile in inputfiles:
        accumulator = load(inputfile)
        samples = list(accumulator["columns"].keys())
        if accumulator["columns"] == {}:
            logger.info("Empty columns, trying to read from parquet files")
            return get_columns_from_parquet(inputfiles, sel_var, filter_lambda, debug)
            # return get_columns_from_parquet(inputfiles, filter_lambda, debug)
        if debug: logger.debug(f"inputfile {inputfile}")
        for sample in samples:
            if debug: logger.debug(f"sample {sample}")
            datasets = list(accumulator["columns"][sample].keys())
            for dataset in datasets:
                if dataset not in total_datasets_list:
                    total_datasets_list.append(dataset)
                if debug: logger.debug(f"dataset {dataset}")
                categories = list(accumulator["columns"][sample][dataset].keys())
                for category in categories:
                    if debug: logger.debug(f"category {category}")
                    if category not in cat_col:
                        cat_col[category] = {}
                    variations = list(accumulator["columns"][sample][dataset][category].keys())
                    logger.debug(sel_var)
                    if sel_var.lower() == "all":
                        logger.debug("loading all variations")
                        for variation in variations:
                            if debug: logger.debug(f"variation {variation}")
                            if variation not in cat_col[category]:
                                cat_col[category][variation] = {}
                            # cat_col[category][variation] = np.concatenate((cat_col[category][variation], fill_category(accumulator["columns"][sample][dataset][category], accumulator["sum_genweights"], dataset, filter_lambda, variation, debug)))
                            for i, column in enumerate(list(accumulator["columns"][sample][dataset][category][variation].keys())):
                                # filter with lamda function
                                if filter_lambda is not None:
                                    if not filter_lambda(column):
                                        if debug: logger.debug(f"Skipping column {column} due to filter")
                                        continue
                                column_array = accumulator["columns"][sample][dataset][category][variation][column].value

                                if column == "weight" and dataset in accumulator["sum_genweights"]:
                                    column_array = column_array / accumulator["sum_genweights"][dataset]

                                if column not in cat_col[category][variation]:
                                    cat_col[category][variation][column] = column_array
                                else:
                                    logger.debug("concatenating")
                                    cat_col[category][variation][column] = np.concatenate(
                                        (cat_col[category][variation][column], column_array)
                                    )
                                if i == 0:
                                    if debug: logger.debug(
                                        f"column {column}",
                                        column_array.shape,
                                        cat_col[category][variation][column].shape,
                                    )
                    elif str(sel_var) in variations:
                        # cat_col[category] = np.concatenate((cat_col[category], fill_category(accumulator["columns"][sample][dataset][category], accumulator["sum_genweights"], dataset, filter_lambda, sel_var, debug)))
                            for i, column in enumerate(list(accumulator["columns"][sample][dataset][category][sel_var].keys())):
                                # filter with lamda function
                                if filter_lambda is not None:
                                    if not filter_lambda(column):
                                        if debug: logger.debug(f"Skipping column {column} due to filter")
                                        continue
                                column_array = accumulator["columns"][sample][dataset][category][str(sel_var)][column].value

                                if column == "weight" and dataset in accumulator["sum_genweights"]:
                                    column_array = column_array / accumulator["sum_genweights"][dataset]

                                if column not in cat_col[category]:
                                    cat_col[category][column] = column_array
                                else:
                                    logger.debug("concatenating")
                                    cat_col[category][column] = np.concatenate(
                                        (cat_col[category][column], column_array)
                                    )
                                if i == 0:
                                    if debug: logger.debug(
                                        f"column {column}",
                                        column_array.shape,
                                        cat_col[category][column].shape,
                                    )
                    else:
                        raise ValueError(f"Variation {sel_var} not found in variations {variations}")
    return cat_col, total_datasets_list


def fill_category(accumulator, sum_genweights, dataset, filter_lambda, variation, debug):
    coldict = {}
    columns = list(
        accumulator[variation].keys()
    )
    for i, column in enumerate(columns):
        # filter with lamda function
        if filter_lambda is not None:
            if not filter_lambda(column):
                if debug: logger.debug(f"Skipping column {column} due to filter")
                continue
        column_array = accumulator[variation][column].value

        if column == "weight" and dataset in sum_genweights:
            column_array = column_array / sum_genweights[dataset]

        if column not in coldict:
            coldict[column] = column_array
        else:
            logger.debug("concatenating")
            coldict[column] = np.concatenate(
                (coldict[column], column_array)
            )
        if i == 0:
            if debug: logger.debug(
                f"column {column}",
                column_array.shape,
                coldict[column].shape,
            )
    return coldict


def get_columns_from_parquet(input_files, sel_var="nominal", filter_lambda=None, debug=False):
    cat_col = {}
    total_datasets_list = []
    dirs_datasets = {}

    for input_file in input_files:
        dir, dset = get_parquet_save_directory(input_file)
        dirs_datasets[dir] = dset

    for rootdir, sel_dataset in dirs_datasets.items():
        if debug:
            logger.debug(f"Scanning {rootdir}")
        datasets = os.listdir(rootdir) if sel_dataset == "all" else [sel_dataset]
        logger.debug(datasets)

        # structure: root/dataset/category/variation/*.parquet
        for dataset in datasets:
            dataset_path = os.path.join(rootdir, dataset)
            logger.debug(dataset_path)
            if not os.path.isdir(dataset_path):
                continue

            if dataset not in total_datasets_list:
                total_datasets_list.append(dataset)

            for category in os.listdir(dataset_path):
                category_path = os.path.join(dataset_path, category)
                if not os.path.isdir(category_path):
                    continue

                if debug:
                    logger.debug(f"dataset {dataset}, category {category}")

                if category not in cat_col:
                    cat_col[category] = {}

                if sel_var == "all" or not sel_var:
                    variations = os.listdir(category_path)
                else:
                    variations = [sel_var]

                single_var = True if len(variations) == 1 else False

                for variation in os.listdir(category_path):
                    if not sel_var.lower() == "all" and sel_var != variation:
                        logger.debug(f"Skipping variation {variation} as not demanded")
                        continue
                    variation_path = os.path.join(category_path, variation)
                    if not os.path.isdir(variation_path):
                        continue

                    if variation not in cat_col[category] and not single_var:
                        cat_col[category][variation] = {}
                        coldict = cat_col[category][variation]
                    elif single_var:
                        coldict = cat_col[category]

                    if debug:
                        logger.debug(f"  variation {variation}, files: {len(parquet_files)}")

                    logger.info(f"Loading datasets in {variation_path}")
                    parquet_files = ds.dataset(variation_path, format="parquet")
                    table = parquet_files.to_table()
                    df = table.to_pandas()
                    logger.info(f"Loaded datasets in {variation_path}")
                    logger.info(df)
                    for i, column in enumerate(df.columns):
                        # filter with lambda function
                        if filter_lambda is not None:
                            if not filter_lambda(column):
                                if debug:
                                    logger.debug(f"Skipping column {column} due to filter")
                                continue

                        column_array = df[column].to_numpy()

                        # normalize weights (if sum_genweights exists somewhere you may pass it separately)
                        if column == "weight" and "sum_genweights" in df.columns:
                            denom = df["sum_genweights"].iloc[0]
                            logger.debug(denom)
                            if denom != 0:
                                column_array = column_array / denom

                        if column not in coldict:
                            coldict[column] = column_array
                        else:
                            coldict[column] = np.concatenate(
                                (coldict[column], column_array)
                            )

                        if i == 0 and debug:
                            logger.debug(
                                f"column {column}",
                                column_array.shape,
                                coldict[column].shape,
                            )

    return cat_col, total_datasets_list


def get_parquet_save_directory(input_parquet):
    dataset = input_parquet.split("/")[-1].split(".")[0].split("_", 1)[-1]
    config_json_path = os.path.join(os.path.dirname(input_parquet), "config.json")
    try:
        with open(config_json_path, "r") as f:
            config = json.load(f)
        col_dir = config["workflow"]["workflow_options"]["dump_columns_as_arrays_per_chunk"]
        # Strip the redirector (e.g. root://t3dcachedb03.psi.ch:1094/) from the path if it exists
        if col_dir is not None and "://" in col_dir:
            col_dir = col_dir.split("://")[-1].split("/", 1)[-1]
            col_dir = "/" + col_dir.split("/", 1)[-1]
        logger.debug(f"dump_columns_as_arrays_per_chunk: {col_dir}")
    except Exception as e:
        logger.debug(f"Could not determine save directory (probably bad config.json): {e}")
        return None
    return col_dir, dataset


def get_columns_from_files_novars(inputfiles, filter_lambda=None, debug=False):
    cat_col = {}
    total_datasets_list = []
    # get the columns
    for inputfile in inputfiles:
        accumulator = load(inputfile)
        samples = list(accumulator["columns"].keys())
        if debug: print(f"inputfile {inputfile}")
        if accumulator["columns"] == {}:
            logger.info("Empty columns, trying to read from parquet files")
            return get_columns_from_parquet(inputfiles, sel_var, filter_lambda, debug)
            # return get_columns_from_parquet(inputfiles, filter_lambda, debug)
        for sample in samples:
            if debug: print(f"sample {sample}")
            datasets = list(accumulator["columns"][sample].keys())
            for dataset in datasets:
                if dataset not in total_datasets_list:
                    total_datasets_list.append(dataset)
                if debug: print(f"dataset {dataset}")
                categories = list(accumulator["columns"][sample][dataset].keys())
                for category in categories:
                    if debug: print(f"category {category}")
                    if category not in cat_col:
                        cat_col[category] = {}
                    columns = list(
                        accumulator["columns"][sample][dataset][category].keys()
                    )
                    for i, column in enumerate(columns):
                        # filter with lamda function
                        if filter_lambda is not None:
                            if not filter_lambda(column):
                                if debug: print(f"Skipping column {column} due to filter")
                                continue
                        column_array = accumulator["columns"][sample][dataset][
                            category
                        ][column].value

                        if column == "weight" and dataset in accumulator["sum_genweights"]:
                            column_array = column_array / accumulator["sum_genweights"][dataset]

                        if column not in cat_col[category]:
                            cat_col[category][column] = column_array
                        else:
                            cat_col[category][column] = np.concatenate(
                                (cat_col[category][column], column_array)
                            )

                        if i == 0:
                            if debug: print(
                                f"column {column}",
                                column_array.shape,
                                cat_col[category][column].shape,
                            )

    return cat_col, total_datasets_list
