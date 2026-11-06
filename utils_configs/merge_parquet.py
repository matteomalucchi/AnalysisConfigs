"""Merge the many small parquet chunks saved by PocketCoffea into a single
parquet file per leaf directory (e.g. <dataset>/<region>/nominal).

Reading thousands of small files from dCache (/pnfs) is dominated by the
per-file latency, so merging them once makes the loading of the columns (e.g.
the training dataset of ML_pytorch) much faster. The directory structure and
the events order are kept, so the merged directory can be used in place of
the original parquet directory.

By default the merged files are written in a directory called
``merged_parquet_files`` next to the original parquet directory (i.e. in the
same parent directory, e.g. ``<...>/parquet_files`` ->
``<...>/merged_parquet_files``). Use -o to choose another directory.

Files on /pnfs cannot be written directly: if the output directory is on
/pnfs, each file is first merged in a temporary directory (--tmp-dir), then
copied to the output directory, the copy is checked (size, adler32 checksum
and number of events) and the temporary file is removed.

With --update-config, the parquet path in the config.json of the PocketCoffea
output directory (``dump_columns_as_arrays_per_chunk`` and ``save_chunk`` in
``workflow.workflow_options``) is overwritten with the path of the merged
files, so that the scripts reading the columns through the config.json use
the merged files directly. The original paths are kept in the same dictionary
with the suffix ``_unmerged`` (and are used as input when merging again).

Usage:
    python merge_parquet.py -i <PocketCoffea output dir or parquet files dir>
                            [-o <output_dir>] [--tmp-dir <tmp_dir>] [-j 32]
                            [--update-config]
"""

import argparse
import json
import logging
import os
import shutil
import zlib
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# keys of config["workflow"]["workflow_options"] with the parquet directory
CONFIG_KEYS = ("dump_columns_as_arrays_per_chunk", "save_chunk")
UNMERGED_SUFFIX = "_unmerged"
DEFAULT_OUTPUT_NAME = "merged_parquet_files"


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


def local_like(original, local_path):
    """local_path written in the same form of original (with the same
    root://host:port/ redirector if original has one)."""
    if isinstance(original, str) and original.startswith("root://"):
        u = urlsplit(original)
        return f"{u.scheme}://{u.netloc}/{os.path.abspath(local_path)}"
    return os.path.abspath(local_path)


def list_parquet_files(path):
    """Sorted list of the files in a parquet directory (searched recursively),
    in the same order used by ML_pytorch to load them.

    As in pyarrow.dataset, the files and directories starting with "." or "_"
    (e.g. _SUCCESS) are ignored.
    """
    if os.path.isfile(path):
        return [path]
    files = []
    for root, dirs, filenames in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_")))
        for name in sorted(filenames):
            if not name.startswith((".", "_")):
                files.append(os.path.join(root, name))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    return files


def find_leaf_dirs(path):
    """Directories directly containing parquet files (sorted)."""
    leaves = []
    for root, dirs, filenames in os.walk(path, followlinks=True):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_")))
        if any(
            f.endswith(".parquet") and not f.startswith((".", "_")) for f in filenames
        ):
            leaves.append(root)
    return leaves


def merge_dir(in_dir, out_file, n_threads, files_per_step):
    files = [f for f in list_parquet_files(in_dir) if f.endswith(".parquet")]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    tmp_file = out_file + ".tmp"
    writer = None
    n_rows = 0
    with ThreadPoolExecutor(n_threads) as ex:
        # read the files in parallel in steps to limit the memory usage,
        # map keeps the order of the files
        for i in range(0, len(files), files_per_step):
            tables = list(ex.map(pq.read_table, files[i : i + files_per_step]))
            if writer is None:
                schema = tables[0].schema
                writer = pq.ParquetWriter(tmp_file, schema)
            tables = [t.cast(schema) for t in tables if t.num_rows > 0]
            if tables:
                # one large row group per step instead of one per small chunk
                table = pa.concat_tables(tables)
                writer.write_table(table, row_group_size=table.num_rows)
                n_rows += table.num_rows
    if writer is not None:
        writer.close()
        os.replace(tmp_file, out_file)
    logger.info(f"Merged {len(files)} files ({n_rows} events) into {out_file}")


def adler32(path, block_size=16 * 1024 * 1024):
    value = 1
    with open(path, "rb") as f:
        while block := f.read(block_size):
            value = zlib.adler32(block, value)
    return value


def remove_empty_parents(path, stop_dir, include_stop=False):
    """Remove the parent directories of path left empty, up to stop_dir
    (removed too only if include_stop)."""
    parent = os.path.dirname(os.path.abspath(path))
    stop_dir = os.path.abspath(stop_dir)
    while (
        parent.startswith(stop_dir)
        and (parent != stop_dir or include_stop)
        and os.path.isdir(parent)
        and not os.listdir(parent)
    ):
        os.rmdir(parent)
        if parent == stop_dir:
            break
        parent = os.path.dirname(parent)


def copy_and_remove(src_file, dst_file, stop_dir):
    """Copy src_file to dst_file, check the copy and remove src_file and its
    parent directories left empty (up to stop_dir excluded).

    If the copy is not correct, dst_file is removed, src_file is kept and an
    error is raised."""
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    # files on dCache cannot be overwritten, they have to be removed first
    if os.path.exists(dst_file):
        os.remove(dst_file)
    logger.info(f"Copying {src_file} to {dst_file}")
    shutil.copyfile(src_file, dst_file)

    errors = []
    src_size, dst_size = os.path.getsize(src_file), os.path.getsize(dst_file)
    if src_size != dst_size:
        errors.append(f"size {dst_size} != {src_size}")
    else:
        src_sum, dst_sum = adler32(src_file), adler32(dst_file)
        if src_sum != dst_sum:
            errors.append(f"adler32 {dst_sum:08x} != {src_sum:08x}")
        src_rows = pq.ParquetFile(src_file).metadata.num_rows
        dst_rows = pq.ParquetFile(dst_file).metadata.num_rows
        if src_rows != dst_rows:
            errors.append(f"events {dst_rows} != {src_rows}")
    if errors:
        os.remove(dst_file)
        raise RuntimeError(
            f"Copy of {src_file} to {dst_file} failed ({', '.join(errors)}): "
            f"the copy was removed and the original file kept"
        )
    logger.info(f"Copy checked (size, adler32, events), removing {src_file}")

    os.remove(src_file)
    remove_empty_parents(src_file, stop_dir)


def read_config(config_file):
    with open(config_file) as f:
        return json.load(f)


def parquet_dir_from_config(config):
    """Original (unmerged) parquet directory stored in the config.json."""
    options = config["workflow"]["workflow_options"]
    for key in CONFIG_KEYS:
        # already updated by a previous run: use the original directory
        value = options.get(key + UNMERGED_SUFFIX) or options.get(key)
        if value:
            return root_to_local(value)
    raise KeyError(
        f"No parquet directory ({', '.join(CONFIG_KEYS)}) in the config.json: "
        "were the columns saved with save_chunk?"
    )


def update_config(config_file, merged_dir):
    """Overwrite the parquet directory in config_file with merged_dir, keeping
    the original value in <key>_unmerged."""
    config = read_config(config_file)
    options = config["workflow"]["workflow_options"]
    for key in CONFIG_KEYS:
        original = options.get(key + UNMERGED_SUFFIX) or options.get(key)
        if not original:
            continue
        options[key + UNMERGED_SUFFIX] = original
        options[key] = local_like(original, merged_dir)
        logger.info(f"config.json: {key} = {options[key]} (was {original})")
    tmp_file = config_file + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(config, f, indent=2)
    if os.path.exists(config_file):
        # files on dCache cannot be overwritten, they have to be removed first
        os.remove(config_file)
    os.replace(tmp_file, config_file)
    logger.info(f"Updated {config_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="PocketCoffea output directory (with config.json, the parquet "
        "location is read from it) or the parquet files directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory of the merged files (default: "
        f"'{DEFAULT_OUTPUT_NAME}' next to the original parquet directory)",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Merge each file in this directory, then copy it to the output "
        "directory, check the copy (size, adler32, events) and remove it from "
        "here. Needed when the output is on /pnfs (default in that case: "
        f"<input>/{DEFAULT_OUTPUT_NAME}_tmp if --input is the PocketCoffea "
        "output directory)",
    )
    parser.add_argument(
        "-j", "--threads", type=int, default=32, help="Number of reading threads"
    )
    parser.add_argument(
        "--files-per-step",
        type=int,
        default=256,
        help="Number of files kept in memory at the same time",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing merged files"
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Overwrite the parquet directory (dump_columns_as_arrays_per_chunk "
        "and save_chunk) in the config.json of --input with the output "
        "directory (the original values are kept as <key>_unmerged)",
    )
    args = parser.parse_args()

    config_file = os.path.join(args.input, "config.json")
    has_config = os.path.isfile(config_file)
    if has_config:
        in_path = parquet_dir_from_config(read_config(config_file))
    elif args.update_config:
        parser.error(f"--update-config needs a config.json in {args.input}")
    else:
        in_path = args.input
    in_path = os.path.abspath(in_path)
    if not os.path.isdir(in_path):
        raise FileNotFoundError(f"Parquet directory not found: {in_path}")

    output = os.path.abspath(
        args.output
        or os.path.join(os.path.dirname(in_path.rstrip("/")), DEFAULT_OUTPUT_NAME)
    )
    if output == in_path:
        parser.error("The output directory cannot be the input parquet directory")

    tmp_dir = args.tmp_dir
    if tmp_dir is None and output.startswith("/pnfs"):
        if not has_config:
            parser.error(
                "The output is on /pnfs, where the files cannot be written "
                "directly: give a local --tmp-dir"
            )
        tmp_dir = os.path.join(args.input, DEFAULT_OUTPUT_NAME + "_tmp")
    if tmp_dir is not None:
        tmp_dir = os.path.abspath(tmp_dir)

    logger.info(f"Merging the parquet files in {in_path} into {output}")
    if tmp_dir is not None:
        logger.info(f"Using the temporary directory {tmp_dir}")

    for leaf in find_leaf_dirs(in_path):
        rel_file = os.path.join(os.path.relpath(leaf, in_path), "merged.parquet")
        final_file = os.path.join(output, rel_file)
        if os.path.exists(final_file) and not args.overwrite:
            logger.info(f"Skipping existing {final_file}")
            continue
        if tmp_dir is None:
            merge_dir(leaf, final_file, args.threads, args.files_per_step)
            continue
        tmp_file = os.path.join(tmp_dir, rel_file)
        # a merged file left by a previous run whose copy failed is reused
        if not os.path.exists(tmp_file) or args.overwrite:
            merge_dir(leaf, tmp_file, args.threads, args.files_per_step)
        else:
            logger.info(f"Reusing existing {tmp_file}")
        # done after each merge, to not keep all the files in the tmp dir
        copy_and_remove(tmp_file, final_file, tmp_dir)

    if tmp_dir is not None and os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
        os.rmdir(tmp_dir)

    if args.update_config:
        update_config(config_file, output)

    logger.info(f"Merged parquet files in {output}")


if __name__ == "__main__":
    main()
