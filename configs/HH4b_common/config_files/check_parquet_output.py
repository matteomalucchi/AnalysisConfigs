#!/usr/bin/env python3
"""
Check, before a PocketCoffea run, that the parquet output directory is free.

If the config dumps the columns as parquet chunks
(`dump_columns_as_arrays_per_chunk` in the workflow options, set from
`save_chunk`), the chunks are written to
`<dump dir>/<dataset>/[<subsample>/]<category>/<variation>/`. If such a
directory already holds parquet files (e.g. from a previous run with a stale
`save_chunk` path), the new files are mixed with the old ones (and the ones
with the same chunk name overwrite them), so this script exits with an error.

Run automatically by run_pocket_coffea.sh (skip it there with
--skip-parquet-check), or by hand:

    python check_parquet_output.py --cfg <config.py> \
        [--filter-datasets a,b] [--filter-samples a,b] [--filter-years a,b]
"""
import argparse
import os
import sys

from pocket_coffea.utils.utils import load_config


def local_path(path):
    """Strip the xrootd redirector: root://host:port//pnfs/... -> /pnfs/..."""
    if "://" in path:
        path = "/" + path.split("://", 1)[1].split("/", 1)[1].lstrip("/")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--cfg", required=True, help="PocketCoffea config file")
    p.add_argument("--filter-datasets", default=None, help="comma separated list")
    p.add_argument("--filter-samples", default=None, help="comma separated list")
    p.add_argument("--filter-years", default=None, help="comma separated list")
    args = p.parse_args()

    cfg = load_config(args.cfg, do_load=True, save_config=False)
    dump_dir = (cfg.workflow_options or {}).get("dump_columns_as_arrays_per_chunk")
    if not dump_dir:
        print("Parquet check: the columns are not dumped as parquet chunks, nothing to check.")
        return 0

    dump_local = local_path(dump_dir)
    print(f"Parquet check: the columns are dumped as parquet chunks in {dump_dir}")
    if not os.path.isdir(dump_local):
        # e.g. /eos not mounted on this node: the directory can not be checked
        top_dir = "/" + dump_local.strip("/").split("/")[0]
        if not os.path.isdir(top_dir):
            print(f"WARNING: {top_dir} is not mounted on this node, the parquet check is skipped.")
        else:
            print(f"Parquet check: {dump_local} does not exist yet: OK.")
        return 0

    # same dataset filters as `pocket-coffea run`
    filesets = cfg.filesets
    if args.filter_years:
        years = args.filter_years.split(",")
        filesets = {d: f for d, f in filesets.items() if f["metadata"]["year"] in years}
    if args.filter_samples:
        samples = args.filter_samples.split(",")
        filesets = {d: f for d, f in filesets.items() if f["metadata"]["sample"] in samples}
    if args.filter_datasets:
        datasets = args.filter_datasets.split(",")
        filesets = {d: f for d, f in filesets.items() if d in datasets}

    categories = set(cfg.categories.keys())

    # {dataset/[subsample/]category/variation: number of parquet files}
    occupied = {}
    for dataset in sorted(filesets):
        dataset_dir = os.path.join(dump_local, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for root, _, files in os.walk(dataset_dir):
            n_parquet = sum(f.endswith(".parquet") for f in files)
            if n_parquet == 0:
                continue
            rel = os.path.relpath(root, dump_local).split(os.sep)
            # rel = [dataset, (subsample,) category, variation]
            if len(rel) >= 3 and rel[-2] in categories:
                occupied["/".join(rel)] = n_parquet

    if not occupied:
        print(f"Parquet check: no parquet files for the {len(filesets)} datasets to run: OK.")
        return 0

    bar = "!" * 100
    print(f"\n{bar}")
    print("ERROR: the parquet output directory already holds parquet files for the datasets,")
    print("regions and variations of this run. The new chunks would be mixed with the old ones")
    print("(and the chunks with the same name would overwrite them).")
    print(f"Parquet directory (`save_chunk` / `dump_columns_as_arrays_per_chunk`):\n    {dump_dir}")
    print("Directories already holding parquet files (dataset/[subsample/]region/variation):")
    for d, n in sorted(occupied.items()):
        print(f"    {d}: {n} files")
    print("Change `save_chunk` in the config, or remove/move the old files.")
    print("Use --skip-parquet-check in run_pocket_coffea to run anyway.")
    print(f"{bar}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
