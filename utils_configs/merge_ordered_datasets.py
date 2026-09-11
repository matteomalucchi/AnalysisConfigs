#!/usr/bin/env python3
"""Merge dataset-definition JSON files into one, ordered like a reference file.

Each input file is a dict of {dataset_name: {"metadata": ..., "files": ...}}.
The output is the union of all inputs, with keys ordered following their
first appearance in --order-ref; keys not found there are appended at the
end, in the order encountered in the inputs.
"""

import argparse
import json
import sys


def merge_ordered(input_paths, order_ref_path):
    merged = {}
    for path in input_paths:
        with open(path) as f:
            data = json.load(f)
        duplicates = set(data) & set(merged)
        if duplicates:
            sys.exit(f"Duplicate dataset key(s) found in {path}: {sorted(duplicates)}")
        merged.update(data)

    with open(order_ref_path) as f:
        order_ref = json.load(f)

    ordered = {k: merged.pop(k) for k in order_ref if k in merged}
    missing_from_ref = list(merged)
    if missing_from_ref:
        print(
            f"Warning: {len(missing_from_ref)} key(s) not found in {order_ref_path}, "
            f"appended at the end: {missing_from_ref}",
            file=sys.stderr,
        )
    ordered.update(merged)
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Dataset-definition JSON files to merge")
    parser.add_argument(
        "-r", "--order-ref", required=True,
        help="JSON file whose top-level key order is used to order the output",
    )
    parser.add_argument("-o", "--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    merged = merge_ordered(args.inputs, args.order_ref)

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=4)
        f.write("\n")

    print(f"Wrote {len(merged)} datasets to {args.output}")


if __name__ == "__main__":
    main()
