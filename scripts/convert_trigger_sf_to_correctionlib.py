#!/usr/bin/env python
'''Convert the ROOT files with the trigger efficiency curves to a correctionlib file.

The trigger efficiencies are measured filter-by-filter: for each filter of the
trigger path (plus the logical OR of the L1 seeds) the ROOT files contain

    <Data/Simulation>__Efficiency_<filter>            the efficiency curve
    <Data/Simulation>__ConfidenceIntervals_<filter>   the 68% CI of the fit
    <Data/Simulation>__Efficiency_<filter>_FitFunction   (not used)
    <Data/Simulation>__Efficiency_<filter>_FitResult     (not used)

The total efficiency is the product of the per-filter efficiencies, therefore the
total scale factor is the product of the per-filter data/MC efficiency ratios:

    SF        = prod_i  eff_data_i             / eff_mc_i
    SF(up)    = prod_i (eff_data_i + err_i)    / (eff_mc_i + err_i)
    SF(down)  = prod_i (eff_data_i - err_i)    / (eff_mc_i - err_i)

where err_i is the error taken from the confidence intervals of the fit.

This script reads the curves with uproot, converts them to binned histograms and
writes one correctionlib correction per filter, with the `systematic`
(`nominal`, `up`, `down`) and the value of the observable as inputs.
The resulting file is applied by the `sf_trigger` weight of PocketCoffea
(see `pocket_coffea.lib.trigger_sf`).

Examples
--------

Inspect the content of the ROOT files:

    python scripts/convert_trigger_sf_to_correctionlib.py -i /path/to/trgSFs_2022_to_2025 --inspect

Build the correctionlib file for the 2022_postEE trigger:

    python scripts/convert_trigger_sf_to_correctionlib.py \
        -i /path/to/trgSFs_2022_to_2025 \
        -y 2022_postEE -o trigger_sf_2022_postEE.json.gz \
        --dump-params trigger_sf_2022_postEE.yaml

Check that the conversion machinery works without any input file:

    python scripts/convert_trigger_sf_to_correctionlib.py --selftest
'''

import argparse
import glob
import gzip
import os
import re
import sys

import numpy as np

try:
    import uproot
except ImportError:  # the self-test does not need uproot
    uproot = None

import correctionlib.schemav2 as cs

# Default names of the objects in the ROOT files
DATA_PREFIX = "Data"
MC_PREFIX = "Simulation"
EFFICIENCY_KEY = "Efficiency"
CONFIDENCE_INTERVALS_KEY = "ConfidenceIntervals"
# Label of the efficiency of the logical OR of all the L1 seeds
L1_LABEL = "L1All"

# Observable used to evaluate the efficiency of each filter, following the
# prescription of the trigger efficiency measurement:
# - the L1 seeds efficiency is evaluated with the Calo-HT
# - the `N`-jet filters with the pt of the N-th leading in pt jet
# - the b-tagging filters with the atanh of the average b-tagging score of the
#   two leading in b-tagging score jets
# - the HT filters with the HT of all the jets
# The rules are applied in order, the first matching one is used.
FILTER_VARIABLE_RULES = [
    (re.compile(r"^L1"), lambda match: {"name": "calojet_ht", "collection": "Jet"}),
    (
        re.compile(r"BTag|BTagMean|PNet2BTag|ParticleNet2BTag", re.IGNORECASE),
        lambda match: {
            "name": "atanh_btag_mean",
            "collection": "JetGood",
            "field": "btagPNetB",
            "n": 2,
        },
    ),
    (
        re.compile(r"HT\d+", re.IGNORECASE),
        lambda match: {"name": "alljet_ht", "collection": "Jet"},
    ),
    (
        re.compile(r"^(?P<index>\d+)\D+Pt\d+$"),
        lambda match: {
            "name": "jet_pt",
            "collection": "JetGood",
            "index": int(match.group("index")),
        },
    ),
]


def get_filter_variable(filter_name):
    '''Observable used to evaluate the efficiency of the filter `filter_name`'''
    for regex, builder in FILTER_VARIABLE_RULES:
        match = regex.search(filter_name)
        if match:
            return builder(match)
    raise ValueError(
        f"Impossible to guess the observable to be used for the filter "
        f"`{filter_name}`. Please add a rule to FILTER_VARIABLE_RULES."
    )


#############################################################################
# Reading of the ROOT objects


def input_files(inputs):
    '''Expand the input arguments (files, directories or globs) to a list of files'''
    files = []
    for path in inputs:
        if os.path.isdir(path):
            files += sorted(glob.glob(os.path.join(path, "**", "*.root"), recursive=True))
        else:
            files += sorted(glob.glob(path)) or [path]
    if len(files) == 0:
        raise Exception(f"No ROOT file found in {inputs}")
    return files


def index_objects(files):
    '''Index the objects of all the input files: {object name: (file, classname)}'''
    index = {}
    for filename in files:
        with uproot.open(filename) as file:
            for key, classname in file.classnames().items():
                # remove the cycle number from the key
                index[key.split(";")[0]] = (filename, classname)
    return index


# Objects stored for completeness, not used to evaluate the efficiencies
AUXILIARY_SUFFIXES = ("_FitFunction", "_FitResult")


def find_object(index, name):
    '''Find an object in the index.

    The name is first looked for as it is, then as a prefix (the objects of the L1
    efficiency have the era appended to their name, e.g. `Efficiency_L1All_2022F`).
    The fit function and the fit result are never returned.
    '''
    if name in index:
        return name
    candidates = sorted(
        key
        for key in index
        if key.startswith(name) and not key.endswith(AUXILIARY_SUFFIXES)
    )
    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        raise Exception(
            f"Multiple objects matching `{name}` found in the input files: "
            f"{candidates}. Please select the era with the `--era` argument."
        )
    return candidates[0]


def read_object(index, name):
    '''Read the object `name` from the file where it has been indexed'''
    filename, _ = index[name]
    with uproot.open(filename) as file:
        return file[name]


def points_from_object(obj):
    '''Extract the points of a TGraph-like or TH1-like object.

    :returns: dictionary with the `x` values, the low/high edges of the points
              (`xlow`, `xhigh`), the `y` values and their errors (`ylow`, `yhigh`)
    '''
    classname = getattr(obj, "classname", type(obj).__name__)

    if classname.startswith("TGraph"):
        x = np.asarray(obj.member("fX"), dtype=np.float64)
        y = np.asarray(obj.member("fY"), dtype=np.float64)
        if classname == "TGraphAsymmErrors":
            exlow = np.asarray(obj.member("fEXlow"), dtype=np.float64)
            exhigh = np.asarray(obj.member("fEXhigh"), dtype=np.float64)
            eylow = np.asarray(obj.member("fEYlow"), dtype=np.float64)
            eyhigh = np.asarray(obj.member("fEYhigh"), dtype=np.float64)
        elif classname in ("TGraphErrors", "TGraphBentErrors"):
            exlow = exhigh = np.asarray(obj.member("fEX"), dtype=np.float64)
            eylow = eyhigh = np.asarray(obj.member("fEY"), dtype=np.float64)
        else:
            exlow = exhigh = np.zeros_like(x)
            eylow = eyhigh = np.zeros_like(y)
        # the points of a graph are not necessarily sorted by x
        order = np.argsort(x)
        return {
            "x": x[order],
            "xlow": (x - exlow)[order],
            "xhigh": (x + exhigh)[order],
            "y": y[order],
            "ylow": eylow[order],
            "yhigh": eyhigh[order],
        }

    if classname.startswith("TH1") or classname.startswith("TProfile"):
        values, edges = obj.to_numpy()
        errors = np.asarray(obj.errors(), dtype=np.float64)
        edges = np.asarray(edges, dtype=np.float64)
        return {
            "x": 0.5 * (edges[:-1] + edges[1:]),
            "xlow": edges[:-1],
            "xhigh": edges[1:],
            "y": np.asarray(values, dtype=np.float64),
            "ylow": errors,
            "yhigh": errors,
        }

    raise ValueError(
        f"Objects of type `{classname}` are not supported. Supported types: "
        f"TGraph, TGraphErrors, TGraphAsymmErrors, TH1, TProfile."
    )


def edges_from_points(points):
    '''Bin edges of the histogram built from the points of a curve.

    The edges are taken from the x errors of the points if they are defined
    (the graphs produced from a TEfficiency store the bin width), otherwise the
    mid-points between consecutive points are used.
    '''
    x, xlow, xhigh = points["x"], points["xlow"], points["xhigh"]
    if len(x) < 2:
        raise ValueError("At least 2 points are needed to build a binned correction")

    if np.all(xhigh[:-1] > xlow[:-1]) and np.allclose(xhigh[:-1], xlow[1:]):
        return np.concatenate([xlow, [xhigh[-1]]])

    middle = 0.5 * (x[:-1] + x[1:])
    return np.concatenate(
        [[x[0] - (middle[0] - x[0])], middle, [x[-1] + (x[-1] - middle[-1])]]
    )


def interpolate(x, points):
    '''Interpolate the y values and the errors of a curve at the positions `x`'''
    order = np.argsort(points["x"])
    xp = points["x"][order]
    return (
        np.interp(x, xp, points["y"][order]),
        np.interp(x, xp, 0.5 * (points["ylow"] + points["yhigh"])[order]),
    )


def efficiency_and_error(eff_points, ci_points, central="efficiency"):
    '''Build the binned efficiency and its error.

    :param eff_points: points of the efficiency curve
    :param ci_points: points of the confidence intervals of the fit (can be None)
    :param central: `efficiency` to take the central value from the efficiency
                    curve, `fit` to take it from the fitted function stored in the
                    confidence intervals object
    :returns: tuple (edges, efficiency, error)
    '''
    if central == "fit":
        if ci_points is None:
            raise Exception(
                "The confidence intervals are needed to take the central value from the fit"
            )
        edges = edges_from_points(ci_points)
        x = 0.5 * (edges[:-1] + edges[1:])
        value, error = interpolate(x, ci_points)
        return edges, value, error

    edges = edges_from_points(eff_points)
    x = 0.5 * (edges[:-1] + edges[1:])
    value, _ = interpolate(x, eff_points)
    if ci_points is None:
        print(
            f"WARNING: no confidence intervals found, the error is set to 0",
            file=sys.stderr,
        )
        return edges, value, np.zeros_like(value)
    _, error = interpolate(x, ci_points)
    return edges, value, error


#############################################################################
# Building of the correctionlib file


def scale_factors(eff_data, err_data, eff_mc, err_mc, epsilon=1e-6):
    '''Data/MC efficiency ratio and its up/down variations.

    The efficiencies are clipped in [epsilon, 1] to protect the ratio: below the
    turn-on the efficiency is 0 both in data and in MC and the ratio is not
    defined. The scale factor is set to 1 in the bins where either the data or
    the MC efficiency is 0.
    '''
    undefined = (eff_data <= 0) | (eff_mc <= 0)

    def ratio(numerator, denominator):
        out = np.clip(numerator, epsilon, 1.0) / np.clip(denominator, epsilon, 1.0)
        return np.where(undefined, 1.0, out)

    return {
        "nominal": ratio(eff_data, eff_mc),
        "up": ratio(eff_data + err_data, eff_mc + err_mc),
        "down": ratio(eff_data - err_data, eff_mc - err_mc),
    }


def build_correction(name, variable_name, edges, content_by_systematic, description):
    '''Build a correctionlib Correction with the `systematic` and the observable as inputs'''
    return cs.Correction(
        name=name,
        description=description,
        version=1,
        inputs=[
            cs.Variable(
                name="systematic",
                type="string",
                description="nominal, up or down variation",
            ),
            cs.Variable(
                name=variable_name,
                type="real",
                description="observable used to evaluate the efficiency of the filter",
            ),
        ],
        output=cs.Variable(name="weight", type="real"),
        data=cs.Category(
            nodetype="category",
            input="systematic",
            content=[
                cs.CategoryItem(
                    key=systematic,
                    value=cs.Binning(
                        nodetype="binning",
                        input=variable_name,
                        edges=list(edges),
                        content=list(content),
                        # the SF of the first (last) bin is applied to the events
                        # below (above) the range of the measurement
                        flow="clamp",
                    ),
                )
                for systematic, content in content_by_systematic.items()
            ],
        ),
    )


def build_correctionset(filter_curves, year, store_efficiencies=True):
    '''Build the CorrectionSet with one correction per filter.

    :param filter_curves: dictionary {filter: (variable, edges, eff_data, err_data,
                          eff_mc, err_mc)}
    :param year: data-taking period, used in the descriptions
    :param store_efficiencies: also store the data and MC efficiencies, which are
                               not used by the weight but are useful for checks
    '''
    corrections = []
    for filter_name, curves in filter_curves.items():
        variable, edges, eff_data, err_data, eff_mc, err_mc = curves
        variable_name = variable["name"]
        corrections.append(
            build_correction(
                f"sf_{filter_name}",
                variable_name,
                edges,
                scale_factors(eff_data, err_data, eff_mc, err_mc),
                f"Data/MC efficiency ratio of the trigger filter {filter_name} "
                f"({year}), as a function of {variable_name}",
            )
        )
        if store_efficiencies:
            for label, efficiency, error in (
                ("data", eff_data, err_data),
                ("mc", eff_mc, err_mc),
            ):
                corrections.append(
                    build_correction(
                        f"eff_{label}_{filter_name}",
                        variable_name,
                        edges,
                        {
                            "nominal": efficiency,
                            "up": efficiency + error,
                            "down": efficiency - error,
                        },
                        f"Efficiency of the trigger filter {filter_name} in {label} "
                        f"({year}), as a function of {variable_name}",
                    )
                )

    return cs.CorrectionSet(
        schema_version=2,
        description=f"Trigger scale factors and efficiencies per trigger filter ({year})",
        corrections=corrections,
    )


def save_correctionset(correction_set, output):
    '''Write the CorrectionSet to a json (or json.gz) file'''
    content = correction_set.model_dump_json(exclude_unset=True, indent=2)
    if output.endswith(".gz"):
        with gzip.open(output, "wt") as file:
            file.write(content)
    else:
        with open(output, "w") as file:
            file.write(content)
    print(f"Trigger scale factors saved in {output}")


def dump_parameters(filter_curves, year, correction_file, output):
    '''Dump the PocketCoffea parameters to apply the scale factors'''
    lines = [
        "# Parameters to apply the trigger scale factors with the `sf_trigger` weight.",
        "# Produced by scripts/convert_trigger_sf_to_correctionlib.py",
        "trigger_scale_factors:",
        f'  "{year}":',
        f"    file: {os.path.abspath(correction_file)}",
        "    corrections:",
    ]
    for filter_name, curves in filter_curves.items():
        variable = curves[0]
        lines.append(f"      - name: sf_{filter_name}")
        lines.append("        variable:")
        for key, value in variable.items():
            lines.append(f"          {key}: {value}")
    with open(output, "w") as file:
        file.write("\n".join(lines) + "\n")
    print(f"PocketCoffea parameters saved in {output}")


#############################################################################
# Command line interface


def get_filters(args):
    '''List of the filters to convert, in the order they are applied.

    The L1 efficiency is always the first one. The HLT filters are taken from the
    `--filters` argument or from the yaml file with the trigger object filters
    (the same file used for the trigger object matching), where each filter is
    defined by the string `type:bit:n_objects:threshold:name`.
    '''
    filters = [] if args.no_l1 else [args.l1_label + (f"_{args.era}" if args.era else "")]

    if args.filters:
        return filters + list(args.filters)

    import yaml

    with open(args.filters_file) as file:
        config = yaml.safe_load(file)
    # allow both {year: {trigger: [filters]}} and {key: {year: {trigger: [filters]}}}
    if args.year not in config:
        config = config[list(config.keys())[0]]
    if args.year not in config:
        raise Exception(f"The year {args.year} is not present in {args.filters_file}")
    triggers = config[args.year]

    for trigger, trigger_filters in triggers.items():
        if args.triggers and trigger not in args.triggers:
            continue
        for trigger_filter in trigger_filters:
            filters.append(trigger_filter.split(":")[-1])
    return filters


def convert(args):
    files = input_files(args.input)
    print(f"Reading {len(files)} ROOT file(s)")
    index = index_objects(files)

    if args.inspect:
        for name, (filename, classname) in sorted(index.items()):
            print(f"{classname:25s} {name:70s} {os.path.basename(filename)}")
        return

    filters = get_filters(args)
    print(f"Converting {len(filters)} filters: {filters}")

    filter_curves = {}
    for filter_name in filters:
        curves = {}
        for label, prefix in (("data", args.data_prefix), ("mc", args.mc_prefix)):
            efficiency_name = find_object(index, f"{prefix}__{EFFICIENCY_KEY}_{filter_name}")
            if efficiency_name is None:
                raise Exception(
                    f"The efficiency `{prefix}__{EFFICIENCY_KEY}_{filter_name}` is not "
                    f"present in the input files. Run with `--inspect` to list the "
                    f"content of the files."
                )
            ci_name = find_object(
                index, f"{prefix}__{CONFIDENCE_INTERVALS_KEY}_{filter_name}"
            )
            print(f"  {filter_name} ({label}): {efficiency_name}, {ci_name}")
            curves[label] = efficiency_and_error(
                points_from_object(read_object(index, efficiency_name)),
                points_from_object(read_object(index, ci_name)) if ci_name else None,
                central=args.central,
            )

        edges_data, eff_data, err_data = curves["data"]
        edges_mc, eff_mc, err_mc = curves["mc"]
        if not np.allclose(edges_data, edges_mc):
            # the MC efficiency is interpolated on the binning of the data one
            x = 0.5 * (edges_data[:-1] + edges_data[1:])
            eff_mc = np.interp(x, 0.5 * (edges_mc[:-1] + edges_mc[1:]), eff_mc)
            err_mc = np.interp(x, 0.5 * (edges_mc[:-1] + edges_mc[1:]), err_mc)

        # the label of the L1 efficiency contains the era: it is removed from the
        # name of the correction to have the same name for all the eras
        label = args.l1_label if filter_name.startswith(args.l1_label) else filter_name
        filter_curves[label] = (
            get_filter_variable(label),
            edges_data,
            eff_data,
            err_data,
            eff_mc,
            err_mc,
        )

    correction_set = build_correctionset(
        filter_curves, args.year, store_efficiencies=not args.no_efficiencies
    )
    save_correctionset(correction_set, args.output)
    if args.dump_params:
        dump_parameters(filter_curves, args.year, args.output, args.dump_params)


def selftest():
    '''Check the conversion machinery on a synthetic efficiency curve.

    No ROOT file is needed: the objects read with uproot are replaced by objects
    exposing the same interface (`classname` and `member`).
    '''
    import tempfile

    import correctionlib

    class FakeGraph:
        '''Minimal implementation of the uproot TGraphAsymmErrors interface'''

        classname = "TGraphAsymmErrors"

        def __init__(self, x, y, ey, width):
            self._members = {
                "fX": np.asarray(x),
                "fY": np.asarray(y),
                "fEXlow": np.full_like(np.asarray(x), width / 2),
                "fEXhigh": np.full_like(np.asarray(x), width / 2),
                "fEYlow": np.asarray(ey),
                "fEYhigh": np.asarray(ey),
            }

        def member(self, key):
            return self._members[key]

    # turn-on curve sampled in 10 GeV wide bins
    x = np.arange(25.0, 205.0, 10.0)
    eff_data = 1.0 / (1.0 + np.exp(-(x - 70.0) / 10.0))
    eff_mc = 1.0 / (1.0 + np.exp(-(x - 65.0) / 10.0))
    err = np.full_like(x, 0.01)

    curves = {}
    for label, efficiency in (("data", eff_data), ("mc", eff_mc)):
        curves[label] = efficiency_and_error(
            points_from_object(FakeGraph(x, efficiency, err, 10.0)),
            points_from_object(FakeGraph(x, efficiency, err, 10.0)),
            central="efficiency",
        )

    edges, eff_data_binned, err_data_binned = curves["data"]
    _, eff_mc_binned, err_mc_binned = curves["mc"]

    assert np.allclose(edges, np.arange(20.0, 210.0, 10.0)), edges
    assert np.allclose(eff_data_binned, eff_data)
    assert np.allclose(err_data_binned, err)

    filter_curves = {
        "1PFCentralJetTightIDPt70": (
            get_filter_variable("1PFCentralJetTightIDPt70"),
            edges,
            eff_data_binned,
            err_data_binned,
            eff_mc_binned,
            err_mc_binned,
        )
    }
    assert filter_curves["1PFCentralJetTightIDPt70"][0]["name"] == "jet_pt"
    assert filter_curves["1PFCentralJetTightIDPt70"][0]["index"] == 1

    with tempfile.TemporaryDirectory() as directory:
        output = os.path.join(directory, "trigger_sf_selftest.json")
        save_correctionset(build_correctionset(filter_curves, "selftest"), output)
        correction_set = correctionlib.CorrectionSet.from_file(output)

        evaluator = correction_set["sf_1PFCentralJetTightIDPt70"]
        expected = scale_factors(eff_data, err, eff_mc, err)
        for systematic in ("nominal", "up", "down"):
            assert np.allclose(evaluator.evaluate(systematic, x), expected[systematic])
        # the flow is clamped to the first and last bin
        assert np.isclose(evaluator.evaluate("nominal", 0.0), expected["nominal"][0])
        assert np.isclose(evaluator.evaluate("nominal", 1e4), expected["nominal"][-1])
        # the efficiencies are stored as well
        assert np.allclose(
            correction_set["eff_data_1PFCentralJetTightIDPt70"].evaluate("nominal", x),
            eff_data,
        )

    # the fit function and the fit result are never picked up when the name of the
    # object is resolved by prefix (the L1 objects have the era appended)
    index = {
        "Data__Efficiency_L1All_2022F": ("f.root", "TGraphAsymmErrors"),
        "Data__Efficiency_L1All_2022F_FitFunction": ("f.root", "TF1"),
        "Data__Efficiency_L1All_2022F_FitResult": ("f.root", "TFitResult"),
    }
    assert find_object(index, "Data__Efficiency_L1All") == "Data__Efficiency_L1All_2022F"
    assert find_object(index, "Data__Efficiency_L1All_2022F") == "Data__Efficiency_L1All_2022F"
    assert find_object(index, "Data__Efficiency_HT") is None

    # the points of a graph are sorted by x
    shuffled = FakeGraph(x[::-1], eff_data[::-1], err[::-1], 10.0)
    assert np.allclose(points_from_object(shuffled)["x"], x)
    assert np.allclose(points_from_object(shuffled)["y"], eff_data)

    # the observables of the filters of the documentation are correctly assigned
    for filter_name, expected_variable in (
        ("L1All", "calojet_ht"),
        ("4PixelOnlyPFCentralJetTightIDPt20", "jet_pt"),
        ("2PFCentralJetTightIDPt50", "jet_pt"),
        ("BTagCentralJetPt35PFParticleNet2BTagSum0p65", "atanh_btag_mean"),
        ("PFCentralJetPt30PNet2BTagMean0p55", "atanh_btag_mean"),
        ("PFHT280Jet30", "alljet_ht"),
        ("4PFCentralJetTightIDPt25", "jet_pt"),
    ):
        variable = get_filter_variable(filter_name)
        assert variable["name"] == expected_variable, (filter_name, variable)
    assert get_filter_variable("3PFCentralJetTightIDPt40")["index"] == 3

    print("Self-test passed")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", nargs="+", help="ROOT files, directories or globs with the efficiency curves")
    parser.add_argument("-y", "--year", help="Data-taking period, e.g. 2022_postEE")
    parser.add_argument("-o", "--output", default=None, help="Output correctionlib file (.json or .json.gz)")
    parser.add_argument("--era", default=None, help="Era appended to the name of the L1 efficiency objects")
    parser.add_argument("--filters-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "HH4b_common", "params", "trigger_object_filters.yaml"), help="yaml file with the trigger filters of each trigger")
    parser.add_argument("--triggers", nargs="+", default=None, help="Triggers to consider in the filters file (default: all)")
    parser.add_argument("--filters", nargs="+", default=None, help="Explicit list of HLT filters, overwrites the filters file")
    parser.add_argument("--no-l1", action="store_true", help="Do not convert the L1 efficiency")
    parser.add_argument("--l1-label", default=L1_LABEL, help=f"Label of the L1 efficiency objects (default: {L1_LABEL})")
    parser.add_argument("--data-prefix", default=DATA_PREFIX, help=f"Prefix of the data objects (default: {DATA_PREFIX})")
    parser.add_argument("--mc-prefix", default=MC_PREFIX, help=f"Prefix of the simulation objects (default: {MC_PREFIX})")
    parser.add_argument("--central", default="efficiency", choices=["efficiency", "fit"], help="Take the central value of the efficiency from the measured curve or from the fitted function")
    parser.add_argument("--no-efficiencies", action="store_true", help="Store only the scale factors, not the efficiencies")
    parser.add_argument("--dump-params", default=None, help="Dump the PocketCoffea parameters in this yaml file")
    parser.add_argument("--inspect", action="store_true", help="Print the content of the input files and exit")
    parser.add_argument("--selftest", action="store_true", help="Run the self-test of the conversion and exit")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists")
    return parser.parse_args()


def main():
    args = get_args()
    if args.selftest:
        selftest()
        return
    if uproot is None:
        raise ImportError("uproot is needed to read the ROOT files")
    if not args.input:
        raise Exception("The input files are needed: use `-i/--input`")
    if not args.inspect:
        if not args.year:
            raise Exception("The data-taking period is needed: use `-y/--year`")
        if not args.output:
            args.output = f"trigger_sf_{args.year}.json.gz"
        if os.path.exists(args.output) and not args.overwrite:
            raise Exception(f"The output file {args.output} already exists, use `--overwrite`")
    convert(args)


if __name__ == "__main__":
    main()
