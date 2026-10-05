"""Resolve the jet algorithms configured for a data-taking year.

The b-tagging and the pT-regression algorithms change from year to year (e.g.
ParticleNet up to 2023 and UParTAK4 from 2024 on). The workflows read them from
the parameters through the helpers below and then work on algorithm-independent
fields, so that no year-specific NanoAOD branch name is hardcoded in the
analysis code or in the saved columns.
"""

import awkward as ak
from pocket_coffea.lib.jets import get_btag_wp_threshold

# NanoAOD suffix of the per-algorithm discriminants -> algorithm-independent
# field name. `btagB`, `btagCvL` and `btagCvB` keep the names that PocketCoffea's
# `jet_selection` gives them when it is called with a `jet_tagger`.
BTAG_GENERIC_FIELDS = {
    "B": "btagB",
    "CvL": "btagCvL",
    "CvB": "btagCvB",
    "QvG": "btagQvG",
}

# pT-regression algorithms supported by PocketCoffea's `JetsCalibrator` 
# and the branch with the pT resolution estimated by the regression.
PT_REGRESSION_ALGORITHMS = {"PNet": "PNetRegPtRawRes", "UParT": "UParTAK4RegPtRawRes"}

# name of the collection holding the pT-regressed jets
PT_REGRESSED_COLLECTION = "JetPtRegressed"


def get_btag_algorithm(params, year):
    """Return the b-tag discriminant branch set as default for `year`."""
    return params["btagging"]["working_point"][year]["btagging_algorithm"]


def get_btag_working_points(params, year, tagger=None):
    """Return `{working point name: threshold}` for a b-tagging algorithm.

    `tagger` defaults to the algorithm set as default for `year`. Both the flat
    (`btagging_WP: {L: ...}`) and the per-algorithm
    (`btagging_WP: {<tagger>: {L: ...}}`) layouts of the parameters are
    supported, as in PocketCoffea's `get_btag_wp_threshold`.
    """
    btag_params = params["btagging"]["working_point"][year]
    if tagger is None:
        tagger = btag_params["btagging_algorithm"]

    working_points = btag_params["btagging_WP"]
    if tagger in working_points:
        working_points = working_points[tagger]

    return {
        wp: get_btag_wp_threshold(btag_params, wp, tagger) for wp in working_points
    }


def add_btag_generic_fields(jets, tagger):
    """Copy the discriminants of `tagger` onto algorithm-independent fields.

    All the discriminants of an algorithm share the prefix of its b-tag branch
    (e.g. `btagPNetB`, `btagPNetCvL`, ... for `btagPNetB`), so the b-tag branch
    alone is enough to find them. The ones present in `jets` are copied to the
    names in `BTAG_GENERIC_FIELDS`.
    """
    if not tagger.endswith("B"):
        raise ValueError(
            f"'{tagger}' does not look like a b-tag discriminant branch: the "
            "name of a b-tagging algorithm is expected to end with 'B'."
        )
    algorithm_prefix = tagger[: -len("B")]

    for suffix, generic_field in BTAG_GENERIC_FIELDS.items():
        branch = f"{algorithm_prefix}{suffix}"
        if branch in jets.fields:
            jets = ak.with_field(jets, jets[branch], generic_field)

    if BTAG_GENERIC_FIELDS["B"] not in jets.fields:
        raise ValueError(
            f"The b-tag discriminant '{tagger}' set as default for this year is "
            f"not available in the input jets. Available fields: {jets.fields}."
        )

    return jets


def get_pt_regression_algorithm(params, year, collection=PT_REGRESSED_COLLECTION):
    """Return the pT-regression algorithm calibrating `collection` in `year`.

    The algorithm is the one that PocketCoffea's `JetsCalibrator` derives from
    the name of the jet type (e.g. `AK4PFPuppiRegression` -> UParTAK4).
    """
    calibrated_collections = params["jets_calibration"]["collection"][year]
    jet_types = [
        jet_type
        for jet_type, collection_name in calibrated_collections.items()
        if collection_name == collection
    ]
    if not jet_types:
        raise ValueError(
            f"No jet type is calibrated into '{collection}' for year {year}. "
            "Make sure the pT regression is configured in jets_calibration."
        )

    for algorithm in PT_REGRESSION_ALGORITHMS.keys():
        if algorithm in jet_types[0]:
            return algorithm

    raise ValueError(
        f"The jet type '{jet_types[0]}' does not name any of the supported pT "
        f"regression algorithms {PT_REGRESSION_ALGORITHMS.keys()}."
    )


def get_pt_regression_resolution_branch(
    params, year, collection=PT_REGRESSED_COLLECTION
):
    """Return the branch with the pT resolution estimated by the regression."""
    algorithm = get_pt_regression_algorithm(params, year, collection)
    return PT_REGRESSION_ALGORITHMS[algorithm]
