"""Generic DeltaR-based trigger object matching, self-contained in AnalysisConfigs.

Unlike PocketCoffea's per-filter `get_trigger_object_matching` (added on the
`claude/trigger-sf-root-files-2ph6y5` branch, which needs a NanoAOD-version-keyed
configuration of every filter bit of every trigger), this module only matches a
given offline collection against the trigger objects by DeltaR, with no notion of
filter bits, trigger names or NanoAOD version. It needs no PocketCoffea support
beyond the stable `Cut` class, so it works by editing this analysis configuration
alone. It is enough to implement the usual analysis-note recipe, e.g.:

    A trigger-object matching is performed using all selected jets in the event
    with a DeltaR(trigger obj., jet) < 0.5.

via `get_trigger_object_matching(collection="JetGood", dr_max=0.5)`.
"""

import awkward as ak
import numpy as np

from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.deltaR_matching import delta_phi


def delta_r_table(obj, other):
    '''DeltaR between all the pairs of objects of two jagged collections.

    :returns: array with dimension (n_events, n_obj, n_other)
    '''
    first, second = ak.unzip(ak.cartesian([obj, other], axis=1, nested=True))
    return np.sqrt(
        (first.eta - second.eta) ** 2 + delta_phi(first.phi, second.phi) ** 2
    )


def trigger_object_matching_mask(objects, trigger_objects, dr_max=0.5):
    '''Per-object mask, True where the object is matched to at least one trigger
    object within `dr_max`.

    :param objects: jagged collection of offline objects (e.g. `events.JetGood`)
    :param trigger_objects: jagged collection of trigger objects (e.g. `events.TrigObj`)
    :param dr_max: maximum deltaR between an object and a trigger object
    :returns: boolean array with the same jagged structure as `objects`
    '''
    dr = delta_r_table(objects, trigger_objects)
    return ak.any(dr < dr_max, axis=-1)


def get_trigger_object_matching(
    collection="JetGood",
    trigger_collection="TrigObj",
    dr_max=0.5,
    require="all",
    name=None,
):
    '''Build the `Cut` requiring the objects of `collection` to be matched, within
    `dr_max`, to a trigger object of `trigger_collection`.

    :param collection: offline collection whose objects must be matched (default `JetGood`).
    :param trigger_collection: collection of trigger objects (default the raw `TrigObj`).
    :param dr_max: maximum deltaR between an object of `collection` and a trigger object.
    :param require: `"all"` requires every object of `collection` to be matched (the
                    usual "matching performed using all selected jets" recipe); `"any"`
                    requires at least one object matched.
    :param name: name of the cut (default derived from `collection` and `dr_max`).

    N.B: an event with zero objects in `collection` passes the `"all"` requirement
    vacuously; apply this cut after a selection already requiring a minimum number
    of objects if that is not the intended behaviour.
    '''
    if require not in ("all", "any"):
        raise ValueError(f"`require` must be `all` or `any`, got `{require}`")
    if name is None:
        name = f"trigger_object_matching_{collection}_dr{dr_max}"

    def _trigger_object_matching_mask(events, params, **kwargs):
        objects = events[params["collection"]]
        trigger_objects = events[params["trigger_collection"]]
        matched = trigger_object_matching_mask(
            objects, trigger_objects, params["dr_max"]
        )
        if params["require"] == "all":
            return ak.all(matched, axis=1)
        return ak.any(matched, axis=1)

    return Cut(
        name=name,
        params={
            "collection": collection,
            "trigger_collection": trigger_collection,
            "dr_max": dr_max,
            "require": require,
        },
        function=_trigger_object_matching_mask,
    )
