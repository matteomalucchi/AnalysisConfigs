import numpy as np
import awkward as ak
from collections import defaultdict

from utils.spanet_evaluation_functions import define_spanet_pairing_inputs, get_pairing_information


def get_input_name(collection, input_name):
    for name in input_name:
        # in case of "events", the last s has to be removed to map to "Event_data"
        if collection == "events":
            collection = "event"
        if collection.lower() in name.lower():
            data_name = f"{name.split('_')[0]}_data"
            mask_name = f"{name.split('_')[0]}_mask"
            return data_name, mask_name
    raise ValueError(f"No {collection} found in {input_name}")


def extract_inputs_global(session, input_name, output_name, events, variables, pad_value, run2):
    """Extract inputs for SPANet global inputs."""
    variables_dict = {}
    for var_name, attributes in variables.items():
        collection = attributes[0]
        feature = attributes[1]
        if len(attributes) > 2:
            scale = attributes[2]
        else:
            scale = None
        data_name, mask_name = get_input_name(collection, input_name)
        if data_name not in variables_dict.keys():
            variables_dict[data_name] = []

        if collection == "events":
            try:
                ak_array = getattr(events, f"{feature}Run2" if run2 else feature)
            except AttributeError:
                ak_array = getattr(events, feature)
        elif ":" in collection:
            try:
                ak_array = getattr(
                    getattr(
                        events,
                        (
                            f"{collection.split(':')[0]}Run2"
                            if run2
                            else collection.split(":")[0]
                        ),
                    ),
                    feature,
                )
            except AttributeError:
                ak_array = getattr(getattr(events, collection.split(":")[0]), feature)
            pos = int(collection.split(":")[1])
            ak_array = ak.fill_none(ak.pad_none(ak_array, pos + 1, clip=True), pad_value)
        else:
            try:
                ak_array = ak.fill_none(
                    getattr(
                        getattr(events, f"{collection}Run2" if run2 else collection),
                        feature,
                    ),
                    pad_value,
                )
            except AttributeError:
                ak_array = ak.fill_none(
                    getattr(getattr(events, collection), feature), pad_value
                )
        if scale and "log" in scale:
            arr = np.array(
                np.log(
                    ak.to_numpy(ak_array, allow_missing=True),
                    dtype=np.float32,
                )
                + 1
            )

            if arr.ndim == 1:
                arr = arr[:, None]   # <-- THIS is the missing axis

            variables_dict[data_name].append(arr)
        else:
            arr = np.array(
                ak.to_numpy(ak_array, allow_missing=True),
                dtype=np.float32,
            )

            if arr.ndim == 1:
                arr = arr[:, None]   # <-- THIS is the missing axis

            variables_dict[data_name].append(arr)
        if mask_name not in variables_dict.keys():
            mask_ak = ak.ones_like(ak_array)
            mask_np = ak.to_numpy(mask_ak, allow_missing=True)
            if mask_np.ndim == 1:
                mask_np = mask_np[:, None]
            variables_dict[mask_name] = mask_np.astype(np.bool_)
    for key, value in variables_dict.items():
        if "data" in key:
            variables_dict[key] = np.stack(value, axis=-1)
    return variables_dict


def extract_inputs(session, input_name, output_name, events, variables, pad_value, run2):
    """Extract inputs for the DNN models."""
    variables_array = []
    for var_name, attributes in variables.items():
        collection = attributes[0]
        feature = attributes[1]
        if len(attributes) > 2:
            scale = attributes[2]
        else:
            scale = None

        if collection == "events":
            try:
                ak_array = getattr(events, f"{feature}Run2" if run2 else feature)
            except AttributeError:
                ak_array = getattr(events, feature)
        elif ":" in collection:
            try:
                ak_array = getattr(
                    getattr(
                        events,
                        (
                            f"{collection.split(':')[0]}Run2"
                            if run2
                            else collection.split(":")[0]
                        ),
                    ),
                    feature,
                )
            except AttributeError:
                ak_array = getattr(getattr(events, collection.split(":")[0]), feature)
            pos = int(collection.split(":")[1])
            ak_array = ak.fill_none(ak.pad_none(ak_array, pos + 1, clip=True), pad_value)[
                :, pos
            ]
        else:
            try:
                ak_array = ak.fill_none(
                    getattr(
                        getattr(events, f"{collection}Run2" if run2 else collection),
                        feature,
                    ),
                    pad_value,
                )
            except AttributeError:
                ak_array = ak.fill_none(
                    getattr(getattr(events, collection), feature), pad_value
                )
        if scale and "log" in scale:
            variables_array.append(
                np.array(
                    np.log(
                        ak.to_numpy(
                            ak_array,
                            allow_missing=True,
                        ),
                        dtype=np.float32,
                    )
                )
            )
        else:
            variables_array.append(
                np.array(
                    ak.to_numpy(
                        ak_array,
                        allow_missing=True,
                    ),
                    dtype=np.float32,
                )
            )

    return np.stack(variables_array, axis=-1)

def get_dnn_prediction(session, input_name, output_name, events, variables, pad_value, run2=False):
    inputs = extract_inputs(session, input_name, output_name, events, variables, pad_value, run2)

    inputs_complete = {input_name[0]: inputs}

    outputs = session.run(output_name, inputs_complete)
    return outputs

def get_collections(input_dict):
    coll_dict = defaultdict(list)
    for val_list in input_dict.values():
        if len(val_list) > 2 and "log" in val_list[2]:
            feature = f"{val_list[1]}:log"
        else:
            feature = val_list[1]
        coll_dict[val_list[0]].append(feature)
    return coll_dict

def get_dnn_prediction_spanet_or_dnn(session, input_name, output_name, events, variables, pad_value, max_num_jets, run2=False):
    if "sequential" in variables:
        assert "global" in variables
        input_name_forpop = input_name.copy()
        inputs_complete = {}
        collection_feature_dict = get_collections(variables["sequential"])
        for collection, features in collection_feature_dict.items():
            sequential_inputs = define_spanet_pairing_inputs(events, max_num_jets, collection, features)  # Currently hardcode jets to 4
            mask = np.array(
                ak.to_numpy(
                    ak.fill_none(
                        ak.pad_none(
                            ak.ones_like(events[collection].pt),
                            max_num_jets,
                            clip=True,
                        ),
                        value=0,
                    ),
                    allow_missing=True,
                ),
                dtype=np.bool_,
            )
            # Added sequential part to inputs
            inputs_complete |= {input_name_forpop.pop(0): sequential_inputs, input_name_forpop.pop(0): mask} # Take always the first element from input_name.
        global_inputs_total = extract_inputs_global(session, input_name_forpop, output_name, events, variables["global"], pad_value, run2)  # use remaining input_name, which is reduced by the pops from before
        inputs_complete |= global_inputs_total
        spanet_score = session.run(output_name, inputs_complete)
        return spanet_score, "spanet"  # Taking first column because I think that is the signal score
    else:
        return get_dnn_prediction(session, input_name, output_name, events, variables, pad_value, run2)[0], "dnn"
