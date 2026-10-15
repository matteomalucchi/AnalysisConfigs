import numpy as np
import awkward as ak
from collections import defaultdict

from utils_configs.spanet_evaluation_functions import (
    define_spanet_pairing_inputs,
)


def get_input_name(collection, input_name):
    for name in input_name:
        # in case of "events", the last s has to be removed to map to "Event_data"
        if collection == "events":
            collection = "event"
        if collection.lower() in name.lower():
            data_name = f"{'_'.join(name.split('_')[:-1])}_data"
            mask_name = f"{'_'.join(name.split('_')[:-1])}_mask"
            return data_name, mask_name
    raise ValueError(f"No {collection} found in {input_name}")


def extract_inputs_global(
    input_name,
    output_name,
    events,
    variables,
    pad_value,
    pad_value_spanet,
    run2,
    mask_type="ones",
):
    """
    Extract inputs for SPANet global inputs.

    The variables defined in the workflow are padded with `pad_value`, so
    those entries are replaced with `pad_value_spanet` to have all the SPANet
    inputs padded with the same value.

    `mask_type` sets the mask of each global input:
    - "ones": the mask is always True
    - "padded": an input is masked away (False) when all its features are
      equal to `pad_value_spanet`, i.e. when the object is missing
    """
    if mask_type not in ["ones", "padded"]:
        raise ValueError(
            f"Unknown SPANet global mask type {mask_type}, use 'ones' or 'padded'"
        )
    variables_dict = {}
    # whether each feature is padded, used to build the mask
    is_padded_dict = defaultdict(list)
    for var_name, attributes in variables.items():
        collection = attributes[0]
        feature = attributes[1]
        if len(attributes) > 2:
            scale = attributes[2]
        else:
            scale = None
        # "Collection:N" selects the N-th object of the collection, which is
        # fed to the global input of the collection
        data_name, mask_name = get_input_name(collection.split(":")[0], input_name)
        if data_name not in variables_dict.keys():
            variables_dict[data_name] = []

        if collection == "events":
            ak_array = getattr(events, feature)
        elif ":" in collection:
            ak_array = getattr(getattr(events, collection.split(":")[0]), feature)
            pos = int(collection.split(":")[1])
            ak_array = ak.fill_none(
                ak.pad_none(ak_array, pos + 1, clip=True), pad_value_spanet
            )[:, pos]
        else:
            ak_array = ak.fill_none(
                getattr(getattr(events, collection), feature), pad_value_spanet
            )
        # replace the workflow padding with the SPANet one
        ak_array = ak.fill_none(ak_array, pad_value_spanet)
        ak_array = ak.where(ak_array == pad_value, pad_value_spanet, ak_array)
        # checked before the log, which would change the padded value
        is_padded = ak.to_numpy(ak_array == pad_value_spanet, allow_missing=True)
        if is_padded.ndim == 1:
            is_padded = is_padded[:, None]
        is_padded_dict[mask_name].append(is_padded)
        if scale and "log" in scale:
            # apply the log to the padded value
            arr = np.array(
                np.log(ak.to_numpy(ak_array, allow_missing=True) + 1),
                dtype=np.float32,
            )

            if arr.ndim == 1:
                arr = arr[:, None]  # <-- THIS is the missing axis

            variables_dict[data_name].append(arr)
        else:
            arr = np.array(
                ak.to_numpy(ak_array, allow_missing=True),
                dtype=np.float32,
            )

            if arr.ndim == 1:
                arr = arr[:, None]  # <-- THIS is the missing axis

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
    if mask_type == "padded":
        for mask_name, is_padded in is_padded_dict.items():
            variables_dict[mask_name] = ~np.all(np.stack(is_padded, axis=-1), axis=-1)
    return variables_dict


def extract_inputs(input_name, output_name, events, variables, pad_value, run2):
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
            ak_array = getattr(events, feature)
        elif ":" in collection:
            ak_array = getattr(getattr(events, collection.split(":")[0]), feature)
            pos = int(collection.split(":")[1])
            ak_array = ak.fill_none(
                ak.pad_none(ak_array, pos + 1, clip=True), pad_value
            )[:, pos]
        else:
            ak_array = ak.fill_none(
                getattr(getattr(events, collection), feature), pad_value
            )
        if scale and "log" in scale:
            # apply the log to the padded value
            variables_array.append(
                np.array(
                    np.log(
                        ak.to_numpy(
                            ak_array,
                            allow_missing=True,
                        )
                        + 1
                    ),
                    dtype=np.float32,
                )
            )
        else:
            try:
                variables_array.append(
                    np.array(
                        ak.to_numpy(
                            ak_array,
                            allow_missing=True,
                        ),
                        dtype=np.float32,
                    )
                )
            except:
                raise ValueError(f"Issue with {collection}{feature}, {ak_array}")

    return np.stack(variables_array, axis=-1)


def get_dnn_prediction(
    session, input_name, output_name, events, variables, pad_value, run2=False
):
    inputs = extract_inputs(
        input_name, output_name, events, variables, pad_value, run2
    )

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


def get_onnx_prediction(
    session,
    input_name,
    output_name,
    events,
    variables,
    pad_value,
    pad_value_spanet,
    max_num_jets_spanet,
    run2=False,
    spanet_global_mask="ones",
):
    if "sequential" in variables:
        # SPANet
        assert "global" in variables
        input_name_forpop = input_name.copy()
        inputs_complete = {}
        collection_feature_dict = get_collections(variables["sequential"])
        for collection, features in collection_feature_dict.items():
            sequential_inputs = define_spanet_pairing_inputs(
                events, max_num_jets_spanet, collection, features, pad_value_spanet
            )  
            mask = np.array(
                ak.to_numpy(
                    ak.fill_none(
                        ak.pad_none(
                            ak.ones_like(events[collection].pt),
                            max_num_jets_spanet,
                            clip=True,
                        ),
                        value=0,
                    ),
                    allow_missing=True,
                ),
                dtype=np.bool_,
            )
            # Added sequential part to inputs
            inputs_complete |= {
                input_name_forpop.pop(0): sequential_inputs,
                input_name_forpop.pop(0): mask,
            }  # Take always the first element from input_name.
        global_inputs_total = extract_inputs_global(
            input_name_forpop,
            output_name,
            events,
            variables["global"],
            pad_value,
            pad_value_spanet,
            run2,
            mask_type=spanet_global_mask,
        )  # use remaining input_name, which is reduced by the pops from before
        inputs_complete |= global_inputs_total
        spanet_output = session.run(output_name, inputs_complete)

        order = ["h1", "h2", "vbf"]

        # separate the outputs
        idx_assignment_prob = np.array(
            [
                i
                for key in order
                for i, name in enumerate(output_name)
                if "assignment_probability" in name and key in name
            ]
        )
        assignment_prob = [spanet_output[i] for i in idx_assignment_prob]
        idx_detection_prob = np.array(
            [
                i
                for key in order
                for i, name in enumerate(output_name)
                if "detection_probability" in name and key in name
            ]
        )
        detection_prob = [spanet_output[i] for i in idx_detection_prob]

        idx_class_prob = np.where(
            [
                "assignment_probability" not in x
                and "detection_probability" not in x
                and ("class" in x or "signal" in x)
                for x in output_name
            ]
        )[0]
        class_prob = [spanet_output[i] for i in idx_class_prob]

        idx_regr_prob = np.where(
            [
                "assignment_probability" not in x
                and "detection_probability" not in x
                and "regr" in x
                for x in output_name
            ]
        )[0]
        regr_value = [spanet_output[i] for i in idx_regr_prob]

        spanet_separated_output = {
            "assignment_prob": assignment_prob,
            "detection_prob": detection_prob,
            "class_prob": class_prob,
            "regr_value": regr_value,
        }

        return (
            spanet_separated_output,
            "spanet",
        )
    else:
        # DNN
        return (
            get_dnn_prediction(
                session, input_name, output_name, events, variables, pad_value, run2
            )[0],
            "dnn",
        )
