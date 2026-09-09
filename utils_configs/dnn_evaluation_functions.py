import numpy as np
import awkward as ak
from collections import defaultdict

from utils_configs.inference_session_onnx import get_model_session
from utils_configs.reconstruct_resonances import reconstruct_vbf_jets_from_idx
from utils_configs.spanet_evaluation_functions import (
    clean_assignment_prob,
    define_spanet_pairing_inputs,
    get_best_pairings,
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


def extract_inputs_global(input_name, output_name, events, variables, pad_value_spanet, run2):
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
            ak_array = getattr(events, feature)
        elif ":" in collection:
            ak_array = getattr(getattr(events, collection.split(":")[0]), feature)
            pos = int(collection.split(":")[1])
            ak_array = ak.fill_none(
                ak.pad_none(ak_array, pos + 1, clip=True), pad_value_spanet
            )
        else:
            ak_array = ak.fill_none(
                getattr(getattr(events, collection), feature), pad_value_spanet
            )
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
            input_name_forpop, output_name, events, variables["global"], pad_value_spanet, run2
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


def get_pairing_collection(input_variables):
    """
    Name of the jet collection the pairing indices of a SPANet model refer to,
    i.e. the collection given as sequential input to the model.
    """
    return [x[0] for x in input_variables["sequential"].values()][0]


def eval_spanet(
    events,
    spanet,
    spanet_input_name,
    pad_value,
    pad_value_spanet,
    max_num_jets_higgs_pairing,
):
    """
    Run the SPANet pairing model and extract the best jet assignment of the
    Higgs candidates (and of the VBF pair, when the model predicts it as well).
    """
    model_session_spanet, input_name_spanet, output_name_spanet = get_model_session(
        spanet, "spanet"
    )

    spanet_output, _ = get_onnx_prediction(
        model_session_spanet,
        input_name_spanet,
        output_name_spanet,
        events,
        spanet_input_name,
        pad_value,
        pad_value_spanet,
        max_num_jets_higgs_pairing,
    )
    # Not needed anymore
    del model_session_spanet, input_name_spanet, output_name_spanet

    jet_coll_pairing = get_pairing_collection(spanet_input_name)

    # if an event has less than 6 jets, than remove the vbf prob matrix
    cleaned_assignment_prob = clean_assignment_prob(
        spanet_output["assignment_prob"], events[jet_coll_pairing]
    )

    (
        pairing_predictions,
        best_pairing_probability,
        second_best_pairing_probability,
        worst_pairing_probability,
    ) = get_best_pairings(cleaned_assignment_prob)

    return (
        pairing_predictions,
        jet_coll_pairing,
        spanet_output,
        best_pairing_probability,
        second_best_pairing_probability,
        worst_pairing_probability,
    )


def eval_vbf_discriminator(
    events,
    vbf_discriminator,
    vbf_discriminator_input_variables,
    pad_value,
    pad_value_spanet,
    max_num_jets_vbf_discriminator,
):
    """
    Run the standalone ggF/VBF discriminator model and return its raw output.

    The same model can provide both the ggF/VBF score and the VBF pairing, so
    the caller is expected to run it once and reuse the output.
    """
    (
        model_session_vbf_discriminator,
        input_name_vbf_discriminator,
        output_name_vbf_discriminator,
    ) = get_model_session(vbf_discriminator, "vbf_discriminator")

    vbf_discriminator_output, _ = get_onnx_prediction(
        model_session_vbf_discriminator,
        input_name_vbf_discriminator,
        output_name_vbf_discriminator,
        events,
        vbf_discriminator_input_variables,
        pad_value,
        pad_value_spanet,
        max_num_jets_vbf_discriminator,
    )

    # Not needed anymore
    del (
        model_session_vbf_discriminator,
        input_name_vbf_discriminator,
        output_name_vbf_discriminator,
    )

    return vbf_discriminator_output


def eval_vbf_pairing(
    events,
    vbf_discriminator_output,
    vbf_discriminator_input_variables,
    min_num_jets=2,
):
    """
    Build the energy ordered VBF jet pair from the assignment probabilities of a
    model which predicts the VBF jets on top of the ggF/VBF classification.

    Returns None when the model does not provide any jet assignment, so that the
    caller can fall back to another VBF pair definition.
    """
    assignment_prob = vbf_discriminator_output["assignment_prob"]
    if len(assignment_prob) == 0:
        return None

    jet_collection = events[get_pairing_collection(vbf_discriminator_input_variables)]

    # an event with less than two jets cannot have a VBF pair: zero its
    # probabilities and mask it after the extraction
    mask_enough_jets = ak.to_numpy(
        ak.count(jet_collection.pt, axis=1) >= min_num_jets
    )
    cleaned_assignment_prob = [np.copy(x) for x in assignment_prob]
    for prob in cleaned_assignment_prob:
        prob[~mask_enough_jets] = 0

    pairing_predictions, *_ = get_best_pairings(cleaned_assignment_prob)

    # the VBF pair is the last resonance predicted by the model
    return reconstruct_vbf_jets_from_idx(
        jet_collection, pairing_predictions[:, -1, :], mask_enough_jets
    )
