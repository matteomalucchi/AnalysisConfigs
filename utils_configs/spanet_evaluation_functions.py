import numpy as np
import awkward as ak
import copy

from utils_configs.prediction_selection import extract_predictions


def define_spanet_sequential_inputs(
    events, max_num_jets_spanet, collection, spanet_input_name_list, pad_value_spanet
):
    """
    Define the sequential (2D arrays) input features for the SPANet model.
    """
    input_dict = {}

    for variable_name in spanet_input_name_list:
        # Determine, if we have a log scale
        islog = False
        if ":" in variable_name:
            variable_name, scale = variable_name.split(":")
            if "log" in scale:
                islog = True

        if variable_name not in ["btag12_ratioSubLead", "btag_ratioAll"]:
            if islog:
                # apply the log to the padded value
                input_dict[variable_name] = np.array(
                    np.log(
                        ak.to_numpy(
                            ak.fill_none(
                                ak.pad_none(
                                    getattr(events[collection], variable_name),
                                    max_num_jets_spanet,
                                    clip=True,
                                ),
                                value=pad_value_spanet,
                            ),
                            allow_missing=True,
                        )
                        + 1
                    ),
                    dtype=np.float32,
                )
            else:
                input_dict[variable_name] = np.array(
                    ak.to_numpy(
                        ak.fill_none(
                            ak.pad_none(
                                getattr(events[collection], variable_name),
                                max_num_jets_spanet,
                                clip=True,
                            ),
                            value=pad_value_spanet,
                        ),
                        allow_missing=True,
                    ),
                    dtype=np.float32,
                )

    # Define btag and variations
    btag_padded = ak.pad_none(
        events[collection].btagPNetB, max_num_jets_spanet, clip=True
    )

    if max_num_jets_spanet >= 4:
        btag_ratio_sum_1 = btag_padded[:, 0] / (btag_padded[:, 0] + btag_padded[:, 1])
        btag_ratio_sum_2 = btag_padded[:, 1] / (btag_padded[:, 0] + btag_padded[:, 1])
        btag_ratio_sum_3 = btag_padded[:, 2] / (btag_padded[:, 2] + btag_padded[:, 3])
        btag_ratio_sum_4 = btag_padded[:, 3] / (btag_padded[:, 2] + btag_padded[:, 3])

        btag12_ratioSubLead_list = [
            btag_padded[:, 0],
            btag_padded[:, 1],
            btag_ratio_sum_3,
            btag_ratio_sum_4,
        ]
        btag_ratioAll_list = [
            btag_ratio_sum_1,
            btag_ratio_sum_2,
            btag_ratio_sum_3,
            btag_ratio_sum_4,
        ]

        if max_num_jets_spanet > 4:
            btag_ratio_sum_5 = btag_padded[:, 4] / (btag_padded[:, 2] + btag_padded[:, 3])

            btag12_ratioSubLead_list.append(btag_ratio_sum_5)
            btag_ratioAll_list.append(btag_ratio_sum_5)

        if "btag12_ratioSubLead" in spanet_input_name_list:
            btag12_ratioSubLead = np.array(
                ak.to_numpy(
                    np.stack(
                        ak.fill_none(
                            btag12_ratioSubLead_list,
                            value=pad_value_spanet,
                        ),
                        axis=-1,
                    ),
                    allow_missing=True,
                ),
                dtype=np.float32,
            )
            input_dict["btag12_ratioSubLead"] = btag12_ratioSubLead

        if "btag_ratioAll" in spanet_input_name_list:
            btag_ratioAll = np.array(
                ak.to_numpy(
                    np.stack(
                        ak.fill_none(
                            btag_ratioAll_list,
                            value=pad_value_spanet,
                        ),
                        axis=-1,
                    ),
                    allow_missing=True,
                ),
                dtype=np.float32,
            )
            input_dict["btag_ratioAll"] = btag_ratioAll

    return input_dict


def define_spanet_pairing_inputs(
    events, max_num_jets_spanet, collection, spanet_input_name_list, pad_value_spanet
):
    """
    Define the input features for the SPANet model used for jet pairing.
    """

    input_dict = define_spanet_sequential_inputs(
        events,
        max_num_jets_spanet,
        collection,
        spanet_input_name_list,
        pad_value_spanet,
    )
    # TODO: add global inputs for the pairing as well

    try:
        assert len(input_dict) == len(spanet_input_name_list)
    except AssertionError:
        print(f"Error: Not all inputs in spanet_input_name_list were defined.")
        print("Available inputs in input_dict:", input_dict.keys())
        # find the missing inputs
        missing_inputs = set(spanet_input_name_list) - set(input_dict.keys())
        print(f"Missing inputs: {missing_inputs}")
        print(f"New inputs can be defined in define_spanet_pairing_inputs function.")
        raise AssertionError

    # order the inputs according to the spanet_input_name_list
    input_list = [
        input_dict[name.split(":")[0]]
        for name in spanet_input_name_list
        if name.split(":")[0] in input_dict
    ]

    inputs = np.stack(input_list, axis=-1)

    return inputs


def get_pairing_information(
    session,
    input_name,
    output_name,
    events,
    max_num_jets_spanet,
    spanet_input_name_list,
    pad_value_spanet,
):
    inputs_complete = {}
    inputs = define_spanet_pairing_inputs(
        events, max_num_jets_spanet, spanet_input_name_list, pad_value_spanet
    )

    mask = np.array(
        ak.to_numpy(
            ak.fill_none(
                ak.pad_none(
                    ak.ones_like(events.JetGood.pt),
                    max_num_jets_spanet,
                    clip=True,
                ),
                value=0,
            ),
            allow_missing=True,
        ),
        dtype=np.bool_,
    )
    inputs_complete |= {input_name[0]: inputs, input_name[1]: mask}

    outputs = session.run(output_name, inputs_complete)

    return outputs


def get_best_pairings(assignment_prob):
    """
    Extract the best jet assignment of every resonance from the predicted
    assignment probabilities.

    `assignment_prob` is the list of the (n_events, n_jets, n_jets) probability
    matrices predicted by SPANet, one per resonance decaying to two jets. It
    works for any number of resonances, so it can be used both for the two Higgs
    candidates and for the VBF pair (alone or together with the Higgs ones).

    Returns the predicted indices, with shape (n_events, n_resonances, 2), and
    the sum over the resonances of the probability of the best, of the second
    best and of the worst assignment.
    """
    assignment_probability = np.stack(tuple(assignment_prob), axis=0)

    num_resonances, num_events = assignment_probability.shape[:2]
    range_num_events = np.arange(num_events)

    prediction_list = []
    pairing_probabilities_sum_list = []

    # the assignments are extracted from the most to the least probable one:
    # every iteration zeroes the assignment it just took, so the loop stops once
    # no assignment is left
    while True:
        # swap axis to have the events on the first axis
        predictions = np.swapaxes(extract_predictions(assignment_probability), 0, 1)
        prediction_list.append(predictions)

        # get the probability of the assignment chosen for each resonance
        pairing_probabilities = np.stack(
            [
                assignment_probability[
                    i,
                    range_num_events,
                    predictions[:, i, 0],
                    predictions[:, i, 1],
                ]
                for i in range(num_resonances)
            ],
            axis=0,
        )
        pairing_probabilities_sum_list.append(np.sum(pairing_probabilities, axis=0))

        # set to zero the probabilities of the chosen jet assignment, of its
        # symmetrization and of the same jet assignment on the other resonances
        for j in range(num_resonances):
            for k in range(2):
                for i in range(num_resonances):
                    assignment_probability[
                        i,
                        range_num_events,
                        predictions[:, j, k],
                        predictions[:, j, 1 - k],
                    ] = 0

        if np.sum(assignment_probability) <= 0:
            break

    # a probability matrix with a single assignment left has no second best one
    second_best_index = min(1, len(pairing_probabilities_sum_list) - 1)

    return (
        prediction_list[0],
        pairing_probabilities_sum_list[0],
        pairing_probabilities_sum_list[second_best_index],
        pairing_probabilities_sum_list[-1],
    )


def clean_assignment_prob(assignment_prob, jet_coll_pairing):
    # Deep copy each element explicitly
    cleaned_assignment_prob = [np.copy(x) for x in assignment_prob]
    # if an event has less than 6 jets, than remove the vbf prob matrix
    if len(cleaned_assignment_prob) == 3:
        # Count non-None jets per event and mask the ones with <6
        mask_bad = ak.count(jet_coll_pairing.pt, axis=1) < 6

        # Replace assignment_prob[2] for bad events
        cleaned_assignment_prob[2][mask_bad] = np.zeros_like(
            cleaned_assignment_prob[2][mask_bad]
        )

    return cleaned_assignment_prob
