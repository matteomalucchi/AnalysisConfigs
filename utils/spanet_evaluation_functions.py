import numpy as np
import awkward as ak

from utils.prediction_selection import extract_predictions

PAD_VALUE_SPANET = 9999.0


def define_spanet_inputs(events, max_num_jets, spanet_input_name_list):

    input_dict = {}
    
    # for variable_name in ["eta", "phi", "btag", "btagPNetB_5wp","btagPNetB_3wp", "btagPNetB_delta5wp", "btagPNetB_delta3wp"]:
    for variable_name in spanet_input_name_list: 
        if variable_name not in ["log_pt", "btag12_ratioSubLead", "btag_ratioAll"]:
            input_dict[variable_name] = np.array(
                ak.to_numpy(
                    ak.fill_none(
                        ak.pad_none(
                            getattr(events.JetGood, variable_name),
                            max_num_jets,
                            clip=True,
                        ),
                        value=PAD_VALUE_SPANET,
                    ),
                    allow_missing=True,
                ),
                dtype=np.float32,
            )

    if "log_pt" in spanet_input_name_list:
        log_pt = np.array(
            np.log(
                ak.to_numpy(
                    ak.fill_none(
                        ak.pad_none(events.JetGood.pt, max_num_jets, clip=True),
                        value=PAD_VALUE_SPANET,
                    ),
                    allow_missing=True,
                )
                + 1
            ),
            dtype=np.float32,
        )
        input_dict["log_pt"] = log_pt

    # Define btag and variations
    btag_padded = ak.pad_none(events.JetGood.btagPNetB, max_num_jets, clip=True)

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

    if max_num_jets > 4:
        btag_ratio_sum_5 = btag_padded[:, 4] / (btag_padded[:, 2] + btag_padded[:, 3])

        btag12_ratioSubLead_list.append(btag_ratio_sum_5)
        btag_ratioAll_list.append(btag_ratio_sum_5)

    if "btag12_ratioSubLead" in spanet_input_name_list:
        btag12_ratioSubLead = np.array(
            ak.to_numpy(
                np.stack(
                    ak.fill_none(
                        btag12_ratioSubLead_list,
                        value=PAD_VALUE_SPANET,
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
                        value=PAD_VALUE_SPANET,
                    ),
                    axis=-1,
                ),
                allow_missing=True,
            ),
            dtype=np.float32,
        )
        input_dict["btag_ratioAll"] = btag_ratioAll
        
    try:
        assert len(input_dict) == len(spanet_input_name_list)
    except AssertionError:
        print(f"Error: Not all inputs in spanet_input_name_list were defined.")
        print("Available inputs in input_dict:", input_dict.keys())
        # find the missing inputs
        missing_inputs = set(spanet_input_name_list) - set(input_dict.keys())
        print(f"Missing inputs: {missing_inputs}")
        print(f"New inputs can be defined in define_spanet_inputs function.")
        raise AssertionError

    # order the inputs according to the spanet_input_name_list
    input_list = [
        input_dict[name] for name in spanet_input_name_list if name in input_dict
    ]

    inputs = np.stack(input_list, axis=-1)

    return inputs


def get_pairing_information(
    session, input_name, output_name, events, max_num_jets, spanet_input_name_list
):

    inputs = define_spanet_inputs(events, max_num_jets, spanet_input_name_list)

    mask = np.array(
        ak.to_numpy(
            ak.fill_none(
                ak.pad_none(
                    ak.ones_like(events.JetGood.pt),
                    max_num_jets,
                    clip=True,
                ),
                value=0,
            ),
            allow_missing=True,
        ),
        dtype=np.bool_,
    )
    inputs_complete = {input_name[0]: inputs, input_name[1]: mask}

    outputs = session.run(output_name, inputs_complete)

    return outputs


def get_best_pairings(outputs):

    # extract the best jet assignment from
    # the predicted probabilities

    # NOTE: here the way this was implemented was changed
    assignment_probability = np.stack((outputs[0], outputs[1]), axis=0)
    # assignment_probability = [outputs[0], outputs[1]]

    # swap axis
    predictions_best = np.swapaxes(extract_predictions(assignment_probability), 0, 1)
    # assignment_probability=np.array(assignment_probability)

    # get the probabilities of the best jet assignment

    # NOTE: here the way this was implemented was changed
    num_events = assignment_probability.shape[1]

    range_num_events = np.arange(num_events)
    best_pairing_probabilities = np.ndarray((2, num_events))
    for i in range(2):
        best_pairing_probabilities[i] = assignment_probability[
            i,
            range_num_events,
            predictions_best[:, i, 0],
            predictions_best[:, i, 1],
        ]
    best_pairing_probabilities_sum = np.sum(best_pairing_probabilities, axis=0)

    # set to zero the probabilities of the best jet assignment, the symmetrization and the same jet assignment on the other target
    for j in range(2):
        for k in range(2):
            assignment_probability[
                j,
                range_num_events,
                predictions_best[:, j, k],
                predictions_best[:, j, 1 - k],
            ] = 0
            assignment_probability[
                1 - j,
                range_num_events,
                predictions_best[:, j, k],
                predictions_best[:, j, 1 - k],
            ] = 0

    # extract the second best jet assignment from
    # the predicted probabilities
    # swap axis
    predictions_second_best = np.swapaxes(
        extract_predictions(assignment_probability), 0, 1
    )

    # get the probabilities of the second best jet assignment
    second_best_pairing_probabilities = np.ndarray((2, num_events))
    for i in range(2):
        second_best_pairing_probabilities[i] = assignment_probability[
            i,
            range_num_events,
            predictions_second_best[:, i, 0],
            predictions_second_best[:, i, 1],
        ]
    second_best_pairing_probabilities_sum = np.sum(
        second_best_pairing_probabilities, axis=0
    )

    return (
        predictions_best,
        best_pairing_probabilities_sum,
        second_best_pairing_probabilities_sum,
    )
