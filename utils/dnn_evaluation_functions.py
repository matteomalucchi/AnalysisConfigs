import numpy as np
import awkward as ak


def get_dnn_prediction(session, input_name, output_name, events, variables, pad_value, run2=False):
    """
    Generates DNN predictions based on the provided session and input data.

    Parameters:
        session: The TensorFlow or ONNX session used for inference.
        input_name (list): A list containing the name(s) of the input tensor(s).
        output_name (list): A list containing the name(s) of the output tensor(s).
        events: An object containing event data with attributes for collections and features.
        variables (dict): A dictionary mapping variable names to their collection and feature.
        pad_value (float): The value used to pad missing data in the arrays.
        run2 (bool, optional): If True, uses Run2-specific attributes. Defaults to False.

    Returns:
        np.ndarray: The output predictions from the DNN model.
    """
    variables_array = []
    for var_name, attributes in variables.items():
        collection, feature = attributes

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
        variables_array.append(
            np.array(
                ak.to_numpy(
                    ak_array,
                    allow_missing=True,
                ),
                dtype=np.float32,
            )
        )
    inputs = np.stack(variables_array, axis=-1)

    inputs_complete = {input_name[0]: inputs}

    outputs = session.run(output_name, inputs_complete)
    return outputs
