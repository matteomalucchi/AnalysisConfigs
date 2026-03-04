def get_DNN_input_list(run2, dnn_input_variables):
    """Create the DNN input variables"""
    column_list = []
    if "sequential" in dnn_input_variables:
        dnn_input_variables = dnn_input_variables["sequential"] | dnn_input_variables["global"]
    for var_name, attributes in dnn_input_variables.items():
        collection = attributes[0]
        feature = attributes[1]
        if len(attributes) > 2:
            scale = attributes[2]
        if run2:
            if ":" in collection:
                coll, pos = collection.split(":")
                column_list.append(f"{coll}Run2_{feature}_{pos}")
            elif collection != "events":
                column_list.append(f"{collection}Run2_{feature}")
            elif "sigma" in feature:
                column_list.append(f"{collection}_{feature}Run2")
            else:
                column_list.append(f"{collection}_{feature}")
        else:
            if ":" in collection:
                coll, pos = collection.split(":")
                column_list.append(f"{coll}_{feature}_{pos}")
            else:
                column_list.append(f"{collection}_{feature}")

    return column_list

if __name__ == "__main__":
    from configs.HH4b_common.dnn_input_variables import sig_bkg_dnn_input_variables

    columns = get_DNN_input_list(False, sig_bkg_dnn_input_variables)
    for var in columns:
        print(var)
