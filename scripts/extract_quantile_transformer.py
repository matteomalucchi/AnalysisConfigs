import logging
import matplotlib.pyplot as plt
import os

import numpy as np

import configs.HH4b_common.dnn_input_variables as dnn_input_variables
from utils.get_DNN_input_list import get_DNN_input_list
from utils.inference_session_onnx import get_model_session
from utils.plot.args_plot import args
from utils.plot.get_columns_from_files import get_columns_from_files
from utils.quantile_transformer import WeightedQuantileTransformer

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger()


def extract_quantile_transformer(cat_col):
    logger.setLevel(logging.INFO)
    spanet = True
    """Compute the transformation function to rebin the scores such that the 4b signal is constant in each bin."""
    cat_dict = {}
    if args.run2:
        cat_dict[f"SR{args.region_suffix}Run2"] = [
            f"4b{args.region_suffix}_signal_regionRun2",
        ]
    elif spanet:
        cat_dict[f"SR{args.region_suffix}"] = [
            f"4b{args.region_suffix}_signal_region",
        ]

    # cat_dict defined on top (global variable)
    for cats_name, cat_list in cat_dict.items():
        dir_cat = f"{outputdir}/{cats_name}_qt"
        if not os.path.exists(dir_cat):
            os.makedirs(dir_cat)
        col_dict = {}

        mc = "MC"
        logger.debug(cat_col.keys())
        vars_tot = list(cat_col[cat_list[0]].keys())
        if args.test:
            vars_tot = vars_tot[:3]
        vars_to_plot = []
        for v in vars_tot:
            if "_N" in v:
                continue
            v_pref = v.split("_")[0]
            if v_pref + "_N" in vars_tot:
                N = cat_col[cat_list[0]][v_pref + "_N"][0]
                try:
                    assert (cat_col[cat_list[0]][v_pref + "_N"] == N).all()
                except AssertionError:
                    logger.warn(f"Variables {v_pref} have different N values: {cat_col[cat_list[0]][v_pref + '_N']}. Skipping...")
                    continue

                for idx in range(N):
                    if f"{v}_{idx}" not in col_dict.keys():
                        col_dict[f"{v}_{idx}"] = {}
                    vars_to_plot.append(f"{v}_{idx}")
                    for cat in cat_list:
                        cat_mc = f"{cat}_{mc}"
                        try:
                            col_dict[f"{v}_{idx}"][cat_mc] = cat_col[cat][v][np.arange(len(cat_col[cat][v])) % N == idx]
                        except KeyError:
                            col_dict[f"{v}_{idx}"][cat_mc] = cat_col[cat][v.replace("Run2", "")][np.arange(len(cat_col[cat][v.replace("Run2", "")])) % N == idx]
            else:
                if v not in col_dict.keys():
                    col_dict[v] = {}
                if v != "weight":
                    vars_to_plot.append(v)
                for cat in cat_list:
                    cat_mc = f"{cat}_{mc}"
                    try:
                        col_dict[v][cat_mc] = cat_col[cat][v]
                    except KeyError:
                        col_dict[v][cat_mc] = cat_col[cat][v.replace("Run2", "")]

        # compute the DNN score if onnx model is given
        if args.onnx_model:
            if any(["score" in v for v in vars_to_plot]):
                logger.info("Found score variables and onnx model")
                logger.info("The score will be overwritten by the onnx model")

            v = f"events_sig_bkg_dnn_score{'Run2' if args.run2 else ''}"
            if v not in col_dict.keys():
                col_dict[v] = {}
            if v not in vars_to_plot:
                vars_to_plot.append(v)

            for cat in cat_list:
                cat_mc = f"{cat}_{mc}"
                input_variables_array = []
                for input_var in dnn_input_list:
                    input_variables_array.append(np.array(col_dict[input_var][cat_mc], dtype=np.float32))
                input_variables_array = np.stack(input_variables_array, axis=-1)
                inputs_complete = {input_name_SIG_BKG_DNN[0]: input_variables_array}
                outputs = model_session_SIG_BKG_DNN.run(output_name_SIG_BKG_DNN, inputs_complete)
                col_dict[v][cat_mc] = outputs[0][:, -1]
                del input_variables_array, inputs_complete, outputs

        vars_to_plot_final = [v for v in vars_to_plot if "score" in v]

        logger.debug(f"col_dict {col_dict}")
        logger.debug(f"vars_to_plot_final {vars_to_plot_final}")

        logger.debug(f"Category name: {cats_name}")
        for var in vars_to_plot_final:
            var_plot_name = var.replace("Run2", "")
            logger.debug(f"Variable: {var}")
            kl = os.path.basename(inputfiles[0]).split("kl-")[-1].split("_")[0].replace("p", ".")
            savesuffix = f"kl_{kl}"
            if kl == "1.00" and "score" in var:
                transformer = WeightedQuantileTransformer(output_distribution="uniform", n_quantiles=21)  # 1000000)
                transformer.fit(col_dict[var][cat_mc], sample_weight=col_dict["weight"][cat_mc])
                transformer.save(os.path.join(dir_cat, f"qt_{var_plot_name}_{savesuffix}.pkl".replace("Run2", "_DHH")))
                # For debugging to see, if transformation is fine.
                transformer2 = WeightedQuantileTransformer(output_distribution="uniform", n_quantiles=0)  # 1000000)
                transformer2.load(os.path.join(dir_cat, f"qt_{var_plot_name}_{savesuffix}.pkl".replace("Run2", "_DHH")))
                print("Saving transformed plot")
                col_den_transformed = transformer2.transform(col_dict[var][cat_mc])
                bins = transformer2.get_quantiles()
                print(bins)
                bins_final = bins
                bins_final[0] = 0.0
                bins_final[-1] = 1.0
                print(f"Current observable {var}, {cat_mc}")
                print(f"bin edges: {bins_final}")
                plt.hist(col_den_transformed, weights=col_dict["weight"][cat_mc], bins=20, label="transformed")
                plt.savefig(f"./test_transforming_{var_plot_name}_{savesuffix}.png")
                print("Printing transformed column")
                hist, bins = np.histogram(col_den_transformed, weights=col_dict["weight"][cat_mc], bins=20)
                print(hist)
                print("Mean:", np.mean(hist), "Std:", np.std(hist), "Rel. Std:", np.std(hist)/np.mean(hist))
                print("Printing normal column with transformed bins")
                hist, bins = np.histogram(col_dict[var][cat_mc], weights=col_dict["weight"][cat_mc], bins=bins_final)
                print(hist)
                print("Mean:", np.mean(hist), "Std:", np.std(hist), "Rel. Std:", np.std(hist)/np.mean(hist))
        del col_dict


if __name__ == "__main__":
    if not args.output:
        if not args.test:
            args.output = "quantile_transformer"
        else:
            args.output = "test_transformer"

    outputdir = args.output

    # In this script I only want MC files. Everything with data should be ignored (only partially supported for parquet files so far).
    # Also, I want the transformation only for signal 4b

    # To mix categories with Run2 and SPANet, put first the Run2 category
    # because first the name of the variables is try with the Run2 string
    # and after without it
    # Second region: data 2b reweighted (unblinded)
    # Third region: mc 4b (unblinded)

    # -- Load the onnx model --
    if args.onnx_model:
        (
            model_session_SIG_BKG_DNN,
            input_name_SIG_BKG_DNN,
            output_name_SIG_BKG_DNN,
        ) = get_model_session(args.onnx_model, "SIG_BKG_DNN")
        # load the variables for the DNN
        # get the list name from the string args.input_variables
        dnn_variables = getattr(dnn_input_variables, args.input_variables)
        dnn_input_list = get_DNN_input_list(args.run2, dnn_variables)
        logger.info(f"Input list for DNN: {dnn_input_list}")

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # Hack, because I only want input mc, but this is easier with -i flag
    args.input = args.input_data

    inputfiles = args.input

    filter_lambda = (
        (
            lambda x: (
                "weight" in x
                or ("score" in x and ("Run2" in x if args.run2 else "Run2" not in x))
            )
        )
        if not args.onnx_model
        else None
    )

    # == Collecting MC dataset ==
    cat_col_mc, total_datasets_list_mc = get_columns_from_files(
        inputfiles, sel_var="nominal", filter_lambda=filter_lambda, debug=False, novars=args.novars
    )
    print(total_datasets_list_mc)

    logger.info("cat_col_mc")
    for key, value in cat_col_mc.items():
        logger.info(f"{key}: {value.keys()}")

    # === Actual plotting command. [datastuff, mcstuff] ===
    extract_quantile_transformer(cat_col_mc)
