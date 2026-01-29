from collections import defaultdict
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib
import argparse
import os
import logging

matplotlib.rcParams["figure.dpi"] = 300
hep.style.use("CMS")

from utils.plot.get_era_lumi import get_era_lumi
from utils.plot.get_columns_from_files import get_columns_from_files
from utils.plot.HEPPlotter import HEPPlotter

parser = argparse.ArgumentParser(description="Plot truth matching efficiencies")
parser.add_argument(
    "-i",
    "--input-file",
    type=str,
    required=True,
    help="Input coffea file",
)
parser.add_argument(
    "-o", "--output", type=str, help="Output directory", default="plots_cutflow"
)
parser.add_argument(
    "--novars",
    action="store_true",
    help="If true, old save format without saved variations is expected",
    default=False,
)
args = parser.parse_args()


# global constants
YEARS = ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix", "2024"]
PROVENANCE = "provenance"
JETS_ORDER = [
    "Jet",
    "JetGood",
    "JetGoodVBFMerged",
    "JetTotalSPANet",
    "JetGoodHiggsPlusVBF1mjj",
]
CATEGORIES_ORDER = ["4b_region", "vbf_4b_region"]


# make output directory if it does not exist
if not os.path.exists(args.output):
    os.makedirs(args.output)

# Logger
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger()
formatter = logging.Formatter("%(asctime)s  - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(
    f"{args.output}/compare_truth_matching_efficiencies.log"
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


def plot_efficiencies_all_categories_all_jets(
    eff_type,
    eff_dict_eff_type,
    title,
    lumitext_str,
    output_base,
):
    # ----------------------------
    # PREPARE AXES CONTENT
    # ----------------------------

    # sort the jet types and categories according to predefined order
    jet_types = sorted(
        eff_dict_eff_type.keys(),
        key=lambda x: JETS_ORDER.index(x) if x in JETS_ORDER else len(JETS_ORDER),
    )
    categories = sorted(
        next(iter(eff_dict_eff_type.values())).keys(),
        key=lambda x: (
            CATEGORIES_ORDER.index(x)
            if x in CATEGORIES_ORDER
            else len(CATEGORIES_ORDER)
        ),
    )

    labels = eff_dict_eff_type[jet_types[0]][categories[0]]["labels"]

    # colors per jet collection (CMS style)
    color_cycle = [c["color"] for c in hep.style.CMS["axes.prop_cycle"]]

    # hatches per category
    hatches = ["", "//", "xx", "..", "\\\\", "++"]

    series_dict = {}

    for j, jet_type in enumerate(jet_types):
        for c, cat in enumerate(categories):
            effs = eff_dict_eff_type[jet_type][cat]["efficiencies"]

            label = f"{jet_type} | {cat}"

            series_dict[label] = {
                "data": {
                    "categories": labels,
                    "values": effs,
                },
                "style": {
                    "color": color_cycle[j % len(color_cycle)],
                    "hatch": hatches[c % len(hatches)],
                    "edgecolor": "white",
                },
            }

    # ----------------------------
    # RUN HEPPlotter
    # ----------------------------
    (
        HEPPlotter(debug=True)
        .set_plot_config(figsize=(18, 10), lumitext=lumitext_str)
        .set_output(f"{output_base}/{eff_type.replace(' ', '_')}_allJets_allCats")
        .set_labels(
            xlabel=None,
            ylabel="Efficiency %",
        )
        .set_data(series_dict, plot_type="categorical")
        .set_options(
            legend=True,
            legend_loc="upper left",
            legend_font_size=26,
            ylim_top_value=1.4,
            ylim_bottom_value=0,
            grid=False,
            rotate_xticks=False,
        )
        .add_annotation(
            0.97,
            0.97,
            title,
            fontsize=18,
            ha="right",
            va="top",
        )
        .run()
    )


def plot_with_HEPPlotter(labels, efficiencies, lumitext_str, title, output_dir):

    series_dict = {
        title: {
            "data": {
                "categories": labels,
                "values": efficiencies,
            },
            "style": {},
        }
    }

    plot_name = title.replace("\n", "_").replace("-", "_").replace(" ", "")
    (
        HEPPlotter(debug=True)
        .set_plot_config(lumitext=lumitext_str)
        .set_output(f"{output_dir}/{plot_name}")
        .set_labels(xlabel=None, ylabel="Efficiency %")
        .set_data(series_dict, plot_type="categorical")
        .add_annotation(
            0.97,
            0.97,
            title,
            fontsize=18,
            ha="right",
            va="top",
        )
        .set_options(legend=False)
        .run()
    )


def remove_year_from_dataset_string(dataset_string):
    for year in YEARS:
        if year in dataset_string:
            dataset_string = dataset_string.replace(f"_{year}", "")
    return dataset_string


def main(cat_cols, lumitext_str, total_datasets_list):
    dataset_string = remove_year_from_dataset_string(
        "_".join(total_datasets_list)
    ).rstrip("_")
    logger.info(f"Processing datasets: {dataset_string}")

    eff_tot_dict = defaultdict(lambda: defaultdict(dict))

    for cat, cols in cat_cols.items():
        logger.info(f"Processing category: {cat}")
        num_events = len(cols["weight"])
        logger.info(f"Number of events for category {cat}: {num_events}")

        # find jet collections
        jets_set = {col.split("_")[0] for col in cols if "Jet" in col}
        jets_list = list(jets_set)

        for jet_type in jets_list:
            prov = ak.values_astype(cols[f"{jet_type}_{PROVENANCE}"], np.int64)
            prov_uflattened = ak.unflatten(prov, cols[f"{jet_type}_N"])
            jet_type_clean = jet_type.replace("Padded", "")

            base_title = f"{jet_type_clean}\n{dataset_string}\n{cat}"

            # ------------------------------------------------------------------
            # 1) Efficiency per jet
            # ------------------------------------------------------------------
            counts = [ak.sum(prov == i) for i in [1, 2, 3]]
            max_counts = [2 * num_events] * 3
            efficiencies = np.asarray(counts) / np.asarray(max_counts)
            labels = ["Higgs 1", "Higgs 2", "VBF"]
            eff_type = "Efficiency per jet"

            logger.info(
                f"{eff_type} for {jet_type_clean} in {cat}: "
                f"{dict(zip(labels, efficiencies))}"
            )

            plot_with_HEPPlotter(
                labels,
                efficiencies,
                lumitext_str,
                f"{eff_type} \n {base_title}",
                args.output,
            )

            eff_tot_dict[eff_type][jet_type_clean][cat] = {
                "labels": labels,
                "efficiencies": efficiencies,
            }

            # ------------------------------------------------------------------
            # 2) Efficiency per resonance
            # ------------------------------------------------------------------
            counts = [
                ak.sum(ak.sum(prov_uflattened == i, axis=1) == 2) for i in [1, 2, 3]
            ]
            efficiencies = np.asarray(counts) / num_events
            labels = ["Higgs 1", "Higgs 2", "VBF"]
            eff_type = "Efficiency per resonance"

            logger.info(
                f"{eff_type} for {jet_type_clean} in {cat}: "
                f"{dict(zip(labels, efficiencies))}"
            )

            plot_with_HEPPlotter(
                labels,
                efficiencies,
                lumitext_str,
                f"{eff_type} \n {base_title}",
                args.output,
            )

            eff_tot_dict[eff_type][jet_type_clean][cat] = {
                "labels": labels,
                "efficiencies": efficiencies,
            }

            # ------------------------------------------------------------------
            # 3) Efficiency per resonance (combined Higgs)
            # ------------------------------------------------------------------
            counts = [
                ak.sum(
                    (ak.sum(prov_uflattened == 1, axis=1) == 2)
                    & (ak.sum(prov_uflattened == 2, axis=1) == 2)
                ),
                ak.sum(ak.sum(prov_uflattened == 3, axis=1) == 2),
            ]
            efficiencies = np.asarray(counts) / num_events
            labels = ["Higgs 1 + Higgs 2", "VBF"]
            eff_type = "Efficiency per resonance combine Higgs"

            plot_with_HEPPlotter(
                labels,
                efficiencies,
                lumitext_str,
                f"{eff_type} \n {base_title}",
                args.output,
            )

            eff_tot_dict[eff_type][jet_type_clean][cat] = {
                "labels": labels,
                "efficiencies": efficiencies,
            }

            # ------------------------------------------------------------------
            # 4) Fully matched events
            # ------------------------------------------------------------------
            counts = [
                ak.sum(
                    (ak.sum(prov_uflattened == 1, axis=1) == 2)
                    & (ak.sum(prov_uflattened == 2, axis=1) == 2)
                    & (ak.sum(prov_uflattened == 3, axis=1) == 2)
                )
            ]
            efficiencies = np.asarray(counts) / num_events
            labels = ["Higgs 1 + Higgs 2 + VBF"]
            eff_type = "Efficiency fully matched events"

            plot_with_HEPPlotter(
                labels,
                efficiencies,
                lumitext_str,
                f"{eff_type} \n {base_title}",
                args.output,
            )

            eff_tot_dict[eff_type][jet_type_clean][cat] = {
                "labels": labels,
                "efficiencies": efficiencies,
            }

    # ----------------------------------------------------------------------
    # Combined plots (all jets, all categories)
    # ----------------------------------------------------------------------
    for eff_type, eff_dict_eff_type in eff_tot_dict.items():
        plot_efficiencies_all_categories_all_jets(
            eff_type=eff_type,
            eff_dict_eff_type=eff_dict_eff_type,
            title=f"{eff_type}\n{dataset_string}",
            lumitext_str=lumitext_str,
            output_base=args.output,
        )


if __name__ == "__main__":
    # Load coffea file
    inputfiles = [args.input_file]
    logger.info(f"Loading coffea file: {inputfiles}")

    cat_col, total_datasets_list = get_columns_from_files(
        inputfiles, "nominal", None, debug=False, novars=args.novars
    )
    lumi, era_string = get_era_lumi(total_datasets_list)

    lumitext_str = f"{era_string} (13.6 TeV)"
    main(cat_col, lumitext_str, total_datasets_list)
