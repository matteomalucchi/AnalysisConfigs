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


YEARS = ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix", "2024"]
PROVENANCE="provenance"

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


def autolabel(ax, bars):
    for bar in bars:
        height = bar.get_height()
        # transform to percentage
        percentage = height * 100

        ax.annotate(
            f"{percentage:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )
        
def _draw_bars(ax, x, effs, width, **bar_kwargs):
    bars = ax.bar(x, effs, width, **bar_kwargs)
    autolabel(ax, bars)
    return bars

def _finalize_efficiency_plot(
    ax,
    x,
    xticklabels,
    title,
    lumitext_str,
    rotate_xticks=False,
    legend=False,
    legend_kwargs=None,
):
    ax.set_ylabel("Efficiency %", fontsize=24)

    ax.text(
        0.97,
        0.97,
        title,
        transform=ax.transAxes,
        fontsize=18,
        ha="right",
        va="top",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        xticklabels,
        rotation=45 if rotate_xticks else 0,
        ha="right" if rotate_xticks else "center",
        fontsize=18,
    )

    ax.set_ylim(0, 1.2)

    hep.cms.lumitext(lumitext_str, ax=ax, fontsize=24)
    hep.cms.text("Preliminary", ax=ax, fontsize=24)

    if legend:
        ax.legend(
            frameon=False,
            **(legend_kwargs or {}),
        )

def plot_efficiencies(list_cuts, efficiencies, title, lumitext_str):
    x = np.arange(len(list_cuts))
    eff_vals = np.asarray(efficiencies)

    fig, ax = plt.subplots()
    width = 0.5

    _draw_bars(ax, x, eff_vals, width)

    _finalize_efficiency_plot(
        ax=ax,
        x=x,
        xticklabels=list_cuts,
        title=title,
        lumitext_str=lumitext_str,
        rotate_xticks=False,
    )

    plt.tight_layout()
    plot_name = title.replace("\n", "_").replace("-", "_").replace(" ", "")
    plt.savefig(f"{args.output}/{plot_name}.png")
    plt.close()

def plot_efficiencies_all_categories_all_jets(
    eff_type,
    eff_dict_eff_type,
    title,
    lumitext_str,
):
    jet_types = list(eff_dict_eff_type.keys())
    categories = list(next(iter(eff_dict_eff_type.values())).keys())
    
    # order the jet types and categories alphabetically
    jet_types.sort()
    categories.sort()
    labels = eff_dict_eff_type[jet_types[0]][categories[0]]["labels"]

    n_labels = len(labels)
    n_jets = len(jet_types)
    n_cats = len(categories)

    x = np.arange(n_labels)
    width = 0.8 / (n_jets * n_cats)

    fig, ax = plt.subplots(figsize=(18, 10))

    # colors per jet collection
    color_dict = list(hep.style.CMS["axes.prop_cycle"])
    color_list = [cycle["color"] for cycle in color_dict]

    # hatches per region
    hatches = ["", "//", "xx", "..", "\\\\", "++"]

    offset = 0
    for j, jet_type in enumerate(jet_types):
        for c, cat in enumerate(categories):
            effs = eff_dict_eff_type[jet_type][cat]["efficiencies"]

            _draw_bars(
                ax=ax,
                x=x + offset,
                effs=effs,
                width=width,
                color=color_list[j],
                hatch=hatches[c % len(hatches)],
                edgecolor="black",
                label=f"{jet_type} | {cat}",
            )

            offset += width

    _finalize_efficiency_plot(
        ax=ax,
        x=x + width * (n_jets * n_cats - 1) / 2,
        xticklabels=labels,
        title=title,
        lumitext_str=lumitext_str,
        legend=True,
        legend_kwargs=dict(fontsize=12, ncol=2),
    )

    plt.tight_layout()
    plot_name = title.replace("\n", "_").replace("-", "_").replace(" ", "")
    plt.savefig(f"{args.output}/{plot_name}_allJets_allCats.png")
    plt.close()


def remove_year_from_dataset_string(dataset_string):
    for year in YEARS:
        if year in dataset_string:
            dataset_string = dataset_string.replace(f"_{year}", "")
    return dataset_string

def main(cat_cols, lumitext_str, total_datasets_list):
    dataset_string = remove_year_from_dataset_string("_".join(total_datasets_list)).rstrip("_")
    logger.info(f"Processing datasets: {dataset_string}")
    
    eff_dict = defaultdict(lambda: defaultdict(dict))
    for cat, cols in cat_cols.items():
        logger.info(f"Processing category: {cat}")
        num_events = len(cols["weight"])
        logger.info(f"Number of events for category {cat}: {num_events}")

        # find the jet collections in the columns
        jets_set = set()
        for col in cols.keys():
            if "Jet" in col:
                jets_set.add(col.split("_")[0])
        jets_list = list(jets_set)

        for jet_type in jets_list:
            # remove year from dataset
            prov = ak.values_astype(cols[f"{jet_type}_{PROVENANCE}"], np.int64)
            prov_uflattened = ak.unflatten(prov, cols[f"{jet_type}_N"])
            jet_type=jet_type.replace("Padded", "")
            title = f"{jet_type} \n {dataset_string.rstrip('_')} \n {cat}"
            # count how many matched jets
            counts = []
            for i in [1, 2, 3]:
                counts.append(ak.sum(prov == i))
            num_max_matched_jets = [2 * num_events, 2 * num_events, 2 * num_events]
            efficiencies = np.array(counts) / np.array(num_max_matched_jets)
            labels = ["Higgs 1", "Higgs 2", "VBF"]
            eff_type="Efficiency per jet"
            logger.info(
                f"{eff_type} for {jet_type} in category {cat}: {dict(zip(labels, efficiencies))}"
            )
            plot_efficiencies(labels, efficiencies, f"{eff_type} \n {title}", lumitext_str)
            eff_dict[eff_type][jet_type][cat] = {
                "labels": labels,
                "efficiencies": efficiencies,
            }

            # count how many matched resonances
            counts = []
            for i in [1, 2, 3]:
                counts.append(ak.sum(ak.sum(prov_uflattened == i, axis=1) == 2))
            efficiencies = np.array(counts) / np.array(num_events)
            labels = ["Higgs 1", "Higgs 2", "VBF"]
            eff_type="Efficiency per resonance"
            logger.info(
                f"{eff_type} for {jet_type} in category {cat}: {dict(zip(labels, efficiencies))}"
            )
            plot_efficiencies(labels, efficiencies, f"{eff_type} \n {title}", lumitext_str)
            eff_dict[eff_type][jet_type][cat] = {
                "labels": labels,
                "efficiencies": efficiencies,
            }

            # count how many matched resonances (combine the Higgs)
            counts = []
            for i in [[1, 2], [3]]:
                counts.append(
                    ak.sum(
                        (ak.sum(prov_uflattened == i[0], axis=1) == 2)
                        & (ak.sum(prov_uflattened == i[-1], axis=1) == 2)
                    )
                )
            efficiencies = np.array(counts) / np.array(num_events)
            labels = ["Higgs 1 + Higgs 2", "VBF"]
            eff_type="Efficiency per resonance combine Higgs"
            logger.info(
                f"{eff_type} for {jet_type} in category {cat}: {dict(zip(labels, efficiencies))}"
            )
            plot_efficiencies(labels, efficiencies, f"{eff_type} \n {title}", lumitext_str)
            eff_dict[eff_type][jet_type][cat] = {
                "labels": labels,
                "efficiencies": efficiencies,
            }

            # count how many fully matched events
            counts = []
            for i in [[1, 2, 3]]:
                counts.append(
                    ak.sum(
                        (ak.sum(prov_uflattened == i[0], axis=1) == 2)
                        & (ak.sum(prov_uflattened == i[1], axis=1) == 2)
                        & (ak.sum(prov_uflattened == i[2], axis=1) == 2)
                    )
                )
            efficiencies = np.array(counts) / np.array(num_events)
            labels = ["Higgs 1 + Higgs 2 + VBF"]
            eff_type="Efficiency fully matched events"
            logger.info(
                f"{eff_type} for {jet_type} in category {cat}: {dict(zip(labels, efficiencies))}"
            )
            plot_efficiencies(labels, efficiencies, f"{eff_type} \n {title}", lumitext_str)
            eff_dict[eff_type][jet_type][cat] = {
                "labels": labels,
                "efficiencies": efficiencies,
            }

    for eff_type, eff_dict_eff_type in eff_dict.items():
        plot_efficiencies_all_categories_all_jets(
            eff_type=eff_type,
            eff_dict_eff_type=eff_dict_eff_type,
            title=f"{eff_type}\n{dataset_string}",
            lumitext_str=lumitext_str,
        )


if __name__ == "__main__":
    # Load coffea file
    inputfiles = [args.input_file]
    logger.info(f"Loading coffea file: {inputfiles}")

    cat_col, total_datasets_list = get_columns_from_files(
        inputfiles, "nominal", None, debug=False, novars=args.novars
    )
    lumi, era_string = get_era_lumi(total_datasets_list)

    lumitext_str = f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)"
    main(cat_col, lumitext_str, total_datasets_list)
