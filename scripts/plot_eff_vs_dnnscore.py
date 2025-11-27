from copy import deepcopy
import os
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib
import vector

from utils.plot.get_era_lumi import get_era_lumi
from utils.plot.get_columns_from_files import get_columns_from_files

# from utils.plot.weighted_quantile import weighted_quantile
# from utils.plot.plot_names import plot_regions_names
import argparse


def calculate_pairing_efficiencies(cols, run2):
    suffix = "Run2" if run2 else ""
    dnn_score = cols[f"events_sig_bkg_dnn_score{suffix}"]
    mask_low = ak.Array(dnn_score) > low
    mask_high = ak.Array(dnn_score) < high
    mask = mask_low & mask_high
    print(f"Amount of events in bin: {sum(mask)}")

    pairings_bin_matched = cols[f"events_correct_prediction{suffix}"][mask][
        cols["events_mask_fully_matched"][mask]
    ]
    eff = sum(pairings_bin_matched) / len(pairings_bin_matched)
    eff_err = (
        sum(pairings_bin_matched)
        / len(pairings_bin_matched)
        * np.sqrt(1 / sum(pairings_bin_matched) + 1 / len(pairings_bin_matched))
    )

    pairings_bin = cols[f"events_correct_prediction{suffix}"][mask]
    tot_eff = sum(pairings_bin) / len(pairings_bin)
    tot_eff_err = (
        sum(pairings_bin)
        / len(pairings_bin)
        * np.sqrt(1 / sum(pairings_bin) + 1 / len(pairings_bin))
    )
    percentage_matched = len(pairings_bin_matched)/len(pairings_bin)
    print(f"Matched {suffix}: {percentage_matched}")
    print(f"Efficiency {suffix}: {eff}")
    print(f"Total Efficiency {suffix}: {tot_eff}")
    return pairings_bin, pairings_bin_matched, eff, eff_err, tot_eff, tot_eff_err, percentage_matched


matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["agg.path.chunksize"] = 10000  # or try 5000, depending on size
hep.style.use("CMS")

vector.register_awkward

NUMBER_OF_BINS = 20

parser = argparse.ArgumentParser(description="Plot 2b morphed vs 4b data")
parser.add_argument(
    "-is",
    "--input-spanet",
    type=str,
    default="/work/tharte/datasets/sig_bkg_classifier_pairing_eff_vs_dnnscore/spanet_ptflat_true_idx_new_fixes/output_GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE.coffea",
)
parser.add_argument(
    "-ir",
    "--input-run2",
    type=str,
    default="/work/tharte/datasets/sig_bkg_classifier_pairing_eff_vs_dnnscore/DHH_method_true_idx_new_fixes/output_GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE.coffea",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Output directory",
    default="./plots_eff_vs_dnnscore",
)
args = parser.parse_args()

# Collecting MC dataset
cat_col_spanet, total_datasets_list_spanet = get_columns_from_files(
    [args.input_spanet], sel_var="nominal", filter_lambda=None, novars=args.novars 
)
cat_col_run2, total_datasets_list_run2 = get_columns_from_files([args.input_run2], sel_var="nominal", filter_lambda=None, novars=args.novars)
print(f"Category_spanet: {cat_col_spanet.keys()}")
print(total_datasets_list_spanet)
print(cat_col_run2.keys())
print(total_datasets_list_run2)

# Get lumi and era string
lumi, era_string = get_era_lumi(total_datasets_list_spanet)

categories = ["4b_region", "4b_signal_region", "4b_control_region"]
categories_Run2 = ["4b_regionRun2", "4b_signal_regionRun2", "4b_control_regionRun2"]
# categories = ["4b_region"]
# categories_Run2 = ["4b_regionRun2"]

pairing_matched = [
    cat_col_spanet[cat]["events_correct_prediction"][
        cat_col_spanet[cat]["events_mask_fully_matched"]
    ]
    for cat in categories
]
pairing_tot = [cat_col_spanet[cat]["events_correct_prediction"] for cat in categories]
pairing_matched = np.concatenate(pairing_matched, axis=0)
pairing_tot = np.concatenate(pairing_tot, axis=0)
print(f"Total amount of events: {len(pairing_tot)}")
print(f"All pairing efficiency SPANet: {sum(pairing_matched)/len(pairing_matched)}")
print(f"All total pairing efficiency SPANet: {sum(pairing_tot)/len(pairing_tot)}")

pairing_matched = [
    cat_col_run2[cat]["events_correct_predictionRun2"][
        cat_col_run2[cat]["events_mask_fully_matched"]
    ]
    for cat in categories_Run2
]
pairing_tot = [
    cat_col_run2[cat]["events_correct_predictionRun2"] for cat in categories_Run2
]
pairing_matched = np.concatenate(pairing_matched, axis=0)
pairing_tot = np.concatenate(pairing_tot, axis=0)
print(f"Total amount of events: {len(pairing_tot)}")
print(f"All pairing efficiency Run2: {sum(pairing_matched)/len(pairing_matched)}")
print(f"All total pairing efficiency Run2: {sum(pairing_tot)/len(pairing_tot)}")

# bin the dataset into bins of same amount of data:
for cat, cat_run2 in zip(categories, categories_Run2):
    cols_spanet = cat_col_spanet[cat]
    cols_run2 = cat_col_run2[cat_run2]

    dnn_scoreR2 = cols_run2["events_sig_bkg_dnn_scoreRun2"]
    # Overwriting the bins due to combination of SPANet and DHH
    bin_edges = np.linspace(0, 1, NUMBER_OF_BINS + 1)
    print(bin_edges)

    eff_dict = {
        "SPANet total bin distribution": [],
        "SPANet bin distribution": [],
        "SPANet efficiency": [],
        "SPANet total efficiency": [],
        "DHH method total bin distribution": [],
        "DHH method bin distribution": [],
        "DHH method efficiency": [],
        "DHH method total efficiency": [],
        "SPANet total matched fraction": [],
        "DHH total matched fraction": [],
    }
    eff_dict_err = deepcopy(eff_dict)

    pairing_matched = cols_spanet["events_correct_prediction"][
        cols_spanet["events_mask_fully_matched"]
    ]
    pairing_tot = cols_spanet["events_correct_prediction"]
    print(f"Region {cat} and {cat_run2}:")
    print(f"Fraction matched SPANet: {len(pairing_matched)/len(pairing_tot)}")
    print(f"Pairing efficiency SPANet: {sum(pairing_matched)/len(pairing_matched)}")
    print(f"Total pairing efficiency SPANet: {sum(pairing_tot)/len(pairing_tot)}")

    pairing_matched = cols_run2["events_correct_predictionRun2"][
        cols_run2["events_mask_fully_matched"]
    ]
    pairing_tot = cols_run2["events_correct_predictionRun2"]
    print(f"Fraction matched Run2: {len(pairing_matched)/len(pairing_tot)}")
    print(f"Pairing efficiency Run2: {sum(pairing_matched)/len(pairing_matched)}")
    print(f"Total pairing efficiency Run2: {sum(pairing_tot)/len(pairing_tot)}")

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        print(f"Range bin: {low} - {high}")

        # Spanet
        (
            pairings_bin,
            pairings_bin_matched,
            eff,
            eff_err,
            tot_eff,
            tot_eff_err,
            perc_matched,
        ) = calculate_pairing_efficiencies(cols_spanet, run2=False)

        # Run2
        (
            pairings_binR2,
            pairings_bin_matchedR2,
            effR2,
            eff_errR2,
            tot_effR2,
            tot_eff_errR2,
            perc_matchedR2,
        ) = calculate_pairing_efficiencies(cols_run2, run2=True)

        # Updating eff_dict
        for name, valeff, valerr in zip(
            list(eff_dict.keys()),
            [
                len(pairings_bin),
                len(pairings_bin_matched),
                eff,
                tot_eff,
                len(pairings_binR2),
                len(pairings_bin_matchedR2),
                effR2,
                tot_effR2,
                perc_matched,
                perc_matchedR2,
            ],
            [
                np.zeros_like(tot_eff),
                np.zeros_like(eff),
                eff_err,
                tot_eff_err,
                np.zeros_like(tot_effR2),
                np.zeros_like(effR2),
                eff_errR2,
                tot_eff_errR2,
                np.zeros_like(perc_matched),
                np.zeros_like(perc_matchedR2),
            ],
        ):
            eff_dict[name].append(valeff)
            eff_dict_err[name].append(valerr)

    labels = ["Pairing Efficiency", "Total Pairing Efficiency"]
    plotnames = ["pairing_efficiency", "total_pairing_efficiency"]
    arrays = [
        [name for name in eff_dict.keys() if "total" not in name],
        [name for name in eff_dict.keys() if "total" in name],
    ]
    # Colors currently hardcoded. To be fixed
    # Expected order: SPANet bins, SPANet eff, DHH bins, DHH eff
    col_list = [
        ["skyblue", "blue", "lawngreen", "limegreen"],
        ["skyblue", "blue", "lawngreen", "limegreen", "darkblue", "darkgreen"],
    ]

    bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    # bins = np.linspace(0, 1, NUMBER_OF_BINS)

    for label, arrays, colors, plotname in zip(labels, arrays, col_list, plotnames):
        fig_events, ax_events = plt.subplots(figsize=[13, 13])
        for idx, array_name in enumerate(arrays):
            if "bin" not in array_name and "matched" not in array_name:
                ax_events.errorbar(
                    bins,
                    eff_dict[array_name],
                    yerr=eff_dict_err[array_name],
                    fmt="*",
                    label=array_name,
                    color=colors[idx],
                )
            elif "matched" in array_name:
                ax_events.step(
                    bins,
                    eff_dict[array_name],
                    where="mid",
                    label=array_name,
                    color=colors[idx],
                )
            else:
                density = [
                    entry / sum(eff_dict[array_name]) for entry in eff_dict[array_name]
                ]
                ax_events.step(
                    bins,
                    density,
                    where="mid",
                    label=array_name,
                    color=colors[idx],
                )

        ax_events.legend(loc="right")
        ax_events.set_yscale("linear")
        hep.cms.lumitext(
            f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)",
            ax=ax_events,
        )
        ax_events.set_xlabel("DNN Score")
        ax_events.set_ylabel(label)
        ax_events.grid()
        fig_events.show()
        fig_events.savefig(
            os.path.join(args.output, f"{cat}_{plotname}.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig_events)

        fig_relative, ax_relative = plt.subplots(figsize=[13, 13])
        binsize_array_spanet = [array for array in arrays if "bin" in array and "SPANet" in array][0]
        binsize_array_run2 = [array for array in arrays if "bin" in array and "SPANet" not in array][0]
        samplearrays = [array for array in arrays if "bin" not in array and "matched" not in array]
        matchedarrays = [array for array in arrays if "matched" in array]
        density_spanet = ak.Array([
            entry / sum(eff_dict[binsize_array_spanet]) for entry in eff_dict[binsize_array_spanet]
        ])
        density_run2 = ak.Array([
            entry / sum(eff_dict[binsize_array_run2]) for entry in eff_dict[binsize_array_run2]
        ])

        # for idx, array_name in enumerate(samplearrays):
        #     dens = density_spanet if "SPANet" in array_name else density_run2
        #     ax_relative.step(
        #         bins,
        #         eff_dict[array_name]*dens,
        #         where="mid",
        #         label=array_name,
        #         color=colors[2*idx + 1],
        #     )
        #     ax_relative.fill_between(
        #         bins,
        #         eff_dict[array_name]*dens,
        #         color=colors[2*idx + 1],
        #         step="mid",
        #         alpha=0.5,
        #     )
        if "total" not in plotname:
            # for idx, (density, binsize_array) in enumerate(zip([density_spanet, density_run2], [binsize_array_spanet, binsize_array_run2])):
            for idx, (density, binsize_array) in enumerate(zip([binsize_array_spanet, binsize_array_run2], [binsize_array_spanet, binsize_array_run2])):
                ax_relative.step(
                    bins,
                    eff_dict[density],
                    where="mid",
                    label=binsize_array,
                    color=colors[2*idx],
                )
        else:
            # for idx, (density, binsize_array) in enumerate(zip([density_spanet, density_run2], [binsize_array_spanet, binsize_array_run2])):
            for idx, (density, binsize_array) in enumerate(zip([binsize_array_spanet, binsize_array_run2], [binsize_array_spanet, binsize_array_run2])):
                ax_relative.step(
                    bins,
                    eff_dict[density],
                    where="mid",
                    label=binsize_array,
                    color=colors[2*idx],
                )
                ax_relative.step(
                    bins,
                    ak.Array(eff_dict[density])*ak.Array(eff_dict[matchedarrays[idx]]),
                    where="mid",
                    label=matchedarrays[idx],
                    color=colors[idx+4],
                )

        ax_relative.legend(loc="upper right")
        ax_relative.set_yscale("linear")
        hep.cms.lumitext(
            f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)",
            ax=ax_relative,
        )
        ax_relative.set_xlabel("DNN Score")
        ax_relative.set_ylabel(label)
        ax_relative.grid()
        fig_relative.show()
        fig_relative.savefig(
            os.path.join(args.output, f"{cat}_{plotname}_relative.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig_relative)
