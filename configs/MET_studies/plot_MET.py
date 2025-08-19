import os
import sys
import re
from matplotlib import pyplot as plt
from coffea.util import load
from omegaconf import OmegaConf
import numpy as np
import awkward as ak
import mplhep as hep
import argparse

from utils.plot.get_columns_from_files import get_columns_from_files
from plot_config import var_dict, color_list, ranges, log_dict


hep.style.use("CMS")


parser = argparse.ArgumentParser(description="Plot MET distributions from coffea files")
parser.add_argument (
    "-i",
    "--input-dir",
    type=str,
    required=True,
    help="Input directory for data with coffea files",
)
parser.add_argument(
    "-o", "--output", type=str, help="Output directory", default=""
)

args = parser.parse_args()

outputdir = args.output if args.output else "plots_MET"
# Create output directory if it does not exist
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

def plot_from_columns(cat_col):
    for category, col_var in cat_col.items():

        for vars_name, var_list in var_dict.items():

            fig, (ax, ax_ratio) = plt.subplots(
                2,
                1,
                figsize=[13, 13],
                sharex=True,
                gridspec_kw={"height_ratios": [2.5, 1]},
            )
            for i, variable in enumerate(var_list):
                col_num = col_var[variable]
                if "phi" in variable:
                    print(variable,  col_num)
                col_den = col_var[var_list[0]]
                var_name = (
                    variable.split("_")[0]
                    if "MuonGood" not in variable
                    else f'{variable.split("_")[0]}_MinusMuons'
                )

                # range_4b = (np.min(col_den), np.max(col_den))
                range_4b = ranges[vars_name]
                mask_range4b = (col_den > range_4b[0]) & (col_den < range_4b[1])
                col_num = col_num[mask_range4b]
                col_den = col_den[mask_range4b]

                h_den, bins = np.histogram(col_den, bins=30, range=range_4b)

                bins_center = (bins[1:] + bins[:-1]) / 2
                # draw the ratio
                h_num, bins = np.histogram(col_num, bins=30, range=range_4b)

                ratio = h_num / h_den
                err_num = np.sqrt(h_num)
                err_den = np.sqrt(h_den)
                ratio_err = np.sqrt(
                    (err_num / h_den) ** 2 + (h_num * err_den / h_den**2) ** 2
                )

                if i == 0:
                    # ax.errorbar(
                    #     bins_center,
                    #     h_den,
                    #     yerr=np.sqrt(h_den),
                    #     label=var_name,
                    #     color=color_list[i],
                    #     fmt=".",
                    # )
                    # ax_ratio.axhline(y=1, linestyle="--", color="black")
                    # ax_ratio.fill_between(
                    #     bins_center,
                    #     1 - ratio_err,
                    #     1 + ratio_err,
                    #     color="grey",
                    #     alpha=0.5,
                    # )
                    
                    hep.histplot(
                        h_den,
                        bins=bins,
                        # yerr=True,
                        w2=h_den,
                        w2method="poisson",
                        histtype="errorbar",
                        label=var_name,
                        ax=ax,
                    )
                else:
                    ax.hist(
                        col_num,
                        bins=30,
                        histtype="step",
                        label=var_name,
                        color=color_list[i],
                        range=range_4b,
                    )
                    ax_ratio.errorbar(
                        bins_center,
                        ratio,
                        yerr=ratio_err,
                        fmt=".",
                        label=var_name,
                        color=color_list[i],
                    )

                del col_den, col_num

            ax.legend(loc="upper right")
            ax.set_yscale("log" if log_dict[vars_name] else "linear")
            hep.cms.lumitext(r"(13.6 TeV)", ax=ax)
            hep.cms.text(text="Simulation Preliminary", ax=ax)

            ax_ratio.set_xlabel(vars_name)
            ax.set_ylabel("Events")
            ax_ratio.set_ylabel("Reg./Std.")

            ax.grid()
            ax_ratio.grid()
            ax_ratio.set_ylim(0.5, 1.5)
            ax.set_ylim(
                top=(
                    1.3 * ax.get_ylim()[1]
                    if not log_dict[vars_name]
                    else ax.get_ylim()[1] ** (1.3)
                )
            )

            fig.savefig(
                os.path.join(outputdir, f"{category}_{variable}.png"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fig)



if __name__ == "__main__":
    
    inputfiles_data = [
        os.path.join(args.input_dir, file)
        for file in os.listdir(args.input_dir)
        if file.endswith(".coffea")
    ]
    
    cat_col, total_datasets_list = get_columns_from_files(inputfiles_data)
    print(f"Total datasets found: {total_datasets_list}")
    print(cat_col)
    plot_from_columns(cat_col)