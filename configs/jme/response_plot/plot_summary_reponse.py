import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import sys
import argparse


hep.style.use("CMS")


sys.path.append("../")
from params.binning import *


parser = argparse.ArgumentParser(description="Run the jme analysis")
parser.add_argument(
    "-d",
    "--dir",
    type=str,
    help="Input dir",
)
parser.add_argument("-f", "--flav", nargs="+", default=["inclusive"], help="Flavour")
parser.add_argument(
    "-t",
    "--type",
    nargs="+",
    help="Type of plot (jec, reg, neutrino)",
    default=["reg", "neutrino"],
)
parser.add_argument(
    "--upart",
    help="Use UparT Regression instead of PNet one",
    action="store_true",
    default=False,
)
args = parser.parse_args()

pt_bins = pt_bins_all if "pnetreg15" in args.dir else (pt_bins_extended if "extendedPT" in args.dir else pt_bins_reduced)
type_plot_dict = {
    "jec": "ResponseJEC",
    "reg": f"Response{'UparT' if args.upart else 'PNet'}Reg",
    "neutrino": f"Response{'UparT' if args.upart else 'PNet'}RegNeutrino",
}

label_dict = {
    "jec": "JEC",
    "reg": f"{'UparT' if args.upart else 'PNet'}",
    "neutrino": f"{'UparT' if args.upart else 'PNet'} incl. neutrinos",
}

tot_string = "Tot" if "splitpnetreg15" in args.dir else ""

if args.upart:
    inclusive_bins = inclusive_bins_upart

for type_string in args.type:
    for flav in args.flav:
        type_plot = type_plot_dict[type_string] + tot_string

        name_file = f"{args.dir}/median_plots_binned/medians_absinclusive_{flav}_{type_plot}.npy"
        print(name_file)
        # load npy file
        data = np.load(
            name_file,
            allow_pickle=True,
        )

        err_name_file = f"{args.dir}/median_plots_binned/err_medians_absinclusive_{flav}_{type_plot}.npy"
        print(err_name_file)
        err_data = np.load(
            err_name_file,
            allow_pickle=True,
        )

        markers = ["o", "*", "^", "s", "v"]

        fig, ax = plt.subplots()

        for eta_bin in range(len(data)):
            ax.errorbar(
                x=pt_bins[:-1],
                y=data[eta_bin],
                yerr=err_data[eta_bin],
                label=f"{inclusive_bins[eta_bin]:.1f} <"
                + r"|$\eta^{reco}$|"
                + f" < {inclusive_bins[eta_bin+1]:.1f}",
                marker=markers[eta_bin],
                markersize=8,
                linestyle="None",
            )

        ax.legend(loc="upper right", frameon=False)
        ax.set_ylabel("Median jet response", loc="top")
        ax.set_xlabel(r"$p_{T}^{ptcl}$ (GeV)", loc="right")

        #ax.set_ylim(top=1.05 * np.nanmax(data), bottom=0.99 * np.nanmin(data))
        ax.set_ylim(top=1.08, bottom=0.92)

        ax.axhline(y=1, color="black", linestyle="--", linewidth=0.7)
        # add +- 1% lines
        ax.axhline(y=1.01, color="black", linestyle="--", linewidth=0.5)
        ax.axhline(y=0.99, color="black", linestyle="--", linewidth=0.5)

        # add +- 0.1% lines
        ax.axhline(y=1.001, color="black", linestyle=":", linewidth=0.5)
        ax.axhline(y=0.999, color="black", linestyle=":", linewidth=0.5)

        ax.set_xscale("log")

        if "2016_PreVFP" in args.dir:
            hep.cms.lumitext("2016_PreVFP (13 TeV)")
        elif "2016_PostVFP" in args.dir:
            hep.cms.lumitext("2016_PostVFP (13 TeV)")
        elif "2017" in args.dir:
            hep.cms.lumitext("2017 (13 TeV)")
        elif "2018" in args.dir:
            hep.cms.lumitext("2018 (13 TeV)")
        elif "2022_preEE" in args.dir:
            hep.cms.lumitext("2022_preEE (13.6 TeV)")
        elif "2022_postEE" in args.dir:
            hep.cms.lumitext("2022_postEE (13.6 TeV)")
        elif "2023_preBPix" in args.dir:
            hep.cms.lumitext("2023_preBPix (13.6 TeV)")
        elif "2023_postBPix" in args.dir:
            hep.cms.lumitext("2023_postBPix (13.6 TeV)")
        elif "2024" in args.dir:
            hep.cms.lumitext("2024 (13.6 TeV)")
        elif "2025" in args.dir:
            hep.cms.lumitext("2025 (13.6 TeV)")
        else:
            raise ValueError("Year string not found in directory name")


        hep.cms.text(
            text="Simulation\nPreliminary",
            loc=2,
        )
        ax.text(
            0.05,
            0.75,
            r"anti-$k_{T}$ R=0.4 (PUPPI)",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )
        ax.text(
            0.05,
            0.7,
            label_dict[type_string] + f" ({flav})",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        fig.savefig(
            f"{args.dir}/median_plots_binned/summary_{flav}_median_{type_plot}.png",
            bbox_inches="tight",
            dpi=300,
        )
        # fig.savefig(
        #     f"{args.dir}/median_plots_binned/summary_{flav}_median_{type_plot}.pdf",
        #     bbox_inches="tight",
        #     dpi=300,
        # )
