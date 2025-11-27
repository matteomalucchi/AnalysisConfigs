# load a coffea file
import argparse
import logging
import os
from utils.plot.get_era_lumi import get_era_lumi
import numpy as np

import matplotlib
from coffea.util import load
from utils.plot.HEPPlotter import HEPPlotter

matplotlib.rcParams["figure.dpi"] = 300


def get_var(coffea_file, sample, datasets, variable, variation="nominal"):
    """Get variable for a sample, summing over all datasets of that sample."""
    logger.debug(f"Loading histogram for sample {sample}, datasets {datasets} in variable {variable}")
    hist_nominal = []
    hist_sf_btag = []
    for dset in datasets:
        hist_nominal.append(coffea_file["variables"][variable][sample][dset][{"cat": "inclusive", "variation": variation}])
        hist_sf_btag.append(coffea_file["variables"][variable][sample][dset][{"cat": "inclusive_sf_btag", "variation": variation}])
    return sum(hist_nominal), sum(hist_sf_btag)


def compare_bin_by_bin(hist_collection, sample_datasets, output):
    """Compare the nominal vs. SF histograms bin-by-bin."""
    logger.info("Comparing histograms bin-by-bin")
    for sample, variables in hist_collection.items():
        lumi, era_string = get_era_lumi(sample_datasets[sample])
        for variable, hist_nom_sf in variables.items():
            logger.info(f"Comparing variable {variable} ratio btag_sf / nominal:")
            hist_ratio = np.divide(hist_nom_sf[1].values(), hist_nom_sf[0].values(), out=np.zeros_like(hist_nom_sf[0].values(), dtype=float), where=hist_nom_sf[0].values() != 0)
            nan_mask = (hist_nom_sf[1].values() == 0) & (hist_nom_sf[1].values() == 0)
            hist_ratio[nan_mask] = np.nan
            os.makedirs(f"{output}/{sample}", exist_ok=True)
            hist_dict = {"nominal": {"data": hist_nom_sf[0], "style": {"is_reference": True, "label": "nominal"}}, "btag_sf": {"data": hist_nom_sf[1], "style": {"label": "btag_sf applied"}}}
            if all(hist.ndim == 1 for hist in hist_nom_sf):
                logger.info(", ".join(f"{bin:.4f}" for bin in hist_ratio))
                logger.info(f"maximal ratio: {max(hist_ratio[~nan_mask]):.4f}, \t minimal ratio: {min(hist_ratio[~nan_mask]):.4f}")
                (
                    HEPPlotter()
                    .set_plot_config(
                    lumitext=f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)",
                        figsize=[13, 13],
                    )
                    .set_output(f"{output}/{sample}/{variable}_comparison")
                    .set_labels(
                        f"{variable}",
                        "Events",
                        ratio_label="with btag-sf / no btag-sf",
                    )
                    .set_options(y_log=True, x_log=False, set_ylim=False, legend=True)
                    .set_data(hist_dict, plot_type="1d").run()
                )
            elif all(hist.ndim == 2 for hist in hist_nom_sf):
                # hist_ratio = hist_nom_sf[1].values() / hist_nom_sf[0].values()
                for row in hist_ratio:
                    logger.info(", ".join(f"{bin:.4f}" for bin in row))
                logger.info(f"maximal ratio: {np.max(hist_ratio[~nan_mask]):.4f}, \t minimal ratio: {np.min(hist_ratio[~nan_mask]):.4f}")
                for njets in range(6):
                    hist_dict = {"nominal": {"data": hist_nom_sf[0][{"events.nJetGood": njets}], "style": {"is_reference": True, "label": "nominal"}}, "btag_sf": {"data": hist_nom_sf[1][{"events.nJetGood": njets}], "style": {"label": "btag_sf applied"}}}
                    (
                        HEPPlotter()
                        .set_plot_config(
                        lumitext=f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)",
                            figsize=[13, 13],
                        )
                        .set_output(f"{output}/{sample}/{variable}_events_with_{njets}_Jets_comparison")
                        .set_labels(
                            f"{hist_nom_sf[0].axes[1].name}",
                            f"Events with {njets} Jets"
                            # ratio_label="with btag-sf / no btag-sf",
                        )
                        .set_options(y_log=True, x_log=False, set_ylim=False, legend=True)
                        .set_data(hist_dict, plot_type="1d").run()
                    )

            else:
                raise ValueError(f"Dimension of histograms either >2 or not the same")


def compare_histograms(coffea_file, sample_datasets, output):
    """Compare the sum over nominal vs. SF histograms as simple check.

    After comparison, call script to compare bin-by-bin and produce plots.
    """
    hist_collection = {}
    for sample, datasets in sample_datasets.items():
        hist_collection[sample] = {}
        logger.info(f"Comparing histogram sums for sample: {sample}")
        logger.info("variable \t nominal sum \t btag_sf sum \t ratio")
        for variable in coffea_file["variables"].keys():
            hist_nom, hist_sf = get_var(coffea_file, sample, datasets, variable)
            hist_collection[sample][variable] = [hist_nom, hist_sf]
            nom_sum = hist_nom.sum(flow=True).value
            sf_sum = hist_sf.sum(flow=True).value
            diff = (sf_sum / nom_sum - 1) * 100
            logger.info(f"{variable} \t {nom_sum:.4f} \t {sf_sum:.4f} \t {diff:.4f} %")
    compare_bin_by_bin(hist_collection, sample_datasets, output)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Compare normalisation with and without btagSF")
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Input coffea file (needs to be single file)",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output directory", default="./btag_sf_comparison"
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Produce plots",
        default=False,
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        filename=f"{args.output}/logger_output.log")
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    coffea_file = load(args.input_file)

    datasets = coffea_file["datasets_metadata"]["by_dataset"].keys()
    sample_datasets = {}
    for dset in datasets:
        sample = coffea_file["datasets_metadata"]["by_dataset"][dset]["sample"]
        if sample in sample_datasets.keys():
            sample_datasets[sample].append(dset)
        else:
            sample_datasets[sample] = [dset]

    compare_histograms(coffea_file, sample_datasets, args.output)
