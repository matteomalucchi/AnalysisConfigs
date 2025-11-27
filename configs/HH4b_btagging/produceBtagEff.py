# np.seterr(divide='ignore', invalid='ignore')
import os
import json
import yaml
import logging
import argparse
import boost_histogram as bh

import correctionlib
import correctionlib.convert
import hist
import matplotlib.pyplot as plt
import numpy as np
from coffea.util import load
from matplotlib.ticker import MultipleLocator


def plotBtagEffiControl(sampleGroups, bJetHistNames, histos, bTaggingAlgorithm, outputpath):
    histo = histos[0]
    markers = ["1", "P", "X", "o", "*"]
    flavors = ["light-jets", "c-jets", "b-jets"]
    colors = ["#f89c20", "#e42536", "#5790fc"]
    sampGroupNames = ["HH4b"]
    for flav in range(len(histo.axes[2].centers)):
        fig = plt.figure(figsize=(14.5, 14))
        for i in range(len(histo.axes[1].edges) - 1):
            ax = plt.subplot(4, 3, i + 1)
            axTitle = r"$|\eta^{jet}|$ $\in$ " + "$[${},{}$]$".format(
                histo.axes[1].edges[i],
                histo.axes[1].edges[i + 1]
            )
            ax.set_title(axTitle)
            for j in range(len(sampleGroups)):
                for wp, bJetHistName in enumerate(bJetHistNames):
                    workingPoint = bJetHistName.split("_")[2]
                    histoId = j * len(bJetHistNames) + wp
                    histo = histos[histoId]
                    bWidth = [(histo.axes[0].edges[i + 1] - histo.axes[0].edges[i]) / 2 for i in range(len(histo.axes[0].edges) - 1)]
                    label = sampGroupNames[j] + ", wp {}".format(workingPoint) if (i == 0) else None
                    ax.set_ylim((0, 1.))
                    ax.set_xlim((15., 500.))
                    ax.errorbar(
                        histo.axes[0].centers,
                        histo.values()[:, i, flav],
                        xerr=bWidth,
                        label=label,
                        marker=markers[wp],
                        markersize=10,
                        fillstyle="none",
                        color=colors[j],
                    )
                    ax.xaxis.set_minor_locator(MultipleLocator(20))
                    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
                    ax.tick_params(direction='in', which="both")
                    ax.set_ylabel("Efficiency")
                    ax.set_xlabel(r"$p_T^{jet}$ [GeV]")
        fig.legend(bbox_to_anchor=(0.62, 0.25), fontsize=16)
        flavText = "Jet flavor: {}\n\nBtagger: {}".format(flavors[flav], bTaggingAlgorithm)
        processText = "Process groups:\n  Top: $t\\overline{t}$, ttH(bb), ttZ, ttW, single t\n"\
            + "  VV: WW, WZ, ZZ\n  V+Jets: DY, W+Jets"
        flavTextProps = dict(boxstyle='round', facecolor="none", pad=0.6)
        fig.text(0.7, 0.17, flavText, fontsize=16, bbox=flavTextProps)
        fig.text(0.7, 0.05, processText, fontsize=16)
        plt.tight_layout(pad=0.7)
        plt.savefig(outputpath + "btaggingEfficiency_{}_{}.pdf".format(bTaggingAlgorithm, flavors[flav]))


def produceBtagEfficiencies(outputpath, inputfile, histCategoryToUse, sampleGroups, bjetHistNames, jetHistName, btaggingAlgorithm, produceControlPlots=False):

    histCollection = bjetHistNames + jetHistName

    # Reading in config and parameters to obtain year
    inputFile = load(inputfile)
    year = list(inputFile["datasets_metadata"]['by_datataking_period'])
    # print("Producing btagging sfs for " + year[0])
    # Reading in jet Histogramms

    samples_found = {k: True for k in sampleGroups.keys()}
    for jetHistVar in histCollection:  # different jet collections (btaggedJets for each wp, jetsTotal)
        for sampleGroup in sampleGroups.keys():  # groupings of samples to calculated efficiencies in
            tempHistColl = []
            if "hists" not in sampleGroups[sampleGroup].keys():
                sampleGroups[sampleGroup]["hists"] = {}
            for sampleName in sampleGroups[sampleGroup]["sampleNames"]:
                if sampleName not in inputFile["variables"][jetHistVar].keys() or not samples_found[sampleGroup]:
                    samples_found[sampleGroup] = False
                    continue
                for dataset, inputHist in inputFile["variables"][jetHistVar][sampleName].items():
                    nominalHist = inputHist[hist.loc(histCategoryToUse), hist.loc("nominal"), :, :, :]
                    tempHistColl.append(nominalHist) # * inputFile["sum_genweights"][dataset])
            if samples_found:
                sampleGroups[sampleGroup]["hists"][jetHistVar] = sum(tempHistColl)  # sum samples in each sample group
    for sampleGroup, found in samples_found.items():
        if found:
            logger.info(f"Found all samples for group {sampleGroup} for algorithm {btaggingAlgorithm}")

    # Calculating Efficiency and creating correction sets

    correction_set_collection = []
    if produceControlPlots:
        effiHistos = []
    for sampleGroup in [group for group, found in samples_found.items() if found]:
        nJetsTotalHist = sampleGroups[sampleGroup]["hists"][jetHistName[0]]

        for bJetHistName in bjetHistNames:
            nbJetsHist = sampleGroups[sampleGroup]["hists"][bJetHistName]
            check_flow(nbJetsHist)
            workingPoint = bJetHistName.split("_")[2]

            # Calculating btaggin efficiency
            btagEfficiencyArray = np.divide(
                nbJetsHist.values(),
                nJetsTotalHist.values(),
                out=np.zeros_like(nbJetsHist.values(), dtype=float),
                where=nJetsTotalHist.values() != 0,
            )
            btagEfficiencyHisto = hist.Hist(
                *nJetsTotalHist.axes,
                data=btagEfficiencyArray,
                name=sampleGroup + "_wp_" + workingPoint,
                label="btag_efficiency"
            )
            if produceControlPlots:
                effiHistos.append(btagEfficiencyHisto)
            # Creating Correctionset
            view = btagEfficiencyHisto.view(flow=True)
            view[...] = np.nan_to_num(view, nan=0.0)
            cset = correctionlib.convert.from_histogram(btagEfficiencyHisto)
            cset.description = "Btagging efficiencies for " + sampleGroup + " samples using " + workingPoint + " working point"
            # For the moment let correctionlib fail evaluation in case a jet is in the under-/overflow bin of the efficiency map.
            cset.data.flow = "error"
            correction_set_collection.append(cset)

    if produceControlPlots:
        plotBtagEffiControl([group for group, found in samples_found.items() if found], bjetHistNames, effiHistos, btaggingAlgorithm, outputpath)

    btag_effi_correction_set = correctionlib.schemav2.CorrectionSet(
        schema_version=2,
        description="btagging efficiencies",
        corrections=correction_set_collection,
    )

    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    # Creating output file
    parsed_json = json.loads(btag_effi_correction_set.model_dump_json(exclude_unset=True))
    with open(f"{outputpath}/btag_efficiencies_" + btaggingAlgorithm + "_" + year[0] + ".json", "w") as fout:
        fout.write(json.dumps(parsed_json, indent=2))


def check_flow(hist):
    """Check if there are entries in over-/ underflow bin of Hist() object and raise exception if so."""
    for ax in range(hist.ndim):
        if np.any(hist[{ax: bh.underflow}].view(flow=True).value):
            raise ValueError(
                    "No entries in under-/ overflows allowed \n" +
                    f"Found underflow bin in axis {hist.axes[ax].name}. Values are: {hist[{ax: bh.underflow}].view(flow=True).value}"
                    )
        elif np.any(hist[{ax: bh.overflow}].view(flow=True).value):
            raise ValueError(
                    "No entries in under-/ overflows allowed \n" +
                    f"Found overflow bin in axis {hist.axes[ax].name}. Values are: {hist[{ax: bh.overflow}].view(flow=True).value}"
                    )


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description="Produce btagWP efficiency files for SF correction.")
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Input coffea file (needs to be single file currently)",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output directory", default="./"
    )
    parser.add_argument(
        "-c", "--category", type=str, help="Category to calculate efficiency in", default="inclusive"
    )
    parser.add_argument(
        "-g", "--groups", type=str, help="YAML file containing information about sample group names", default=f"{os.path.dirname(os.path.realpath(__file__))}/../HH4b_common/params/btagging_sampleGroups.yaml"
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Produce control plots",
        default=False,
    )
    args = parser.parse_args()

    with open(args.groups, 'r') as file:
        sampleGroups = yaml.safe_load(file)["btagging"]["sampleGroups"]

    # Collection of histograms output with config_BtagEff_fixed_wp

    bjetHist_dictionary = {
        "btagDeepFlavB": [
            "bjets_deepJet_L_pt_eta_flav",
            "bjets_deepJet_M_pt_eta_flav",
            "bjets_deepJet_T_pt_eta_flav",
            "bjets_deepJet_XT_pt_eta_flav",
            "bjets_deepJet_XXT_pt_eta_flav"
            ],
        "btagPNetB": [
            "bjets_particleNet_L_pt_eta_flav",
            "bjets_particleNet_M_pt_eta_flav",
            "bjets_particleNet_T_pt_eta_flav",
            "bjets_particleNet_XT_pt_eta_flav",
            "bjets_particleNet_XXT_pt_eta_flav"
            ],
        "btagRobustParTAK4B": [
            "bjets_robustParticleTransformer_L_pt_eta_flav",
            "bjets_robustParticleTransformer_M_pt_eta_flav",
            "bjets_robustParticleTransformer_T_pt_eta_flav",
            "bjets_robustParticleTransformer_XT_pt_eta_flav",
            "bjets_robustParticleTransformer_XXT_pt_eta_flav"
        ]
    }
    jetHistName = ["jets_pt_eta_flav"]

    for btaggingAlgorithm, bjetHistNames in bjetHist_dictionary.items():
        produceBtagEfficiencies(
            args.output,
            args.input_file,
            args.category,
            sampleGroups,
            bjetHistNames,
            jetHistName,
            btaggingAlgorithm,
            args.plot,
        )
