from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.lib.categorization import CartesianSelection, MultiCut

from pocket_coffea.parameters import defaults
import os

from configs.jme.workflow import *

from configs.jme.cuts import *
from configs.jme.custom_functions import *

from configs.jme.custom_cut_functions import *
from configs.jme.params.binning import *
from configs.jme.variables_def import get_variables_dict


localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")


year = os.environ.get("YEAR", "2022_preEE")
# adding object preselection
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection{'_extendedPT' if (int(os.environ.get('EXTENDED_PT_BINS', 0)) == 1) else ''}.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    update=True,
)

if int(os.environ.get("UPART", 0)) == 0:
    eta_bins = eta_bins
else:
    eta_bins = eta_bins_upart


mc_truth_corr_pnetreg = None
corr_files_pnetreg = {
    "2022_preEE": f"{localdir}/params/Summer22Run3_V3_MC_L2Relative_AK4PFPNet.txt",
    "2022_postEE": f"{localdir}/params/Summer22EERun3_V3_MC_L2Relative_AK4PFPNet.txt",
    "2023_preBPix": f"{localdir}/params/Summer23Run3_V3_MC_L2Relative_AK4PFPNet.txt",
    "2023_postBPix": f"{localdir}/params/Summer23BPixRun3_V3_MC_L2Relative_AK4PFPNet.txt",
    "2024": f"{localdir}/params/Summer24Run3_V3_MC_L2Relative_AK4PFPNet.txt",
    "2025": f"{localdir}/params/Winter25Run3_V3_MC_L2Relative_AK4PFPNet.txt",
}
if int(os.environ.get("CLOSURE", 0)) == 1:
    print(f"Performing closure test with {corr_files_pnetreg[year]}")
    mc_truth_corr_pnetreg = get_closure_function_information(corr_files_pnetreg[year])

mc_truth_corr_upartreg = None
corr_files_upartreg = {
    "2022_preEE": f"{localdir}/params/Summer22Run3_V3_MC_L2Relative_AK4PFUparT.txt",
    "2022_postEE": f"{localdir}/params/Summer22EERun3_V3_MC_L2Relative_AK4PFUparT.txt",
    "2023_preBPix": f"{localdir}/params/Summer23Run3_V3_MC_L2Relative_AK4PFUparT.txt",
    "2023_postBPix": f"{localdir}/params/Summer23BPixRun3_V3_MC_L2Relative_AK4PFUparT.txt",
    "2024": f"{localdir}/params/Summer24Run3_V3_MC_L2Relative_AK4PFUparT.txt",
    "2025": f"{localdir}/params/Winter25Run3_V3_MC_L2Relative_AK4PFUparT.txt",
}
if int(os.environ.get("CLOSURE", 0)) == 1:
    print(f"Performing closure test with {corr_files_upartreg[year]}")
    mc_truth_corr_upartreg = get_closure_function_information(corr_files_upartreg[year])

mc_truth_corr_pnetreg_neutrino = None
corr_files_pnetreg_neutrino = {
    "2022_preEE": f"{localdir}/params/Summer22Run3_V3_MC_L2Relative_AK4PFPNetPlusNeutrino.txt",
    "2022_postEE": f"{localdir}/params/Summer22EERun3_V3_MC_L2Relative_AK4PFPNetPlusNeutrino.txt",
    "2023_preBPix": f"{localdir}/params/Summer23Run3_V3_MC_L2Relative_AK4PFPNetPlusNeutrino.txt",
    "2023_postBPix": f"{localdir}/params/Summer23BPixRun3_V3_MC_L2Relative_AK4PFPNetPlusNeutrino.txt",
    "2024": f"{localdir}/params/Summer24Run3_V3_MC_L2Relative_AK4PFPNetPlusNeutrino.txt",
    "2025": f"{localdir}/params/Winter25Run3_V3_MC_L2Relative_AK4PFPNetPlusNeutrino.txt",
}
if int(os.environ.get("CLOSURE", 0)) == 1:
    print(f"Performing closure test with {corr_files_pnetreg_neutrino[year]}")
    mc_truth_corr_pnetreg_neutrino = get_closure_function_information(
        corr_files_pnetreg_neutrino[year]
    )

mc_truth_corr_upartreg_neutrino = None
corr_files_upartreg_neutrino = {
    "2022_preEE": f"{localdir}/params/Summer22Run3_V3_MC_L2Relative_AK4PFUparTPlusNeutrino.txt",
    "2022_postEE": f"{localdir}/params/Summer22EERun3_V3_MC_L2Relative_AK4PFUparTPlusNeutrino.txt",
    "2023_preBPix": f"{localdir}/params/Summer23Run3_V3_MC_L2Relative_AK4PFUparTPlusNeutrino.txt",
    "2023_postBPix": f"{localdir}/params/Summer23BPixRun3_V3_MC_L2Relative_AK4PFUparTPlusNeutrino.txt",
    "2024": f"{localdir}/params/Summer24Run3_V3_MC_L2Relative_AK4PFUparTPlusNeutrino.txt",
    "2025": f"{localdir}/params/Winter25Run3_V3_MC_L2Relative_AK4PFUparTPlusNeutrino.txt",
}
if int(os.environ.get("CLOSURE", 0)) == 1:
    print(f"Performing closure test with {corr_files_upartreg_neutrino[year]}")
    mc_truth_corr_upartreg_neutrino = get_closure_function_information(
        corr_files_upartreg_neutrino[year]
    )

mc_truth_corr = None
corr_files = {
    "2022_preEE": f"{localdir}/params/Summer22Run3_V1_MC_L2Relative_AK4PUPPI.txt",
    "2022_postEE": f"{localdir}/params/Summer22EEVetoRun3_V1_MC_L2Relative_AK4PUPPI.txt",
    "2023_preBPix": f"{localdir}/params/Summer23Run3_V1_MC_L2Relative_AK4PUPPI.txt",
    "2023_postBPix": f"{localdir}/params/Summer23BPixRun3_V3_MC_L2Relative_AK4PUPPI.txt",
    "2024": f"{localdir}/params/Summer24Prompt24_V1_MC_L2Relative_AK4PFPuppi.txt",
    "2025": f"{localdir}/params/Winter25Prompt25_V1_MC_L2Relative_AK4PFPuppi.txt",
}
print(f"Reapplying corrections {corr_files[year]}")
mc_truth_corr = get_closure_function_information(corr_files[year])

cuts_eta = []
cuts_names_eta = []
cuts_eta_neutrino = []
cuts_names_eta_neutrino = []
cuts_reco_eta = []
cuts_names_reco_eta = []

if int(os.environ.get("NEUTRINO", 1)) == 0:
    print("RECO JET ETA CUTS NEUTRINO==0")
    for i in range(len(eta_bins) - 1):
        eta_low, eta_high = eta_bins[i], eta_bins[i + 1]
        cuts_reco_eta.append(get_reco_etabin(eta_low, eta_high))
        cuts_names_reco_eta.append(f"MatchedJets_reco_eta{eta_low}to{eta_high}")
elif int(os.environ.get("NEUTRINO", 0)) == 1:
    print("RECO JET ETA CUTS NEUTRINO==1")
    for i in range(len(eta_bins) - 1):
        eta_low, eta_high = eta_bins[i], eta_bins[i + 1]
        cuts_reco_eta.append(get_reco_neutrino_etabin(eta_low, eta_high))
        cuts_names_reco_eta.append(f"MatchedJetsNeutrino_reco_eta{eta_low}to{eta_high}")
elif int(os.environ.get("ABS_ETA_INCLUSIVE", 0)) == 1:
    print("RUNNING ABS_ETA_INCLUSIVE ETA BINS")
    for i in range(len(eta_bins) - 1):
        eta_low, eta_high = eta_bins[i], eta_bins[i + 1]
        cuts_reco_eta.append(get_reco_neutrino_abs_etabin(eta_low, eta_high))
        cuts_names_reco_eta.append(
            f"MatchedJetsNeutrino_reco_abseta{eta_low}to{eta_high}"
        )
else:
    print("RECO JET ETA CUTS")
    for i in range(len(eta_bins) - 1):
        eta_low, eta_high = eta_bins[i], eta_bins[i + 1]
        cuts_reco_eta.append(get_reco_neutrino_etabin(eta_low, eta_high))
        cuts_names_reco_eta.append(f"MatchedJetsNeutrino_reco_eta{eta_low}to{eta_high}")


variables_dict=get_variables_dict(cuts_names_eta, cuts_names_reco_eta, cuts_names_eta_neutrino)

samples_dict = {
    "2022_preEE": "QCD_PT-15to7000_JMENano_Summer22",
    "2022_postEE": "QCD_PT-15to7000_JMENano_Summer22EE",
    "2023_preBPix": "QCD_PT-15to7000_JMENano_Summer23",
    "2023_postBPix": "QCD_PT-15to7000_JMENano_Summer23BPix",
    "2024" : "QCD_PT-15to7000_JMENano_Summer24",
    "2025" : "QCD_PT-15to7000_JMENano_Winter25",
}
samples_PNetReg15_dict = {
    "2022_preEE": "QCD_PT-15to7000_PNetReg15_JMENano_Summer22",
    "2022_postEE": "QCD_PT-15to7000_PNetReg15_JMENano_Summer22EE",
    "2023_preBPix": "QCD_PT-15to7000_PNetReg15_JMENano_Summer23",
    "2023_postBPix": "QCD_PT-15to7000_PNetReg15_JMENano_Summer23BPix",
}
# No UparTReg15 used for now but may be added later if necessary

multicuts = [
    MultiCut(
        name="eta",
        cuts=cuts_eta + cuts_eta_neutrino + cuts_reco_eta,
        cuts_names=cuts_names_eta + cuts_names_eta_neutrino + cuts_names_reco_eta,
    ),
]

common_cats = {
    # "baseline": [passthrough],
    "PV_presel": [PV_presel],
}

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            #f"{localdir}/datasets/QCD.json",
            f"{localdir}/datasets/QCD_redirector.json",
            f"{localdir}/datasets/QCD_PNetReg15.json",
        ],
        "filter": {
            "samples": [
                (
                    samples_PNetReg15_dict[year]
                    if (
                        int(os.environ.get("PNETREG15", 0)) == 1
                        or int(os.environ.get("SPLITPNETREG15", 0)) == 1
                    )
                    else samples_dict[year]
                )
            ],
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {},
    },
    workflow=QCDBaseProcessor,
    workflow_options={
        "donotscale_sumgenweights": True,
        "mc_truth_corr_pnetreg": mc_truth_corr_pnetreg,
        "mc_truth_corr_pnetreg_neutrino": mc_truth_corr_pnetreg_neutrino,
        "mc_truth_corr_upartreg": mc_truth_corr_upartreg,
        "mc_truth_corr_upartreg_neutrino": mc_truth_corr_upartreg_neutrino,
        "mc_truth_corr": mc_truth_corr,
        "DeltaR_matching": 0.2,
        "SetRegResponseToZero": True,
        "GenJetPtCut": (
            15
            if (
                int(os.environ.get("PNETREG15", 0)) == 1
                or int(os.environ.get("SPLITPNETREG15", 0)) == 1
            )
            else 
            (0 
            if (
                int(os.environ.get("EXTENDED_PT_BINS", 0)) == 1
            )
            else 50
            )
        ),
    },
    skim=[],
    preselections=[],
    categories=CartesianSelection(multicuts=multicuts, common_cats=common_cats),
    # categories={
    #             **common_cats,
    #     # **eta_cuts,
    # },
    weights={
        "common": {
            "inclusive": [
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations={
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory": {},
            },
            "bysample": {},
        }
    },
    variables=variables_dict,
    columns={
    },
)
