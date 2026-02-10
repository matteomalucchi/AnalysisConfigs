import os
from omegaconf import DictConfig

from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.lib.cut_functions import get_HLTsel

from pocket_coffea.parameters import defaults
import pocket_coffea.lib.calibrators.legacy.legacy_calibrators as legacy_cal
from pocket_coffea.lib.calibrators.common.common import JetsCalibrator
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
    # get_L1sel,
    goldenJson,
    eventFlags,
    get_nPVgood,
)

from configs.MET_studies.workflow import METProcessor
import custom_cuts as cuts
from output_quantities import get_met_columns, get_met_variables

# Define the saving method
SAVE_COLUMNS = True
DUMP_COLUMNS_AS_ARRAYS_PER_CHUNK = True
SAVE_HISTOGRAMS = False

if SAVE_HISTOGRAMS and (SAVE_COLUMNS or DUMP_COLUMNS_AS_ARRAYS_PER_CHUNK):
    raise ValueError("You can either save histograms or columns, not both.")
if not SAVE_HISTOGRAMS and not SAVE_COLUMNS:
    raise ValueError("You have to save something, either histograms or columns.")

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")


# adding object preselection
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    # DictConfig({}),
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/triggers.yaml",
    # f"{localdir}/params/jets_calibration_legacy_type1met.yaml",
    f"{localdir}/params/jets_calibration_regression_json_noMETcorr.yaml",
    update=True,
)


### Configuring the MET studies config ###
year = "2023_postBPix"
# dataset = "DYJetsToLL_M-50"
# dataset = "DYto2L-4Jets_MLL-50-v12"
dataset = "DYto2L-4Jets_MLL-50-v15"
option = "option_2"
add_str = ""
output_chunks_name = (
    # f"/scratch/mmalucch/out_MET/out_{option}_{dataset}_{year}{add_str}/parquet_files"
    f"root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/mmalucch/out_MET/out_{option}_{dataset}_{year}{add_str}/parquet_files"
)
print("Output chunks path:", output_chunks_name)
##########################################


common_cats = {
    "baseline": [passthrough],
}

# Define the columns to save
met_cols = get_met_columns()

# Define the variables to save
met_vars = get_met_variables()


cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/DYJetsToLL_M-50_pnfs_redirector.json",
            f"{localdir}/datasets/DYto2L-4Jets_MLL-50_pnfs_redirector.json",
            # f"{localdir}/datasets/DYJetsToLL_M-50_redirector.json",
        ],
        "filter": {
            "samples": [
                (
                    # dataset
                    # "DYto2L-4Jets_MLL-50-v12"
                    "DYto2L-4Jets_MLL-50-v15"
                    # "DYJetsToLL_M-50"
                    # "DYJetsToLL_M-50_local"
                )
            ],
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {},
    },
    workflow=METProcessor,
    workflow_options={
        # "donotscale_sumgenweights": True,
        "only_physical_jet": True,
        "rescale_MET_with_regressed_pT": True,
        "jec_pt_threshold": 15.0,
        "consider_all_jets": True,
        "add_low_pt_jets": True,
        "jet_regressed_option": option,
        "dump_columns_as_arrays_per_chunk": (
            output_chunks_name if DUMP_COLUMNS_AS_ARRAYS_PER_CHUNK else ""
        ),
    },
    skim=[
        get_HLTsel(primaryDatasets=["SingleMuon"]),
        # eventFlags,
        # get_nPVgood(1),
        # goldenJson,
    ],
    preselections=[
        # cuts.custom_JetVetoMap,
        cuts.PV_presel,
        cuts.at_least_one_jet,
        cuts.dimuon_presel,
    ],
    categories={
        **common_cats,
    },
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    # calibrators=[legacy_cal.JetsCalibrator, legacy_cal.JetsPtRegressionCalibrator],
    calibrators=[JetsCalibrator],
    variations={
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory": {},
            },
            "bysample": {},
        },
        # "shape": {
        #     "common": {
        #         "inclusive": ["jet_calibration"],
        #         },
        #     }
    },
    variables=met_vars if SAVE_HISTOGRAMS else {},
    columns={
        "common": {
            "inclusive": (
                (
                    [
                        ColOut("ll", ["mass", "pt", "eta", "phi"]),
                        ColOut("GenMET", ["pt", "phi"]),
                    ]
                    + met_cols
                )
                if SAVE_COLUMNS
                else []
            ),
        },
        "bysample": {},
    },
)
