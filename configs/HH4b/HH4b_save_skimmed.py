import os


from configs.HH4b_common.config_files.__config_file__ import (
    config_options_dict,
)

# from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters import defaults
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_HLTsel
from workflow_dummy import HH4bbQuarkMatchingProcessorDummy
import configs.HH4b_common.custom_cuts_common as cuts

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = ["2023_preBPix", "2023_postBPix", "2024"]
# year = ["2022_postEE", "2022_preEE", "2023_preBPix", "2023_postBPix"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/../HH4b_common/params/triggers_skim.yaml",
    update=True,
)

cfg = Configurator(
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_skimmed",
    #    save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_F_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_ParkingHH_2023_D_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/testing/DATA_JetMET_JMENano_C_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/bevila_t/PostDoc/HH4b/skimmed_files/ggHH_boosted_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/bevila_t/PostDoc/HH4b/skimmed_files/ttbar_boosted_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/bevila_t/PostDoc/HH4b/skimmed_files/vbf_boosted_skimmed",
    save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/mmalucch/HH4b/skimmed_files/DATA_ParkingHH_resolved_skimmed",
    parameters=parameters,
    datasets={
        "jsons": [
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet_redirector.json",
            #e f"{localdir}/../HH4b_common/datasets/background_TTtoX_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_JetMET_pnfs_redirector.json",
            f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH_pnfs_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b_redirector.json",
        ],
        "filter": {
            "samples": (
                [
                    # "DATA_JetMET",
                    # "DATA_JetMET0_HH4bResolved",
                    # "DATA_JetMET1_HH4bResolved",
                    "DATA_ParkingHH",
                ]
            ),
            "samples_exclude": [],
            "year": year,
        },
        "subsamples": {},
    },
    workflow=HH4bbQuarkMatchingProcessorDummy,
    workflow_options={},
    # skim=cuts.skimming_cut_list,
    # skim=cuts.skimming_cut_list(config_options_dict),
    skim=[
        get_HLTsel(),
    ],
    preselections=[
        #
    ],
    categories={
        #
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
    variations={
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory": {},
            },
            "bysample": {},
        }
    },
    variables={
        #
    },
    columns={
        #
    },
)
