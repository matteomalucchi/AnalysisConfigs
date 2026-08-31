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
year = ["2022_postEE", "2022_preEE", "2023_preBPix", "2023_postBPix", "2024"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/../HH4b_common/params/triggers_skim.yaml",
    update=True,
)

vbf_hh4b_2024 = [
    # samples for 2024
    "VBFHHto4B-CV-1-C2V-0-C3-1",
    "VBFHHto4B-CV-1-C2V-1-C3-1",
    "VBFHHto4B-CV-1p74-C2V-1p37-C3-14p4",
    "VBFHHto4B-CV-m2p12-C2V-3p87-C3-m5p96",
    "VBFHHto4B-CV-2p12-C2V-3p87-C3-m5p96",
    "VBFHHto4B-CV-m0p012-C2V-0p030-C3-10p2",
    "VBFHHto4B-CV-m0p758-C2V-1p44-C3-m19p3",
    "VBFHHto4B-CV-m0p962-C2V-0p959-C3-m1p43",
    "VBFHHto4B-CV-m1p21-C2V-1p94-C3-m0p94",
    "VBFHHto4B-CV-m1p60-C2V-2p72-C3-m1p36",
    "VBFHHto4B-CV-m1p83-C2V-3p57-C3-m3p39",
]


ggf_hh4b_2024 = [
    # samples for 2024
    "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00",
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00",
    "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00",
    "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00",
    # new samples for 2024
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10",
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p35",
    "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-1p00",
    "GluGlutoHHto4B_kl-m20p00_kt-1p00_c2-2p24",
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-3p00",
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-m2p00",
]

cfg = Configurator(
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_F_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_ParkingHH_2023_D_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/testing/DATA_JetMET_JMENano_C_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/bevila_t/PostDoc/HH4b/skimmed_files/ggHH_boosted_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/bevila_t/PostDoc/HH4b/skimmed_files/ttbar_boosted_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/bevila_t/PostDoc/HH4b/skimmed_files/vbf_boosted_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/mmalucch/HH4b/skimmed_files/DATA_ParkingHH_resolved_skimmed",
    save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/mmalucch/HH4b/skimmed_files/ggF_VBF_HH4b_skimmed",
    parameters=parameters,
    datasets={
        "jsons": [
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet_redirector.json",
            # e f"{localdir}/../HH4b_common/datasets/background_TTtoX_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_JetMET_pnfs_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH_pnfs_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b_redirector.json",
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_official_pnfs_redirector.json",
            f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b_pnfs_redirector_pnfs_redirector.json",
        ],
        "filter": {
            "samples": (
                [
                    # "DATA_JetMET",
                    # "DATA_JetMET0_HH4bResolved",
                    # "DATA_JetMET1_HH4bResolved",
                    # "DATA_ParkingHH",
                ]
                + ggf_hh4b_2024
                + vbf_hh4b_2024
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
