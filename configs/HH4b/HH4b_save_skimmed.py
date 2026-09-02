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
# year = ["2022_postEE", "2022_preEE", "2023_preBPix", "2023_postBPix", "2024"]
year = ["2024"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/../HH4b_common/params/triggers_skim.yaml",
    update=True,
)

vbf_hh4b = [
    "VBFHHto4B_CV_1_C2V_0_C3_1",
    "VBFHHto4B_CV_1_C2V_1_C3_1",
    "VBFHHto4B_CV_1p74_C2V_1p37_C3_14p4",
    "VBFHHto4B_CV_m2p12_C2V_3p87_C3_m5p96", # not in 2024
    "VBFHHto4B_CV_2p12_C2V_3p87_C3_m5p96", # only in 2024
    "VBFHHto4B_CV_m0p012_C2V_0p030_C3_10p2",
    "VBFHHto4B_CV_m0p758_C2V_1p44_C3_m19p3",
    "VBFHHto4B_CV_m0p962_C2V_0p959_C3_m1p43",
    "VBFHHto4B_CV_m1p21_C2V_1p94_C3_m0p94",
    "VBFHHto4B_CV_m1p60_C2V_2p72_C3_m1p36",
    "VBFHHto4B_CV_m1p83_C2V_3p57_C3_m3p39",
]


ggf_hh4b = [
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
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/mmalucch/HH4b/skimmed_files/signal_ggF_HH4b_skimmed",
    save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/mmalucch/HH4b/skimmed_files/signal_VBF_HH4b_skimmed",
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
            f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b_pnfs_redirector.json",
        ],
        "filter": {
            "samples": (
                [
                    # "DATA_JetMET",
                    # "DATA_JetMET0_HH4bResolved",
                    # "DATA_JetMET1_HH4bResolved",
                    # "DATA_ParkingHH",
                ]
                + ggf_hh4b
                # + vbf_hh4b
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
