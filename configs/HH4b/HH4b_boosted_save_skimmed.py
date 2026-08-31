import os


from configs.HH4b_common.config_files.__config_file__ import (
    config_options_dict,
)
from pocket_coffea.lib.cut_functions import get_HLTsel
# from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters import defaults
from pocket_coffea.utils.configurator import Configurator
from workflow_dummy import HH4bbQuarkMatchingProcessorDummy
import configs.HH4b_common.custom_cuts_common as cuts

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = ["2022_postEE", "2022_preEE", "2023_preBPix", "2023_postBPix"]
# year = ["2024"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/../HH4b_common/params/object_preselection_{config_options_dict['approach']}_approach.yaml",
    f"{localdir}/../HH4b_common/params/triggers_boosted_skim.yaml",
    update=True,
)

cfg = Configurator(
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_skimmed",
#    save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_F_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_ParkingHH_2023_D_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/testing/DATA_JetMET_JMENano_C_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/bevila_t/PostDoc/HH4b/skimmed_files/ggHH_boosted_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/bevila_t/PostDoc/HH4b/skimmed_files/ttbar_boosted_skimmed",
    save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/skimmed_files/vbf_boosted_skimmed",
    parameters=parameters,
    datasets={
        "jsons": [
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet.json",
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_official.json",
            f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet_redirector.json",
            f"{localdir}/../HH4b_common/datasets/background_TTtoX_pnfs_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET_pnfs_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_JetMET_redirector_24EraG.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_official_pnfs_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b_pnfs_redirector_pnfs_redirector.json",
        ],
        "filter": {
            "samples": (
                [
                    "DATA_JetMET",
                    # "DATA_JetMET0_HH4bBoosted",
                    # "DATA_JetMET1_HH4bBoosted",
                    "TTto4Q",
                    "TTtoLNu2Q",
                    "TTto2L2Nu",
                    "GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00",
                    "GluGlutoHHto4B_spanet_kl-m2p00_kt-1p00_c2-0p00",
                    "GluGlutoHHto4B_spanet_kl-m1p00_kt-1p00_c2-0p00",
                    "GluGlutoHHto4B_spanet_kl-5p00_kt-1p00_c2-0p00",
                    "GluGlutoHHto4B_spanet_kl-2p45_kt-1p00_c2-0p00",
                    "GluGlutoHHto4B_spanet_kl-0p00_kt-0p00_c2-0p00",
                    "GluGlutoHHto4B_spanet_kl-3p50_kt-1p00_c2-0p00",
                    "GluGlutoHHto4B_spanet_kl-4p00_kt-1p00_c2-0p00",
                    "GluGlutoHHto4B_spanet_kl-3p00_kt-1p00_c2-0p00",
                    "GluGlutoHHto4B_spanet_kl-2p00_kt-1p00_c2-0p00",
                    "GluGlutoHHto4B_spanet_kl-1p50_kt-1p00_c2-0p00",
                    "GluGlutoHHto4B_spanet_kl-0p50_kt-1p00_c2-0p00",
                    "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4",
                    "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2",
                    "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3",
                    "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43",
                    "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94",
                    "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36",
                    "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39",
                    "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96",
                    "VBFHHto4B_CV-2p12_C2V-3p87_C3-m5p96",
                    "VBFHHto4B_CV_1_C2V_0_C3_1",
                    "VBFHHto4B_CV_1_C2V_1_C3_1",
                    # "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00",
                    # "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00",
                    # "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00",
                    # "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00",
                    # "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10",
                    # "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p35",
                    # "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-1p00",
                    # "GluGlutoHHto4B_kl-m20p00_kt-1p00_c2-2p24",
                    # "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-3p00",
                    # "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-m2p00",
                    ]
            ),
            "samples_exclude": [],
            "year": year,
            # "year": year,
        },
        "subsamples": {},
    },
    workflow=HH4bbQuarkMatchingProcessorDummy,
    workflow_options={
    },
    # skim=cuts.skimming_cut_list,
    # skim=cuts.skimming_cut_list(config_options_dict),
    skim=[
        get_HLTsel(),
    ],
    # skim=[
    #     get_HLTsel(primaryDatasets=["JetMET"]),
    #     # get_HLTsel(primaryDatasets=["ParkingHH"]),
    # ],
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
