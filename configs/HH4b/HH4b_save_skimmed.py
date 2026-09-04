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
year = ["2022_postEE", "2023_preBPix"]
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
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/mmalucch/HH4b/skimmed_files/DATA_ParkingHH_resolved_skimmed",
    save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/bevila_t/PostDoc/HH4b/skimmed_files/ZZ_ZH_resolved_private_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/bevila_t/PostDoc/HH4b/skimmed_files/vbf_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/bevila_t/PostDoc/HH4b/skimmed_files/data_parking_2023_PostBPix_skimmed",
    parameters=parameters,
    datasets={
        "jsons": [
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet.json",
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_official.json",
            # f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet_redirector.json",
            #e f"{localdir}/../HH4b_common/datasets/background_TTtoX_redirector.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH.json",
            # f"{localdir}/../HH4b_common/datasets/DATA_JetMET_pnfs_redirector.json",
            f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH_pnfs_redirector.json",
            # TODO: these dataset definitions are not committed in the repository,
            # re-enable them once they are added (otherwise the Configurator
            # crashes with a FileNotFoundError). NOTE: the redirector file used
            # below in their place only defines the 2023_postBPix ZZ/ZH
            # datasets, so `year` has to be adapted before running it:
            # f"{localdir}/../HH4b_common/datasets/background_ZZ_ZH_private_2022_postEE.json",
            # f"{localdir}/../HH4b_common/datasets/background_ZZ_ZH_private_2023_preBPix.json",
            # f"{localdir}/../HH4b_common/datasets/background_ZZ_ZH_private.json",
            f"{localdir}/../HH4b_common/datasets/background_ZZ_ZH_private_redirector.json",
            f"{localdir}/../HH4b_common/datasets/signal_VBF_HH4b.json",
        ],
        "filter": {
            "samples": (
                [
                    #"GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00",
                    #"GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00",
                    #"GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00",
                    # "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00",
                    "ZZTo4B01j",
                    "ggZH_HToBB_ZToBB",
                    "ZH_ZToBB_HToBB"
                    # "DATA_JetMET",
                    # "DATA_JetMET0_HH4bResolved",
                    # "DATA_JetMET1_HH4bResolved",
                    # "DATA_ParkingHH",
                    # "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4",
                    # "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2",
                    # "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3",
                    # "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43",
                    # "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94",
                    # "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36",
                    # "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39",
                    # "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96",
                    # "VBFHHto4B_CV_1_C2V_0_C3_1",
                    # "VBFHHto4B_CV_1_C2V_1_C3_1",
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
