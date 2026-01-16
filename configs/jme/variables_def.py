from pocket_coffea.parameters.histograms import HistConf, Axis
from params.binning import *
from workflow import *


def get_variables_dict(cuts_names_eta, cuts_names_reco_eta, cuts_names_eta_neutrino):
    variables_dict = (
        {
            **{
                f"MatchedJets{flav}_ResponseJECVSpt": HistConf(
                    [
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="ResponseJEC",
                            bins=response_bins,
                            pos=None,
                            label=f"MatchedJets{flav}_ResponseJEC",
                        ),
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="pt",
                            bins=pt_bins,
                            label=f"MatchedJets{flav}_pt",
                            type="variable",
                            pos=None,
                        ),
                    ]
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
            **{
                f"MatchedJets{flav}_JetPtJECVSpt": HistConf(
                    [
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="JetPtJEC",
                            bins=jet_pt_bins,
                            pos=None,
                            label=f"MatchedJets{flav}_JetPtJEC",
                        ),
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="pt",
                            bins=pt_bins,
                            label=f"MatchedJets{flav}_pt",
                            type="variable",
                            pos=None,
                        ),
                    ]
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
            **{
                f"MatchedJets{flav}_ResponseRawVSpt": HistConf(
                    [
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="ResponseRaw",
                            bins=response_bins,
                            pos=None,
                            label=f"MatchedJets{flav}_ResponseRaw",
                        ),
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="pt",
                            bins=pt_bins,
                            label=f"MatchedJets{flav}_pt",
                            type="variable",
                            pos=None,
                        ),
                    ]
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
            **{
                f"MatchedJets{flav}_JetPtRawVSpt": HistConf(
                    [
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="JetPtRaw",
                            bins=jet_pt_bins,
                            pos=None,
                            label=f"MatchedJets{flav}_JetPtRaw",
                        ),
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="pt",
                            bins=pt_bins,
                            label=f"MatchedJets{flav}_pt",
                            type="variable",
                            pos=None,
                        ),
                    ]
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
        }
        if int(os.environ.get("NEUTRINO", 0)) == 0
        else {}
    )
    if int(os.environ.get("PNET", 0)) == 1 and int(os.environ.get("NEUTRINO", 0)) == 0:
        variables_dict.update(
            {
                **{
                    f"MatchedJets{flav}_ResponsePNetRegVSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJets{flav}",
                                field="ResponsePNetReg",
                                bins=response_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_ResponsePNetReg",
                            ),
                            Axis(
                                coll=f"MatchedJets{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
                **{
                    f"MatchedJets{flav}_JetPtPNetRegVSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJets{flav}",
                                field="JetPtPNetReg",
                                bins=jet_pt_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_JetPtPNetReg",
                            ),
                            Axis(
                                coll=f"MatchedJets{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
            }
        )
        if int(os.environ.get("SPLITPNETREG15", 0)) == 1:
            variables_dict.update(
                {
                    **{
                        f"MatchedJets{flav}_ResponsePNetRegSplit15VSpt": HistConf(
                            [
                                Axis(
                                    coll=f"MatchedJetsSplit15{flav}",
                                    field="ResponsePNetReg",
                                    bins=response_bins,
                                    pos=None,
                                    label=f"MatchedJets{flav}_ResponsePNetReg",
                                ),
                                Axis(
                                    coll=f"MatchedJetsSplit15{flav}",
                                    field="pt",
                                    bins=pt_bins,
                                    label=f"MatchedJets{flav}_pt",
                                    type="variable",
                                    pos=None,
                                ),
                            ],
                            only_categories=cuts_names_eta + cuts_names_reco_eta,
                        )
                        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                    },
                    **{
                        f"MatchedJets{flav}_JetPtPNetRegSplit15VSpt": HistConf(
                            [
                                Axis(
                                    coll=f"MatchedJetsSplit15{flav}",
                                    field="JetPtPNetReg",
                                    bins=jet_pt_bins,
                                    pos=None,
                                    label=f"MatchedJets{flav}_JetPtPNetReg",
                                ),
                                Axis(
                                    coll=f"MatchedJetsSplit15{flav}",
                                    field="pt",
                                    bins=pt_bins,
                                    label=f"MatchedJets{flav}_pt",
                                    type="variable",
                                    pos=None,
                                ),
                            ],
                            only_categories=cuts_names_eta + cuts_names_reco_eta,
                        )
                        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                    },
                }
            )

    if int(os.environ.get("PNET", 0)) == 1 and int(os.environ.get("NEUTRINO", 1)) == 1:
        variables_dict.update(
            {
                **{
                    f"MatchedJets{flav}_ResponsePNetRegNeutrinoVSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="ResponsePNetRegNeutrino",
                                bins=response_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_ResponsePNetRegNeutrino",
                            ),
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
                **{
                    f"MatchedJets{flav}_JetPtPNetRegNeutrinoVSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="JetPtPNetRegNeutrino",
                                bins=jet_pt_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_JetPtPNetRegNeutrino",
                            ),
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
            }
        )
        if int(os.environ.get("SPLITPNETREG15", 0)) == 1:
            variables_dict.update(
                {
                    **{
                        f"MatchedJets{flav}_ResponsePNetRegNeutrinoSplit15VSpt": HistConf(
                            [
                                Axis(
                                    coll=f"MatchedJetsNeutrinoSplit15{flav}",
                                    field="ResponsePNetRegNeutrino",
                                    bins=response_bins,
                                    pos=None,
                                    label=f"MatchedJets{flav}_ResponsePNetRegNeutrino",
                                ),
                                Axis(
                                    coll=f"MatchedJetsNeutrinoSplit15{flav}",
                                    field="pt",
                                    bins=pt_bins,
                                    label=f"MatchedJets{flav}_pt",
                                    type="variable",
                                    pos=None,
                                ),
                            ],
                            only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                        )
                        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                    },
                    **{
                        f"MatchedJets{flav}_JetPtPNetRegNeutrinoSplit15VSpt": HistConf(
                            [
                                Axis(
                                    coll=f"MatchedJetsNeutrinoSplit15{flav}",
                                    field="JetPtPNetRegNeutrino",
                                    bins=jet_pt_bins,
                                    pos=None,
                                    label=f"MatchedJets{flav}_JetPtPNetRegNeutrino",
                                ),
                                Axis(
                                    coll=f"MatchedJetsNeutrinoSplit15{flav}",
                                    field="pt",
                                    bins=pt_bins,
                                    label=f"MatchedJets{flav}_pt",
                                    type="variable",
                                    pos=None,
                                ),
                            ],
                            only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                        )
                        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                    },
                }
            )
    if int(os.environ.get("UPART", 0)) == 1 and int(os.environ.get("NEUTRINO", 0)) == 0:
        variables_dict.update(
            {
                **{
                    f"MatchedJets{flav}_ResponseUparTRegVSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJets{flav}",
                                field="ResponseUparTReg",
                                bins=response_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_ResponseUparTReg",
                            ),
                            Axis(
                                coll=f"MatchedJets{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
                **{
                    f"MatchedJets{flav}_JetPtUparTRegVSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJets{flav}",
                                field="JetPtUparTReg",
                                bins=jet_pt_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_JetPtUparTReg",
                            ),
                            Axis(
                                coll=f"MatchedJets{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
            }
        )

    if int(os.environ.get("UPART", 0)) == 1 and int(os.environ.get("NEUTRINO", 1)) == 1:
        variables_dict.update(
            {
                **{
                    f"MatchedJets{flav}_ResponseUparTRegNeutrinoVSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="ResponseUparTRegNeutrino",
                                bins=response_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_ResponseUparTRegNeutrino",
                            ),
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
                **{
                    f"MatchedJets{flav}_JetPtUparTRegNeutrinoVSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="JetPtUparTRegNeutrino",
                                bins=jet_pt_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_JetPtUparTRegNeutrino",
                            ),
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
            }
        )
        
    if int(os.environ.get("NEUTRINO", 1)) == 1 and int(os.environ.get("CLOSURE", 0)) == 1:
        variables_dict.update(
            {
                **{
                    f"MatchedJets{flav}_ResponseJECNeutrinoVSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="ResponseJEC",
                                bins=response_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_ResponseJECNeutrino",
                            ),
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ]
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
                **{
                    f"MatchedJets{flav}_JetPtJECNeutrinoVSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="JetPtJEC",
                                bins=jet_pt_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_JetPtJECNeutrino",
                            ),
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ]
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
                # **{
                #     f"MatchedJets{flav}_ResponseRawNeutrinoVSpt": HistConf(
                #         [
                #             Axis(
                #                 coll=f"MatchedJetsNeutrino{flav}",
                #                 field="ResponseRaw",
                #                 bins=response_bins,
                #                 pos=None,
                #                 label=f"MatchedJets{flav}_ResponseRawNeutrino",
                #             ),
                #             Axis(
                #                 coll=f"MatchedJetsNeutrino{flav}",
                #                 field="pt",
                #                 bins=pt_bins,
                #                 label=f"MatchedJets{flav}_pt",
                #                 type="variable",
                #                 pos=None,
                #             ),
                #         ]
                #     )
                #     for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                # },
                # **{
                #     f"MatchedJets{flav}_JetPtRawNeutrinoVSpt": HistConf(
                #         [
                #             Axis(
                #                 coll=f"MatchedJetsNeutrino{flav}",
                #                 field="JetPtRaw",
                #                 bins=jet_pt_bins,
                #                 pos=None,
                #                 label=f"MatchedJets{flav}_JetPtRawNeutrino",
                #             ),
                #             Axis(
                #                 coll=f"MatchedJetsNeutrino{flav}",
                #                 field="pt",
                #                 bins=pt_bins,
                #                 label=f"MatchedJets{flav}_pt",
                #                 type="variable",
                #                 pos=None,
                #             ),
                #         ]
                #     )
                #     for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                # },
            }
        )

    # if int(os.environ.get("CLOSURE", 0)) == 1:
    if False:
        variables_dict.update(
            {
                **{
                    f"MatchedJets{flav}_MCTruthCorrPNetRegVSJetPtPNetReg": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJets{flav}",
                                field="MCTruthCorrPNetReg",
                                bins=list(np.linspace(0, 2, 1000)),
                                pos=None,
                                label=f"MatchedJets{flav}_MCTruthCorrPNetReg",
                            ),
                            Axis(
                                coll=f"MatchedJets{flav}",
                                field="JetPtPNetReg",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_JetPtPNetReg",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
                **{
                    f"MatchedJets{flav}_MCTruthCorrPNetRegNeutrinoVSJetPtPNetRegNeutrino": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="MCTruthCorrPNetRegNeutrino",
                                bins=list(np.linspace(0, 2, 1000)),
                                pos=None,
                                label=f"MatchedJets{flav}_MCTruthCorrPNetRegNeutrino",
                            ),
                            Axis(
                                coll=f"MatchedJetsNeutrino{flav}",
                                field="JetPtPNetRegNeutrino",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_JetPtPNetRegNeutrino",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
            }
        )
    return variables_dict