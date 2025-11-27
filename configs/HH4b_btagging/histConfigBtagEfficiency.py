from pocket_coffea.lib.hist_manager import Axis, HistConf


def btag_sf_hist(colName):
    btagHistConf = HistConf(
        axes=[
            Axis(
                coll=colName,
                field="pt",
                bins=[0., 20., 30., 50., 70., 100., 140., 200., 300., 600., 1000., 14000.],
                label=colName + " pT",
                pos=None),
            Axis(
                coll=colName,
                field="abseta",
                bins=[0, 2.4, 2.5],
                label=colName + " eta",
                pos=None),
            Axis(
                coll=colName,
                field="hadronFlavour",
                bins=[-0.5, 3.5, 4.5, 5.5],
                label=colName + " flav",
                pos=None)
        ],
        no_weights=True
    )
    return btagHistConf
