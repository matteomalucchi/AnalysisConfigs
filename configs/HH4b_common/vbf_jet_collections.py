"""
Names of the VBF jet collections that the categories, the saved columns and any
VBF study outside the workflow have to use, shared by the workflows and the
config files so that the two always agree.

- Creating the SPANet training inputs (no `spanet` model and no `run2`
  pairing): `JetGoodVBFMergedProvVBFPadded`, i.e. the VBF pair fed to SPANet
  (`spanet_vbf_candidates`) followed by the additional VBF jets. Its first two
  jets are the VBF pair.
- Applying a pairing (`spanet` model or `run2`): `JetGoodVBFSelectedPair`, the
  VBF pair assigned after the Higgs pairing, ordered by pt.
- Boosted analysis: `JetGoodVBFEnergyOrdered`, the pair leading in mjj among
  the VBF candidates, ordered by energy (built by the boosted workflow).
"""

VBF_TRAINING_JET_COLL = "JetGoodVBFMergedProvVBFPadded"
VBF_SELECTED_PAIR_COLL = "JetGoodVBFSelectedPair"
VBF_BOOSTED_PAIR_COLL = "JetGoodVBFEnergyOrdered"


def applies_pairing(spanet, run2):
    """True when a Higgs pairing is applied (SPANet model or Run 2 D_HH), i.e.
    when the workflow is not creating the SPANet training inputs."""
    return bool(spanet) or bool(run2)


def get_vbf_jet_collection(spanet, run2, boosted=False):
    """VBF jet collection to use for the categories and the saved columns. Only
    its first two jets are the VBF pair."""
    if boosted:
        return VBF_BOOSTED_PAIR_COLL
    if applies_pairing(spanet, run2):
        return VBF_SELECTED_PAIR_COLL
    return VBF_TRAINING_JET_COLL
