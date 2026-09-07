"""Default options for the HH4b configuration files.

Every config in ``configs/HH4b_common/config_files/`` starts from the two
dictionaries defined here and overrides only the entries it needs, e.g.

    from configs.HH4b_common.config_files.default_config import (
        default_onnx_model_dict as onnx_model_dict,
        default_config_options_dict as config_options_dict,
    )

    onnx_model_dict |= {"spanet": "/path/to/model.onnx"}
    config_options_dict |= {"run2": False, ...} | onnx_model_dict

``config_options_dict`` is passed to the PocketCoffea ``Configurator`` as
``workflow_options``; the processor (``HH4bCommonProcessor.__init__``) turns
every key into an attribute of ``self``, so an option ``"foo"`` is read inside
the workflows as ``self.foo``. The same dictionary is also read directly by the
config templates (``configs/HH4b/*.py``, ``configs/VBF_HH4b*/*.py``) to build
the categories, the columns and the preselections.

See ``configs/HH4b_common/README.md`` ("Configuration options") for a longer
description of each option.
"""

import configs.HH4b_common.dnn_input_variables as dnn_vars


# ----------------------------------------------------------------------------
# ONNX MODELS
# ----------------------------------------------------------------------------
# Paths to the ONNX models applied on the fly during the processing.
# An empty string means "do not run this model": every model is optional and
# the presence/absence of a model also drives which categories and columns are
# created (e.g. no model at all => only the `4b_region` used for SPANet
# training is defined).
default_onnx_model_dict = {
    # Jet-to-Higgs pairing model. If set, the pairing is taken from SPANet
    # instead of the Run-2 D_HH algorithm or the gen-level truth matching.
    "spanet": "",
    # ggF vs VBF classifier. It can either be the same file as `spanet` (the
    # class-probability output of the pairing model is used) or a standalone
    # model, in which case `vbf_discriminator_input_variables` and
    # `max_num_jets_vbf_discriminator` must be set accordingly.
    "vbf_discriminator": "",
    # DNN reweighting 2b (or mixed) data to 4b data; produces the per-event
    # `bkg_morphing_dnn_weight` applied in the `*_postW` categories.
    "bkg_morphing_dnn": "",
    # Signal vs background classifier; produces `sig_bkg_dnn_score`.
    "sig_bkg_dnn": "",
    # Ensemble of morphing models used to estimate the spread (systematic
    # uncertainty) of the background morphing weights.
    "bkg_morphing_spread_dnn": "",
}

default_config_options_dict = {
    # ------------------------------------------------------------------
    # ANALYSIS FLAVOUR
    # ------------------------------------------------------------------
    # Jet pt definition / object preselection to use. It selects the
    # `params/object_preselection_<approach>_approach.yaml` file and the way
    # the `Jet` collection is built in `apply_object_preselection`:
    #   "first"   : use the PNet+Neutrino regressed pt when it is available
    #               (HIG-24-010 first approach)
    #   "second"  : use the regressed pt also when the jet passes the loose
    #               b-tag WP (HIG-24-010 second approach)
    #   "boosted" : skip the resolved jet-pt handling and the b-tag WP
    #               definition (used for the boosted analysis)
    "approach": "first",
    # Run the boosted (AK8 / FatJet) analysis: builds the `FatJetGood`
    # collection, the boosted categories and the boosted DNN variables.
    "boosted": False,
    # Use the Run-2 D_HH pairing algorithm instead of SPANet.
    # Mutually exclusive with `random_pt`.
    "run2": False,
    # Run the VBF part of the analysis: builds the VBF jet collections, the
    # VBF variables and the VBF categories.
    "vbf_analysis": False,
    # Run on the mixed-data samples (data-driven background model) instead of
    # the 2b data: the 2b/4b `preW`/`postW` categories are replaced by 4b ones,
    # the HLT/L1 skim cuts and the jet calibration are switched off.
    "mixeddata": False,

    # ------------------------------------------------------------------
    # SKIM AND PRESELECTION
    # ------------------------------------------------------------------
    # Apply the tight jet pt cuts (`pt_tight` in the object preselection) and
    # require the pt preselection on the 4 b-tagged Higgs candidate jets only.
    "tight_cuts": False,
    # Use the VBF preselection. NOTE: not supported anymore, `define_preselection`
    # raises a ValueError if this is True (the cut acts on the wrong jet
    # collection).
    "vbf_presel": False,
    # Use the boosted preselection (>= 2 FatJets) instead of the resolved one.
    # It also disables the jet veto map cut.
    "boosted_presel": False,
    # Drop the b-tag requirement from the preselection (`hh4b_presel_nobtag`).
    # Needed by the b-tag WP efficiency measurement, which has to be performed in
    # a region where no cut on the b-tag score is applied. The configs in
    # `configs/HH4b_btagging` turn it on with
    # `define_preselection(config_options_dict | {"no_btag": True})`.
    "no_btag": False,
    # Legacy flag for the semi-tight VBF jet selection; currently only accepted
    # as an argument of `jet_selection_nopu` and not used by any workflow.
    "semi_tight_vbf": True,
    # Drop the L1 seed requirement from the skim (`get_L1sel`). Needed for the
    # samples/eras for which the L1 emulation is not available.
    "noL1": False,

    # ------------------------------------------------------------------
    # TRIGGER SCALE FACTORS
    # ------------------------------------------------------------------
    # Apply the `sf_trigger` weight of PocketCoffea: the trigger scale factor is
    # the product of the data/MC efficiency ratios of the single filters of the
    # trigger path (plus the OR of the L1 seeds). It requires the parameters of
    # `params/trigger_scale_factors.yaml` and the correctionlib file referenced
    # there, produced from the ROOT files with the efficiency curves with
    # `scripts/convert_trigger_sf_to_correctionlib.py`.
    "trigger_sf": False,
    # Add the `sf_trigger` up/down variations to the output.
    "trigger_sf_variations": False,
    # Require the offline jets to be matched to the trigger objects firing each of
    # the filters of the trigger (`params/trigger_object_filters.yaml`). The
    # efficiencies are derived filter-by-filter, so this matching is part of the
    # selection in which the trigger scale factors are valid.
    "trigger_object_matching": False,

    # ------------------------------------------------------------------
    # TRUTH MATCHING (MC only)
    # ------------------------------------------------------------------
    # Which copy of the b-quarks from H->bb is matched to the jets:
    #   "first"                  : the direct children of the Higgs
    #   "last"                   : the last copy, found by walking the decay chain
    #   "last_numba"             : the last copy, found with the numba helper
    #   "last_numba_with_status" : same, but the b-quarks are selected with status==23
    "which_bquark": "last",
    # Maximum deltaR between a parton and a jet for the matching to succeed
    # (both for the Higgs b-quarks and for the VBF quarks).
    "parton_jet_min_dR": 0.4,

    # ------------------------------------------------------------------
    # VBF
    # ------------------------------------------------------------------
    # Run the gen-level matching of the VBF quarks to the jets (VBF MC only);
    # fills the `provenance_vbf` field.
    "vbf_parton_matching": False,
    # How the VBF quarks are identified at gen level:
    #   "with_mothers_children" : quarks whose mother also has two Higgs children
    #   "with_status"           : outgoing (status==23) non-b partons
    "which_vbf_quark": "with_mothers_children",
    # Number of additional VBF jet candidates (on top of the leading-mjj pair)
    # kept in `JetAdditionalGoodVBF` and merged into the SPANet input collections.
    "max_num_jets_add_vbf": 2,
    # Field used to order the additional VBF jets, e.g. "energy" or "pt".
    "jets_add_vbf_order": "energy",
    # Run the SPANet Higgs pairing first and define the VBF candidates from the
    # jets left over by the pairing, instead of using the b-tag ordered
    # `JetGoodClip` collection. Requires `spanet`.
    "vbf_matching_after_higgs_pairing": False,
    # Threshold on the ggF-vs-VBF discriminator score (`VBF_ggF_score`) used to
    # split the pass/fail VBF categories.
    "ggf_vbf_threshold": 0.95,
    # Boosted VBF only: overrides `vbf_analysis` when defining the categories in
    # `VBF_HH4b_boosted_config.py`, so that the VBF regions can be built without
    # switching on the full VBF jet reconstruction of the workflow.
    # `None` means "not set": `vbf_analysis` is used instead.
    "vbf_selection": None,

    # ------------------------------------------------------------------
    # FOX-WOLFRAM MOMENTA
    # ------------------------------------------------------------------
    # Maximum order l of the Fox-Wolfram moments computed on `JetGood`.
    # 0 disables the computation; otherwise the columns `FW_H{i}_{norm}` and
    # `FW_R{i}_{norm}` are added to `events` for i in range(max_order_FW).
    "max_order_FW": 0,
    # Normalisation schemes used for the Fox-Wolfram moments, e.g. ["W_T"].
    "FW_momenta_norms": ["W_T"],

    # ------------------------------------------------------------------
    # JET COLLECTIONS AND PAIRING
    # ------------------------------------------------------------------
    # Number of `JetGood` jets kept for the VBF analysis (`JetGoodClip`) and
    # used as the offset of the VBF pair inside the merged SPANet collections.
    "max_num_jets_good": 5,
    # Number of jets given to the SPANet pairing model; the true pairing is
    # also truncated to this number when computing the pairing efficiency.
    "max_num_jets_higgs_pairing": 5,
    # Number of jets given to a standalone ggF/VBF discriminator model.
    # None means "use all the jets of the input collection".
    "max_num_jets_vbf_discriminator": None,
    # Number of jets given to a SPANet-like signal-vs-background classifier
    # (only used when `sig_bkg_dnn` is a SPANet-format model).
    "max_num_jets_spanet_class": 4,
    # Ordering of the jets beyond the 4 Higgs candidates: "pt" re-sorts the
    # 5th+ jets by pt (the first 4 stay b-tag ordered); anything else keeps
    # the pure b-tag ordering.
    "fifth_jet": "pt",
    # Sort the `Jet` collection by regressed pt before the good-jet selection
    # and order the additional jet (`JetNotFromHiggs`) by b-tag score or pt
    # depending on whether the pairing picked the 5th jet.
    "add_jet_spanet": False,
    # Use the old b-tag working-point convention, where the WP index starts at
    # -1 (no WP passed) instead of 0. Must match the convention of the trained
    # SPANet/DNN models.
    "old_wp_def": False,
    # Boosted only: order the FatJets by `btagBBTXbb` instead of `btagBB`.
    "TXbb_order": False,
    # b-tag SF studies only: compute the b-tag scale factors using only the
    # 5 leading jets instead of all the `JetGood`.
    "only5jetsbSF": False,

    # ------------------------------------------------------------------
    # MODEL INPUTS AND PADDING
    # ------------------------------------------------------------------
    # Input features of the SPANet pairing model, as an OrderedDict with a
    # "sequential" (per-jet) and a "global" (per-event) block. Must match the
    # event file used for the SPANet training; see `dnn_input_variables.py`.
    "spanet_input_name": dnn_vars.pairing_spanet_btag,
    # Flat list of the SPANet sequential input names. Only its last entry is
    # inspected by the config templates, to decide whether the b-tag
    # working-point columns have to be saved for the SPANet training.
    # `None` means "derive it from `spanet_input_name`", which is what the
    # templates do; set it explicitly only to override that.
    "spanet_input_name_list": None,
    # Input features of the signal-vs-background DNN. Also used to build the
    # list of columns saved when `dnn_variables` is True.
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_dnn_input_variables,
    # Input features of the background morphing DNN (and of the spread model).
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_dnn_input_variables,
    # Input features of a standalone ggF/VBF discriminator model.
    "vbf_discriminator_input_variables": None,
    # Value used to fill missing entries (padding, unmatched jets, ...) for the
    # DNN inputs and for the padded pairing-probability variable.
    "pad_value": -999.0,
    # Value used to pad the jet arrays fed to the SPANet models. It is kept
    # separate from `pad_value` because SPANet was trained with a different
    # padding convention.
    "pad_value_spanet": 9999.0,

    # ------------------------------------------------------------------
    # PAIRING PROBABILITY VARIABLES (SPANet only)
    # ------------------------------------------------------------------
    # Bin edge used to build `Binned_Arctanh_Delta_pairing_probabilities`, the
    # binned version of arctanh(p_best - p_second_best).
    "arctanh_delta_prob_bin_edge": 2.44,
    # Upper limit above which `Arctanh_Delta_pairing_probabilities` is replaced
    # by `pad_value` in `Padded_Arctanh_Delta_pairing_probabilities`.
    "arctanh_delta_prob_pad_limit": 2.0,

    # ------------------------------------------------------------------
    # CATEGORIES AND REGIONS
    # ------------------------------------------------------------------
    # Use the VR1 validation regions (Higgs mass planes centred at
    # (185, 180) GeV) instead of the nominal signal/control regions.
    "vr1": False,
    # Use the wide control region (30 < Rhh < 80 instead of 30 < Rhh < 55)
    # for the morphing `preW`/`postW` categories.
    "expandCR": False,
    # Add the blinded copies of the signal regions, keeping only the events
    # with `sig_bkg_dnn_score` below the blinding threshold.
    "blind": False,
    # Boosted only: split the QCD control region into the `qcd_A`/`qcd_B`/`qcd_C`
    # sub-regions instead of defining a single `qcd` region.
    "split_qcd": True,
    # Path to the pickled quantile transformer used to define variable-width
    # bins of `sig_bkg_dnn_score` (constant SM signal per bin) for the
    # datacards. One file per era; None disables the transformed histogram.
    "qt_postEE": None,
    "qt_preEE": None,

    # ------------------------------------------------------------------
    # SPANET TRAINING SAMPLE PRODUCTION
    # ------------------------------------------------------------------
    # Randomly scale the jet pt (and mass) to flatten the pt spectrum of the
    # SPANet training sample; saves the `*PtFlatten*` collections and the
    # `random_pt_weights` column. Cannot be used together with `run2`.
    "random_pt": False,
    # Range of the random pt scale factor used by `random_pt`:
    #   0.5 -> [0.5, 1.5], 0.3 -> [0.3, 1.7], 0.1 -> [0.1, 10.0]
    "rand_type": 0.3,

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------
    # Compute and save the DNN input variables (Higgs/HH kinematics, additional
    # jet, sigma_mbb, ...). If False, only the SPANet training columns (or the
    # default jet columns) are saved.
    "dnn_variables": True,
    # Dump the columns as parquet files per chunk instead of accumulating them
    # in the coffea output (sets `dump_columns_as_arrays_per_chunk`). The
    # output path ends up in the `config.json` of the run.
    "save_chunk": False,
    # Do not normalise the MC by the sum of the generator weights. Kept for
    # backward compatibility with the PocketCoffea `Configurator` option; it is
    # currently not forwarded by any of the config templates.
    "donotscale_sumgenweights": False,

    # ------------------------------------------------------------------
    # BOOSTED BDT
    # ------------------------------------------------------------------
    # Path to the XGBoost model of the other-group boosted analysis; produces
    # `boosted_bdt_score` and `boosted_bdt_vbf_score`. Empty string disables it.
    "bdt_model": "",
} | default_onnx_model_dict
