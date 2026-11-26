from pocket_coffea.lib.weights.weights import WeightLambda

bkg_morphing_dnn_weight = WeightLambda.wrap_func(
    name="bkg_morphing_dnn_weight",
    function=lambda params, metadata, events, size, shape_variations: events.bkg_morphing_dnn_weight,
    has_variations=False,
    isMC_only=False,
)

# Template version of the morphing weight (bkg_morphing_dnn_template): the numerator of the ttbar
# subtraction is not clipped to >= 0, so the weight can be negative. Applied instead of
# `bkg_morphing_dnn_weight` with bkg_morphing_weight_mode "template".
bkg_morphing_dnn_weight_template = WeightLambda.wrap_func(
    name="bkg_morphing_dnn_weight_template",
    function=lambda params, metadata, events, size, shape_variations: events.bkg_morphing_dnn_weight_template,
    has_variations=False,
    isMC_only=False,
)
