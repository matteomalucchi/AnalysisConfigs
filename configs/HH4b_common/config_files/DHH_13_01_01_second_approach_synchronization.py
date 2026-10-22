"""Second approach (HIG-24-010) without models: D_HH pairing and VBF pair made of
the leading-pt jet and the leading-pt jet in the opposite eta hemisphere.

See configs/HH4b_common/README.md ("Second approach").
"""
from configs.HH4b_common.config_files.default_config import (
    default_config_options_dict as config_options_dict,
)
from configs.HH4b_common.config_files.default_config import (
    default_onnx_model_dict as onnx_model_dict,
)

config_options_dict |= {
    "approach": "second",
    # D_HH pairing (minimum distance to the diagonal)
    "run2": True,
    "vbf_analysis": True,
    "max_num_jets_higgs_pairing": 4,
    # only >= 4 JetGood jets (|eta| < 2.5), no pt thresholds
    "only_4jets_presel": True,
} | onnx_model_dict
