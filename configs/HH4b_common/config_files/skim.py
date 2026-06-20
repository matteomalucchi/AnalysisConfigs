import os

localdir = os.path.dirname(os.path.abspath(__file__))
import configs.HH4b_common.dnn_input_variables as dnn_vars
from configs.HH4b_common.dnn_input_variables import pairing_spanet_btag

from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict

config_options_dict |= {
} | onnx_model_dict
