from configs.HH4b_common.config_files.__config_file__ import onnx_model_dict

from utils_configs.onnx_executor_factory import OnnxExecutorFactory


def get_executor_factory(executor_name, **kwargs):
    return OnnxExecutorFactory(onnx_model_dict=onnx_model_dict, **kwargs)
