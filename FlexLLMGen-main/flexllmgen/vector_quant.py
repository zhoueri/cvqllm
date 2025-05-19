import torch
import numpy as np

from flexllmgen.pytorch_backend import (TorchTensor, TorchDevice,
    DeviceType, general_copy, fix_recursive_import)
from flexllmgen.utils import np_dtype_to_torch_dtype

class TorchVQDevice:

    def __init__(self, base_device):
        '''Manager tensor in a vector quant format.'''
        self.name = "vector_quant"
        self.device_type = DeviceType.VECTOR_QUANT
        self.base_device = base_device
        self.data_dequant_workspace = None
        self.workspace_pt = 0

    def allocate(self, shape, dtype, comp_config, pin_memory=None, name=None):
        
        return 

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            comp_config=policy.comp_cache_config, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            comp_config=policy.comp_cache_config, pin_memory=pin_memory)
        return k_cache, v_cache
    
    def quantize(self, tensor):
        return
    
    def dequantize(self, tensor):
        return
    
    