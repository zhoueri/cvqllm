import torch
import numpy as np
import dataclasses

from typing import Optional, Union, Tuple

from flexllmgen.pytorch_backend import (TorchTensor, TorchDevice,
     DeviceType, general_copy, global_cpu_device, ConfidentialTensor)
from flexllmgen.utils import np_dtype_to_torch_dtype

from flexllmgen.gptq.vq_quant import kpp_parallel_sampled, mahalanobis_init, VQQuantizer, vq_quantize

@dataclasses.dataclass
class VectorQuantConfig:
    """向量量化配置类，用于设置和存储向量量化的参数"""
    
    # 基本量化参数
    wbit: float = 16                      # 量化位宽
    vq_dim: int = 2                    # 向量维度
    groupsize: int = 1024               # 分组大小
    
    # 码本结构参数
    columns_per_group: Optional[int] = None  # 每组列数
    codebook_bitwidth: Optional[int] = None  # 码本位宽
    
    # K-means算法参数
    kmeans_init_method: str = "mahalanobis"  # K-means初始化方法: "kpp", "mahalanobis", "cdf"
    kmeans_iters: int = 10                   # K-means迭代次数
    assignment_chunk_size: Optional[int] = None  # 分配块大小
    kpp_n_subsample: int = 100000            # KPP子采样数量
    quantize_during_kmeans: bool = False     # 是否在K-means过程中量化中心点
    
    # 缩放参数
    vq_scaling_blocksize: int = -1           # 向量缩放块大小，-1表示禁用
    vq_scaling_norm: str = "max"             # 向量缩放范数类型
    vq_scaling_n_bits: int = 4               # 向量缩放位数
    vq_scaling_domain: str = "log"           # 向量缩放域
    
    # 高级选项
    quantize_per_codebook: bool = True       # 每码本量化
    
    def __post_init__(self):
        """验证配置参数的合理性"""
        if self.wbit * self.vq_dim > 16:
            print(f"警告: wbit({self.wbit}) * vq_dim({self.vq_dim}) = {self.wbit*self.vq_dim} > 16，"
                  f"码本大小(2^{self.wbit*self.vq_dim})可能过大")
            
        if self.kmeans_init_method not in ["kpp", "mahalanobis", "cdf"]:
            raise ValueError(f"不支持的K-means初始化方法: {self.kmeans_init_method}")
            
        if self.vq_scaling_norm not in ["max", "l2"]:
            raise ValueError(f"不支持的向量缩放范数类型: {self.vq_scaling_norm}")
            
        if self.vq_scaling_domain not in ["log", "linear"]:
            raise ValueError(f"不支持的向量缩放域: {self.vq_scaling_domain}")

'''Security tag should be set when loading sensitive data.'''
def general_copy_confidential(dst: ConfidentialTensor, dst_indices: Tuple[slice],
                 src: ConfidentialTensor, src_indices: Tuple[slice]):
    if src.is_confidential:
        assert dst.is_confidential, "dst must be confidential"
        assert dst.device.device_type == src.device.device_type == DeviceType.VECTORQUANT, "dst and src must be on the same device"
        if dst.device.base_device.device_type != src.device.base_device.device_type:  # Cross-device transfer occurs
            set_secure_flag()
            general_copy(dst, dst_indices, src, src_indices)
            clear_secure_flag()
        else:
            general_copy(dst, dst_indices, src, src_indices)
    else:
        general_copy(dst, dst_indices, src, src_indices)

def set_secure_flag():
    print("Secure flag set")
    pass

def clear_secure_flag():
    print("Secure flag cleared")
    pass

class TorchVectorQuantDevice:

    def __init__(self, base_device):
        '''Manager tensor in a vector quant format.'''
        self.name = "vector_quant"
        self.device_type = DeviceType.VECTORQUANT
        self.base_device = base_device
        self.data_dequant_workspace = None
        self.workspace_pt = 0

    def allocate(self, shape, dtype, vectorquant_config, pin_memory=None, name=None, codebook=False, quantizer=None):
        '''Allocate a tensor in vector quant format.'''
        wbits = vectorquant_config.wbit
        vq_dim = vectorquant_config.vq_dim
        n_centroids = 2 ** (wbits * vq_dim) # "K = 2^(wbits * vq_dim)"

        assert len(shape) == 2, "Only 2D tensor is supported"
        if quantizer is None:
            quantizer = VectorQuantizer(vectorquant_config)
            quantizer.configure(vectorquant_config.wbit) 
        groupsize, centroids_G = quantizer.get_groupsize(shape, vectorquant_config.groupsize) # "In most case, G = shape[0]"
        n_groups = shape[1] // groupsize # "Number of quant groups"
        assert shape[1] % groupsize == 0, f"shape[1] {shape[1]} is not divisible by groupsize {groupsize}"
        
        if codebook:
            centroids_shape = (int(n_groups), int(centroids_G), int(n_centroids), int(vq_dim)) # "N x G x K x D"
            quantizer.centroids_shape = centroids_shape
            quantizer.centroids_dtype = dtype
            centroids = self.base_device.allocate(centroids_shape, dtype,
                pin_memory=pin_memory, name=name)
            
            return ConfidentialTensor(shape, np_dtype_to_torch_dtype[dtype], 
                            (centroids, quantizer, vectorquant_config), self, is_confidential=True, name = name, is_codebook=True)
        else:
            idx_shape = (int(n_groups), int(shape[0]), int(groupsize // vq_dim)) # "N x G x R // N // D "
            '''The data type of the index tensor is related to the number of codebooks'''
            if n_centroids <= 2**8: 
                idx_dtype = np.uint8
            elif n_centroids <= 2**16:  
                idx_dtype = np.uint16
            else:  
                idx_dtype = np.int64

            quantizer.idx_shape = idx_shape
            quantizer.idx_dtype = idx_dtype
            idx = self.base_device.allocate(idx_shape, idx_dtype, pin_memory=pin_memory, name=name) 
            return ConfidentialTensor(shape, np_dtype_to_torch_dtype[dtype], 
                            (idx, quantizer, vectorquant_config), self, is_confidential=False, name = name)
        

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
    
    def simple_vq_quant(self, W, idx, centroids, quantizer, vectorquant_config):
        W1 = W.data.clone()
        assert len(W1.shape) == 2, "Only 2D tensor is supported"
        W1 = W1.float()
        # idx = torch.zeros(quantizer.idx_shape, dtype=quantizer.idx_dtype, device=W1.device)
        # centroids = torch.zeros(quantizer.centroids_shape, dtype=W1.dtype, device=W1.device)

        Losses = torch.zeros_like(W1)
        n_group = 0
        for i in range(0, W.shape[1], quantizer.groupsize):
            end = min(i + quantizer.groupsize, W.shape[1])
            W_group = W1[:, i:end].clone()
            Losses1 = torch.zeros_like(W_group)
            centroids[n_group] = quantizer.find_param(W_group)
            
            for j in range(quantizer.groupsize):
                if j % quantizer.vq_dim == 0:
                    w = W_group[:, j:j+quantizer.vq_dim]
                    q, assmt = vq_quantize(w, quantizer, centroids=centroids[n_group])
                    idx[n_group,:,j // quantizer.vq_dim] = assmt
                    Losses1[:, j:j+quantizer.vq_dim] = (w - q)**2
            Losses[:, i:end] = Losses1
            n_group += 1
        
        idx, centroids = self.optimize_index_desensitization(idx, centroids, quantizer)
        return (
            ConfidentialTensor(idx.shape, idx.dtype, (idx, quantizer, vectorquant_config), self), 
            ConfidentialTensor(centroids.shape, centroids.dtype, (centroids, quantizer, vectorquant_config), self, is_confidential=True, is_codebook=True),
        )
    
    def optimize_index_desensitization(self, idx_tensor, centroids_tensor, quantizer):
        """
        优化索引矩阵和码本，降低索引矩阵信息熵，实现脱敏

        Args:
            idx_tensor: 索引张量，ConfidentialTensor类型
            centroids_tensor: 码本张量，ConfidentialTensor类型
            quantizer: VectorQuantizer对象
    
        Returns:
            优化后的(idx_tensor, centroids_tensor)
        """
        # 获取原始数据
        idx = idx_tensor.data.clone() if isinstance(idx_tensor.data, torch.Tensor) else idx_tensor.data
        centroids = centroids_tensor.data.clone() if isinstance(centroids_tensor.data, torch.Tensor) else centroids_tensor.data
    
        # 获取形状
        batch_size, rows, cols = idx.shape
        _, centroids_G, n_centroids, vq_dim = centroids.shape

        # 保存原始码本-索引映射的副本用于验证
        original_mapping = {}
        for n in range(batch_size):
            original_mapping[n] = {}
            for i in range(n_centroids):
                original_mapping[n][i] = centroids[n, 0, i, :].clone()
    
        # 对每个批次独立处理
        for n in range(batch_size):
            # 1. 计算当前索引矩阵的统计信息
            idx_flat = idx[n].reshape(-1).to(torch.long)
            idx_freq = torch.bincount(idx_flat, minlength=n_centroids)

            # 2. 设计置换策略 - 使用频率打乱
            # 频率相似的索引应该被分散，降低局部相关性
            freq_sorted_indices = torch.argsort(idx_freq)

            # 3. 创建新的码本排列 - 采用交错模式
            # 这将使高频和低频索引交错分布，降低信息熵
            new_indices = torch.zeros_like(freq_sorted_indices)
            half = (n_centroids + 1) // 2
            new_indices[:half] = freq_sorted_indices[::2]  # 偶数位放置
            new_indices[half:] = freq_sorted_indices[1::2]  # 奇数位放置

            # 4. 创建新旧索引映射
            old_to_new = {old_idx.item(): new_idx.item() 
                           for new_idx, old_idx in enumerate(new_indices)}

            # 5. 构建新的码本
            new_centroids = torch.zeros_like(centroids[n])
            for old_idx, new_idx in old_to_new.items():
                new_centroids[:, new_idx] = centroids[n, :, old_idx]

            # 6. 更新索引矩阵
            for r in range(rows):
                for c in range(cols):
                    old_idx = idx[n, r, c].item()
                    idx[n, r, c] = old_to_new[old_idx]

            # 7. 应用变更
            centroids[n] = new_centroids

        # 8. 验证变换保持一致性
        verify_consistency(idx, centroids, original_mapping, quantizer)

        # 9. 创建结果张量
        # 如果idx和centroids本身就是torch.Tensor，则直接使用
        idx_tensor_new = TorchTensor.create_from_torch(idx, self.base_device)
        centroids_tensor_new = TorchTensor.create_from_torch(centroids, self.base_device)
        # 返回优化后的张量
        return idx_tensor_new, centroids_tensor_new
    
    def dequantize(self, idx_tensor, centroids_tensor):
        idx, quantizer, vectorquant_config = idx_tensor.data
        centroids = centroids_tensor.data[0]
        w = quantizer.dequantize(idx, centroids)
        return w
    
    def create_tmp_tensor(self, X, quantizer):
        """创建临时张量"""
        idx_shape = quantizer.idx_shape
        centroids_shape = quantizer.centroids_shape
        idx = torch.zeros(idx_shape, dtype=np_dtype_to_torch_dtype[quantizer.idx_dtype], device=X.device)
        centroids = torch.zeros(centroids_shape, dtype=np_dtype_to_torch_dtype[quantizer.centroids_dtype], device=X.device)
        return idx, centroids


class VectorQuantizer(VQQuantizer):
    def __init__(
        self,
        vectorquant_config,
    ):
        super().__init__(
            vq_dim=vectorquant_config.vq_dim,
            columns_per_group=vectorquant_config.columns_per_group,
            vq_scaling_blocksize=vectorquant_config.vq_scaling_blocksize,
            vq_scaling_norm=vectorquant_config.vq_scaling_norm,
            vq_scaling_n_bits=vectorquant_config.vq_scaling_n_bits,
            vq_scaling_domain=vectorquant_config.vq_scaling_domain,
            kmeans_init_method=vectorquant_config.kmeans_init_method,
            assignment_chunk_size=vectorquant_config.assignment_chunk_size,
            kmeans_iters=vectorquant_config.kmeans_iters,
            codebook_bitwidth=vectorquant_config.codebook_bitwidth,
            quantize_per_codebook=vectorquant_config.quantize_per_codebook,
            quantize_during_kmeans=vectorquant_config.quantize_during_kmeans,
            n_subsample=vectorquant_config.kpp_n_subsample
        )
    
    def get_groupsize(self, shape, groupsize):
        if self.columns_per_group is not None:
            if groupsize < self.columns_per_group:
                assert self.columns_per_group % groupsize == 0
                self.columns_per_group = groupsize

            assert groupsize % self.columns_per_group == 0
            assert shape[1] % self.columns_per_group == 0

            self.rows_per_group = groupsize // self.columns_per_group
            assert shape[0] % self.rows_per_group == 0

            self.groups_per_column = shape[0] // self.rows_per_group
            self.groupsize = self.columns_per_group
            return self.columns_per_group, self.groups_per_column

        if groupsize < shape[1]:
            assert shape[1] % groupsize == 0
            self.groups_per_column = shape[0]
            self.groupsize = groupsize
            return groupsize, self.groups_per_column

        if groupsize % shape[1] != 0:
            print(
                f"Requested groupsize {groupsize} doesn't fit tensor shape[0] {shape[0]}. "
                f"Upscaling to {int(np.ceil(groupsize / shape[0]) * shape[0])}"
            )

        rows_per_group = int(np.ceil(groupsize / shape[1]))
        self.groups_per_column = shape[0] // rows_per_group
        self.groupsize = shape[1]
        return shape[1], self.groups_per_column
    
    def find_param(self, w):
        assert len(w.shape) == 2, "Only 2D tensor is supported"
        super().find_param(w)
        return self.all_centroids[-1]

    def dequantize(self, idx, centroids):
        batch_size, rows, cols = idx.shape
    
        output = torch.zeros(batch_size, rows, cols * self.vq_dim, 
                            device=idx.device, dtype=centroids.dtype)
    
        for n in range(batch_size):
            for r in range(rows):
                for c in range(cols):
                    index = idx[n, r, c].item()
                    vec = centroids[n, 0, index, :]
                    output[n, r, c*self.vq_dim:(c+1)*self.vq_dim] = vec

        original_shape = (rows, batch_size, cols * self.vq_dim)
        return output.permute(1, 0, 2).reshape(original_shape)



def verify_consistency(idx, centroids, original_mapping, quantizer):
    """验证变换前后的解码一致性"""
    batch_size, rows, cols = idx.shape
    vq_dim = quantizer.vq_dim
    
    for n in range(batch_size):
        for r in range(rows):
            for c in range(cols):
                # 获取新索引和对应向量
                new_idx = idx[n, r, c].item()
                new_vec = centroids[n, 0, new_idx, :]
                
                # 获取旧索引对应的原始向量
                old_idx = None
                for i, vec in original_mapping[n].items():
                    if torch.allclose(new_vec, vec, rtol=1e-5, atol=1e-5):
                        old_idx = i
                        break
                
                # 验证结果
                if old_idx is None:
                    print(f"警告: 在批次{n}行{r}列{c}处找不到匹配的原始向量")