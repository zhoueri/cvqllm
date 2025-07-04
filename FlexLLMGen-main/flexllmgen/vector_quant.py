import torch
import numpy as np
import dataclasses
import torch.nn.functional as F
from typing import Optional, Union, Tuple

from flexllmgen.pytorch_backend import (TorchTensor, TorchDevice,
     DeviceType, general_copy, global_cpu_device, ConfidentialTensor)
from flexllmgen.utils import np_dtype_to_torch_dtype

from flexllmgen.gptq.vq_quant import kpp_parallel_sampled, mahalanobis_init, VQQuantizer, vq_quantize

from flexllmgen.encrypt_engine.encrypt import encrypt_tensor_aes_ctr, decrypt_tensor_aes_ctr, encrypt_tensor_cuda_aes_ctr, decrypt_tensor_cuda_aes_ctr


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
    kmeans_init_method: str = "kpp"  # K-means初始化方法: "kpp", "mahalanobis", "cdf"
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
            # 创建临时加密张量
            encrypted_tensor = create_encrypted_copy(src)
            try:
                # 传输加密数据
                general_copy(dst.data[0], dst_indices, encrypted_tensor, src_indices)
                # 就地解密目标数据
                dst.data[0].data = tensor_decrypt_inplace(dst.data[0].data, dst.device.base_device)
                print("✓ 加密传输成功完成")
            except Exception as e:
                print(f"✗ 加密传输失败: {e}")
                raise
            finally:
                # 清理临时加密张量
                del encrypted_tensor
        else:
            general_copy(dst.data[0], dst_indices, src.data[0], src_indices)
    else:
        general_copy(dst.data[0], dst_indices, src.data[0], src_indices)

def create_encrypted_copy(tensor: ConfidentialTensor):
    data_copy = tensor.data[0].data.clone()
    if tensor.device.base_device.device_type == DeviceType.CUDA:
        encrypt_tensor = encrypt_tensor_cuda_aes_ctr(data_copy)
    else:
        encrypt_tensor = encrypt_tensor_aes_ctr(data_copy)
    
    return TorchTensor.create_from_torch(encrypt_tensor, tensor.device.base_device)

def tensor_decrypt_inplace(tensor: torch.Tensor, base_device: TorchDevice):
    if base_device.device_type == DeviceType.CUDA:
        decrypt_tensor = decrypt_tensor_cuda_aes_ctr(tensor)
    else:
        decrypt_tensor = decrypt_tensor_aes_ctr(tensor)
        
    return decrypt_tensor

class TorchVectorQuantDevice:

    def __init__(self, base_device):
        '''Manager tensor in a vector quant format.'''
        self.name = "vector_quant"
        self.device_type = DeviceType.VECTORQUANT
        self.base_device = base_device

    def allocate(self, shape, dtype, vectorquant_config, pin_memory=None, name=None, codebook=False, quantizer=None):
        '''Allocate a tensor in vector quant format.'''
        wbits = vectorquant_config.wbit
        vq_dim = vectorquant_config.vq_dim
        n_centroids = 2 ** (wbits * vq_dim) # "K = 2^(wbits * vq_dim)"
        
        if len(shape) == 2:
            row, col = shape
        elif len(shape) == 3:
            row, col = shape[1], shape[2] # "for kv cache"
            print(f"shape of kv cache: {shape}, row: {row}, col: {col}")
        else:
            raise ValueError(f"Unsupported shape {shape}, only 2D or 3D tensors are supported")
        new_shape = (row, col)
        if quantizer is None:
            quantizer = VectorQuantizer(vectorquant_config)
            quantizer.configure(vectorquant_config.wbit) 
        groupsize, centroids_G = quantizer.get_groupsize(new_shape, vectorquant_config.groupsize) # "In most case, G = shape[0]"
        print(f"groupsize: {groupsize}, centroids_G: {centroids_G}")
        quantizer.groupsize = groupsize
        quantizer.centroids_G = centroids_G
        n_groups = (row // centroids_G) * (col // groupsize) # "Number of quant groups"
        quantizer.n_groups = n_groups
        assert col % groupsize == 0, f"shape[1] {col} is not divisible by groupsize {groupsize}"
        if n_centroids <= 2**8: 
            idx_dtype = np.uint8
        elif n_centroids <= 2**16:  
            idx_dtype = np.uint16
        else:  
            idx_dtype = np.int64
        quantizer.idx_dtype = idx_dtype
        if codebook:
            assert len(shape) == 2, "Only 2D tensor is supported"
            centroids_shape = (np.int64(n_groups), np.int64(centroids_G), np.int64(n_centroids), np.int64(vq_dim)) # "N x G x K x D"
            print(f"shape of centroids: {centroids_shape}, dtype: {dtype}")
            quantizer.centroids_shape = centroids_shape
            quantizer.centroids_dtype = dtype
            centroids = self.base_device.allocate(centroids_shape, dtype,
                pin_memory=pin_memory, name=name)
            
            return ConfidentialTensor(shape, np_dtype_to_torch_dtype[dtype], 
                            (centroids, quantizer, vectorquant_config), self, is_confidential=True, name = name, is_codebook=True)
        else:
            if len(shape) == 2:
                idx_shape = (np.int64(n_groups), np.int64(centroids_G), np.int64(groupsize // vq_dim)) # "N x G x R // N // D "
            else:
                idx_shape = (shape[0] ,np.int64(n_groups), np.int64(centroids_G), np.int64(groupsize // vq_dim)) # shape[0] is number of token

            quantizer.idx_shape = idx_shape

            idx = self.base_device.allocate(idx_shape, idx_dtype, pin_memory=pin_memory, name=name) 
            return ConfidentialTensor(shape, np_dtype_to_torch_dtype[dtype], 
                            (idx, quantizer, vectorquant_config), self, is_confidential=False, name = name)
        

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = ((prompt_len + gen_len - 1) , (gpu_batch_size * num_head), hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        print(f"shape of cache: {shape}")
        pin_memory = False
        k_cache_idx = self.allocate(shape, np.float16,
            vectorquant_config=policy.vector_quant_cache_config, pin_memory=pin_memory, codebook=False)
        v_cache_idx = self.allocate(shape, np.float16,
            vectorquant_config=policy.vector_quant_cache_config, pin_memory=pin_memory, codebook=False)   
        return k_cache_idx, v_cache_idx
    
    def init_cache_one_gpu_batch_codebook(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (gpu_batch_size * num_head, hidden_size // num_head)
        print(f"shape of cache: {shape}, dtype: {np.float16}")
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache_codebook = self.allocate(shape, np.float16,
            vectorquant_config=policy.vector_quant_cache_config, pin_memory=pin_memory, codebook=True)
        v_cache_codebook = self.allocate(shape, np.float16,
            vectorquant_config=policy.vector_quant_cache_config, pin_memory=pin_memory, codebook=True)   
        return k_cache_codebook, v_cache_codebook
    
    def simple_quant_cache(self, W, idx, centroids, quantizer, vectorquant_config):
        assert len(W.shape) == 3, "Cache tensor must be 3D"
        print(f"shape of W: {W.shape}, dtype: {W.dtype}, device: {W.device}")
        W1 = W.data.clone().float()
        idx_shape = (quantizer.n_groups, quantizer.centroids_G, quantizer.groupsize // vectorquant_config.vq_dim)
        print(f"shape of idx: {idx_shape}, dtype: {np_dtype_to_torch_dtype[quantizer.idx_dtype]}, device: {W.device.dev}")
        if W1.shape[0] != 1: # prefill
            quantized_indices_list = []
            train_codebook = True
            for i in range(W1.shape[0]):
                single_token_kv = W1[i:i+1]  
                single_token_kv_2d = single_token_kv.squeeze(0)
                print(f"shape of single_token_kv_2d: {single_token_kv_2d.shape}, dtype: {single_token_kv_2d.dtype}, device: {W.device}")  
                idx_tmp = torch.zeros(idx_shape, dtype=np_dtype_to_torch_dtype[quantizer.idx_dtype], device=W.device.dev)
                idx_tmp, centroids_tmp = self.simple_vq_quant(
                    single_token_kv_2d, idx_tmp, centroids.data, quantizer, vectorquant_config, train_codebook=train_codebook)
                quantized_indices_list.append(idx_tmp.data[0].data)
                if i == 0:
                    train_codebook = False  # Only train codebook for the first token
                
            idx_tensor = torch.stack(quantized_indices_list, dim=0)
            idx_tensor = TorchTensor.create_from_torch(idx_tensor, self.base_device)
            return (ConfidentialTensor(W.shape, W.dtype, (idx_tensor, quantizer, vectorquant_config), self),
                    centroids_tmp)
        else:  # generation
            single_token_kv = W1.squeeze(0)  # [B, R, C]
            idx_tmp = torch.zeros(idx_shape, dtype=np_dtype_to_torch_dtype[quantizer.idx_dtype], device=W.device.dev)
            idx_tmp, centroids_tmp = self.simple_vq_quant(
                single_token_kv, idx_tmp, centroids.data, quantizer, vectorquant_config)
            return idx_tmp, centroids_tmp
    
    def simple_dequant_cache(self, idx, centroids):
        idx, quantizer, _ = idx.data
        centroids = centroids.data[0]
        cache_list = []
        for i in range(idx.shape[0]):
            single_token_idx = idx[i:i+1]
            single_token_idx_2d = single_token_idx.squeeze(0)  # [R, C]
            w = quantizer.dequantize(single_token_idx_2d, centroids)
            cache_list.append(w)
        cache_tensor = torch.stack(cache_list, dim=0)
        return cache_tensor
        

    def init_attention_compute_workspace(self, config, task, policy):
        if self.base_device.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        b = policy.gpu_batch_size
        n_head = config.n_head
        head_dim = config.input_dim // n_head
        max_seq_len = task.prompt_len + task.gen_len - 1
        shape = (max_seq_len, b * n_head, head_dim)

        group_size, group_dim = (
            policy.comp_cache_config.group_size, policy.comp_cache_config.group_dim)
        num_groups = (shape[group_dim] + group_size - 1) // group_size
        new_shape = (shape[:group_dim] + (num_groups, group_size) +
                     shape[group_dim+1:])

        self.data_decompress_workspace = [
            torch.empty(*new_shape, dtype=torch.float32,
                device=self.base_device.dev),
            torch.empty(*new_shape, dtype=torch.float32,
                device=self.base_device.dev),
        ]
    
    def simple_vq_quant(self, W, idx, centroids, quantizer, vectorquant_config, train_codebook=False):
        import time
        total_start = time.time()
        W1 = W.data.clone()
        assert len(W1.shape) == 2, "Only 2D tensor is supported"
        W1 = W1.float()
        prep_end = time.time()
        print(f"Preparation time: {prep_end - total_start:.4f} seconds")
        # idx = torch.zeros(quantizer.idx_shape, dtype=quantizer.idx_dtype, device=W1.device)
        # centroids = torch.zeros(quantizer.centroids_shape, dtype=W1.dtype, device=W1.device)
        n_group = 0
        total_find_param_time = 0
        total_vq_quantize_time = 0

        main_loop_start = time.time()
        for i in range(0, W.shape[1], quantizer.groupsize):
            group_start = time.time()
            end = min(i + quantizer.groupsize, W.shape[1])
            W_group = W1[:, i:end].clone()
            
            if train_codebook:
                find_param_start = time.time()
                centroids[n_group] = quantizer.find_param(W_group)
                find_param_end = time.time()
                find_param_time = find_param_end - find_param_start
                total_find_param_time += find_param_time

            vq_start = time.time()
            for j in range(quantizer.groupsize):
                if j % quantizer.vq_dim == 0:
                    w = W_group[:, j:j+quantizer.vq_dim]
                    q, assmt = vq_quantize(w, quantizer, centroids=centroids[n_group])
                    idx[n_group,:,j // quantizer.vq_dim] = assmt.squeeze(-1)
            vq_end = time.time()
            vq_time = vq_end - vq_start
            total_vq_quantize_time += vq_time
            n_group += 1
            group_end = time.time()
            if n_group % 10 == 0:  # 每10组打印一次
                print(f"组 {n_group}/{W.shape[1]//quantizer.groupsize}: "
                    f"码本计算耗时={find_param_time:.4f}秒, "
                    f"向量量化耗时={vq_time:.4f}秒, "
                    f"总耗时={group_end-group_start:.4f}秒")
        main_loop_end = time.time()
        print(f"主循环总耗时: {main_loop_end - main_loop_start:.4f}秒")
        print(f"- 码本计算总耗时: {total_find_param_time:.4f}秒 ({total_find_param_time/(main_loop_end-main_loop_start)*100:.1f}%)")
        print(f"- 向量量化总耗时: {total_vq_quantize_time:.4f}秒 ({total_vq_quantize_time/(main_loop_end-main_loop_start)*100:.1f}%)")
   
        # desensitize_start = time.time()
        # idx, centroids = self.optimize_index_desensitization(idx, centroids, quantizer)
        # desensitize_end = time.time()
        # print(f"索引脱敏优化耗时: {desensitize_end - desensitize_start:.4f}秒")
        idx = TorchTensor.create_from_torch(idx, self.base_device)
        centroids = TorchTensor.create_from_torch(centroids, self.base_device)
   
        return (
            ConfidentialTensor(W.shape, W.dtype, (idx, quantizer, vectorquant_config), self), 
            ConfidentialTensor(W.shape, W.dtype, (centroids, quantizer, vectorquant_config), self, is_confidential=True, is_codebook=True),
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
        # idx = idx_tensor.data.clone() if isinstance(idx_tensor.data, torch.Tensor) else idx_tensor.data
        # centroids = centroids_tensor.data.clone() if isinstance(centroids_tensor.data, torch.Tensor) else centroids_tensor.data
        idx = idx_tensor.data
        centroids = centroids_tensor.data 
        # 获取形状
        batch_size, rows, cols = idx.shape
        _, centroids_G, n_centroids, vq_dim = centroids.shape

        # 保存原始码本-索引映射的副本用于验证
        # original_mapping = {}
        # for n in range(batch_size):
        #     original_mapping[n] = {}
        #     for i in range(n_centroids):
        #         original_mapping[n][i] = centroids[n, 0, i, :].clone()
    
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
            old_to_new = {old_idx.item(): new_idx 
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
        # verify_consistency(idx, centroids, original_mapping, quantizer)

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
    
    def create_tmp_tensor(self, quantizer):
        """创建临时张量"""
        idx_shape = quantizer.idx_shape
        centroids_shape = quantizer.centroids_shape
        idx = torch.zeros(idx_shape, dtype=np_dtype_to_torch_dtype[quantizer.idx_dtype])
        centroids = torch.zeros(centroids_shape, dtype=np_dtype_to_torch_dtype[quantizer.centroids_dtype])
        # idx = TorchTensor.create_from_torch(idx, self.base_device)
        # centroids = TorchTensor.create_from_torch(centroids, self.base_device)
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
            # 获取关键参数
        vq_dim = self.vq_dim
        n_centroids = self.n_centroids
    
        W_reshaped = w.reshape(self.groups_per_column, -1, vq_dim)  # G x N x D
    
            # 向量化计算：所有组的min和max同时计算
        min_vals = W_reshaped.min(dim=1)[0]  # 形状: [G, D]
        max_vals = W_reshaped.max(dim=1)[0]  # 形状: [G, D]
    
        # 计算范围，确保不为零
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals > 0, range_vals, torch.ones_like(range_vals))
    
        # 创建线性分布并调整维度以便广播
        t_values = torch.linspace(0, 1, n_centroids, device=w.device).view(1, -1, 1)
    
        # 调整min_vals和range_vals的维度以便广播
        min_vals = min_vals.unsqueeze(1)   # [G, 1, D]
        range_vals = range_vals.unsqueeze(1)  # [G, 1, D]
    
        # 利用广播机制一次性计算所有码本向量
        centroids = min_vals + t_values * range_vals

        # return super().find_params(w)
        return centroids

    def dequantize(self, idx_tensor, centroids_tensor):
        import time
        total_start = time.time()
        prep_start = time.time()
        idx = idx_tensor.data  # 索引张量
        centroids = centroids_tensor.data  # 码本张量
        B, R, C = idx.shape
        D = self.vq_dim
        prep_end = time.time()
        prep_time = prep_end - prep_start
    
        index_start = time.time()
        cent_flat = centroids.view(B*R, -1, D)      # [B*R, K, D]
        idx_flat  = idx.view(B*R, C) 
        index_end = time.time()
        index_time = index_end - index_start

        reshape_start = time.time()
        idx_exp   = idx_flat.unsqueeze(-1).expand(-1, -1, D).to(torch.int64)
        lookup    = torch.gather(cent_flat, 1, idx_exp)  # [B*R, C, D]

        # 最后合并第二、三维度 [rows, batch_size, cols*vq_dim] -> [rows, batch_size*cols*vq_dim]
        output = lookup.reshape(B, R, C*D).permute(1, 0, 2).reshape(R, B*C*D)
        reshape_end = time.time()
        reshape_time = reshape_end - reshape_start

        # 总时间统计
        total_end = time.time()
        total_time = total_end - total_start
    
        # 打印性能数据
        print(f"解量化性能分析 (形状: {R}x{B*C*self.vq_dim}):")
        print(f"  数据准备时间: {prep_time*1000:.2f}ms ({prep_time/total_time*100:.1f}%)")
        print(f"  索引创建时间: {index_time*1000:.2f}ms ({index_time/total_time*100:.1f}%)")
        print(f"  维度重塑时间: {reshape_time*1000:.2f}ms ({reshape_time/total_time*100:.1f}%)")
        print(f"  总执行时间: {total_time*1000:.2f}ms")
    
        return output



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



