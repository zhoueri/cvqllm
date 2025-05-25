# 1. 创建新文件 flexllmgen/utils/memory_monitor.py
import gc
import psutil
import os
import sys
import torch

def print_memory_usage(location="未指定位置"):
    """打印当前内存使用情况"""
    # 获取当前进程
    process = psutil.Process(os.getpid())
    # 获取内存信息 (RSS: 实际物理内存使用量)
    memory_info = process.memory_info()
    
    # 获取PyTorch GPU内存信息
    gpu_allocated = 0
    gpu_reserved = 0
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
    
    # 获取已分配的张量数量
    tensor_count = 0
    tensor_memory = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor_count += 1
                tensor_memory += obj.element_size() * obj.nelement()
        except:
            pass
    
    print(f"位置: {location}")
    print(f"CPU内存: {memory_info.rss / (1024 ** 3):.3f} GB")
    print(f"PyTorch张量: {tensor_count}个, 共{tensor_memory / (1024 ** 3):.3f} GB")
    print(f"GPU已分配: {gpu_allocated:.3f} GB, 保留: {gpu_reserved:.3f} GB")
    print("-" * 50)
    sys.stdout.flush()  # 确保立即输出