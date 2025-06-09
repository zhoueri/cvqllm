import cuda_aes
import torch
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util import Counter

# 测试密钥
key = "2B7E151628AED2A6ABF7158809CF4F3C"
iv = "3243F6A8885A308D313198A200000000"

def encrypt_tensor_aes_ctr(tensor: torch.Tensor, key_hex: str = key, iv_hex: str = iv) -> torch.Tensor:
    """
    使用AES-CTR 128模式加密PyTorch张量，保持张量形状不变
    
    Args:
        tensor: 输入的PyTorch张量
        key_hex: 16字节的十六进制密钥字符串 (128位)
        iv_hex: 16字节的十六进制初始向量字符串
    
    Returns:
        加密后的张量，形状与输入张量相同
    """
    # 保存原始形状和数据类型
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    original_device = tensor.device
    
    # 将张量移动到CPU进行加密
    tensor_cpu = tensor
    
    # 将密钥和IV从十六进制字符串转换为字节
    key_bytes = bytes.fromhex(key_hex)
    iv_bytes = bytes.fromhex(iv_hex)
    
    # 确保密钥长度为16字节 (128位)
    if len(key_bytes) != 16:
        raise ValueError(f"密钥长度必须为16字节，当前为{len(key_bytes)}字节")
    
    # 将张量数据转换为字节
    tensor_bytes = tensor_cpu.numpy().tobytes()
    
    # 创建CTR模式的Counter对象
    # AES块大小为16字节，使用前12字节作为nonce，后4字节作为计数器
    nonce = iv_bytes[:12]  # 取前12字节作为nonce
    counter = Counter.new(32, nonce=nonce, initial_value=int.from_bytes(iv_bytes[12:16], 'big'))
    
    # 创建AES-CTR加密器
    cipher = AES.new(key_bytes, AES.MODE_CTR, counter=counter)
    
    # 加密数据
    encrypted_bytes = cipher.encrypt(tensor_bytes)
    
    # 将加密后的字节转换回numpy数组
    encrypted_array = np.frombuffer(encrypted_bytes, dtype=tensor_cpu.numpy().dtype)
    
    # 重塑为原始形状
    encrypted_array = encrypted_array.reshape(original_shape)
    
    # 转换回PyTorch张量并移动到原始设备
    encrypted_tensor = torch.from_numpy(encrypted_array.copy()).to(original_device)
    
    return encrypted_tensor

def decrypt_tensor_aes_ctr(encrypted_tensor: torch.Tensor, key_hex: str = key, iv_hex: str = iv) -> torch.Tensor:
    """
    使用AES-CTR 128模式解密PyTorch张量
    
    Args:
        encrypted_tensor: 加密的PyTorch张量
        key_hex: 16字节的十六进制密钥字符串 (128位)
        iv_hex: 16字节的十六进制初始向量字符串
    
    Returns:
        解密后的张量，形状与输入张量相同
    """
    # 保存原始形状和设备
    original_shape = encrypted_tensor.shape
    original_device = encrypted_tensor.device
    
    # 将张量移动到CPU进行解密
    tensor_cpu = encrypted_tensor
    
    # 将密钥和IV从十六进制字符串转换为字节
    key_bytes = bytes.fromhex(key_hex)
    iv_bytes = bytes.fromhex(iv_hex)
    
    # 将张量数据转换为字节
    encrypted_bytes = tensor_cpu.numpy().tobytes()
    
    # 创建CTR模式的Counter对象（与加密时相同）
    nonce = iv_bytes[:12]
    counter = Counter.new(32, nonce=nonce, initial_value=int.from_bytes(iv_bytes[12:16], 'big'))
    
    # 创建AES-CTR解密器
    cipher = AES.new(key_bytes, AES.MODE_CTR, counter=counter)
    
    # 解密数据
    decrypted_bytes = cipher.decrypt(encrypted_bytes)
    
    # 将解密后的字节转换回numpy数组
    decrypted_array = np.frombuffer(decrypted_bytes, dtype=tensor_cpu.numpy().dtype)
    
    # 重塑为原始形状
    decrypted_array = decrypted_array.reshape(original_shape)
    
    # 转换回PyTorch张量并移动到原始设备
    decrypted_tensor = torch.from_numpy(decrypted_array.copy()).to(original_device)
    
    return decrypted_tensor

def encrypt_tensor_cuda_aes_ctr(tensor: torch.Tensor, key_hex: str = key, iv_hex: str = iv) -> torch.Tensor:
    encrypted_tensor, enc_stats = cuda_aes.encrypt_tensor_gpu_direct(
        tensor, key, iv, verbose=True
    )
    return encrypted_tensor

def decrypt_tensor_cuda_aes_ctr(encrypted_tensor: torch.Tensor, key_hex: str = key, iv_hex: str = iv) -> torch.Tensor:
    decrypted_tensor, dec_stats = cuda_aes.decrypt_tensor_gpu_direct(
        encrypted_tensor, key, iv, verbose=True
    )
    return decrypted_tensor

