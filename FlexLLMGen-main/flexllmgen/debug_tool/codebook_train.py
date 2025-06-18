import torch
from torch.nn import functional as F
import vector_quant
from vector_quant import simple_vector_quantize_gpu, simple_vector_dequantize_gpu_optimized

def train_codebook_simple(W, groupsize, vq_dim, G_centroids, n_centroids, 
                         kmeans_iters=10, init_method="kmeans++", verbose=True):
    """
    简单的码本训练函数，使用K-means聚类
    
    Args:
        W: 输入张量 (x, y, z)
        groupsize: 每组的列数
        vq_dim: 每个向量的维度  
        G_centroids: 码本分组数
        n_centroids: 每个码本的中心点数量
        kmeans_iters: K-means迭代次数
        init_method: 初始化方法 ("kmeans++", "random", "uniform")
        verbose: 是否打印训练过程
    
    Returns:
        centroids: 训练好的码本 (n_groups, G_centroids, n_centroids, vq_dim)
    """
    import time
    
    if verbose:
        print("=== 码本训练开始 ===")
    
    assert len(W.shape) == 3, "Only 3D tensor is supported"
    x, y, z = W.shape
    W1 = W.view(x*y, z).float()
    
    n_groups = z // groupsize
    num_vectors = x * y
    vectors_per_g = num_vectors // G_centroids
    actual_vectors = vectors_per_g * G_centroids
    
    if verbose:
        print(f"输入形状: {W.shape} -> {W1.shape}")
        print(f"分组数: {n_groups}, 每组大小: {groupsize}")
        print(f"向量维度: {vq_dim}, 每个码本中心点数: {n_centroids}")
        print(f"码本分组数: {G_centroids}, 每组向量数: {vectors_per_g}")
    
    # 初始化码本
    device = W.device
    centroids = torch.zeros((n_groups, G_centroids, n_centroids, vq_dim), 
                           dtype=torch.float32, device=device)
    
    total_start_time = time.time()
    
    # 为每个组训练码本
    for n_group in range(n_groups):
        group_start_time = time.time()
        
        # 提取当前组的数据
        col_start = n_group * groupsize
        col_end = min(col_start + groupsize, z)
        W_group = W1[:, col_start:col_end].clone()
        
        if verbose:
            print(f"\n--- 训练第 {n_group+1}/{n_groups} 组码本 (列 {col_start}:{col_end}) ---")
        
        # 处理每个vq_dim向量位置
        for j in range(0, col_end - col_start, vq_dim):
            if j + vq_dim <= col_end - col_start:
                vector_idx = j // vq_dim
                
                # 提取当前向量
                w = W_group[:, j:j+vq_dim]  # (num_vectors, vq_dim)
                
                if actual_vectors > 0:
                    # 重塑为分组结构
                    w_truncated = w[:actual_vectors]
                    w_reshaped = w_truncated.reshape(G_centroids, vectors_per_g, vq_dim)
                    
                    # 为每个码本分组训练聚类中心
                    for g in range(G_centroids):
                        vectors_g = w_reshaped[g]  # (vectors_per_g, vq_dim)
                        
                        if verbose and vector_idx == 0:
                            print(f"  训练码本分组 {g+1}/{G_centroids}, 向量数: {vectors_g.shape[0]}")
                        
                        # 执行K-means聚类
                        centers = kmeans_clustering(
                            vectors_g, n_centroids, kmeans_iters, 
                            init_method, verbose=(verbose and g == 0 and vector_idx == 0)
                        )
                        
                        centroids[n_group, g, :, :] = centers
        
        group_end_time = time.time()
        if verbose:
            print(f"第 {n_group+1} 组训练完成，耗时: {group_end_time - group_start_time:.3f}秒")
    
    total_end_time = time.time()
    if verbose:
        print(f"\n=== 码本训练完成 ===")
        print(f"总耗时: {total_end_time - total_start_time:.3f}秒")
        print(f"码本形状: {centroids.shape}")
    
    return centroids


def kmeans_clustering(data, n_centers, max_iters=10, init_method="kmeans++", verbose=False):
    """
    K-means聚类算法实现
    
    Args:
        data: 输入数据 (n_samples, n_features)
        n_centers: 聚类中心数量
        max_iters: 最大迭代次数
        init_method: 初始化方法
        verbose: 是否打印过程
    
    Returns:
        centers: 聚类中心 (n_centers, n_features)
    """
    n_samples, n_features = data.shape
    device = data.device
    
    # 如果数据点数量少于聚类中心数量，直接返回数据点加随机扩充
    if n_samples <= n_centers:
        if verbose:
            print(f"    数据点数({n_samples}) <= 聚类中心数({n_centers})，使用数据点直接初始化")
        
        centers = torch.zeros(n_centers, n_features, device=device)
        centers[:n_samples] = data
        
        # 用数据的均值加随机噪声填充剩余中心
        if n_samples < n_centers:
            data_mean = data.mean(dim=0)
            data_std = data.std(dim=0) + 1e-6
            for i in range(n_samples, n_centers):
                centers[i] = data_mean + torch.randn(n_features, device=device) * data_std * 0.1
        
        return centers
    
    # 初始化聚类中心
    centers = initialize_centers(data, n_centers, init_method)
    
    if verbose:
        print(f"    K-means: {n_samples}个向量 -> {n_centers}个中心")
    
    for iter_idx in range(max_iters):
        # 计算距离并分配样本到最近的中心
        distances = torch.cdist(data, centers)  # (n_samples, n_centers)
        assignments = distances.argmin(dim=1)   # (n_samples,)
        
        # 更新聚类中心
        new_centers = torch.zeros_like(centers)
        
        for i in range(n_centers):
            mask = (assignments == i)
            center_counts = mask.sum()
            
            if center_counts > 0:
                new_centers[i] = data[mask].mean(dim=0)
            else:
                # 如果某个中心没有分配到任何点，随机重新初始化
                new_centers[i] = data[torch.randint(n_samples, (1,))].squeeze()
        
        # 检查收敛
        center_shift = torch.norm(new_centers - centers, dim=1).max()
        centers = new_centers
        
        if verbose and (iter_idx + 1) % 5 == 0:
            print(f"      迭代 {iter_idx+1}/{max_iters}, 中心偏移: {center_shift:.6f}")
        
        # 收敛判断
        if center_shift < 1e-6:
            if verbose:
                print(f"      在第 {iter_idx+1} 次迭代收敛")
            break
    
    return centers


def initialize_centers(data, n_centers, method="kmeans++"):
    """
    初始化聚类中心
    
    Args:
        data: 输入数据 (n_samples, n_features)
        n_centers: 中心数量
        method: 初始化方法
    
    Returns:
        centers: 初始化的中心 (n_centers, n_features)
    """
    n_samples, n_features = data.shape
    device = data.device
    
    if method == "random":
        # 随机选择数据点作为初始中心
        indices = torch.randperm(n_samples, device=device)[:n_centers]
        return data[indices].clone()
    
    elif method == "uniform":
        # 在数据范围内均匀分布初始化
        data_min = data.min(dim=0)[0]
        data_max = data.max(dim=0)[0]
        centers = torch.rand(n_centers, n_features, device=device)
        centers = data_min + centers * (data_max - data_min)
        return centers
    
    elif method == "kmeans++":
        # K-means++初始化
        centers = torch.zeros(n_centers, n_features, device=device)
        
        # 随机选择第一个中心
        centers[0] = data[torch.randint(n_samples, (1,), device=device)]
        
        for i in range(1, n_centers):
            # 计算每个点到最近已选中心的距离
            distances = torch.cdist(data, centers[:i])  # (n_samples, i)
            min_distances = distances.min(dim=1)[0]     # (n_samples,)
            
            # 按距离平方进行概率选择
            probabilities = min_distances ** 2
            probabilities = probabilities / probabilities.sum()
            
            # 根据概率选择下一个中心
            selected_idx = torch.multinomial(probabilities, 1)
            centers[i] = data[selected_idx]
        
        return centers
    
    else:
        raise ValueError(f"未知的初始化方法: {method}")


def test_codebook_training():
    """
    测试码本训练和量化的完整流程
    """
    print("=== 码本训练和量化测试 ===")
    
    # 创建测试数据
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x, y, z = 2, 4, 8
    
    # 创建有一定结构的数据，便于观察聚类效果
    W = torch.randn(x, y, z, device=device) * 2 + 1
    # 添加一些模式让聚类更明显
    W[:, :, :4] += 5  # 前4列加偏移
    W[:, :, 4:] += 10  # 后4列加更大偏移
    
    # 参数设置
    groupsize = 4
    vq_dim = 2
    G_centroids = 2
    n_centroids = 4
    
    print(f"输入数据形状: {W.shape}")
    print(f"设备: {device}")
    
    print(f"\n原始数据示例:")
    print(f"W[0,0,:]: {W[0, 0, :].tolist()}")
    
    # === 1. 训练码本 ===
    print(f"\n=== 1. 训练码本 ===")
    centroids = train_codebook_simple(
        W, groupsize, vq_dim, G_centroids, n_centroids,
        kmeans_iters=20, init_method="kmeans++", verbose=True
    )
    
    print(f"\n训练后的码本:")
    for group_idx in range(centroids.shape[0]):
        print(f"组 {group_idx}:")
        for g in range(G_centroids):
            print(f"  码本分组 {g}: {centroids[group_idx, g]}")
    
    # === 2. 执行量化 ===
    print(f"\n=== 2. 执行量化 ===")
    indices = simple_vector_quantize_gpu(W, centroids, groupsize, vq_dim, G_centroids)
    print(f"量化索引形状: {indices.shape}")
    print(f"索引值范围: [{indices.min().item()}, {indices.max().item()}]")
    
    # === 3. 反量化验证 ===
    print(f"\n=== 3. 反量化验证 ===")
    W_reconstructed = simple_vector_dequantize_gpu_optimized(
        indices, centroids, W.shape, groupsize, vq_dim, G_centroids
    )
    
    # === 4. 计算重构误差 ===
    print(f"\n=== 4. 重构质量评估 ===")
    mse_error = torch.mean((W - W_reconstructed) ** 2).item()
    max_error = torch.max(torch.abs(W - W_reconstructed)).item()
    cosine_sim = F.cosine_similarity(W.view(-1), W_reconstructed.view(-1), dim=0).item()
    
    print(f"重构张量形状: {W_reconstructed.shape}")
    print(f"均方误差: {mse_error:.6f}")
    print(f"最大绝对误差: {max_error:.6f}")
    print(f"余弦相似度: {cosine_sim:.6f}")
    
    # === 5. 对比结果 ===
    print(f"\n=== 5. 原始 vs 重构对比 ===")
    print(f"原始[0,0,:]: {W[0, 0, :].tolist()}")
    print(f"重构[0,0,:]: {W_reconstructed[0, 0, :].tolist()}")
    print(f"差值[0,0,:]: {(W[0, 0, :] - W_reconstructed[0, 0, :]).tolist()}")
    
    # === 6. 质量评估 ===
    if cosine_sim > 0.9:
        print("✓ 码本训练质量优秀")
    elif cosine_sim > 0.7:
        print("⚠ 码本训练质量良好，可考虑调整参数")
    else:
        print("✗ 码本训练质量较差，建议检查参数设置")
    
    # === 7. 压缩比计算 ===
    original_size = W.numel() * 4  # float32 = 4 bytes
    compressed_size = indices.numel() * 1 + centroids.numel() * 4  # indices用1字节，码本用4字节
    compression_ratio = original_size / compressed_size
    
    print(f"\n=== 6. 压缩分析 ===")
    print(f"原始大小: {original_size} bytes")
    print(f"压缩大小: {compressed_size} bytes (索引: {indices.numel()}, 码本: {centroids.numel()*4})")
    print(f"压缩比: {compression_ratio:.2f}x")
    
    return centroids, indices, W_reconstructed


# 将测试函数添加到原有的 test_vector_quantization() 中
def test_vector_quantization_with_training():
    """
    使用训练码本的量化测试
    """
    # 首先测试训练的码本
    test_codebook_training()
    
    # 然后测试原有的固定码本
    print("\n" + "="*60)
    print("对比：使用固定码本的量化结果")



if __name__ == "__main__":
    # 运行码本训练测试
    test_codebook_training()
    
    # 如果想对比固定码本和训练码本的效果，可以运行：
    # test_vector_quantization_with_training()