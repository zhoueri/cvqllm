import torch
import time
import torch.nn.functional as F

def simple_vector_quantize_gpu(W, centroids, groupsize, vq_dim, G_centroids):
    assert len(W.shape) == 3, "Only 3D tensor is supported"
    x, y, z = W.shape
    W1 = W.view(x*y, z)
    W1 = W1.float()
    
    n_groups = z // groupsize
    num_vectors = x * y
    vectors_per_group = groupsize // vq_dim
    
    print(f"\n=== 量化过程详细信息 ===")
    print(f"输入张量: {W.shape} -> 重塑为: {W1.shape}")
    print(f"总分组数: {n_groups}, 每组向量数: {vectors_per_group}")
    print(f"每组处理 {num_vectors} 个向量，分为 {G_centroids} 个码本分组")
    
    # 初始化索引张量
    indices = torch.zeros((n_groups, num_vectors, vectors_per_group), 
                         dtype=torch.long, device=W1.device)
    print(f"索引张量形状: {indices.shape}")
    
    n_group = 0
    for i in range(0, W1.shape[1], groupsize):
        end = min(i + groupsize, W1.shape[1])
        W_group = W1[:, i:end].clone()
        codebook = centroids[n_group] # G x K x D
        
        print(f"\n--- 处理第 {n_group} 组 (列 {i}:{end}) ---")
        print(f"W_group 形状: {W_group.shape}")
        print(f"码本形状: {codebook.shape}")
        
        for j in range(0, end-i, vq_dim):  # 修正1：直接按vq_dim步长循环
            if j + vq_dim <= end - i:
                vector_idx = j // vq_dim
                print(f"\n  处理 vector_idx={vector_idx}, 列位置 {j}:{j+vq_dim}")
                
                w = W_group[:, j:j+vq_dim]  # (num_vectors, vq_dim)
                print(f"  提取的向量 w 形状: {w.shape}")
                print(f"  w 内容 (前4个向量):")
                print(w[:4])
                
                # 修正2：处理分组逻辑
                vectors_per_g = num_vectors // G_centroids
                actual_vectors = vectors_per_g * G_centroids
                
                if actual_vectors > 0:
                    # 只处理能被G_centroids整除的部分
                    w_truncated = w[:actual_vectors]  # (actual_vectors, vq_dim)
                    w_reshaped = w_truncated.reshape(G_centroids, vectors_per_g, vq_dim)  # G x N x D
                    print(f"  重塑后 w 形状: {w_reshaped.shape}")
                    
                    # 修正3：正确的归一化维度
                    codebook_norm = F.normalize(codebook, dim=2)
                    w_norm = F.normalize(w_reshaped, dim=2)  # 修正：dim=2
                    
                    print(f"  归一化完成:")
                    print(f"    码本归一化形状: {codebook_norm.shape}")
                    print(f"    向量归一化形状: {w_norm.shape}")
                    
                    # 计算相似度矩阵
                    sim = torch.matmul(w_norm, codebook_norm.transpose(1, 2))  # (G_centroids, vectors_per_g, n_centroids)
                    print(f"  相似度矩阵形状: {sim.shape}")
                    print(f"  相似度范围: [{sim.min().item():.3f}, {sim.max().item():.3f}]")
                    
                    # 找到最相似的码本索引
                    best_indices = sim.argmax(dim=2)  # (G_centroids, vectors_per_g)
                    print(f"  最佳索引形状: {best_indices.shape}")
                    print(f"  各组的最佳索引:")
                    for g in range(G_centroids):
                        print(f"    组{g}: {best_indices[g].tolist()}")
                    
                    # 重塑回原始向量顺序
                    best_indices_flat = best_indices.transpose(0, 1).reshape(-1)  # (actual_vectors,)
                    print(f"  展平后索引形状: {best_indices_flat.shape}")
                    print(f"  展平后索引内容: {best_indices_flat.tolist()}")
                    
                    # 修正4：直接存储到对应的vector_idx位置，不使用ptr_to_num
                    if vector_idx < vectors_per_group:
                        print(f"  存储到 indices[{n_group}, :len(best_indices_flat), {vector_idx}]")
                        
                        # 确保不越界
                        store_length = min(len(best_indices_flat), indices.shape[1])
                        indices[n_group, :store_length, vector_idx] = best_indices_flat[:store_length]
                        
                        print(f"  实际存储长度: {store_length}")
                        print(f"  存储后 indices[{n_group}, :, {vector_idx}]: {indices[n_group, :store_length, vector_idx].tolist()}")
                        
                        # 验证匹配结果
                        print(f"  验证前3个向量的匹配:")
                        for vec_id in range(min(3, vectors_per_g)):
                            for g in range(G_centroids):
                                global_vec_id = g * vectors_per_g + vec_id
                                if global_vec_id < actual_vectors:
                                    original_vec = w_reshaped[g, vec_id]
                                    selected_idx = best_indices[g, vec_id].item()
                                    selected_centroid = codebook[g, selected_idx]
                                    cosine_sim = F.cosine_similarity(original_vec, selected_centroid, dim=0)
                                    print(f"    向量[{g},{vec_id}]: {original_vec.tolist()} -> "
                                          f"码本[{g},{selected_idx}]: {selected_centroid.tolist()} "
                                          f"(相似度: {cosine_sim:.3f})")
                    else:
                        print(f"  跳过 vector_idx={vector_idx} (超出范围)")
        
        print(f"组 {n_group} 处理完成")
        n_group += 1
    
    print(f"\n=== 量化完成 ===")
    print(f"最终索引张量形状: {indices.shape}")
    print(f"索引值范围: [{indices.min().item()}, {indices.max().item()}]")
    
    return indices


def simple_vector_dequantize_gpu_optimized(indices, centroids, original_shape, groupsize, vq_dim, G_centroids):
    """
    优化版本的向量恢复函数，提高计算效率
    """
    total_start = time.time()
    
    # 参数提取
    x, y, z = original_shape
    num_vectors = x * y
    n_groups = z // groupsize
    vectors_per_g = num_vectors // G_centroids
    actual_vectors = vectors_per_g * G_centroids
    
    # 设备统一
    device = indices.device
    centroids = centroids.to(device)
    
    # 初始化输出
    W_restored_2d = torch.zeros((num_vectors, z), dtype=centroids.dtype, device=device)
    
    print(f"\n=== 反量化过程 ===")
    print(f"处理向量数: {actual_vectors}/{num_vectors}")
    
    # 批量处理优化
    for group_idx in range(n_groups):
        col_start = group_idx * groupsize
        col_end = min(col_start + groupsize, z)
        actual_groupsize = col_end - col_start
        
        print(f"\n处理第 {group_idx} 组 (列 {col_start}:{col_end})")
        
        # 获取当前组数据
        codebook = centroids[group_idx]  # (G_centroids, n_centroids, vq_dim)
        group_indices = indices[group_idx]  # (num_vectors, vectors_per_group)
        
        # 批量处理每个vq_dim向量
        for vec_idx in range(0, actual_groupsize, vq_dim):
            if vec_idx + vq_dim <= actual_groupsize:
                vector_position = vec_idx // vq_dim
                print(f"  处理 vector_position={vector_position}, 列位置 {vec_idx}:{vec_idx+vq_dim}")
                
                # 获取索引
                current_indices = group_indices[:actual_vectors, vector_position]
                print(f"  使用的索引: {current_indices.tolist()}")
                
                # 分组处理
                restored_parts = []
                for g in range(G_centroids):
                    start_v = g * vectors_per_g
                    end_v = start_v + vectors_per_g
                    
                    if end_v <= actual_vectors:
                        g_indices = current_indices[start_v:end_v]
                        g_codebook = codebook[g]
                        
                        print(f"    组{g}: 索引{g_indices.tolist()}")
                        
                        # 使用embedding进行批量查找
                        restored_part = F.embedding(g_indices, g_codebook)
                        restored_parts.append(restored_part)
                        
                        print(f"    组{g}: 恢复向量形状{restored_part.shape}")
                
                # 合并并存储
                if restored_parts:
                    restored_full = torch.cat(restored_parts, dim=0)
                    target_col_start = col_start + vec_idx
                    target_col_end = target_col_start + vq_dim
                    
                    print(f"  合并后形状: {restored_full.shape}")
                    print(f"  存储到列 {target_col_start}:{target_col_end}")
                    
                    if target_col_end <= z:
                        W_restored_2d[:restored_full.shape[0], target_col_start:target_col_end] = restored_full
    
    # 重塑为原始3D形状
    W_restored = W_restored_2d.view(x, y, z)
    
    total_time = time.time() - total_start
    print(f"反量化完成，耗时: {total_time:.4f}秒，输出形状: {W_restored.shape}")
    
    return W_restored


def test_vector_quantization():
    """
    创建简单的矩阵和码本，测试量化过程并观察索引结果
    """
    print("=== 简单向量量化测试 ===\n")
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # === 1. 创建简单的测试矩阵 ===
    print("\n1. 创建测试矩阵:")
    x, y, z = 2, 4, 8  # 小尺寸便于观察
    W = torch.tensor([
        # 第一个2D矩阵 (x=0)
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],   # y=0
         [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],   # y=1
         [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # y=2
         [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]], # y=3
        # 第二个2D矩阵 (x=1)
        [[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],  # y=0
         [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], # y=1
         [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],# y=2
         [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]# y=3
    ], dtype=torch.float32, device=device)
    
    print(f"   原始矩阵形状: {W.shape}")
    print(f"   重塑后: {W.view(x*y, z).shape}")
    
    # === 2. 设置量化参数 ===
    print("\n2. 量化参数设置:")
    groupsize = 4      # 每组4列
    vq_dim = 2         # 每个向量2维
    G_centroids = 2    # 2个码本分组
    n_centroids = 4    # 每个码本4个中心点
    n_groups = z // groupsize  # 8 // 4 = 2组
    
    print(f"   groupsize: {groupsize} (每组列数)")
    print(f"   vq_dim: {vq_dim} (向量维度)")
    print(f"   G_centroids: {G_centroids} (码本分组数)")
    print(f"   n_centroids: {n_centroids} (每个码本的中心点数)")
    print(f"   n_groups: {n_groups} (总分组数)")
    print(f"   vectors_per_group: {groupsize // vq_dim} (每组向量数)")
    print(f"   vectors_per_g: {(x*y) // G_centroids} (每个码本分组的向量数)")
    
    # === 3. 创建简单的码本 ===
    print("\n3. 创建码本:")
    centroids = torch.zeros(n_groups, G_centroids, n_centroids, vq_dim, device=device)
    
    # 第一组码本 (处理列 0-3)
    centroids[0, 0] = torch.tensor([  # 第一个分组的码本
        [0.0, 1.0],   # 中心点0
        [2.0, 3.0],   # 中心点1
        [4.0, 5.0],   # 中心点2
        [6.0, 7.0]    # 中心点3
    ], device=device)
    
    centroids[0, 1] = torch.tensor([  # 第二个分组的码本
        [1.0, 2.0],   # 中心点0
        [3.0, 4.0],   # 中心点1
        [5.0, 6.0],   # 中心点2
        [7.0, 8.0]    # 中心点3
    ], device=device)
    
    # 第二组码本 (处理列 4-7)
    centroids[1, 0] = torch.tensor([  # 第一个分组的码本
        [4.0, 5.0],   # 中心点0
        [6.0, 7.0],   # 中心点1
        [8.0, 9.0],   # 中心点2
        [10.0, 11.0]  # 中心点3
    ], device=device)
    
    centroids[1, 1] = torch.tensor([  # 第二个分组的码本
        [5.0, 6.0],   # 中心点0
        [7.0, 8.0],   # 中心点1
        [9.0, 10.0],  # 中心点2
        [11.0, 12.0]  # 中心点3
    ], device=device)
    
    print(f"   码本形状: {centroids.shape}")
    print(f"   第0组码本:")
    print(f"     分组0: {centroids[0, 0]}")
    print(f"     分组1: {centroids[0, 1]}")
    
    # === 4. 执行量化 ===
    print("\n4. 执行量化:")
    try:
        indices = simple_vector_quantize_gpu(W, centroids, groupsize, vq_dim, G_centroids)
        
        print("\n5. 量化索引结果:")
        print(f"   索引形状: {indices.shape}")
        print(f"   索引内容:")
        for group_idx in range(indices.shape[0]):
            print(f"   组 {group_idx}:")
            print(f"     {indices[group_idx]}")
        
        # === 6. 验证反量化 ===
        print("\n6. 验证反量化:")
        W_reconstructed = simple_vector_dequantize_gpu_optimized(
            indices, centroids, W.shape, groupsize, vq_dim, G_centroids
        )
        
        # 计算误差
        mse_error = torch.mean((W - W_reconstructed) ** 2).item()
        max_error = torch.max(torch.abs(W - W_reconstructed)).item()
        
        print(f"\n7. 误差分析:")
        print(f"   重构张量形状: {W_reconstructed.shape}")
        print(f"   均方误差: {mse_error:.6f}")
        print(f"   最大绝对误差: {max_error:.6f}")
        
        print(f"\n8. 原始 vs 重构对比:")
        print(f"   原始[0,0,:]: {W[0, 0, :].tolist()}")
        print(f"   重构[0,0,:]: {W_reconstructed[0, 0, :].tolist()}")
        
    except Exception as e:
        print(f"   ✗ 量化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_vector_quantization()