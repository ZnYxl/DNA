# models/step2_refine.py
"""
Step2: Evidence-Guided Cluster Refinement (Vectorized Optimized Version)
核心：用Step1学到的证据强度来修正簇结构
✅ 相对不确定性原则：簇内比较
✅ 向量化加速：移除关键路径上的Python循环
"""
import torch
import torch.nn.functional as F

def split_confidence_by_percentile(strength, cluster_labels, p=0.2):
    """
    ✅ Phase A: 簇内相对证据筛选
    每个簇里，取strength最低的p%作为低置信度
    """
    low_conf_mask = torch.zeros_like(cluster_labels, dtype=torch.bool)
    stats = {'processed_clusters': 0, 'skipped_clusters': 0, 'total_low_conf': 0}
    
    unique_labels = torch.unique(cluster_labels)
    
    print(f"    🎯 簇内相对筛选 (p={p:.1%}):")
    
    # 这里保持循环是OK的，因为簇的数量远小于Reads数量，且quantile操作无法简单的全局向量化（每个簇阈值不同）
    for c in unique_labels:
        if c < 0:  # 跳过噪声
            continue
            
        mask = cluster_labels == c
        cluster_size = mask.sum().item()
        
        if cluster_size < 5:  # 太小的簇跳过，直接视为高置信度或由后续逻辑处理
            # print(f"      簇{c}: {cluster_size} reads (太小，保留)")
            stats['skipped_clusters'] += 1
            continue
        
        # 该簇的strength
        s = strength[mask]
        tau = torch.quantile(s, p)  # 第p分位数作为阈值
        
        # 标记该簇内的低置信度reads
        cluster_low_conf = s <= tau
        # 注意：这里需要利用mask索引回填
        # 这种写法在PyTorch中是安全的：low_conf_mask[mask]的shape等于cluster_low_conf
        low_conf_mask[mask] = cluster_low_conf
        
        low_count = cluster_low_conf.sum().item()
        stats['total_low_conf'] += low_count
        stats['processed_clusters'] += 1
        
        # 只有在调试模式下才打印每个簇的详情，防止刷屏
        # print(f"      簇{c}: τ={tau:.3f}, 低置信度={low_count}")
    
    high_conf_mask = ~low_conf_mask
    
    print(f"\n    📊 相对筛选结果:")
    print(f"      处理簇数: {stats['processed_clusters']}")
    print(f"      跳过簇数: {stats['skipped_clusters']}")
    print(f"      高置信度: {high_conf_mask.sum()}/{len(strength)} ({high_conf_mask.float().mean():.1%})")
    print(f"      低置信度: {low_conf_mask.sum()}/{len(strength)} ({low_conf_mask.float().mean():.1%})")
    
    return low_conf_mask, stats

def compute_cluster_centroids(embeddings, labels, high_conf_mask):
    """
    ✅ 向量化计算簇中心 (只用高置信度reads)
    使用 scatter/index_add 加速，避免 Python 循环
    """
    # 1. 筛选有效数据
    valid_mask = (labels >= 0) & high_conf_mask
    if valid_mask.sum() == 0:
        return {}, {}
        
    valid_emb = embeddings[valid_mask]
    valid_labels = labels[valid_mask]
    
    # 2. 映射 Label 到 0..K-1 的连续索引，以便向量化累加
    unique_labels, inverse_indices = torch.unique(valid_labels, return_inverse=True)
    num_clusters = len(unique_labels)
    dim = embeddings.shape[1]
    
    # 3. 初始化累加容器
    sum_emb = torch.zeros(num_clusters, dim, device=embeddings.device)
    counts = torch.zeros(num_clusters, device=embeddings.device)
    
    # 4. 向量化累加 (Scatter Add / Index Add)
    # counts: 统计每个簇有多少个点
    counts.index_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float))
    
    # sum_emb: 累加向量
    # index_add_ 需要 dim 匹配，所以对 inverse_indices 不需要做特殊处理，但 index_add_ 是按行操作的
    sum_emb.index_add_(0, inverse_indices, valid_emb)
    
    # 5. 计算均值
    # clamp避免除以0（虽然逻辑上unique保证了至少有1个，但安全第一）
    means = sum_emb / counts.unsqueeze(1).clamp(min=1)
    
    # 6. 转回 Dict 格式 (兼容后续接口，同时过滤极小簇)
    centroids = {}
    cluster_sizes = {}
    valid_clusters_count = 0
    
    # 将 Tensor 数据转回 CPU 字典
    means_cpu = means  # 保持在原设备或 .cpu() 取决于后续需求，这里建议保持原设备直到最后
    counts_cpu = counts
    unique_labels_cpu = unique_labels
    
    for i in range(num_clusters):
        lbl = int(unique_labels_cpu[i].item())
        cnt = int(counts_cpu[i].item())
        
        if cnt >= 2: # 至少2个高置信度reads才算有效中心
            centroids[lbl] = means_cpu[i] # 保持 Tensor
            cluster_sizes[lbl] = cnt
            valid_clusters_count += 1
    
    print(f"\n    📍 簇中心统计 (Vectorized):")
    print(f"      有效簇数: {valid_clusters_count}/{num_clusters}")
    if cluster_sizes:
        avg_size = sum(cluster_sizes.values()) / len(cluster_sizes)
        print(f"      平均簇大小: {avg_size:.1f}")
    
    return centroids, cluster_sizes

def compute_adaptive_delta(embeddings, labels, centroids, high_conf_mask, percentile=95):
    """
    ✅ 自适应 Delta 计算
    逻辑：计算【高置信度点】到【自身簇中心】的距离分布，取第 p 分位数。
    这意味着：如果一个点距离中心的距离超过了95%的正常点，它就被视为“太远”。
    """
    if not centroids:
        return 0.5 # fallback
        
    valid_mask = (labels >= 0) & high_conf_mask
    valid_indices = torch.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return 0.5
        
    # 为了向量化，我们需要构造一个 "aligned_centroids" 矩阵
    # 即：对于第 i 个 read，找到它 label 对应的 centroid
    
    # 1. 准备数据
    curr_embs = embeddings[valid_indices]
    curr_labels = labels[valid_indices]
    
    # 2. 构造中心张量
    # 并不是所有 valid_labels 都在 centroids 字典里 (因为 centroids 过滤了 count<2 的)
    # 所以我们需要筛选
    keys_tensor = torch.tensor(list(centroids.keys()), device=embeddings.device)
    vals_stack = torch.stack(list(centroids.values()))
    
    # 创建一个查找表 (Label -> Index in vals_stack)
    # 假设 Label 范围可能很大，不能直接用数组 lookup。
    # 这里用一个简单的技巧：只计算那些 label 在 centroids 里的点
    
    # 将 dict 转为 lookup 可能会慢，不如直接遍历计算 distances (如果簇很少)
    # 或者，更简单的方法：
    distances = []
    
    # 这里用循环簇的方式比较安全，因为我们需要“点到自身中心”的距离
    # 且 valid_indices 的数量可能很大，但 unique labels 不会太大
    unique_valid_labels = torch.unique(curr_labels)
    
    for k in unique_valid_labels:
        k_item = int(k.item())
        if k_item not in centroids:
            continue
            
        ck = centroids[k_item] # (D,)
        
        # 找出属于簇 k 的高置信度点
        mask_k = (curr_labels == k)
        embs_k = curr_embs[mask_k]
        
        # 计算距离
        dists_k = torch.norm(embs_k - ck.unsqueeze(0), dim=1)
        distances.append(dists_k)
    
    if not distances:
        print("    ⚠️ 无法计算 Delta，使用默认值 0.5")
        return 0.5
        
    all_distances = torch.cat(distances)
    
    # 取分位数
    delta = torch.quantile(all_distances, percentile / 100.0).item()
    
    print(f"    🎯 自适应 Delta: {delta:.4f} (基于高置信度点分布的 {percentile}%)")
    return delta

def refine_low_confidence_reads(embeddings, labels, low_conf_mask, 
                                centroids, delta):
    """
    ✅ Phase B: 簇修正 (Vectorized Matrix Version)
    使用矩阵运算一次性计算所有低置信度点到所有中心的距离
    """
    low_conf_indices = torch.where(low_conf_mask)[0]
    num_low = len(low_conf_indices)
    
    if num_low == 0 or not centroids:
        return labels, torch.zeros_like(labels, dtype=torch.bool), {'reassigned': 0, 'marked_noise': 0}
    
    print(f"\n    🔄 向量化修正 {num_low} 个低置信度 reads...")
    
    # 1. 准备 Query 数据 (M, D)
    query_embs = embeddings[low_conf_indices]
    
    # 2. 准备 Reference 数据 (Centroids) -> Matrix (K, D)
    # 必须排序 keys 以便后续映射回 label
    sorted_keys = sorted(centroids.keys())
    centroid_labels = torch.tensor(sorted_keys, device=embeddings.device)
    centroid_matrix = torch.stack([centroids[k] for k in sorted_keys])
    
    # 3. 矩阵计算距离 (M, K) - 核心加速点
    # cdist 计算 query 中每一行到 centroid_matrix 中每一行的欧氏距离
    dists_matrix = torch.cdist(query_embs, centroid_matrix)
    
    # 4. 找到最近的簇
    min_dists, min_indices = torch.min(dists_matrix, dim=1) # (M,) values, (M,) indices
    
    # 5. 决策
    # 满足距离阈值的 mask
    assign_mask = min_dists < delta
    
    # 6. 生成新标签
    new_labels = labels.clone()
    noise_mask = torch.zeros_like(labels, dtype=torch.bool)
    
    # A. 重新分配的点
    # 获取 centroid_matrix 的索引 -> 映射回 真实 Label ID
    target_centroid_idx = min_indices[assign_mask]
    target_labels = centroid_labels[target_centroid_idx]
    
    # 获取原始 reads 的全局索引
    reassigned_global_indices = low_conf_indices[assign_mask]
    
    # 赋值
    new_labels[reassigned_global_indices] = target_labels
    
    # B. 标记为噪声的点
    noise_global_indices = low_conf_indices[~assign_mask]
    new_labels[noise_global_indices] = -1
    noise_mask[noise_global_indices] = True
    
    # 统计
    reassigned_count = len(reassigned_global_indices)
    noise_count = len(noise_global_indices)
    unchanged_count = num_low - reassigned_count - noise_count # 逻辑上应该是0，因为要么assign要么noise
    
    print(f"    ✅ 修正完成:")
    print(f"      重新分配: {reassigned_count}")
    print(f"      标记噪声: {noise_count}")
    
    stats = {
        'reassigned': reassigned_count,
        'marked_noise': noise_count
    }
    
    return new_labels, noise_mask, stats
