# models/step2_refine.py
"""
Step2: Evidence-Guided Cluster Refinement
核心：用Step1学到的证据强度来修正簇结构
✅ 相对不确定性原则：簇内比较，不是全局阈值
"""
import torch
import torch.nn.functional as F

def split_confidence_by_percentile(strength, cluster_labels, p=0.2):
    """
    ✅ Phase A: 簇内相对证据筛选（核心修改）
    每个簇里，取strength最低的p%作为低置信度
    
    Args:
        strength: (N,) evidence strength
        cluster_labels: (N,) 簇标签
        p: float, 低置信度百分比 (0.2 = 20%)
    
    Returns:
        low_conf_mask: (N,) bool, True表示低置信度
        stats: dict, 统计信息
    """
    low_conf_mask = torch.zeros_like(cluster_labels, dtype=torch.bool)
    stats = {'processed_clusters': 0, 'skipped_clusters': 0, 'total_low_conf': 0}
    
    unique_labels = torch.unique(cluster_labels)
    
    print(f"   🎯 簇内相对筛选 (p={p:.1%}):")
    
    for c in unique_labels:
        if c < 0:  # 跳过噪声
            continue
            
        mask = cluster_labels == c
        cluster_size = mask.sum().item()
        
        if cluster_size < 5:  # 太小的簇跳过
            print(f"      簇{c}: {cluster_size} reads (太小，跳过)")
            stats['skipped_clusters'] += 1
            continue
        
        # 该簇的strength
        s = strength[mask]
        tau = torch.quantile(s, p)  # 第p分位数作为阈值
        
        # 标记该簇内的低置信度reads
        cluster_low_conf = s <= tau
        low_conf_mask[mask] = cluster_low_conf
        
        low_count = cluster_low_conf.sum().item()
        stats['total_low_conf'] += low_count
        stats['processed_clusters'] += 1
        
        print(f"      簇{c}: {cluster_size} reads, τ={tau:.3f}, 低置信度={low_count} ({low_count/cluster_size:.1%})")
    
    high_conf_mask = ~low_conf_mask
    
    print(f"\n   📊 相对筛选结果:")
    print(f"      处理簇数: {stats['processed_clusters']}")
    print(f"      跳过簇数: {stats['skipped_clusters']}")
    print(f"      高置信度: {high_conf_mask.sum()}/{len(strength)} ({high_conf_mask.float().mean():.1%})")
    print(f"      低置信度: {low_conf_mask.sum()}/{len(strength)} ({low_conf_mask.float().mean():.1%})")
    
    return low_conf_mask, stats

def compute_cluster_centroids(embeddings, labels, high_conf_mask):
    """
    ✅ [性能优化版] 快速计算簇中心
    复杂度降低为 O(N)，适配 100万+ 数据量
    """
    print(f"\n   🧮 正在快速计算簇中心 (Total Reads: {len(labels)})...")
    
    device = embeddings.device
    
    # 1. 过滤：只保留高置信度且非噪声的reads
    # label >= 0 且 high_conf_mask 为 True
    valid_mask = (labels >= 0) & high_conf_mask
    
    valid_embeddings = embeddings[valid_mask] # (M, D)
    valid_labels = labels[valid_mask]         # (M,)
    
    if len(valid_labels) == 0:
        print("   ⚠️ 没有有效的高置信度Reads用于计算中心")
        return {}, {}

    # 2. 获取所有出现的簇ID
    unique_cluster_ids = torch.unique(valid_labels)
    max_id = int(valid_labels.max().item())
    
    # 3. 初始化累加器 (使用 max_id + 1 大小的张量作为散列表)
    # sum_embeddings[k] 存储簇 k 的向量和
    sum_embeddings = torch.zeros(max_id + 1, embeddings.shape[1], device=device)
    # count_reads[k] 存储簇 k 的数量
    count_reads = torch.zeros(max_id + 1, device=device)
    
    # 4. 核心优化：使用 scatter_add 或 index_add_ (这里用 index_add_ 更通用)
    # 将 valid_embeddings 加到对应的 sum_embeddings 行中
    sum_embeddings.index_add_(0, valid_labels, valid_embeddings)
    
    # 计算计数 (加1.0)
    ones = torch.ones_like(valid_labels, dtype=torch.float)
    count_reads.index_add_(0, valid_labels, ones)
    
    # 5. 转换为字典输出 (保持原有接口兼容性)
    centroids = {}
    cluster_sizes = {}
    
    valid_clusters_count = 0
    
    # 将 Tensor 转回 CPU 处理字典 (因为此时 K 只有 10000，循环很快)
    # 避免在 GPU 上做大规模字典操作
    sum_emb_cpu = sum_embeddings.cpu()
    counts_cpu = count_reads.cpu()
    unique_ids_cpu = unique_cluster_ids.cpu().numpy()
    
    for k in unique_ids_cpu:
        count = counts_cpu[k].item()
        if count < 2: # 保持你之前的逻辑：至少2个reads
            continue
            
        # 计算平均值
        centroid = sum_emb_cpu[k] / count
        
        centroids[int(k)] = centroid
        cluster_sizes[int(k)] = int(count)
        valid_clusters_count += 1
        
    print(f"   📍 簇中心计算完成:")
    print(f"      有效簇数: {valid_clusters_count}")
    if cluster_sizes:
        avg_size = sum(cluster_sizes.values()) / len(cluster_sizes)
        print(f"      平均有效簇大小: {avg_size:.1f}")
        
    return centroids, cluster_sizes

def refine_low_confidence_reads(embeddings, labels, low_conf_mask, centroids, delta):
    """
    ✅ [向量化优化版] 
    使用矩阵运算替代双重循环，秒级完成 20万 x 1万 的匹配
    """
    new_labels = labels.clone()
    noise_mask = torch.zeros_like(labels, dtype=torch.bool)
    
    # 提取低置信度的 embeddings
    low_conf_indices = torch.where(low_conf_mask)[0]
    num_low_conf = len(low_conf_indices)
    
    if num_low_conf == 0:
        return new_labels, noise_mask, {'reassigned': 0, 'marked_noise': 0, 'kept_unchanged': 0}

    print(f"\n   🔄 正在批量修正 {num_low_conf} 个低置信度reads...")
    
    # 1. 准备簇中心矩阵
    # 将字典转换为 tensor: (K, D)
    sorted_cluster_ids = sorted(centroids.keys())
    cluster_matrix = torch.stack([centroids[k] for k in sorted_cluster_ids]) # (K, D)
    cluster_ids_tensor = torch.tensor(sorted_cluster_ids, device=embeddings.device) # (K,)
    
    # 2. 准备查询向量
    query_embeddings = embeddings[low_conf_indices] # (M, D)
    
    # 3. 计算距离矩阵 (M, K)
    # 为了防止显存爆炸 (如果 M*K 很大)，我们可以分块计算
    # 20万 * 1万 * 4 bytes ≈ 8GB，如果你显存有24G，可以直接算。保险起见分块。
    
    batch_size = 5000 # 每次处理 5000 个 reads
    reassigned = 0
    marked_noise = 0
    
    for i in range(0, num_low_conf, batch_size):
        end = min(i + batch_size, num_low_conf)
        batch_queries = query_embeddings[i:end] # (B, D)
        
        # 计算该批次到所有簇中心的距离 (B, K)
        dists = torch.cdist(batch_queries, cluster_matrix)
        
        # 找到最近的簇
        min_dists, min_indices = torch.min(dists, dim=1) # (B,)
        
        # 获取对应的 Cluster ID
        best_cluster_ids = cluster_ids_tensor[min_indices]
        
        # 决策
        # 满足 delta 阈值
        valid_mask = min_dists < delta
        
        # 当前批次在全局的索引
        global_indices = low_conf_indices[i:end]
        
        # 1. 重新分配 (valid_mask 为 True 的部分)
        valid_indices = global_indices[valid_mask]
        new_assignments = best_cluster_ids[valid_mask]
        
        # 统计重新分配的数量 (标签发生变化的)
        original_labels = labels[valid_indices]
        reassigned += (original_labels != new_assignments).sum().item()
        
        new_labels[valid_indices] = new_assignments
        
        # 2. 标记噪声 (valid_mask 为 False 的部分)
        noise_indices = global_indices[~valid_mask]
        new_labels[noise_indices] = -1
        noise_mask[noise_indices] = True
        marked_noise += len(noise_indices)

    print(f"   ✅ 修正完成:")
    print(f"      重新分配: {reassigned}")
    print(f"      标记噪声: {marked_noise}")
    print(f"      保持不变: {num_low_conf - reassigned - marked_noise}")

    stats = {
        'reassigned': reassigned,
        'marked_noise': marked_noise,
        'kept_unchanged': num_low_conf - reassigned - marked_noise
    }

    return new_labels, noise_mask, stats

def compute_adaptive_delta(embeddings, centroids, percentile=10):
    """
    ✅ 自适应计算delta阈值
    
    Args:
        embeddings: (N, D)
        centroids: dict[label] -> (D,)
        percentile: int, 百分位数（接收最近的X%）
    
    Returns:
        delta: float
    """
    all_distances = []
    
    for k, ck in centroids.items():
        dists = torch.norm(embeddings - ck.unsqueeze(0), dim=1)
        all_distances.append(dists)
    
    all_distances = torch.cat(all_distances)
    delta = torch.quantile(all_distances, percentile / 100.0).item()
    
    print(f"   🎯 自适应delta: {delta:.4f} (接收最近{percentile}%的reads)")
    
    return delta