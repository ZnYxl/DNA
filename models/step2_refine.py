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

def refine_low_confidence_reads(embeddings, labels, low_conf_mask, 
                                centroids, delta):
    """
    ✅ Phase B: 簇修正（只处理低置信度reads）
    低置信度reads重新分配或标记为噪声
    
    Args:
        embeddings: (N, D)
        labels: (N,) 当前标签
        low_conf_mask: (N,) 低置信度mask
        centroids: dict[label] -> (D,)
        delta: float, 距离阈值
    
    Returns:
        new_labels: (N,) 修正后的标签
        noise_mask: (N,) bool, 新增噪声mask
        stats: dict, 统计信息
    """
    new_labels = labels.clone()
    noise_mask = torch.zeros_like(labels, dtype=torch.bool)
    
    # 统计信息
    reassigned = 0
    marked_noise = 0
    
    low_conf_indices = torch.where(low_conf_mask)[0]
    
    print(f"\n   🔄 只处理 {len(low_conf_indices)} 个低置信度reads...")
    
    for idx in low_conf_indices:
        i = idx.item()
        zi = embeddings[i]
        
        # 找最近的簇中心
        best_k = None
        best_dist = float('inf')
        
        for k, ck in centroids.items():
            dist = torch.norm(zi - ck).item()
            if dist < best_dist:
                best_dist = dist
                best_k = k
        
        # 决策规则
        if best_k is not None and best_dist < delta:
            # ✅ 重新分配到最近簇
            if new_labels[i] != best_k:
                reassigned += 1
            new_labels[i] = best_k
        else:
            # ❌ 标记为噪声
            new_labels[i] = -1
            noise_mask[i] = True
            marked_noise += 1
    
    print(f"   ✅ 修正完成:")
    print(f"      重新分配: {reassigned}")
    print(f"      标记噪声: {marked_noise}")
    print(f"      保持不变: {len(low_conf_indices) - reassigned - marked_noise}")
    
    stats = {
        'reassigned': reassigned,
        'marked_noise': marked_noise,
        'kept_unchanged': len(low_conf_indices) - reassigned - marked_noise
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