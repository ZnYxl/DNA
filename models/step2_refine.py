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
    ✅ 只用高置信度reads计算簇中心
    
    Args:
        embeddings: (N, D) Step1的embedding
        labels: (N,) 当前簇标签
        high_conf_mask: (N,) bool, 高置信度mask
    
    Returns:
        centroids: dict[label] -> (D,) 簇中心
        cluster_sizes: dict[label] -> int 簇大小
    """
    centroids = {}
    cluster_sizes = {}
    
    unique_labels = torch.unique(labels)
    valid_clusters = 0
    
    for k in unique_labels:
        if k < 0:  # 跳过噪声
            continue
        
        # ✅ 只用高置信度reads
        mask = (labels == k) & high_conf_mask
        count = mask.sum().item()
        
        if count < 2:  # 至少2个高置信度reads
            print(f"   ⚠️ 簇 {k}: 只有 {count} 个高置信度reads，跳过")
            continue
        
        centroids[int(k.item())] = embeddings[mask].mean(dim=0)
        cluster_sizes[int(k.item())] = count
        valid_clusters += 1
    
    print(f"\n   📍 簇中心统计:")
    print(f"      有效簇数: {valid_clusters}/{len(unique_labels)-1}")  # -1排除噪声
    if cluster_sizes:
        print(f"      平均簇大小: {sum(cluster_sizes.values())/len(cluster_sizes):.1f}")
    
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