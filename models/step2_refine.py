# models/step2_refine.py
"""
Step2: Evidence-Guided Cluster Refinement
核心：用Step1学到的证据强度来修正簇结构
"""
import torch
import torch.nn.functional as F

def select_high_confidence_reads(strength, tau=None, quantile=0.5):
    """
    ✅ Phase A: 证据筛选
    根据evidence strength区分高/低置信度reads
    
    Args:
        strength: (N,) 每条read的evidence strength
        tau: float or None, 自定义阈值
        quantile: float, 分位数（当tau=None时使用）
    
    Returns:
        high_conf_mask: (N,) bool, True表示高置信度
        tau_used: float, 实际使用的阈值
    """
    if tau is None:
        tau = torch.quantile(strength, quantile)
    
    high_conf_mask = strength >= tau
    
    print(f"   📊 置信度统计:")
    print(f"      阈值 τ: {tau:.4f}")
    print(f"      高置信度: {high_conf_mask.sum()}/{len(strength)} ({high_conf_mask.float().mean()*100:.1f}%)")
    print(f"      低置信度: {(~high_conf_mask).sum()}/{len(strength)} ({(~high_conf_mask).float().mean()*100:.1f}%)")
    
    return high_conf_mask, tau


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
    print(f"      平均簇大小: {sum(cluster_sizes.values())/len(cluster_sizes):.1f}")
    
    return centroids, cluster_sizes


def refine_low_confidence_reads(embeddings, labels, high_conf_mask, 
                                centroids, delta):
    """
    ✅ Phase B: 簇修正
    低置信度reads重新分配或标记为噪声
    
    Args:
        embeddings: (N, D)
        labels: (N,) 当前标签
        high_conf_mask: (N,)
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
    
    low_conf_indices = torch.where(~high_conf_mask)[0]
    
    print(f"\n   🔄 处理 {len(low_conf_indices)} 个低置信度reads...")
    
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
