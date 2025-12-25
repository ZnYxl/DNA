# models/step2_refine.py
"""
Step2: Evidence-Guided Cluster Refinement
核心：用Step1学到的证据强度来修正簇结构
✅ 修复版 v2：解决 "bitwise_and_cuda not implemented for Float" 报错
"""
import torch
import torch.nn.functional as F
import random

def split_confidence_by_percentile(strength, cluster_labels, p=0.2):
    """
    Phase A: 簇内相对证据筛选
    """
    low_conf_mask = torch.zeros_like(cluster_labels, dtype=torch.bool)
    stats = {'processed_clusters': 0, 'skipped_clusters': 0, 'total_low_conf': 0}

    unique_labels = torch.unique(cluster_labels)

    print(f"   🎯 簇内相对筛选 (p={p:.1%}):")

    for c in unique_labels:
        if c < 0: continue # 跳过噪声

        mask = cluster_labels == c
        cluster_size = mask.sum().item()

        if cluster_size < 5:
            stats['skipped_clusters'] += 1
            continue

        s = strength[mask]
        tau = torch.quantile(s, p)

        cluster_low_conf = s <= tau
        low_conf_mask[mask] = cluster_low_conf

        stats['total_low_conf'] += cluster_low_conf.sum().item()
        stats['processed_clusters'] += 1
        
    print(f"\n   📊 相对筛选结果:")
    print(f"      处理簇数: {stats['processed_clusters']}")
    print(f"      跳过簇数: {stats['skipped_clusters']}")
    print(f"      高置信度: { (~low_conf_mask).sum() }")
    print(f"      低置信度: { low_conf_mask.sum() }")

    return low_conf_mask, stats


def compute_cluster_centroids(embeddings, labels, high_conf_mask):
    """
    快速计算簇中心 (返回 CPU 字典以节省 GPU 显存)
    """
    print(f"\n   🧮 正在快速计算簇中心 (Total Reads: {len(labels)})...")

    device = embeddings.device
    valid_mask = (labels >= 0) & high_conf_mask
    valid_embeddings = embeddings[valid_mask]
    valid_labels = labels[valid_mask]

    if len(valid_labels) == 0:
        return {}, {}

    max_id = int(valid_labels.max().item())
    
    # 在 GPU 上累加
    sum_embeddings = torch.zeros(max_id + 1, embeddings.shape[1], device=device)
    count_reads = torch.zeros(max_id + 1, device=device)
    
    sum_embeddings.index_add_(0, valid_labels, valid_embeddings)
    ones = torch.ones_like(valid_labels, dtype=torch.float)
    count_reads.index_add_(0, valid_labels, ones)

    # 转回 CPU 处理字典
    centroids = {}
    cluster_sizes = {}
    
    sum_emb_cpu = sum_embeddings.cpu()
    counts_cpu = count_reads.cpu()
    unique_ids_cpu = torch.unique(valid_labels).cpu().numpy()

    for k in unique_ids_cpu:
        count = counts_cpu[k].item()
        if count < 2: continue
        
        # ⚠️ 注意：这里返回的是 CPU Tensor
        centroids[int(k)] = sum_emb_cpu[k] / count
        cluster_sizes[int(k)] = int(count)

    print(f"   📍 簇中心计算完成: 有效簇数 {len(centroids)}")
    return centroids, cluster_sizes


def refine_low_confidence_reads(embeddings, labels, low_conf_mask, centroids, delta):
    """
    ✅ [修复+加速版] 簇修正
    解决了 CPU Centroids 与 GPU Embeddings 的设备冲突
    使用矩阵运算替代循环
    """
    new_labels = labels.clone()
    noise_mask = torch.zeros_like(labels, dtype=torch.bool)
    
    low_conf_indices = torch.where(low_conf_mask)[0]
    num_low_conf = len(low_conf_indices)
    
    if num_low_conf == 0:
        return new_labels, noise_mask, {'reassigned': 0, 'marked_noise': 0, 'kept_unchanged': 0}

    print(f"\n   🔄 正在批量修正 {num_low_conf} 个低置信度reads (Matrix Mode)...")
    
    # 1. 准备簇中心矩阵 (并移动到 GPU!)
    sorted_cluster_ids = sorted(centroids.keys())
    if not sorted_cluster_ids:
        print("   ⚠️ 没有有效的簇中心，跳过修正")
        return new_labels, noise_mask, {}

    # 将 CPU 的 centroids 堆叠后，一次性搬运到 GPU
    cluster_matrix = torch.stack([centroids[k] for k in sorted_cluster_ids]).to(embeddings.device) # (K, D)
    cluster_ids_tensor = torch.tensor(sorted_cluster_ids, device=embeddings.device) # (K,)
    
    # 2. 准备查询向量 (已经在 GPU 上)
    query_embeddings = embeddings[low_conf_indices] # (M, D)
    
    # 3. 分块计算距离矩阵 (防止显存爆炸)
    batch_size = 5000 
    reassigned = 0
    marked_noise = 0
    
    for i in range(0, num_low_conf, batch_size):
        end = min(i + batch_size, num_low_conf)
        batch_queries = query_embeddings[i:end] # (B, D)
        
        # 计算距离 (B, K)
        dists = torch.cdist(batch_queries, cluster_matrix)
        
        # 找最近
        min_dists, min_indices = torch.min(dists, dim=1) # (B,)
        best_cluster_ids = cluster_ids_tensor[min_indices]
        
        # 决策
        valid_mask = min_dists < delta
        
        # 写回
        global_indices = low_conf_indices[i:end]
        
        # 有效的：重新分配
        valid_indices = global_indices[valid_mask]
        valid_assignments = best_cluster_ids[valid_mask]
        
        # 统计变化
        original_labels = labels[valid_indices]
        reassigned += (original_labels != valid_assignments).sum().item()
        
        new_labels[valid_indices] = valid_assignments
        
        # 无效的：标记噪声
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
    ✅ [修复版] 自适应计算 Delta
    """
    device = embeddings.device
    print(f"   🎯 计算自适应 Delta (Percentile={percentile})...")
    
    # 优化逻辑：只采样 100 个簇来计算阈值，避免 100 万次计算卡死
    sample_dists = []
    sampled_keys = random.sample(list(centroids.keys()), min(100, len(centroids)))
    
    for k in sampled_keys:
        ck_gpu = centroids[k].to(device)
        # 随机采 100 个 embedding 算一下距离分布
        indices = torch.randint(0, len(embeddings), (100,), device=device)
        dists = torch.norm(embeddings[indices] - ck_gpu.unsqueeze(0), dim=1)
        sample_dists.append(dists)
        
    all_dists = torch.cat(sample_dists)
    delta = torch.quantile(all_dists, percentile / 100.0).item()

    print(f"   🎯 自适应delta: {delta:.4f}")
    return delta


def merge_similar_clusters(embeddings, labels, centroids, merge_threshold=0.1):
    """
    ✅ [修复版] 强力合并
    修复了 'bitwise_and_cuda' not implemented for 'Float' 报错
    """
    print(f"\n   🧲 开始执行簇合并 (阈值={merge_threshold})...")
    device = embeddings.device
    
    # 1. 准备数据
    sorted_ids = sorted(list(centroids.keys()))
    if len(sorted_ids) < 2: return labels, {}
    
    # 转为 Tensor 矩阵
    center_matrix = torch.stack([centroids[k] for k in sorted_ids]).to(device) # (K, D)
    
    # 2. 计算两两距离 (K, K)
    dists = torch.cdist(center_matrix, center_matrix)
    
    # 排除自身 (设为无穷大)
    eye_mask = torch.eye(len(sorted_ids), device=device).bool()
    dists.masked_fill_(eye_mask, float('inf'))
    
    # 3. 贪婪合并策略
    merge_map = {} # old_id -> new_id
    
    # 🔴 关键修复：强制将上三角掩码转为 bool 类型
    # 原代码: torch.triu(torch.ones_like(dists), diagonal=1) -> Float
    # 新代码: torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1) -> Bool
    
    upper_tri_mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
    
    # 获取满足条件的索引
    pairs = torch.nonzero((dists < merge_threshold) & upper_tri_mask)
    
    # 按距离从小到大排序，优先合并最近的
    if len(pairs) > 0:
        pair_dists = dists[pairs[:, 0], pairs[:, 1]]
        sorted_idx = torch.argsort(pair_dists)
        pairs = pairs[sorted_idx]
    
    merge_count = 0
    
    for idx in range(len(pairs)):
        i, j = pairs[idx].tolist()
        id_a, id_b = sorted_ids[i], sorted_ids[j]
        
        # 检查是否已经被合并过
        root_a = id_a
        while root_a in merge_map: root_a = merge_map[root_a]
        
        root_b = id_b
        while root_b in merge_map: root_b = merge_map[root_b]
        
        if root_a != root_b:
            # 总是把大的 ID 合并到小的 ID (保持稳定)
            target = min(root_a, root_b)
            source = max(root_a, root_b)
            merge_map[source] = target
            merge_count += 1
            
    print(f"      发现 {merge_count} 对相似簇需要合并")
    
    if merge_count == 0:
        return labels, {}

    # 4. 执行合并 (更新 Labels)
    new_labels = labels.clone()
    
    # 批量更新
    for src, dst in merge_map.items():
        mask = (labels == src)
        new_labels[mask] = dst
        
    print(f"   ✅ 合并完成！簇数量: {len(sorted_ids)} -> {len(sorted_ids) - merge_count}")
    return new_labels, merge_map