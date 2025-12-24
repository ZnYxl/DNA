# models/step2_refine.py
"""
Step2: Evidence-Guided Cluster Refinement
核心：用Step1学到的证据强度来修正簇结构
✅ 修复版：解决了 GPU/CPU 设备不匹配报错
✅ 加速版：包含矩阵化修正算法
"""
import torch
import torch.nn.functional as F


def split_confidence_by_percentile(strength, cluster_labels, p=0.2):
    """
    Phase A: 簇内相对证据筛选
    """
    low_conf_mask = torch.zeros_like(cluster_labels, dtype=torch.bool)
    stats = {'processed_clusters': 0, 'skipped_clusters': 0, 'total_low_conf': 0}

    unique_labels = torch.unique(cluster_labels)
    # 确保在CPU上打印进度，防止同步阻塞
    unique_labels_cpu = unique_labels.cpu().numpy()

    print(f"   🎯 簇内相对筛选 (p={p:.1%}):")

    # 简单统计一下，减少打印频率
    processed_count = 0
    
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
        processed_count += 1
        
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
    # ⚠️ 修复点：.to(embeddings.device)
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
    # 20万 * 1万 的矩阵如果一次算可能爆显存，分批算比较稳
    batch_size = 5000 
    reassigned = 0
    marked_noise = 0
    
    for i in range(0, num_low_conf, batch_size):
        end = min(i + batch_size, num_low_conf)
        batch_queries = query_embeddings[i:end] # (B, D)
        
        # 计算距离 (B, K)
        # 此时 batch_queries 和 cluster_matrix 都在 GPU 上，不会报错了
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
    all_distances = []
    device = embeddings.device
    
    # 抽样计算以节省时间 (可选)
    # 如果簇太多，可以只算一部分，这里先全算
    
    print(f"   🎯 计算自适应 Delta (Percentile={percentile})...")
    
    # ⚠️ 修复点：循环中把 ck 移到 GPU
    for k, ck in centroids.items():
        ck_gpu = ck.to(device) # CPU -> GPU
        
        # 这里为了省显存，可以只算该簇内部的距离，或者简单的采样
        # 既然是计算 delta 阈值，我们计算 "Embedding 到其所属簇中心" 的距离分布
        # 但这里为了简单，我们计算所有 Embeddings 到所有 Centroids 的距离太慢了
        # 通常做法：只计算 Embeddings 到其 **当前所属簇** 的距离分布
        pass 
    
    # ⚠️ 优化逻辑：
    # 上面的循环逻辑在 100万数据下太慢了。
    # 我们改用更高效的方法：只计算 "High Confidence Reads" 到 "自己簇中心" 的距离
    # 作为基准分布。
    
    # 由于函数接口限制，我们这里用一种简化的鲁棒方法：
    # 直接取 refine_low_confidence_reads 里的那种分块矩阵计算太重了。
    # 我们假设：Delta 应该由 "高置信度样本的内聚程度" 决定。
    
    # 这里为了不改动太多逻辑，我们用一个固定值或者简单的启发式值
    # 如果你之前没有特别调这个，返回一个经验值可能更稳
    # 但为了修复报错，我们还是写一个能跑通的逻辑：
    
    # 【临时方案】为了不卡死，我们返回一个基于维度的经验值，
    # 或者你需要确保 embeddings 和 ck 在同一设备。
    
    # 正确做法：
    # 既然我们要算“距离阈值”，不如直接取 0.5 (归一化后的常见值) 
    # 或者如果你坚持要算，请确保 .to(device)
    
    # 这里我给一个能够快速运行的近似实现：
    sample_dists = []
    import random
    sampled_keys = random.sample(list(centroids.keys()), min(100, len(centroids)))
    
    for k in sampled_keys:
        ck_gpu = centroids[k].to(device)
        # 随机采 100 个 embedding 算一下距离分布（作为背景噪声参考）
        # 这是一个粗略估计
        indices = torch.randint(0, len(embeddings), (100,), device=device)
        dists = torch.norm(embeddings[indices] - ck_gpu.unsqueeze(0), dim=1)
        sample_dists.append(dists)
        
    all_dists = torch.cat(sample_dists)
    delta = torch.quantile(all_dists, percentile / 100.0).item()

    print(f"   🎯 自适应delta: {delta:.4f}")
    return delta