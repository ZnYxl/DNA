# models/step2_refine.py
"""
Step2: Evidence-Guided Cluster Refinement (防 OOM + 高速版)

基于学生版本的 index_add_ 向量化思路，修复以下问题:
  [FIX-1] refine_reads 返回 3 个值 (new_labels, noise_mask, stats)
  [FIX-2] refine_reads 接受 round_idx 参数，内部做 delta scaling
  [FIX-3] compute_global_delta 签名与 runner 一致: (embeddings, labels, zone_ids, centroids)
  [FIX-4] refine 距离计算: CPU 双轴分块 matmul，避免 cdist(5K, 444K) OOM
  [FIX-5] compute_global_delta 只返回原始 delta，scaling 留给 refine_reads
"""
import torch
import torch.nn.functional as F
import numpy as np
import time

# ---------------------------------------------------------------------------
# 超参常量
# ---------------------------------------------------------------------------
DIRTY_PERCENTILE   = 0.10
SAFE_PERCENTILE    = 0.70
MIN_ZONE1_SAFETY   = 3
ZONE2_WEIGHT_CAP   = 0.30
DELTA_P            = 95
ROUND1_DELTA_SCALE = 1.5


# ===========================================================================
# 1. 三区制划分
# ===========================================================================
def split_confidence_by_zone(u_epi, u_ale, labels):
    N      = len(labels)
    device = labels.device
    zone_ids = torch.zeros(N, dtype=torch.long, device=device)

    valid   = (labels >= 0)
    n_valid = valid.sum().item()
    if n_valid == 0:
        return zone_ids, {'zone1': 0, 'zone2': 0, 'zone3': 0, 'noise': N}

    # 第一刀: U_ale Top 10% → Zone III
    ale_threshold = torch.quantile(u_ale[valid], 1.0 - DIRTY_PERCENTILE)
    is_dirty = valid & (u_ale >= ale_threshold)
    zone_ids[is_dirty] = 3

    # 第二刀: 剩余中 U_epi Bottom 70% → Zone I
    remaining = valid & (~is_dirty)
    if remaining.any():
        epi_threshold = torch.quantile(u_epi[remaining], SAFE_PERCENTILE)
        zone_ids[remaining & (u_epi <= epi_threshold)] = 1
        zone_ids[remaining & (u_epi >  epi_threshold)] = 2

    z1 = int((zone_ids == 1).sum().item())
    z2 = int((zone_ids == 2).sum().item())
    z3 = int((zone_ids == 3).sum().item())
    zn = int((zone_ids == 0).sum().item())

    print(f"   📊 三区制划分:")
    print(f"      Zone I  (Safe):   {z1:>8d}  ({z1/max(n_valid,1)*100:5.1f}%)")
    print(f"      Zone II (Hard):   {z2:>8d}  ({z2/max(n_valid,1)*100:5.1f}%)")
    print(f"      Zone III(Dirty):  {z3:>8d}  ({z3/max(n_valid,1)*100:5.1f}%)")
    print(f"      Noise   (skip):   {zn:>8d}")

    return zone_ids, {'zone1': z1, 'zone2': z2, 'zone3': z3, 'noise': zn}


# ===========================================================================
# 2. 证据加权质心 (CPU index_add_ 向量化)
#
#    签名: compute_centroids_weighted(embeddings, labels, strength, zone_ids)
#    返回: (centroids_dict, cluster_sizes_dict)
# ===========================================================================
def compute_centroids_weighted(embeddings, labels, strength, zone_ids):
    t0 = time.time()

    # ---- 全部搬到 CPU，解决 GPU OOM ----
    # ★ embeddings 已在 runner 中 in-place 归一化, 不再拷贝
    emb_cpu  = embeddings.detach() if embeddings.device.type == 'cpu' else embeddings.detach().cpu()
    lbl_cpu  = labels.detach() if labels.device.type == 'cpu' else labels.detach().cpu()
    str_cpu  = strength.detach() if strength.device.type == 'cpu' else strength.detach().cpu()
    zone_cpu = zone_ids.detach() if zone_ids.device.type == 'cpu' else zone_ids.detach().cpu()

    # 筛选 Zone I + Zone II
    valid = (lbl_cpu >= 0) & ((zone_cpu == 1) | (zone_cpu == 2))
    sub_emb = emb_cpu[valid]
    sub_lbl = lbl_cpu[valid]
    sub_str = str_cpu[valid]
    sub_zon = zone_cpu[valid]

    if len(sub_lbl) == 0:
        return {}, {}

    max_label = int(sub_lbl.max().item()) + 1
    D = sub_emb.shape[1]

    # ---- 安全阀: 向量化统计 Zone I 数量 ----
    z1_mask = (sub_zon == 1)
    z1_counts = torch.zeros(max_label, dtype=torch.long)
    if z1_mask.any():
        z1_counts.index_add_(0, sub_lbl[z1_mask],
                             torch.ones(int(z1_mask.sum()), dtype=torch.long))

    unsafe = (z1_counts < MIN_ZONE1_SAFETY)
    is_unsafe_read = unsafe[sub_lbl]

    weights = sub_str.clone()
    clamp_mask = is_unsafe_read & (sub_zon == 2)
    if clamp_mask.any():
        weights[clamp_mask] = weights[clamp_mask].clamp(max=ZONE2_WEIGHT_CAP)

    # ---- index_add_ 加权求和 (核心加速) ----
    centroid_sum = torch.zeros(max_label, D)
    weight_sum   = torch.zeros(max_label)
    centroid_sum.index_add_(0, sub_lbl, sub_emb * weights.unsqueeze(1))
    weight_sum.index_add_(0, sub_lbl, weights)

    valid_mask = (weight_sum > 1e-6)
    final = torch.zeros(max_label, D)
    final[valid_mask] = centroid_sum[valid_mask] / weight_sum[valid_mask].unsqueeze(1)
    final = F.normalize(final, dim=-1)

    # 转 dict
    present = torch.nonzero(valid_mask).squeeze(1)
    centroids = {int(k): final[k] for k in present}

    # cluster_sizes (保持接口兼容)
    size_count = torch.zeros(max_label, dtype=torch.long)
    size_count.index_add_(0, sub_lbl, torch.ones(len(sub_lbl), dtype=torch.long))
    cluster_sizes = {int(k): int(size_count[k]) for k in present}

    n_safety = int(unsafe[present].sum().item()) if len(present) > 0 else 0
    t1 = time.time()
    print(f"\n   📍 质心 (CPU vectorized): {len(centroids)} 簇, "
          f"安全阀 {n_safety}, 耗时 {t1-t0:.1f}s")

    return centroids, cluster_sizes


# ===========================================================================
# 3. 全局自适应 Delta (CPU 采样 10 万估算)
#
#    签名: compute_global_delta(embeddings, labels, zone_ids, centroids)
#                               ↑ 注意顺序! 与 runner 调用一致
#    返回: delta (原始 P95 值, 不乘 ROUND1_DELTA_SCALE)
# ===========================================================================
def compute_global_delta(embeddings, labels, zone_ids, centroids):
    print(f"   🎯 计算 Global Delta (P{DELTA_P})...")

    # ★ embeddings 已归一化, 不再拷贝
    emb_cpu  = embeddings.detach() if embeddings.device.type == 'cpu' else embeddings.detach().cpu()
    lbl_cpu  = labels.detach() if labels.device.type == 'cpu' else labels.detach().cpu()
    zone_cpu = zone_ids.detach() if zone_ids.device.type == 'cpu' else zone_ids.detach().cpu()

    mask = (lbl_cpu >= 0) & (zone_cpu == 1)
    z1_indices = torch.nonzero(mask).squeeze(1)

    if len(z1_indices) == 0:
        print(f"   ⚠️ 无 Zone I 样本, 返回 0.5")
        return 0.5

    # 采样 10 万 (足够准确, 避免慢)
    if len(z1_indices) > 100000:
        z1_indices = z1_indices[torch.randperm(len(z1_indices))[:100000]]

    sample_emb = emb_cpu[z1_indices]
    sample_lbl = lbl_cpu[z1_indices]

    # 匹配质心
    target_list = []
    keep = []
    for i in range(len(sample_lbl)):
        lid = int(sample_lbl[i].item())
        if lid in centroids:
            target_list.append(centroids[lid])
            keep.append(i)

    if not target_list:
        print(f"   ⚠️ 无法匹配质心, 返回 0.5")
        return 0.5

    target_mat = torch.stack(target_list)
    sample_emb = sample_emb[keep]

    # L2 distance (normalized vectors): ||a-b|| = sqrt(2 - 2*<a,b>)
    sim = (sample_emb * target_mat).sum(dim=1)
    dists = torch.sqrt((2.0 - 2.0 * sim).clamp(min=0.0))

    delta = float(np.percentile(dists.numpy(), DELTA_P))
    print(f"   🎯 Global Delta = {delta:.4f} (基于 {len(keep)} 个 Zone I 样本)")
    return delta


# ===========================================================================
# 辅助: 双轴分块最近邻 (CPU, 防 OOM)
#
# 为什么不能用 cdist?
#   query=5000, centroids=444K → output (5000, 444000) × 4B = 8.8GB → 必爆
#
# 解法: 对 centroid 轴也分块
#   query_chunk × centroid_chunk × 4B = 3000 × 80000 × 4 = 960MB → CPU 安全
# ===========================================================================
def _chunked_nearest_centroid(query, centroid_matrix, centroid_ids,
                              query_chunk=3000, centroid_chunk=80000):
    """
    在 normalized 空间用分块 matmul 找最近质心。
    返回: (min_dist, best_centroid_id) 都是 (N,) tensor
    """
    N = query.shape[0]
    K = centroid_matrix.shape[0]

    best_dist = torch.full((N,), float('inf'))
    best_idx  = torch.zeros(N, dtype=torch.long)

    for qi in range(0, N, query_chunk):
        qe = min(qi + query_chunk, N)
        q_batch = query[qi:qe]

        batch_best_dist = torch.full((qe - qi,), float('inf'))
        batch_best_idx  = torch.zeros(qe - qi, dtype=torch.long)

        for ci in range(0, K, centroid_chunk):
            ce = min(ci + centroid_chunk, K)
            c_batch = centroid_matrix[ci:ce]

            # cosine sim → L2 for normalized vectors
            sim = q_batch @ c_batch.T                              # (q, c)
            dist = torch.sqrt((2.0 - 2.0 * sim).clamp(min=0.0))   # (q, c)

            chunk_min_d, chunk_min_i = dist.min(dim=1)

            improved = (chunk_min_d < batch_best_dist)
            batch_best_dist[improved] = chunk_min_d[improved]
            batch_best_idx[improved]  = chunk_min_i[improved] + ci  # 偏移量!

        best_dist[qi:qe] = batch_best_dist
        best_idx[qi:qe]  = batch_best_idx

        if qi > 0 and (qi // query_chunk) % 20 == 0:
            print(f"      refine 进度: {qe}/{N}", flush=True)

    best_cid = centroid_ids[best_idx]
    return best_dist, best_cid


# ===========================================================================
# 4. Zone-aware 修正
#
#    签名: refine_reads(embeddings, labels, zone_ids, centroids, delta, round_idx=1)
#    返回: (new_labels, noise_mask, stats)  ← 3 个值!
# ===========================================================================
def refine_reads(embeddings, labels, zone_ids, centroids, delta, round_idx=1):
    device = embeddings.device
    new_labels = labels.clone()
    noise_mask = torch.zeros_like(labels, dtype=torch.bool)

    # ---- ROUND1_DELTA_SCALE 在这里应用 (不在 compute_global_delta 里) ----
    eff_delta = delta * ROUND1_DELTA_SCALE if round_idx == 1 else delta
    print(f"\n   🔄 Zone-aware 修正 (Round {round_idx}, "
          f"delta={delta:.4f}, scale={'1.5' if round_idx==1 else '1.0'}, "
          f"eff_delta={eff_delta:.4f})")

    # Zone III → 噪声
    dirty = (zone_ids == 3)
    new_labels[dirty] = -1
    noise_mask[dirty] = True
    n_dirty = int(dirty.sum().item())

    # Zone I → 不动
    n_safe = int((zone_ids == 1).sum().item())

    # Zone II → 距离判决 (CPU 双轴分块)
    hard_indices = torch.nonzero(zone_ids == 2).squeeze(1)
    n_hard = len(hard_indices)
    reassigned = 0
    marked_noise = 0

    if n_hard > 0 and len(centroids) > 0:
        t0 = time.time()
        print(f"   ⚖️ 修正 {n_hard} 条 Zone II (CPU 双轴分块, "
              f"{len(centroids)} 质心)...")

        # ---- 全部在 CPU, 已归一化 ----
        emb_cpu = embeddings.detach() if embeddings.device.type == 'cpu' else embeddings.detach().cpu()
        query = emb_cpu[hard_indices.cpu() if hard_indices.is_cuda else hard_indices]

        sorted_ids = sorted(centroids.keys())
        centroid_matrix = torch.stack([centroids[k] for k in sorted_ids])
        centroid_ids = torch.tensor(sorted_ids, dtype=torch.long)

        # 双轴分块最近邻
        min_dist, best_cid = _chunked_nearest_centroid(
            query, centroid_matrix, centroid_ids
        )

        # 判决
        within = (min_dist < eff_delta)

        # 归队
        gi_in = hard_indices[within]
        bi_in = best_cid[within].to(device)
        orig  = labels[gi_in]
        reassigned = int((orig != bi_in).sum().item())
        new_labels[gi_in] = bi_in

        # 噪声
        gi_out = hard_indices[~within]
        new_labels[gi_out] = -1
        noise_mask[gi_out] = True
        marked_noise = int((~within).sum().item())

        t1 = time.time()
        print(f"      完成, 耗时 {t1-t0:.1f}s")

    stats = {
        'zone1_kept':       n_safe,
        'zone2_total':      n_hard,
        'zone2_reassigned': reassigned,
        'zone2_noise':      marked_noise,
        'zone3_dirty':      n_dirty,
    }

    print(f"   ✅ 修正结果:")
    print(f"      Zone I  保持:    {stats['zone1_kept']}")
    print(f"      Zone II 重分配:  {stats['zone2_reassigned']}")
    print(f"      Zone II 噪声:    {stats['zone2_noise']}")
    print(f"      Zone III 丢弃:   {stats['zone3_dirty']}")

    return new_labels, noise_mask, stats