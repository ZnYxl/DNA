# models/step2_refine.py
"""
Step2: Evidence-Guided Cluster Refinement  —  三区制版本

修改清单 C 的全部实现：
  1. split_confidence_by_zone        — 解耦双重筛选（先切 U_ale，再切 U_epi）
  2. compute_centroids_weighted      — Zone I+II 证据加权质心 + 小簇安全阀
  3. compute_global_delta            — 用 Safe 样本到自己质心的距离分布取 P95
  4. refine_reads                    — Zone-aware 修正 + Round-aware delta 调度
"""
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 超参常量
# ---------------------------------------------------------------------------
DIRTY_PERCENTILE   = 0.10   # U_ale 全局 Top 10% → Zone III
SAFE_PERCENTILE    = 0.70   # 剩余中 U_epi Bottom 70% → Zone I
MIN_ZONE1_SAFETY   = 3      # 簇内 Zone I 不足此数时激活安全阀
ZONE2_WEIGHT_CAP   = 0.30   # 安全阀激活后，Zone II 每个样本的权重上限
DELTA_P            = 95     # Global Delta 取 P95
ROUND1_DELTA_SCALE = 1.5    # Round 1 宽松倍数


# ===========================================================================
# 1. 三区制划分
#    第一刀：全局 U_ale Top 10% → Zone III (Dirty)
#    第二刀：剩余中 U_epi Bottom 70% → Zone I (Safe)，其余 → Zone II (Hard)
#    噪声标签 (label < 0) 的位置 zone = 0，不参与任何后续流程
# ===========================================================================
def split_confidence_by_zone(u_epi, u_ale, labels):
    """
    u_epi:  (N,)  认知不确定性
    u_ale:  (N,)  偶然不确定性
    labels: (N,)  当前簇标签

    返回:
        zone_ids:   (N,) LongTensor  1=Safe 2=Hard 3=Dirty 0=噪声
        zone_stats: dict
    """
    N       = len(labels)
    device  = labels.device
    zone_ids = torch.zeros(N, dtype=torch.long, device=device)

    valid   = (labels >= 0)
    n_valid = valid.sum().item()
    if n_valid == 0:
        return zone_ids, {'zone1': 0, 'zone2': 0, 'zone3': 0, 'noise': N}

    # ---- 第一刀: U_ale 切 Dirty ----
    u_ale_valid   = u_ale[valid]
    ale_threshold = torch.quantile(u_ale_valid, 1.0 - DIRTY_PERCENTILE)
    dirty_global  = torch.zeros(N, dtype=torch.bool, device=device)
    dirty_global[valid] = (u_ale_valid >= ale_threshold)
    zone_ids[dirty_global] = 3

    # ---- 第二刀: 在"有效且非Dirty"中切 Safe / Hard ----
    remaining = valid & (~dirty_global)
    n_rem     = remaining.sum().item()

    if n_rem > 0:
        u_epi_rem     = u_epi[remaining]
        epi_threshold = torch.quantile(u_epi_rem, SAFE_PERCENTILE)

        safe_local  = (u_epi_rem <= epi_threshold)
        hard_local  = ~safe_local

        rem_indices = torch.where(remaining)[0]
        zone_ids[rem_indices[safe_local]] = 1
        zone_ids[rem_indices[hard_local]] = 2

    # ---- 统计 ----
    z1 = int((zone_ids == 1).sum().item())
    z2 = int((zone_ids == 2).sum().item())
    z3 = int((zone_ids == 3).sum().item())
    zn = int((zone_ids == 0).sum().item())

    print(f"   📊 三区制划分结果:")
    print(f"      Zone I  (Safe):   {z1:>7d}  ({z1/max(n_valid,1)*100:5.1f}%)")
    print(f"      Zone II (Hard):   {z2:>7d}  ({z2/max(n_valid,1)*100:5.1f}%)")
    print(f"      Zone III(Dirty):  {z3:>7d}  ({z3/max(n_valid,1)*100:5.1f}%)")
    print(f"      Noise   (skip):   {zn:>7d}")

    return zone_ids, {'zone1': z1, 'zone2': z2, 'zone3': z3, 'noise': zn}


# ===========================================================================
# 2. 证据加权质心（Zone I + Zone II，含安全阀）
#
#    C_k = Σ(S_i · z_i) / Σ(S_i)    S_i = strength of read i
#
#    安全阀触发条件: 簇内 Zone I 数量 < MIN_ZONE1_SAFETY
#    触发后: Zone II 样本的 S 被夹紧到 ZONE2_WEIGHT_CAP
#    效果: 归一化后 Zone II 单个样本权重极低，不影响质心偏移
#          但它们的存在增加了样本数，降低方差
# ===========================================================================
def compute_centroids_weighted(embeddings, labels, strength, zone_ids):
    """
    embeddings: (N, D)
    labels:     (N,)
    strength:   (N,)   序列级别 evidence strength
    zone_ids:   (N,)   三区制标签

    返回:
        centroids:     dict { cluster_id(int) -> CPU Tensor (D,) }
        cluster_sizes: dict { cluster_id(int) -> int }
    """
    print(f"\n   🧮 计算证据加权质心 (Zone I+II，含安全阀)...")
    device = embeddings.device

    # 只让 Zone I 和 Zone II 参与
    participate = ((zone_ids == 1) | (zone_ids == 2)) & (labels >= 0)
    p_emb    = embeddings[participate]
    p_labels = labels[participate]
    p_str    = strength[participate]
    p_zones  = zone_ids[participate]

    if p_emb.shape[0] == 0:
        return {}, {}

    centroids     = {}
    cluster_sizes = {}
    safety_count  = 0

    for k in torch.unique(p_labels):
        k_int = int(k.item())
        mask  = (p_labels == k)
        n_k   = int(mask.sum().item())
        if n_k < 2:
            continue

        z1_count = int((p_zones[mask] == 1).sum().item())
        s_k      = p_str[mask].clone()

        # ---- 安全阀 ----
        if z1_count < MIN_ZONE1_SAFETY:
            z2_local = (p_zones[mask] == 2)
            if z2_local.any():
                s_k[z2_local] = s_k[z2_local].clamp(max=ZONE2_WEIGHT_CAP)
            safety_count += 1

        # 加权平均
        w        = s_k / s_k.sum()
        centroid = (w.unsqueeze(1) * p_emb[mask]).sum(dim=0)

        centroids[k_int]     = centroid.cpu()
        cluster_sizes[k_int] = n_k

    print(f"   📍 质心计算完成: 有效簇数 {len(centroids)}, 安全阀触发 {safety_count} 次")
    return centroids, cluster_sizes


# ===========================================================================
# 3. 全局自适应 Delta
#
#    统计所有 Zone I (Safe) 样本到自己所属簇质心的距离分布，取 P95。
#    含义：Safe 样本中 95% 都在此半径内，超出的就是离群点。
# ===========================================================================
def compute_global_delta(embeddings, labels, zone_ids, centroids):
    """
    返回: delta (float)
    """
    print(f"   🎯 计算 Global Delta (Safe→自己质心的距离分布, P{DELTA_P})...")
    device = embeddings.device

    safe_mask   = (zone_ids == 1) & (labels >= 0)
    safe_emb    = embeddings[safe_mask]
    safe_labels = labels[safe_mask]

    if safe_emb.shape[0] == 0:
        print(f"   ⚠️ 无 Safe 样本，返回经验值 0.5")
        return 0.5

    sorted_ids  = sorted(centroids.keys())
    if len(sorted_ids) == 0:
        return 0.5

    id_to_row       = {kid: i for i, kid in enumerate(sorted_ids)}
    centroid_matrix = torch.stack(
        [centroids[k] for k in sorted_ids]
    ).to(device)                                          # (K, D)

    # label → centroid_matrix 行索引
    safe_labels_list   = safe_labels.cpu().tolist()
    valid_sample_idx   = []
    valid_centroid_idx = []

    for i, lbl in enumerate(safe_labels_list):
        if lbl in id_to_row:
            valid_sample_idx.append(i)
            valid_centroid_idx.append(id_to_row[lbl])

    if len(valid_sample_idx) == 0:
        print(f"   ⚠️ Safe 样本无法匹配到质心，返回经验值 0.5")
        return 0.5

    valid_sample_idx   = torch.tensor(valid_sample_idx,  device=device)
    valid_centroid_idx = torch.tensor(valid_centroid_idx, device=device)

    # 分块距离
    chunk  = 10000
    dists  = []
    for i in range(0, len(valid_sample_idx), chunk):
        end = min(i + chunk, len(valid_sample_idx))
        emb = safe_emb[valid_sample_idx[i:end]]
        cen = centroid_matrix[valid_centroid_idx[i:end]]
        dists.append(torch.norm(emb - cen, dim=1))

    all_dists = torch.cat(dists)
    delta     = torch.quantile(all_dists, DELTA_P / 100.0).item()

    print(f"   🎯 Global Delta = {delta:.4f}  (基于 {len(all_dists)} 个 Safe 样本)")
    return delta


# ===========================================================================
# 4. Zone-aware 修正
#
#    Zone I:   完全信任，不动
#    Zone II:  距离判决 → < eff_delta 归队，≥ 噪声
#    Zone III: 直接置 -1
#    Round 1 用 delta*1.5（宽松），Round 2+ 用 delta（严格）
# ===========================================================================
def refine_reads(embeddings, labels, zone_ids, centroids, delta, round_idx=1):
    """
    返回:
        new_labels: (N,)
        noise_mask: (N,) bool
        stats:      dict
    """
    new_labels = labels.clone()
    noise_mask = torch.zeros_like(labels, dtype=torch.bool)
    device     = embeddings.device

    eff_delta = delta * ROUND1_DELTA_SCALE if round_idx == 1 else delta
    print(f"\n   🔄 Zone-aware 修正  (Round {round_idx}, eff_delta={eff_delta:.4f})")

    # ---- Zone III → 直接噪声 ----
    dirty      = (zone_ids == 3)
    new_labels[dirty] = -1
    noise_mask[dirty] = True
    n_dirty    = int(dirty.sum().item())

    # ---- Zone I → 不动 ----
    n_safe     = int((zone_ids == 1).sum().item())

    # ---- Zone II → 距离判决 ----
    hard_mask    = (zone_ids == 2)
    hard_indices = torch.where(hard_mask)[0]
    n_hard       = len(hard_indices)

    reassigned   = 0
    marked_noise = 0

    if n_hard > 0 and len(centroids) > 0:
        sorted_ids     = sorted(centroids.keys())
        cluster_matrix = torch.stack(
            [centroids[k] for k in sorted_ids]
        ).to(device)
        cluster_ids_t  = torch.tensor(sorted_ids, device=device)

        query = embeddings[hard_indices]

        chunk = 5000
        for i in range(0, n_hard, chunk):
            end   = min(i + chunk, n_hard)
            batch = query[i:end]

            dists          = torch.cdist(batch, cluster_matrix)
            min_d, min_idx = torch.min(dists, dim=1)
            best_ids       = cluster_ids_t[min_idx]

            within     = (min_d < eff_delta)
            global_idx = hard_indices[i:end]

            # 归队
            gi_in  = global_idx[within]
            bi_in  = best_ids[within]
            orig   = labels[gi_in]
            reassigned += int((orig != bi_in).sum().item())
            new_labels[gi_in] = bi_in

            # 噪声
            gi_out = global_idx[~within]
            new_labels[gi_out] = -1
            noise_mask[gi_out] = True
            marked_noise += int((~within).sum().item())

    stats = {
        'zone1_kept':       n_safe,
        'zone2_total':      n_hard,
        'zone2_reassigned': reassigned,
        'zone2_noise':      marked_noise,
        'zone3_dirty':      n_dirty,
    }

    print(f"   ✅ 修正完成:")
    print(f"      Zone I  保持不动:  {stats['zone1_kept']}")
    print(f"      Zone II 重新分配:  {stats['zone2_reassigned']}")
    print(f"      Zone II 标记噪声:  {stats['zone2_noise']}")
    print(f"      Zone III 直接丢弃: {stats['zone3_dirty']}")

    return new_labels, noise_mask, stats
