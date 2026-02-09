# models/step2_refine.py
"""
Step2: Evidence-Guided Cluster Refinement — 三区制版本

修复清单:
  [FIX-#3]  Zone II 距离判决改用 cosine distance (与对比学习一致)
            embeddings 先 L2 normalize，再用 L2 距离 ≡ cosine distance 的单调变换
"""
import torch
import torch.nn.functional as F


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
    N       = len(labels)
    device  = labels.device
    zone_ids = torch.zeros(N, dtype=torch.long, device=device)

    valid   = (labels >= 0)
    n_valid = valid.sum().item()
    if n_valid == 0:
        return zone_ids, {'zone1': 0, 'zone2': 0, 'zone3': 0, 'noise': N}

    # 第一刀: U_ale Top 10% → Zone III
    u_ale_valid   = u_ale[valid]
    ale_threshold = torch.quantile(u_ale_valid, 1.0 - DIRTY_PERCENTILE)
    dirty_global  = torch.zeros(N, dtype=torch.bool, device=device)
    dirty_global[valid] = (u_ale_valid >= ale_threshold)
    zone_ids[dirty_global] = 3

    # 第二刀: 剩余中 U_epi Bottom 70% → Zone I
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
# 2. 证据加权质心 + 安全阀
#    [FIX-#3] 质心在 normalized 空间计算
# ===========================================================================
def compute_centroids_weighted(embeddings, labels, strength, zone_ids):
    print(f"\n   🧮 计算证据加权质心 (Zone I+II, cosine 空间)...")
    device = embeddings.device

    # [FIX-#3] normalize embeddings
    embeddings_norm = F.normalize(embeddings, dim=-1)

    participate = ((zone_ids == 1) | (zone_ids == 2)) & (labels >= 0)
    p_emb    = embeddings_norm[participate]
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

        if z1_count < MIN_ZONE1_SAFETY:
            z2_local = (p_zones[mask] == 2)
            if z2_local.any():
                s_k[z2_local] = s_k[z2_local].clamp(max=ZONE2_WEIGHT_CAP)
            safety_count += 1

        w        = s_k / s_k.sum()
        centroid = (w.unsqueeze(1) * p_emb[mask]).sum(dim=0)
        # Re-normalize centroid to stay on unit sphere
        centroid = F.normalize(centroid, dim=0)

        centroids[k_int]     = centroid.cpu()
        cluster_sizes[k_int] = n_k

    print(f"   📍 质心计算完成: {len(centroids)} 簇, 安全阀触发 {safety_count} 次")
    return centroids, cluster_sizes


# ===========================================================================
# 3. 全局自适应 Delta (cosine 空间)
# ===========================================================================
def compute_global_delta(embeddings, labels, zone_ids, centroids):
    print(f"   🎯 计算 Global Delta (cosine 空间, P{DELTA_P})...")
    device = embeddings.device

    # [FIX-#3] normalize
    embeddings_norm = F.normalize(embeddings, dim=-1)

    safe_mask   = (zone_ids == 1) & (labels >= 0)
    safe_emb    = embeddings_norm[safe_mask]
    safe_labels = labels[safe_mask]

    if safe_emb.shape[0] == 0:
        print(f"   ⚠️ 无 Safe 样本，返回经验值 0.5")
        return 0.5

    sorted_ids  = sorted(centroids.keys())
    if len(sorted_ids) == 0:
        return 0.5

    id_to_row       = {kid: i for i, kid in enumerate(sorted_ids)}
    centroid_matrix = torch.stack([centroids[k] for k in sorted_ids]).to(device)

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

    chunk  = 10000
    dists  = []
    for i in range(0, len(valid_sample_idx), chunk):
        end = min(i + chunk, len(valid_sample_idx))
        emb = safe_emb[valid_sample_idx[i:end]]
        cen = centroid_matrix[valid_centroid_idx[i:end]]
        dists.append(torch.norm(emb - cen, dim=1))

    all_dists = torch.cat(dists)
    delta     = torch.quantile(all_dists, DELTA_P / 100.0).item()

    print(f"   🎯 Global Delta = {delta:.4f}  ({len(all_dists)} Safe 样本)")
    return delta


# ===========================================================================
# 4. Zone-aware 修正 (cosine 空间)
# ===========================================================================
def refine_reads(embeddings, labels, zone_ids, centroids, delta, round_idx=1):
    new_labels = labels.clone()
    noise_mask = torch.zeros_like(labels, dtype=torch.bool)
    device     = embeddings.device

    # [FIX-#3] normalize
    embeddings_norm = F.normalize(embeddings, dim=-1)

    eff_delta = delta * ROUND1_DELTA_SCALE if round_idx == 1 else delta
    print(f"\n   🔄 Zone-aware 修正 (Round {round_idx}, eff_delta={eff_delta:.4f}, cosine 空间)")

    # Zone III → 噪声
    dirty      = (zone_ids == 3)
    new_labels[dirty] = -1
    noise_mask[dirty] = True
    n_dirty    = int(dirty.sum().item())

    # Zone I → 不动
    n_safe     = int((zone_ids == 1).sum().item())

    # Zone II → 距离判决
    hard_mask    = (zone_ids == 2)
    hard_indices = torch.where(hard_mask)[0]
    n_hard       = len(hard_indices)

    reassigned   = 0
    marked_noise = 0

    if n_hard > 0 and len(centroids) > 0:
        sorted_ids     = sorted(centroids.keys())
        cluster_matrix = torch.stack([centroids[k] for k in sorted_ids]).to(device)
        cluster_ids_t  = torch.tensor(sorted_ids, device=device)

        query = embeddings_norm[hard_indices]

        chunk = 5000
        for i in range(0, n_hard, chunk):
            end   = min(i + chunk, n_hard)
            batch = query[i:end]

            dists          = torch.cdist(batch, cluster_matrix)
            min_d, min_idx = torch.min(dists, dim=1)
            best_ids       = cluster_ids_t[min_idx]

            within     = (min_d < eff_delta)
            global_idx = hard_indices[i:end]

            gi_in  = global_idx[within]
            bi_in  = best_ids[within]
            orig   = labels[gi_in]
            reassigned += int((orig != bi_in).sum().item())
            new_labels[gi_in] = bi_in

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
    print(f"      Zone I  保持:    {stats['zone1_kept']}")
    print(f"      Zone II 重分配:  {stats['zone2_reassigned']}")
    print(f"      Zone II 噪声:    {stats['zone2_noise']}")
    print(f"      Zone III 丢弃:   {stats['zone3_dirty']}")

    return new_labels, noise_mask, stats