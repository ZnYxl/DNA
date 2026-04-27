# models/step2_refine.py
"""
Step2: Evidence-Guided Cluster Refinement (防 OOM + 高速版)

v3 变更:
  [NEW] 自适应三区制划分:
        第一刀: CDF 拐点检测 (Kneedle) 切 Zone III (U_ale 长尾)
        第二刀: K=2 GMM 切 Zone I / Zone II (U_epi 双峰)
        Fallback: 拐点找不到 → P90, GMM 不收敛 → P70

保留:
  [FIX-1] refine_reads 返回 3 个值 (new_labels, noise_mask, stats)
  [FIX-2] refine_reads 接受 round_idx 参数，内部做 delta scaling
  [FIX-3] compute_global_delta 签名与 runner 一致
  [FIX-4] refine 距离计算: CPU 双轴分块 matmul，避免 OOM
  [FIX-5] compute_global_delta 只返回原始 delta

G老师审查修复清单:
  [G老师-Bug1-FIX] GMM 加权中点公式纠正 (_find_gmm_threshold 第160行)
    原来: threshold = (mu0*w1 + mu1*w0) / (w0+w1)  ← w0/w1位置互换
    w0=1/std0 很大(Zone I集中)被错误地乘到 mu1 上，阈值被拉向 Zone II 山峰，
    导致大量 Zone II reads 被误判为 Zone I，三区制防线失效。
    修复: threshold = (mu0*w0 + mu1*w1) / (w0+w1)，阈值正确靠近 Zone I。

  [G老师-Bug2-FIX] compute_centroids_weighted 改为 O(N) 哈希索引
    原来: for k in unique_labels: mask = (labels==k)
    K=40000 × N=400万 = 1600亿次布尔比较，每次 Step2 卡死数分钟。
    修复: 一次遍历建 defaultdict {label: [indices]}，之后 O(1) 直接取值。

  [G老师-Bug3-FIX] compute_global_delta 改为纯向量化
    原来: for k in unique: km = (z1_labels==k)
    Zone I ~240万 reads × ~40000 簇 = 同款算力黑洞。
    修复: 提取每条 Zone I read 对应质心堆成矩阵，一次性批量计算距离。
"""
import torch
import torch.nn.functional as F
import numpy as np
import time

# ---------------------------------------------------------------------------
# 超参常量
# ---------------------------------------------------------------------------
# Fallback percentiles (当自适应方法失败时使用)
FALLBACK_DIRTY_PERCENTILE = 0.10
FALLBACK_SAFE_PERCENTILE  = 0.70

MIN_ZONE1_SAFETY   = 3
ZONE2_WEIGHT_CAP   = 0.30
DELTA_P            = 75    # [v2] 提升到 P95，配合小簇保护提高复活率
# ROUND1_DELTA_SCALE 已删除: refine_reads 被 FIX-ZONE2 废弃, 该常量不再使用
MAX_ZONE3_RATIO = 0.15   # [FIX-Bug#2] Zone III 绝对上限
SMALL_CLUSTER_THRESHOLD = 5    # [v2-策略一] 小簇保护: <5 reads 的簇免疫 Zone III v4改为10了



# ===========================================================================
# 自适应阈值工具函数
# ===========================================================================
def _find_cdf_knee(values_sorted):
    """
    CDF 拐点检测 (Kneedle 算法简化版)

    对排序后的值计算经验 CDF, 找曲率最大的点。
    物理含义: CDF 从平缓爬升突然变陡的位置 = 长尾噪声开始的边界。

    Args:
        values_sorted: 1D numpy array, 已升序排序

    Returns:
        knee_value: float, 拐点对应的值; 或 None (找不到)
    """
    N = len(values_sorted)
    if N < 100:
        return None

    # 经验 CDF: x = 值, y = 累积比例
    # 为了稳定性, 下采样到 1000 个点
    n_points = min(1000, N)
    indices = np.linspace(0, N - 1, n_points, dtype=int)
    x = values_sorted[indices].astype(np.float64)
    y = indices / (N - 1)  # CDF: [0, 1]

    # 归一化到 [0, 1] × [0, 1]
    x_min, x_max = x[0], x[-1]
    if x_max - x_min < 1e-10:
        return None
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = y  # 已经是 [0, 1]

    # Kneedle: 找离对角线 (0,0)→(1,1) 最远的点
    # U_ale 是右偏分布: 大量 reads 集中在低值, CDF 先快速爬升再变缓 → 凹函数
    # 凹 CDF 曲线全程在对角线上方, 即 y_norm > x_norm 全程成立
    # 拐点 = 曲线离对角线最远处 = y_norm - x_norm 最大的点
    # [FIX] 原代码用 x_norm - y_norm (凸函数公式), 对凹 CDF 全程为负,
    #       argmax 只会落在边界点而非真正的肘点, 导致几乎永远触发 fallback
    diff = y_norm - x_norm

    # 忽略首尾 5% (避免边界效应)
    margin = max(1, n_points // 20)
    search_region = diff[margin:-margin]
    if len(search_region) == 0:
        return None

    knee_idx = margin + np.argmax(search_region)
    knee_value = float(x[knee_idx])

    # 验证: 拐点切出的比例应在 [2%, 30%] 之间, 否则不可信
    knee_ratio = 1.0 - y[knee_idx]  # 拐点右侧的比例 = Zone III 比例
    if knee_ratio < 0.02 or knee_ratio > 0.30:
        return None

    return knee_value


def _find_gmm_threshold(values, fallback_percentile=0.70):
    """
    K=2 一维 GMM 拟合, 返回两个高斯分量的交叉点阈值。

    物理含义: U_epi 的两个分量分别对应
      - 低 U_epi (模型确定) → Zone I
      - 高 U_epi (模型不确定) → Zone II

    Args:
        values: 1D numpy array
        fallback_percentile: GMM 失败时 fallback 到的百分位

    Returns:
        threshold: float
        method: str ('gmm' or 'fallback')
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print("      ⚠️ sklearn 不可用, fallback 到百分位")
        return float(np.quantile(values, fallback_percentile)), 'fallback'

    if len(values) < 100:
        return float(np.quantile(values, fallback_percentile)), 'fallback'

    # 下采样加速 (GMM 对 400 万数据太慢)
    max_samples = 200000
    if len(values) > max_samples:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(values), max_samples, replace=False)
        fit_data = values[sample_idx].reshape(-1, 1)
    else:
        fit_data = values.reshape(-1, 1)

    try:
        gmm = GaussianMixture(
            n_components=2,
            covariance_type='full',
            n_init=5,
            max_iter=200,
            random_state=42
        )
        gmm.fit(fit_data)

        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())

        # 确保 component 0 是低值(Zone I), component 1 是高值(Zone II)
        order = np.argsort(means)
        mu0, mu1 = means[order[0]], means[order[1]]
        std0, std1 = stds[order[0]], stds[order[1]]

        # 检查两个分量是否真的分开
        separation = abs(mu1 - mu0) / max(std0 + std1, 1e-10)
        if separation < 0.3:
            # 两个分量几乎重合, GMM 不可信
            print(f"      ⚠️ GMM 两分量重合 (sep={separation:.2f}), fallback")
            return float(np.quantile(values, fallback_percentile)), 'fallback'

        # [问题5修复] 原来只扫 [mu0, mu1]，如果真正的 PDF 交叉点在 mu1 右侧
        # （std1 >> std0 时可能出现），会漏掉交叉点，导致阈值直接取 mu1（偏大）。
        # 修复：扫描范围扩展到 [mu0 - std0, mu1 + std1]，覆盖两侧尾部，
        # 代价只是 linspace 的范围略大，精度不变。
        scan_lo = mu0 - std0
        scan_hi = mu1 + std1
        x_scan = np.linspace(scan_lo, scan_hi, 2000).reshape(-1, 1)
        proba  = gmm.predict_proba(x_scan)
        order_idx = order[0]
        p0 = proba[:, order_idx]
        cross = np.where(p0 < 0.5)[0]
        if len(cross) == 0:
            threshold = float(mu1)
        else:
            threshold = float(x_scan[cross[0], 0])

        # 验证: 阈值切出的 Zone I 比例应在 [40%, 95%]
        zone1_ratio = (values < threshold).sum() / len(values)
        if zone1_ratio < 0.40 or zone1_ratio > 0.95:
            print(f"      ⚠️ GMM 阈值不合理 (Zone I={zone1_ratio:.1%}), fallback")
            return float(np.quantile(values, fallback_percentile)), 'fallback'

        return float(threshold), 'gmm'

    except Exception as e:
        print(f"      ⚠️ GMM 拟合失败 ({e}), fallback")
        return float(np.quantile(values, fallback_percentile)), 'fallback'


# ===========================================================================
# 1. 三区制划分 (自适应版)
# ===========================================================================
def split_confidence_by_zone(u_epi, u_ale, labels, include_noise=False):
    """
    自适应三区制划分:
      第一刀: CDF 拐点 (Kneedle) → Zone III (U_ale 长尾噪声)
      第二刀: K=2 GMM            → Zone I / Zone II (U_epi 认知边界)

    [FIX-Bug#2] 新增安全阀: Zone III 实际比例超过 30% 时强制回退
    [FIX-Bug#6] 打印实际比例而非硬编码 "≈10%"

    [v18 Zone 全量判定]
      include_noise=False (默认, v17 及以前行为):
          Zone 判定只看 labels >= 0 的 reads.
          问题: -1 reads 被排除, 一旦标 -1 就失去被重新评估的机会,
                导致每轮 5% 的 reads 单向流失到 -1, 累积到 R3 时占 14%.
      include_noise=True (v18 行为):
          labels=-1 的 reads 也参与 Zone 判定. 它们的 u_ale 可能不长尾
          (不是 Zone III), 应该分到 Zone I/II 并通过质心匹配重获 label.
    """
    N      = len(labels)
    device = labels.device
    zone_ids = torch.zeros(N, dtype=torch.long, device=device)

    if include_noise:
        # [v18] 全量参与 Zone 判定, -1 reads 也能被重新审判
        valid = torch.ones(N, dtype=torch.bool, device=device)
    else:
        # [v17] 只看正常 label reads (会导致 -1 累积)
        valid = (labels >= 0)
    n_valid = valid.sum().item()
    if n_valid == 0:
        return zone_ids, {'zone1': 0, 'zone2': 0, 'zone3': 0, 'noise': N}

    # ===== 第一刀: U_ale → Zone III =====
    ale_valid = u_ale[valid].cpu().numpy()
    ale_sorted = np.sort(ale_valid)

    knee_value = _find_cdf_knee(ale_sorted)

    # [FIX-Bug#2] Kneedle 成功时也要验证比例
    if knee_value is not None:
        ale_threshold = knee_value
        zone3_ratio = (ale_valid >= ale_threshold).sum() / len(ale_valid)
        if zone3_ratio > MAX_ZONE3_RATIO:
            print(f"   🔪 第一刀 (U_ale → Zone III): CDF knee 比例={zone3_ratio:.1%} 过高, 降级 fallback")
            knee_value = None  # 强制走 fallback
        else:
            print(f"   🔪 第一刀 (U_ale → Zone III): CDF knee")
            print(f"      阈值 = {ale_threshold:.6f}, Zone III = {zone3_ratio:.1%}")

    if knee_value is None:
        # Fallback: P90
        ale_threshold = float(np.quantile(ale_valid, 1.0 - FALLBACK_DIRTY_PERCENTILE))
        ale_method = f'fallback P{int((1-FALLBACK_DIRTY_PERCENTILE)*100)}'
        print(f"   🔪 第一刀 (U_ale → Zone III): {ale_method} (拐点未找到)")
        print(f"      阈值 = {ale_threshold:.6f}")

    # [FIX-Bug#2] 安全阀：验证实际 Zone III 比例，防止天花板导致雪崩
    actual_zone3_ratio = float((ale_valid >= ale_threshold).mean())
    if actual_zone3_ratio > MAX_ZONE3_RATIO:
        # 阈值切太多了（典型原因：大量数据挤在同一个值上）
        # 策略：用严格大于 + epsilon 打破平局
        ale_threshold_new = float(np.quantile(ale_valid, 1.0 - FALLBACK_DIRTY_PERCENTILE)) + 1e-6
        new_ratio = float((ale_valid >= ale_threshold_new).mean())
        if new_ratio > MAX_ZONE3_RATIO:
            # 仍然超标：按排序直接切最脏的 10%
            target_idx = int(len(ale_sorted) * (1.0 - FALLBACK_DIRTY_PERCENTILE))
            target_idx = min(target_idx, len(ale_sorted) - 1)
            ale_threshold_new = float(ale_sorted[target_idx]) + 1e-6
            new_ratio = float((ale_valid >= ale_threshold_new).mean())
        print(f"      ⚠️ 安全阀触发: 原比例={actual_zone3_ratio:.1%} 过高, "
              f"新阈值={ale_threshold_new:.6f}, 新比例={new_ratio:.1%}")
        ale_threshold = ale_threshold_new
        actual_zone3_ratio = new_ratio

    print(f"      Zone III 实际比例 = {actual_zone3_ratio:.1%}")

    is_dirty = valid & (u_ale >= ale_threshold)

    # ── [v2-策略一] 小簇保护 ──
    # 簇大小 < SMALL_CLUSTER_THRESHOLD 的 reads 免疫 Zone III，降级为 Zone II。
    # 打薄后每簇仅 ~30 reads，Zone III 切掉 5-7% 就可能让小簇跌破
    # consensus 多数投票的最低法定人数。保护底线：5 条 reads。
    label_counts = torch.bincount(labels.clamp(min=0).long())
    read_cluster_sizes = label_counts[labels.clamp(min=0).long()]
    protected_mask = (read_cluster_sizes < SMALL_CLUSTER_THRESHOLD) & valid
    n_protected = int((is_dirty & protected_mask).sum().item())
    if n_protected > 0:
        is_dirty = is_dirty & (~protected_mask)
        print(f"      🛡️ 小簇保护: {n_protected} reads 从 Zone III 降级为 Zone II "
              f"(簇大小 < {SMALL_CLUSTER_THRESHOLD})")

    zone_ids[is_dirty] = 3

    # ===== 第二刀: U_epi → Zone I / Zone II =====
    remaining = valid & (~is_dirty)
    n_remaining = remaining.sum().item()

    if n_remaining > 0:
        epi_remaining = u_epi[remaining].cpu().numpy()

        epi_threshold, epi_method = _find_gmm_threshold(
            epi_remaining, FALLBACK_SAFE_PERCENTILE
        )

        zone1_mask = remaining & (u_epi <= epi_threshold)
        zone2_mask = remaining & (u_epi > epi_threshold)
        zone_ids[zone1_mask] = 1
        zone_ids[zone2_mask] = 2

        z1_ratio = zone1_mask.sum().item() / max(n_remaining, 1)
        print(f"   🔪 第二刀 (U_epi → Zone I/II): {epi_method}")
        print(f"      阈值 = {epi_threshold:.6f}, Zone I = {z1_ratio:.1%} of remaining")

    # ===== 汇总 =====
    z1 = int((zone_ids == 1).sum().item())
    z2 = int((zone_ids == 2).sum().item())
    z3 = int((zone_ids == 3).sum().item())
    zn = int((zone_ids == 0).sum().item())

    print(f"   📊 三区制划分 (自适应):")
    print(f"      Zone I  (Safe):   {z1:>8d}  ({z1/max(n_valid,1)*100:5.1f}%)")
    print(f"      Zone II (Hard):   {z2:>8d}  ({z2/max(n_valid,1)*100:5.1f}%)")
    print(f"      Zone III(Dirty):  {z3:>8d}  ({z3/max(n_valid,1)*100:5.1f}%)")
    print(f"      Noise   (skip):   {zn:>8d}")

    return zone_ids, {'zone1': z1, 'zone2': z2, 'zone3': z3, 'noise': zn}


# ===========================================================================
# 2. 加权质心计算
# ===========================================================================
def compute_centroids_weighted(embeddings, labels, strength, zone_ids):
    """
    Zone I + Zone II 参与, Zone III 不参与
    权重 = Strength, 小簇安全阀触发时 Zone II 权重截断

    [G老师-Bug2-FIX] 用 O(N) 哈希倒排索引替换 O(K×N) 全局扫描。
    原来对每个簇执行 `labels == k`，K=40000 × N=400万 = 1600亿次布尔比较。
    现在：一次遍历建好 {label: [indices]} 字典，之后直接用下标取值。
    """
    from collections import defaultdict

    D = embeddings.shape[1]
    centroids = {}
    cluster_sizes = {}

    # O(N) 建立倒排索引：label → 属于该簇的所有 read 下标
    label_to_idx = defaultdict(list)
    labels_cpu = labels.cpu().numpy()
    for i, l in enumerate(labels_cpu):
        if l >= 0:
            label_to_idx[int(l)].append(i)

    for k_int, idxs in label_to_idx.items():
        mask_idx = torch.tensor(idxs, dtype=torch.long, device=embeddings.device)

        z_ids     = zone_ids[mask_idx]
        zone_mask = (z_ids == 1) | (z_ids == 2)
        if not zone_mask.any():
            continue

        valid_idx = mask_idx[zone_mask]
        emb = embeddings[valid_idx]
        w   = strength[valid_idx].clone()

        # 安全阀: Zone I < 3 条时, Zone II 权重截断
        z1_count = int((z_ids == 1).sum().item())
        if z1_count < MIN_ZONE1_SAFETY:
            is_z2 = zone_ids[valid_idx] == 2
            w[is_z2] = w[is_z2].clamp(max=ZONE2_WEIGHT_CAP)

        w_sum = w.sum()
        if w_sum < 1e-10:
            centroids[k_int] = emb.mean(dim=0)
        else:
            centroids[k_int] = (emb * w.unsqueeze(1)).sum(dim=0) / w_sum

        cluster_sizes[k_int] = len(idxs)   # 全簇 reads 数（含 Zone III）

    print(f"   📍 质心: {len(centroids)} 个簇")
    return centroids, cluster_sizes


# ===========================================================================
# 3. Global Delta
# ===========================================================================
def compute_global_delta(embeddings, labels, zone_ids, centroids):
    """Zone I reads 到自身质心的距离分布 P95 → delta

    [G老师-Bug3-FIX] 用纯向量化替换 O(K×N) 循环。
    原来对每个唯一簇执行 `z1_labels == k`，Zone I 有 ~240万 reads，
    ~40000 个簇，同样是数百亿次布尔比较。
    修复：提取每条 Zone I read 对应质心的向量，一次性批量计算距离。
    """
    z1_mask = (zone_ids == 1)
    if z1_mask.sum() == 0:
        print("   ⚠️ Zone I 为空, delta 设为 1.0")
        return 1.0

    z1_labels = labels[z1_mask]
    z1_emb    = embeddings[z1_mask]

    # 过滤掉不在 centroids 里的 reads（极少数边界情况）
    z1_labels_list = z1_labels.tolist()
    valid_flags = torch.tensor(
        [l in centroids for l in z1_labels_list],
        dtype=torch.bool, device=z1_emb.device
    )

    if not valid_flags.any():
        print("   ⚠️ 无有效 Zone I 距离, delta 设为 1.0")
        return 1.0

    z1_labels_valid = [z1_labels_list[i] for i in range(len(z1_labels_list)) if valid_flags[i]]
    z1_emb_valid    = z1_emb[valid_flags]

    # 为每条 valid Zone I read 取出其对应质心，堆成 (M, D) 矩阵
    aligned_centroids = torch.stack([centroids[l] for l in z1_labels_valid])

    # 一次性计算所有 reads 到自身质心的 L2 距离
    all_dist = torch.norm(z1_emb_valid - aligned_centroids, dim=1)

    delta = float(torch.quantile(all_dist, DELTA_P / 100.0).item())
    print(f"   📏 Global Delta = {delta:.4f} (P{DELTA_P} of {len(all_dist)} Zone I distances)")
    return delta


# ===========================================================================
# 4. Zone-aware 修正 (已废弃 — FIX-ZONE2)
# ===========================================================================
# [DEAD CODE] refine_reads 不再被 step2_runner 调用。
# 原因: FIX-ZONE2 决定不做 Zone II 重分配，直接使用 MNN 合并后的标签，
#       避免 Zone I 无法覆盖全部 GT 时强制重分配丢失有效 reads。
# 保留函数体仅供参考，请勿在新代码中调用。
def refine_reads(embeddings, labels, zone_ids, centroids, delta,
                 round_idx=1, chunk_size=5000):
    """
    [DEPRECATED — FIX-ZONE2] 此函数不再被调用，保留仅供参考。

    Zone I:   保持
    Zone II:  距离判决 (< eff_delta → 最近簇, ≥ → -1)
    Zone III: -1
    """
    raise RuntimeError("❌ refine_reads 已废弃，请勿调用！使用 merge_close_centroids 替代")
    t_start = time.time()

    # [DEPRECATED] ROUND1_DELTA_SCALE 已删除，eff_delta 直接等于 delta
    eff_delta = delta
    print(f"\n   🔧 修正 (round={round_idx}, delta={delta:.4f}) [DEPRECATED, 不应被调用]")

    N = len(labels)
    new_labels = labels.clone()

    # Zone III → -1
    z3_mask = (zone_ids == 3)
    new_labels[z3_mask] = -1

    # Zone II → 距离判决
    z2_mask = (zone_ids == 2)
    z2_indices = torch.where(z2_mask)[0]
    n_z2 = len(z2_indices)

    if n_z2 == 0 or len(centroids) == 0:
        noise_mask = (new_labels == -1)
        print(f"   ✅ 修正完成 (Zone II 为空)")
        return new_labels, noise_mask, {
            'zone1_kept': int((zone_ids == 1).sum()),
            'zone2_reassigned': 0,
            'zone2_to_noise': 0,
            'zone3_dirty': int(z3_mask.sum()),
        }

    cids = sorted(centroids.keys())
    centroid_matrix = torch.stack([centroids[c] for c in cids])  # (K, D)

    reassigned = 0
    to_noise = 0

    for start in range(0, n_z2, chunk_size):
        end = min(start + chunk_size, n_z2)
        batch_idx = z2_indices[start:end]
        batch_emb = embeddings[batch_idx]

        # 分块距离 (防 OOM)
        dists = torch.cdist(batch_emb, centroid_matrix)  # (chunk, K)
        min_dists, min_idx = dists.min(dim=1)

        for i in range(len(batch_idx)):
            idx = batch_idx[i]
            if min_dists[i] < eff_delta:
                new_labels[idx] = cids[min_idx[i].item()]
                reassigned += 1
            else:
                new_labels[idx] = -1
                to_noise += 1

        if (start + chunk_size) % 50000 < chunk_size:
            print(f"      Zone II 进度: {min(end, n_z2)}/{n_z2}", flush=True)

    noise_mask = (new_labels == -1)
    elapsed = time.time() - t_start

    stats = {
        'zone1_kept': int((zone_ids == 1).sum()),
        'zone2_reassigned': reassigned,
        'zone2_to_noise': to_noise,
        'zone3_dirty': int(z3_mask.sum()),
    }

    total_noise = int(noise_mask.sum())
    total_assigned = N - total_noise
    print(f"   ✅ 修正完成 ({elapsed:.1f}s)")
    print(f"      Zone I 保持:      {stats['zone1_kept']:>8d}")
    print(f"      Zone II 重分配:   {reassigned:>8d}")
    print(f"      Zone II → 噪声:   {to_noise:>8d}")
    print(f"      Zone III 丢弃:    {stats['zone3_dirty']:>8d}")
    print(f"      总分配: {total_assigned:,} / {N:,} ({total_assigned/N*100:.1f}%)")

    return new_labels, noise_mask, stats