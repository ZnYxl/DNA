#!/usr/bin/env python3
"""
models/merge_clusters.py — 安全版簇合并
========================================

上一版的致命问题:
  Union-Find 传递闭包: A-B相似 + B-C相似 → A,B,C全合并
  61K 簇 → 773 簇, 最大簇 3.88M reads, Purity 从 0.974 崩到 0.028

安全版设计 (两道保险):
  1. Mutual Nearest Neighbor (MNN):
     只合并互为最近邻的质心对。A 的最近邻是 B，且 B 的最近邻也是 A，才合并。
     这从根本上阻止了链式传播: A-B-C-D... 的传递闭包不会发生。

  2. 最大簇大小上限 (max_cluster_size):
     合并后的簇大小超过上限时拒绝合并。
     exp_1 的 GT: 11710 分子 / 3.92M reads → 平均 335 reads/分子。
     上限设 2000，覆盖 >99% 的 GT 簇，同时阻止巨型簇。

  3. 迭代合并 (多轮):
     每轮只做 MNN 合并 → 重算质心 → 下一轮。
     簇数逐步收敛，不会一步跳崩。

G老师审查修复清单:
  [G老师-Bug1-FIX] seq_jaccard_threshold: 0.5 → 0.05
    原来设 0.5 假设 consensus 质量好，但 Round2+ 的 consensus 与 GT 平均
    ED≈57，两条烂序列的 Jaccard 趋近于 0，几乎所有 MNN 正确找到的同源碎片
    对都被拒绝——R2+R3 合并骤降至 90 对的根本原因。
    0.05 只拦截序列完全不相关的误合并，不再误杀正确合并对。

  [G老师-Bug2-FIX] 质心重算：O(K×N) 全局 → 只重算本轮 keep 簇
    原来对所有 K 个簇执行 `labels == cid`，K=40000, N=400万时产生
    1600亿次布尔比较。实际只有本轮 keep 簇的成员发生了变化，
    其他簇质心完全没变。新增 keep_cids_this_round 集合，只重算这些簇。

  [G老师-Bug3-FIX] 补上遗失的 Zone II 安全阀（与 step2_refine.py 一致）
    原来直接用全部 strength，碎片簇合并后 Zone II 的异常高 strength 会
    主导新质心方向，引发质心漂移。
    安全阀：Zone I reads < 3 条时，Zone II 的 strength 截断到 0.30。

调用位置: step2_runner.py 中 compute_centroids_weighted() 之后,
          compute_global_delta() 之前
"""
import torch
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict


# ---------------------------------------------------------------------------
# 序列双重校验工具函数
# ---------------------------------------------------------------------------
_BASE_MAP = ['A', 'C', 'G', 'T']

def _onehot_to_seq(one_hot: torch.Tensor) -> str:
    """将 (L, 4) one-hot tensor 转为碱基字符串，padding(全零)位截断"""
    has_vote = one_hot.sum(dim=-1) > 0          # (L,) bool
    L = int(has_vote.sum().item())
    if L == 0:
        return ''
    indices = one_hot[:L].argmax(dim=-1).tolist()
    return ''.join(_BASE_MAP[i] for i in indices)


def _kmer_jaccard(seq_a: str, seq_b: str, k: int = 8) -> float:
    """
    k-mer Jaccard 相似度。
    相比编辑距离，对 IDS 错误更鲁棒，且计算复杂度 O(L) 而非 O(L²)。
    返回 [0, 1]，越高越相似。
    """
    if len(seq_a) < k or len(seq_b) < k:
        return 0.0
    set_a = set(seq_a[i:i+k] for i in range(len(seq_a) - k + 1))
    set_b = set(seq_b[i:i+k] for i in range(len(seq_b) - k + 1))
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def merge_close_centroids(centroids, labels, cluster_sizes,
                          embeddings, zone_ids, strength,
                          threshold=0.98,
                          max_cluster_size=2000,
                          max_rounds=60,
                          chunk_size=2000,
                          consensus_dict=None,
                          seq_jaccard_threshold=0.05,
                          kmer_k=8,
                          target_clusters=None):
    """
    安全版簇合并: MNN + 序列双重校验 + 最大簇大小约束 + 先验簇数硬停止 + 迭代。

    Args:
        centroids:             dict {cluster_id: tensor(D,)}
        labels:                tensor(N,) 当前标签
        cluster_sizes:         dict {cluster_id: int}
        embeddings:            tensor(N, D) 全量 embeddings
        zone_ids:              tensor(N,) zone 标记
        strength:              tensor(N,) strength
        threshold:             float, cosine similarity 合并阈值 (默认 0.98)
        max_cluster_size:      int, 合并后簇大小上限 (默认 2000)
        max_rounds:            int, 最大迭代轮数 (默认 60)
        chunk_size:            int, 分块大小
        consensus_dict:        dict {cluster_id: tensor(L,4)} 上一轮的 consensus，
                               用于序列双重校验。为 None 时跳过序列校验。
        target_clusters:       int or None, 目标簇数下限（先验约束）。
                               MNN 合并按相似度降序执行，一旦簇数降至该值则立即停止。
                               数学上等价于 HAC dendrogram 在 K 处的贪心最优截断。
                               None 时不设下限（向后兼容）。
        seq_jaccard_threshold: float, k-mer Jaccard 最低阈值
                               [G老师-Bug1-FIX] 从 0.5 降到 0.05。
                               原设 0.5 假设 consensus 质量好，但 Round2+
                               的 consensus 与 GT 平均 ED≈57，两条烂序列的
                               Jaccard 趋近于 0，几乎所有 MNN 正确找到的
                               同源碎片对都被拒绝——R2+R3 合并骤降至 90 对
                               的根本原因。0.05 只拦截序列完全不相关的误合并。
        kmer_k:                int, k-mer 大小 (默认 8)

    Returns:
        new_centroids, new_labels, merge_stats
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    seq_check_str = f", seq_jaccard≥{seq_jaccard_threshold}" if consensus_dict else ", 无序列校验(Round1)"
    target_str = f", target_K≥{target_clusters}" if target_clusters else ""
    print(f"🔗 安全簇合并 (MNN, threshold={threshold}, max_size={max_cluster_size}{seq_check_str}{target_str})")
    print(f"{'='*60}")

    # 先验簇数保护：已经不多于目标，不合并
    if target_clusters is not None and len(centroids) <= target_clusters:
        print(f"   🛑 当前簇数 {len(centroids)} 已 ≤ 目标 {target_clusters}，跳过合并")
        early_sizes = {cid: int((labels == cid).sum().item()) for cid in centroids}
        return centroids, labels, {
            'clusters_before': len(centroids),
            'clusters_after':  len(centroids),
            'n_merges': 0,
            'time_seconds': 0.0,
            'threshold': threshold,
            'max_cluster_size': max_cluster_size,
        }, early_sizes

    if len(centroids) < 2:
        print(f"   ⚠️ 簇数 < 2, 跳过")
        # [问题1修复] 早返回路径补上 new_cluster_sizes，与正常路径返回值数量一致（4个）。
        # 原来只返回3个值，调用方解包4个值时直接 crash。
        early_sizes = {cid: int((labels == cid).sum().item()) for cid in centroids}
        return centroids, labels, {
            'clusters_before': len(centroids),
            'clusters_after':  len(centroids),
            'n_merges': 0,
            'time_seconds': 0.0,
            'threshold': threshold,
            'max_cluster_size': max_cluster_size,
        }, early_sizes

    initial_K = len(centroids)
    total_merges = 0
    labels = labels.clone()

    # [问题2修复] 用 Counter 一次 O(N) 遍历建立簇大小字典，替换原来的 O(K×N) 循环。
    # 原来: for cid in centroids: (labels == cid).sum() → 67K × 390万 = 2600亿次比较。
    from collections import Counter as _Counter
    _label_counts = _Counter(labels.tolist())
    current_sizes = {cid: _label_counts.get(cid, 0) for cid in centroids}

    for round_idx in range(max_rounds):
        # ── 1. 构建质心矩阵 ──
        cids = sorted(centroids.keys())
        K = len(cids)
        if K < 2:
            break

        # [问题3修复] 删除 cid_to_idx——全函数从未使用，每轮白白构建一次字典。
        centroid_matrix = torch.stack([centroids[c] for c in cids])  # (K, D)
        centroid_normed = F.normalize(centroid_matrix, dim=1)

        # ── 2. 分块计算全量 cosine similarity → 找每个质心的最近邻 ──
        # nn_idx[i] = 质心 i 的最近邻在 cids 中的索引
        # nn_sim[i] = 对应的 cosine similarity
        nn_sim = torch.full((K,), -1.0)
        nn_idx = torch.full((K,), -1, dtype=torch.long)

        for i_start in range(0, K, chunk_size):
            i_end = min(i_start + chunk_size, K)
            chunk = centroid_normed[i_start:i_end]  # (chunk, D)
            sim = chunk @ centroid_normed.T          # (chunk, K)

            # 排除自身
            for local_i in range(i_end - i_start):
                global_i = i_start + local_i
                sim[local_i, global_i] = -2.0  # 排除自身

            max_sim, max_idx = sim.max(dim=1)
            nn_sim[i_start:i_end] = max_sim
            nn_idx[i_start:i_end] = max_idx

        # ── 3. 找 Mutual Nearest Neighbor 对 + 序列双重校验 ──
        merge_pairs = []
        seq_rejected = 0
        for i in range(K):
            j = nn_idx[i].item()
            if j < 0:
                continue
            # MNN 条件: i 的最近邻是 j，且 j 的最近邻也是 i
            if nn_idx[j].item() == i and nn_sim[i].item() > threshold:
                # 去重: 只保留 i < j 的对
                if i < j:
                    cid_a = cids[i]
                    cid_b = cids[j]
                    # 大小约束: 合并后不超过上限
                    size_a = current_sizes.get(cid_a, 0)
                    size_b = current_sizes.get(cid_b, 0)
                    if size_a + size_b <= max_cluster_size:
                        # [序列双重校验] 有 consensus_dict 时验证序列相似度
                        if consensus_dict is not None:
                            ca = consensus_dict.get(cid_a)
                            cb = consensus_dict.get(cid_b)
                            if ca is not None and cb is not None:
                                seq_a = _onehot_to_seq(ca)
                                seq_b = _onehot_to_seq(cb)
                                jaccard = _kmer_jaccard(seq_a, seq_b, k=kmer_k)
                                if jaccard < seq_jaccard_threshold:
                                    seq_rejected += 1
                                    continue  # 序列不相似，拒绝合并
                        merge_pairs.append((nn_sim[i].item(), cid_a, cid_b))

        if seq_rejected > 0:
            print(f"   🔒 序列校验拒绝: {seq_rejected} 对 (Jaccard<{seq_jaccard_threshold})")

        if len(merge_pairs) == 0:
            print(f"   Round {round_idx+1}: 无可合并的 MNN 对, 终止")
            break

        # ── 4. 按 similarity 降序执行合并 ──
        merge_pairs.sort(key=lambda x: -x[0])

        # 每轮中已被合并的簇不能再参与（避免冲突）
        merged_this_round = set()
        keep_cids_this_round = set()   # [G老师-Bug2-FIX] 记录本轮实际发生合并的 keep 簇
        round_merges = 0

        for sim_val, cid_a, cid_b in merge_pairs:
            # 【先验簇数硬停止】贪心最优截断：合并列表按相似度降序排列，
            # 已执行的都是模型最确信的同源对；一旦簇数触底，后续的都是风险对。
            if target_clusters is not None and len(centroids) <= target_clusters:
                print(f"   🛑 先验保护：簇数 {len(centroids)} 已达目标底线 {target_clusters}，停止本轮合并")
                break

            if cid_a in merged_this_round or cid_b in merged_this_round:
                continue

            # 大簇吸收小簇
            size_a = current_sizes.get(cid_a, 0)
            size_b = current_sizes.get(cid_b, 0)
            if size_a >= size_b:
                keep, absorb = cid_a, cid_b
            else:
                keep, absorb = cid_b, cid_a

            # 更新标签
            mask = (labels == absorb)
            labels[mask] = keep

            # 更新簇大小
            current_sizes[keep] = current_sizes.get(keep, 0) + current_sizes.get(absorb, 0)
            current_sizes.pop(absorb, None)

            # 删除被吸收的质心
            centroids.pop(absorb, None)

            merged_this_round.add(cid_a)
            merged_this_round.add(cid_b)
            keep_cids_this_round.add(keep)   # [G老师-Bug2-FIX] 记录 keep 簇
            round_merges += 1

        total_merges += round_merges

        # ── 5. 重算合并后的质心 ──
        # [G老师-Bug2-FIX] 只重算本轮吸收了新成员的 keep 簇，不遍历全部 K 个簇。
        # 原来 for cid in centroids.keys() 在 K=40000, N=400万 时会产生
        # 40000 × 400万 = 1600亿次布尔比较，是严重的算力黑洞。
        # 只有 keep 簇的成员发生了变化，其他簇质心完全没变，不需要重算。
        for cid in keep_cids_this_round:
            if cid not in centroids:
                continue
            mask = (labels == cid)
            zone_mask = (zone_ids == 1) | (zone_ids == 2)
            valid_mask = mask & zone_mask
            count = valid_mask.sum().item()
            if count == 0:
                continue
            emb = embeddings[valid_mask]
            w = strength[valid_mask].clone()

            # [G老师-Bug3-FIX] 补上遗失的 Zone II 安全阀，与 step2_refine.py 保持一致。
            # 原来直接用全部 strength，碎片簇合并后 Zone II 的异常高 strength
            # 会直接主导新质心方向，引发质心漂移。
            # 安全阀：Zone I reads < 3 条时，Zone II 的 strength 截断到 0.30。
            z1_count = int((mask & (zone_ids == 1)).sum().item())
            if z1_count < 3:
                is_z2 = zone_ids[valid_mask] == 2
                w[is_z2] = w[is_z2].clamp(max=0.30)   # ZONE2_WEIGHT_CAP

            w_sum = w.sum()
            if w_sum < 1e-10:
                centroids[cid] = emb.mean(dim=0)
            else:
                centroids[cid] = (emb * w.unsqueeze(1)).sum(dim=0) / w_sum

        new_K = len(centroids)
        print(f"   Round {round_idx+1}: 合并 {round_merges} 对 MNN, "
              f"簇数 {K} → {new_K}", flush=True)

        if round_merges == 0:
            break

        # 先验簇数保护：外层循环也检查，防止下一轮继续合并
        if target_clusters is not None and len(centroids) <= target_clusters:
            print(f"   🛑 已达目标簇数底线 {target_clusters}，终止迭代合并")
            break

    t1 = time.time()

    final_K = len(centroids)
    # [问题2修复] new_cluster_sizes 同样改用 Counter O(N) 替换 O(K×N) 循环。
    # 合并后 K≈15K，15K × 390万 = 585亿次比较，同样是性能黑洞。
    _final_counts = _Counter(labels.tolist())
    new_cluster_sizes = {cid: _final_counts.get(cid, 0) for cid in centroids}

    merge_stats = {
        'clusters_before': initial_K,
        'clusters_after': final_K,
        'n_merges': total_merges,
        'time_seconds': t1 - t0,
        'threshold': threshold,
        'max_cluster_size': max_cluster_size,
        'target_clusters': target_clusters,
    }

    print(f"\n   📊 合并总结:")
    print(f"      簇数: {initial_K} → {final_K} (减少 {initial_K - final_K})")
    if target_clusters:
        print(f"      目标底线: {target_clusters} ({'✅ 已触底' if final_K <= target_clusters else f'距目标还差 {final_K - target_clusters}'})")
    print(f"      合并次数: {total_merges}")
    print(f"      耗时: {t1-t0:.1f}s")

    if new_cluster_sizes:
        sizes = sorted(new_cluster_sizes.values(), reverse=True)
        print(f"      簇大小: max={sizes[0]}, median={sizes[len(sizes)//2]}, min={sizes[-1]}")

    # [残留问题1修复] new_cluster_sizes 算好了但只打日志，没有返回出去。
    # 调用方用的仍是合并前的旧 cluster_sizes（含已被吸收消失的簇 ID），
    # 保存到磁盘的元数据是错的。修复：把它加入返回值。
    return centroids, labels, merge_stats, new_cluster_sizes