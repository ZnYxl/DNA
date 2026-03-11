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

调用位置: step2_runner.py 中 compute_centroids_weighted() 之后,
          compute_global_delta() 之前
"""
import torch
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict


def merge_close_centroids(centroids, labels, cluster_sizes,
                          embeddings, zone_ids, strength,
                          threshold=0.95,
                          max_cluster_size=2000,
                          max_rounds=30,
                          chunk_size=2000):
    """
    安全版簇合并: MNN + 最大簇大小约束 + 迭代。

    Args:
        centroids:        dict {cluster_id: tensor(D,)}
        labels:           tensor(N,) 当前标签
        cluster_sizes:    dict {cluster_id: int}
        embeddings:       tensor(N, D) 全量 embeddings
        zone_ids:         tensor(N,) zone 标记
        strength:         tensor(N,) strength
        threshold:        float, cosine similarity 合并阈值 (默认 0.95，比上次的 0.90 更保守)
        max_cluster_size: int, 合并后簇大小上限 (默认 2000)
        max_rounds:       int, 最大迭代轮数
        chunk_size:       int, 分块大小

    Returns:
        new_centroids, new_labels, merge_stats
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"🔗 安全簇合并 (MNN, threshold={threshold}, max_size={max_cluster_size})")
    print(f"{'='*60}")

    if len(centroids) < 2:
        print(f"   ⚠️ 簇数 < 2, 跳过")
        return centroids, labels, {'n_merges': 0}

    initial_K = len(centroids)
    total_merges = 0
    labels = labels.clone()

    # 维护动态簇大小
    current_sizes = {}
    for cid in centroids:
        current_sizes[cid] = int((labels == cid).sum().item())

    for round_idx in range(max_rounds):
        # ── 1. 构建质心矩阵 ──
        cids = sorted(centroids.keys())
        K = len(cids)
        if K < 2:
            break

        cid_to_idx = {c: i for i, c in enumerate(cids)}
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

        # ── 3. 找 Mutual Nearest Neighbor 对 ──
        merge_pairs = []
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
                        merge_pairs.append((nn_sim[i].item(), cid_a, cid_b))

        if len(merge_pairs) == 0:
            print(f"   Round {round_idx+1}: 无可合并的 MNN 对, 终止")
            break

        # ── 4. 按 similarity 降序执行合并 ──
        merge_pairs.sort(key=lambda x: -x[0])

        # 每轮中已被合并的簇不能再参与（避免冲突）
        merged_this_round = set()
        round_merges = 0

        for sim_val, cid_a, cid_b in merge_pairs:
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
            round_merges += 1

        total_merges += round_merges

        # ── 5. 重算合并后的质心 ──
        for cid in list(centroids.keys()):
            mask = (labels == cid)
            zone_mask = (zone_ids == 1) | (zone_ids == 2)
            valid_mask = mask & zone_mask
            count = valid_mask.sum().item()
            if count == 0:
                continue
            emb = embeddings[valid_mask]
            w = strength[valid_mask].clone()
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

    t1 = time.time()

    final_K = len(centroids)
    # 更新 cluster_sizes
    new_cluster_sizes = {}
    for cid in centroids:
        new_cluster_sizes[cid] = int((labels == cid).sum().item())

    merge_stats = {
        'clusters_before': initial_K,
        'clusters_after': final_K,
        'n_merges': total_merges,
        'time_seconds': t1 - t0,
        'threshold': threshold,
        'max_cluster_size': max_cluster_size,
    }

    print(f"\n   📊 合并总结:")
    print(f"      簇数: {initial_K} → {final_K} (减少 {initial_K - final_K})")
    print(f"      合并次数: {total_merges}")
    print(f"      耗时: {t1-t0:.1f}s")

    if new_cluster_sizes:
        sizes = sorted(new_cluster_sizes.values(), reverse=True)
        print(f"      簇大小: max={sizes[0]}, median={sizes[len(sizes)//2]}, min={sizes[-1]}")

    return centroids, labels, merge_stats