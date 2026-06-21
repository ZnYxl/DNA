# models/cluster_split.py
# -*- coding: utf-8 -*-
"""
SSI-EC v21 迭代引擎: 簇内拆分 (Intra-cluster Split)
====================================================
背景(三个 spike 的结论):
  - 合并方向救 0 个 success(死), edit 合并已废弃。
  - Clover 初始聚类的真损失在"欠分割": 体量相当的两个 GT 分子被错并进同一簇,
    次要 GT 被压制拿不到自己的 consensus。
  - 拆分上界: 上帝视角 +581 success; 无监督全量模拟(本机制)实测净 +518(τ=5)。
  - 机制: 簇内 read 做 edit 距离层次聚类二分 → 两子簇各自 MV consensus →
          若两 consensus 的 edit ≥ τ(默认5) 判定"两个分子被错并"→ 拆; 否则不拆。
  - 纯簇天然受保护: 同源 read 二分后两半 consensus 几乎一致(edit 0-1), 过不了 τ 门控。
  - τ=5 在"近似重复异源最小间距=2"之上留安全垫, 误伤仅 6 个, 泛化稳。

为什么这是"让迭代真正有用"的引擎:
  关闭 Zone III 后 step2 空转(labels 逐字节不变)。本拆分是唯一会改变 labels 的机制,
  且只重组已分配 read、不产生 -1, 对 read_util(覆盖轴)中性 → 不会引发覆盖率塌陷。
  拆分改变 labels → 改变下一轮训练 consensus 靶点 → 形成真实迭代闭环。

插入位置(step2_runner.py):
  在 `new_labels_np = new_labels.cpu().numpy()` 之后、首个 run_feddna_decode 之前。
  对 new_labels_np 原地拆分, 下游 consensus / 落盘 / 下一轮训练自动跟随, 零改动。

依赖: edlib(eval 已用) + scipy。复用 eval_reconstruction 的 levenshtein 口径由调用方注入。
"""
import numpy as np
from collections import defaultdict, Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def _mv_consensus(read_seqs, ref_length):
    """逐位多数投票 + 50% has_vote 门限。与 compute_mv_consensus / ds_fusion_masked 同口径。"""
    N = len(read_seqs)
    if N == 0:
        return ""
    thresh = max(N * 0.5, 1)
    out = []
    for pos in range(ref_length):
        cnt = Counter()
        valid = 0
        for s in read_seqs:
            if pos < len(s):
                valid += 1
                b = s[pos]
                if b in 'ACGT':
                    cnt[b] += 1
        if valid >= thresh and cnt:
            out.append(cnt.most_common(1)[0][0])
    return ''.join(out)


def _split_two(seqs, levenshtein, max_pairwise=80, seed=0):
    """
    对一组 read 序列做 edit 距离层次聚类二分。
    返回两组的局部 index list (a_local, b_local)。
    簇内 read > max_pairwise 时抽样建矩阵, 其余按到两子簇 consensus 的距离就近分配。
    """
    n = len(seqs)
    if n < 2:
        return list(range(n)), []

    if n <= max_pairwise:
        idxs = list(range(n))
        sub = seqs
    else:
        rng = np.random.default_rng(seed)
        idxs = sorted(rng.choice(n, max_pairwise, replace=False).tolist())
        sub = [seqs[i] for i in idxs]

    m = len(sub)
    D = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(i + 1, m):
            d = levenshtein(sub[i], sub[j])
            D[i, j] = D[j, i] = d
    Z = linkage(squareform(D, checks=False), method='average')
    lab = fcluster(Z, t=2, criterion='maxclust')  # 值域 {1,2}
    a_local = [idxs[i] for i in range(m) if lab[i] == 1]
    b_local = [idxs[i] for i in range(m) if lab[i] == 2]

    if n > max_pairwise:
        ca = _mv_consensus([seqs[i] for i in a_local],
                           max((len(seqs[i]) for i in a_local), default=0)) if a_local else ""
        cb = _mv_consensus([seqs[i] for i in b_local],
                           max((len(seqs[i]) for i in b_local), default=0)) if b_local else ""
        assigned = set(idxs)
        for i in range(n):
            if i in assigned:
                continue
            da = levenshtein(seqs[i], ca) if ca else 1e9
            db = levenshtein(seqs[i], cb) if cb else 1e9
            (a_local if da <= db else b_local).append(i)

    return a_local, b_local


def split_clusters(
    new_labels_np,
    flat_real_indices,
    data_loader,
    levenshtein,
    ref_length=196,
    tau=5,
    min_split_size=6,
    max_pairwise=80,
    verbose=True,
):
    """
    对 new_labels_np 原地做簇内拆分(只读 data_loader.reads, 不改任何文件)。

    Args:
        new_labels_np:     (M,) int, didx 索引空间的标签(-1 表噪声, 已被排除拆分)
        flat_real_indices: didx -> data_loader.reads 真实索引
        data_loader:       提供 data_loader.reads[real_idx] -> 序列字符串
        levenshtein:       edit 距离函数(由调用方注入, 用 eval_reconstruction 的 edlib 版)
        ref_length:        consensus 截断长度(Seq_1D=196)
        tau:               门控阈值, 两子簇 consensus edit >= tau 才拆(默认5)
        min_split_size:    簇 read 数 < 此值不尝试拆
        max_pairwise:      簇内建距离矩阵的最大 read 数(超出抽样)

    Returns:
        out_labels:  (M,) int, 拆分后的新标签(新增子簇分配了新 cluster_id)
        stats:       dict, 拆分统计
    """
    labels = new_labels_np.copy()

    # didx 按 cluster 分组(只处理 label >= 0)
    cl_to_didx = defaultdict(list)
    for didx, lab in enumerate(labels):
        if lab >= 0:
            cl_to_didx[int(lab)].append(didx)

    # 新 ID 从 max+1 起, 保证不撞号
    next_id = int(labels.max()) + 1 if (labels >= 0).any() else 0

    n_split = 0
    n_examined = 0
    split_dists = []

    for cid, didxs in cl_to_didx.items():
        if len(didxs) < min_split_size:
            continue
        n_examined += 1

        seqs = [data_loader.reads[flat_real_indices[d]] for d in didxs]
        a_loc, b_loc = _split_two(seqs, levenshtein, max_pairwise=max_pairwise)
        if len(a_loc) < 1 or len(b_loc) < 1:
            continue

        consA = _mv_consensus([seqs[i] for i in a_loc], ref_length)
        consB = _mv_consensus([seqs[i] for i in b_loc], ref_length)
        if not consA or not consB:
            continue
        dAB = levenshtein(consA, consB)

        if dAB >= tau:
            # 拆: A 子簇保留原 cid, B 子簇分配新 id
            b_didxs = [didxs[i] for i in b_loc]
            labels[b_didxs] = next_id
            next_id += 1
            n_split += 1
            split_dists.append(dAB)

    stats = {
        'tau': tau,
        'clusters_examined': n_examined,
        'clusters_split': n_split,
        'new_clusters_added': n_split,
        'n_clusters_before': len(cl_to_didx),
        'n_clusters_after': len(cl_to_didx) + n_split,
        'split_dist_median': float(np.median(split_dists)) if split_dists else 0.0,
    }

    if verbose:
        print(f"\n   ✂️  [v21] 簇内拆分 (τ={tau}, min_size={min_split_size})")
        print(f"      考察簇(size>={min_split_size}): {n_examined:,}")
        print(f"      实际拆分:                  {n_split:,}")
        print(f"      簇数: {stats['n_clusters_before']:,} → {stats['n_clusters_after']:,}")
        if split_dists:
            print(f"      拆分对 consensus edit 中位: {stats['split_dist_median']:.0f}")

    return labels, stats