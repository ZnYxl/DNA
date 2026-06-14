#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spike_split_gate.py
===================
v21 拆分机制的最后一道验证(过了就动代码)。

前两个 spike 已证:
  - 合并方向救 0 个(死)
  - 拆分上界 +581(上帝视角), 无监督"序列或embedding"可达 420/581
  - 主信号是序列edit(68.8% > embedding 42.7%)
  - 156个近似重复(consensus≤2)是硬边界, 放弃

本 spike 模拟一个完整的无监督拆分算法跑全量 11648 簇, 测真实净ΔSR:
  对每个簇(全量, 无监督下分不清纯/混):
    1. 簇内 read 做 edit 距离 层次聚类 二分 -> 子簇A,B
    2. A,B 各自 MV consensus
    3. 门控: edit(consA, consB) >= τ 才判"两个分子被错并"-> 拆; 否则不拆
    4. 递归一层(拆出的子簇不再拆), 防碎片化
  净ΔSR = 拆对救回的 success  −  纯簇误拆掉的 success

τ 扫 {3,5,8,10,15} -> 给完整曲线, 看拐点。
GT 仅用于最后评判净ΔSR, 算法本身全程无 GT。

只读。不改任何文件。需 edlib(你服务器eval已用)+ scipy。
"""
import argparse
import os
import sys
import numpy as np
from collections import defaultdict, Counter


def _add_code_root():
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(8):
        if os.path.exists(os.path.join(d, 'models', 'eval_reconstruction.py')):
            if d not in sys.path: sys.path.insert(0, d)
            m = os.path.join(d, 'models')
            if m not in sys.path: sys.path.insert(0, m)
            return d
        p = os.path.dirname(d)
        if p == d: break
        d = p
    if here not in sys.path: sys.path.insert(0, here)
    return None

_root = _add_code_root()
if _root: print(f"[path] code 根: {_root}")

try:
    from eval_reconstruction import (
        levenshtein, load_reads_from_readtxt, load_gt_tags_file,
        load_gt_refs_fasta, build_tag_to_ref_mapping, match_reads_to_gt, find_read_txt,
    )
except ImportError as e:
    print(f"❌ import 失败: {e}"); sys.exit(1)

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def mv_consensus(read_seqs, ref_length):
    N = len(read_seqs)
    if N == 0: return ""
    thresh = max(N * 0.5, 1)
    out = []
    for pos in range(ref_length):
        cnt = Counter(); valid = 0
        for s in read_seqs:
            if pos < len(s):
                valid += 1
                if s[pos] in 'ACGT': cnt[s[pos]] += 1
        if valid >= thresh and cnt:
            out.append(cnt.most_common(1)[0][0])
    return ''.join(out)


def split_two(seqs, max_pairwise=80):
    """
    对一组 read 序列做 edit 距离层次聚类二分。
    返回两组的 index list (idxA, idxB)。
    为控开销: 簇内 read 数 > max_pairwise 时, 随机抽 max_pairwise 条建距离矩阵,
    其余 read 按到两个子簇 medoid 的距离就近分配。
    """
    n = len(seqs)
    if n < 2:
        return list(range(n)), []
    if n <= max_pairwise:
        idxs = list(range(n))
        sub = seqs
    else:
        rng = np.random.default_rng(0)
        idxs = sorted(rng.choice(n, max_pairwise, replace=False).tolist())
        sub = [seqs[i] for i in idxs]

    m = len(sub)
    D = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(i + 1, m):
            d = levenshtein(sub[i], sub[j])
            D[i, j] = D[j, i] = d
    Z = linkage(squareform(D, checks=False), method='average')
    lab = fcluster(Z, t=2, criterion='maxclust')  # 1/2
    a_local = [idxs[i] for i in range(m) if lab[i] == 1]
    b_local = [idxs[i] for i in range(m) if lab[i] == 2]

    if n > max_pairwise:
        # 用两组 medoid 把剩余 read 就近分配
        def medoid(group_local):
            gs = [seqs[i] for i in group_local]
            cons = mv_consensus(gs, max(len(x) for x in gs)) if gs else ""
            return cons
        ca = medoid(a_local); cb = medoid(b_local)
        assigned = set(idxs)
        for i in range(n):
            if i in assigned: continue
            da = levenshtein(seqs[i], ca) if ca else 1e9
            db = levenshtein(seqs[i], cb) if cb else 1e9
            (a_local if da <= db else b_local).append(i)
    return a_local, b_local


def main():
    ap = argparse.ArgumentParser(description="v21 拆分门控可行性 spike(全量模拟+净ΔSR)")
    ap.add_argument('--experiment_dir', required=True)
    ap.add_argument('--gt_refs', required=True)
    ap.add_argument('--gt_tags', required=True)
    ap.add_argument('--read_txt', default=None)
    ap.add_argument('--ref_length', type=int, default=196)
    ap.add_argument('--taus', default='3,5,8,10,15',
                    help='门控阈值扫描(逗号分隔)')
    ap.add_argument('--max_pairwise', type=int, default=80,
                    help='簇内建距离矩阵的最大read数(超出抽样)')
    ap.add_argument('--min_split_size', type=int, default=6,
                    help='簇read数<此值不尝试拆(太小拆了两边都投不出consensus)')
    args = ap.parse_args()
    taus = [int(x) for x in args.taus.split(',')]

    print("=" * 72)
    print("  v21 拆分门控 可行性 spike (全量模拟 / 无监督算法 / GT仅评判)")
    print("=" * 72)
    print(f"  ref_length={args.ref_length}  taus={taus}  min_split_size={args.min_split_size}")

    # ── GT 链路 ───────────────────────────────────────────────────────────────
    read_txt = args.read_txt or find_read_txt(args.experiment_dir)
    print(f"\n[1] reads: {read_txt}")
    reads, clover_labels = load_reads_from_readtxt(read_txt)
    print(f"\n[2] GT tags: {args.gt_tags}")
    seq_to_tag, tag_to_reads = load_gt_tags_file(args.gt_tags)
    print(f"\n[3] GT refs: {args.gt_refs}")
    gt_refs = load_gt_refs_fasta(args.gt_refs)
    ref_len_med = int(np.median([len(s) for s in gt_refs.values()]))
    tag_to_ref = build_tag_to_ref_mapping(tag_to_reads, gt_refs, ref_len=ref_len_med)
    print(f"\n[4] reads -> GT ref id")
    gt_ref_ids = match_reads_to_gt(reads, seq_to_tag, tag_to_ref)

    cl_to_ridx = defaultdict(list)
    for i, c in enumerate(clover_labels):
        cl_to_ridx[int(c)].append(i)
    all_gt = set(int(g) for g in gt_ref_ids[gt_ref_ids >= 0].tolist())
    n_gt = len(all_gt)

    def seqs_of(ridx_list):
        return [reads[ri] for ri in ridx_list]

    # ── 现状 success 集合(baseline) ───────────────────────────────────────────
    # 口径同前: 一个 read-group 的 MV==某GT的ref 即该GT被该group覆盖success。
    # 这里直接以"每个簇 MV -> 命中哪个GT的ref(ED==0)"来统计 success 的GT集合,
    # 与"按主GT"等价但更贴近拆分后评估(拆分会产生新group)。
    def success_gts_of_groups(groups):
        """groups: list of ridx_list. 返回被success命中的GT集合。"""
        succ = set()
        for g_ridx in groups:
            if len(g_ridx) < 1: continue
            cons = mv_consensus(seqs_of(g_ridx), args.ref_length)
            if not cons: continue
            # 该group的多数GT(用于只允许命中自己的主GT, 避免一个group碰巧投出别人的ref)
            cnt = Counter(int(gt_ref_ids[ri]) for ri in g_ridx if gt_ref_ids[ri] >= 0)
            if not cnt: continue
            mg = cnt.most_common(1)[0][0]
            if levenshtein(cons, gt_refs[mg]) == 0:
                succ.add(mg)
        return succ

    print(f"\n[5] 现状 baseline ...")
    base_groups = [cl_to_ridx[c] for c in cl_to_ridx]
    base_succ = success_gts_of_groups(base_groups)
    print(f"    现状 success GT: {len(base_succ):,}/{n_gt:,} (SR={len(base_succ)/n_gt:.4f})")

    # ── 预拆: 对每个簇算一次二分(只算一次, 各τ复用) ───────────────────────────
    print(f"\n[6] 全量簇二分(edit层次聚类) + 记录子簇consensus距离 ...")
    # 对每个簇记录: 原ridx, 二分A/B ridx, consA, consB, dAB
    split_info = {}
    n_clusters = len(cl_to_ridx)
    for k, c in enumerate(cl_to_ridx):
        ridx = cl_to_ridx[c]
        if len(ridx) < args.min_split_size:
            split_info[c] = None  # 太小, 不拆
            continue
        seqs = seqs_of(ridx)
        a_loc, b_loc = split_two(seqs, args.max_pairwise)
        a_ridx = [ridx[i] for i in a_loc]
        b_ridx = [ridx[i] for i in b_loc]
        if len(a_ridx) < 1 or len(b_ridx) < 1:
            split_info[c] = None; continue
        consA = mv_consensus(seqs_of(a_ridx), args.ref_length)
        consB = mv_consensus(seqs_of(b_ridx), args.ref_length)
        dAB = levenshtein(consA, consB) if (consA and consB) else 0
        split_info[c] = (a_ridx, b_ridx, dAB)
        if (k + 1) % 2000 == 0:
            print(f"    ... {k+1}/{n_clusters}")

    n_splittable = sum(1 for v in split_info.values() if v is not None)
    print(f"    可二分的簇(size>={args.min_split_size}): {n_splittable:,}/{n_clusters:,}")

    # ── 扫 τ: 应用门控, 算净ΔSR ────────────────────────────────────────────────
    print(f"\n[7] 扫描门控阈值 τ ...")
    print(f"\n  {'τ':>4} {'拆的簇数':>8} {'新SR':>8} {'ΔSR':>9} {'救回':>6} {'误伤':>6} {'净':>6}")
    print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*9} {'-'*6} {'-'*6} {'-'*6}")

    results = []
    for tau in taus:
        groups = []
        n_split = 0
        for c in cl_to_ridx:
            info = split_info[c]
            if info is None:
                groups.append(cl_to_ridx[c]); continue
            a_ridx, b_ridx, dAB = info
            if dAB >= tau:
                groups.append(a_ridx); groups.append(b_ridx)  # 拆
                n_split += 1
            else:
                groups.append(cl_to_ridx[c])  # 不拆
        new_succ = success_gts_of_groups(groups)
        rescued = len(new_succ - base_succ)   # 拆后新增success
        broken  = len(base_succ - new_succ)   # 拆后丢失(误伤)
        net = rescued - broken
        new_sr = len(new_succ) / n_gt
        dsr = new_sr - len(base_succ) / n_gt
        results.append((tau, n_split, new_sr, dsr, rescued, broken, net))
        print(f"  {tau:>4} {n_split:>8} {new_sr:>8.4f} {dsr:>+9.4f} "
              f"{rescued:>6} {broken:>6} {net:>+6}")

    # ── 结论 ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    best = max(results, key=lambda r: r[6])
    print(f"  最优 τ={best[0]}: 净+{best[6]} success, 新SR={best[2]:.4f} "
          f"(现状 {len(base_succ)/n_gt:.4f})")
    print(f"  救回 {best[4]} / 误伤 {best[5]}")
    print(f"  对照: 上帝视角上界 0.9546 / 无监督可分上界(420) ≈ 0.9408 / FedDNA完美簇 0.9726")
    if best[6] <= 0:
        print(f"  ⚠️ 净增益<=0: 门控误伤≥救回, 当前簇内edit二分方案不可用, 需换门控信号。")
    elif best[6] < 100:
        print(f"  净增益有限(+{best[6]}), 拆分能做但折损大; 看是否值得 vs 复杂度。")
    else:
        print(f"  ✅ 净增益可观(+{best[6]}), 簇内edit二分+consensus门控可作v21迭代引擎。")
    print("=" * 72)


if __name__ == "__main__":
    main()