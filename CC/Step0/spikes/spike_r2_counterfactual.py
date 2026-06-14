#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spike_r2_counterfactual.py
==========================
R2 回退根因已定位到两个机制(read层面):
  - 二次拆分: R1拆好的簇在R2又被拆 (→R2新簇, 6576个R1拆分簇read被动)
  - 质心重分配: 簇间挪动 read (12794个)
  encoder无辜(AUROC 0.9997), Clover老簇几乎没动(16个)。

本spike用"反事实标签重构"在labels层面直接验证, 不重放step2:
  场景0  R2原始labels         → baseline SR (应≈0.9164)
  场景1  R2"还原二次拆分"      → 把R2中被二次拆成新簇的read, 还原回它在R1的簇
  场景2  R2"还原质心重分配"    → 把R1老簇间被挪动的read, 还原回R1的簇归属
  场景3  R2"两个都还原"        → 完全回到R1的簇结构(应≈R1 SR 0.9539, 验证逻辑自洽)

哪个场景把SR拉回0.95, 就是主因, 对应解法:
  场景1有效 → 只需"拆分单向化"(已拆簇不再拆)
  场景2有效 → 只需"冻结质心重分配"
  都需要    → 两处都改

只读, 纯labels+MV, 不加载模型。几分钟出。
"""
import argparse, os, sys
import numpy as np
from collections import defaultdict, Counter


def _add_code_root():
    here = os.path.dirname(os.path.abspath(__file__)); d = here
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

from eval_reconstruction import (
    levenshtein, load_reads_from_readtxt, load_gt_tags_file,
    load_gt_refs_fasta, build_tag_to_ref_mapping, match_reads_to_gt, find_read_txt,
)


def mv_consensus(read_seqs, ref_length):
    N = len(read_seqs)
    if N == 0: return ""
    thresh = max(N*0.5, 1)
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


def compute_sr(labels, reads, gt_ref_ids, gt_refs, ref_length, n_gt_total):
    """每GT在以它为主GT的簇里任一簇MV==ref(ED==0)即success。返回success数。"""
    cl_to_didx = defaultdict(list)
    for i, c in enumerate(labels):
        if c >= 0: cl_to_didx[int(c)].append(i)
    cl_majgt = {}
    for c, ridxs in cl_to_didx.items():
        cnt = Counter(int(gt_ref_ids[ri]) for ri in ridxs if gt_ref_ids[ri] >= 0)
        if cnt: cl_majgt[c] = cnt.most_common(1)[0][0]
    majgt_to_clusters = defaultdict(list)
    for c, mg in cl_majgt.items():
        majgt_to_clusters[mg].append(c)
    all_gt = set(int(g) for g in gt_ref_ids[gt_ref_ids >= 0].tolist())
    succ = 0
    for g in all_gt:
        for c in majgt_to_clusters.get(g, []):
            cons = mv_consensus([reads[ri] for ri in cl_to_didx[c]], ref_length)
            if cons and levenshtein(cons, gt_refs[g]) == 0:
                succ += 1; break
    return succ, len(all_gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment_dir', required=True)
    ap.add_argument('--gt_refs', required=True)
    ap.add_argument('--gt_tags', required=True)
    ap.add_argument('--read_txt', default=None)
    ap.add_argument('--r1_labels', required=True)
    ap.add_argument('--r2_labels', required=True)
    ap.add_argument('--ref_length', type=int, default=196)
    ap.add_argument('--split_id_threshold', type=int, default=11648,
                    help='R1拆分新簇起始ID(Clover原簇0..此值-1)')
    ap.add_argument('--r1_max_id', type=int, default=15663,
                    help='R1最大簇数(R2中>=此值=R2二次拆分新簇)')
    args = ap.parse_args()

    print("="*72)
    print("  R2 回退 反事实验证 (labels层面, 不重放step2)")
    print("="*72)

    read_txt = args.read_txt or find_read_txt(args.experiment_dir)
    print(f"\n[1] reads"); reads, _ = load_reads_from_readtxt(read_txt)
    print(f"\n[2] GT tags"); seq_to_tag, tag_to_reads = load_gt_tags_file(args.gt_tags)
    print(f"\n[3] GT refs"); gt_refs = load_gt_refs_fasta(args.gt_refs)
    ref_len_med = int(np.median([len(s) for s in gt_refs.values()]))
    tag_to_ref = build_tag_to_ref_mapping(tag_to_reads, gt_refs, ref_len=ref_len_med)
    print(f"\n[4] reads->GT"); gt_ref_ids = match_reads_to_gt(reads, seq_to_tag, tag_to_ref)

    r1 = np.loadtxt(args.r1_labels, dtype=int)
    r2 = np.loadtxt(args.r2_labels, dtype=int)
    n_gt = len(set(int(g) for g in gt_ref_ids[gt_ref_ids >= 0].tolist()))
    print(f"\n   R1簇 {len(set(r1[r1>=0]))}, R2簇 {len(set(r2[r2>=0]))}, GT分子 {n_gt}")

    SPLIT_TH = args.split_id_threshold
    R1_MAX = args.r1_max_id

    # ── 场景0: R2原始 ──
    print(f"\n[场景0] R2 原始 labels ...")
    s0, _ = compute_sr(r2, reads, gt_ref_ids, gt_refs, args.ref_length, n_gt)
    print(f"   SR = {s0/n_gt:.4f}  ({s0}/{n_gt})")

    # ── 场景1: 还原二次拆分 ──
    # R2中标签>=R1_MAX的read(=R2新拆出的), 还原回它在R1的簇标签。
    cf1 = r2.copy()
    mask_2ndsplit = (r2 >= R1_MAX)
    cf1[mask_2ndsplit] = r1[mask_2ndsplit]
    print(f"\n[场景1] 还原二次拆分 (把 {mask_2ndsplit.sum()} 个R2新簇read 还原回R1簇) ...")
    s1, _ = compute_sr(cf1, reads, gt_ref_ids, gt_refs, args.ref_length, n_gt)
    print(f"   SR = {s1/n_gt:.4f}  ({s1}/{n_gt})   Δ vs 场景0: {(s1-s0)/n_gt*100:+.2f}pt")

    # ── 场景2: 还原质心重分配 ──
    # "簇间重分配" = R1!=R2 且 R2标签<R1_MAX(没进新拆簇)的read, 还原回R1标签。
    cf2 = r2.copy()
    mask_realloc = (r1 != r2) & (r2 < R1_MAX)
    cf2[mask_realloc] = r1[mask_realloc]
    print(f"\n[场景2] 还原质心重分配 (把 {mask_realloc.sum()} 个簇间挪动read 还原回R1) ...")
    s2, _ = compute_sr(cf2, reads, gt_ref_ids, gt_refs, args.ref_length, n_gt)
    print(f"   SR = {s2/n_gt:.4f}  ({s2}/{n_gt})   Δ vs 场景0: {(s2-s0)/n_gt*100:+.2f}pt")

    # ── 场景3: 两个都还原(应回到R1) ──
    print(f"\n[场景3] 两个都还原 (完全回到R1簇结构, 自洽检查) ...")
    s3, _ = compute_sr(r1, reads, gt_ref_ids, gt_refs, args.ref_length, n_gt)
    print(f"   SR = {s3/n_gt:.4f}  ({s3}/{n_gt})   [应≈R1的0.9539]")

    # ── 结论 ──
    print(f"\n{'='*72}")
    print(f"  SR 汇总:")
    print(f"    场景0 R2原始          : {s0/n_gt:.4f}")
    print(f"    场景1 还原二次拆分    : {s1/n_gt:.4f}  ({(s1-s0)/n_gt*100:+.2f}pt)")
    print(f"    场景2 还原质心重分配  : {s2/n_gt:.4f}  ({(s2-s0)/n_gt*100:+.2f}pt)")
    print(f"    场景3 回到R1(自洽)    : {s3/n_gt:.4f}")
    print(f"\n  判定:")
    gain1 = s1 - s0
    gain2 = s2 - s0
    target = s3 - s0  # 完全修复能回收多少
    print(f"    完全修复可回收: {target} success")
    if gain1 >= target * 0.7 and gain1 > gain2 * 1.5:
        print(f"    ➜ 主因是【二次拆分】。解法: 拆分单向化(R1已拆簇R2/R3不再拆)。")
        print(f"      光改这一处即可回收 {gain1}/{target} 的回退。")
    elif gain2 >= target * 0.7 and gain2 > gain1 * 1.5:
        print(f"    ➜ 主因是【质心重分配】。解法: 冻结质心重分配(不动已分配read)。")
        print(f"      光改这一处即可回收 {gain2}/{target} 的回退。")
    else:
        print(f"    ➜ 两个机制共同致回退(场景1+2均贡献)。两处都要改:")
        print(f"      拆分单向化(回收{gain1}) + 冻结质心重分配(回收{gain2})。")
    print("="*72)


if __name__ == "__main__":
    main()