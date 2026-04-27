#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  diagnose_zone3_tracking.py — Zone III 身份追踪 + GT purity 诊断
================================================================================

不需要 embedding 就能回答的两个问题:

  ① Zone III reads 的身份重叠率 (每轮都是同一批 reads 吗?)
     - 分别找出 R1/R2/R3 的 Zone III read_idx 集合
     - 算两两 Jaccard 重叠率, 以及"三轮都是 Zone III"的核心集大小
     - 高重叠 → encoder 稳定, 只是这批 reads 确实"脏"
     - 低重叠 → encoder 判据在漂, 每轮新旧交替

  ② Zone III reads 的 GT purity (它们真的是脏 reads, 还是被误判的好 reads?)
     - 用 tag → gt_id 映射 (借用 diagnose_sr_regression.py 的流程)
     - 对每轮 Zone III reads, 统计它们真实映射到的 GT 分子
     - 如果一个 Zone III reads 簇里同 GT 的 reads 很多 → "其实是好 read 被 Zone III 误杀"
     - 如果 Zone III reads 分布在一堆不同 GT 里 → "确实是脏 reads"

输入 (v17 实验结果目录):
  --experiment_dir .../Experiments/seq_1d/
  自动发现 04_Iterative_Labels/read_state_HHMMSS.pt 里的 zone_ids
  和 refined_labels_HHMMSS.txt 里的 labels

用法:
  cd /mnt/st_data/liangxinyi/code/
  python diagnose_zone3_tracking.py \\
      --experiment_dir /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/ \\
      --gt_refs /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/reads.fasta \\
      --gt_tags /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/seq1d_tags_reads.txt

可选:
  --skip_tag_mapping   跳过 tag→gt_id 映射, 只做身份追踪 (<10 秒)
"""

import os
import sys
import glob
import argparse
import time
from collections import Counter, defaultdict
from typing import List, Dict, Set

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════
# ED 引擎
# ═══════════════════════════════════════════════════════════════
def _resolve_ed_engine():
    try:
        import edlib
        return lambda a, b: edlib.align(a, b, mode="NW", task="distance")['editDistance'], "edlib"
    except ImportError:
        pass
    def _dp(a, b):
        m, n = len(a), len(b)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0]*n
            for j in range(1, n + 1):
                c = 0 if a[i-1] == b[j-1] else 1
                curr[j] = min(curr[j-1]+1, prev[j]+1, prev[j-1]+c)
            prev = curr
        return prev[n]
    return _dp, "dp_fallback"

ED_FUNC, ED_ENGINE = _resolve_ed_engine()


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════
def load_reads_and_clover_labels(read_path):
    """从 read.txt 同时解析 reads 和 Clover cluster id (块索引)"""
    reads = []
    labels = []
    cid = 0
    with open(read_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if '分隔符' in s or s.startswith('====='):
                cid += 1
                continue
            reads.append(s)
            labels.append(cid)
    return reads, labels


def discover_rounds(exp_dir):
    """
    自动发现 04_Iterative_Labels/ 下的 (round_idx, labels_path, state_path, centroids_path)
    按文件 mtime 排序, 假定 R1 < R2 < R3.
    """
    label_dir = os.path.join(exp_dir, "04_Iterative_Labels")
    labels_files = sorted(glob.glob(os.path.join(label_dir, "refined_labels_*.txt")),
                          key=os.path.getmtime)
    states_files = sorted(glob.glob(os.path.join(label_dir, "read_state_*.pt")),
                          key=os.path.getmtime)
    centroids_files = sorted(glob.glob(os.path.join(label_dir, "centroids_*.pt")),
                             key=os.path.getmtime)
    rounds = []
    for r, (l, s, c) in enumerate(zip(labels_files, states_files, centroids_files), 1):
        rounds.append((r, l, s, c))
    return rounds


def load_gt_tags_detailed(tags_path):
    seq_to_tag = {}
    tag_to_reads = defaultdict(list)
    with open(tags_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    tag = int(parts[0])
                    seq = parts[1].strip().upper()
                    seq_to_tag[seq] = tag
                    tag_to_reads[tag].append(seq)
                except ValueError:
                    pass
    return seq_to_tag, tag_to_reads


def load_gt_refs_fasta(path):
    refs = {}
    name = None
    cur = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if s.startswith('>'):
                if name is not None:
                    refs[len(refs)] = ''.join(cur).upper()
                name = s[1:]
                cur = []
            else:
                cur.append(s)
        if name is not None:
            refs[len(refs)] = ''.join(cur).upper()
    return refs


def build_tag_to_ref(tag_to_reads, gt_refs):
    """tag → ref_id (同 diagnose_sr_regression.py 流程)"""
    print(f"   构建 pseudo-references ({len(tag_to_reads)} tags)...")
    tag_pseudo = {}
    for tag, seqs in tag_to_reads.items():
        if not seqs:
            continue
        max_len = max(len(s) for s in seqs)
        cons = []
        for pos in range(max_len):
            cnt = Counter()
            for s in seqs:
                if pos < len(s):
                    cnt[s[pos]] += 1
            cons.append(cnt.most_common(1)[0][0])
        tag_pseudo[tag] = ''.join(cons)

    ref_seq_to_id = {seq: rid for rid, seq in gt_refs.items()}
    tag_to_ref = {}
    exact = 0
    unmatched = []
    for tag, pseudo in tag_pseudo.items():
        rid = ref_seq_to_id.get(pseudo)
        if rid is not None:
            tag_to_ref[tag] = rid
            exact += 1
        else:
            unmatched.append(tag)
    print(f"   精确匹配: {exact}/{len(tag_pseudo)} tags")

    if unmatched:
        print(f"   ED 近似匹配剩余 {len(unmatched)} tags...")
        ref_by_len = defaultdict(list)
        for rid, seq in gt_refs.items():
            ref_by_len[len(seq)].append((rid, seq))
        ed_matched = 0
        for idx, tag in enumerate(unmatched):
            if (idx + 1) % 500 == 0:
                print(f"      进度: {idx+1}/{len(unmatched)}", end='\r')
            pseudo = tag_pseudo[tag]
            plen = len(pseudo)
            best_ed = 999; best_rid = -1
            for dl in range(4):
                for sign in [0, 1, -1]:
                    for rid, ref_seq in ref_by_len.get(plen + sign*dl, []):
                        ed = ED_FUNC(pseudo, ref_seq)
                        if ed < best_ed:
                            best_ed = ed; best_rid = rid
                        if ed == 0: break
                    if best_ed == 0: break
                if best_ed == 0: break
            if best_ed <= 20:
                tag_to_ref[tag] = best_rid
                ed_matched += 1
        print()
        print(f"   ED 近似匹配: {ed_matched}")
    print(f"   总映射: {len(tag_to_ref)}/{len(tag_pseudo)}")
    return tag_to_ref


def match_reads_to_gt(reads, seq_to_tag, tag_to_ref):
    n = len(reads)
    gt_ids = [-1] * n
    matched = 0
    for i, seq in enumerate(reads):
        tag = seq_to_tag.get(seq.upper(), -1)
        if tag >= 0:
            rid = tag_to_ref.get(tag, -1)
            gt_ids[i] = rid
            if rid >= 0:
                matched += 1
    print(f"   Read→Ref 匹配: {matched}/{n}")
    return np.array(gt_ids, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment_dir', required=True)
    ap.add_argument('--gt_refs', required=True)
    ap.add_argument('--gt_tags', required=True)
    ap.add_argument('--skip_tag_mapping', action='store_true',
                    help='跳过 tag→gt_id 映射, 只做身份追踪')
    args = ap.parse_args()

    t_start = time.time()
    print("=" * 70)
    print("  🔬 Zone III 身份追踪 + GT purity 诊断")
    print("=" * 70)

    # ---- 加载 reads + clover labels ----
    exp_dir = args.experiment_dir
    read_path = os.path.join(exp_dir, "03_FedDNA_In", "read.txt")
    print(f"\n📂 加载 reads...")
    reads, clover_labels = load_reads_and_clover_labels(read_path)
    print(f"   Reads: {len(reads):,}")

    # ---- 发现轮次 ----
    rounds = discover_rounds(exp_dir)
    print(f"\n🔍 发现 {len(rounds)} 轮次 (按 mtime 排序)")
    for r, l, s, c in rounds:
        print(f"   R{r}: {os.path.basename(l)}")

    if len(rounds) == 0:
        print("❌ 未找到轮次数据")
        sys.exit(1)

    # ---- 加载每轮 zone_ids ----
    print(f"\n📊 加载每轮 Zone IDs 和 labels...")
    round_data = []
    for rnd, labels_path, state_path, cent_path in rounds:
        labels = np.loadtxt(labels_path, dtype=int)
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        zone_ids = state['zone_ids']
        # zone_ids 可能是 torch tensor 也可能是 numpy
        if isinstance(zone_ids, torch.Tensor):
            zone_ids = zone_ids.numpy()
        cent_data = torch.load(cent_path, map_location='cpu', weights_only=False)
        delta = cent_data.get('delta', None)
        round_data.append({
            'round': rnd,
            'labels': labels,
            'zone_ids': zone_ids,
            'delta': delta,
            'n_reads': len(labels),
        })
        n_z3 = int((zone_ids == 3).sum())
        n_neg1 = int((labels == -1).sum())
        print(f"   R{rnd}: Zone III={n_z3:,}  label=-1 存盘={n_neg1:,}  "
              f"delta={delta:.4f}" if delta is not None else f"   R{rnd}: Zone III={n_z3:,}")

    # ---- 核心 ①: Zone III 身份追踪 ----
    print(f"\n" + "=" * 70)
    print(f"  ①  Zone III 身份追踪 (每轮都是同一批 reads 吗?)")
    print(f"=" * 70)

    z3_sets = []
    for rd in round_data:
        z3_idx = set(int(i) for i in np.where(rd['zone_ids'] == 3)[0])
        z3_sets.append(z3_idx)

    # 两两 Jaccard
    print(f"\n  两两 Jaccard 重叠率:")
    for i in range(len(z3_sets)):
        for j in range(i+1, len(z3_sets)):
            inter = len(z3_sets[i] & z3_sets[j])
            union = len(z3_sets[i] | z3_sets[j])
            jac = inter / union if union > 0 else 0
            print(f"     R{i+1} ∩ R{j+1}: |inter|={inter:,}, |union|={union:,}, "
                  f"Jaccard={jac:.3f}")

    # 核心集 (所有轮都是 Zone III)
    if len(z3_sets) >= 2:
        core = z3_sets[0]
        for s in z3_sets[1:]:
            core = core & s
        print(f"\n  核心集 (所有 {len(z3_sets)} 轮都是 Zone III): {len(core):,} reads")

    # 每轮独有
    for i, s in enumerate(z3_sets):
        others = set()
        for j, t in enumerate(z3_sets):
            if i != j:
                others |= t
        only = s - others
        print(f"  R{i+1} 独有 (其他轮不是 Zone III): {len(only):,}")

    # ---- 核心 ②: Zone III reads 的 GT purity ----
    if args.skip_tag_mapping:
        print(f"\n  ⏭️  跳过 GT purity 分析 (--skip_tag_mapping)")
    else:
        print(f"\n" + "=" * 70)
        print(f"  ②  Zone III reads 的 GT purity (它们是被误判还是真脏?)")
        print(f"=" * 70)

        print(f"\n📂 构建 tag → gt_id 映射...")
        gt_refs = load_gt_refs_fasta(args.gt_refs)
        seq_to_tag, tag_to_reads = load_gt_tags_detailed(args.gt_tags)
        tag_to_ref = build_tag_to_ref(tag_to_reads, gt_refs)
        gt_ids = match_reads_to_gt(reads, seq_to_tag, tag_to_ref)

        # 对每轮 Zone III 分析
        for rd in round_data:
            rnd = rd['round']
            z3_mask = (rd['zone_ids'] == 3)
            z3_positions = np.where(z3_mask)[0]
            z3_gt = gt_ids[z3_positions]

            total = len(z3_positions)
            with_gt = int((z3_gt >= 0).sum())
            no_gt = total - with_gt

            # GT 分布
            gt_hist = Counter()
            for g in z3_gt:
                if g >= 0:
                    gt_hist[int(g)] += 1

            # 统计: 多少个 GT 有 ≥1/2/5/10 条 Zone III reads
            unique_gts = len(gt_hist)
            gt_ge2 = sum(1 for c in gt_hist.values() if c >= 2)
            gt_ge5 = sum(1 for c in gt_hist.values() if c >= 5)
            gt_ge10 = sum(1 for c in gt_hist.values() if c >= 10)

            print(f"\n  ── Round {rnd} Zone III ({total:,} reads) ──")
            print(f"    有 GT 映射:    {with_gt:,}  ({with_gt/total*100:.1f}%)")
            print(f"    无 GT 映射:    {no_gt:,}")
            print(f"    涉及 GT 分子:  {unique_gts:,}")
            print(f"      其中有 ≥2 reads:  {gt_ge2:,}")
            print(f"      其中有 ≥5 reads:  {gt_ge5:,}")
            print(f"      其中有 ≥10 reads: {gt_ge10:,}")

            # 核心判据: Zone III reads 的"聚集度" 
            # 如果很多 reads 聚在少数 GT 上, 说明它们是"GT 已有大簇, 但这几条被判进 Zone III"
            # = 被误判的好 reads (值得救)
            # 反之, 大量 reads 散在不同 GT 上, 说明它们确实是"找不到家"的脏 reads
            if with_gt > 0:
                # 平均每 GT 几条 Zone III reads?
                mean_per_gt = with_gt / unique_gts if unique_gts > 0 else 0
                print(f"    Mean reads/GT: {mean_per_gt:.2f}")

                # Top 10 拥有 Zone III reads 最多的 GT
                top10 = gt_hist.most_common(10)
                print(f"    Top 10 GT (按 Zone III reads 数):")
                for gt, cnt in top10[:5]:
                    print(f"      GT#{gt}: {cnt} reads")

        # ---- 跨轮交集的 GT purity ----
        if len(z3_sets) >= 2:
            core = z3_sets[0]
            for s in z3_sets[1:]:
                core = core & s
            if core:
                print(f"\n  ── 核心集 ({len(core):,} reads, 所有轮都是 Zone III) ──")
                core_idx = np.array(sorted(core))
                core_gt = gt_ids[core_idx]
                core_with_gt = int((core_gt >= 0).sum())
                core_hist = Counter()
                for g in core_gt:
                    if g >= 0:
                        core_hist[int(g)] += 1
                print(f"    有 GT 映射: {core_with_gt:,}")
                print(f"    涉及 GT:    {len(core_hist):,}")
                if core_with_gt > 0:
                    mean_per = core_with_gt / len(core_hist)
                    print(f"    Mean reads/GT: {mean_per:.2f}")

    # ---- 结论 ----
    print(f"\n" + "=" * 70)
    print(f"  🎯 诊断结论 (读我!)")
    print(f"=" * 70)

    # 推断判据
    if len(z3_sets) >= 2:
        jaccards = []
        for i in range(len(z3_sets)):
            for j in range(i+1, len(z3_sets)):
                inter = len(z3_sets[i] & z3_sets[j])
                union = len(z3_sets[i] | z3_sets[j])
                if union > 0:
                    jaccards.append(inter / union)
        mean_jac = np.mean(jaccards) if jaccards else 0

        if mean_jac >= 0.7:
            print(f"  Mean Jaccard = {mean_jac:.2f} ≥ 0.70")
            print(f"  → encoder 判据稳定, Zone III 每轮是同一批 reads")
            print(f"  → 这批 reads 被 encoder 一致地认为【不确定】")
            print(f"  → 但【同一批】本身不说明它们真的【脏】, 看下面 purity 结果")
        elif mean_jac >= 0.4:
            print(f"  Mean Jaccard = {mean_jac:.2f} (中等)")
            print(f"  → encoder 判据部分飘移")
            print(f"  → 核心集是每轮固定的 Zone III, 边缘集每轮变化")
        else:
            print(f"  Mean Jaccard = {mean_jac:.2f} < 0.40")
            print(f"  → encoder 判据大幅飘移, Zone III 每轮换了大半")
            print(f"  → 真凶可能是 encoder 本身在退化")

    if not args.skip_tag_mapping:
        print(f"\n  下一步 (根据 GT purity 结果):")
        print(f"  - 如果 Top GTs 都有很多 Zone III reads (5+ 条/GT) → "
              f"方案 β (降 Zone III 比例)")
        print(f"  - 如果 Zone III reads 均匀分散 (<2 条/GT) → "
              f"保持 v17, 找别的改进点")

    print(f"\n  总耗时: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()