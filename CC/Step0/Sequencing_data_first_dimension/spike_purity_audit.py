#!/usr/bin/env python3
"""
spike_purity_audit.py
=====================
核对 GradHC purity 到底对不对。测三件事:
  1. 撞串里有多少是「跨 tag」的（决定字典法是否失真）
  2. 字典法 (seq→单tag) 的 purity   —— eval_gradhc_metrics.py 用的口径
  3. 行序对齐法 (read.txt 行 ↔ GT 行) 的 purity —— 撞串-鲁棒口径
两个 purity 若一致 → 74% 可信；若差很多 → 用行序法。

前提：read.txt 的 reads 顺序 与 seq1d_tags_reads.txt 的行顺序一一对应
      （因为 pipeline 写 read.txt 和写 tag 文件都来自同一个 cleaned 列表的同序遍历）。
本 spike 会先验证这个前提是否成立，不成立则只信字典法的诊断部分。

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d_gradhc
    python spike_purity_audit.py
"""
import os
from collections import Counter, defaultdict

EXP = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d_gradhc"
READ_TXT = os.path.join(EXP, "03_FedDNA_In", "read.txt")
GT_TAGS  = os.path.join(EXP, "seq1d_tags_reads.txt")

# ---- 1. 读 read.txt：reads + GradHC 簇标签（=====分隔符） ----
reads, pred = [], []
cid = 0
with open(READ_TXT) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("====="):
            cid += 1
        else:
            reads.append(line.upper())
            pred.append(cid)
n_pred_clusters = cid
print(f"read.txt: {len(reads):,} reads, {n_pred_clusters:,} GradHC 簇")

# ---- 2. 读 GT tags：行序列表 + seq→tag 字典 ----
gt_tag_by_line = []      # 按文件行序的 tag
gt_seq_by_line = []
seq_to_tag_dict = {}     # 字典法（后写覆盖先写）
seq_to_tags_all = defaultdict(set)  # 每个 seq 对应的所有 tag（测跨tag撞串）
with open(GT_TAGS) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            try:
                tag = int(parts[0]); seq = parts[1].strip().upper()
            except ValueError:
                continue
            gt_tag_by_line.append(tag)
            gt_seq_by_line.append(seq)
            seq_to_tag_dict[seq] = tag
            seq_to_tags_all[seq].add(tag)
print(f"GT tags: {len(gt_tag_by_line):,} 行, {len(seq_to_tag_dict):,} 唯一序列")

# ---- 3. 跨 tag 撞串分析 ----
cross_tag_seqs = sum(1 for s, tags in seq_to_tags_all.items() if len(tags) > 1)
cross_tag_reads = sum(c for s, c in Counter(gt_seq_by_line).items()
                      if len(seq_to_tags_all[s]) > 1)
print(f"\n===== 撞串分析 =====")
print(f"唯一序列:        {len(seq_to_tags_all):,}")
print(f"跨tag序列:       {cross_tag_seqs:,}  ({cross_tag_seqs/len(seq_to_tags_all)*100:.1f}% 的唯一序列对应多个GT分子)")
print(f"跨tag read实例:  {cross_tag_reads:,}  ({cross_tag_reads/len(gt_seq_by_line)*100:.1f}% 的read受字典覆盖影响)")

# ---- 4. 验证行序对齐前提：read.txt 的 reads 是否是 GT 行序的一个排列? ----
# GradHC 重排了 reads，所以 read.txt 行序 ≠ GT 行序。
# 但若两者是同一 multiset，可用「seq→tag列表」消耗式对齐（撞串-鲁棒）。
read_multiset = Counter(reads)
gt_multiset = Counter(gt_seq_by_line)
same_multiset = (read_multiset == gt_multiset)
print(f"\nread.txt 与 GT 是否同一序列multiset: {same_multiset}")
if not same_multiset:
    only_read = sum((read_multiset - gt_multiset).values())
    only_gt = sum((gt_multiset - read_multiset).values())
    print(f"  仅在read.txt: {only_read:,}, 仅在GT: {only_gt:,}（差异越小越好）")

# ---- 5. 字典法 purity（eval_gradhc 口径）----
def purity_dict():
    c2g = defaultdict(list)
    matched = 0
    for seq, p in zip(reads, pred):
        g = seq_to_tag_dict.get(seq)
        if g is not None:
            c2g[p].append(g); matched += 1
    correct = sum(Counter(gs).most_common(1)[0][1] for gs in c2g.values())
    return correct / max(matched, 1), matched, len(c2g)

# ---- 6. 消耗式对齐 purity（撞串-鲁棒）----
def purity_consume():
    # 用 seq→tag 的「可消耗队列」，同序列的多个tag按出现次数分配
    pool = defaultdict(list)
    for seq, tag in zip(gt_seq_by_line, gt_tag_by_line):
        pool[seq].append(tag)
    # 注意：不同分配顺序结果可能不同，这里取「该seq的众数tag」做稳健估计
    c2g = defaultdict(list)
    matched = 0
    for seq, p in zip(reads, pred):
        tags = pool.get(seq)
        if tags:
            # 用众数tag（撞串里最可能的归属），不pop，避免顺序依赖
            maj_tag = Counter(tags).most_common(1)[0][0]
            c2g[p].append(maj_tag); matched += 1
    correct = sum(Counter(gs).most_common(1)[0][1] for gs in c2g.values())
    return correct / max(matched, 1), matched, len(c2g)

pd, md, nd = purity_dict()
pc, mc, nc = purity_consume()
print(f"\n===== Purity 对比 =====")
print(f"字典法 (eval_gradhc 口径):  {pd*100:.2f}%   匹配 {md:,}/{len(reads):,}, 簇 {nd:,}")
print(f"众数法 (撞串-鲁棒):         {pc*100:.2f}%   匹配 {mc:,}/{len(reads):,}, 簇 {nc:,}")
print(f"差异:                       {abs(pd-pc)*100:.2f} 个百分点")
print(f"\n判读:")
print(f"  · 差异 <1pp → 撞串无害,74% 口径可信,直接用 eval_gradhc_metrics.py")
print(f"  · 差异 >2pp → 字典覆盖失真,需改评估脚本用众数法对齐")