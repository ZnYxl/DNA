#!/usr/bin/env python3
"""
gradhc_giant_cluster_diagnose.py
================================
GradHC 巨簇诊断 spike（只读，不改任何数据）。

背景
----
q=8 + sd_high=0.40 版 GradHC 在 Seq_1D p0.2 上 Purity 仅 84%，且出现 max=33398 的巨簇。
本 spike 拆解这个巨簇（及前 N 大簇）的 GT tag 成分，定位病根：

  问题1：巨簇里混了多少个不同 GT？
  问题2：是「一个 GT 占主导 + 少量污染」（轻度），还是「几百个 GT 均匀混杂」（滚雪球误并）？
  问题3：误并集中在少数巨簇，还是普遍现象？

判读
----
  • 若巨簇 = 1 个主导 GT（>80%）+ 长尾污染  → 轻度，调 sd 阈值可能有救
  • 若巨簇 = 几百个 GT 每个占百分之几     → 滚雪球误并，问题在 chunk_partitioning 阶段
                                            （该阶段只看公共子串、完全不看 sd_high，调 sd 无用）

数据源（均为 84% 版已落盘产物，只读）
  • read.txt   : 按簇分隔的 reads（=====分隔符===== 切簇），各簇即 GradHC 聚类结果
  • tags_file  : tag<TAB>read 同源 GT 映射

用法
----
    python gradhc_giant_cluster_diagnose.py
    python gradhc_giant_cluster_diagnose.py --top_n 20 --read_file /path/read.txt
"""

import os
import sys
import argparse
from collections import Counter, defaultdict

DEFAULT_READ_FILE = "/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/gradhc_out_p0.2/04_FedDNA_In/read.txt"
DEFAULT_TAGS_FILE = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d_gradhc_p0.2/seq1d_tags_reads.txt"
SEPARATOR = "=====分隔符====="


def load_read_to_tags(tags_file):
    """读 tag<TAB>read，建 read→[tags] 映射（一条 read 可能对应多个 tag 实例）。"""
    read_to_tags = defaultdict(list)
    n = 0
    with open(tags_file) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            tag, read = parts
            read_to_tags[read].append(tag)
            n += 1
    print(f"  [tags] 读入 {n:,} 条 (tag,read)，唯一 read {len(read_to_tags):,}")
    return read_to_tags


def parse_clusters(read_file):
    """读 read.txt，按 =====分隔符===== 切簇，返回 [[read,...], ...]。"""
    clusters = []
    cur = []
    with open(read_file) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == SEPARATOR:
                if cur:
                    clusters.append(cur)
                cur = []
            elif line:
                cur.append(line)
    if cur:
        clusters.append(cur)
    return clusters


def cluster_gt_composition(cluster_reads, pool):
    """用可消耗的 read→tag 队列回查每条 read 的 GT tag。返回 Counter(tag→count)。"""
    tags = []
    for read in cluster_reads:
        cand = pool.get(read)
        if cand:
            tags.append(cand.pop())
    return Counter(tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read_file", default=DEFAULT_READ_FILE)
    ap.add_argument("--tags_file", default=DEFAULT_TAGS_FILE)
    ap.add_argument("--top_n", type=int, default=10, help="诊断前 N 大簇")
    args = ap.parse_args()

    print("=" * 64)
    print("  GradHC 巨簇诊断 spike（只读）")
    print("=" * 64)
    print(f"  read_file: {args.read_file}")
    print(f"  tags_file: {args.tags_file}\n")

    for p in (args.read_file, args.tags_file):
        if not os.path.exists(p):
            print(f"✗ 文件不存在: {p}")
            sys.exit(1)

    read_to_tags = load_read_to_tags(args.tags_file)
    clusters = parse_clusters(args.read_file)
    print(f"  [read] 解析到 {len(clusters):,} 个簇\n")

    sizes = sorted([len(c) for c in clusters], reverse=True)
    print(f"  簇大小: max={sizes[0]:,}  median={sizes[len(sizes)//2]}  min={sizes[-1]}")
    print(f"  >1000 的簇: {sum(1 for s in sizes if s > 1000)}  "
          f">100 的簇: {sum(1 for s in sizes if s > 100)}\n")

    # 按大小排序，诊断前 N 大
    clusters_sorted = sorted(clusters, key=len, reverse=True)
    pool = {r: list(tags) for r, tags in read_to_tags.items()}

    print("─" * 64)
    print(f"  前 {args.top_n} 大簇的 GT 成分拆解")
    print("─" * 64)
    for rank, c in enumerate(clusters_sorted[:args.top_n], start=1):
        comp = cluster_gt_composition(c, pool)
        size = len(c)
        n_gt = len(comp)
        matched = sum(comp.values())
        if matched == 0:
            print(f"  #{rank}  size={size:,}  [无法回查 GT]")
            continue
        top_tag, top_cnt = comp.most_common(1)[0]
        top_frac = top_cnt / matched * 100
        # 判读
        if n_gt == 1:
            verdict = "纯簇 ✓"
        elif top_frac >= 80:
            verdict = f"主导GT占{top_frac:.0f}%，轻度污染"
        elif top_frac >= 30:
            verdict = f"主导GT仅{top_frac:.0f}%，中度混杂"
        else:
            verdict = f"主导GT仅{top_frac:.0f}%，{n_gt}个GT滚雪球误并 ⚠"
        print(f"  #{rank}  size={size:,}  含{n_gt:,}个不同GT  主导GT={top_cnt:,}({top_frac:.1f}%)  → {verdict}")
        # 巨簇额外展示前5大GT分布
        if rank <= 3 and n_gt > 1:
            top5 = comp.most_common(5)
            dist = "  ".join(f"{cnt}({cnt/matched*100:.1f}%)" for _, cnt in top5)
            print(f"        前5大GT: {dist}")

    # ── 全局误并量化 ──
    print(f"\n{'─'*64}")
    print(f"  全局误并量化")
    print(f"{'─'*64}")
    pool2 = {r: list(tags) for r, tags in read_to_tags.items()}
    total_reads = 0
    total_pure = 0
    giant_misassign = 0   # >1000 巨簇里非主导GT的reads(误并量)
    for c in clusters_sorted:
        comp = cluster_gt_composition(c, pool2)
        matched = sum(comp.values())
        if matched == 0:
            continue
        total_reads += matched
        top_cnt = comp.most_common(1)[0][1]
        total_pure += top_cnt
        if len(c) > 1000:
            giant_misassign += (matched - top_cnt)
    print(f"  总匹配 reads:        {total_reads:,}")
    print(f"  纯度(主导GT占比):    {total_pure/total_reads*100:.2f}%")
    print(f"  >1000巨簇内误并reads: {giant_misassign:,} "
          f"({giant_misassign/total_reads*100:.2f}% of all)")
    print(f"\n  → 若巨簇误并占比高 → 病根是少数滚雪球巨簇，应聚焦切断 chunk 阶段的串联")
    print("=" * 64)


if __name__ == "__main__":
    main()