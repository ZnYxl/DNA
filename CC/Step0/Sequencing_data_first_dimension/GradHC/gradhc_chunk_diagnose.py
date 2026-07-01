#!/usr/bin/env python3
"""
gradhc_chunk_diagnose.py  —  只读 spike(不改源码 / 不写 production 文件 / 不跑后续聚类)
================================================================================
目的:确认 84% 版的 33K 巨簇,是否在 chunk_partitioning【第一阶段】就已经成型。

设计要点(均来自源码精读):
  1. chunk_partitioning() 不返回值,结果落在 self.chunks(chunk_rep -> [read indices])
     和并查集 self.chunk_parent。子类只跑到 chunk 阶段就停,绝不进 clustering_in_chunks。
  2. ⚠ GradHC 内部 all_reads 的 index 与输入文件行序【无关】:
     process_input() 里先 C_reps.sort(by read 串) 又 random.shuffle(all_reads)。
     所以 chunk 里的整数 index 只能通过 all_reads[idx](=read 字符串)映回 tag。
     => 严格遵守 "sequence as key,never line number"。
  3. chunk 阶段用 cmn_substr(随机 w-mer 公共子串),完全不碰 q / numsets。
     因此本诊断结论与 q=8 还是 q=6 无关——chunk 划分对 q 不敏感。

用法(在 GRADHC_DIR 下跑,因为要 import GradHC_clustering):
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/GradHC
    python /path/to/gradhc_chunk_diagnose.py \
        --gradhc_input  /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/gradhc_out_p0.2/01_gradhc_input.txt \
        --tag_file      /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d_p0.2/seq1d_tags_reads.txt \
        --q 8 --top_n 10

注意:--gradhc_input 用 pipeline 已经写好的 01_gradhc_input.txt(GradHC 输入格式),
      --tag_file 用 Clover 固化的 tag<TAB>read(建 read->tag 映射,sequence as key)。
      q 在 chunk 阶段无影响,这里传 q 只是为了构造对象(pre_step 会算 numsets,
      但我们不用它;给 q=8 是为了和 production 一致、避免 pre_step 内存异常)。
"""

import argparse
import os
import sys
import random
from collections import Counter, defaultdict


def build_read_to_tag(tag_file):
    """sequence as key:read 串 -> 主 tag(众数)。重复 read 共享时取出现最多的 tag。"""
    read_tags = defaultdict(Counter)
    n_lines = 0
    with open(tag_file) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            tag, read = parts
            read_tags[read][tag] += 1
            n_lines += 1
    read_to_tag = {r: c.most_common(1)[0][0] for r, c in read_tags.items()}
    # 同时记录有多少 read 串本身就横跨多个 tag(重复 read 的 GT 歧义,作为噪声下限参考)
    ambiguous = sum(1 for c in read_tags.values() if len(c) > 1)
    return read_to_tag, n_lines, len(read_to_tag), ambiguous


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gradhc_input', required=True, help='pipeline 写好的 GradHC 输入(01_gradhc_input.txt)')
    ap.add_argument('--tag_file', required=True, help='Clover 固化 tag<TAB>read(建 read->tag 映射)')
    ap.add_argument('--q', type=int, default=8)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--m', type=int, default=40)
    ap.add_argument('--L', type=int, default=32)
    ap.add_argument('--dist', type=int, default=12)
    ap.add_argument('--top_n', type=int, default=10, help='拆解前 N 大 chunk 的 GT 成分')
    ap.add_argument('--giant', type=int, default=1000, help='巨 chunk 阈值(size > giant)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)  # GradHC 内部多处用 random;固定种子让 spike 可复现

    # ---- import GradHC(必须在 GRADHC_DIR 下,WORKING_DIR_ALGORITHMS 在 import 时固定)----
    try:
        from GradHC_clustering import GradHCBasedCluster
    except ImportError:
        sys.stderr.write(
            "❌ 无法 import GradHC_clustering。请在 GRADHC_DIR 下运行本脚本:\n"
            "   cd .../Sequencing_data_first_dimension/GradHC && python <此脚本>\n")
        sys.exit(1)

    # ---- 子类:只跑到 chunk 阶段就停,不进 clustering_in_chunks / final_clustering ----
    class ChunkOnlyDiagnose(GradHCBasedCluster):
        def run_until_chunks(self):
            self.pre_step()
            self.chunk_partitioning()
            # 故意不调用 clustering_in_chunks() / final_clustering() / export_file()

    print("=" * 70)
    print("  GradHC chunk_partitioning 只读诊断 spike")
    print("=" * 70)
    print(f"  gradhc_input : {args.gradhc_input}")
    print(f"  tag_file     : {args.tag_file}")
    print(f"  q={args.q}(chunk 阶段不受 q 影响,仅用于构造对象)")
    print()

    # ---- read 串 -> tag 映射(sequence as key)----
    read_to_tag, n_tag_lines, n_unique_reads, ambiguous = build_read_to_tag(args.tag_file)
    print(f"  [映射] tag 文件行数        : {n_tag_lines:,}")
    print(f"  [映射] 唯一 read 串        : {n_unique_reads:,}")
    print(f"  [映射] 横跨多 tag 的 read 串: {ambiguous:,} "
          f"({ambiguous / max(n_unique_reads,1) * 100:.3f}%  ← GT 歧义下限,纯度上限受此影响)")
    print()

    # ---- 跑到 chunk 阶段 ----
    cluster = ChunkOnlyDiagnose(
        args.gradhc_input,
        q=args.q, k=args.k, m=args.m, L=args.L,
        distance_threshold=args.dist,
        serial=True, export=False,
    )
    cluster.run_until_chunks()

    # ---- 取 chunk 划分:self.chunks[rep] = [read indices],空的跳过 ----
    chunks = [(rep, reads) for rep, reads in enumerate(cluster.chunks) if len(reads) > 0]
    sizes = sorted((len(r) for _, r in chunks), reverse=True)
    total_reads = sum(sizes)
    n_chunks = len(chunks)

    print()
    print("=" * 70)
    print("  [结果 1] chunk 大小分布(chunk 阶段产出,未经任何 sd 聚类)")
    print("=" * 70)
    print(f"  chunk 数        : {n_chunks:,}")
    print(f"  归入 chunk reads: {total_reads:,}")
    if sizes:
        print(f"  max={sizes[0]:,}  median={sizes[len(sizes)//2]:,}  min={sizes[-1]:,}")
    buckets = [
        ('size == 1     ', sum(1 for s in sizes if s == 1)),
        ('2 - 10        ', sum(1 for s in sizes if 2 <= s <= 10)),
        ('11 - 30       ', sum(1 for s in sizes if 11 <= s <= 30)),
        ('31 - 100      ', sum(1 for s in sizes if 31 <= s <= 100)),
        ('101 - 1000    ', sum(1 for s in sizes if 101 <= s <= 1000)),
        (f'> {args.giant}      ', sum(1 for s in sizes if s > args.giant)),
    ]
    for label, cnt in buckets:
        print(f"    {label}: {cnt:>8,}")

    # ---- GT 成分拆解:read index -> all_reads[idx](read 串)-> tag ----
    def chunk_gt_profile(read_indices):
        tag_counter = Counter()
        unmapped = 0
        for idx in read_indices:
            read = cluster.all_reads[idx]          # ★ sequence as key
            tag = read_to_tag.get(read)
            if tag is None:
                unmapped += 1
            else:
                tag_counter[tag] += 1
        return tag_counter, unmapped

    chunks_by_size = sorted(chunks, key=lambda x: len(x[1]), reverse=True)

    print()
    print("=" * 70)
    print(f"  [结果 2] 前 {args.top_n} 大 chunk 的 GT 成分(滚雪球判定)")
    print("=" * 70)
    print("  判读:主导 GT 占比低 + 含大量不同 GT 且分布均匀 = 链式滚雪球误并")
    print("       主导 GT 占比高(>90%)= chunk 干净,不是病灶")
    print()
    for rank, (rep, reads) in enumerate(chunks_by_size[:args.top_n], 1):
        tag_counter, unmapped = chunk_gt_profile(reads)
        size = len(reads)
        n_distinct_gt = len(tag_counter)
        if tag_counter:
            top5 = tag_counter.most_common(5)
            dom_tag, dom_cnt = top5[0]
            dom_frac = dom_cnt / size
            top5_str = "  ".join(f"{t}({c/size*100:.1f}%)" for t, c in top5)
        else:
            dom_frac = 0.0
            top5_str = "(全部 unmapped)"
        flag = " ⚠滚雪球" if (n_distinct_gt >= 20 and dom_frac < 0.30) else ""
        print(f"  #{rank:<2} size={size:>7,}  含{n_distinct_gt:>5}个GT  "
              f"主导={dom_frac*100:5.1f}%  unmapped={unmapped}{flag}")
        print(f"       top5: {top5_str}")

    # ---- 全局 chunk 纯度(每个 chunk 取主导 GT 的 reads 数之和 / 总 reads)----
    print()
    print("=" * 70)
    print("  [结果 3] 全局 chunk 纯度(chunk 阶段天花板,后续聚类无法超过它)")
    print("=" * 70)
    total_pure = 0
    total_mapped = 0
    giant_misjoin_reads = 0   # 巨 chunk 内非主导 GT 的 reads(误并量)
    giant_count = 0
    for rep, reads in chunks:
        tag_counter, _ = chunk_gt_profile(reads)
        mapped = sum(tag_counter.values())
        if mapped == 0:
            continue
        dom_cnt = tag_counter.most_common(1)[0][1]
        total_pure += dom_cnt
        total_mapped += mapped
        if len(reads) > args.giant:
            giant_count += 1
            giant_misjoin_reads += (mapped - dom_cnt)
    chunk_purity = total_pure / max(total_mapped, 1)
    print(f"  全局 chunk 纯度     : {chunk_purity*100:.2f}%  (基于 mapped reads={total_mapped:,})")
    print(f"  巨 chunk 数(>{args.giant})  : {giant_count}")
    print(f"  巨 chunk 内误并 reads: {giant_misjoin_reads:,} "
          f"({giant_misjoin_reads / max(total_mapped,1) * 100:.2f}% of mapped)")
    print()
    print("=" * 70)
    print("  [诊断结论 · 人工判读]")
    print("=" * 70)
    print("  ① 若 chunk 纯度已 ≈ 84%(接近最终 Purity)→ 巨簇在第一阶段成型,")
    print("     病根确认在 chunk_partitioning,修复方向 = w/t/min_work,与 sd_high 无关。")
    print("  ② 若 chunk 纯度明显 > 84%(如 >95%)→ 巨簇是阶段 2/3 造成,需回头查 sd 路径。")
    print("  ③ 若 chunk 阶段已有 >1000 巨 chunk 且滚雪球 → 直接锁定 chunk 公共子串串联。")
    print("=" * 70)


if __name__ == '__main__':
    main()