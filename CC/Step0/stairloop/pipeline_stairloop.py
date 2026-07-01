#!/usr/bin/env python3
"""
pipeline_stairloop.py
=====================
StairLoop CA 数据集专用：reads + BWA-GT → Clover 输入 + 数据分布报告

与 Seq_1D 的关键差异：
  - Seq_1D 的 GT(tag) 内嵌在 output.txt 每行；StairLoop 的 GT 来自 BWA 比对
    (read2ref_q60.tsv: read_id -> ref_id)，需要 join。
  - Seq_1D 28x 高深度，打薄到 30；StairLoop 仅 ~7.5x，关闭打薄（保全部冗余，
    避免饿死本就很小的簇）。
  - 全部 reads 喂 Clover（聚类面对真实噪声）；只用 q60 GT 评 Purity/Coverage。

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/stairloop
    python pipeline_stairloop.py                    # 只做格式转换 + 分布报告
    python pipeline_stairloop.py --run_clover        # 顺便跑 Clover
    python pipeline_stairloop.py --max_reads_per_tag 30   # 如果想打薄

输出:
    out/01_clover_input.txt    Clover 输入(无tag: "序号 序列")
    out/01_gt_tags.txt         GT 标签(对齐 SSI-EC: "ref_id\tseq")
    out/00_distribution.txt    数据分布报告
"""

import os
import sys
import gzip
import time
import random
import argparse
from collections import defaultdict, Counter

# ============================================================
# 配置
# ============================================================
BASE_DIR  = '/mnt/st_data/liangxinyi/code/CC/Step0/stairloop'
FASTQ     = os.path.join(BASE_DIR, 'sequencing reads', 'filter_sequences3.fastq')
GT_TSV    = os.path.join(BASE_DIR, 'read2ref_q60.tsv')   # 第1列 read_id, 第2列 ref_id
REF_FASTA = os.path.join(BASE_DIR, 'test_encode.fasta')
OUT_DIR   = os.path.join(BASE_DIR, 'out')

REF_LEN   = 130
LEN_MIN   = REF_LEN - 8     # 122  (reads 实测 127-133)
LEN_MAX   = REF_LEN + 8     # 138
N_GT_TAGS = 45360           # test_encode.fasta 参考池大小
MAX_READS_PER_TAG = None    # None = 不打薄；给整数则按 FedDNA 策略打薄
MIN_READS_PER_CLUSTER = 2   # Clover 后小簇过滤下限(评测用)
RANDOM_SEED = 42

# 引物(论文给定，trim 用；这里 reads 已 trim，仅作长度核对参考)
PRIMER_F = 'GTAAAACGACGGCCAG'
PRIMER_R = 'GTCATAGCTGTTTCCTG'


def banner(t):
    print(f"\n{'─'*60}\n  {t}\n{'─'*60}")


def open_maybe_gz(path):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r')


# ============================================================
# Step 1: 读 GT 映射  read_id -> ref_id
# ============================================================
def load_gt(gt_path):
    banner("Step 1  读 BWA GT 映射 (read2ref_q60.tsv)")
    read2ref = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                read2ref[parts[0]] = parts[1]
    print(f"  GT 条目: {len(read2ref):,}  (read_id -> ref_id)")
    return read2ref


# ============================================================
# Step 2: 读 fastq，join GT，长度过滤
# ============================================================
def load_reads(fastq_path, read2ref):
    banner("Step 2  读 fastq + join GT + 长度过滤")
    reads = []          # [(read_id, seq, ref_id or None), ...]
    total = n_drop_len = n_drop_n = 0
    len_counter = Counter()

    with open_maybe_gz(fastq_path) as f:
        while True:
            h = f.readline()
            if not h:
                break
            seq = f.readline().strip()
            f.readline()  # +
            f.readline()  # qual
            total += 1
            rid = h[1:].split()[0]  # 去掉 '@'，取第一段

            L = len(seq)
            len_counter[L] += 1
            if 'N' in seq.upper():
                n_drop_n += 1
                continue
            if not (LEN_MIN <= L <= LEN_MAX):
                n_drop_len += 1
                continue

            ref_id = read2ref.get(rid)   # 可能为 None(未比对/低MAPQ)
            reads.append((rid, seq, ref_id))

    n_with_gt = sum(1 for _, _, r in reads if r is not None)
    print(f"  fastq 总 reads:       {total:,}")
    print(f"  含 N 剔除:            {n_drop_n:,}")
    print(f"  长度不合格剔除:       {n_drop_len:,}  (窗口 [{LEN_MIN},{LEN_MAX}])")
    print(f"  保留 reads:           {len(reads):,}")
    print(f"  其中有 GT(q60):       {n_with_gt:,}  ({100*n_with_gt/max(len(reads),1):.2f}%)")
    print(f"  无 GT(模糊/未比对):   {len(reads)-n_with_gt:,}")
    return reads, len_counter, total


# ============================================================
# Step 3: 可选打薄
# ============================================================
def thin(reads, max_per_tag):
    if max_per_tag is None:
        print("\n  打薄: 关闭(保留全部冗余)")
        return reads
    banner(f"Step 3  打薄 (每 ref ≤ {max_per_tag})")
    by_ref = defaultdict(list)
    no_gt = []
    for r in reads:
        (by_ref[r[2]].append(r) if r[2] is not None else no_gt.append(r))
    rng = random.Random(RANDOM_SEED)
    kept, dropped = [], 0
    for ref, lst in by_ref.items():
        if len(lst) > max_per_tag:
            kept.extend(rng.sample(lst, max_per_tag))
            dropped += len(lst) - max_per_tag
        else:
            kept.extend(lst)
    kept.extend(no_gt)
    print(f"  打薄丢弃: {dropped:,}   打薄后: {len(kept):,}")
    return kept


# ============================================================
# Step 4: 写 Clover 输入 + GT 标签文件
# ============================================================
def write_outputs(reads, ref_seqs):
    banner("Step 4  写 Clover 输入 + GT 标签")
    os.makedirs(OUT_DIR, exist_ok=True)
    clover_in = os.path.join(OUT_DIR, '01_clover_input.txt')
    gt_tags   = os.path.join(OUT_DIR, '01_gt_tags.txt')

    # Clover 输入: "序号 序列"  (序号即 Clover 内部 idx，从 1 开始)
    # 同时记录 idx -> ref_id，用于聚类后评测
    idx2ref = {}
    with open(clover_in, 'w') as f:
        for i, (rid, seq, ref) in enumerate(reads, 1):
            f.write(f"{i} {seq}\n")
            idx2ref[i] = ref

    # GT 标签: 只写有 GT 的，格式 "ref_id\tseq"(对齐 dl.load_gt_tags)
    with open(gt_tags, 'w') as f:
        for rid, seq, ref in reads:
            if ref is not None:
                f.write(f"{ref}\t{seq}\n")

    print(f"  Clover 输入: {clover_in}  ({len(reads):,} reads)")
    print(f"  GT 标签:     {gt_tags}")
    return clover_in, idx2ref


# ============================================================
# Step 5: 数据分布报告
# ============================================================
def distribution_report(reads, len_counter, total_raw, ref_seqs):
    banner("Step 5  数据分布报告")
    rpt = os.path.join(OUT_DIR, '00_distribution.txt')

    # 簇大小(按 ref_id 分组，仅 q60 GT)
    cluster_sizes = Counter()
    for _, _, ref in reads:
        if ref is not None:
            cluster_sizes[ref] += 1
    sizes = sorted(cluster_sizes.values())
    n_clusters = len(cluster_sizes)
    n_gt_reads = sum(sizes)

    size_hist = Counter(sizes)
    singletons = size_hist.get(1, 0)
    le3 = sum(c for s, c in size_hist.items() if s <= 3)

    covered = n_clusters
    coverage = covered / N_GT_TAGS
    depth = n_gt_reads / N_GT_TAGS
    # SR 理论上限：去掉无法重建的单 read 簇
    sr_ceiling = (n_clusters - singletons) / N_GT_TAGS

    lines = []
    def p(s=""):
        print(s); lines.append(s)

    p(f"  ── reads 长度分布 ──")
    for L in sorted(len_counter):
        bar = '█' * int(50 * len_counter[L] / max(len_counter.values()))
        p(f"    {L:3d} bp  {len_counter[L]:>7,}  {bar}")
    p()
    p(f"  ── 规模 ──")
    p(f"    原始 reads:          {total_raw:,}")
    p(f"    过滤后 reads:        {len(reads):,}")
    p(f"    参考池 (GT分子):     {N_GT_TAGS:,}")
    p(f"    q60 GT reads:        {n_gt_reads:,}")
    p()
    p(f"  ── 覆盖 & 深度 ──")
    p(f"    被覆盖的 GT 分子:    {covered:,} / {N_GT_TAGS:,}  ({coverage*100:.2f}%)")
    p(f"    dropout(无read分子): {N_GT_TAGS-covered:,}  ({(1-coverage)*100:.2f}%)")
    p(f"    有效深度:            {depth:.2f}x")
    p()
    p(f"  ── 簇大小分布 (q60 GT) ──")
    buckets = [('1 (单read)', lambda s: s == 1),
               ('2-3',        lambda s: 2 <= s <= 3),
               ('4-10',       lambda s: 4 <= s <= 10),
               ('11-30',      lambda s: 11 <= s <= 30),
               ('>30',        lambda s: s > 30)]
    for label, cond in buckets:
        c = sum(v for s, v in size_hist.items() if cond(s))
        p(f"    {label:12s}  {c:>7,}  ({100*c/max(n_clusters,1):5.1f}%)")
    if sizes:
        p(f"\n    max={sizes[-1]}, median={sizes[len(sizes)//2]}, min={sizes[0]}")
    p()
    p(f"  ── SR 上限估计 ──")
    p(f"    单read簇(不可重建):  {singletons:,}  ({100*singletons/max(n_clusters,1):.1f}%)")
    p(f"    size≤3 困难簇:       {le3:,}  ({100*le3/max(n_clusters,1):.1f}%)")
    p(f"    SR 理论上限(去单簇): {sr_ceiling*100:.2f}%")
    p()
    p(f"  ⚠ 难点: 深度{depth:.1f}x偏低 + payload重复motif(簇间区分度低)")

    with open(rpt, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\n  💾 {rpt}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_reads_per_tag', type=int, default=None,
                    help='打薄上限(默认 None=不打薄)')
    ap.add_argument('--run_clover', action='store_true', help='顺便跑 Clover')
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("  🚀  StairLoop CA Pipeline")
    print("=" * 60)
    print(f"  fastq:   {FASTQ}")
    print(f"  GT:      {GT_TSV}")
    print(f"  ref:     {REF_FASTA}  (REF_LEN={REF_LEN}, N_GT={N_GT_TAGS})")

    # 读 ref（仅用于核对，可选）
    ref_seqs = {}
    if os.path.exists(REF_FASTA):
        with open(REF_FASTA) as f:
            rid = None
            for line in f:
                if line.startswith('>'):
                    rid = line[1:].strip()
                elif rid:
                    ref_seqs[rid] = line.strip()
                    rid = None

    read2ref = load_gt(GT_TSV)
    reads, len_counter, total_raw = load_reads(FASTQ, read2ref)
    reads = thin(reads, args.max_reads_per_tag)
    clover_in, idx2ref = write_outputs(reads, ref_seqs)
    distribution_report(reads, len_counter, total_raw, ref_seqs)

    if args.run_clover:
        banner("可选: 运行 Clover")
        print("  请手动执行(参数已按 REF_LEN=130 调整):")
        print(f"    cd /mnt/st_data/liangxinyi/code/CC/Step0/Clover")
        print(f"    python -m clover.main -I {clover_in} \\")
        print(f"      -O {OUT_DIR}/02_clover_result -L {REF_LEN} -P 0 \\")
        print(f"      -D 20 -V 3 -H 3 --no-tag")

    print(f"\n  总耗时: {time.time()-t0:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()