#!/usr/bin/env python3
"""
pipeline_seq1d_gradhc.py
========================
Sequencing_data_first_dimension 专用 Pipeline（GradHC baseline 版 · 同源输入）

  Clover 固化的 seq1d_tags_reads.txt → GradHC 聚类 → 小簇过滤 → 统计 → read.txt + ref.txt

【关键改动 · 同源对比】
  老师已拍板：三个聚类 baseline（Clover / GradHC / NIPS'17）必须消费【同一份打薄后数据】，
  而不是各自独立打薄。打薄属于"数据预处理"，固化一次、三方法共用，这样对比出来的差异
  才纯粹是"聚类算法"的差异，可证明 SSI-EC 对聚类器选择鲁棒。

  因此本版【删除了原 step0 的独立打薄】（原来每 tag≤30 条），改为直接读取
  Clover pipeline 全局随机打薄后固化的 seq1d_tags_reads.txt（tag<TAB>read）。
  --keep_ratio 与 Clover 的 pipeline_seq1d.py 一一对应：
      Clover : CC/Step0/Experiments/seq_1d_p{kr}/seq1d_tags_reads.txt   ← 数据源
      GradHC : CC/Step0/Experiments/seq_1d_gradhc_p{kr}/                 ← 输出，对标 Clover

  GradHC 聚类逻辑 / 参数 (q=8, sd_high=0.40 等) / 输出格式 全部保留不变。

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension
    python pipeline_seq1d_gradhc.py                      # 默认 keep_ratio=0.2，对标 Clover seq_1d_p0.2
    python pipeline_seq1d_gradhc.py --keep_ratio 0.1     # 对标 Clover seq_1d_p0.1
    python pipeline_seq1d_gradhc.py --skip_gradhc        # 跳过 GradHC，复用已有结果
"""

import os
import re
import sys
import glob
import subprocess
import time
import random
from collections import defaultdict, Counter


class _Tee:
    """把 stdout 同时写到屏幕和日志文件。"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ============================================================
# 配置
# ============================================================
BASE_DIR    = '/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension'
GRADHC_DIR  = '/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/GradHC'
EXP_ROOT    = '/mnt/st_data/liangxinyi/code/CC/Step0/Experiments'
OUTPUT_DIR  = None   # 运行时按 keep_ratio 设置
EXPERIMENT_DIR = None  # 运行时按 keep_ratio 设置

REF_LEN     = 196
MIN_READS_PER_CLUSTER = 5   # 聚类后小簇过滤
N_GT_TAGS   = 11826          # design 总数
KEEP_RATIO  = 0.2            # 对标 Clover 的 keep_ratio（仅用于定位同源数据 + 输出目录命名）

# GradHC 参数（保留原版，针对 196bp 长 read 调过；spike 结果支持"防误并优先"，不改）
#   q=8：置换空间 4^8=65536，把 numset 占比降到 0.3%，消除 196bp 上的 MinHash 塌缩。
#   sd_high=0.40：落在"同源 sd p5=0.925 / 异源 consensus sd p99=0.235"的空隙正中，
#                 同源保留 99.9%（不伤召回）、跨 GT 误并降到 0.01%（掐断链式滚雪球）。
#   spike 实测 Seq_1D 簇间极远（q-gram 间隔 195），误并代价大，保留该保护更稳妥。
GRADHC_Q    = 8
GRADHC_K    = 3
GRADHC_M    = 40
GRADHC_L    = 32
GRADHC_DIST = 12
GRADHC_TECH = 'minion_idt'
GRADHC_SD_HIGH = 0.40


def banner(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


# ============================================================
# Step 0: 读取 Clover 固化的同源打薄数据（不再独立打薄）
# ============================================================
def step0_load_shared_thinned(tags_file, keep_ratio):
    """
    直接读 Clover pipeline 固化的 seq1d_tags_reads.txt（tag<TAB>read）。
    这就是全局随机打薄后的同一批 reads —— 与 Clover 输入完全一致（同源对比）。
    不做任何过滤 / 打薄：固化文件已是预处理 + 打薄的最终产物。
    """
    banner("Step 0  读取 Clover 固化的同源打薄数据")

    if not os.path.exists(tags_file):
        raise FileNotFoundError(
            f"找不到 Clover 固化数据: {tags_file}\n"
            f"请先运行 Clover 的 pipeline_seq1d.py --keep_ratio {keep_ratio} 生成同源数据。")

    cleaned = []  # [(tag, read), ...]
    with open(tags_file) as f:
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
            cleaned.append((tag, read))

    n_tags = len(set(t for t, _ in cleaned))
    sizes = Counter(t for t, _ in cleaned)
    size_vals = list(sizes.values())

    print(f"  同源数据源:          {tags_file}")
    print(f"  （= Clover 全局随机打薄后固化的同一批 reads）")
    print()
    print(f"  读入 reads:          {len(cleaned):,}")
    print(f"  GT tags:             {n_tags:,}")
    print(f"  每 tag reads:        avg={sum(size_vals)/len(size_vals):.1f}, "
          f"max={max(size_vals)}, med={sorted(size_vals)[len(size_vals)//2]}, min={min(size_vals)}")
    print()
    print(f"  ⚠ 注意：本脚本不再独立打薄，直接消费 Clover 固化数据，确保三 baseline 同源对比。")

    return cleaned


# ============================================================
# Step 1: 写 GradHC 输入（分块格式，无监督）
# ============================================================
def step1_write_gradhc_input(cleaned, gradhc_input_path, gradhc_tag_path):
    """
    GradHC 输入格式:
        <rep 占位串>
        *****************************
        <read 1>
        <read 2>
        ...
        <空行><空行>
    全部 reads 放进同一块，无监督（rep 用占位串，不泄露 GT）。
    """
    banner("Step 1  写 GradHC 输入文件")

    placeholder_rep = 'A' * REF_LEN

    with open(gradhc_input_path, 'w', newline='\n') as f:
        f.write(placeholder_rep + '\n')
        f.write('*' * 29 + '\n')
        for tag, read in cleaned:
            f.write(read + '\n')
        f.write('\n\n')

    print(f"  GradHC 输入:    {gradhc_input_path}")
    print(f"                  {len(cleaned):,} 条 reads（单块、无监督）")

    read_to_tags = defaultdict(list)
    for tag, read in cleaned:
        read_to_tags[read].append(tag)

    with open(gradhc_tag_path, 'w') as f:
        for tag, read in cleaned:
            f.write(f"{tag}\t{read}\n")

    n_tags = len(set(t for t, _ in cleaned))
    n_unique_reads = len(read_to_tags)
    dup_reads = len(cleaned) - n_unique_reads

    print(f"  tag 映射文件:   {gradhc_tag_path}")
    print(f"  唯一 read 数:   {n_unique_reads:,}  (重复 read: {dup_reads:,})")
    print(f"  💡 GT tags 数: {n_tags:,}")

    return read_to_tags


# ============================================================
# Step 2: 运行 GradHC
# ============================================================
def step2_run_gradhc(gradhc_input_path):
    banner("Step 2  运行 GradHC")

    results_dir = os.path.join(GRADHC_DIR, 'Results')
    os.makedirs(results_dir, exist_ok=True)

    input_base = os.path.basename(gradhc_input_path)
    old_pattern = os.path.join(results_dir, input_base + '_*.clustering_results')
    for old in glob.glob(old_pattern):
        os.remove(old)
        print(f"  🧹 清除旧结果: {os.path.basename(old)}")

    prev_cwd = os.getcwd()
    os.chdir(GRADHC_DIR)
    if GRADHC_DIR not in sys.path:
        sys.path.insert(0, GRADHC_DIR)
    from GradHC_clustering import GradHCBasedCluster

    class GradHCSdHigh(GradHCBasedCluster):
        _SD_HIGH = GRADHC_SD_HIGH
        def clustering_given_chunk(self, chunk_rep, sd_high=None, sd_low=0.28):
            if sd_high is None:
                sd_high = self._SD_HIGH
            return super().clustering_given_chunk(chunk_rep, sd_high=sd_high, sd_low=sd_low)
        def final_clustering(self, sd_high=None, sd_low=0.22, low_work_rate=0.005,
                             high_work_rate=0.03, rounds_before_refresh=8, min_rounds=300):
            if sd_high is None:
                sd_high = self._SD_HIGH
            return super().final_clustering(
                sd_high=sd_high, sd_low=sd_low,
                low_work_rate=low_work_rate, high_work_rate=high_work_rate,
                rounds_before_refresh=rounds_before_refresh, min_rounds=min_rounds)

    print(f"  GradHC: q={GRADHC_Q} k={GRADHC_K} m={GRADHC_M} L={GRADHC_L} dist={GRADHC_DIST}")
    print(f"          sd_high={GRADHC_SD_HIGH} (覆盖默认，掐断链式滚雪球)")
    print(f"  运行中 ...（GradHC 在数十万 read 量级上较慢，请耐心）\n")

    t0 = time.time()
    try:
        cluster = GradHCSdHigh(
            gradhc_input_path,
            q=GRADHC_Q, k=GRADHC_K, m=GRADHC_M, L=GRADHC_L,
            distance_threshold=GRADHC_DIST,
            serial=True, export=True,
        )
        cluster.run()
    finally:
        os.chdir(prev_cwd)
    elapsed = time.time() - t0

    matches = glob.glob(old_pattern)
    if not matches:
        raise FileNotFoundError(f"GradHC 输出文件未找到: {old_pattern}")
    result_path = max(matches, key=os.path.getmtime)

    print(f"\n  ✅ GradHC 完成，耗时 {elapsed:.1f}s")
    print(f"  结果文件: {result_path}")
    return result_path


# ============================================================
# Step 3: 解析 GradHC 输出（分块格式）→ cid → [reads]
# ============================================================
def step3_parse_gradhc(gradhc_output_path):
    banner("Step 3  解析 GradHC 输出")

    clusters = []
    cur_reads = None
    expect_rep = True

    with open(gradhc_output_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if line == '':
                if cur_reads is not None and len(cur_reads) > 0:
                    clusters.append(cur_reads)
                cur_reads = None
                expect_rep = True
                continue
            if line[0] == '*':
                expect_rep = False
                cur_reads = []
                continue
            if expect_rep:
                cur_reads = None
                expect_rep = True
                continue
            else:
                if cur_reads is None:
                    cur_reads = []
                cur_reads.append(line)

    if cur_reads is not None and len(cur_reads) > 0:
        clusters.append(cur_reads)

    cid_to_reads = {cid: reads for cid, reads in enumerate(clusters)}

    sizes = [len(v) for v in cid_to_reads.values()]
    total_reads = sum(sizes)

    print(f"  解析到簇数: {len(cid_to_reads):,}")
    print(f"  归簇 reads: {total_reads:,}")
    if sizes:
        print(f"  max={max(sizes)}, med={sorted(sizes)[len(sizes)//2]}, min={min(sizes)}")

    return cid_to_reads


# ============================================================
# Step 3.5: 过滤小簇
# ============================================================
def step3_5_filter_small_clusters(cid_to_reads, min_reads):
    banner(f"Step 3.5  过滤小簇 (reads < {min_reads})")

    before_n = len(cid_to_reads)
    before_r = sum(len(v) for v in cid_to_reads.values())

    filtered = {cid: reads for cid, reads in cid_to_reads.items()
                if len(reads) >= min_reads}

    after_n = len(filtered)
    after_r = sum(len(v) for v in filtered.values())

    print(f"  过滤前:  {before_n:,} 簇,  {before_r:,} reads")
    print(f"  过滤后:  {after_n:,} 簇,  {after_r:,} reads")
    print(f"  丢弃:    {before_n - after_n:,} 簇 ({before_r - after_r:,} reads)")

    return filtered


# ============================================================
# Step 4: 统计（用 read→tag 回查）
# ============================================================
def step4_statistics(cid_to_reads, read_to_tags, min_reads, stats_path):
    banner("Step 4  聚类统计（过滤后）")

    total_reads = sum(len(v) for v in cid_to_reads.values())
    n_clusters = len(cid_to_reads)
    sizes = sorted([len(v) for v in cid_to_reads.values()], reverse=True)

    pool = {r: list(tags) for r, tags in read_to_tags.items()}

    total_pure = 0
    gt_tags_covered = set()

    for cid, reads in cid_to_reads.items():
        tags = []
        for read in reads:
            cand = pool.get(read)
            if cand:
                tags.append(cand.pop())
        if not tags:
            continue
        tag_counts = Counter(tags)
        majority_tag, majority_count = tag_counts.most_common(1)[0]
        total_pure += majority_count
        gt_tags_covered.add(majority_tag)

    purity = total_pure / max(total_reads, 1)
    coverage = len(gt_tags_covered) / N_GT_TAGS

    buckets = [
        (f'{min_reads} - 10', sum(1 for s in sizes if min_reads <= s <= 10)),
        ('11 - 30',           sum(1 for s in sizes if 11 <= s <= 30)),
        ('31 - 100',          sum(1 for s in sizes if 31 <= s <= 100)),
        ('> 100',             sum(1 for s in sizes if s > 100)),
    ]

    print(f"  聚类簇数:              {n_clusters:,}")
    print(f"  GT tag 数:             {N_GT_TAGS:,}")
    print(f"  reads:                 {total_reads:,}")
    print(f"  min reads/簇:          {min_reads}")
    print()
    print(f"  Purity:                {purity*100:.2f}%")
    print(f"  Coverage:              {coverage*100:.2f}%  ({len(gt_tags_covered)}/{N_GT_TAGS})")
    print()
    print(f"  簇大小分布:")
    for label, count in buckets:
        print(f"    {label:15s}  {count:>7,}")
    if sizes:
        print(f"\n  max={sizes[0]}, median={sizes[len(sizes)//2]}, min={sizes[-1]}")

    with open(stats_path, 'w') as f:
        f.write(f"Seq_1D GradHC 聚类统计（同源打薄 + 小簇过滤）\n{'='*40}\n\n")
        f.write(f"聚类簇数:      {n_clusters:,}\nGT tag 数:     {N_GT_TAGS:,}\n")
        f.write(f"reads:         {total_reads:,}\nmin reads/簇:  {min_reads}\n\n")
        f.write(f"Purity:        {purity*100:.2f}%\nCoverage:      {coverage*100:.2f}%\n\n")
        f.write("簇大小分布:\n")
        for label, count in buckets:
            f.write(f"  {label:15s}  {count:,}\n")
        if sizes:
            f.write(f"\nmax={sizes[0]}, median={sizes[len(sizes)//2]}, min={sizes[-1]}\n")

    print(f"\n  💾 {stats_path}")
    return purity, coverage


# ============================================================
# Step 5+6: Majority Vote → read.txt + ref.txt
# ============================================================
def majority_vote(reads, ref_len):
    vote = [Counter() for _ in range(ref_len)]
    for read in reads:
        for pos in range(min(len(read), ref_len)):
            b = read[pos].upper()
            if b in 'ACGT':
                vote[pos][b] += 1
    result, last = [], 'A'
    for pos in range(ref_len):
        if vote[pos]:
            last = vote[pos].most_common(1)[0][0]
        result.append(last)
    return ''.join(result)


def step56_write_output(cid_to_reads, read_path, ref_path):
    banner("Step 5+6  Majority Vote → read.txt + ref.txt")

    SEPARATOR = "=====分隔符=====\n"
    n_clusters = n_reads = 0

    with open(read_path, 'w') as fr, open(ref_path, 'w') as ff:
        for cid in sorted(cid_to_reads.keys()):
            reads = cid_to_reads[cid]
            if not reads:
                continue
            for read in reads:
                fr.write(read + '\n')
            fr.write(SEPARATOR)
            ff.write(majority_vote(reads, REF_LEN) + '\n')
            n_clusters += 1
            n_reads += len(reads)

    print(f"  ✅ 簇数: {n_clusters:,},  reads: {n_reads:,}")
    print(f"  read.txt: {read_path}")
    print(f"  ref.txt:  {ref_path}")


# ============================================================
# Step 7: 部署到实验目录
# ============================================================
def step7_deploy(gradhc_tag_path, keep_ratio):
    banner("Step 7  部署到实验目录")

    feddna_src = os.path.join(OUTPUT_DIR, '04_FedDNA_In')
    feddna_dst = os.path.join(EXPERIMENT_DIR, '03_FedDNA_In')
    os.makedirs(feddna_dst, exist_ok=True)

    import shutil
    for fname in ['read.txt', 'ref.txt']:
        src = os.path.join(feddna_src, fname)
        dst = os.path.join(feddna_dst, fname)
        shutil.copy2(src, dst)
        print(f"  ✅ {src} → {dst}")

    # GT tags：gradhc_tag_path 已是 "tag\tread"，直接复制
    gt_tags_path = os.path.join(EXPERIMENT_DIR, 'seq1d_tags_reads.txt')
    shutil.copy2(gradhc_tag_path, gt_tags_path)
    print(f"  ✅ GT tags: {gt_tags_path}")

    # GT refs
    refs_fasta = os.path.join(BASE_DIR, 'reads.fasta')
    gt_refs_path = os.path.join(EXPERIMENT_DIR, 'seq1d_refs.txt')
    with open(refs_fasta, 'r') as fin, open(gt_refs_path, 'w') as fout:
        for line in fin:
            if not line.startswith('>'):
                fout.write(line)
    print(f"  ✅ GT refs: {gt_refs_path}")
    print()

    ratio_tag = f"p{keep_ratio}"
    exp_rel = f"CC/Step0/Experiments/seq_1d_gradhc_{ratio_tag}"
    log_name = f"seq1d_gradhc_{ratio_tag}.log"
    print(f"  🚀 运行实验:")
    print(f"     cd /mnt/st_data/liangxinyi/code")
    print(f"     python -m models.main_loop \\")
    print(f"       --experiment_dir {exp_rel}/ \\")
    print(f"       --feddna_checkpoint result/FLDNA_I/I_1214234233/model/epoch1_I.pth \\")
    print(f"       --gt_tags_file {exp_rel}/seq1d_tags_reads.txt \\")
    print(f"       --gt_refs_file {exp_rel}/seq1d_refs.txt \\")
    print(f"       --max_iterations 3 --max_length 201 \\")
    print(f"       --cl_mode ours --ref_length {REF_LEN} --primer_prefix 20 --primer_suffix 20 \\")
    print(f"       --split_tau 5 --split_min_size 6 \\")
    print(f"       2>&1 | tee {exp_rel}/{log_name}")
    print()
    print(f"  📊 评估命令（实验跑完后执行）:")
    print(f"     cd /mnt/st_data/liangxinyi/code/models")
    print(f"     python eval_reconstruction.py \\")
    print(f"       --experiment_dir /mnt/st_data/liangxinyi/code/{exp_rel}/ \\")
    print(f"       --gt_refs /mnt/st_data/liangxinyi/code/{exp_rel}/seq1d_refs.txt \\")
    print(f"       --gt_tags /mnt/st_data/liangxinyi/code/{exp_rel}/seq1d_tags_reads.txt \\")
    print(f"       --out reconstruction_eval_seq1d_gradhc_{ratio_tag}.tsv")


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Seq_1D GradHC Pipeline（同源输入版）')
    parser.add_argument('--skip_gradhc', action='store_true',
                        help='跳过 GradHC，复用已有结果')
    parser.add_argument('--gradhc_result', type=str, default=None,
                        help='--skip_gradhc 时指定已有的 .clustering_results 路径')
    parser.add_argument('--keep_ratio', type=float, default=KEEP_RATIO,
                        help=f'对标 Clover 的 keep_ratio（默认 {KEEP_RATIO}）。'
                             f'决定读哪份 Clover 固化数据 + 输出目录命名。')
    parser.add_argument('--min_reads', type=int, default=MIN_READS_PER_CLUSTER,
                        help=f'聚类后簇最少 reads 数（默认 {MIN_READS_PER_CLUSTER}）')
    args = parser.parse_args()

    global OUTPUT_DIR, EXPERIMENT_DIR
    ratio_tag = f"p{args.keep_ratio}"

    # 数据源：Clover 固化的同源打薄数据
    clover_tags_file = os.path.join(EXP_ROOT, f"seq_1d_{ratio_tag}", "seq1d_tags_reads.txt")

    # 输出：对标 Clover，单独的 gradhc 实验目录
    OUTPUT_DIR = os.path.join(BASE_DIR, f'gradhc_out_{ratio_tag}')
    EXPERIMENT_DIR = os.path.join(EXP_ROOT, f'seq_1d_gradhc_{ratio_tag}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, '04_FedDNA_In'), exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    gradhc_input     = os.path.join(OUTPUT_DIR, '01_gradhc_input.txt')
    gradhc_tag_input = os.path.join(OUTPUT_DIR, '01_gradhc_tag_input.txt')
    stats_path       = os.path.join(OUTPUT_DIR, '03_stats.txt')
    read_path        = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'read.txt')
    ref_path         = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'ref.txt')

    # ── 开启 Tee：全程输出同时写到实验目录 data_detail.txt（与 Clover 一致）──
    detail_path = os.path.join(EXPERIMENT_DIR, 'data_detail.txt')
    _detail_fh = open(detail_path, 'w')
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _detail_fh)

    print()
    print("=" * 60)
    print("  🚀  Seq_1D Pipeline（GradHC baseline · 同源输入）")
    print("=" * 60)
    print()
    print(f"  数据源(同源):     {clover_tags_file}")
    print(f"  GradHC 目录:      {GRADHC_DIR}")
    print(f"  输出目录:         {OUTPUT_DIR}")
    print(f"  实验目录:         {EXPERIMENT_DIR}")
    print(f"  日志文件:         {detail_path}")
    print()
    print(f"  ref_len:          {REF_LEN}bp")
    print(f"  keep_ratio:       {args.keep_ratio}  ← 对标 Clover seq_1d_{ratio_tag}")
    print(f"  min_reads/簇:     {args.min_reads}")
    print(f"  GradHC:           q={GRADHC_Q} k={GRADHC_K} m={GRADHC_M} L={GRADHC_L} "
          f"dist={GRADHC_DIST} sd_high={GRADHC_SD_HIGH}")

    t_start = time.time()

    # Step 0: 读 Clover 固化的同源打薄数据（不独立打薄）
    cleaned = step0_load_shared_thinned(clover_tags_file, args.keep_ratio)

    # Step 1: 写 GradHC 输入 + read→tag 映射
    read_to_tags = step1_write_gradhc_input(cleaned, gradhc_input, gradhc_tag_input)
    del cleaned

    # Step 2: GradHC
    if not args.skip_gradhc:
        gradhc_result = step2_run_gradhc(gradhc_input)
    else:
        banner("Step 2  跳过 GradHC")
        if args.gradhc_result:
            gradhc_result = args.gradhc_result
        else:
            pattern = os.path.join(GRADHC_DIR, 'Results',
                                   os.path.basename(gradhc_input) + '_*.clustering_results')
            matches = glob.glob(pattern)
            if not matches:
                raise FileNotFoundError(f"未找到已有结果: {pattern}（用 --gradhc_result 指定）")
            gradhc_result = max(matches, key=os.path.getmtime)
        print(f"  使用: {gradhc_result}")
        if not os.path.exists(gradhc_result):
            raise FileNotFoundError(f"不存在: {gradhc_result}")

    # Step 3: 解析
    cid_to_reads = step3_parse_gradhc(gradhc_result)

    # Step 3.5: 过滤小簇
    cid_to_reads = step3_5_filter_small_clusters(cid_to_reads, args.min_reads)

    # Step 4: 统计
    purity, coverage = step4_statistics(cid_to_reads, read_to_tags, args.min_reads, stats_path)

    # Step 5+6: 写输出
    step56_write_output(cid_to_reads, read_path, ref_path)

    # Step 7: 部署
    step7_deploy(gradhc_tag_input, args.keep_ratio)

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print("  🎉  Pipeline 完成!")
    print("=" * 60)
    print()
    print(f"  总耗时:         {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  数据源:         Clover 固化同源数据 (keep_ratio={args.keep_ratio})")
    print(f"  Purity:         {purity*100:.2f}%")
    print(f"  Coverage:       {coverage*100:.2f}%")
    print()

    # ── 关闭 Tee ──
    sys.stdout = _orig_stdout
    _detail_fh.close()


if __name__ == '__main__':
    main()