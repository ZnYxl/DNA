#!/usr/bin/env python3
"""
pipeline_seq1d_gradhc.py
========================
Sequencing_data_first_dimension 专用 Pipeline（GradHC baseline 版）

  output.txt → 预处理 → 打薄(每tag最多N条) → GradHC聚类 → 小簇过滤 → 统计 → read.txt + ref.txt

与 Clover 版 (pipeline_seq1d_thin.py) 的差异（仅聚类器替换，其余完全一致）：
  - Step 1：GradHC 输入是「分块格式」(rep + ***** + reads + 双空行)，且必须无监督，
            故每块 rep 用占位串，不泄露 GT。
  - Step 2：调用 GradHCBasedCluster(...).run()，结果落在 GradHC/Results/*.clustering_results
  - Step 3：按分块格式解析，用 read 字符串回查 tag（GradHC 内部 shuffle + 按序列索引，没有行号 idx）

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension
    python pipeline_seq1d_gradhc.py                          # 完整运行（默认每tag最多30条）
    python pipeline_seq1d_gradhc.py --max_reads_per_tag 15   # 更激进打薄
    python pipeline_seq1d_gradhc.py --skip_gradhc            # 跳过 GradHC，复用已有结果

打薄策略（与 FedDNA / Clover 版完全一致）:
    对每个 BWA tag，随机采样最多 max_reads_per_tag 条 reads（FedDNA: 5-30），打薄在聚类前执行。
"""

import os
import re
import sys
import glob
import subprocess
import time
import random
from collections import defaultdict, Counter

# ============================================================
# 配置
# ============================================================
BASE_DIR    = '/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension'
GRADHC_DIR  = '/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/GradHC'          # ← GradHC 仓库根目录（含 GradHC_clustering.py）
OUTPUT_DIR  = os.path.join(BASE_DIR, 'gradhc_out')
EXPERIMENT_DIR = '/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d_gradhc'

REF_LEN     = 196
LEN_MIN     = REF_LEN - 5   # 191
LEN_MAX     = REF_LEN + 5   # 201
MIN_READS_PER_CLUSTER = 5   # 聚类后小簇过滤
N_GT_TAGS   = 11826          # design 总数
MAX_READS_PER_TAG = 30       # 每个 tag 最多保留的 reads 数（FedDNA: 5-30）
RANDOM_SEED = 42

# GradHC 参数
# 注意：默认 q=6 (置换空间 4^6=4096) 在 196bp 长 read 上会导致 MinHash 碰撞塌缩
#       (numset 均值 ~173，占置换空间 4.2%，远超 5% 警戒线 → 海量异源 read 进同桶 → 巨簇)。
#       q=8 (置换空间 65536) 将占比降到 0.3%，消除塌缩，同源 sd 仍 0.99、异源 p95 仅 0.19。
#       这是为适配 196bp read 长度的必要配置（对等于 Clover 针对 196bp 调的 tree_depth/index 参数）。
GRADHC_Q    = 8
GRADHC_K    = 3
GRADHC_M    = 40
GRADHC_L    = 32
GRADHC_DIST = 12             # distance_threshold
GRADHC_TECH = 'minion_idt'

# sd_high 覆盖：默认 final=0.25 / chunk=0.32 在本数据上会让「链式滚雪球」误并不同 GT 分子
#   (诊断：巨簇内不同 GT 的 consensus 经边缘相似对串联成 29,627 条巨簇)。
#   权衡扫描：同源对 sd p5=0.925、异源 consensus sd p99=0.235，在 0.44~0.92 间有大空隙。
#   sd_high=0.40 落在空隙正中：同源保留 99.9%（不伤召回）、跨GT误并降到 0.01%（掐断雪球）。
GRADHC_SD_HIGH = 0.40


def banner(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


# ============================================================
# Step 0: 预处理 + 打薄  （与 Clover 版完全相同）
# ============================================================
def step0_preprocess_and_thin(input_file, max_reads_per_tag):
    """读 output.txt → 去N + 长度过滤 → 按 tag 分组 → 每 tag 随机采样 ≤ max_reads_per_tag 条。"""
    banner("Step 0  预处理 + 打薄")

    total = 0
    n_dropped = 0
    len_dropped = 0
    tag_to_reads = defaultdict(list)  # tag → [(tag, read), ...]

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            tag, seq = parts

            if 'N' in seq.upper():
                n_dropped += 1
                continue

            if not (LEN_MIN <= len(seq) <= LEN_MAX):
                len_dropped += 1
                continue

            tag_to_reads[tag].append((tag, seq))

    after_filter = sum(len(v) for v in tag_to_reads.values())
    n_tags_before = len(tag_to_reads)

    # ── 打薄：每 tag 随机采样 ──
    rng = random.Random(RANDOM_SEED)
    cleaned = []
    thinned_reads = 0

    for tag in sorted(tag_to_reads.keys(), key=lambda x: int(x)):
        reads = tag_to_reads[tag]
        if len(reads) > max_reads_per_tag:
            sampled = rng.sample(reads, max_reads_per_tag)
            thinned_reads += len(reads) - max_reads_per_tag
        else:
            sampled = reads
        cleaned.extend(sampled)

    n_tags_after = len(set(t for t, _ in cleaned))

    print(f"  输入文件:            {input_file}")
    print(f"  ref_len:             {REF_LEN}bp, 范围 [{LEN_MIN}, {LEN_MAX}]")
    print()
    print(f"  总行数:              {total:,}")
    print(f"  因含 N 剔除:         {n_dropped:,}")
    print(f"  因长度不合格剔除:    {len_dropped:,}")
    print(f"  过滤后 reads:        {after_filter:,}  ({n_tags_before:,} tags)")
    print()
    print(f"  ── 打薄 ──")
    print(f"  max_reads_per_tag:   {max_reads_per_tag}")
    print(f"  打薄丢弃:            {thinned_reads:,} 条 reads")
    print(f"  打薄后 reads:        {len(cleaned):,}  ({n_tags_after:,} tags)")
    print()

    before_sizes = [len(v) for v in tag_to_reads.values()]
    after_tag_counts = Counter(t for t, _ in cleaned)
    after_sizes = list(after_tag_counts.values())

    print(f"  每 tag reads 数变化:")
    print(f"    打薄前:  avg={sum(before_sizes)/len(before_sizes):.1f}, "
          f"max={max(before_sizes)}, med={sorted(before_sizes)[len(before_sizes)//2]}")
    print(f"    打薄后:  avg={sum(after_sizes)/len(after_sizes):.1f}, "
          f"max={max(after_sizes)}, med={sorted(after_sizes)[len(after_sizes)//2]}")

    return cleaned


# ============================================================
# Step 1: 写 GradHC 输入（分块格式，无监督）
# ============================================================
def step1_write_gradhc_input(cleaned, gradhc_input_path, gradhc_tag_path):
    """
    GradHC 输入格式（见 README / process_input）:
        <rep 串>
        *****************************
        <噪声拷贝 1>
        <噪声拷贝 2>
        ...
        <空行>
        <空行>
        <下一块 rep 串>
        ...

    无监督约束:
        GradHC 用 '*' 上一行作为该块的「真值 rep」(original_strand_dict)，只用于它内部
        的精度统计，不参与聚类决策。为不泄露 GT，我们给每块 rep 填一个占位串。
        这里把整份打薄数据当作「一个待聚类的 input」——即所有 reads 放进同一块，让 GradHC
        从零开始聚类（这才是 baseline 该做的：不告诉它任何分簇先验）。

    同时建立 read → [tags] 映射写入 gradhc_tag_path，供 Step 3 回查 tag。
    """
    banner("Step 1  写 GradHC 输入文件")

    # 占位 rep：用一个不会与真实 read 混淆的串（长度等于 REF_LEN 的 'A'）
    placeholder_rep = 'A' * REF_LEN

    # 全部 reads 放进同一块 —— baseline 不提供任何分簇先验
    with open(gradhc_input_path, 'w', newline='\n') as f:
        f.write(placeholder_rep + '\n')
        f.write('*' * 29 + '\n')
        for tag, read in cleaned:
            f.write(read + '\n')
        f.write('\n\n')

    print(f"  GradHC 输入:    {gradhc_input_path}")
    print(f"                  {len(cleaned):,} 条 reads（单块、无监督）")

    # read → tag 映射（用于 Step 3 回查；同一 read 可能对应多个 tag，保留列表）
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
    print()
    print(f"  💡 GT tags 数: {n_tags:,}")

    return read_to_tags


# ============================================================
# Step 2: 运行 GradHC
# ============================================================
def step2_run_gradhc(gradhc_input_path):
    """
    通过 import 调用 GradHCBasedCluster（而非 subprocess 调脚本），以便传入
    适配 196bp 的 q 等参数（GradHC 的 __main__ 没暴露 q 的 CLI 接口）。
    不修改 GradHC 源码，仅在构造时传参——对等于 Clover 针对 196bp 调 tree_depth/index。

    GradHC 的 export_file() 用 os.getcwd() 决定 Results 输出路径，故调用前 chdir 到
    GRADHC_DIR，调用后切回，保证输出落在 <GRADHC_DIR>/Results/ 且下游 glob 解析零改动。
    """
    banner("Step 2  运行 GradHC")

    results_dir = os.path.join(GRADHC_DIR, 'Results')
    os.makedirs(results_dir, exist_ok=True)

    input_base = os.path.basename(gradhc_input_path)
    # 清掉旧结果（同前缀），避免 glob 抓到上一次的
    old_pattern = os.path.join(results_dir, input_base + '_*.clustering_results')
    for old in glob.glob(old_pattern):
        os.remove(old)
        print(f"  🧹 清除旧结果: {os.path.basename(old)}")

    # ⚠ 关键：GradHC 模块顶层有 WORKING_DIR_ALGORITHMS = os.getcwd()+"/"，
    #   它在 import 那一刻就固定。export_file 用它拼 Results/ 输出路径。
    #   因此必须在 import 之前 chdir 到 GRADHC_DIR，否则结果会写到错误目录。
    prev_cwd = os.getcwd()
    os.chdir(GRADHC_DIR)
    if GRADHC_DIR not in sys.path:
        sys.path.insert(0, GRADHC_DIR)
    from GradHC_clustering import GradHCBasedCluster

    # 子类多态覆盖 sd_high（不改 GradHC 源码）：
    #   clustering_in_chunks 内部 self.clustering_given_chunk(...) 与 run() 里 self.final_clustering()
    #   均走子类版（多态），注入 sd_high=GRADHC_SD_HIGH，其余参数原样 super() 传回。
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
    print(f"          sd_high={GRADHC_SD_HIGH} (覆盖默认 final=0.25/chunk=0.32，掐断链式滚雪球)")
    print(f"  调用方式: import GradHCSdHigh(q={GRADHC_Q}, sd_high={GRADHC_SD_HIGH}).run()")
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
    """
    GradHC 输出每块：
        <rep 串(它选的代表)>
        *****************************
        <read 1>
        <read 2>
        ...
        <空行><空行>
    我们按块切分，每块的 reads（跳过 rep 行和 ***** 行）构成一个簇。
    返回 cid → [read 序列列表]。
    """
    banner("Step 3  解析 GradHC 输出")

    clusters = []          # list of [read, read, ...]
    cur_reads = None
    expect_rep = True      # 块内第一行是 rep

    with open(gradhc_output_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if line == '':
                # 空行 → 当前块结束
                if cur_reads is not None and len(cur_reads) > 0:
                    clusters.append(cur_reads)
                cur_reads = None
                expect_rep = True
                continue
            if line[0] == '*':
                # ***** 分隔行，下一行起是 reads
                expect_rep = False
                cur_reads = []
                continue
            if expect_rep:
                # rep 行（GradHC 选的代表串），跳过；开启新块
                cur_reads = None
                expect_rep = True
                # rep 行后应紧跟 ***** 行，这里不收集
                continue
            else:
                if cur_reads is None:
                    cur_reads = []
                cur_reads.append(line)

    # 收尾
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
# Step 3.5: 过滤小簇  （与 Clover 版一致）
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
# Step 4: 统计（用 read→tag 回查，而非 idx）
# ============================================================
def step4_statistics(cid_to_reads, read_to_tags, min_reads, stats_path):
    banner("Step 4  聚类统计（过滤后）")

    total_reads = sum(len(v) for v in cid_to_reads.values())
    n_clusters = len(cid_to_reads)
    sizes = sorted([len(v) for v in cid_to_reads.values()], reverse=True)

    # 用一个可消耗的 read→tag 队列副本，避免同一 read 重复消费同一 tag
    pool = {r: list(tags) for r, tags in read_to_tags.items()}

    total_pure = 0
    gt_tags_covered = set()

    for cid, reads in cid_to_reads.items():
        tags = []
        for read in reads:
            cand = pool.get(read)
            if cand:
                tags.append(cand.pop())   # 消费一个 tag 实例
            # read 不在映射里（理论上不会发生）则跳过
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
        f.write(f"Seq_1D GradHC 聚类统计（打薄 + 小簇过滤）\n{'='*40}\n\n")
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
# Step 5+6: Majority Vote → read.txt + ref.txt  （与 Clover 版一致）
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
# Step 7: 部署到实验目录  （与 Clover 版一致，路径换 GradHC 实验目录）
# ============================================================
def step7_deploy(gradhc_tag_path):
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

    # GT tags：gradhc_tag_path 已是 "tag\tread" 格式，直接复制
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
    print(f"  🚀 运行实验:")
    print(f"     cd /mnt/st_data/liangxinyi/code/models")
    print(f"     python main_loop.py \\")
    print(f"       --experiment_dir {EXPERIMENT_DIR}/ \\")
    print(f"       --feddna_checkpoint /mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth \\")
    print(f"       --max_iterations 3 --max_length {REF_LEN} --cl_mode ours \\")
    print(f"       --gt_tags_file {gt_tags_path} \\")
    print(f"       --gt_refs_file {gt_refs_path} \\")
    print(f"       2>&1 | tee seq1d_gradhc_v1.log")


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Seq_1D GradHC Pipeline（打薄版）')
    parser.add_argument('--skip_gradhc', action='store_true',
                        help='跳过 GradHC，复用已有结果')
    parser.add_argument('--gradhc_result', type=str, default=None,
                        help='--skip_gradhc 时指定已有的 .clustering_results 路径')
    parser.add_argument('--max_reads_per_tag', type=int, default=MAX_READS_PER_TAG,
                        help=f'每个 tag 最多保留的 reads 数（默认 {MAX_READS_PER_TAG}，FedDNA 用 5-30）')
    parser.add_argument('--min_reads', type=int, default=MIN_READS_PER_CLUSTER,
                        help=f'聚类后簇最少 reads 数（默认 {MIN_READS_PER_CLUSTER}）')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, '04_FedDNA_In'), exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    input_file       = os.path.join(BASE_DIR, 'output.txt')
    gradhc_input     = os.path.join(OUTPUT_DIR, '01_gradhc_input.txt')
    gradhc_tag_input = os.path.join(OUTPUT_DIR, '01_gradhc_tag_input.txt')
    stats_path       = os.path.join(OUTPUT_DIR, '03_stats.txt')
    read_path        = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'read.txt')
    ref_path         = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'ref.txt')

    print()
    print("=" * 60)
    print("  🚀  Seq_1D Pipeline（GradHC baseline）")
    print("=" * 60)
    print()
    print(f"  数据目录:         {BASE_DIR}")
    print(f"  GradHC 目录:      {GRADHC_DIR}")
    print(f"  输出目录:         {OUTPUT_DIR}")
    print(f"  实验目录:         {EXPERIMENT_DIR}")
    print()
    print(f"  ref_len:          {REF_LEN}bp")
    print(f"  max_reads/tag:    {args.max_reads_per_tag}  ← 打薄参数")
    print(f"  min_reads/簇:     {args.min_reads}")
    print(f"  GradHC:           q={GRADHC_Q} k={GRADHC_K} m={GRADHC_M} L={GRADHC_L} dist={GRADHC_DIST}")

    t_start = time.time()

    # Step 0: 预处理 + 打薄
    cleaned = step0_preprocess_and_thin(input_file, args.max_reads_per_tag)

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
    step7_deploy(gradhc_tag_input)

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print("  🎉  Pipeline 完成!")
    print("=" * 60)
    print()
    print(f"  总耗时:         {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  打薄:           每 tag ≤ {args.max_reads_per_tag} reads")
    print(f"  Purity:         {purity*100:.2f}%")
    print(f"  Coverage:       {coverage*100:.2f}%")
    print()


if __name__ == '__main__':
    main()