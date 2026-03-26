#!/usr/bin/env python3
"""
preprocess_and_cluster.py
=========================
完整 Pipeline: output.txt → 预处理 → Clover聚类 → 统计 → Majority Vote → read.txt + ref.txt

用法示例:
    # 处理 id20 (有引物)
    python preprocess_and_cluster.py --dataset id20

    # 处理其他数据集
    python preprocess_and_cluster.py --dataset P10_5
    python preprocess_and_cluster.py --dataset PE_AYB
    python preprocess_and_cluster.py --dataset Seq_1D

    # 自定义路径
    python preprocess_and_cluster.py --dataset id20 --base_dir /your/path

Pipeline 步骤:
    Step 0: 预处理 (去引物/长度筛选/去N)
    Step 1: 写 Clover 输入文件 (idx read, --no-tag 格式)
    Step 2: 运行 Clover 聚类
    Step 3: 解析 Clover 输出 → {cid: [idx, ...]}
    Step 4: 统计聚类结果 (purity, 簇大小分布等)
    Step 5: Majority Vote → 伪ref
    Step 6: 写 read.txt + ref.txt (FedDNA/SSI-EC 输入格式)
"""

import os
import re
import sys
import subprocess
import argparse
import time
from collections import defaultdict, Counter

# ============================================================
# 数据集配置
# ============================================================
# ref_len: 参考序列长度 (用于长度筛选 ±5 和 majority vote 截断)
# n_clusters: 比对到reads的参考序列数 (给 Clover -T 参数用，仅用于统计)
# has_primer: 是否需要去引物
# primer_start/primer_end: 仅 id20 使用，截取 [AGTG ... GCCG] 之间的序列

DATASET_CONFIGS = {
    'id20': {
        'base_dir': '/mnt/st_data/liangxinyi/code/CC/Step0/data-nbt17',
        'ref_len': 150,
        'has_primer': True,
        'primer_start': 'AGTG',
        'primer_end': 'GCCG',
        'n_clusters': 596669,
    },
    'P10_5': {
        'base_dir': '/mnt/st_data/liangxinyi/code/CC/Step0/P10_5_BDDP210000009',
        'ref_len': 200,
        'has_primer': False,
        'n_clusters': 209185,
    },
    'PE_AYB': {
        'base_dir': '/mnt/st_data/liangxinyi/code/CC/Step0/PE_AYB',
        'ref_len': 183,
        'has_primer': False,
        'n_clusters': 153331,
    },
    'Seq_1D': {
        'base_dir': '/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension',
        'ref_len': 196,
        'has_primer': False,
        'n_clusters': 11751,
    },
}


# ============================================================
# Step 0: 预处理
# ============================================================

def extract_primer_id20(read: str, primer_start: str, primer_end: str) -> str:
    """
    id20 专用去引物: 截取从 primer_start 到 primer_end(含) 之间的序列。
    与师姐 generate_original.py 的 remove_prefix_suffix_AGTG_GCCG 逻辑一致。

    返回截取后的序列，若找不到锚点则返回空字符串。
    """
    start_idx = read.find(primer_start)
    end_idx = read.rfind(primer_end)  # 用 rfind 取最后一个，与师姐一致

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        return read[start_idx: end_idx + len(primer_end)]
    return ''


def preprocess_reads(output_txt: str, config: dict) -> list:
    """
    读取 output.txt，按照数据集配置做预处理。

    输入格式: tag\tread (每行)
    输出: [(tag, cleaned_read), ...]  仅保留通过所有过滤的条目

    过滤规则:
      1. id20: 去引物 (AGTG...GCCG截取)
      2. 所有数据集: 去N (read 中含 N 则丢弃)
      3. 所有数据集: 长度过滤 [ref_len-5, ref_len+5]
    """
    ref_len = config['ref_len']
    len_min = ref_len - 5
    len_max = ref_len + 5
    has_primer = config.get('has_primer', False)
    primer_start = config.get('primer_start', '')
    primer_end = config.get('primer_end', '')

    cleaned = []
    total = 0
    kept = 0
    skipped_N = 0
    skipped_len = 0
    skipped_primer = 0

    print(f"   📖 读取 {output_txt} ...")
    with open(output_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1

            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            tag, read = parts[0], parts[1]

            # Step 1: 去引物 (仅 id20)
            if has_primer:
                read = extract_primer_id20(read, primer_start, primer_end)
                if not read:
                    skipped_primer += 1
                    continue

            # Step 2: 去 N
            if 'N' in read or 'n' in read:
                skipped_N += 1
                continue

            # Step 3: 长度过滤
            if not (len_min <= len(read) <= len_max):
                skipped_len += 1
                continue

            cleaned.append((tag, read))
            kept += 1

            if total % 1_000_000 == 0:
                print(f"      已处理 {total:,} 条...", end='\r')

    print(f"\n   ✅ 预处理完成:")
    print(f"      总 reads:     {total:,}")
    print(f"      保留:         {kept:,} ({kept/max(total,1)*100:.1f}%)")
    if has_primer:
        print(f"      去引物失败:   {skipped_primer:,}")
    print(f"      含N丢弃:      {skipped_N:,}")
    print(f"      长度不符:     {skipped_len:,}")

    return cleaned


# ============================================================
# Step 1: 写 Clover 输入文件
# ============================================================

def write_clover_input(cleaned_reads: list, clover_input_path: str):
    """
    写 Clover --no-tag 模式的输入文件。
    格式: idx read  (idx 从 1 开始，与原始行对应)

    同时返回 idx → (tag, read) 的映射字典，用于后续解析。
    idx 是 1-based 全局行号，与 Clover 输出的 idx 一致。
    """
    print(f"   ✍️  写 Clover 输入: {clover_input_path} ...")
    idx_to_info = {}  # {idx: (tag, read)}

    with open(clover_input_path, 'w') as f:
        for i, (tag, read) in enumerate(cleaned_reads, start=1):
            f.write(f"{i} {read}\n")
            idx_to_info[i] = (tag, read)

    print(f"   ✅ 写入 {len(cleaned_reads):,} 条 reads")
    return idx_to_info


# ============================================================
# Step 2: 运行 Clover
# ============================================================

def run_clover(clover_input_path: str, clover_output_base: str,
               ref_len: int, processes: int = 0):
    """
    调用 Clover 进行聚类 (--no-tag 模式)。

    Args:
        clover_input_path:   输入文件路径
        clover_output_base:  输出文件基名 (Clover 会生成 base.txt)
        ref_len:             参考序列长度 (-L 参数)
        processes:           进程数 (-P 参数, 0=单进程)
    """
    print(f"\n[Step 2] 运行 Clover ...")
    print(f"   输入:  {clover_input_path}")
    print(f"   输出:  {clover_output_base}.txt")
    print(f"   L={ref_len}, P={processes}")

    cmd = [
        sys.executable, "-m", "clover.main",
        "-I", clover_input_path,
        "-O", clover_output_base,
        "-L", str(ref_len),
        "-P", str(processes),
        "--no-tag",
    ]

    # 将 Clover 源码目录加入 PYTHONPATH（与 run_real_data_exp1_Fixed.py 一致）
    clover_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Clover")
    env = os.environ.copy()
    env["PYTHONPATH"] = clover_src + os.pathsep + env.get("PYTHONPATH", "")

    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    t1 = time.time()

    clover_output_txt = clover_output_base + ".txt"
    if not os.path.exists(clover_output_txt):
        raise FileNotFoundError(f"Clover 输出文件不存在: {clover_output_txt}")

    print(f"   ✅ Clover 完成，耗时 {t1-t0:.1f}s")
    return clover_output_txt


# ============================================================
# Step 3: 解析 Clover 输出
# ============================================================

def parse_clover_output(clover_output_txt: str) -> dict:
    """
    解析 Clover --no-tag 模式的输出文件。

    Clover 输出格式: 每行或整体包含 ('idx', 'cid') 或 (idx, cid) 的 tuple。
    返回: {cid: [idx, ...]} 字典
    """
    print(f"\n[Step 3] 解析 Clover 输出: {clover_output_txt} ...")

    with open(clover_output_txt, 'r') as f:
        content = f.read()

    # 匹配 ('idx', 'cid') 或 (idx, cid) 格式
    pairs = re.findall(r"\('?(\d+)'?,\s*'?(\d+)'?\)", content)
    print(f"   解析到 {len(pairs):,} 条 (idx, cid) 对")

    cid_to_idxs = defaultdict(list)
    for idx_str, cid_str in pairs:
        cid_to_idxs[int(cid_str)].append(int(idx_str))

    print(f"   ✅ 共 {len(cid_to_idxs):,} 个聚类簇")
    return dict(cid_to_idxs)


# ============================================================
# Step 4: 统计聚类结果
# ============================================================

def compute_statistics(cid_to_idxs: dict, idx_to_info: dict,
                       n_gt_clusters: int, stats_path: str):
    """
    计算并保存聚类统计信息:
      - 簇数量、reads 分布
      - Purity: 对每个聚类簇，找最多数标签的比例
      - Coverage: 有多少 GT 簇被覆盖
    """
    print(f"\n[Step 4] 统计聚类结果 ...")

    total_reads_clustered = sum(len(v) for v in cid_to_idxs.values())
    n_clusters = len(cid_to_idxs)

    # 簇大小分布
    sizes = sorted([len(v) for v in cid_to_idxs.values()], reverse=True)
    size_counter = Counter(sizes)

    # Purity 计算
    total_pure = 0
    gt_tags_covered = set()

    for cid, idxs in cid_to_idxs.items():
        tags = [idx_to_info[idx][0] for idx in idxs if idx in idx_to_info]
        if not tags:
            continue
        tag_counts = Counter(tags)
        majority_tag, majority_count = tag_counts.most_common(1)[0]
        total_pure += majority_count
        gt_tags_covered.add(majority_tag)

    purity = total_pure / max(total_reads_clustered, 1)
    coverage = len(gt_tags_covered) / max(n_gt_clusters, 1)

    # 簇大小分布 buckets
    buckets = {
        '1':       sum(1 for s in sizes if s == 1),
        '2-5':     sum(1 for s in sizes if 2 <= s <= 5),
        '6-10':    sum(1 for s in sizes if 6 <= s <= 10),
        '11-30':   sum(1 for s in sizes if 11 <= s <= 30),
        '31-100':  sum(1 for s in sizes if 31 <= s <= 100),
        '>100':    sum(1 for s in sizes if s > 100),
    }

    lines = [
        "=" * 60,
        "Clover 聚类统计",
        "=" * 60,
        f"聚类簇数:              {n_clusters:,}",
        f"GT 参考簇数:           {n_gt_clusters:,}",
        f"已聚类 reads 总数:     {total_reads_clustered:,}",
        f"",
        f"Purity:                {purity*100:.2f}%",
        f"Coverage:              {coverage*100:.2f}%  ({len(gt_tags_covered)}/{n_gt_clusters})",
        f"",
        f"簇大小分布:",
        f"  单条簇 (size=1):     {buckets['1']:,}",
        f"  2-5 reads:           {buckets['2-5']:,}",
        f"  6-10 reads:          {buckets['6-10']:,}",
        f"  11-30 reads:         {buckets['11-30']:,}",
        f"  31-100 reads:        {buckets['31-100']:,}",
        f"  >100 reads:          {buckets['>100']:,}",
        f"",
        f"簇大小: max={sizes[0]}, median={sizes[len(sizes)//2]}, min={sizes[-1]}",
    ]

    report = "\n".join(lines)
    print(report)

    with open(stats_path, 'w') as f:
        f.write(report + "\n")
    print(f"\n   💾 统计报告: {stats_path}")

    return {'purity': purity, 'coverage': coverage, 'n_clusters': n_clusters}


# ============================================================
# Step 5: Majority Vote → 伪 ref
# ============================================================

def majority_vote(reads: list, ref_len: int) -> str:
    """
    对一个簇内的 reads 做逐位多数投票，生成 pseudo-ref。

    处理 IDS 导致的长度差异:
      - 对每个位置 0..ref_len-1，统计该位置有碱基的 reads
      - 取票数最多的碱基 (A/C/G/T)
      - 若某位置所有 reads 都比该位置短 (不覆盖)，用前一位置的碱基填充

    返回长度恰好为 ref_len 的字符串。
    """
    vote = [Counter() for _ in range(ref_len)]
    for read in reads:
        for pos in range(min(len(read), ref_len)):
            base = read[pos].upper()
            if base in 'ACGT':
                vote[pos][base] += 1

    consensus = []
    last_base = 'A'
    for pos in range(ref_len):
        if vote[pos]:
            base = vote[pos].most_common(1)[0][0]
            last_base = base
        else:
            base = last_base  # 该位置无覆盖，用前一位补充
        consensus.append(base)

    return ''.join(consensus)


# ============================================================
# Step 6: 写 read.txt + ref.txt
# ============================================================

def write_feddna_output(cid_to_idxs: dict, idx_to_info: dict,
                        ref_len: int, read_txt_path: str, ref_txt_path: str,
                        min_reads_per_cluster: int = 5): # 新增阈值参数
    """
    将聚类结果写成 SSI-EC/FedDNA 的输入格式:
      read.txt: 每个簇的 reads，簇间用 ===...=== 分隔
      ref.txt:  每行一条 pseudo-ref，与 read.txt 簇顺序对应

    格式与 step1_data.py 的 CloverDataLoader 兼容:
      - 分隔符以 "=====" 开头 (代码里是 line.startswith("====="))
      - 每个簇: reads行 → 分隔符行

    过滤: 只保留 size >= 1 的簇 (全部保留，模型训练时动态采样控制)
    """
    print(f"\n[Step 6] 写 FedDNA 格式输出 ...")

    SEPARATOR = "===============================\n"

    n_written_clusters = 0
    n_written_reads = 0

    # 按 cid 排序保证确定性
    sorted_cids = sorted(cid_to_idxs.keys())

    with open(read_txt_path, 'w') as fr, open(ref_txt_path, 'w') as ff:
        for cid in sorted_cids:
            idxs = cid_to_idxs[cid]
            reads = [idx_to_info[idx][1] for idx in idxs if idx in idx_to_info]
            
            # 🔥 核心修改：如果该簇内分配到的有效序列少于 5 条，直接整簇丢弃！
            if len(reads) < min_reads_per_cluster:
                continue

            # 计算 pseudo-ref (majority vote)
            pseudo_ref = majority_vote(reads, ref_len)

            # 写 read.txt
            for read in reads:
                fr.write(read + '\n')
            fr.write(SEPARATOR)

            # 写 ref.txt
            ff.write(pseudo_ref + '\n')

            n_written_clusters += 1
            n_written_reads += len(reads)

            if n_written_clusters % 10000 == 0:
                print(f"      已写 {n_written_clusters:,} 个簇...", end='\r')

    print(f"\n   ✅ 写入完成:")
    print(f"      簇数:  {n_written_clusters:,}")
    print(f"      reads: {n_written_reads:,}")
    print(f"      read.txt: {read_txt_path}")
    print(f"      ref.txt:  {ref_txt_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DNA存储数据集预处理 + Clover聚类 Pipeline')
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()),
                        help='数据集名称')
    parser.add_argument('--base_dir', default=None,
                        help='数据目录 (覆盖配置文件中的默认路径)')
    parser.add_argument('--output_dir', default=None,
                        help='输出目录 (默认: base_dir/clover_pipeline_out)')
    parser.add_argument('--clover_processes', type=int, default=0,
                        help='Clover 进程数 (0=单进程, 1=4进程, 2=16进程)')
    parser.add_argument('--skip_clover', action='store_true',
                        help='跳过 Clover 步骤 (用于复用已有 Clover 输出)')
    args = parser.parse_args()

    config = DATASET_CONFIGS[args.dataset].copy()
    if args.base_dir:
        config['base_dir'] = args.base_dir

    base_dir = config['base_dir']
    output_dir = args.output_dir or os.path.join(base_dir, 'clover_pipeline_out')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🚀 Dataset: {args.dataset}")
    print(f"   base_dir:   {base_dir}")
    print(f"   output_dir: {output_dir}")
    print(f"   ref_len:    {config['ref_len']}")
    print(f"   has_primer: {config['has_primer']}")
    print(f"{'='*60}\n")

    t_total = time.time()

    # ── 子目录定义 (与 run_real_data_exp1_Fixed.py 保持一致) ──
    dir_formatted = os.path.join(output_dir, "01_FormattedInput")
    dir_clover    = os.path.join(output_dir, "02_CloverOut")
    dir_feddna    = os.path.join(output_dir, "03_FedDNA_In")
    dir_temp      = os.path.join(output_dir, "99_Temp")
    for d in [dir_formatted, dir_clover, dir_feddna, dir_temp]:
        os.makedirs(d, exist_ok=True)

    # ── 路径定义 ──
    output_txt      = os.path.join(base_dir, 'output.txt')
    clover_input    = os.path.join(dir_formatted, 'clover_input.txt')
    clover_out_base = os.path.join(dir_clover, 'clover_out')
    clover_out_txt  = clover_out_base + '.txt'
    stats_path      = os.path.join(dir_clover, 'cluster_stats.txt')
    read_txt        = os.path.join(dir_feddna, 'read.txt')
    ref_txt         = os.path.join(dir_feddna, 'ref.txt')

    # ── Step 0: 预处理 ──
    print(f"[Step 0] 预处理 ...")
    cleaned_reads = preprocess_reads(output_txt, config)

    # ── Step 1: 写 Clover 输入 ──
    print(f"\n[Step 1] 写 Clover 输入文件 ...")
    idx_to_info = write_clover_input(cleaned_reads, clover_input)
    del cleaned_reads  # 释放内存，后续通过 idx_to_info 访问

    # ── Step 2: 运行 Clover ──
    if not args.skip_clover:
        run_clover(clover_input, clover_out_base,
                   config['ref_len'], args.clover_processes)
    else:
        print(f"\n[Step 2] 跳过 Clover (--skip_clover)，使用: {clover_out_txt}")
        if not os.path.exists(clover_out_txt):
            raise FileNotFoundError(f"Clover 输出不存在: {clover_out_txt}")

    # ── Step 3: 解析 Clover 输出 ──
    cid_to_idxs = parse_clover_output(clover_out_txt)

    # ── Step 4: 统计 ──
    compute_statistics(cid_to_idxs, idx_to_info,
                       config['n_clusters'], stats_path)

    # ── Step 5 + 6: Majority Vote + 写输出 ──
    write_feddna_output(cid_to_idxs, idx_to_info,
                        config['ref_len'], read_txt, ref_txt)

    print(f"\n{'='*60}")
    print(f"🎉 完成! 总耗时: {(time.time()-t_total)/60:.1f} 分钟")
    print(f"   输出目录: {output_dir}")
    print(f"   ├── 01_FormattedInput/clover_input.txt  ← Clover 输入 (可删)")
    print(f"   ├── 02_CloverOut/clover_out.txt         ← Clover 原始输出 (可删)")
    print(f"   ├── 02_CloverOut/cluster_stats.txt      ← 聚类统计 (Purity/Coverage等)")
    print(f"   └── 03_FedDNA_In/")
    print(f"       ├── read.txt  ← SSI-EC 输入")
    print(f"       └── ref.txt   ← Majority Vote 伪ref")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()