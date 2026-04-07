#!/usr/bin/env python3
"""
pipeline_seq1d.py
=================
Sequencing_data_first_dimension 专用 Pipeline（打薄版）

  output.txt → 预处理 → 打薄(每tag最多N条) → Clover聚类 → 小簇过滤 → 统计 → read.txt + ref.txt

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension
    python pipeline_seq1d.py                          # 完整运行（默认每tag最多30条）
    python pipeline_seq1d.py --max_reads_per_tag 15   # 更激进打薄
    python pipeline_seq1d.py --skip_clover            # 跳过 Clover，复用已有结果

打薄策略（与 FedDNA 论文一致）:
    FedDNA: "for each DNA cluster, 5 to 30 reads were randomly sampled"
    本脚本: 对每个 BWA tag，随机采样最多 max_reads_per_tag 条 reads
    打薄在 Clover 之前执行 → Clover 可用冗余减少 → 聚类质量下降 → SSI-EC 有提升空间
"""

import os
import re
import sys
import subprocess
import time
import random
from collections import defaultdict, Counter

# ============================================================
# 配置
# ============================================================
BASE_DIR    = '/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension'
CLOVER_DIR  = '/mnt/st_data/liangxinyi/code/CC/Step0/Clover'
OUTPUT_DIR  = os.path.join(BASE_DIR, 'clover_out')
EXPERIMENT_DIR = '/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d'

REF_LEN     = 196
LEN_MIN     = REF_LEN - 5   # 191
LEN_MAX     = REF_LEN + 5   # 201
MIN_READS_PER_CLUSTER = 5   # Clover 聚类后小簇过滤
N_GT_TAGS   = 11826          # design 总数
MAX_READS_PER_TAG = 30       # 每个 tag 最多保留的 reads 数（FedDNA: 5-30）
RANDOM_SEED = 42

# Clover 参数
CLOVER_TREE_DEPTH = 20
CLOVER_V_DRIFT    = 3
CLOVER_H_DRIFT    = 3
CLOVER_H_INDEX    = 20
CLOVER_E_INDEX    = 20


def banner(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


# ============================================================
# Step 0: 预处理 + 打薄
# ============================================================
def step0_preprocess_and_thin(input_file, max_reads_per_tag):
    """
    读 output.txt → 去N + 长度过滤 → 按 tag 分组 → 每 tag 随机采样 ≤ max_reads_per_tag 条。
    """
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

    # ── 丢弃打薄后 < MIN_READS_PER_CLUSTER 的 tag ──
    # （打薄不会让 tag 变少，因为 max_reads_per_tag >= MIN_READS_PER_CLUSTER）

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

    # 打薄前后的每 tag 统计
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
# Step 1: 写 Clover 输入
# ============================================================
def step1_write_clover_input(cleaned, clover_input_path, clover_tag_path):
    banner("Step 1  写 Clover 输入文件")

    idx_map = {}

    with open(clover_input_path, 'w') as f:
        for i, (tag, read) in enumerate(cleaned, start=1):
            f.write(f"{i} {read}\n")
            idx_map[i] = (tag, read)

    print(f"  --no-tag 输入:  {clover_input_path}")
    print(f"                  {len(idx_map):,} 条 reads")

    with open(clover_tag_path, 'w') as f:
        for tag, read in cleaned:
            f.write(f"{tag} {read}\n")

    n_tags = len(set(t for t, _ in cleaned))

    print(f"  tag 模式输入:   {clover_tag_path}")
    print()
    print(f"  💡 验证 accuracy:")
    print(f"     cd {CLOVER_DIR}")
    print(f"     python -m clover.main -I {clover_tag_path} "
          f"-L {REF_LEN} -T {n_tags} -D {CLOVER_TREE_DEPTH} -V {CLOVER_V_DRIFT} -H {CLOVER_H_DRIFT}")

    return idx_map


# ============================================================
# Step 2: 运行 Clover
# ============================================================
def step2_run_clover(clover_input_path, clover_output_base):
    banner("Step 2  运行 Clover")

    config_path = os.path.join(CLOVER_DIR, 'clover', 'load_config.py')
    with open(config_path, 'r') as f:
        content = f.read()

    patches = {
        'h_index_nums':  CLOVER_H_INDEX,
        'e_index_nums':  CLOVER_E_INDEX,
        'thd_tree_loc':  72,
        'four_tree_loc': 124,
    }

    modified = False
    for key, val in patches.items():
        pattern = rf'"{key}"\s*:\s*\d+'
        replacement = f'"{key}" : {val}'
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            content = new_content
            modified = True

    if modified:
        with open(config_path, 'w') as f:
            f.write(content)
        print(f"  ✅ 已更新 load_config.py")
        for k, v in patches.items():
            print(f"     {k:20s} = {v}")
        print()

    cmd = [
        sys.executable, "-m", "clover.main",
        "-I", clover_input_path,
        "-O", clover_output_base,
        "-L", str(REF_LEN),
        "-P", "0",
        "-D", str(CLOVER_TREE_DEPTH),
        "-V", str(CLOVER_V_DRIFT),
        "-H", str(CLOVER_H_DRIFT),
        "--no-tag",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = CLOVER_DIR + os.pathsep + env.get("PYTHONPATH", "")

    print(f"  Clover: -L {REF_LEN} -D {CLOVER_TREE_DEPTH} -V {CLOVER_V_DRIFT} -H {CLOVER_H_DRIFT}")
    print(f"  运行中 ...\n")

    t0 = time.time()
    subprocess.run(cmd, check=True, env=env, cwd=CLOVER_DIR)
    elapsed = time.time() - t0

    result_path = clover_output_base + ".txt"
    if not os.path.exists(result_path):
        alt_path = os.path.join(CLOVER_DIR, os.path.basename(clover_output_base) + ".txt")
        if os.path.exists(alt_path):
            os.rename(alt_path, result_path)
        else:
            raise FileNotFoundError(f"Clover 输出文件不存在: {result_path}")

    print(f"\n  ✅ Clover 完成，耗时 {elapsed:.1f}s")
    return result_path


# ============================================================
# Step 3: 解析 Clover 输出
# ============================================================
def step3_parse_clover(clover_output_path):
    banner("Step 3  解析 Clover 输出")

    with open(clover_output_path, 'r') as f:
        content = f.read()

    pairs = re.findall(r"\('?(\d+)'?,\s*'?(\d+)'?\)", content)

    cid_to_idxs = defaultdict(list)
    for idx_str, cid_str in pairs:
        cid_to_idxs[int(cid_str)].append(int(idx_str))

    sizes = [len(v) for v in cid_to_idxs.values()]

    print(f"  (idx,cid): {len(pairs):,} 条")
    print(f"  原始簇数:  {len(cid_to_idxs):,}")
    if sizes:
        print(f"  max={max(sizes)}, med={sorted(sizes)[len(sizes)//2]}, min={min(sizes)}")

    return dict(cid_to_idxs)


# ============================================================
# Step 3.5: 过滤小簇
# ============================================================
def step3_5_filter_small_clusters(cid_to_idxs, min_reads):
    banner(f"Step 3.5  过滤小簇 (reads < {min_reads})")

    before_n = len(cid_to_idxs)
    before_r = sum(len(v) for v in cid_to_idxs.values())

    filtered = {cid: idxs for cid, idxs in cid_to_idxs.items()
                if len(idxs) >= min_reads}

    after_n = len(filtered)
    after_r = sum(len(v) for v in filtered.values())

    print(f"  过滤前:  {before_n:,} 簇,  {before_r:,} reads")
    print(f"  过滤后:  {after_n:,} 簇,  {after_r:,} reads")
    print(f"  丢弃:    {before_n - after_n:,} 簇 ({before_r - after_r:,} reads)")

    return filtered


# ============================================================
# Step 4: 统计
# ============================================================
def step4_statistics(cid_to_idxs, idx_map, min_reads, stats_path):
    banner("Step 4  聚类统计（过滤后）")

    total_reads = sum(len(v) for v in cid_to_idxs.values())
    n_clusters = len(cid_to_idxs)
    sizes = sorted([len(v) for v in cid_to_idxs.values()], reverse=True)

    total_pure = 0
    gt_tags_covered = set()

    for cid, idxs in cid_to_idxs.items():
        tags = [idx_map[idx][0] for idx in idxs if idx in idx_map]
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
        f.write(f"Seq_1D 聚类统计（打薄 + 小簇过滤）\n{'='*40}\n\n")
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
# Step 5+6: Majority Vote + 写输出
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


def step56_write_output(cid_to_idxs, idx_map, read_path, ref_path):
    banner("Step 5+6  Majority Vote → read.txt + ref.txt")

    SEPARATOR = "=====分隔符=====\n"
    n_clusters = n_reads = 0

    with open(read_path, 'w') as fr, open(ref_path, 'w') as ff:
        for cid in sorted(cid_to_idxs.keys()):
            idxs = cid_to_idxs[cid]
            reads = [idx_map[idx][1] for idx in idxs if idx in idx_map]
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
# Step 7: 自动部署到实验目录
# ============================================================
def step7_deploy(idx_map, clover_tag_path):
    """复制 read.txt/ref.txt 到实验目录，生成 GT tags 文件。"""
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

    # GT tags: 从 tag 模式输入转成 tab 分隔
    gt_tags_path = os.path.join(EXPERIMENT_DIR, 'seq1d_tags_reads.txt')
    with open(clover_tag_path, 'r') as fin, open(gt_tags_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # 输入是 "tag read"（空格分隔），转为 "tag\tread"
            parts = line.split(' ', 1)
            if len(parts) == 2:
                fout.write(f"{parts[0]}\t{parts[1]}\n")

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
    print(f"       2>&1 | tee seq1d_ours_v1.log")


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Seq_1D Pipeline（打薄版）')
    parser.add_argument('--skip_clover', action='store_true',
                        help='跳过 Clover，复用已有结果')
    parser.add_argument('--max_reads_per_tag', type=int, default=MAX_READS_PER_TAG,
                        help=f'每个 tag 最多保留的 reads 数（默认 {MAX_READS_PER_TAG}，FedDNA 用 5-30）')
    parser.add_argument('--min_reads', type=int, default=MIN_READS_PER_CLUSTER,
                        help=f'Clover 聚类后簇最少 reads 数（默认 {MIN_READS_PER_CLUSTER}）')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, '04_FedDNA_In'), exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    input_file       = os.path.join(BASE_DIR, 'output.txt')
    clover_input     = os.path.join(OUTPUT_DIR, '01_clover_input.txt')
    clover_tag_input = os.path.join(OUTPUT_DIR, '01_clover_tag_input.txt')
    clover_out_base  = os.path.join(OUTPUT_DIR, '02_clover_result')
    clover_out_txt   = clover_out_base + '.txt'
    stats_path       = os.path.join(OUTPUT_DIR, '03_stats.txt')
    read_path        = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'read.txt')
    ref_path         = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'ref.txt')

    print()
    print("=" * 60)
    print("  🚀  Seq_1D Pipeline（打薄版）")
    print("=" * 60)
    print()
    print(f"  数据目录:         {BASE_DIR}")
    print(f"  输出目录:         {OUTPUT_DIR}")
    print(f"  实验目录:         {EXPERIMENT_DIR}")
    print()
    print(f"  ref_len:          {REF_LEN}bp")
    print(f"  max_reads/tag:    {args.max_reads_per_tag}  ← 打薄参数")
    print(f"  min_reads/簇:     {args.min_reads}")
    print(f"  Clover:           D={CLOVER_TREE_DEPTH} V={CLOVER_V_DRIFT} H={CLOVER_H_DRIFT}")

    t_start = time.time()

    # Step 0: 预处理 + 打薄
    cleaned = step0_preprocess_and_thin(input_file, args.max_reads_per_tag)

    # Step 1: 写 Clover 输入
    idx_map = step1_write_clover_input(cleaned, clover_input, clover_tag_input)
    del cleaned

    # Step 2: Clover
    if not args.skip_clover:
        clover_out_txt = step2_run_clover(clover_input, clover_out_base)
    else:
        banner("Step 2  跳过 Clover")
        print(f"  使用: {clover_out_txt}")
        if not os.path.exists(clover_out_txt):
            raise FileNotFoundError(f"不存在: {clover_out_txt}")

    # Step 3: 解析
    cid_to_idxs = step3_parse_clover(clover_out_txt)

    # Step 3.5: 过滤小簇
    cid_to_idxs = step3_5_filter_small_clusters(cid_to_idxs, args.min_reads)

    # Step 4: 统计
    purity, coverage = step4_statistics(cid_to_idxs, idx_map, args.min_reads, stats_path)

    # Step 5+6: 写输出
    step56_write_output(cid_to_idxs, idx_map, read_path, ref_path)

    # Step 7: 部署到实验目录
    step7_deploy(idx_map, clover_tag_input)

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