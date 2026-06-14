#!/usr/bin/env python3
"""
pipeline_p10_5_thin.py
======================
P10_5_BDDP210000009 数据集 Pipeline（打薄版）

  output.txt → 预处理 → 打薄(每tag最多N条) → tag→ref_id 重映射
            → Clover 聚类 → 小簇过滤 → 统计 → read.txt + ref.txt

数据集差异 (vs Seq_1D):
  REF_LEN:        196 → 200      reads 主流长度
  GT_REF_LEN:     196 → 201      ref.fasta 长度（比 reads 长 1bp）
  N_GT_TAGS:      11826 → 209185
  h_index_nums:   20 → 24        ("CCTGCAGAGTAGCATGTCATTGAT")
  e_index_nums:   20 → 18        ("CTGACACTGATGCATCCG")
  thd_tree_loc:   72 → 73
  four_tree_loc:  124 → 127
  GT FASTA:       reads.fasta → ref.fasta
  ★ 新增: tag→ref_id 显式映射
     P10_5 的 ref FASTA header 是散列 ID (>101010102, >101010104, ...) 不连续，
     必须在 Step 0 建立 ref_id (str) → row_index (int) 的映射表，
     把 BWA tag 替换成行号写入 GT tags 文件，
     否则 SSI-EC 主代码按 gt_refs[int(tag)-1] 查 ref 会越界/错位。

注意:
  ‣ MV 输出 200bp（reads 长度），GT ref 201bp
  ‣ SSI-EC 主代码: --max_length 201 --ref_length 200
    (v15 修复保证只在 FASTA 保存时裁剪、consensus_dict 不裁)

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/P10_5_BDDP210000009
    python pipeline_p10_5_thin.py                          # 完整运行
    python pipeline_p10_5_thin.py --max_reads_per_tag 15   # 更激进打薄
    python pipeline_p10_5_thin.py --skip_clover            # 跳过 Clover，复用已有结果
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
BASE_DIR    = '/mnt/st_data/liangxinyi/code/CC/Step0/P10_5_BDDP210000009'
CLOVER_DIR  = '/mnt/st_data/liangxinyi/code/CC/Step0/Clover'
OUTPUT_DIR  = os.path.join(BASE_DIR, 'clover_out_noisy')
EXPERIMENT_DIR = '/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/p10_5_noisy'

# ── 数据集核心常量 ──
REF_LEN     = 200            # reads 主流长度 (Seq_1D: 196)
LEN_MIN     = REF_LEN - 5    # 195
LEN_MAX     = REF_LEN + 5    # 205
GT_REF_LEN  = 201            # ref.fasta 长度 (Seq_1D: 196 = REF_LEN)

MIN_READS_PER_CLUSTER = 5
N_GT_TAGS   = 209185         # P10_5 mapped refs (Seq_1D: 11826)
MAX_READS_PER_TAG = 30
RANDOM_SEED = 42

# Clover 参数
CLOVER_TREE_DEPTH = 20
CLOVER_V_DRIFT    = 3
CLOVER_H_DRIFT    = 3
CLOVER_H_INDEX    = 24       # P10_5 上游 primer (Seq_1D: 20)
CLOVER_E_INDEX    = 18       # P10_5 下游 primer (Seq_1D: 20)
CLOVER_THD_LOC    = 73       # 36% × 200 (Seq_1D: 72)
CLOVER_FOUR_LOC   = 127      # 63% × 200 (Seq_1D: 124)


def banner(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


# ============================================================
# v3 新增: ref FASTA 解析 + 建立映射表
# ============================================================
def load_ref_fasta_with_mapping(ref_fasta_path):
    """
    读 ref.fasta，返回:
      - ref_id_to_idx: {ref_id (str) → 1-based row index (int)}
      - ref_seqs:      [(ref_id, seq), ...] 按文件顺序
    """
    ref_id_to_idx = {}
    ref_seqs = []

    with open(ref_fasta_path, 'r') as f:
        cur_id = None
        cur_seq_parts = []
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if cur_id is not None:
                    ref_id_to_idx[cur_id] = len(ref_seqs) + 1   # 1-based
                    ref_seqs.append((cur_id, ''.join(cur_seq_parts)))
                cur_id = line[1:].strip()
                cur_seq_parts = []
            else:
                cur_seq_parts.append(line)
        if cur_id is not None:
            ref_id_to_idx[cur_id] = len(ref_seqs) + 1
            ref_seqs.append((cur_id, ''.join(cur_seq_parts)))

    return ref_id_to_idx, ref_seqs


# ============================================================
# Step 0: 预处理 + 打薄 (vs Seq_1D 多了 tag→ref_id 映射逻辑)
# ============================================================
def step0_preprocess_and_thin(input_file, ref_id_to_idx, max_reads_per_tag):
    banner("Step 0  预处理 + 打薄 + tag→ref_id 重映射")

    total = 0
    n_dropped = 0
    len_dropped = 0
    unmapped_tag_dropped = 0   # ★ 新增：BWA tag 不在 ref FASTA 中
    tag_to_reads = defaultdict(list)

    unmapped_examples = []

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            bwa_tag, seq = parts

            if 'N' in seq.upper():
                n_dropped += 1
                continue

            if not (LEN_MIN <= len(seq) <= LEN_MAX):
                len_dropped += 1
                continue

            # ★ 关键: BWA tag → row_index 映射
            row_idx = ref_id_to_idx.get(bwa_tag)
            if row_idx is None:
                unmapped_tag_dropped += 1
                if len(unmapped_examples) < 5:
                    unmapped_examples.append(bwa_tag)
                continue

            # 用 row_idx (str) 作为 tag，下游代码与 Seq_1D 完全兼容
            tag_to_reads[str(row_idx)].append((str(row_idx), seq))

    after_filter = sum(len(v) for v in tag_to_reads.values())
    n_tags_before = len(tag_to_reads)

    # 打薄
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
    print(f"  GT ref 长度:         {GT_REF_LEN}bp (用于 SSI-EC --ref_length)")
    print(f"  ref 词典大小:        {len(ref_id_to_idx):,}  (来自 ref.fasta)")
    print()
    print(f"  总行数:              {total:,}")
    print(f"  因含 N 剔除:         {n_dropped:,}")
    print(f"  因长度不合格剔除:    {len_dropped:,}")
    print(f"  ★ tag 未映射剔除:    {unmapped_tag_dropped:,}")
    if unmapped_examples:
        print(f"     示例未匹配 tag:  {unmapped_examples}")
    print(f"  过滤后 reads:        {after_filter:,}  ({n_tags_before:,} tags)")
    print()
    print(f"  ── 打薄 ──")
    print(f"  max_reads_per_tag:   {max_reads_per_tag}")
    print(f"  打薄丢弃:            {thinned_reads:,} 条 reads")
    print(f"  打薄后 reads:        {len(cleaned):,}  ({n_tags_after:,} tags)")
    print()

    if tag_to_reads:
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
        'thd_tree_loc':  CLOVER_THD_LOC,
        'four_tree_loc': CLOVER_FOUR_LOC,
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
        f.write(f"P10_5 聚类统计（打薄 + 小簇过滤）\n{'='*40}\n\n")
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
    print(f"  read.txt: {read_path}  (reads {REF_LEN}bp)")
    print(f"  ref.txt:  {ref_path}   (MV consensus {REF_LEN}bp)")


# ============================================================
# Step 7: 部署到实验目录
# ============================================================
def step7_deploy(idx_map, clover_tag_path, ref_seqs):
    """复制 read.txt/ref.txt + 生成 GT tags 文件 + 生成 GT refs 文件（重排为 1..N 顺序）。"""
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

    # GT tags: 注意此时 tag 已经是 row_index (1-based)，不是 BWA 散列 ID
    gt_tags_path = os.path.join(EXPERIMENT_DIR, 'p10_5_noisy_tags_reads.txt')
    with open(clover_tag_path, 'r') as fin, open(gt_tags_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                fout.write(f"{parts[0]}\t{parts[1]}\n")

    print(f"  ✅ GT tags: {gt_tags_path}  (tag 已重映射为 row_index)")

    # ★ GT refs: 按 ref_seqs 顺序写出（即 row_index 1..N），下游代码可直接 gt_refs[tag-1]
    gt_refs_path = os.path.join(EXPERIMENT_DIR, 'p10_5_noisy_refs.txt')
    with open(gt_refs_path, 'w') as fout:
        for ref_id, seq in ref_seqs:
            fout.write(seq + '\n')

    print(f"  ✅ GT refs: {gt_refs_path}  ({len(ref_seqs):,} 行, {GT_REF_LEN}bp)")
    print()
    print(f"  🚀 运行实验 (完整 v19 参数):")
    print(f"     cd /mnt/st_data/liangxinyi/code/")
    print(f"     python main_loop.py \\")
    print(f"       --experiment_dir {EXPERIMENT_DIR}/ \\")
    print(f"       --feddna_checkpoint /mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth \\")
    print(f"       --gt_tags_file {gt_tags_path} \\")
    print(f"       --gt_refs_file {gt_refs_path} \\")
    print(f"       --max_iterations 3 \\")
    print(f"       --max_length 201 \\")
    print(f"       --target_clusters 209500 \\")
    print(f"       --cl_mode ours \\")
    print(f"       --ref_length {REF_LEN} \\")
    print(f"       --primer_prefix 24 \\")
    print(f"       --primer_suffix 18 \\")
    print(f"       --disable_merge \\")
    print(f"       --consensus_source mv \\")
    print(f"       --fasta_source mv_strict \\")
    print(f"       --zone_include_noise True \\")
    print(f"       --rebirth_mode nearest \\")
    print(f"       2>&1 | tee p10_5_noisy_v19.log")
    print()
    print(f"  ⚠️  GT ref 实际 {GT_REF_LEN}bp，reads/MV {REF_LEN}bp，主代码按 reads 长度评估")


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='P10_5 Pipeline（打薄版）')
    parser.add_argument('--skip_clover', action='store_true',
                        help='跳过 Clover，复用已有结果')
    parser.add_argument('--max_reads_per_tag', type=int, default=MAX_READS_PER_TAG,
                        help=f'每个 tag 最多保留的 reads 数（默认 {MAX_READS_PER_TAG}）')
    parser.add_argument('--min_reads', type=int, default=MIN_READS_PER_CLUSTER,
                        help=f'Clover 聚类后簇最少 reads 数（默认 {MIN_READS_PER_CLUSTER}）')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, '04_FedDNA_In'), exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    input_file       = os.path.join(BASE_DIR, 'output_noisy.txt')

    if not os.path.exists(input_file):
        print(f"\n  ❌ 输入文件不存在: {input_file}")
        print(f"     请先运行: python add_noise_p10_5.py\n")
        sys.exit(1)
    ref_fasta_path   = os.path.join(BASE_DIR, 'ref.fasta')
    clover_input     = os.path.join(OUTPUT_DIR, '01_clover_input.txt')
    clover_tag_input = os.path.join(OUTPUT_DIR, '01_clover_tag_input.txt')
    clover_out_base  = os.path.join(OUTPUT_DIR, '02_clover_result')
    clover_out_txt   = clover_out_base + '.txt'
    stats_path       = os.path.join(OUTPUT_DIR, '03_stats.txt')
    read_path        = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'read.txt')
    ref_path         = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'ref.txt')

    print()
    print("=" * 60)
    print("  🧬  P10_5 加噪版 Pipeline (打薄 + 噪声注入)")
    print("=" * 60)
    print()
    print(f"  数据目录:         {BASE_DIR}")
    print(f"  输出目录:         {OUTPUT_DIR}")
    print(f"  实验目录:         {EXPERIMENT_DIR}")
    print()
    print(f"  ref_len (reads):  {REF_LEN}bp")
    print(f"  ref_len (GT):     {GT_REF_LEN}bp")
    print(f"  N_GT_TAGS:        {N_GT_TAGS:,}")
    print(f"  max_reads/tag:    {args.max_reads_per_tag}")
    print(f"  min_reads/簇:     {args.min_reads}")
    print(f"  Clover:           D={CLOVER_TREE_DEPTH} V={CLOVER_V_DRIFT} H={CLOVER_H_DRIFT}")
    print(f"                    h_idx={CLOVER_H_INDEX} e_idx={CLOVER_E_INDEX} "
          f"thd={CLOVER_THD_LOC} four={CLOVER_FOUR_LOC}")

    t_start = time.time()

    # ★ 先加载 ref.fasta 建立映射
    banner("Step -1  加载 ref.fasta 建立 tag→ref_id 映射")
    ref_id_to_idx, ref_seqs = load_ref_fasta_with_mapping(ref_fasta_path)
    print(f"  ref FASTA:        {ref_fasta_path}")
    print(f"  ref 总条数:       {len(ref_seqs):,}")
    print(f"  ref 长度示例:     {len(ref_seqs[0][1])}bp (header={ref_seqs[0][0]})")
    print(f"  映射示例:         {ref_seqs[0][0]} → row 1")
    print(f"                   {ref_seqs[1][0]} → row 2")
    print(f"                   {ref_seqs[2][0]} → row 3")

    # Step 0
    cleaned = step0_preprocess_and_thin(input_file, ref_id_to_idx, args.max_reads_per_tag)
    if not cleaned:
        print("\n  ❌ Step 0 后无 reads，终止。")
        sys.exit(1)

    # Step 1
    idx_map = step1_write_clover_input(cleaned, clover_input, clover_tag_input)
    del cleaned

    # Step 2
    if not args.skip_clover:
        clover_out_txt = step2_run_clover(clover_input, clover_out_base)
    else:
        banner("Step 2  跳过 Clover")
        print(f"  使用: {clover_out_txt}")
        if not os.path.exists(clover_out_txt):
            raise FileNotFoundError(f"不存在: {clover_out_txt}")

    # Step 3
    cid_to_idxs = step3_parse_clover(clover_out_txt)

    # Step 3.5
    cid_to_idxs = step3_5_filter_small_clusters(cid_to_idxs, args.min_reads)

    # Step 4
    purity, coverage = step4_statistics(cid_to_idxs, idx_map, args.min_reads, stats_path)

    # Step 5+6
    step56_write_output(cid_to_idxs, idx_map, read_path, ref_path)

    # Step 7
    step7_deploy(idx_map, clover_tag_input, ref_seqs)

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print("  🎉  Pipeline 完成!")
    print("=" * 60)
    print()
    print(f"  总耗时:         {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  打薄:           每 tag ≤ {args.max_reads_per_tag} reads")
    print(f"  Purity:         {purity*100:.2f}%")
    print(f"  Coverage:       {coverage*100:.2f}%  (vs 原版 99.24%)")
    print()
    print(f"  对照实验: 原版 P10_5 Clover Purity 99.86% / Coverage 99.24%")
    print(f"           噪声版相比原版的下降即 SSI-EC 的提升空间")
    print()


if __name__ == '__main__':
    main()