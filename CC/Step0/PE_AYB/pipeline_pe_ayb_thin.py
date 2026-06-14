#!/usr/bin/env python3
"""
pipeline_pe_ayb_thin.py
=======================
PE_AYB 数据集 Pipeline（打薄版）

  output.txt → 预处理 → 打薄 → Clover 聚类 → 小簇过滤 → 统计 → read.txt + ref.txt

数据集差异 (vs Seq_1D):
  REF_LEN:        196 → 117      reads 主流长度（无 adapter）
  GT_REF_LEN:     196 → 117      ref payload 抽取后长度
  RAW_REF_LEN:    —  → 183       ref.fasta 原始长度
  PAYLOAD_START:  0  → 33        Illumina adapter 长度（reads 对齐 ref[33:150]）
  N_GT_TAGS:      11826 → 153331
  h_index_nums:   20 → 0         reads 已无 adapter, 无固定锚点
  e_index_nums:   20 → 0
  thd_tree_loc:   72 → 42        36% × 117
  four_tree_loc:  124 → 74       63% × 117

★ 关键差异: reads 是 ref payload (33:150) 的测序读，不是 ref 完整 183bp。
  Step 7 生成 GT refs 时必须抽取 ref[33:150]，否则 SSI-EC SR 评估会全错。

注意:
  ‣ reads 没有 primer/adapter 锚点，Clover 难度比 Seq_1D/P10_5 大
  ‣ 这正是 SSI-EC 能体现优势的场景（Coverage 预期 < P10_5）
  ‣ SSI-EC 主代码: --max_length 201 --ref_length 117

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/PE_AYB
    python pipeline_pe_ayb_thin.py                          # 完整运行
    python pipeline_pe_ayb_thin.py --max_reads_per_tag 15   # 更激进打薄
    python pipeline_pe_ayb_thin.py --skip_clover            # 跳过 Clover
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
BASE_DIR    = '/mnt/st_data/liangxinyi/code/CC/Step0/PE_AYB'
CLOVER_DIR  = '/mnt/st_data/liangxinyi/code/CC/Step0/Clover'
OUTPUT_DIR  = os.path.join(BASE_DIR, 'clover_out')
EXPERIMENT_DIR = '/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/pe_ayb'

# ── 数据集核心常量 ──
REF_LEN     = 117            # reads 主流长度（无 adapter）
LEN_MIN     = 115            # 覆盖 95%+ reads
LEN_MAX     = 120
GT_REF_LEN  = 117            # ref payload 抽取后长度
RAW_REF_LEN = 183            # ref.fasta 原始长度
PAYLOAD_START = 33           # Illumina adapter 长度

MIN_READS_PER_CLUSTER = 5
N_GT_TAGS   = 153331         # PE_AYB mapped refs (Seq_1D: 11826)
MAX_READS_PER_TAG = 30
RANDOM_SEED = 42

# Clover 参数
CLOVER_TREE_DEPTH = 20
CLOVER_V_DRIFT    = 3
CLOVER_H_DRIFT    = 3
CLOVER_H_INDEX    = 0        # ★ reads 无 adapter, 无锚点
CLOVER_E_INDEX    = 0
CLOVER_THD_LOC    = 42       # 36% × 117
CLOVER_FOUR_LOC   = 74       # 63% × 117


def banner(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


# ============================================================
# Step 0: 预处理 + 打薄
# ============================================================
def step0_preprocess_and_thin(input_file, max_reads_per_tag):
    banner("Step 0  预处理 + 打薄")

    total = 0
    n_dropped = 0
    len_dropped = 0
    tag_to_reads = defaultdict(list)

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
    print(f"  GT ref 长度:         {GT_REF_LEN}bp (从 ref[{PAYLOAD_START}:{PAYLOAD_START+GT_REF_LEN}] 抽取)")
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
        f.write(f"PE_AYB 聚类统计（打薄 + 小簇过滤）\n{'='*40}\n\n")
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
# Step 7: 部署到实验目录（★ 关键：抽取 ref payload）
# ============================================================
def step7_deploy(idx_map, clover_tag_path):
    """复制 read.txt/ref.txt + 生成 GT tags + 生成 GT refs (抽取 ref[33:150] payload)。"""
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

    # GT tags
    gt_tags_path = os.path.join(EXPERIMENT_DIR, 'pe_ayb_tags_reads.txt')
    with open(clover_tag_path, 'r') as fin, open(gt_tags_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                fout.write(f"{parts[0]}\t{parts[1]}\n")

    print(f"  ✅ GT tags: {gt_tags_path}")

    # ★ GT refs: 必须从 ref.fasta 抽取 [33:150] 的 117bp payload，不是完整 183bp
    refs_fasta = os.path.join(BASE_DIR, 'ref.fasta')
    gt_refs_path = os.path.join(EXPERIMENT_DIR, 'pe_ayb_refs.txt')
    n_refs_written = 0
    n_refs_skipped = 0
    payload_end = PAYLOAD_START + GT_REF_LEN  # 33 + 117 = 150

    with open(refs_fasta, 'r') as fin, open(gt_refs_path, 'w') as fout:
        cur_seq_parts = []
        for line in fin:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if cur_seq_parts:
                    full_seq = ''.join(cur_seq_parts)
                    if len(full_seq) >= payload_end:
                        fout.write(full_seq[PAYLOAD_START:payload_end] + '\n')
                        n_refs_written += 1
                    else:
                        # 兜底：长度不够时写空行保持行号对齐
                        fout.write('\n')
                        n_refs_skipped += 1
                cur_seq_parts = []
            else:
                cur_seq_parts.append(line)
        # 最后一条
        if cur_seq_parts:
            full_seq = ''.join(cur_seq_parts)
            if len(full_seq) >= payload_end:
                fout.write(full_seq[PAYLOAD_START:payload_end] + '\n')
                n_refs_written += 1
            else:
                fout.write('\n')
                n_refs_skipped += 1

    print(f"  ✅ GT refs: {gt_refs_path}")
    print(f"     抽取范围:    ref[{PAYLOAD_START}:{payload_end}]  ({GT_REF_LEN}bp)")
    print(f"     成功 refs:   {n_refs_written:,}")
    if n_refs_skipped > 0:
        print(f"     长度不足:    {n_refs_skipped:,}  (写空行保持行号对齐)")

    print()
    print(f"  🚀 运行实验:")
    print(f"     cd /mnt/st_data/liangxinyi/code/models")
    print(f"     python main_loop.py \\")
    print(f"       --experiment_dir {EXPERIMENT_DIR}/ \\")
    print(f"       --feddna_checkpoint /mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth \\")
    print(f"       --max_iterations 3 --max_length 201 --ref_length {REF_LEN} --cl_mode ours \\")
    print(f"       --gt_tags_file {gt_tags_path} \\")
    print(f"       --gt_refs_file {gt_refs_path} \\")
    print(f"       2>&1 | tee pe_ayb_ours_v1.log")
    print()
    print(f"  ⚠️  GT refs 已抽取 payload (117bp)，与 reads/MV 同长，无需在主代码再做对齐")


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='PE_AYB Pipeline（打薄版）')
    parser.add_argument('--skip_clover', action='store_true')
    parser.add_argument('--max_reads_per_tag', type=int, default=MAX_READS_PER_TAG)
    parser.add_argument('--min_reads', type=int, default=MIN_READS_PER_CLUSTER)
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
    print("  🚀  PE_AYB Pipeline（打薄版）")
    print("=" * 60)
    print()
    print(f"  数据目录:         {BASE_DIR}")
    print(f"  输出目录:         {OUTPUT_DIR}")
    print(f"  实验目录:         {EXPERIMENT_DIR}")
    print()
    print(f"  ref_len (reads):  {REF_LEN}bp")
    print(f"  ref_len (GT):     {GT_REF_LEN}bp (从 ref[{PAYLOAD_START}:{PAYLOAD_START+GT_REF_LEN}] 抽取)")
    print(f"  raw ref:          {RAW_REF_LEN}bp (含 adapter)")
    print(f"  N_GT_TAGS:        {N_GT_TAGS:,}")
    print(f"  max_reads/tag:    {args.max_reads_per_tag}")
    print(f"  min_reads/簇:     {args.min_reads}")
    print(f"  Clover:           D={CLOVER_TREE_DEPTH} V={CLOVER_V_DRIFT} H={CLOVER_H_DRIFT}")
    print(f"                    h_idx={CLOVER_H_INDEX} e_idx={CLOVER_E_INDEX} "
          f"thd={CLOVER_THD_LOC} four={CLOVER_FOUR_LOC}")

    t_start = time.time()

    cleaned = step0_preprocess_and_thin(input_file, args.max_reads_per_tag)
    if not cleaned:
        print("\n  ❌ Step 0 后无 reads，终止。")
        sys.exit(1)

    idx_map = step1_write_clover_input(cleaned, clover_input, clover_tag_input)
    del cleaned

    if not args.skip_clover:
        clover_out_txt = step2_run_clover(clover_input, clover_out_base)
    else:
        banner("Step 2  跳过 Clover")
        print(f"  使用: {clover_out_txt}")

    cid_to_idxs = step3_parse_clover(clover_out_txt)
    cid_to_idxs = step3_5_filter_small_clusters(cid_to_idxs, args.min_reads)
    purity, coverage = step4_statistics(cid_to_idxs, idx_map, args.min_reads, stats_path)
    step56_write_output(cid_to_idxs, idx_map, read_path, ref_path)
    step7_deploy(idx_map, clover_tag_input)

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print("  🎉  Pipeline 完成!")
    print("=" * 60)
    print()
    print(f"  总耗时:         {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Purity:         {purity*100:.2f}%")
    print(f"  Coverage:       {coverage*100:.2f}%")
    print()


if __name__ == '__main__':
    main()