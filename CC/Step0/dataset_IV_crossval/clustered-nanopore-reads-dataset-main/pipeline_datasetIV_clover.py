#!/usr/bin/env python3
"""
pipeline_datasetIV_clover.py
============================
dataset IV (Srinivasavaradhan 2021 / Microsoft CNR) 的 Clover baseline pipeline

定位:
    dataset IV = "又脏又小"的困难起点 (110bp, 269,709 reads, 9,984 GT簇,
    sub/ins/del≈1.7/2.0/2.2%, 且原作者声明GT有长程依赖缺陷)。
    作者补充材料明确: 所有聚类算法在 dataset IV 上都表现差。
    用途: 作为 SSI-EC "无论起点多差都能提升" 的 Clover 困难起点。

与 Seq_1D 版 (pipeline_seq1d_thin.py) 的差异:
    - 不打薄 (dataset IV 本就脏小簇, 平均27 reads/簇, 打薄会伤覆盖; 且对齐作者口径)
    - 数据源: 直接读 prep/01_gt_seq_to_tag.txt (已有 seq→tag), 不从output.txt预处理
    - REF_LEN=110, N_GT=9984
    - Clover 论文参数: D=15, V=3, H=3 (补充材料2.1节)
    - load_config: read_len=110, read_len_min=105, thd_tree_loc=53, four_tree_loc=92
      (152→110 比例缩放: 73*110/152≈53, 127*110/152≈92, 均<110不越界)
    - h_index/e_index: dataset IV 无 primer, 先用默认值跑, 崩则调

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Clover
    python pipeline_datasetIV_clover.py
    python pipeline_datasetIV_clover.py --skip_clover   # 复用已有结果
"""

import os
import re
import sys
import subprocess
import time
from collections import defaultdict, Counter

# ============================================================
# 配置
# ============================================================
CLOVER_DIR  = '/mnt/st_data/liangxinyi/code/CC/Step0/Clover'
PREP_DIR    = '/mnt/st_data/liangxinyi/code/CC/Step0/dataset_IV_crossval/prep'
CENTERS_TXT = '/mnt/st_data/liangxinyi/code/CC/Step0/dataset_IV_crossval/clustered-nanopore-reads-dataset-main/Centers.txt'
OUTPUT_DIR  = '/mnt/st_data/liangxinyi/code/CC/Step0/dataset_IV_crossval/clover_out'
EXPERIMENT_DIR = '/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/dataset_IV_clover'

GT_SEQ_TAG  = os.path.join(PREP_DIR, '01_gt_seq_to_tag.txt')   # tag\tread, 全部reads

REF_LEN     = 110
MIN_READS_PER_CLUSTER = 5
N_GT_TAGS   = 9984

# Clover 论文参数 (补充材料 2.1)
CLOVER_TREE_DEPTH = 15
CLOVER_V_DRIFT    = 3
CLOVER_H_DRIFT    = 3
# load_config 长度相关 (110bp 校正)
CFG_READ_LEN      = 110
CFG_READ_LEN_MIN  = 105
CFG_THD_TREE_LOC  = 40    # 73 * 110/152 ≈ 53
CFG_FOUR_TREE_LOC = 69    # 127 * 110/152 ≈ 92
CFG_H_INDEX       = 11    # dataset IV 无primer, 先用默认; 崩则调小
CFG_E_INDEX       = 11


def banner(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}\n")


# ============================================================
# Step 0: 读 prep (不打薄, 全部 reads)
# ============================================================
def step0_load_reads():
    """读 01_gt_seq_to_tag.txt (tag\tread) → cleaned [(tag, read), ...], 不打薄"""
    banner("Step 0  读取 dataset IV reads (不打薄)")
    cleaned = []
    with open(GT_SEQ_TAG) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 2:
                continue
            tag, seq = parts
            cleaned.append((tag, seq.upper()))
    n_tags = len(set(t for t, _ in cleaned))
    sizes = Counter(t for t, _ in cleaned)
    sz = sorted(sizes.values(), reverse=True)
    print(f"  输入:        {GT_SEQ_TAG}")
    print(f"  reads:       {len(cleaned):,}")
    print(f"  GT tags:     {n_tags:,}  (论文 dataset IV: 9,984)")
    print(f"  簇大小: max={sz[0]}, med={sz[len(sz)//2]}, min={sz[-1]}, "
          f"mean={len(cleaned)/n_tags:.1f}")
    print(f"  ⚠️ 不打薄, 不预过滤 (保持脏小簇原貌, 对齐作者口径)")
    return cleaned


# ============================================================
# Step 1: 写 Clover 输入 (idx read)
# ============================================================
def step1_write_clover_input(cleaned, clover_input_path, clover_tag_path):
    banner("Step 1  写 Clover 输入文件")
    idx_map = {}
    with open(clover_input_path, 'w') as f:
        for i, (tag, read) in enumerate(cleaned, start=1):
            f.write(f"{i} {read}\n")
            idx_map[i] = (tag, read)
    with open(clover_tag_path, 'w') as f:
        for tag, read in cleaned:
            f.write(f"{tag} {read}\n")
    print(f"  --no-tag 输入:  {clover_input_path}  ({len(idx_map):,} reads)")
    print(f"  tag 模式输入:   {clover_tag_path}")
    return idx_map


# ============================================================
# Step 2: 运行 Clover (论文参数 + 110bp config 校正)
# ============================================================
def step2_run_clover(clover_input_path, clover_output_base):
    banner("Step 2  运行 Clover (论文参数 D=15 V=3 H=3, 110bp校正)")

    config_path = os.path.join(CLOVER_DIR, 'clover', 'load_config.py')
    with open(config_path, 'r') as f:
        content = f.read()

    # patch load_config 的长度/索引参数 (110bp)
    patches = {
        'read_len':       CFG_READ_LEN,
        'read_len_min':   CFG_READ_LEN_MIN,
        'thd_tree_loc':   CFG_THD_TREE_LOC,
        'four_tree_loc':  CFG_FOUR_TREE_LOC,
        'h_index_nums':   CFG_H_INDEX,
        'e_index_nums':   CFG_E_INDEX,
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
        print(f"  ✅ 已更新 load_config.py (110bp):")
        for k, v in patches.items():
            print(f"     {k:18s} = {v}")
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
        alt = os.path.join(CLOVER_DIR, os.path.basename(clover_output_base) + ".txt")
        if os.path.exists(alt):
            os.rename(alt, result_path)
        else:
            raise FileNotFoundError(f"Clover 输出不存在: {result_path}")
    print(f"\n  ✅ Clover 完成, 耗时 {elapsed:.1f}s")
    return result_path


# ============================================================
# Step 3: 解析 Clover 输出
# ============================================================
def step3_parse_clover(clover_output_path):
    banner("Step 3  解析 Clover 输出")
    with open(clover_output_path) as f:
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


def step3_5_filter(cid_to_idxs, min_reads):
    banner(f"Step 3.5  过滤小簇 (reads < {min_reads})")
    before_n = len(cid_to_idxs)
    before_r = sum(len(v) for v in cid_to_idxs.values())
    filtered = {c: i for c, i in cid_to_idxs.items() if len(i) >= min_reads}
    after_n, after_r = len(filtered), sum(len(v) for v in filtered.values())
    print(f"  过滤前:  {before_n:,} 簇, {before_r:,} reads")
    print(f"  过滤后:  {after_n:,} 簇, {after_r:,} reads")
    print(f"  丢弃:    {before_n-after_n:,} 簇 ({before_r-after_r:,} reads)")
    return filtered


# ============================================================
# Step 4: 统计 (Purity/Coverage, 用 idx_map 回查GT)
# ============================================================
def step4_stats(cid_to_idxs, idx_map, min_reads, stats_path):
    banner("Step 4  聚类统计 (过滤后)")
    total_reads = sum(len(v) for v in cid_to_idxs.values())
    n_clusters = len(cid_to_idxs)
    sizes = sorted([len(v) for v in cid_to_idxs.values()], reverse=True)
    total_pure = 0
    gt_covered = set()
    for cid, idxs in cid_to_idxs.items():
        tags = [idx_map[idx][0] for idx in idxs if idx in idx_map]
        if not tags:
            continue
        maj_tag, maj_n = Counter(tags).most_common(1)[0]
        total_pure += maj_n
        gt_covered.add(maj_tag)
    purity = total_pure / max(total_reads, 1)
    coverage = len(gt_covered) / N_GT_TAGS
    print(f"  聚类簇数:   {n_clusters:,}")
    print(f"  GT tag 数:  {N_GT_TAGS:,}")
    print(f"  reads:      {total_reads:,}")
    print(f"  Purity:     {purity*100:.2f}%")
    print(f"  Coverage:   {coverage*100:.2f}%  ({len(gt_covered)}/{N_GT_TAGS})")
    if sizes:
        print(f"  簇大小: max={sizes[0]}, med={sizes[len(sizes)//2]}, min={sizes[-1]}")
    with open(stats_path, 'w') as f:
        f.write(f"dataset IV Clover 聚类统计 (不打薄+过滤<{min_reads})\n{'='*40}\n")
        f.write(f"簇数: {n_clusters}\nGT: {N_GT_TAGS}\nreads: {total_reads}\n")
        f.write(f"Purity: {purity*100:.2f}%\nCoverage: {coverage*100:.2f}%\n")
    print(f"  💾 {stats_path}")
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


def step56_write_output(cid_to_idxs, idx_map, read_path, ref_path):
    banner("Step 5+6  Majority Vote → read.txt + ref.txt")
    SEP = "=====分隔符=====\n"
    nc = nr = 0
    with open(read_path, 'w') as fr, open(ref_path, 'w') as ff:
        for cid in sorted(cid_to_idxs.keys()):
            reads = [idx_map[i][1] for i in cid_to_idxs[cid] if i in idx_map]
            if not reads:
                continue
            for r in reads:
                fr.write(r + '\n')
            fr.write(SEP)
            ff.write(majority_vote(reads, REF_LEN) + '\n')
            nc += 1
            nr += len(reads)
    print(f"  ✅ 簇数: {nc:,}, reads: {nr:,}")
    print(f"  read.txt: {read_path}")
    print(f"  ref.txt:  {ref_path}")


# ============================================================
# Step 7: 部署到实验目录 (对接 SSI-EC + eval)
# ============================================================
def step7_deploy(idx_map, clover_tag_path):
    banner("Step 7  部署到实验目录")
    feddna_dst = os.path.join(EXPERIMENT_DIR, '03_FedDNA_In')
    os.makedirs(feddna_dst, exist_ok=True)
    import shutil
    src_dir = os.path.join(OUTPUT_DIR, '04_FedDNA_In')
    for fname in ['read.txt', 'ref.txt']:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(feddna_dst, fname))
        print(f"  ✅ {fname} → {feddna_dst}")

    # GT tags: tag\tread (eval 和 SSI-EC 用)
    gt_tags_path = os.path.join(EXPERIMENT_DIR, 'datasetIV_tags_reads.txt')
    with open(clover_tag_path) as fin, open(gt_tags_path, 'w') as fout:
        for line in fin:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                fout.write(f"{parts[0]}\t{parts[1]}\n")
    print(f"  ✅ GT tags: {gt_tags_path}")

    # GT refs: Centers.txt (dataset IV 真值, 110bp, 10000条)
    gt_refs_path = os.path.join(EXPERIMENT_DIR, 'datasetIV_refs.txt')
    shutil.copy2(CENTERS_TXT, gt_refs_path)
    print(f"  ✅ GT refs: {gt_refs_path}  (Centers.txt)")
    print()
    print(f"  🚀 接 SSI-EC 三轮 (注意 --max_length 110):")
    print(f"     cd /mnt/st_data/liangxinyi/code/models")
    print(f"     python main_loop.py \\")
    print(f"       --experiment_dir {EXPERIMENT_DIR}/ \\")
    print(f"       --feddna_checkpoint /mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth \\")
    print(f"       --max_iterations 3 --max_length 110 --cl_mode ours \\")
    print(f"       --gt_tags_file {gt_tags_path} \\")
    print(f"       --gt_refs_file {gt_refs_path} \\")
    print(f"       2>&1 | tee datasetIV_clover_v1.log")


def main():
    import argparse
    ap = argparse.ArgumentParser(description='dataset IV Clover pipeline (不打薄)')
    ap.add_argument('--skip_clover', action='store_true')
    ap.add_argument('--min_reads', type=int, default=MIN_READS_PER_CLUSTER)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, '04_FedDNA_In'), exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    clover_input     = os.path.join(OUTPUT_DIR, '01_clover_input.txt')
    clover_tag_input = os.path.join(OUTPUT_DIR, '01_clover_tag_input.txt')
    clover_out_base  = os.path.join(OUTPUT_DIR, '02_clover_result')
    clover_out_txt   = clover_out_base + '.txt'
    stats_path       = os.path.join(OUTPUT_DIR, '03_stats.txt')
    read_path        = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'read.txt')
    ref_path         = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'ref.txt')

    print("="*60)
    print("  🚀  dataset IV Pipeline (Clover baseline, 不打薄)")
    print("="*60)
    print(f"  ref_len: {REF_LEN}bp,  Clover: D={CLOVER_TREE_DEPTH} V={CLOVER_V_DRIFT} H={CLOVER_H_DRIFT}")

    t_start = time.time()
    cleaned = step0_load_reads()
    idx_map = step1_write_clover_input(cleaned, clover_input, clover_tag_input)
    del cleaned

    if not args.skip_clover:
        clover_out_txt = step2_run_clover(clover_input, clover_out_base)
    else:
        banner("Step 2  跳过 Clover")
        if not os.path.exists(clover_out_txt):
            raise FileNotFoundError(f"不存在: {clover_out_txt}")

    cid_to_idxs = step3_parse_clover(clover_out_txt)
    cid_to_idxs = step3_5_filter(cid_to_idxs, args.min_reads)
    purity, coverage = step4_stats(cid_to_idxs, idx_map, args.min_reads, stats_path)
    step56_write_output(cid_to_idxs, idx_map, read_path, ref_path)
    step7_deploy(idx_map, clover_tag_input)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\n  🎉 完成! 耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Purity: {purity*100:.2f}%,  Coverage: {coverage*100:.2f}%\n{'='*60}")


if __name__ == '__main__':
    main()