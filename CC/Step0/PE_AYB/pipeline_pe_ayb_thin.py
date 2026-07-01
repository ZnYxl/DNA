#!/usr/bin/env python3
"""
pipeline_pe_ayb_thin.py
=======================
PE_AYB 数据集 Pipeline（全局随机打薄版）

  output.txt → 预处理 → 全局随机打薄(keep_ratio) → 分布保持验证 → Clover 聚类 → 小簇过滤 → 统计 → read.txt + ref.txt

打薄策略（全局随机采样，无 GT 先验）:
  对过滤后的全体 reads 做一次全局随机采样，精确保留 keep_ratio 比例。
  不按 tag/簇分组、不设保底、不看任何 label —— 与真实"无 GT 测序数据"场景一致。
  打薄后整体冗余分布形状保持不变（仅整体缩放），由 thinning_verify 模块量化验证。

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

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/PE_AYB
    python pipeline_pe_ayb_thin.py --keep_ratio 0.1     # 默认 1/10
    python pipeline_pe_ayb_thin.py --keep_ratio 0.05
    python pipeline_pe_ayb_thin.py --keep_ratio 0.025
    python pipeline_pe_ayb_thin.py --skip_clover        # 跳过 Clover
"""

import os
import re
import sys
import subprocess
import time
import random
from collections import defaultdict, Counter

# 分布保持验证模块（公共模块，位于 CC/Step0/）
sys.path.insert(0, '/mnt/st_data/liangxinyi/code/CC/Step0')
from thinning_verify import verify_distribution_preserved


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
BASE_DIR    = '/mnt/st_data/liangxinyi/code/CC/Step0/PE_AYB'
CLOVER_DIR  = '/mnt/st_data/liangxinyi/code/CC/Step0/Clover'
OUTPUT_DIR  = None   # 运行时根据 keep_ratio 设置
EXPERIMENT_DIR = None  # 运行时根据 keep_ratio 设置

# ── 数据集核心常量 ──
REF_LEN     = 117            # reads 主流长度（无 adapter）
LEN_MIN     = 115
LEN_MAX     = 120
GT_REF_LEN  = 117            # ref payload 抽取后长度
RAW_REF_LEN = 183            # ref.fasta 原始长度
PAYLOAD_START = 33           # Illumina adapter 长度

MIN_READS_PER_CLUSTER = 5
N_GT_TAGS   = 153331         # PE_AYB mapped refs
KEEP_RATIO  = 0.1            # 全局随机打薄：保留比例（默认 1/10）
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
# Step 0: 预处理 + 全局随机打薄 + 分布保持验证
# ============================================================
def step0_preprocess_and_thin(input_file, keep_ratio, png_path):
    """
    去N + 长度过滤 → 对全体 reads 做一次全局随机采样，精确保留 keep_ratio。
    不分组、不保底、不看任何 label。tag 随 read 保留（仅供下游写 GT 文件 / 验证用）。
    """
    banner("Step 0  预处理 + 全局随机打薄")

    total = 0
    n_dropped = 0
    len_dropped = 0
    all_reads = []  # [(tag, seq), ...] 过滤后全体（不分组）

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

            all_reads.append((tag, seq))

    after_filter = len(all_reads)
    n_tags_before = len(set(t for t, _ in all_reads))

    # ── 全局随机打薄：精确保留 keep_ratio ──
    rng = random.Random(RANDOM_SEED)
    n_keep = int(round(after_filter * keep_ratio))
    cleaned = rng.sample(all_reads, n_keep)
    thinned_reads = after_filter - n_keep
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
    print(f"  ── 全局随机打薄 ──")
    print(f"  keep_ratio:          {keep_ratio}  (1/{1/keep_ratio:.1f})")
    print(f"  random_seed:         {RANDOM_SEED}")
    print(f"  打薄丢弃:            {thinned_reads:,} 条 reads")
    print(f"  打薄后 reads:        {len(cleaned):,}  ({n_tags_after:,} tags)")
    lost_tags = n_tags_before - n_tags_after
    print(f"  随机抽稀导致 {lost_tags:,} 个低冗余 tag 完全丢失（真实低采样的自然结果）")

    # ── 分布保持验证（五项证据 + PNG）──
    verify_distribution_preserved(
        all_reads, cleaned, keep_ratio,
        dataset_name="PE_AYB", png_path=png_path,
    )

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
        f.write(f"PE_AYB 聚类统计（全局随机打薄 + 小簇过滤）\n{'='*40}\n\n")
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
def step7_deploy(idx_map, clover_tag_path, keep_ratio):
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
                        fout.write('\n')
                        n_refs_skipped += 1
                cur_seq_parts = []
            else:
                cur_seq_parts.append(line)
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
    ratio_tag = f"p{keep_ratio}"
    log_name = f"pe_ayb_{ratio_tag}_verify.log"
    exp_rel = f"CC/Step0/Experiments/pe_ayb_{ratio_tag}"
    run_cmd = (
        f"cd /mnt/st_data/liangxinyi/code\n"
        f"python -m models.main_loop \\\n"
        f"    --experiment_dir {exp_rel}/ \\\n"
        f"    --feddna_checkpoint result/FLDNA_I/I_1214234233/model/epoch1_I.pth \\\n"
        f"    --gt_tags_file {exp_rel}/pe_ayb_tags_reads.txt \\\n"
        f"    --gt_refs_file {exp_rel}/pe_ayb_refs.txt \\\n"
        f"    --max_iterations 3 --max_length 201 \\\n"
        f"    --cl_mode ours --ref_length {REF_LEN} --primer_prefix 0 --primer_suffix 0 \\\n"
        f"    --split_tau 5 --split_min_size 6 \\\n"
        f"    2>&1 | tee {exp_rel}/{log_name}"
    )
    print(f"  🚀 运行实验:")
    for ln in run_cmd.split("\n"):
        print(f"     {ln}")

    # ── 评估命令（实验跑完后执行）──
    # ★ gt_refs 必须用抽取 payload 后的 pe_ayb_refs.txt（117bp），不是原始 183bp ref.fasta
    eval_cmd = (
        f"cd /mnt/st_data/liangxinyi/code/models\n"
        f"python eval_reconstruction.py \\\n"
        f"    --experiment_dir /mnt/st_data/liangxinyi/code/{exp_rel}/ \\\n"
        f"    --gt_refs /mnt/st_data/liangxinyi/code/{exp_rel}/pe_ayb_refs.txt \\\n"
        f"    --gt_tags /mnt/st_data/liangxinyi/code/{exp_rel}/pe_ayb_tags_reads.txt \\\n"
        f"    --out reconstruction_eval_pe_ayb_{ratio_tag}.tsv"
    )
    print()
    print(f"  📊 评估命令（实验跑完后执行）:")
    for ln in eval_cmd.split("\n"):
        print(f"     {ln}")
    print()
    print(f"  ⚠️  GT refs 已抽取 payload (117bp)，与 reads/MV 同长，无需在主代码再做对齐")
    print(f"  ⚠️  primer_prefix/suffix 设为 0：reads 无 adapter，payload 即全长")


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='PE_AYB Pipeline（全局随机打薄版）')
    parser.add_argument('--skip_clover', action='store_true')
    parser.add_argument('--keep_ratio', type=float, default=KEEP_RATIO,
                        help=f'全局随机打薄保留比例（默认 {KEEP_RATIO} = 1/10）')
    parser.add_argument('--min_reads', type=int, default=MIN_READS_PER_CLUSTER)
    args = parser.parse_args()

    # ── 根据 keep_ratio 生成目录后缀，多个 p 完全对称、互不覆盖 ──
    global OUTPUT_DIR, EXPERIMENT_DIR
    ratio_tag = f"p{args.keep_ratio}"
    OUTPUT_DIR = os.path.join(BASE_DIR, f'clover_out_{ratio_tag}')
    EXPERIMENT_DIR = f'/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/pe_ayb_{ratio_tag}'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, '04_FedDNA_In'), exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    # ── 开启 Tee：全程输出同时写实验目录 data_detail.txt ──
    detail_path = os.path.join(EXPERIMENT_DIR, 'data_detail.txt')
    _detail_fh = open(detail_path, 'w')
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _detail_fh)
    print("=" * 60)
    print("  运行命令:")
    print(f"    python {os.path.abspath(__file__)} --keep_ratio {args.keep_ratio}"
          + (f" --min_reads {args.min_reads}" if args.min_reads != MIN_READS_PER_CLUSTER else "")
          + (" --skip_clover" if args.skip_clover else ""))
    print(f"  日志文件: {detail_path}")
    print("=" * 60)

    input_file       = os.path.join(BASE_DIR, 'output.txt')
    clover_input     = os.path.join(OUTPUT_DIR, '01_clover_input.txt')
    clover_tag_input = os.path.join(OUTPUT_DIR, '01_clover_tag_input.txt')
    clover_out_base  = os.path.join(OUTPUT_DIR, '02_clover_result')
    clover_out_txt   = clover_out_base + '.txt'
    stats_path       = os.path.join(OUTPUT_DIR, '03_stats.txt')
    read_path        = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'read.txt')
    ref_path         = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'ref.txt')
    dist_png         = os.path.join(EXPERIMENT_DIR, f'dist_preserve_{ratio_tag}.png')

    print()
    print("=" * 60)
    print("  🚀  PE_AYB Pipeline（全局随机打薄版）")
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
    print(f"  keep_ratio:       {args.keep_ratio}  ← 全局随机打薄参数")
    print(f"  min_reads/簇:     {args.min_reads}")
    print(f"  Clover:           D={CLOVER_TREE_DEPTH} V={CLOVER_V_DRIFT} H={CLOVER_H_DRIFT}")
    print(f"                    h_idx={CLOVER_H_INDEX} e_idx={CLOVER_E_INDEX} "
          f"thd={CLOVER_THD_LOC} four={CLOVER_FOUR_LOC}")

    t_start = time.time()

    cleaned = step0_preprocess_and_thin(input_file, args.keep_ratio, dist_png)
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
    step7_deploy(idx_map, clover_tag_input, args.keep_ratio)

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print("  🎉  Pipeline 完成!")
    print("=" * 60)
    print()
    print(f"  总耗时:         {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  打薄:           全局随机保留 {args.keep_ratio} (seed={RANDOM_SEED})")
    print(f"  Purity:         {purity*100:.2f}%")
    print(f"  Coverage:       {coverage*100:.2f}%")
    print()

    # ── 关闭 Tee ──
    sys.stdout = _orig_stdout
    _detail_fh.close()


if __name__ == '__main__':
    main()