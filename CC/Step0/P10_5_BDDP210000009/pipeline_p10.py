#!/usr/bin/env python3
"""
pipeline_p10.py
===============
P10_5_BDDP210000009 专用 Pipeline（打薄版）

  output.txt → 预处理 → 打薄(每tag最多N条) → Clover聚类 → 小簇过滤 → 统计 → read.txt + ref.txt → 部署

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/P10_5_BDDP210000009
    python pipeline_p10.py                          # 完整运行（默认每tag最多30条）
    python pipeline_p10.py --max_reads_per_tag 15   # 更激进打薄
    python pipeline_p10.py --skip_clover            # 跳过 Clover，复用已有结果

═══════════════════════════════════════════════════════════════════
P10 vs Seq_1D 关键差异（基于服务器实测 + spike 验证，全部已确认）:
═══════════════════════════════════════════════════════════════════
  参数            Seq_1D    P10        依据
  ─────────────────────────────────────────────────────────────────
  REF_LEN         196       200        主峰长度(1385万条/200bp)
  N_GT_TAGS       11826     209185     实测唯一 tag 数(= 对照表"比对到读取序列"栏)
  h_index         20        24         前导引物 CCTGCAGAGTAGCATGTCATTGAT 保守24bp
  e_index         20        18         尾部引物 CTGACACTGATGCATCCG = 18bp
  thd_tree_loc    72        50         相对 payload(158bp)前1/3
  four_tree_loc   124       50         相对尾部偏移(中段树占比<0.3%,无关紧要)
  GT refs 来源    reads.fasta ref.fasta tag = ref.fasta header 数字(精确对应)
  GT 构建方式     行号顺序   tag→序列字典  ref.fasta 210000条,只取比对上的tag

Clover 中段树 bug（已 spike 验证，不修，归档）:
  insert 用未偏移坐标、检索用 h_index 偏移坐标,h_index=24 时错位24bp。
  实测碱基一致率 25%(完全随机),中段树彻底失效。
  但 P10 上中段树命中占比 <0.3%(a_tree 命中 82%),实测影响可忽略,
  故标记为 true bug 但归档不修(改源码引入复现风险)。供 I16 等中段树占比高的数据集参考。

打薄策略（与 FedDNA 论文一致）:
  FedDNA: "for each DNA cluster, 5 to 30 reads were randomly sampled"
  本脚本对每个 BWA tag 随机采样最多 max_reads_per_tag 条 reads
  P10 全量平均 72.8 read/tag(中位71),冗余充足,打薄到30后聚类变难 → SSI-EC 有提升空间
═══════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import subprocess
import time
import random
from collections import defaultdict, Counter

# ============================================================
# 配置  ★ = P10 相对 Seq_1D 的改动
# ============================================================
BASE_DIR    = '/mnt/st_data/liangxinyi/code/CC/Step0/P10_5_BDDP210000009'   # ★
CLOVER_DIR  = '/mnt/st_data/liangxinyi/code/CC/Step0/Clover'
OUTPUT_DIR  = os.path.join(BASE_DIR, 'clover_out')
EXPERIMENT_DIR = '/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/p10'    # ★

REF_LEN     = 200            # ★ Seq_1D=196
LEN_MIN     = REF_LEN - 5    # 195
LEN_MAX     = REF_LEN + 5    # 205
MIN_READS_PER_CLUSTER = 5
N_GT_TAGS   = 209185         # ★ 实测唯一 tag 数（Seq_1D=11826）
MAX_READS_PER_TAG = 30
RANDOM_SEED = 42

# Clover 参数
CLOVER_TREE_DEPTH = 20
CLOVER_V_DRIFT    = 3
CLOVER_H_DRIFT    = 3
CLOVER_H_INDEX    = 24       # ★ 前导引物 24bp（Seq_1D=20）
CLOVER_E_INDEX    = 18       # ★ 尾部引物 18bp（Seq_1D=20）
CLOVER_THD_LOC    = 50       # ★ 相对 payload（Seq_1D=72）
CLOVER_FOUR_LOC   = 50       # ★ 相对尾部（Seq_1D=124）

# GT 来源  ★ P10 用 ref.fasta（带 header），按 tag 取序列
REF_FASTA   = os.path.join(BASE_DIR, 'ref.fasta')


def banner(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


# ============================================================
# Step 0: 预处理 + 打薄
# ============================================================
def step0_preprocess_and_thin(input_file, max_reads_per_tag):
    """读 output.txt → 去N + 长度过滤 → 按 tag 分组 → 每 tag 随机采样 ≤ max_reads_per_tag。"""
    banner("Step 0  预处理 + 打薄")

    total = n_dropped = len_dropped = 0
    tag_to_reads = defaultdict(list)

    with open(input_file, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
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

    # tag 是纯数字字符串，按数值排序保证可复现
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
    print(f"  💡 验证 accuracy（tag mode，打薄后）:")
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

    # ── 备份（幂等：已存在则不覆盖原始备份）──
    bak_path = config_path + '.bak_p10'
    if not os.path.exists(bak_path):
        import shutil
        shutil.copy2(config_path, bak_path)
        print(f"  ✅ 备份: {bak_path}")

    with open(config_path, 'r') as f:
        content = f.read()

    patches = {
        'h_index_nums':  CLOVER_H_INDEX,    # ★ 24
        'e_index_nums':  CLOVER_E_INDEX,    # ★ 18
        'thd_tree_loc':  CLOVER_THD_LOC,    # ★ 50
        'four_tree_loc': CLOVER_FOUR_LOC,   # ★ 50
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

    # 改后 grep 验证
    with open(config_path, 'r') as f:
        verify = f.read()
    for key, val in patches.items():
        m = re.search(rf'"{key}"\s*:\s*(\d+)', verify)
        got = m.group(1) if m else '?'
        flag = '✓' if got == str(val) else '✗ 不一致!'
        print(f"     验证 {key} = {got}  {flag}")
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

    print(f"  Clover: -L {REF_LEN} -D {CLOVER_TREE_DEPTH} -V {CLOVER_V_DRIFT} -H {CLOVER_H_DRIFT} (h={CLOVER_H_INDEX} e={CLOVER_E_INDEX})")
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
        f.write(f"P10 聚类统计（打薄 + 小簇过滤）\n{'='*40}\n\n")
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
# Step 7: 部署到实验目录
# ★ P10 GT refs 改造：tag = ref.fasta header，按 tag 取序列（绝不按行号）
# ============================================================
def load_ref_fasta(ref_fasta_path):
    """读 ref.fasta → {header_tag: sequence} 字典。"""
    d = {}
    cur_tag = None
    buf = []
    with open(ref_fasta_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if cur_tag is not None:
                    d[cur_tag] = ''.join(buf)
                cur_tag = line[1:].strip()   # 去掉 '>'
                buf = []
            else:
                buf.append(line)
        if cur_tag is not None:
            d[cur_tag] = ''.join(buf)
    return d


def step7_deploy(idx_map, clover_tag_path):
    """复制 read.txt/ref.txt 到实验目录，生成 GT tags + GT refs（按 tag→序列字典）。"""
    banner("Step 7  部署到实验目录")

    import shutil
    feddna_src = os.path.join(OUTPUT_DIR, '04_FedDNA_In')
    feddna_dst = os.path.join(EXPERIMENT_DIR, '03_FedDNA_In')
    os.makedirs(feddna_dst, exist_ok=True)

    for fname in ['read.txt', 'ref.txt']:
        src = os.path.join(feddna_src, fname)
        dst = os.path.join(feddna_dst, fname)
        shutil.copy2(src, dst)
        print(f"  ✅ {src} → {dst}")

    # ── GT tags: tag 模式输入(空格分隔) → tab 分隔 ──
    gt_tags_path = os.path.join(EXPERIMENT_DIR, 'p10_tags_reads.txt')
    n_gt_lines = 0
    with open(clover_tag_path, 'r') as fin, open(gt_tags_path, 'w') as fout:
        for line in fin:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                fout.write(f"{parts[0]}\t{parts[1]}\n")
                n_gt_lines += 1
    print(f"  ✅ GT tags: {gt_tags_path}  ({n_gt_lines:,} 行)")

    # ── GT refs: ★ 按 tag → ref.fasta 序列字典构建 ──
    # 只导出"实际出现在打薄数据里"的 tag，且按 tag 数值排序，保证可复现
    ref_dict = load_ref_fasta(REF_FASTA)
    print(f"  ref.fasta 载入: {len(ref_dict):,} 条 design")

    used_tags = sorted(set(t for t, _ in idx_map.values()), key=lambda x: int(x))
    gt_refs_path = os.path.join(EXPERIMENT_DIR, 'p10_refs.txt')
    gt_tag_order_path = os.path.join(EXPERIMENT_DIR, 'p10_ref_tag_order.txt')

    n_written = n_missing = 0
    with open(gt_refs_path, 'w') as fout, open(gt_tag_order_path, 'w') as ftag:
        for tag in used_tags:
            if tag in ref_dict:
                fout.write(ref_dict[tag] + '\n')
                ftag.write(tag + '\n')   # 与 refs 行号一一对应，供下游按序列/tag 对齐
                n_written += 1
            else:
                n_missing += 1

    print(f"  ✅ GT refs: {gt_refs_path}  ({n_written:,} 条)")
    print(f"  ✅ GT ref tag 顺序: {gt_tag_order_path}  (与 refs 行号一一对应)")
    if n_missing:
        print(f"  ⚠️  {n_missing:,} 个 tag 在 ref.fasta 中找不到（异常，应为0，请检查）")

    print()
    print(f"  🚀 运行 SSI-EC 实验:")
    print(f"     cd /mnt/st_data/liangxinyi/code/models")
    print(f"     python main_loop.py \\")
    print(f"       --experiment_dir {EXPERIMENT_DIR}/ \\")
    print(f"       --feddna_checkpoint /mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth \\")
    print(f"       --max_iterations 3 --max_length {REF_LEN} --cl_mode ours \\")
    print(f"       --gt_tags_file {gt_tags_path} \\")
    print(f"       --gt_refs_file {gt_refs_path} \\")
    print(f"       2>&1 | tee p10_ours_v1.log")
    print()
    print(f"  注: epoch1_I.pth 训练于 ID20 150bp，length_adapter 权重会因长度不符")
    print(f"      被 strict=False 跳过——P10=200bp，与你 memory 里记录的处理方式一致。")


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='P10 Pipeline（打薄版）')
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
    print("  🚀  P10 Pipeline（打薄版）")
    print("=" * 60)
    print()
    print(f"  数据目录:         {BASE_DIR}")
    print(f"  输出目录:         {OUTPUT_DIR}")
    print(f"  实验目录:         {EXPERIMENT_DIR}")
    print()
    print(f"  ref_len:          {REF_LEN}bp")
    print(f"  N_GT_TAGS:        {N_GT_TAGS:,}")
    print(f"  max_reads/tag:    {args.max_reads_per_tag}  ← 打薄参数")
    print(f"  min_reads/簇:     {args.min_reads}")
    print(f"  Clover:           D={CLOVER_TREE_DEPTH} V={CLOVER_V_DRIFT} H={CLOVER_H_DRIFT} "
          f"h={CLOVER_H_INDEX} e={CLOVER_E_INDEX}")

    t_start = time.time()

    cleaned = step0_preprocess_and_thin(input_file, args.max_reads_per_tag)
    idx_map = step1_write_clover_input(cleaned, clover_input, clover_tag_input)
    del cleaned

    if not args.skip_clover:
        clover_out_txt = step2_run_clover(clover_input, clover_out_base)
    else:
        banner("Step 2  跳过 Clover")
        print(f"  使用: {clover_out_txt}")
        if not os.path.exists(clover_out_txt):
            raise FileNotFoundError(f"不存在: {clover_out_txt}")

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
    print(f"  打薄:           每 tag ≤ {args.max_reads_per_tag} reads")
    print(f"  Purity:         {purity*100:.2f}%")
    print(f"  Coverage:       {coverage*100:.2f}%")
    print()


if __name__ == '__main__':
    main()