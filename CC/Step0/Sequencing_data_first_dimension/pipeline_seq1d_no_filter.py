#!/usr/bin/env python3
"""
pipeline_seq1d_no_filter.py
=================
Sequencing_data_first_dimension 专用 Pipeline

  output.txt → 预处理 → Clover聚类 → 统计 → Majority Vote → read.txt + ref.txt

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension
    python pipeline_seq1d.py                    # 完整运行
    python pipeline_seq1d.py --skip_clover      # 跳过 Clover，复用已有结果

数据集参数:
    ref_len        = 196bp
    长度过滤       = [191, 201]
    引物           = 无（但 Clover 需设 h_index=20, e_index=20 跳过共享前后缀）
    Clover 参数    = -D 20 -V 3 -H 3

输出:
    clover_out/
    ├── 01_clover_input.txt         ← Clover 输入 (idx read, --no-tag)
    ├── 01_clover_tag_input.txt     ← Clover 验证 (tag read, tag模式)
    ├── 02_clover_result.txt        ← Clover 原始输出
    ├── 03_stats.txt                ← 聚类统计
    └── 04_FedDNA_In/
        ├── read.txt                ← SSI-EC 输入 (簇间 ===== 分隔)
        └── ref.txt                 ← Majority Vote 伪 ref
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
BASE_DIR    = '/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension'
CLOVER_DIR  = '/mnt/st_data/liangxinyi/code/CC/Step0/Clover'
OUTPUT_DIR  = os.path.join(BASE_DIR, 'clover_out')

REF_LEN     = 196
LEN_MIN     = REF_LEN - 5   # 191
LEN_MAX     = REF_LEN + 5   # 201
N_GT_TAGS   = 11826          # design 总数

# Clover 参数
CLOVER_TREE_DEPTH = 20
CLOVER_V_DRIFT    = 3
CLOVER_H_DRIFT    = 3
CLOVER_H_INDEX    = 20       # 前缀共享区长度
CLOVER_E_INDEX    = 20       # 后缀共享区长度


def banner(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


# ============================================================
# Step 0: 预处理
# ============================================================
def step0_preprocess(input_file):
    """
    读 output.txt (tag\\tread)，做去N + 长度过滤。
    不做簇级过滤——所有通过的 reads 都保留给 Clover 参与建树。
    """
    banner("Step 0  预处理")

    total = 0
    n_dropped = 0
    len_dropped = 0
    cleaned = []

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

            cleaned.append((tag, seq))

    n_tags = len(set(t for t, _ in cleaned))

    print(f"  输入文件:            {input_file}")
    print(f"  参考序列长度:        {REF_LEN}bp")
    print(f"  长度过滤范围:        [{LEN_MIN}, {LEN_MAX}]")
    print()
    print(f"  总行数:              {total:,}")
    print(f"  因含 N 剔除:         {n_dropped:,}")
    print(f"  因长度不合格剔除:    {len_dropped:,}")
    print()
    print(f"  ✅ 保留 reads:       {len(cleaned):,}")
    print(f"  ✅ 保留 tag 数:      {n_tags:,}")

    return cleaned


# ============================================================
# Step 1: 写 Clover 输入
# ============================================================
def step1_write_clover_input(cleaned, clover_input_path, clover_tag_path):
    """
    写两份输入文件：
      - --no-tag 模式:  idx read  （实际聚类用）
      - tag 模式:       tag read  （手动验证 accuracy 用）
    返回 idx → (tag, read) 映射。
    """
    banner("Step 1  写 Clover 输入文件")

    idx_map = {}

    with open(clover_input_path, 'w') as f:
        for i, (tag, read) in enumerate(cleaned, start=1):
            f.write(f"{i} {read}\n")
            idx_map[i] = (tag, read)

    print(f"  --no-tag 输入:  {clover_input_path}")
    print(f"                  {len(idx_map):,} 条 reads")
    print()

    with open(clover_tag_path, 'w') as f:
        for tag, read in cleaned:
            f.write(f"{tag} {read}\n")

    n_tags = len(set(t for t, _ in cleaned))

    print(f"  tag 模式输入:   {clover_tag_path}")
    print()
    print(f"  💡 手动验证 accuracy:")
    print()
    print(f"     cd {CLOVER_DIR}")
    print(f"     python -m clover.main \\")
    print(f"       -I {clover_tag_path} \\")
    print(f"       -L {REF_LEN} -T {n_tags} \\")
    print(f"       -D {CLOVER_TREE_DEPTH} -V {CLOVER_V_DRIFT} -H {CLOVER_H_DRIFT}")

    return idx_map


# ============================================================
# Step 2: 运行 Clover
# ============================================================
def step2_run_clover(clover_input_path, clover_output_base):
    """
    运行 Clover (--no-tag 模式)。
    运行前自动修改 load_config.py 确保参数正确。
    """
    banner("Step 2  运行 Clover")

    # ── 修改 load_config.py ──
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
        print(f"  ✅ 已更新 load_config.py:")
        print()
        for k, v in patches.items():
            print(f"     {k:20s} = {v}")
        print()
    else:
        print(f"  ℹ️  load_config.py 参数已正确")
        print()

    # ── 构建命令 ──
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

    print(f"  Clover 参数:")
    print(f"    -L {REF_LEN}  -D {CLOVER_TREE_DEPTH}  -V {CLOVER_V_DRIFT}  -H {CLOVER_H_DRIFT}")
    print(f"    h_index = {CLOVER_H_INDEX}")
    print(f"    e_index = {CLOVER_E_INDEX}")
    print(f"    thd_tree = 72")
    print(f"    four_tree = 124")
    print()
    print(f"  运行中 ...")
    print()

    t0 = time.time()
    subprocess.run(cmd, check=True, env=env, cwd=CLOVER_DIR)
    elapsed = time.time() - t0

    result_path = clover_output_base + ".txt"
    if not os.path.exists(result_path):
        alt_path = os.path.join(
            CLOVER_DIR, os.path.basename(clover_output_base) + ".txt")
        if os.path.exists(alt_path):
            os.rename(alt_path, result_path)
        else:
            raise FileNotFoundError(
                f"Clover 输出文件不存在:\n  尝试1: {result_path}\n  尝试2: {alt_path}")

    print()
    print(f"  ✅ Clover 完成")
    print(f"     耗时:  {elapsed:.1f}s")
    print(f"     输出:  {result_path}")

    return result_path


# ============================================================
# Step 3: 解析 Clover 输出
# ============================================================
def step3_parse_clover(clover_output_path):
    """解析 Clover 输出的 (idx, cid) 对 → {cid: [idx, ...]}"""
    banner("Step 3  解析 Clover 输出")

    with open(clover_output_path, 'r') as f:
        content = f.read()

    pairs = re.findall(r"\('?(\d+)'?,\s*'?(\d+)'?\)", content)

    cid_to_idxs = defaultdict(list)
    for idx_str, cid_str in pairs:
        cid_to_idxs[int(cid_str)].append(int(idx_str))

    sizes = [len(v) for v in cid_to_idxs.values()]

    print(f"  文件:      {clover_output_path}")
    print(f"  (idx,cid): {len(pairs):,} 条")
    print(f"  聚类簇数:  {len(cid_to_idxs):,}")
    print()

    if sizes:
        print(f"  簇大小概览:")
        print(f"    max = {max(sizes)}")
        print(f"    med = {sorted(sizes)[len(sizes)//2]}")
        print(f"    min = {min(sizes)}")

    return dict(cid_to_idxs)


# ============================================================
# Step 4: 统计
# ============================================================
def step4_statistics(cid_to_idxs, idx_map, stats_path):
    """计算 Purity / Coverage / 簇大小分布。"""
    banner("Step 4  聚类统计")

    total_reads = sum(len(v) for v in cid_to_idxs.values())
    n_clusters = len(cid_to_idxs)
    sizes = sorted([len(v) for v in cid_to_idxs.values()], reverse=True)

    # Purity
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
        ('size = 1',   sum(1 for s in sizes if s == 1)),
        ('2 - 5',      sum(1 for s in sizes if 2 <= s <= 5)),
        ('6 - 10',     sum(1 for s in sizes if 6 <= s <= 10)),
        ('11 - 30',    sum(1 for s in sizes if 11 <= s <= 30)),
        ('31 - 100',   sum(1 for s in sizes if 31 <= s <= 100)),
        ('> 100',      sum(1 for s in sizes if s > 100)),
    ]

    print(f"  聚类簇数:              {n_clusters:,}")
    print(f"  GT tag 数 (design):    {N_GT_TAGS:,}")
    print(f"  已聚类 reads:          {total_reads:,}")
    print()
    print(f"  Purity:                {purity*100:.2f}%")
    print(f"  Coverage:              {coverage*100:.2f}%  ({len(gt_tags_covered)}/{N_GT_TAGS})")
    print()
    print(f"  簇大小分布:")
    print()
    for label, count in buckets:
        bar = '█' * min(count // max(n_clusters // 50, 1), 40)
        print(f"    {label:15s}  {count:>7,}  {bar}")
    print()
    print(f"  max = {sizes[0]},  median = {sizes[len(sizes)//2]},  min = {sizes[-1]}")

    # 保存
    with open(stats_path, 'w') as f:
        f.write("Seq_1D Clover 聚类统计\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"聚类簇数:    {n_clusters:,}\n")
        f.write(f"GT tag 数:   {N_GT_TAGS:,}\n")
        f.write(f"reads:       {total_reads:,}\n\n")
        f.write(f"Purity:      {purity*100:.2f}%\n")
        f.write(f"Coverage:    {coverage*100:.2f}%\n\n")
        f.write("簇大小分布:\n")
        for label, count in buckets:
            f.write(f"  {label:15s}  {count:,}\n")
        f.write(f"\nmax={sizes[0]}, median={sizes[len(sizes)//2]}, min={sizes[-1]}\n")

    print()
    print(f"  💾 保存: {stats_path}")

    return purity, coverage


# ============================================================
# Step 5+6: Majority Vote + 写 read.txt / ref.txt
# ============================================================
def majority_vote(reads, ref_len):
    """逐位多数投票生成 pseudo-ref。"""
    vote = [Counter() for _ in range(ref_len)]
    for read in reads:
        for pos in range(min(len(read), ref_len)):
            b = read[pos].upper()
            if b in 'ACGT':
                vote[pos][b] += 1

    result = []
    last = 'A'
    for pos in range(ref_len):
        if vote[pos]:
            last = vote[pos].most_common(1)[0][0]
        result.append(last)
    return ''.join(result)


def step56_write_output(cid_to_idxs, idx_map, read_path, ref_path):
    """
    写 read.txt (簇间 ===== 分隔) + ref.txt (每行一条 pseudo-ref)。
    所有 Clover 输出的簇都保留，不做 min_reads 过滤。
    """
    banner("Step 5+6  Majority Vote → read.txt + ref.txt")

    SEPARATOR = "=====分隔符=====\n"
    n_clusters = 0
    n_reads = 0

    sorted_cids = sorted(cid_to_idxs.keys())

    with open(read_path, 'w') as fr, open(ref_path, 'w') as ff:
        for cid in sorted_cids:
            idxs = cid_to_idxs[cid]
            reads = [idx_map[idx][1] for idx in idxs if idx in idx_map]
            if not reads:
                continue

            pseudo_ref = majority_vote(reads, REF_LEN)

            for read in reads:
                fr.write(read + '\n')
            fr.write(SEPARATOR)

            ff.write(pseudo_ref + '\n')

            n_clusters += 1
            n_reads += len(reads)

            if n_clusters % 5000 == 0:
                print(f"    已写 {n_clusters:,} 个簇 ...", end='\r')

    print(f"  ✅ 写入完成")
    print()
    print(f"  簇数:      {n_clusters:,}")
    print(f"  reads:     {n_reads:,}")
    print()
    print(f"  read.txt:  {read_path}")
    print(f"  ref.txt:   {ref_path}")


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Seq_1D Pipeline')
    parser.add_argument('--skip_clover', action='store_true',
                        help='跳过 Clover，复用已有结果')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    feddna_dir = os.path.join(OUTPUT_DIR, '04_FedDNA_In')
    os.makedirs(feddna_dir, exist_ok=True)

    # 路径
    input_file       = os.path.join(BASE_DIR, 'output.txt')
    clover_input     = os.path.join(OUTPUT_DIR, '01_clover_input.txt')
    clover_tag_input = os.path.join(OUTPUT_DIR, '01_clover_tag_input.txt')
    clover_out_base  = os.path.join(OUTPUT_DIR, '02_clover_result')
    clover_out_txt   = clover_out_base + '.txt'
    stats_path       = os.path.join(OUTPUT_DIR, '03_stats.txt')
    read_path        = os.path.join(feddna_dir, 'read.txt')
    ref_path         = os.path.join(feddna_dir, 'ref.txt')

    print()
    print("=" * 60)
    print("  🚀  Sequencing_data_first_dimension Pipeline")
    print("=" * 60)
    print()
    print(f"  数据目录:   {BASE_DIR}")
    print(f"  输出目录:   {OUTPUT_DIR}")
    print()
    print(f"  ref_len = {REF_LEN}bp")
    print(f"  长度范围: [{LEN_MIN}, {LEN_MAX}]")
    print()
    print(f"  Clover 参数:")
    print(f"    D={CLOVER_TREE_DEPTH}  V={CLOVER_V_DRIFT}  H={CLOVER_H_DRIFT}")
    print(f"    h_index={CLOVER_H_INDEX}  e_index={CLOVER_E_INDEX}")

    t_start = time.time()

    # Step 0
    cleaned = step0_preprocess(input_file)

    # Step 1
    idx_map = step1_write_clover_input(cleaned, clover_input, clover_tag_input)
    del cleaned

    # Step 2
    if not args.skip_clover:
        clover_out_txt = step2_run_clover(clover_input, clover_out_base)
    else:
        banner("Step 2  跳过 Clover")
        print(f"  使用已有结果: {clover_out_txt}")
        if not os.path.exists(clover_out_txt):
            raise FileNotFoundError(f"Clover 输出不存在: {clover_out_txt}")

    # Step 3
    cid_to_idxs = step3_parse_clover(clover_out_txt)

    # Step 4
    purity, coverage = step4_statistics(cid_to_idxs, idx_map, stats_path)

    # Step 5+6
    step56_write_output(cid_to_idxs, idx_map, read_path, ref_path)

    # 总结
    elapsed = time.time() - t_start

    print()
    print("=" * 60)
    print("  🎉  Pipeline 完成!")
    print("=" * 60)
    print()
    print(f"  总耗时:     {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print()
    print(f"  Purity:     {purity*100:.2f}%")
    print(f"  Coverage:   {coverage*100:.2f}%")
    print()
    print(f"  输出文件:")
    print(f"    📄 {read_path}")
    print(f"    📄 {ref_path}")
    print()


if __name__ == '__main__':
    main()