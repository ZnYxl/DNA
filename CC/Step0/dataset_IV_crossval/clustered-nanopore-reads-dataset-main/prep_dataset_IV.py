#!/usr/bin/env python3
"""
prep_dataset_IV.py
==================
GradHC 交叉验证 —— dataset IV (Srinivasavaradhan 2021 / Microsoft CNR) 数据准备

输入:
    Centers.txt   10,000 条原始链 (length 110)，第 i 行 = 第 i 簇的真值中心
    Clusters.txt  269,709 条噪声 reads，按簇排列，簇间用 "====..." 分隔
                  第 i 个 ==== 块 = Centers 第 i 行的噪声拷贝 (已验证顺序对应)

输出 (供 GradHC + 评估使用):
    01_gradhc_input.txt   GradHC 单块输入格式 (占位rep + ***** + 全部reads)
                          —— 与 Seq_1D pipeline 同口径: 全部 reads 塞进单块，
                             让 GradHC 从零聚类，不泄露任何分簇先验
    01_gt_seq_to_tag.txt  GT 映射: "tag\tread"  (tag = 簇序号 0..9999)
                          —— 评估时用 seq→tag 字典回查 (GT对齐铁律: 序列做key)

设计与 Seq_1D pipeline 完全对齐:
    - 输入: 单块、占位rep、无监督 (同 pipeline_seq1d_gradhc.py step1)
    - GT:   read→tag 用序列做key (同 GT对齐铁律)
    差异仅: 数据源 (dataset IV 而非 Seq_1D)，且不做打薄/小簇过滤 (保持原始口径，
            先验证 GradHC 本身；如需对齐可后续加)

用法:
    python prep_dataset_IV.py \
        --centers  .../Centers.txt \
        --clusters .../Clusters.txt \
        --outdir   .../dataset_IV_crossval/prep
"""

import os
import argparse
from collections import Counter

REF_LEN = 110
PLACEHOLDER_REP = 'A' * REF_LEN   # 占位 rep，与 Seq_1D pipeline 同思路


def banner(t):
    print(f"\n{'─'*60}\n  {t}\n{'─'*60}")


def load_centers(path):
    centers = []
    with open(path) as f:
        for line in f:
            s = line.strip().upper()
            if s:
                centers.append(s)
    return centers


def parse_clusters(path):
    """
    解析 Clusters.txt。返回 list[list[read]]，外层 index = 簇序号。
    分隔符: 任意以 '=' 开头的行 (===============================)。
    结构: <====> reads... <====> reads... ...
    """
    clusters = []
    cur = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line[0] == '=':
                # 新簇开始
                if cur is not None:
                    clusters.append(cur)
                cur = []
                continue
            if cur is None:
                # 文件开头若无分隔符前缀，容错
                cur = []
            cur.append(line.upper())
    if cur is not None and len(cur) > 0:
        clusters.append(cur)
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--centers',  required=True)
    ap.add_argument('--clusters', required=True)
    ap.add_argument('--outdir',   required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── 加载 ──
    banner("加载 Centers / Clusters")
    centers = load_centers(args.centers)
    clusters = parse_clusters(args.clusters)
    print(f"  Centers 链数:   {len(centers):,}")
    print(f"  Clusters 簇数:  {len(clusters):,}")

    if len(centers) != len(clusters):
        print(f"  ⚠️  Centers({len(centers)}) ≠ Clusters({len(clusters)})！")
        print(f"      顺序对应可能不成立，请人工核查。继续但需警惕。")
    else:
        print(f"  ✅ Centers 与 Clusters 数量一致，顺序对应口径成立")

    # ── 体检: 簇大小分布 + 长度分布 ──
    banner("数据体检")
    sizes = [len(c) for c in clusters]
    total_reads = sum(sizes)
    sizes_sorted = sorted(sizes, reverse=True)
    n_empty   = sum(1 for s in sizes if s == 0)
    n_single  = sum(1 for s in sizes if s == 1)
    print(f"  总 reads:       {total_reads:,}")
    print(f"  簇大小: max={sizes_sorted[0]}, "
          f"med={sizes_sorted[len(sizes_sorted)//2]}, min={sizes_sorted[-1]}")
    print(f"  空簇: {n_empty}, 单条簇: {n_single}")
    print(f"  成簇(size>=1): {sum(1 for s in sizes if s>=1):,}  "
          f"(论文 dataset IV: 9,984)")

    # read 长度分布 (抽样前 5 万条)
    sample_lens = []
    for c in clusters:
        for r in c:
            sample_lens.append(len(r))
            if len(sample_lens) >= 50000:
                break
        if len(sample_lens) >= 50000:
            break
    lc = Counter(sample_lens)
    print(f"  read 长度 (前5万条抽样) top5: "
          f"{', '.join(f'{l}bp×{n}' for l,n in lc.most_common(5))}")

    # ── 写 GradHC 单块输入 ──
    banner("写 GradHC 输入 (单块、无监督)")
    gradhc_input = os.path.join(args.outdir, '01_gradhc_input.txt')
    with open(gradhc_input, 'w', newline='\n') as f:
        f.write(PLACEHOLDER_REP + '\n')
        f.write('*' * 29 + '\n')
        for c in clusters:
            for r in c:
                f.write(r + '\n')
        f.write('\n\n')
    print(f"  ✅ {gradhc_input}")
    print(f"     {total_reads:,} reads → 单块 (与 Seq_1D 同口径)")

    # ── 写 GT 映射: tag(簇序号) \t read ──
    banner("写 GT 映射 (seq→tag, 序列做key)")
    gt_path = os.path.join(args.outdir, '01_gt_seq_to_tag.txt')
    # 检查跨簇重复 read (同一序列出现在不同簇)
    seq_first_tag = {}
    n_cross_dup = 0
    with open(gt_path, 'w') as f:
        for tag, c in enumerate(clusters):
            for r in c:
                f.write(f"{tag}\t{r}\n")
                if r in seq_first_tag and seq_first_tag[r] != tag:
                    n_cross_dup += 1
                else:
                    seq_first_tag[r] = tag
    print(f"  ✅ {gt_path}")
    print(f"     唯一 read 序列: {len(seq_first_tag):,}")
    print(f"     跨簇重复 read 实例: {n_cross_dup:,}  "
          f"(理想=0；>0 说明不同GT分子产生了相同噪声read，评估时按多数处理)")

    banner("完成")
    print(f"  下一步: 跑 GradHC 默认参数")
    print(f"    python run_gradhc_default.py \\")
    print(f"      --input   {gradhc_input} \\")
    print(f"      --gt      {gt_path} \\")
    print(f"      --gradhc_dir <GradHC仓库路径>")


if __name__ == '__main__':
    main()