#!/usr/bin/env python3
"""
gradhc_to_ssiec_datasetIV.py
============================
把 GradHC 在 dataset IV 的聚类输出 → SSI-EC 输入格式

背景:
    GradHC 在 dataset IV 上塌缩成 ~1921 大簇 (欠分割, 0.19×)。
    这正是 SSI-EC 拆分引擎 (簇内编辑距离层次二分) 的【完美对象】:
    GradHC 把异源reads粘成大簇, SSI-EC 通过迭代拆分修复。
    叙事: GradHC塌缩→SSI-EC拆分修复, 证明SSI-EC能救GradHC救不了的。

输入:
    GradHC 输出 (分块格式: rep + ***** + reads, 双空行分块)
    GT: prep/01_gt_seq_to_tag.txt (tag\tread)
    Centers.txt (110bp 真值, 做 gt_refs)

输出 (部署到 EXPERIMENT_DIR):
    03_FedDNA_In/read.txt   每簇reads + =====分隔符=====
    03_FedDNA_In/ref.txt    每簇 majority vote consensus
    datasetIV_tags_reads.txt  tag\tread (GT, eval+SSI-EC用)
    datasetIV_refs.txt        Centers.txt (gt_refs)

用法:
    python gradhc_to_ssiec_datasetIV.py \
        --gradhc_result /abs/.../01_gradhc_input_blocked.txt_badbddd.clustering_results \
        --gt /abs/.../prep/01_gt_seq_to_tag.txt \
        --centers /abs/.../Centers.txt \
        --experiment_dir /abs/.../Experiments/dataset_IV_gradhc
"""

import os
import re
import argparse
import shutil
from collections import Counter

REF_LEN = 110
MIN_READS = 5


def banner(t):
    print(f"\n{'─'*60}\n  {t}\n{'─'*60}")


def parse_gradhc_output(path):
    """解析 GradHC 分块输出 → list[[reads]]。rep行+*****行+reads, 双空行分块。"""
    clusters = []
    cur, expect_rep = None, True
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if line == '':
                if cur:
                    clusters.append(cur)
                cur, expect_rep = None, True
                continue
            if line[0] == '*':
                expect_rep = False
                cur = []
                continue
            if expect_rep:
                # rep行, 跳过(它是consensus占位/真值, 不是read)
                cur, expect_rep = None, True
                continue
            else:
                if cur is None:
                    cur = []
                cur.append(line.upper())
    if cur:
        clusters.append(cur)
    return clusters


def majority_vote(reads, ref_len):
    vote = [Counter() for _ in range(ref_len)]
    for r in reads:
        for pos in range(min(len(r), ref_len)):
            b = r[pos].upper()
            if b in 'ACGT':
                vote[pos][b] += 1
    out, last = [], 'A'
    for pos in range(ref_len):
        if vote[pos]:
            last = vote[pos].most_common(1)[0][0]
        out.append(last)
    return ''.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gradhc_result', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--centers', required=True)
    ap.add_argument('--experiment_dir', required=True)
    ap.add_argument('--min_reads', type=int, default=MIN_READS)
    args = ap.parse_args()

    os.makedirs(args.experiment_dir, exist_ok=True)
    feddna_dir = os.path.join(args.experiment_dir, '03_FedDNA_In')
    os.makedirs(feddna_dir, exist_ok=True)

    # ── 解析 GradHC 输出 ──
    banner("解析 GradHC 输出")
    clusters = parse_gradhc_output(args.gradhc_result)
    sizes = [len(c) for c in clusters]
    print(f"  原始簇数:   {len(clusters):,}")
    print(f"  总reads:    {sum(sizes):,}")
    if sizes:
        ss = sorted(sizes, reverse=True)
        print(f"  簇大小: max={ss[0]}, med={ss[len(ss)//2]}, min={ss[-1]}")

    # ── 过滤小簇 ──
    banner(f"过滤小簇 (<{args.min_reads})")
    kept = [c for c in clusters if len(c) >= args.min_reads]
    print(f"  过滤前: {len(clusters):,} 簇, {sum(sizes):,} reads")
    print(f"  过滤后: {len(kept):,} 簇, {sum(len(c) for c in kept):,} reads")
    print(f"  丢弃:   {len(clusters)-len(kept):,} 簇")

    # ── 写 read.txt + ref.txt ──
    banner("写 read.txt + ref.txt (majority vote)")
    SEP = "=====分隔符=====\n"
    read_path = os.path.join(feddna_dir, 'read.txt')
    ref_path = os.path.join(feddna_dir, 'ref.txt')
    nc = nr = 0
    with open(read_path, 'w') as fr, open(ref_path, 'w') as ff:
        for reads in kept:
            for r in reads:
                fr.write(r + '\n')
            fr.write(SEP)
            ff.write(majority_vote(reads, REF_LEN) + '\n')
            nc += 1
            nr += len(reads)
    print(f"  ✅ 簇数: {nc:,}, reads: {nr:,}")
    print(f"  read.txt: {read_path}")
    print(f"  ref.txt:  {ref_path}")

    # ── 部署 GT tags (复用prep, 已是 tag\tread) ──
    banner("部署 GT tags + refs")
    gt_tags_dst = os.path.join(args.experiment_dir, 'datasetIV_tags_reads.txt')
    shutil.copy2(args.gt, gt_tags_dst)
    print(f"  ✅ GT tags: {gt_tags_dst}")

    gt_refs_dst = os.path.join(args.experiment_dir, 'datasetIV_refs.txt')
    shutil.copy2(args.centers, gt_refs_dst)
    print(f"  ✅ GT refs: {gt_refs_dst} (Centers.txt)")

    # ── 快速质量预览 (过分割/欠分割?) ──
    banner("起点质量预览")
    # 加载GT
    seq_to_tag = {}
    with open(args.gt) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 2:
                seq_to_tag[p[1].upper()] = int(p[0])
    n_gt = len(set(seq_to_tag.values()))

    total_pure = total = 0
    covered = set()
    from collections import defaultdict
    pool = defaultdict(list)
    for s, t in seq_to_tag.items():
        pool[s].append(t)
    poolc = {s: list(t) for s, t in pool.items()}
    for reads in kept:
        tags = []
        for r in reads:
            c = poolc.get(r)
            if c:
                tags.append(c.pop())
        if not tags:
            continue
        mt, mn = Counter(tags).most_common(1)[0]
        total_pure += mn
        total += len(tags)
        covered.add(mt)
    purity = total_pure/max(total,1)
    print(f"  预测簇: {nc:,}  GT: {n_gt:,}  → 倍率 {nc/n_gt:.2f}×")
    print(f"  Purity: {purity:.4f}")
    print(f"  Coverage: {len(covered)/n_gt:.4f} ({len(covered)}/{n_gt})")
    if nc < n_gt:
        print(f"  → 欠分割 (簇<GT): SSI-EC拆分引擎对症! 预期TS单调上升")
    else:
        print(f"  → 过分割 (簇>GT): 拆分引擎可能不对症, 注意观察")

    banner("接 SSI-EC 三轮")
    ckpt = "/mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth"
    print(f"  cd /mnt/st_data/liangxinyi/code")
    print(f"  python -m models.main_loop \\")
    print(f"    --experiment_dir {args.experiment_dir}/ \\")
    print(f"    --feddna_checkpoint {ckpt} \\")
    print(f"    --gt_tags_file {gt_tags_dst} \\")
    print(f"    --gt_refs_file {gt_refs_dst} \\")
    print(f"    --max_iterations 3 --max_length 110 --target_clusters {n_gt} \\")
    print(f"    --cl_mode ours --ref_length 110 \\")
    print(f"    --primer_prefix 0 --primer_suffix 0 \\")
    print(f"    --split_tau 5 --split_min_size 6 \\")
    print(f"    2>&1 | tee datasetIV_gradhc_C.log")

    print(f"\n✅ 转换完成")


if __name__ == '__main__':
    main()