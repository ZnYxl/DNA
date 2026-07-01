#!/usr/bin/env python3
"""
run_50clusters_crossval.py
==========================
GradHC 复现验证 —— 作者自带样例 samples/50_clusters.txt (干净GT, 作者认证)

为什么用这份数据:
    - GradHC 作者亲手放进仓库的测试样例 (README 示例就是它)
    - GT 完全干净 (跨簇重复=0), 均匀随机设计, 无 dataset IV 的长程依赖缺陷
    - 默认参数若跑出近完美 → 证明 GradHC 安装/调用完全正确 (复现铁证)

流程:
    1. 从 50_clusters.txt 提取 GT (每个 ***** 上一行rep = 该块真值tag)
       —— 直接把它当 GradHC 输入 (本就是标准分块格式, 不用转换)
    2. 默认参数跑 GradHC (原版类, q=6,k=3,m=40, sd全默认)
    3. seq→tag 字典评估 Purity/ARI/簇数

用法:
    python run_50clusters_crossval.py \
        --input      /abs/.../samples/50_clusters.txt \
        --gradhc_dir /abs/.../GradHC
"""

import os
import sys
import glob
import time
import argparse
from collections import Counter, defaultdict


def banner(t):
    print(f"\n{'─'*60}\n  {t}\n{'─'*60}")


def parse_blocks(path):
    """解析标准分块格式 → list[(rep, [reads])]。***** 上一行=rep。"""
    clusters = []
    rep, reads, expect_rep = None, None, True
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if line == '':
                if rep is not None and reads:
                    clusters.append((rep, reads))
                rep, reads, expect_rep = None, None, True
                continue
            if line[0] == '*':
                expect_rep = False
                reads = []
                continue
            if expect_rep:
                rep = line
                expect_rep = True   # rep行后紧跟*行
                continue
            else:
                if reads is None:
                    reads = []
                reads.append(line.upper())
    if rep is not None and reads:
        clusters.append((rep, reads))
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',      required=True, help='50_clusters.txt')
    ap.add_argument('--gradhc_dir', required=True)
    args = ap.parse_args()

    print("="*60)
    print("  GradHC 复现验证 — 50_clusters.txt (作者自带干净数据)")
    print("="*60)

    # ── 提取 GT (块序号=tag, read序列做key) ──
    banner("提取 GT")
    clusters = parse_blocks(args.input)
    seq_to_tag = {}
    seq_to_tags_all = defaultdict(list)
    for tag, (rep, reads) in enumerate(clusters):
        for r in reads:
            seq_to_tag[r] = tag
            seq_to_tags_all[r].append(tag)
    n_gt = len(clusters)
    total_gt_reads = sum(len(rs) for _, rs in clusters)
    print(f"  GT 簇数:    {n_gt}")
    print(f"  GT reads:   {total_gt_reads}")
    print(f"  唯一序列:   {len(seq_to_tag)}")

    # ── 默认参数跑 GradHC ──
    banner("运行 GradHC (默认参数, 直接喂 50_clusters.txt)")
    results_dir = os.path.join(args.gradhc_dir, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    input_base = os.path.basename(args.input)
    old_pat = os.path.join(results_dir, input_base + '_*.clustering_results')
    for old in glob.glob(old_pat):
        os.remove(old)

    prev = os.getcwd()
    os.chdir(args.gradhc_dir)
    if args.gradhc_dir not in sys.path:
        sys.path.insert(0, args.gradhc_dir)
    from GradHC_clustering import GradHCBasedCluster

    print(f"  调用: GradHCBasedCluster(input, serial=True, export=True) ← 全默认")
    t0 = time.time()
    try:
        c = GradHCBasedCluster(args.input, serial=True, export=True)
        c.run()
    finally:
        os.chdir(prev)
    print(f"  完成, 耗时 {time.time()-t0:.1f}s")

    matches = glob.glob(old_pat)
    if not matches:
        raise FileNotFoundError(f"无输出: {old_pat}")
    result_path = max(matches, key=os.path.getmtime)
    print(f"  结果: {result_path}")

    # ── 解析输出 ──
    banner("解析输出")
    pred_clusters = []
    cur, expect_rep = None, True
    with open(result_path) as f:
        for raw in f:
            line = raw.strip()
            if line == '':
                if cur:
                    pred_clusters.append(cur)
                cur, expect_rep = None, True
                continue
            if line[0] == '*':
                expect_rep = False
                cur = []
                continue
            if expect_rep:
                cur, expect_rep = None, True
                continue
            else:
                if cur is None:
                    cur = []
                cur.append(line.upper())
    if cur:
        pred_clusters.append(cur)
    cid_to_reads = {i: rs for i, rs in enumerate(pred_clusters)}
    print(f"  预测簇数: {len(cid_to_reads)}")
    print(f"  归簇reads: {sum(len(v) for v in cid_to_reads.values())}")

    # ── 评估 ──
    banner("评估")
    pool = {s: list(t) for s, t in seq_to_tags_all.items()}
    total_reads = sum(len(v) for v in cid_to_reads.values())
    total_pure = 0
    gt_covered = set()
    pred_labels, true_labels = [], []
    n_unmatched = 0

    for cid, reads in cid_to_reads.items():
        tags = []
        for read in reads:
            cand = pool.get(read)
            if cand:
                t = cand.pop()
                tags.append(t)
                pred_labels.append(cid)
                true_labels.append(t)
            else:
                n_unmatched += 1
        if not tags:
            continue
        maj_tag, maj_n = Counter(tags).most_common(1)[0]
        total_pure += maj_n
        gt_covered.add(maj_tag)

    purity = total_pure / max(total_reads, 1)
    coverage = len(gt_covered) / max(n_gt, 1)

    print(f"  预测簇数:   {len(cid_to_reads)}")
    print(f"  GT 簇数:    {n_gt}")
    print(f"  归簇reads:  {total_reads}  (未匹配: {n_unmatched})")
    print()
    print(f"  ◆ Purity:    {purity:.4f}  ({purity*100:.2f}%)")
    print(f"  ◆ Coverage:  {coverage:.4f}  ({len(gt_covered)}/{n_gt})")

    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        if pred_labels:
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            print(f"  ◆ ARI:       {ari:.4f}")
            print(f"  ◆ NMI:       {nmi:.4f}")
    except ImportError:
        print("  ⚠️ sklearn 未装")

    banner("判读")
    print(f"  若 Purity>0.95 且 簇数≈{n_gt} 且 ARI>0.9:")
    print(f"    → GradHC 安装/调用【完全正确】(作者自带干净数据+默认参数复现成功)")
    print(f"    → Seq_1D / dataset IV 低分 = 数据特性 (高相似/高错误), 非复现错误")
    print(f"  若仍塌缩:")
    print(f"    → 调用/安装真有问题, 需排查 (但据 process_input 逻辑, 应成功)")

    print(f"\n✅ 完成")


if __name__ == '__main__':
    main()