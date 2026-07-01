#!/usr/bin/env python3
"""
run_gradhc_default.py
=====================
GradHC 交叉验证 —— 在 dataset IV 上用【纯默认参数】跑 GradHC 并评估

目的:
    验证"我们的 GradHC 是不是真的 GradHC"。
    若 dataset IV (干净随机110bp设计) 上默认参数跑出近完美聚类 (Purity≈1, ARI≈1)，
    则证明调用链完全正确，Seq_1D 上的低分纯属数据特性 (196bp高相似区=论文4.1.1失败模式)。

⚠️ 关键: 这里用【原版 GradHCBasedCluster】，所有参数【默认】:
         q=6, k=3, m=40, L=32, distance_threshold=12, sd 全默认
         —— 不用 GradHCSdHigh 子类，不调任何参数。
         这才是纯净的"真的 GradHC"验证。

调用链与 Seq_1D pipeline 完全一致 (chdir→sys.path→import)，确保单变量:
    只换数据 (Seq_1D→dataset IV)、只换参数 (q=8/sd=0.40 → 全默认)。

评估口径:
    GT 对齐铁律 —— seq→tag 字典回查 (序列做key)。
    Purity   = Σ 各簇内多数tag的read数 / 总read数
    ARI/NMI  = sklearn (read级标签 vs GT标签)
    簇数对照  = 预测簇数 vs GT簇数(10000)

用法:
    python run_gradhc_default.py \
        --input      .../prep/01_gradhc_input.txt \
        --gt         .../prep/01_gt_seq_to_tag.txt \
        --gradhc_dir /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/GradHC
"""

import os
import sys
import glob
import time
import argparse
from collections import Counter, defaultdict


def banner(t):
    print(f"\n{'─'*60}\n  {t}\n{'─'*60}")


# ════════════════════════════════════════════════════════════
# Step A: 跑 GradHC (默认参数，复用 Seq_1D 调用链)
# ════════════════════════════════════════════════════════════
def run_gradhc_default(input_path, gradhc_dir):
    banner("运行 GradHC (纯默认参数 q=6,k=3,m=40,L=32,dist=12，sd全默认)")

    results_dir = os.path.join(gradhc_dir, 'Results')
    os.makedirs(results_dir, exist_ok=True)

    input_base = os.path.basename(input_path)
    old_pattern = os.path.join(results_dir, input_base + '_*.clustering_results')
    for old in glob.glob(old_pattern):
        os.remove(old)
        print(f"  🧹 清除旧结果: {os.path.basename(old)}")

    # ⚠ 关键: WORKING_DIR_ALGORITHMS = os.getcwd()+"/" 在 import 时固定，
    #   必须先 chdir 到 gradhc_dir 再 import (与 Seq_1D pipeline 完全一致)
    prev_cwd = os.getcwd()
    os.chdir(gradhc_dir)
    if gradhc_dir not in sys.path:
        sys.path.insert(0, gradhc_dir)
    from GradHC_clustering import GradHCBasedCluster

    print(f"  调用: GradHCBasedCluster(input, serial=True, export=True)  ← 全默认")
    print(f"  运行中 ...(26.9万 read 量级，请耐心)\n")

    t0 = time.time()
    try:
        # 原版类，全默认参数，不传 q/k/m/sd —— 用 __init__ 的默认值
        cluster = GradHCBasedCluster(
            input_path,
            serial=True,
            export=True,
        )
        cluster.run()
    finally:
        os.chdir(prev_cwd)
    elapsed = time.time() - t0

    matches = glob.glob(old_pattern)
    if not matches:
        raise FileNotFoundError(f"GradHC 输出未找到: {old_pattern}")
    result_path = max(matches, key=os.path.getmtime)
    print(f"\n  ✅ 完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  结果: {result_path}")
    return result_path


# ════════════════════════════════════════════════════════════
# Step B: 解析 GradHC 输出 (分块格式，复用 Seq_1D step3 逻辑)
# ════════════════════════════════════════════════════════════
def parse_gradhc(output_path):
    banner("解析 GradHC 输出")
    clusters = []
    cur_reads = None
    expect_rep = True

    with open(output_path) as f:
        for raw in f:
            line = raw.strip()
            if line == '':
                if cur_reads is not None and len(cur_reads) > 0:
                    clusters.append(cur_reads)
                cur_reads = None
                expect_rep = True
                continue
            if line[0] == '*':
                expect_rep = False
                cur_reads = []
                continue
            if expect_rep:
                cur_reads = None
                expect_rep = True
                continue
            else:
                if cur_reads is None:
                    cur_reads = []
                cur_reads.append(line)

    if cur_reads is not None and len(cur_reads) > 0:
        clusters.append(cur_reads)

    cid_to_reads = {cid: reads for cid, reads in enumerate(clusters)}
    sizes = [len(v) for v in cid_to_reads.values()]
    print(f"  解析簇数:   {len(cid_to_reads):,}")
    print(f"  归簇 reads: {sum(sizes):,}")
    if sizes:
        ss = sorted(sizes, reverse=True)
        print(f"  簇大小: max={ss[0]}, med={ss[len(ss)//2]}, min={ss[-1]}")
    return cid_to_reads


# ════════════════════════════════════════════════════════════
# Step C: 加载 GT (seq→tag 字典，序列做key)
# ════════════════════════════════════════════════════════════
def load_gt(gt_path):
    banner("加载 GT (seq→tag, 序列做key)")
    seq_to_tag = {}          # 序列 → tag (后写覆盖，跨簇重复极少)
    seq_to_tags_all = defaultdict(list)  # 保留所有(用于多实例消费)
    with open(gt_path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 2:
                continue
            tag, seq = parts
            tag = int(tag)
            seq = seq.upper()
            seq_to_tag[seq] = tag
            seq_to_tags_all[seq].append(tag)
    n_gt_clusters = len(set(seq_to_tag.values()))
    print(f"  唯一 read 序列: {len(seq_to_tag):,}")
    print(f"  GT 簇数:        {n_gt_clusters:,}")
    return seq_to_tag, seq_to_tags_all, n_gt_clusters


# ════════════════════════════════════════════════════════════
# Step D: 评估 Purity / ARI / NMI / 簇数
# ════════════════════════════════════════════════════════════
def evaluate(cid_to_reads, seq_to_tags_all, n_gt_clusters):
    banner("评估")

    total_reads = sum(len(v) for v in cid_to_reads.values())
    n_pred = len(cid_to_reads)

    # 可消费副本 (同一read多实例时逐个分配tag，避免重复消费)
    pool = {s: list(tags) for s, tags in seq_to_tags_all.items()}

    total_pure = 0
    gt_covered = set()
    pred_labels = []   # read级: 预测簇id
    true_labels = []   # read级: GT tag

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
        cnt = Counter(tags)
        maj_tag, maj_n = cnt.most_common(1)[0]
        total_pure += maj_n
        gt_covered.add(maj_tag)

    purity = total_pure / max(total_reads, 1)
    coverage = len(gt_covered) / max(n_gt_clusters, 1)

    print(f"  预测簇数:   {n_pred:,}")
    print(f"  GT 簇数:    {n_gt_clusters:,}")
    print(f"  过分割比:   {n_pred / n_gt_clusters:.3f}×")
    print(f"  归簇 reads: {total_reads:,}  (未匹配GT: {n_unmatched:,})")
    print()
    print(f"  ◆ Purity:    {purity:.4f}  ({purity*100:.2f}%)")
    print(f"  ◆ Coverage:  {coverage:.4f}  ({len(gt_covered)}/{n_gt_clusters})")

    # ARI / NMI
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        if len(pred_labels) > 0:
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            print(f"  ◆ ARI:       {ari:.4f}")
            print(f"  ◆ NMI:       {nmi:.4f}")
        else:
            print("  ⚠️ 无可评估read")
    except ImportError:
        print("  ⚠️ sklearn 未安装，跳过 ARI/NMI (pip install scikit-learn)")

    banner("结论判读")
    print(f"  对照 Seq_1D (q=8,sd=0.40): Purity=0.883, ARI=0.604, 簇数=10355")
    print(f"  对照 Clover Seq_1D:        Purity=0.928, ARI=0.884")
    print(f"  论文 dataset IV (γ=0.95):  Acc/TS 很高 (GradHC 在此类数据近完美)")
    print()
    print(f"  判读:")
    print(f"   • 若本次 Purity>0.95 且 ARI>0.9 且 簇数≈10000")
    print(f"     → GradHC 调用链【正确】，Seq_1D 低分=数据特性 (铁证)")
    print(f"   • 若本次也塌缩/低分")
    print(f"     → 调用/安装有 bug，需排查 (与数据无关)")

    return {
        'n_pred': n_pred, 'n_gt': n_gt_clusters,
        'purity': purity, 'coverage': coverage,
        'total_reads': total_reads,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',      required=True, help='01_gradhc_input.txt')
    ap.add_argument('--gt',         required=True, help='01_gt_seq_to_tag.txt')
    ap.add_argument('--gradhc_dir', required=True, help='GradHC 仓库根目录')
    ap.add_argument('--skip_run',   action='store_true',
                    help='跳过GradHC，复用已有 .clustering_results')
    ap.add_argument('--result',     default=None, help='--skip_run 时指定结果文件')
    args = ap.parse_args()

    print("="*60)
    print("  GradHC 交叉验证 —— dataset IV (默认参数)")
    print("="*60)

    if args.skip_run:
        if args.result:
            result_path = args.result
        else:
            pat = os.path.join(args.gradhc_dir, 'Results',
                               os.path.basename(args.input) + '_*.clustering_results')
            m = glob.glob(pat)
            if not m:
                raise FileNotFoundError(f"无已有结果: {pat}")
            result_path = max(m, key=os.path.getmtime)
        print(f"  复用结果: {result_path}")
    else:
        result_path = run_gradhc_default(args.input, args.gradhc_dir)

    cid_to_reads = parse_gradhc(result_path)
    seq_to_tag, seq_to_tags_all, n_gt = load_gt(args.gt)
    evaluate(cid_to_reads, seq_to_tags_all, n_gt)

    print(f"\n✅ 交叉验证完成")


if __name__ == '__main__':
    main()