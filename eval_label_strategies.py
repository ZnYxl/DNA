"""
eval_label_strategies.py - 聚类评估: 不同 -1 处理策略对比 (exp_1)

对每轮的 refined_labels, 评估三种策略:
  A. 忽略 -1 reads (只评估已分配的)
  B. -1 退回 Clover 原始标签
  C. Clover 原始 (baseline)

使用 CloverDataLoader 保证索引空间与训练代码完全一致。

用法:
  cd /mnt/st_data/liangxinyi/code
  python eval_label_strategies.py
"""
import os
import sys
import glob
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 确保能 import models
CODE_DIR = "/mnt/st_data/liangxinyi/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.step1_data import CloverDataLoader

# ================= 路径配置 =================
EXP_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_Real_last"
GT_TAGS_FILE = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1/exp1_tags_reads.txt"
LABELS_DIR = os.path.join(EXP_DIR, "04_Iterative_Labels")
FINAL_LABELS = os.path.join(EXP_DIR, "results/final/final_labels.txt")
# ============================================


def compute_metrics(pred_labels, gt_labels, name=""):
    """计算全部指标, 只在两边都有标签 (>=0) 的 reads 上评估"""
    valid = (pred_labels >= 0) & (gt_labels >= 0)
    n_valid = valid.sum()

    if n_valid == 0:
        print(f"    ⚠️ [{name}] 无有效 reads")
        return {}

    pred = pred_labels[valid]
    gt = gt_labels[valid]

    n_pred_clusters = len(np.unique(pred))
    n_gt_clusters = len(np.unique(gt))

    # Purity
    cluster_gt = defaultdict(Counter)
    for p, g in zip(pred, gt):
        cluster_gt[int(p)][int(g)] += 1
    total_correct = sum(c.most_common(1)[0][1] for c in cluster_gt.values())
    purity = total_correct / len(pred)

    # Recovery
    all_gt_covered = set()
    for c in cluster_gt.values():
        all_gt_covered.update(c.keys())
    recovery = len(all_gt_covered) / n_gt_clusters

    # Recall@cluster
    gt_cluster = defaultdict(Counter)
    for p, g in zip(pred, gt):
        gt_cluster[int(g)][int(p)] += 1
    recall_sum = 0
    for g, counter in gt_cluster.items():
        best = counter.most_common(1)[0][1]
        total = sum(counter.values())
        recall_sum += best / total
    recall_at_cluster = recall_sum / n_gt_clusters

    # ARI & NMI
    ari = adjusted_rand_score(gt, pred)
    nmi = normalized_mutual_info_score(gt, pred, average_method='arithmetic')

    return {
        'n_reads': int(n_valid),
        'n_pred': n_pred_clusters,
        'n_gt': n_gt_clusters,
        'purity': purity,
        'recovery': recovery,
        'recall_at_cluster': recall_at_cluster,
        'ari': ari,
        'nmi': nmi,
    }


def print_metrics(m, name):
    if not m:
        return
    print(f"    [{name}]")
    print(f"      评估 reads: {m['n_reads']:>10,}  |  Pred簇: {m['n_pred']:,}  |  GT簇: {m['n_gt']:,}")
    print(f"      ARI:            {m['ari']:.4f}")
    print(f"      NMI:            {m['nmi']:.4f}")
    print(f"      Purity:         {m['purity']:.4f}  ({m['purity']*100:.2f}%)")
    print(f"      Recovery:       {m['recovery']:.4f}  ({m['recovery']*100:.2f}%)")
    print(f"      Recall@cluster: {m['recall_at_cluster']:.4f}  ({m['recall_at_cluster']*100:.2f}%)")


def main():
    # =================================================================
    # 1. 用 CloverDataLoader 加载 (保证索引一致)
    # =================================================================
    print("[1] 用 CloverDataLoader 加载数据...")
    data_loader = CloverDataLoader(EXP_DIR)
    TOTAL = len(data_loader.reads)
    print(f"    data_loader.reads: {TOTAL:,}")

    clover_labels = np.array(data_loader.clover_labels, dtype=int)
    print(f"    Clover 有效标签: {(clover_labels >= 0).sum():,}")
    print(f"    Clover 簇数:     {len(set(clover_labels[clover_labels >= 0])):,}")

    print("[2] 加载 GT tags...")
    data_loader.load_gt_tags(GT_TAGS_FILE)
    gt_labels = np.array(data_loader.gt_labels, dtype=int)
    print(f"    GT 有效标签: {(gt_labels >= 0).sum():,}")

    # 释放内存
    data_loader.reads = []

    # =================================================================
    # 2. 扫描每轮 labels
    # =================================================================
    print("[3] 扫描 refined_labels...")
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "refined_labels_*.txt")))
    print(f"    找到 {len(label_files)} 个:")
    for f in label_files:
        print(f"      {os.path.basename(f)}")

    # =================================================================
    # 3. Clover baseline
    # =================================================================
    print("\n" + "=" * 75)
    print("📊 聚类评估: 不同 -1 处理策略对比")
    print("=" * 75)

    print(f"\n{'─' * 60}")
    print(f"📌 Clover Baseline")
    print(f"{'─' * 60}")
    m_clover = compute_metrics(clover_labels, gt_labels, "Clover")
    print_metrics(m_clover, "Clover 原始")

    # =================================================================
    # 4. 每轮评估
    # =================================================================
    all_results = [("Clover", m_clover)]

    for round_idx, lf in enumerate(label_files, 1):
        print(f"\n{'─' * 60}")
        print(f"📌 Round {round_idx} ({os.path.basename(lf)})")
        print(f"{'─' * 60}")

        ssi_labels = np.loadtxt(lf, dtype=int)
        if len(ssi_labels) != TOTAL:
            print(f"    ⚠️ 长度不匹配: labels={len(ssi_labels)}, data_loader={TOTAL}")
            continue

        n_noise = (ssi_labels == -1).sum()
        n_assigned = TOTAL - n_noise
        print(f"    已分配: {n_assigned:,} ({n_assigned/TOTAL*100:.1f}%)")
        print(f"    -1 噪声: {n_noise:,} ({n_noise/TOTAL*100:.1f}%)")

        # 策略 A: 忽略 -1
        print(f"\n    === 策略 A: 忽略 -1 reads ===")
        m_a = compute_metrics(ssi_labels, gt_labels, f"R{round_idx} ignore-1")
        print_metrics(m_a, "忽略 -1")

        # 策略 B: -1 退回 Clover 标签
        print(f"\n    === 策略 B: -1 退回 Clover 标签 ===")
        fallback_labels = ssi_labels.copy()
        noise_mask = (fallback_labels == -1)
        fallback_labels[noise_mask] = clover_labels[noise_mask]
        still_noise = (fallback_labels == -1).sum()
        print(f"    退回 {noise_mask.sum():,} 条, 仍有 -1: {still_noise:,}")
        m_b = compute_metrics(fallback_labels, gt_labels, f"R{round_idx} fallback")
        print_metrics(m_b, "-1→Clover")

        all_results.append((f"R{round_idx} 忽略-1", m_a))
        all_results.append((f"R{round_idx} -1→Clover", m_b))

    # =================================================================
    # 5. Post-processing
    # =================================================================
    if os.path.exists(FINAL_LABELS):
        print(f"\n{'─' * 60}")
        print(f"📌 Post-processing (强制分配)")
        print(f"{'─' * 60}")
        final_labels = np.loadtxt(FINAL_LABELS, dtype=int)
        if len(final_labels) == TOTAL:
            n_noise = (final_labels == -1).sum()
            print(f"    -1 残留: {n_noise:,}")
            m_pp = compute_metrics(final_labels, gt_labels, "Post-proc")
            print_metrics(m_pp, "Post-processing")
            all_results.append(("Post-proc 强制分配", m_pp))
        else:
            print(f"    ⚠️ 长度不匹配: {len(final_labels)} vs {TOTAL}")

    # =================================================================
    # 6. 汇总表
    # =================================================================
    print(f"\n{'=' * 90}")
    print("📋 汇总对比表")
    print(f"{'=' * 90}")
    print(f"{'方法':<28s} {'Reads':>10s} {'ARI':>8s} {'NMI':>8s} {'Purity':>8s} {'Recovery':>8s} {'R@C':>8s}")
    print("─" * 90)

    for name, m in all_results:
        if m:
            print(f"{name:<28s} {m['n_reads']:>10,} {m['ari']:>8.4f} {m['nmi']:>8.4f} "
                  f"{m['purity']:>8.4f} {m['recovery']:>8.4f} {m['recall_at_cluster']:>8.4f}")

    print("─" * 90)
    print("✅ 评估完成")


if __name__ == "__main__":
    main()