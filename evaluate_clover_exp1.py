"""
evaluate_clover_exp1.py - 评估 Clover 原始聚类的全部指标
用修复后的 idx 映射, 与 SSI-EC 结果直接对比
"""
import re
import numpy as np
from collections import Counter, defaultdict

# ================= 路径配置 =================
CLOVER_RESULT = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_Real_last/02_CloverOut/clover_result_merged.txt"
GT_TAGS_FILE = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1/exp1_tags_reads.txt"
# ============================================

def main():
    # 1. 解析 Clover idx → cid
    print("[1] 解析 Clover 输出...")
    with open(CLOVER_RESULT, 'r') as f:
        content = f.read()
    pairs = re.findall(r"\('?(\d+)'?,\s*'?(\d+)'?\)", content)
    idx_to_cid = {int(idx): int(cid) for idx, cid in pairs}
    del content, pairs
    print(f"    Clover 非噪声: {len(idx_to_cid)} 条")

    # 2. 加载 GT tags
    print("[2] 加载 GT tags...")
    gt_tags = {}  # line_idx_1based → tag_id
    with open(GT_TAGS_FILE, 'r') as f:
        for line_idx_0based, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 2:
                gt_tags[line_idx_0based + 1] = int(parts[0])
    print(f"    GT 条目: {len(gt_tags)}")

    # 3. 构建 pred_labels 和 gt_labels (只保留两边都有的)
    print("[3] 构建标签对...")
    pred_list = []
    gt_list = []
    for line_idx, cid in idx_to_cid.items():
        if line_idx in gt_tags:
            pred_list.append(cid)
            gt_list.append(gt_tags[line_idx])

    pred_arr = np.array(pred_list)
    gt_arr = np.array(gt_list)
    print(f"    匹配: {len(pred_arr)} 条")

    n_pred_clusters = len(set(pred_list))
    n_gt_clusters = len(set(gt_list))
    print(f"    Clover 簇数: {n_pred_clusters}")
    print(f"    GT 簇数:     {n_gt_clusters}")

    # 4. 计算指标
    print("\n[4] 计算指标...")

    # --- Purity ---
    cluster_gt_counts = defaultdict(Counter)
    for p, g in zip(pred_list, gt_list):
        cluster_gt_counts[p][g] += 1

    total_correct = sum(counter.most_common(1)[0][1] for counter in cluster_gt_counts.values())
    purity = total_correct / len(pred_list)

    # --- Recovery ---
    all_gt_covered = set()
    for counter in cluster_gt_counts.values():
        all_gt_covered.update(counter.keys())
    recovery = len(all_gt_covered) / n_gt_clusters

    # --- Micro Accuracy (= Purity for non-overlapping) ---
    micro_acc = purity

    # --- Recall@cluster ---
    gt_cluster_counts = defaultdict(Counter)
    for p, g in zip(pred_list, gt_list):
        gt_cluster_counts[g][p] += 1

    recall_sum = 0
    for g, counter in gt_cluster_counts.items():
        best_pred_count = counter.most_common(1)[0][1]
        total_in_gt = sum(counter.values())
        recall_sum += best_pred_count / total_in_gt
    recall_at_cluster = recall_sum / n_gt_clusters

    # --- ARI ---
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    print("    计算 ARI (可能需要几秒)...")
    ari = adjusted_rand_score(gt_arr, pred_arr)

    # --- NMI ---
    print("    计算 NMI...")
    nmi = normalized_mutual_info_score(gt_arr, pred_arr, average_method='arithmetic')

    # 5. 打印结果
    print("\n" + "=" * 60)
    print("📊 Clover 原始聚类评估 (exp_1)")
    print("=" * 60)
    print(f"  评估 reads:     {len(pred_arr):,}")
    print(f"  Clover 簇数:    {n_pred_clusters:,}")
    print(f"  GT 簇数:        {n_gt_clusters:,}")
    print(f"  ────────────────────────────────────────")
    print(f"  AI 标准指标:")
    print(f"    ARI:            {ari:.4f}")
    print(f"    NMI:            {nmi:.4f}")
    print(f"  Clover 对比指标:")
    print(f"    Purity:         {purity:.4f}  ({purity*100:.2f}%)")
    print(f"    Recovery Rate:  {recovery:.4f}  ({recovery*100:.2f}%)")
    print(f"    Micro Accuracy: {micro_acc:.4f}  ({micro_acc*100:.2f}%)")
    print(f"  补充指标:")
    print(f"    Recall@cluster: {recall_at_cluster:.4f}  ({recall_at_cluster*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()