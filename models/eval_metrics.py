# models/eval_metrics.py
"""
SSI-EC 聚类评估指标体系

包含:
  - ARI (Adjusted Rand Index)
  - NMI (Normalized Mutual Information)
  - Purity
  - Recovery Rate
  - Micro Accuracy
  - Recall@cluster
  - 分层分析 (按簇大小)
"""
import numpy as np
from collections import Counter, defaultdict


def compute_ari(pred_labels, gt_labels):
    """Adjusted Rand Index — 校正随机一致性, [-1, 1]"""
    try:
        from sklearn.metrics import adjusted_rand_score
        return adjusted_rand_score(gt_labels, pred_labels)
    except ImportError:
        print("   ⚠️ sklearn 未安装, ARI 跳过")
        return None


def compute_nmi(pred_labels, gt_labels):
    """Normalized Mutual Information — 信息论视角, [0, 1]"""
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return normalized_mutual_info_score(gt_labels, pred_labels)
    except ImportError:
        print("   ⚠️ sklearn 未安装, NMI 跳过")
        return None


def compute_purity(pred_labels, gt_labels):
    """
    Purity: 每个预测簇中最多的GT类别占比, reads数加权平均
    Clover 原始指标
    """
    cluster_gt = defaultdict(list)
    for p, g in zip(pred_labels, gt_labels):
        if p >= 0 and g >= 0:
            cluster_gt[p].append(g)

    if not cluster_gt:
        return 0.0

    total_correct = 0
    total_count = 0
    for cid, gt_list in cluster_gt.items():
        counter = Counter(gt_list)
        majority = counter.most_common(1)[0][1]
        total_correct += majority
        total_count += len(gt_list)

    return total_correct / max(total_count, 1)


def compute_recovery_rate(pred_labels, gt_labels):
    """
    Recovery Rate: 被至少一个预测簇覆盖的GT簇比例
    Clover 原始指标
    """
    # 找出所有 GT 簇
    gt_clusters = set(g for g in gt_labels if g >= 0)
    if not gt_clusters:
        return 0.0

    # 对每个预测簇, 找到其 majority GT
    cluster_gt = defaultdict(list)
    for p, g in zip(pred_labels, gt_labels):
        if p >= 0 and g >= 0:
            cluster_gt[p].append(g)

    recovered_gt = set()
    for cid, gt_list in cluster_gt.items():
        counter = Counter(gt_list)
        majority_gt = counter.most_common(1)[0][0]
        recovered_gt.add(majority_gt)

    return len(recovered_gt) / len(gt_clusters)


def compute_micro_accuracy(pred_labels, gt_labels):
    """
    Micro Accuracy: 被正确分配的 reads 总数 / 参与聚类的 reads 总数
    "正确分配" 定义: read 的 GT 标签 == 该 read 所在预测簇的 majority GT 标签
    Clover 原始指标
    """
    # 先找每个预测簇的 majority GT
    cluster_gt = defaultdict(list)
    read_assignments = []  # (pred_cluster, gt_label)

    for p, g in zip(pred_labels, gt_labels):
        if p >= 0 and g >= 0:
            cluster_gt[p].append(g)
            read_assignments.append((p, g))

    cluster_majority = {}
    for cid, gt_list in cluster_gt.items():
        counter = Counter(gt_list)
        cluster_majority[cid] = counter.most_common(1)[0][0]

    correct = sum(1 for p, g in read_assignments if cluster_majority.get(p) == g)
    return correct / max(len(read_assignments), 1)


def compute_recall_at_cluster(pred_labels, gt_labels):
    """
    Recall@cluster: 每个 GT 簇被正确召回的 reads 比例, 取平均
    展示方法在每个 GT 簇上的召回能力
    """
    # 建立 GT 簇 → reads 映射
    gt_to_reads = defaultdict(list)
    for i, g in enumerate(gt_labels):
        if g >= 0:
            gt_to_reads[g].append(i)

    if not gt_to_reads:
        return 0.0

    # 建立 pred 簇 → majority GT 映射
    cluster_gt_lists = defaultdict(list)
    for i, (p, g) in enumerate(zip(pred_labels, gt_labels)):
        if p >= 0 and g >= 0:
            cluster_gt_lists[p].append(g)

    cluster_majority = {}
    for cid, gt_list in cluster_gt_lists.items():
        counter = Counter(gt_list)
        cluster_majority[cid] = counter.most_common(1)[0][0]

    # 对每个 GT 簇, 计算被正确召回的比例
    recalls = []
    for gt_id, read_indices in gt_to_reads.items():
        total = len(read_indices)
        correct = 0
        for idx in read_indices:
            p = pred_labels[idx]
            if p >= 0 and cluster_majority.get(p) == gt_id:
                correct += 1
        recalls.append(correct / total)

    return np.mean(recalls)


def compute_stratified_analysis(pred_labels, gt_labels, initial_cluster_sizes):
    """
    分层分析: 按 Clover 初始簇大小分层报告指标
    initial_cluster_sizes: dict {cluster_id: size}

    分层:
      Singleton (1 read)
      Small (2-5 reads)
      Medium (6-50 reads)
      Large (>50 reads)
    """
    # 将 reads 按其所在初始簇的大小分层
    strata = {
        'singleton': {'pred': [], 'gt': []},
        'small':     {'pred': [], 'gt': []},
        'medium':    {'pred': [], 'gt': []},
        'large':     {'pred': [], 'gt': []},
    }

    for i, (p, g) in enumerate(zip(pred_labels, gt_labels)):
        if p < 0 or g < 0:
            continue
        # 这个 read 的初始簇大小
        size = initial_cluster_sizes.get(p, 0)
        if size <= 1:
            key = 'singleton'
        elif size <= 5:
            key = 'small'
        elif size <= 50:
            key = 'medium'
        else:
            key = 'large'
        strata[key]['pred'].append(p)
        strata[key]['gt'].append(g)

    results = {}
    for stratum, data in strata.items():
        if len(data['pred']) > 0:
            results[stratum] = {
                'count': len(data['pred']),
                'purity': compute_purity(data['pred'], data['gt']),
                'micro_acc': compute_micro_accuracy(data['pred'], data['gt']),
            }
        else:
            results[stratum] = {'count': 0, 'purity': 0.0, 'micro_acc': 0.0}

    return results


def compute_all_metrics(pred_labels, gt_labels, verbose=True):
    """
    计算全套评估指标

    Args:
        pred_labels: np.array, 预测簇标签 (全量, 无 -1)
        gt_labels:   np.array, GT 标签 (全量, -1 表示无 GT)
        verbose:     是否打印

    Returns:
        dict of metrics
    """
    # 过滤: 只保留 pred >= 0 且 gt >= 0 的 reads
    valid_mask = (pred_labels >= 0) & (gt_labels >= 0)
    pred_valid = pred_labels[valid_mask]
    gt_valid = gt_labels[valid_mask]

    if len(pred_valid) == 0:
        print("   ⚠️ 无有效评估样本")
        return {}

    metrics = {}

    # AI 社区标准指标
    metrics['ARI'] = compute_ari(pred_valid, gt_valid)
    metrics['NMI'] = compute_nmi(pred_valid, gt_valid)

    # Clover 原始指标
    metrics['Purity'] = compute_purity(pred_valid, gt_valid)
    metrics['Recovery_Rate'] = compute_recovery_rate(pred_valid, gt_valid)
    metrics['Micro_Accuracy'] = compute_micro_accuracy(pred_valid, gt_valid)

    # 展示方法优势
    metrics['Recall_at_cluster'] = compute_recall_at_cluster(pred_valid, gt_valid)

    # 统计信息
    n_pred_clusters = len(set(pred_valid))
    n_gt_clusters = len(set(gt_valid))
    metrics['n_pred_clusters'] = n_pred_clusters
    metrics['n_gt_clusters'] = n_gt_clusters
    metrics['n_evaluated_reads'] = len(pred_valid)

    if verbose:
        print(f"\n   {'='*60}")
        print(f"   📊 聚类评估结果")
        print(f"   {'='*60}")
        print(f"   评估 reads: {metrics['n_evaluated_reads']:,}")
        print(f"   预测簇数:   {n_pred_clusters:,}")
        print(f"   GT 簇数:    {n_gt_clusters:,}")
        print(f"   {'─'*60}")
        print(f"   AI 标准指标:")
        if metrics['ARI'] is not None:
            print(f"      ARI:              {metrics['ARI']:.4f}")
        if metrics['NMI'] is not None:
            print(f"      NMI:              {metrics['NMI']:.4f}")
        print(f"   Clover 对比指标:")
        print(f"      Purity:           {metrics['Purity']:.4f}  ({metrics['Purity']*100:.2f}%)")
        print(f"      Recovery Rate:    {metrics['Recovery_Rate']:.4f}  ({metrics['Recovery_Rate']*100:.2f}%)")
        print(f"      Micro Accuracy:   {metrics['Micro_Accuracy']:.4f}  ({metrics['Micro_Accuracy']*100:.2f}%)")
        print(f"   补充指标:")
        print(f"      Recall@cluster:   {metrics['Recall_at_cluster']:.4f}  ({metrics['Recall_at_cluster']*100:.2f}%)")
        print(f"   {'='*60}")

    return metrics


def save_metrics_report(metrics, output_path, round_info=""):
    """保存评估报告到文件"""
    import json
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 清理 None 值
    clean_metrics = {k: v for k, v in metrics.items() if v is not None}

    with open(output_path, 'w') as f:
        f.write(f"SSI-EC Clustering Evaluation Report\n")
        f.write(f"{'='*60}\n")
        if round_info:
            f.write(f"Info: {round_info}\n")
        f.write(f"\n")
        for k, v in clean_metrics.items():
            if isinstance(v, float):
                f.write(f"{k:25s}: {v:.6f}\n")
            else:
                f.write(f"{k:25s}: {v}\n")

    # 同时保存 JSON 格式
    json_path = output_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(clean_metrics, f, indent=2, default=str)

    print(f"   💾 评估报告: {output_path}")