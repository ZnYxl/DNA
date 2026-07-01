#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_clustering_i16.py  ——  I16 聚类质量统一评估 (Clover / GradHC 共用)

口径对齐 GradHC 论文 (Ben Shabat et al. 2024) Table 5, 这样数字能直接和
论文 dataset VI 的 0.0042 / 0.8669 并排比较。

输入: 一个 "预测簇分配" 文件, 每行  read_seq <TAB> pred_cluster_id
      一个 GT 标签文件 i16_gt_labels.txt, 每行  read_seq <TAB> gt_id <TAB> ed
评估: 把两边按 read_seq 对齐 (GT 铁律: 以序列为 key, 不靠行号)

输出两个论文指标:
  1) Accuracy(γ): 一个【GT簇】被算"正确召回", 当且仅当存在某个预测簇,
     它包含该 GT 簇 ≥γ 比例的成员, 且该预测簇里这些成员占比也达标(纯且全)。
     accuracy = 正确召回的GT簇数 / 总GT簇数。  论文默认 γ=0.95。
  2) TS (类纯度-召回综合): 对每个预测簇, 取其多数 GT 标签, 该簇贡献
     min(本簇命中多数标签数 / 多数标签的GT簇总大小, ...) —— 这里用论文式的
     "归一化正确聚类比例"。实现为: 每个 GT 簇找到最佳匹配预测簇的 F1, 求均值。

注: 论文 TS 的精确定义基于 Rashtchian 的 γ-accuracy 框架。这里给出两个稳健、
    可复现的量: gamma-accuracy(严格) 与 mean-F1(TS 近似), 都报出来。
"""
import sys, argparse, collections


def load_pred(path):
    """read_seq -> pred_cid"""
    d = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            d[parts[0]] = parts[1]
    return d


def load_gt(path):
    """read_seq -> gt_id (忽略 -1 未归属)"""
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            seq, gid = parts[0], parts[1]
            if gid == "-1":
                continue
            d[seq] = gid
    return d


def evaluate(pred, gt, gamma=0.95):
    # 只在两边都有的 read 上评估
    common = set(pred) & set(gt)
    if not common:
        sys.exit("[ERR] 预测与GT没有公共read(检查序列是否一致)")

    # GT簇 -> set(read), 预测簇 -> set(read)
    gt_clusters = collections.defaultdict(set)
    pred_clusters = collections.defaultdict(set)
    for r in common:
        gt_clusters[gt[r]].add(r)
        pred_clusters[pred[r]].add(r)

    n_gt = len(gt_clusters)
    n_pred = len(pred_clusters)

    # 建 read->pred 便于查
    # 对每个 GT 簇, 找到与之重叠最多的预测簇, 算 gamma-accuracy 和 F1
    correct_gamma = 0
    f1_sum = 0.0
    for gid, gset in gt_clusters.items():
        # 该GT簇成员落在哪些预测簇
        cnt = collections.Counter(pred[r] for r in gset)
        best_pred, overlap = cnt.most_common(1)[0]
        pset = pred_clusters[best_pred]
        recall = overlap / len(gset)            # 该GT簇被这个预测簇覆盖的比例
        precision = overlap / len(pset)         # 这个预测簇里属于该GT簇的比例
        # gamma-accuracy: 论文式严格判定
        if recall >= gamma and precision >= gamma:
            correct_gamma += 1
        # F1 (TS 近似)
        if precision + recall > 0:
            f1_sum += 2 * precision * recall / (precision + recall)

    accuracy_gamma = correct_gamma / n_gt
    mean_f1 = f1_sum / n_gt

    # 过度切分指标: 预测簇数 / GT簇数
    oversplit = n_pred / n_gt

    return {
        "n_common_reads": len(common),
        "n_gt_clusters": n_gt,
        "n_pred_clusters": n_pred,
        "oversplit_ratio": oversplit,
        f"accuracy_gamma{gamma}": accuracy_gamma,
        "correct_gt_clusters": correct_gamma,
        "mean_F1(TS_approx)": mean_f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="预测簇分配: read_seq\\tpred_cid")
    ap.add_argument("--gt", default="i16_gt_labels.txt", help="GT标签: read_seq\\tgt_id\\ted")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--name", default="clustering", help="方法名(打印用)")
    args = ap.parse_args()

    pred = load_pred(args.pred)
    gt = load_gt(args.gt)
    res = evaluate(pred, gt, args.gamma)

    print(f"\n========== I16 聚类评估: {args.name} ==========")
    print(f"  公共 reads          : {res['n_common_reads']:,}")
    print(f"  GT 簇数             : {res['n_gt_clusters']:,}")
    print(f"  预测簇数            : {res['n_pred_clusters']:,}")
    print(f"  过度切分比(pred/GT) : {res['oversplit_ratio']:.2f}x")
    print(f"  ----------------------------------------")
    print(f"  Accuracy(γ={args.gamma})     : {res[f'accuracy_gamma{args.gamma}']:.4f}  "
          f"({res['correct_gt_clusters']:,}/{res['n_gt_clusters']:,} GT簇正确召回)")
    print(f"  Mean-F1 (TS近似)    : {res['mean_F1(TS_approx)']:.4f}")
    print(f"  ========================================")
    print(f"  对照 GradHC论文 dataset VI:")
    print(f"    GradHC  acc=0.8669  Clover acc=0.0042  Rashtchian acc=0.6858")


if __name__ == "__main__":
    main()