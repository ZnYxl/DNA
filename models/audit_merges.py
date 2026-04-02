#!/usr/bin/env python3
"""
GT 合并审计脚本 (Merge Audit)
==============================
目的：对 SSI-EC 每一轮的簇合并操作，逐对判定"正确合并"还是"错误合并"。

原理：
  Round N-1 的标签 → Round N 的标签，如果多个旧簇 ID 映射到同一个新簇 ID，
  说明这些旧簇被合并了。对每个合并组，查看各旧簇的 majority GT label：
    - 所有旧簇的 majority GT 相同 → 正确合并（同源碎片回收）
    - 存在不同的 majority GT → 错误合并（异源污染）

输出：每轮的合并精度、错误合并的详细信息、污染 reads 数量估算。

用法：
  python audit_merges.py [--exp_dir PATH]

  直接放到数据集目录运行也可以，会自动检测路径。
"""
import os
import sys
import argparse
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime


def load_reads_and_clover_labels(read_txt_path):
    """从 read.txt 加载 reads 和 Clover 初始标签"""
    reads = []
    clover_labels = []
    current_cluster = 0

    with open(read_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('====='):
                current_cluster += 1
                continue
            reads.append(line.upper())
            clover_labels.append(current_cluster)

    print(f"   ✅ Reads: {len(reads)}, Clover 簇: {current_cluster + 1}")
    return reads, np.array(clover_labels, dtype=np.int64)


def load_refined_labels(label_path):
    """加载 refined_labels.txt"""
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            labels.append(int(line.strip()))
    return np.array(labels, dtype=np.int64)


def load_gt_tags(gt_path, reads):
    """加载 GT tags，返回每条 read 的 GT cluster ID"""
    # 构建 sequence → GT_id 字典
    seq_to_gt = {}
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                gt_id = int(parts[0])
                seq = parts[1].upper()
                seq_to_gt[seq] = gt_id

    # 匹配每条 read
    gt_labels = np.full(len(reads), -1, dtype=np.int64)
    matched = 0
    for i, read in enumerate(reads):
        if read in seq_to_gt:
            gt_labels[i] = seq_to_gt[read]
            matched += 1

    print(f"   ✅ GT 匹配: {matched}/{len(reads)} ({100*matched/len(reads):.1f}%)")
    return gt_labels


def get_cluster_gt_majority(labels, gt_labels, valid_mask=None):
    """
    对每个簇，计算其 majority GT label 和 purity。
    返回: {cluster_id: (majority_gt, purity, n_reads)}
    """
    cluster_info = {}
    cluster_reads = defaultdict(list)

    for i in range(len(labels)):
        cid = labels[i]
        if cid < 0:
            continue
        if valid_mask is not None and not valid_mask[i]:
            continue
        if gt_labels[i] >= 0:
            cluster_reads[cid].append(gt_labels[i])

    for cid, gt_list in cluster_reads.items():
        counter = Counter(gt_list)
        majority_gt, majority_count = counter.most_common(1)[0]
        purity = majority_count / len(gt_list)
        cluster_info[cid] = (majority_gt, purity, len(gt_list))

    return cluster_info


def find_merges(old_labels, new_labels):
    """
    找出从 old_labels 到 new_labels 发生的合并。

    逻辑：对于 new_labels 中的每个簇 ID，找出它包含了哪些 old_labels 中的簇 ID。
    如果一个 new 簇对应了多个 old 簇，就是发生了合并。

    注意：需要排除 label=-1 的 reads（噪声），以及标签没变的簇。

    返回: list of (new_cid, [old_cid_1, old_cid_2, ...])
    """
    # 对每条 read，记录 (old_label, new_label) 对
    new_to_old_clusters = defaultdict(set)

    for i in range(len(old_labels)):
        old_cid = old_labels[i]
        new_cid = new_labels[i]
        if old_cid < 0 or new_cid < 0:
            continue
        new_to_old_clusters[new_cid].add(old_cid)

    # 提取合并事件（new 簇对应 2+ 个 old 簇）
    merges = []
    for new_cid, old_cids in new_to_old_clusters.items():
        if len(old_cids) >= 2:
            merges.append((new_cid, sorted(old_cids)))

    return merges


def audit_one_round(round_idx, old_labels, new_labels, gt_labels, reads):
    """
    审计一轮的合并操作。

    返回审计结果 dict。
    """
    print(f"\n{'='*70}")
    print(f"  🔍 Round {round_idx} 合并审计")
    print(f"{'='*70}")

    # 计算旧标签下每个簇的 GT 信息
    old_cluster_info = get_cluster_gt_majority(old_labels, gt_labels)
    new_cluster_info = get_cluster_gt_majority(new_labels, gt_labels)

    # 找合并事件
    merges = find_merges(old_labels, new_labels)

    n_merges = len(merges)
    n_correct = 0
    n_wrong = 0
    wrong_details = []
    total_contaminated_reads = 0

    for new_cid, old_cids in merges:
        # 收集每个旧簇的 majority GT
        gt_set = set()
        old_info_list = []
        for old_cid in old_cids:
            if old_cid in old_cluster_info:
                maj_gt, pur, n_reads = old_cluster_info[old_cid]
                gt_set.add(maj_gt)
                old_info_list.append((old_cid, maj_gt, pur, n_reads))
            else:
                old_info_list.append((old_cid, -1, 0.0, 0))

        if len(gt_set) <= 1:
            n_correct += 1
        else:
            n_wrong += 1
            # 估算污染量：合并后簇中 minority reads 的数量
            if new_cid in new_cluster_info:
                _, new_pur, new_n = new_cluster_info[new_cid]
                contaminated = int(new_n * (1 - new_pur))
                total_contaminated_reads += contaminated
            else:
                contaminated = 0

            wrong_details.append({
                'new_cid': new_cid,
                'old_clusters': old_info_list,
                'n_gt_mixed': len(gt_set),
                'contaminated_reads': contaminated,
            })

    # --- 统计已有簇的标签=-1 丢失 ---
    n_old_valid = int((old_labels >= 0).sum())
    n_new_valid = int((new_labels >= 0).sum())
    n_lost_to_noise = n_old_valid - n_new_valid

    # --- 输出 ---
    precision = n_correct / n_merges if n_merges > 0 else 1.0

    print(f"\n  📊 合并统计:")
    print(f"     合并组数:       {n_merges}")
    print(f"     ✅ 正确合并:     {n_correct}  (同 GT 分子碎片)")
    print(f"     ❌ 错误合并:     {n_wrong}  (异 GT 分子混合)")
    print(f"     🎯 合并精度:     {precision:.4f}  ({precision*100:.2f}%)")
    print(f"     ☠️  污染 reads:   ~{total_contaminated_reads}")
    print(f"     🗑️  丢弃到 -1:   {n_lost_to_noise}")

    if n_wrong > 0:
        # 按污染量排序，打印 Top 错误
        wrong_details.sort(key=lambda x: -x['contaminated_reads'])
        n_show = min(20, len(wrong_details))
        print(f"\n  📋 Top {n_show} 错误合并详情:")
        print(f"     {'NewCID':>8}  {'#OldClusters':>12}  {'#GT混合':>7}  {'污染reads':>8}  旧簇详情")
        print(f"     {'─'*80}")
        for d in wrong_details[:n_show]:
            old_str = " + ".join(
                f"C{oc}(GT{gt},n={n})" 
                for oc, gt, pur, n in d['old_clusters']
                if gt >= 0
            )
            print(f"     {d['new_cid']:>8}  {len(d['old_clusters']):>12}  "
                  f"{d['n_gt_mixed']:>7}  {d['contaminated_reads']:>8}  {old_str}")

    # 统计错误合并的 GT pair 频率
    if n_wrong > 0:
        gt_pair_counter = Counter()
        for d in wrong_details:
            gts = sorted(set(
                gt for _, gt, _, _ in d['old_clusters'] if gt >= 0
            ))
            for i in range(len(gts)):
                for j in range(i+1, len(gts)):
                    gt_pair_counter[(gts[i], gts[j])] += 1
        
        print(f"\n  📊 错误合并涉及的 GT pair 频率 (Top 10):")
        for (g1, g2), cnt in gt_pair_counter.most_common(10):
            print(f"     GT {g1} ↔ GT {g2}: {cnt} 次")

    return {
        'round': round_idx,
        'n_merges': n_merges,
        'n_correct': n_correct,
        'n_wrong': n_wrong,
        'precision': precision,
        'contaminated_reads': total_contaminated_reads,
        'lost_to_noise': n_lost_to_noise,
        'wrong_details': wrong_details,
    }


def main():
    parser = argparse.ArgumentParser(description='GT Merge Audit')
    parser.add_argument('--exp_dir', type=str,
                        default='/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/',
                        help='实验目录')
    parser.add_argument('--gt_tags', type=str, default=None,
                        help='GT tags 文件路径 (默认自动检测)')
    args = parser.parse_args()

    exp_dir = args.exp_dir
    read_txt = os.path.join(exp_dir, '03_FedDNA_In/read.txt')
    label_dir = os.path.join(exp_dir, '04_Iterative_Labels/')

    # 自动检测 GT tags 路径
    gt_candidates = [
        args.gt_tags,
        os.path.join(exp_dir, 'seq1d_tags_reads.txt'),
        os.path.join(exp_dir, '..', '给师妹的clover数据集', 'seq_1d', 'seq1d_tags_reads.txt'),
    ]
    gt_path = None
    for p in gt_candidates:
        if p and os.path.exists(p):
            gt_path = p
            break

    if gt_path is None:
        print("❌ 找不到 GT tags 文件。请用 --gt_tags 指定路径。")
        sys.exit(1)

    print("=" * 70)
    print("  🔬 SSI-EC 合并审计 (GT Merge Audit)")
    print("=" * 70)
    print(f"  实验目录: {exp_dir}")
    print(f"  GT tags:  {gt_path}")
    print(f"  时间:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 加载数据 ──
    print(f"\n{'─'*70}")
    print("  📂 加载数据")
    print(f"{'─'*70}")

    reads, clover_labels = load_reads_and_clover_labels(read_txt)
    gt_labels = load_gt_tags(gt_path, reads)

    # ── 发现 refined labels 文件 ──
    label_files = sorted([
        f for f in os.listdir(label_dir)
        if f.startswith('refined_labels_') and f.endswith('.txt')
    ])
    print(f"   ✅ 发现 {len(label_files)} 轮标签文件")

    # ── 逐轮审计 ──
    all_results = []
    prev_labels = clover_labels.copy()

    for i, lf in enumerate(label_files):
        round_idx = i + 1
        label_path = os.path.join(label_dir, lf)
        print(f"\n   📂 加载 Round {round_idx}: {lf}")
        new_labels = load_refined_labels(label_path)

        if len(new_labels) != len(prev_labels):
            print(f"   ❌ 标签长度不匹配: prev={len(prev_labels)}, new={len(new_labels)}")
            continue

        result = audit_one_round(round_idx, prev_labels, new_labels, gt_labels, reads)
        all_results.append(result)

        # 下一轮的 prev 是当前轮的 new
        prev_labels = new_labels.copy()

    # ── 汇总表 ──
    print(f"\n\n{'='*70}")
    print("  📋 合并审计汇总表")
    print(f"{'='*70}")
    print(f"  {'Round':>6}  {'合并数':>6}  {'✅正确':>6}  {'❌错误':>6}  "
          f"{'精度':>8}  {'☠️污染reads':>10}  {'🗑️丢弃-1':>10}")
    print(f"  {'─'*70}")

    total_correct = 0
    total_wrong = 0
    total_contaminated = 0

    for r in all_results:
        print(f"  R{r['round']:>5}  {r['n_merges']:>6}  {r['n_correct']:>6}  "
              f"{r['n_wrong']:>6}  {r['precision']:>8.4f}  "
              f"{r['contaminated_reads']:>10}  {r['lost_to_noise']:>10}")
        total_correct += r['n_correct']
        total_wrong += r['n_wrong']
        total_contaminated += r['contaminated_reads']

    total_merges = total_correct + total_wrong
    overall_precision = total_correct / total_merges if total_merges > 0 else 1.0
    print(f"  {'─'*70}")
    print(f"  {'总计':>6}  {total_merges:>6}  {total_correct:>6}  "
          f"{total_wrong:>6}  {overall_precision:>8.4f}  "
          f"{total_contaminated:>10}")

    # ── 关键诊断结论 ──
    print(f"\n{'='*70}")
    print("  🩺 诊断结论")
    print(f"{'='*70}")
    if overall_precision >= 0.95:
        print(f"  A 层精度 {overall_precision:.1%} ≥ 95%: 合并算法本身可靠。")
        print(f"  退化主因大概率在 B 层（consensus 毒化传导）。")
    elif overall_precision >= 0.85:
        print(f"  A 层精度 {overall_precision:.1%} 在 85-95%: 合并有一定错误率。")
        print(f"  A 层和 B 层可能都有贡献，需要实验 2 来区分。")
    else:
        print(f"  A 层精度 {overall_precision:.1%} < 85%: 合并算法本身有严重问题。")
        print(f"  优先修 A 层（提高 MNN threshold、加更多校验）。")

    print(f"\n  累计污染 reads: ~{total_contaminated}")
    print(f"  占总 reads 比例: ~{total_contaminated/len(reads)*100:.2f}%")

    # ── 保存结果 ──
    out_path = os.path.join(exp_dir, 'merge_audit_report.txt')
    with open(out_path, 'w') as f:
        f.write(f"SSI-EC Merge Audit Report\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Exp: {exp_dir}\n\n")
        for r in all_results:
            f.write(f"Round {r['round']}: "
                    f"merges={r['n_merges']}, correct={r['n_correct']}, "
                    f"wrong={r['n_wrong']}, precision={r['precision']:.4f}, "
                    f"contaminated_reads={r['contaminated_reads']}\n")
            if r['wrong_details']:
                for d in r['wrong_details'][:50]:
                    old_str = " + ".join(
                        f"C{oc}(GT{gt},n={n})"
                        for oc, gt, pur, n in d['old_clusters']
                        if gt >= 0
                    )
                    f.write(f"  WRONG: new={d['new_cid']}, "
                            f"gt_mixed={d['n_gt_mixed']}, "
                            f"contaminated={d['contaminated_reads']}, "
                            f"{old_str}\n")
        f.write(f"\nOverall precision: {overall_precision:.4f}\n")
        f.write(f"Total contaminated reads: {total_contaminated}\n")

    print(f"\n  💾 详细报告: {out_path}")
    print(f"\n✅ 审计完成")


if __name__ == '__main__':
    main()