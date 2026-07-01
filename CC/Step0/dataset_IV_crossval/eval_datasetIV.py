#!/usr/bin/env python3
"""
eval_clover_vs_gradhc.py
========================
对比 1：纯聚类器横向对照 —— Clover vs GradHC（同口径，全指标）

两者从同一份打薄 read（output.txt → seed=42, max_reads_per_tag=30）出发，
各自聚类 + 小簇过滤(<5)，因此保留的 read 数不同：
  - Clover : 11,648 簇, 330,788 reads
  - GradHC :  8,713 簇, 350,836 reads
purity 分母不同，故汇总表显式列出每方法的「保留 reads」(n_valid)，避免误读。

指标实现与 eval_gradhc_metrics.py 完全一致（逐字复用，未改计算逻辑）。

运行:
  cd /mnt/st_data/liangxinyi/code/models
  python eval_clover_vs_gradhc.py 2>&1 | tee eval_clover_vs_gradhc.txt
"""

import os, sys, time
import numpy as np
from collections import Counter, defaultdict

# ═══════════════════════════════════════════════════════════════
# 路径配置 —— 两个方法各自的 read.txt + GT tags（内容相同，路径自洽）
# ═══════════════════════════════════════════════════════════════
METHODS = [
    {
        "name": "Clover (dataset IV)",
        "read_txt": "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/dataset_IV_clover/03_FedDNA_In/read.txt",
        "gt_tags":  "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/dataset_IV_clover/datasetIV_tags_reads.txt",
    },
]

N_GT_DESIGN = 9984   # design 总分子数（仅作参考打印）

# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_reads_and_labels(read_txt):
    """从 read.txt 加载 reads + 聚类标签（=====分隔符法）"""
    reads, labels = [], []
    cid = 0
    with open(read_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("====="):
                cid += 1
            else:
                reads.append(line.upper())
                labels.append(cid)
    return reads, np.array(labels, dtype=np.int64)


def load_gt_labels(reads, gt_tags_file):
    """GT 加载：sequence→cluster_id 字典精确匹配（铁律：用 sequence 作 key）"""
    seq_to_gt = {}
    with open(gt_tags_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    seq_to_gt[parts[1].strip().upper()] = int(parts[0])
                except ValueError:
                    pass
    gt = np.full(len(reads), -1, dtype=np.int64)
    matched = 0
    for i, r in enumerate(reads):
        g = seq_to_gt.get(r)
        if g is not None:
            gt[i] = g
            matched += 1
    rate = matched / max(len(reads), 1) * 100
    print(f"   GT 唯一序列: {len(seq_to_gt):,}   匹配率: {matched:,}/{len(reads):,} ({rate:.1f}%)")
    return gt


# ═══════════════════════════════════════════════════════════════
# 核心指标实现（与 eval_gradhc_metrics.py 逐字一致）
# ═══════════════════════════════════════════════════════════════

def _build_cluster_maps(pred, gt):
    valid = (pred >= 0) & (gt >= 0)
    pv, gv = pred[valid], gt[valid]
    c2g = defaultdict(list)
    g2c = defaultdict(Counter)
    for pi, gi in zip(pv, gv):
        c2g[int(pi)].append(int(gi))
        g2c[int(gi)][int(pi)] += 1
    gt_sizes = {gid: sum(cnt.values()) for gid, cnt in g2c.items()}
    return c2g, g2c, gt_sizes, int(valid.sum())


def compute_purity(c2g, n_valid):
    correct = sum(Counter(gs).most_common(1)[0][1] for gs in c2g.values())
    return correct / max(n_valid, 1), correct


def compute_pcr(c2g):
    perfect = sum(1 for gs in c2g.values() if len(set(gs)) == 1)
    return perfect / max(len(c2g), 1), perfect, len(c2g)


def compute_recovery_rate(c2g, gt):
    all_gt = set(g for g in gt if g >= 0)
    recovered = set()
    for gs in c2g.values():
        maj = Counter(gs).most_common(1)[0][0]
        recovered.add(maj)
    return len(recovered) / max(len(all_gt), 1), len(recovered), len(all_gt)


def compute_threat_score(c2g, g2c, gt_sizes, n_valid):
    pred_clusters = sorted(c2g.keys(), key=lambda c: len(c2g[c]), reverse=True)
    matched_gt = set()
    TP = 0
    pred_sizes = {}
    for cid in pred_clusters:
        gs = c2g[cid]
        pred_sizes[cid] = len(gs)
        cnt = Counter(gs)
        for gid, overlap in cnt.most_common():
            if gid not in matched_gt:
                matched_gt.add(gid)
                TP += overlap
                break
    total_pred_reads = sum(pred_sizes.values())
    FP = total_pred_reads - TP
    total_gt_reads = sum(gt_sizes.values())
    FN = total_gt_reads - TP
    ts = TP / max(TP + FP + FN, 1)
    return ts, TP, FP, FN


def compute_accuracy_gamma(c2g, g2c, gt_sizes, gammas=(0.5, 0.75, 0.9, 1.0)):
    gt_best_coverage = defaultdict(float)
    for cid, gs in c2g.items():
        if len(set(gs)) != 1:
            continue
        gid = gs[0]
        ratio = len(gs) / max(gt_sizes[gid], 1)
        if ratio > gt_best_coverage[gid]:
            gt_best_coverage[gid] = ratio
    n_gt = len(gt_sizes)
    results = {}
    for gamma in gammas:
        covered = sum(1 for r in gt_best_coverage.values() if r >= gamma)
        results[gamma] = covered / max(n_gt, 1)
    return results, gt_best_coverage


def compute_over_segmentation(pred, gt):
    valid = (pred >= 0) & (gt >= 0)
    gt2pred = defaultdict(set)
    for p, g in zip(pred[valid], gt[valid]):
        gt2pred[int(g)].add(int(p))
    if not gt2pred:
        return {}, 0
    frags = [len(s) for s in gt2pred.values()]
    stats = {
        'n_gt': len(gt2pred),
        'mean_frags': np.mean(frags),
        'median_frags': np.median(frags),
        'max_frags': max(frags),
        'ratio': len(set(pred[valid].tolist())) / max(len(gt2pred), 1),
        'frag1': sum(1 for f in frags if f == 1),
        'frag2_5': sum(1 for f in frags if 2 <= f <= 5),
        'frag6_20': sum(1 for f in frags if 6 <= f <= 20),
        'frag20plus': sum(1 for f in frags if f > 20),
    }
    return stats, frags


def compute_ari_nmi(pred, gt):
    valid = (pred >= 0) & (gt >= 0)
    pv, gv = pred[valid], gt[valid]
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        return adjusted_rand_score(gv, pv), normalized_mutual_info_score(gv, pv)
    except ImportError:
        print("   ⚠️ sklearn 未安装，ARI/NMI 跳过")
        return None, None


# ═══════════════════════════════════════════════════════════════
# 综合评估
# ═══════════════════════════════════════════════════════════════

def evaluate_all(pred, gt, name, compute_ari_nmi_flag=True):
    print(f"\n{'═'*68}")
    print(f"  📊  {name}")
    print(f"{'═'*68}")

    valid = (pred >= 0) & (gt >= 0)
    n_valid = int(valid.sum())
    n_pred = len(set(pred[valid].tolist()))
    n_gt_total = len(set(gt[gt >= 0].tolist()))
    n_gt_covered_reads = len(set(gt[valid].tolist()))

    print(f"  有效 reads:     {n_valid:>12,} / {len(pred):,}")
    print(f"  预测簇数:       {n_pred:>12,}")
    print(f"  GT 分子数:      {n_gt_total:>12,}  (含 reads: {n_gt_covered_reads:,})")

    t0 = time.time()
    c2g, g2c, gt_sizes, _ = _build_cluster_maps(pred, gt)

    purity, correct = compute_purity(c2g, n_valid)
    print(f"\n  [Purity]         {purity:.6f}  ({correct:,}/{n_valid:,})")

    pcr, perfect, n_pred_c = compute_pcr(c2g)
    print(f"  [PCR]            {pcr:.6f}  ({perfect:,}/{n_pred_c:,} 簇全纯)")

    rec, n_rec, n_gt_all = compute_recovery_rate(c2g, gt)
    print(f"  [Recovery Rate]  {rec:.6f}  ({n_rec:,}/{n_gt_all:,} GT分子被覆盖)")

    ts, TP, FP, FN = compute_threat_score(c2g, g2c, gt_sizes, n_valid)
    print(f"\n  [Threat Score]   {ts:.6f}  (TP={TP:,}  FP={FP:,}  FN={FN:,})")

    gammas = [0.5, 0.75, 0.9, 1.0]
    acc_gamma, _ = compute_accuracy_gamma(c2g, g2c, gt_sizes, gammas)
    print(f"\n  [Accuracy(γ)]")
    for g, v in acc_gamma.items():
        covered = int(v * len(gt_sizes))
        print(f"     γ={g:.2f}:  {v:.6f}  ({covered:,}/{len(gt_sizes):,} GT分子)")

    ari, nmi = None, None
    if compute_ari_nmi_flag:
        print(f"\n  [ARI/NMI] 计算中...")
        ari, nmi = compute_ari_nmi(pred, gt)
        if ari is not None:
            print(f"     ARI: {ari:.6f}")
            print(f"     NMI: {nmi:.6f}")

    os_stats, _ = compute_over_segmentation(pred, gt)
    if os_stats:
        print(f"\n  [过分割]  n_pred/n_gt = {os_stats['ratio']:.2f}×")
        print(f"     GT分子: {os_stats['n_gt']:,}")
        print(f"     均/中位/最大碎片: {os_stats['mean_frags']:.2f} / "
              f"{os_stats['median_frags']:.0f} / {os_stats['max_frags']}")
        print(f"     1片: {os_stats['frag1']:,}  2-5片: {os_stats['frag2_5']:,}  "
              f"6-20片: {os_stats['frag6_20']:,}  >20片: {os_stats['frag20plus']:,}")

    print(f"\n  ⏱  耗时: {time.time()-t0:.1f}s")

    return {
        'name': name,
        'n_pred': n_pred,
        'n_valid': n_valid,
        'n_gt': n_gt_covered_reads,
        'purity': purity,
        'pcr': pcr,
        'recovery_rate': rec,
        'threat_score': ts,
        'TP': TP, 'FP': FP, 'FN': FN,
        'accuracy_gamma': acc_gamma,
        'ari': ari, 'nmi': nmi,
        'over_seg_ratio': os_stats.get('ratio', 0),
        'mean_frags': os_stats.get('mean_frags', 0),
    }


# ═══════════════════════════════════════════════════════════════
# 汇总表
# ═══════════════════════════════════════════════════════════════

def print_summary_table(results):
    print(f"\n\n{'═'*132}")
    print(f"{'📋  对比 1：Clover vs GradHC 纯聚类器横向对照':^132s}")
    print(f"{'═'*132}")

    hdr = (f"  {'Method':<14s}  {'保留reads':>10s}  {'簇数':>8s}  {'Purity':>8s}  "
           f"{'PCR':>8s}  {'Recovery':>9s}  {'TS':>8s}  "
           f"{'Acc0.50':>8s}  {'Acc0.75':>8s}  {'Acc0.90':>8s}  {'Acc1.00':>8s}  "
           f"{'OvrSeg':>7s}  {'ARI':>7s}  {'NMI':>7s}")
    print(hdr)
    print(f"  {'─'*128}")
    for r in results:
        ag = r['accuracy_gamma']
        ari = f"{r['ari']:.4f}" if r['ari'] is not None else "  -  "
        nmi = f"{r['nmi']:.4f}" if r['nmi'] is not None else "  -  "
        print(f"  {r['name']:<14s}  {r['n_valid']:>10,}  {r['n_pred']:>8,}  "
              f"{r['purity']:>8.4f}  {r['pcr']:>8.4f}  {r['recovery_rate']:>9.4f}  "
              f"{r['threat_score']:>8.4f}  "
              f"{ag.get(0.5,0):>8.4f}  {ag.get(0.75,0):>8.4f}  {ag.get(0.9,0):>8.4f}  {ag.get(1.0,0):>8.4f}  "
              f"{r['over_seg_ratio']:>6.2f}x  {ari:>7s}  {nmi:>7s}")
    print(f"{'═'*132}")

    print()
    print("  ⚠ 关键阅读提示：")
    print("    · 两方法从同一份打薄 read 出发，但小簇过滤后「保留reads」不同（分母不同）。")
    print("      Purity 高不代表更好——需结合 Recovery 与保留reads 一起看取舍。")
    print()
    print("  指标说明:")
    print("    保留reads : 该方法聚类+小簇过滤(<5)后剩余 read 数（purity 的分母）")
    print("    Purity    : 每预测簇内多数GT类别比例（reads加权均值）")
    print("    PCR       : 簇内完全纯一的簇比例")
    print("    Recovery  : 被至少一个预测簇主导覆盖的GT分子比例（召回维度）")
    print("    TS        : Threat Score = TP/(TP+FP+FN)，GradHC 主指标")
    print("    Acc(γ)    : 存在零FP且覆盖≥γ的GT分子比例（γ近1=最严苛）")
    print("    OvrSeg    : n_pred / n_gt，过分割倍率（越接近1越好）")

    # Δ 对照（GradHC 相对 Clover）
    if len(results) == 2:
        base, cmp = results[0], results[1]
        print(f"\n  📈 [{cmp['name']}] 相对 [{base['name']}] 的变化:")
        def d(k): return cmp[k] - base[k]
        def dg(g): return cmp['accuracy_gamma'].get(g,0) - base['accuracy_gamma'].get(g,0)
        sgn = lambda x: "+" if x >= 0 else ""
        print(f"    ΔPurity   : {sgn(d('purity'))}{d('purity'):.4f}")
        print(f"    ΔRecovery : {sgn(d('recovery_rate'))}{d('recovery_rate'):.4f}")
        print(f"    ΔTS       : {sgn(d('threat_score'))}{d('threat_score'):.4f}")
        print(f"    ΔAcc0.75  : {sgn(dg(0.75))}{dg(0.75):.4f}")
        print(f"    ΔAcc1.00  : {sgn(dg(1.0))}{dg(1.0):.4f}")
        print(f"    Δ簇数     : {sgn(cmp['n_pred']-base['n_pred'])}{cmp['n_pred']-base['n_pred']:,}")
        print(f"    Δ保留reads: {sgn(cmp['n_valid']-base['n_valid'])}{cmp['n_valid']-base['n_valid']:,}")


# ═══════════════════════════════════════════════════════════════
# 论文速查
# ═══════════════════════════════════════════════════════════════

def print_paper_block(results):
    print(f"\n{'═'*68}")
    print(f"  📄 论文数据速查（可直接填入对比表）")
    print(f"{'═'*68}")
    for r in results:
        ag = r['accuracy_gamma']
        print(f"\n  [{r['name']}]")
        print(f"    保留 reads:      {r['n_valid']:,}")
        print(f"    簇数:            {r['n_pred']:,}")
        print(f"    Purity:          {r['purity']:.4f}")
        print(f"    Recovery Rate:   {r['recovery_rate']:.4f}")
        print(f"    Threat Score:    {r['threat_score']:.4f}")
        print(f"    Accuracy(γ=0.75):{ag.get(0.75, 0):.4f}")
        print(f"    Accuracy(γ=0.90):{ag.get(0.9, 0):.4f}")
        print(f"    Accuracy(γ=1.00):{ag.get(1.0, 0):.4f}")
        print(f"    PCR:             {r['pcr']:.4f}")
        if r['ari'] is not None:
            print(f"    ARI:             {r['ari']:.4f}")
            print(f"    NMI:             {r['nmi']:.4f}")
        print(f"    过分割:          {r['over_seg_ratio']:.2f}×")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  对比 1：Clover vs GradHC 纯聚类器评估（同口径全指标）")
    print("=" * 68)

    all_results = []
    for m in METHODS:
        if not os.path.exists(m["read_txt"]):
            print(f"\n⚠️ read.txt 不存在，跳过: {m['read_txt']}")
            continue
        print(f"\n\n{'━'*68}")
        print(f"  📂 加载 {m['name']}")
        print(f"     read.txt: {m['read_txt']}")
        reads, labels = load_reads_and_labels(m["read_txt"])
        print(f"     reads: {len(reads):,},  簇: {labels.max()+1 if len(labels) else 0:,}")
        gt = load_gt_labels(reads, m["gt_tags"])
        r = evaluate_all(labels, gt, m["name"], compute_ari_nmi_flag=True)
        all_results.append(r)

    print_summary_table(all_results)
    print_paper_block(all_results)
    print(f"\n✅ 完成")


if __name__ == '__main__':
    main()