#!/usr/bin/env python3
"""
SSI-EC 论文指标全口径评估 — GradHC 同口径版
=============================================
新增指标 (对齐 GradHC / Bioinformatics 2024):
  1. Threat Score (TS)    = TP / (TP + FP + FN)
  2. Accuracy(γ)          γ ∈ {0.5, 0.75, 0.9, 1.0}
  3. Recovery Rate        被覆盖的 GT 分子比例

已有指标:
  4. Purity               
  5. Perfect Cluster Rate (PCR)
  6. ARI / NMI
  7. Over-segmentation    n_pred / n_gt

评估对象:
  - CLOVER 基线         (read.txt)
  - SSI-EC Round 1      (refined_labels_084145.txt)
  - SSI-EC Round 2      (refined_labels_110601.txt)

运行:
  python eval_gradhc_metrics.py 2>&1 | tee eval_gradhc.txt
"""

import os, sys, glob, time
import numpy as np
from collections import Counter, defaultdict

# ═══════════════════════════════════════════════════════════════
# 路径配置（按你实际服务器路径修改）
# ═══════════════════════════════════════════════════════════════
EXP_DIR      = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/pe_ayb"
GT_TAGS_FILE = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/pe_ayb/pe_ayb_tags_reads.txt"
GT_REFS_FILE = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/pe_ayb/pe_ayb_refs.txt"

# 指定要评估的 refined_labels 文件（按时间戳顺序）
REFINED_LABELS = [
    os.path.join(EXP_DIR, "04_Iterative_Labels", "refined_labels_005939.txt"),  # Round 1
    os.path.join(EXP_DIR, "04_Iterative_Labels", "refined_labels_063153.txt"),  # Round 2
    os.path.join(EXP_DIR, "04_Iterative_Labels", "refined_labels_104409.txt"),  # Round 3
]
ROUND_NAMES = ["SSI-EC R1","SSI-EC R2","SSI-EC R3"]

# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_reads_and_clover():
    """从 read.txt 加载 reads + CLOVER 标签（=====分隔符法）"""
    read_path = os.path.join(EXP_DIR, "03_FedDNA_In", "read.txt")
    reads, clover_labels = [], []
    cid = 0
    with open(read_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("====="):
                cid += 1
            else:
                reads.append(line.upper())
                clover_labels.append(cid)
    print(f"   CLOVER reads: {len(reads):,},  簇: {cid:,}")
    return reads, np.array(clover_labels, dtype=np.int64)


def load_gt_labels(reads):
    """GT 加载：sequence→cluster_id 字典精确匹配"""
    print(f"\n📋 加载 GT: {os.path.basename(GT_TAGS_FILE)}")
    seq_to_gt = {}
    with open(GT_TAGS_FILE) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    seq_to_gt[parts[1].strip().upper()] = int(parts[0])
                except ValueError:
                    pass
    print(f"   GT 唯一序列: {len(seq_to_gt):,}")

    gt = np.full(len(reads), -1, dtype=np.int64)
    matched = 0
    for i, r in enumerate(reads):
        g = seq_to_gt.get(r)
        if g is not None:
            gt[i] = g
            matched += 1
    rate = matched / len(reads) * 100
    print(f"   GT 匹配率:   {matched:,}/{len(reads):,} ({rate:.1f}%)")
    return gt


def load_refined_labels(path, n_reads):
    """加载 refined_labels_*.txt"""
    labels = np.loadtxt(path, dtype=np.int64)
    assert len(labels) == n_reads, f"长度不匹配: {len(labels)} vs {n_reads}"
    return labels


# ═══════════════════════════════════════════════════════════════
# 核心指标实现
# ═══════════════════════════════════════════════════════════════

def _build_cluster_maps(pred, gt):
    """
    一次遍历，构建所有指标所需的映射。
    返回:
      c2g: dict  pred_id → list of gt_ids
      g2c: dict  gt_id   → Counter{pred_id: count}  (每个GT分子被哪些预测簇覆盖)
      gt_sizes: dict  gt_id → total reads
    """
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
    """Purity = Σ majority_in_cluster / total_reads"""
    correct = sum(Counter(gs).most_common(1)[0][1] for gs in c2g.values())
    return correct / max(n_valid, 1), correct


def compute_pcr(c2g):
    """Perfect Cluster Rate = 簇内 GT 全相同的簇 / 总预测簇数"""
    perfect = sum(1 for gs in c2g.values() if len(set(gs)) == 1)
    return perfect / max(len(c2g), 1), perfect, len(c2g)


def compute_recovery_rate(c2g, gt):
    """
    Recovery Rate = 被至少一个预测簇"主导覆盖"的GT分子数 / GT分子总数
    对应 CLOVER 论文中的 Recovery Rate。
    规则: 预测簇的 majority GT 视为该簇"代表"的GT分子。
    """
    all_gt = set(g for g in gt if g >= 0)
    recovered = set()
    for gs in c2g.values():
        maj = Counter(gs).most_common(1)[0][0]
        recovered.add(maj)
    return len(recovered) / max(len(all_gt), 1), len(recovered), len(all_gt)


def compute_threat_score(c2g, g2c, gt_sizes, n_valid):
    """
    Threat Score (TS / Jaccard) = TP / (TP + FP + FN)

    贪心匹配策略（与 GradHC 一致）：
      对每个预测簇 C̃, 找重叠最多的 GT 簇作为其匹配对象。
      每个 GT 簇只能被匹配一次（first-come 按簇大小降序）。

    TP = Σ |C̃_i ∩ C_matched(i)|
    FP = Σ |C̃_i| - TP          (预测簇中的非TP reads)
    FN = Σ |C_j| - TP           (GT簇中未被覆盖的reads)
    """
    # 为每个预测簇找最佳 GT 匹配
    pred_clusters = sorted(c2g.keys(),
                           key=lambda c: len(c2g[c]), reverse=True)

    matched_gt = set()
    TP = 0
    pred_sizes = {}

    for cid in pred_clusters:
        gs = c2g[cid]
        pred_sizes[cid] = len(gs)
        cnt = Counter(gs)
        # 按 GT 重叠数降序，选尚未被匹配的
        for gid, overlap in cnt.most_common():
            if gid not in matched_gt:
                matched_gt.add(gid)
                TP += overlap
                break

    # FP: 预测簇中不属于其匹配 GT 的 reads
    total_pred_reads = sum(pred_sizes.values())
    FP = total_pred_reads - TP

    # FN: GT 簇中未被任何预测簇匹配的 reads
    total_gt_reads = sum(gt_sizes.values())
    FN = total_gt_reads - TP

    ts = TP / max(TP + FP + FN, 1)
    return ts, TP, FP, FN


def compute_accuracy_gamma(c2g, g2c, gt_sizes, gammas=(0.5, 0.75, 0.9, 1.0)):
    """
    Accuracy(γ) — GradHC 论文定义：
      Aγ = |{ GT簇 Ci : ∃ 预测簇 C̃j ⊆ Ci  且  |C̃j| ≥ γ|Ci| }| / |C|

    "C̃j ⊆ Ci" 等价于"C̃j 中所有 reads 都属于 Ci（零 FP）"。
    即：c2g[j] 中全部元素均等于 Ci 的 GT 标签，且 |c2g[j]| / |Ci| ≥ γ。

    匹配规则：每个 GT 簇取使 |C̃j|/|Ci| 最大的纯预测簇。
    """
    # 对每个 GT 簇，找最大的纯预测子簇（zero-FP）
    # c2g[pred_id] → list of GT ids
    # 纯预测簇: len(set(gs)) == 1
    gt_best_coverage = defaultdict(float)  # gt_id → best coverage ratio

    for cid, gs in c2g.items():
        if len(set(gs)) != 1:
            continue  # 非纯簇，不满足 zero-FP 约束
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
    """过分割率 = n_pred / n_gt，以及每个GT分子平均被切成几个子簇"""
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
# 综合评估函数
# ═══════════════════════════════════════════════════════════════

def evaluate_all(pred, gt, name, compute_ari_nmi_flag=True):
    """对一组标签计算所有指标，返回结果字典"""
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
    print(f"  过分割率:       {n_pred/max(n_gt_covered_reads/max(n_gt_covered_reads,1),1):.2f}×"
          f"  ({n_pred:,} pred / {n_gt_covered_reads:,} gt reachable)")

    t0 = time.time()

    c2g, g2c, gt_sizes, _ = _build_cluster_maps(pred, gt)

    # 1. Purity
    purity, correct = compute_purity(c2g, n_valid)
    print(f"\n  [Purity]         {purity:.6f}  ({correct:,}/{n_valid:,})")

    # 2. PCR
    pcr, perfect, n_pred_c = compute_pcr(c2g)
    print(f"  [PCR]            {pcr:.6f}  ({perfect:,}/{n_pred_c:,} 簇全纯)")

    # 3. Recovery Rate
    rec, n_rec, n_gt_all = compute_recovery_rate(c2g, gt)
    print(f"  [Recovery Rate]  {rec:.6f}  ({n_rec:,}/{n_gt_all:,} GT分子被覆盖)")

    # 4. Threat Score
    ts, TP, FP, FN = compute_threat_score(c2g, g2c, gt_sizes, n_valid)
    print(f"\n  [Threat Score]   {ts:.6f}  (TP={TP:,}  FP={FP:,}  FN={FN:,})")

    # 5. Accuracy(γ)
    gammas = [0.5, 0.75, 0.9, 1.0]
    acc_gamma, best_cov = compute_accuracy_gamma(c2g, g2c, gt_sizes, gammas)
    print(f"\n  [Accuracy(γ)]")
    for g, v in acc_gamma.items():
        covered = int(v * len(gt_sizes))
        print(f"     γ={g:.2f}:  {v:.6f}  ({covered:,}/{len(gt_sizes):,} GT分子)")

    # 6. ARI / NMI
    ari, nmi = None, None
    if compute_ari_nmi_flag:
        print(f"\n  [ARI/NMI] 计算中...")
        ari, nmi = compute_ari_nmi(pred, gt)
        if ari is not None:
            print(f"     ARI: {ari:.6f}")
            print(f"     NMI: {nmi:.6f}")

    # 7. Over-segmentation
    os_stats, frags = compute_over_segmentation(pred, gt)
    if os_stats:
        print(f"\n  [过分割]  n_pred/n_gt = {os_stats['ratio']:.2f}×")
        print(f"     GT分子: {os_stats['n_gt']:,}")
        print(f"     均/中位/最大碎片: {os_stats['mean_frags']:.2f} / "
              f"{os_stats['median_frags']:.0f} / {os_stats['max_frags']}")
        print(f"     1片: {os_stats['frag1']:,}  2-5片: {os_stats['frag2_5']:,}  "
              f"6-20片: {os_stats['frag6_20']:,}  >20片: {os_stats['frag20plus']:,}")

    elapsed = time.time() - t0
    print(f"\n  ⏱  耗时: {elapsed:.1f}s")

    return {
        'name': name,
        'n_pred': n_pred,
        'n_gt': n_gt_covered_reads,
        'n_valid': n_valid,
        'purity': purity,
        'pcr': pcr,
        'recovery_rate': rec,
        'threat_score': ts,
        'TP': TP, 'FP': FP, 'FN': FN,
        'accuracy_gamma': acc_gamma,
        'ari': ari,
        'nmi': nmi,
        'over_seg_ratio': os_stats.get('ratio', 0),
        'mean_frags': os_stats.get('mean_frags', 0),
    }


# ═══════════════════════════════════════════════════════════════
# 汇总表
# ═══════════════════════════════════════════════════════════════

def print_summary_table(results):
    gammas = [0.5, 0.75, 0.9, 1.0]

    print(f"\n\n{'═'*120}")
    print(f"{'📋  综合对比表（GradHC 同口径）':^120s}")
    print(f"{'═'*120}")

    # Header
    hdr1 = f"  {'Method':<20s}  {'Purity':>8s}  {'PCR':>8s}  {'Recovery':>9s}  {'TS':>9s}"
    hdr2 = f"  {'Acc(γ=0.50)':>11s}  {'Acc(γ=0.75)':>11s}  {'Acc(γ=0.90)':>11s}  {'Acc(γ=1.00)':>11s}"
    hdr3 = f"  {'OvrSeg×':>8s}  {'n_pred':>8s}  {'n_gt':>7s}"
    print(hdr1 + hdr2 + hdr3)
    print(f"  {'─'*115}")

    for r in results:
        ag = r['accuracy_gamma']
        line = (
            f"  {r['name']:<20s}"
            f"  {r['purity']:>8.4f}"
            f"  {r['pcr']:>8.4f}"
            f"  {r['recovery_rate']:>9.4f}"
            f"  {r['threat_score']:>9.4f}"
            f"  {ag.get(0.5, 0):>11.4f}"
            f"  {ag.get(0.75, 0):>11.4f}"
            f"  {ag.get(0.9, 0):>11.4f}"
            f"  {ag.get(1.0, 0):>11.4f}"
            f"  {r['over_seg_ratio']:>8.2f}x"
            f"  {r['n_pred']:>8,}"
            f"  {r['n_gt']:>7,}"
        )
        print(line)

    print(f"{'═'*120}")
    print()
    print("  指标说明:")
    print("  Purity        : 每预测簇内多数GT类别比例（reads加权均值）")
    print("  PCR           : Perfect Cluster Rate，簇内完全纯一的簇比例")
    print("  Recovery Rate : 被至少一个预测簇主导覆盖的GT分子比例（召回率维度）")
    print("  TS            : Threat Score = TP/(TP+FP+FN)，GradHC主指标")
    print("  Acc(γ)        : 存在零FP且覆盖≥γ比例的GT簇的比例，GradHC对比指标")
    print("  OvrSeg×       : n_pred / n_gt，过分割倍率")
    print()

    # Delta table
    if len(results) >= 2:
        base = results[0]
        print(f"  📈 相对 [{base['name']}] 的变化:")
        print(f"  {'Method':<20s}  {'ΔPurity':>9s}  {'ΔRecovery':>10s}  {'ΔTS':>9s}  "
              f"{'ΔAcc0.75':>9s}  {'ΔAcc1.0':>8s}  {'Δ簇数':>8s}")
        print(f"  {'─'*80}")
        for r in results[1:]:
            dp = r['purity'] - base['purity']
            drec = r['recovery_rate'] - base['recovery_rate']
            dts = r['threat_score'] - base['threat_score']
            da75 = r['accuracy_gamma'].get(0.75, 0) - base['accuracy_gamma'].get(0.75, 0)
            da10 = r['accuracy_gamma'].get(1.0, 0) - base['accuracy_gamma'].get(1.0, 0)
            dn = r['n_pred'] - base['n_pred']
            sign = lambda x: "+" if x >= 0 else ""
            print(f"  {r['name']:<20s}"
                  f"  {sign(dp)}{dp:>+8.4f}"
                  f"  {sign(drec)}{drec:>+9.4f}"
                  f"  {sign(dts)}{dts:>+8.4f}"
                  f"  {sign(da75)}{da75:>+8.4f}"
                  f"  {sign(da10)}{da10:>+7.4f}"
                  f"  {sign(dn)}{dn:>+7,}")
        print()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  SSI-EC × GradHC 同口径评估")
    print("=" * 68)

    # 加载数据
    print("\n📂 加载 reads + CLOVER 标签...")
    reads, clover_labels = load_reads_and_clover()
    N = len(reads)

    gt = load_gt_labels(reads)
    n_gt_total = len(set(gt[gt >= 0].tolist()))
    print(f"   GT 分子总数:  {n_gt_total:,}")

    all_results = []

    # ─── 评估 0: CLOVER 基线 ─────────────────────────────────
    print(f"\n\n{'═'*68}")
    print(f"  🔵  CLOVER 基线评估")
    r = evaluate_all(clover_labels, gt, "CLOVER (基线)", compute_ari_nmi_flag=True)
    all_results.append(r)

    # ─── 评估 1-N: SSI-EC 每轮 ──────────────────────────────
    for i, (lpath, rname) in enumerate(zip(REFINED_LABELS, ROUND_NAMES)):
        if not os.path.exists(lpath):
            print(f"\n⚠️ 文件不存在，跳过: {lpath}")
            continue
        print(f"\n\n{'═'*68}")
        print(f"  🟢  {rname} 评估")
        labels = load_refined_labels(lpath, N)
        r = evaluate_all(labels, gt, rname, compute_ari_nmi_flag=True)
        all_results.append(r)

    # ─── 汇总表 ─────────────────────────────────────────────
    print_summary_table(all_results)

    # ─── 论文数据速查（直接复制进表格）───────────────────────
    print(f"\n{'═'*68}")
    print(f"  📄 论文数据速查（可直接填入对比表）")
    print(f"{'═'*68}")
    for r in all_results:
        ag = r['accuracy_gamma']
        print(f"\n  [{r['name']}]")
        print(f"    Purity:          {r['purity']:.4f}")
        print(f"    Recovery Rate:   {r['recovery_rate']:.4f}")
        print(f"    Threat Score:    {r['threat_score']:.4f}")
        print(f"    Accuracy(γ=0.75):{ag.get(0.75, 0):.4f}")
        print(f"    Accuracy(γ=1.00):{ag.get(1.0, 0):.4f}")
        print(f"    PCR:             {r['pcr']:.4f}")
        if r['ari'] is not None:
            print(f"    ARI:             {r['ari']:.4f}")
            print(f"    NMI:             {r['nmi']:.4f}")
        print(f"    n_pred / n_gt:   {r['n_pred']:,} / {r['n_gt']:,}"
              f"  ({r['over_seg_ratio']:.2f}×)")

    print(f"\n✅ 完成")


if __name__ == '__main__':
    main()