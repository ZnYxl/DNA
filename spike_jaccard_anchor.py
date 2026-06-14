#!/usr/bin/env python
"""
spike_jaccard_anchor.py —— v20.A 前置 spike (5 分钟)

验证假设: 
  cross-cluster + same-GT pair 的 k-mer Jaccard 显著高于 
  cross-cluster + diff-GT pair → Jaccard 是好 anchor

如果通过, v20.A (Jaccard mask) 写出来有效;
如果失败, Jaccard 不能区分"该保护"vs"该推开", patch 白写.

5 分钟内出结果. 纯 CPU + numpy.

输出:
  Δ_median > 0.10  🟢 commit Patch A
  Δ_median 0.05-0.10 🟡 区分度中等, 用推荐 θ_j 跑一次
  Δ_median < 0.05  🔴 Jaccard 失效, 试 k=4 或 k=6
"""
import os, sys, argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_reads_from_read_txt(path):
    """跳过分隔符行, 只保留 ACGTN 序列"""
    reads = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or len(line) < 50: continue
            if all(c in 'ACGTN' for c in line):
                reads.append(line)
    return reads


def load_gt_tags(path, n_reads):
    tags = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            tags.append(parts[0] if parts else '__PAD__')
    if len(tags) < n_reads:
        tags = tags + ['__PAD__'] * (n_reads - len(tags))
    elif len(tags) > n_reads:
        tags = tags[:n_reads]
    uniq = sorted(set(tags))
    tag_id = {t: i for i, t in enumerate(uniq)}
    gt = np.array([tag_id[t] for t in tags], dtype=int)
    pad_id = tag_id.get('__PAD__', -1)
    if pad_id >= 0:
        gt[gt == pad_id] = -1
    return gt


def seq_to_kmer_set(seq, k=5):
    return frozenset(
        seq[i:i+k] for i in range(len(seq) - k + 1)
        if all(c in 'ACGT' for c in seq[i:i+k])
    )


def jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--read_txt', required=True,
                   help='03_FedDNA_In/read.txt')
    p.add_argument('--labels', required=True,
                   help='refined_labels (建议用 R3 最深训练状态)')
    p.add_argument('--gt_tags', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--n_pairs_per_group', type=int, default=5000)
    p.add_argument('--k', type=int, default=5)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 输出: {args.output_dir}")

    # ── 加载 ───────────────────────────────────────────
    print("\n加载数据...")
    reads = load_reads_from_read_txt(args.read_txt)
    print(f"  reads: {len(reads):,}")
    labels = np.loadtxt(args.labels, dtype=int)
    print(f"  labels: {len(labels):,} (label≥0: {int((labels>=0).sum()):,})")
    gt = load_gt_tags(args.gt_tags, len(reads))
    print(f"  gt: {len(gt):,} (valid: {int((gt>=0).sum()):,})")

    if len(reads) != len(labels):
        print(f"\n❌ reads ({len(reads)}) != labels ({len(labels)}), "
              f"read.txt 解析有误"); sys.exit(1)

    # ── 准备索引 ─────────────────────────────────────
    cluster_to_reads = defaultdict(list)
    gt_to_clusters = defaultdict(lambda: defaultdict(list))
    for i, (l, g) in enumerate(zip(labels, gt)):
        if l >= 0 and g >= 0:
            cluster_to_reads[int(l)].append(i)
            gt_to_clusters[int(g)][int(l)].append(i)

    multi_clusters = [c for c, rs in cluster_to_reads.items() if len(rs) >= 2]
    fragmented_gts = {g: cs for g, cs in gt_to_clusters.items() if len(cs) >= 2}
    print(f"  multi-read clusters: {len(multi_clusters):,}")
    print(f"  fragmented GTs (同 GT 切成 2+ cluster): {len(fragmented_gts):,}")

    if len(fragmented_gts) < 100:
        print(f"\n⚠️ fragmented GTs 太少 ({len(fragmented_gts)}). spike 信号弱.")

    # ── 抽样 ────────────────────────────────────────
    print(f"\n抽样三组 pair (each {args.n_pairs_per_group:,})...")
    rng = np.random.default_rng(42)
    pairs_sc, pairs_csg, pairs_cdg = [], [], []

    # group 1: same cluster
    for _ in range(args.n_pairs_per_group):
        c = rng.choice(multi_clusters)
        rs = cluster_to_reads[c]
        i, j = rng.choice(rs, size=2, replace=False)
        pairs_sc.append((int(i), int(j)))

    # group 2: cross cluster + same GT (我们的目标 anchor)
    if len(fragmented_gts) >= 1:
        fgs = list(fragmented_gts.keys())
        attempts = 0
        max_attempts = args.n_pairs_per_group * 10
        while len(pairs_csg) < args.n_pairs_per_group and attempts < max_attempts:
            attempts += 1
            g = rng.choice(fgs)
            cs = list(fragmented_gts[g].keys())
            c1, c2 = rng.choice(cs, size=2, replace=False)
            i = rng.choice(fragmented_gts[g][int(c1)])
            j = rng.choice(fragmented_gts[g][int(c2)])
            pairs_csg.append((int(i), int(j)))

    # group 3: cross cluster + diff GT (背景)
    valid_idx = np.where((labels >= 0) & (gt >= 0))[0]
    while len(pairs_cdg) < args.n_pairs_per_group:
        i, j = rng.choice(valid_idx, size=2, replace=False)
        if labels[i] != labels[j] and gt[i] != gt[j]:
            pairs_cdg.append((int(i), int(j)))

    print(f"  采到: same-cluster {len(pairs_sc)}, "
          f"cross-cluster same-GT {len(pairs_csg)}, "
          f"cross-cluster diff-GT {len(pairs_cdg)}")

    # ── k-mer 缓存 ──────────────────────────────────
    print(f"\n计算 {args.k}-mer set (with cache)...")
    needed = set()
    for grp in [pairs_sc, pairs_csg, pairs_cdg]:
        for i, j in grp:
            needed.add(i); needed.add(j)
    print(f"  unique reads 需要 k-mer: {len(needed):,}")
    cache = {}
    for cnt, ridx in enumerate(needed):
        if cnt % 5000 == 0:
            print(f"    {cnt}/{len(needed)}")
        cache[ridx] = seq_to_kmer_set(reads[ridx], args.k)

    def jacc_group(grp):
        return np.array([jaccard(cache[i], cache[j]) for i, j in grp])

    print("\n计算 Jaccard...")
    j_sc = jacc_group(pairs_sc)
    j_csg = jacc_group(pairs_csg) if pairs_csg else np.array([])
    j_cdg = jacc_group(pairs_cdg)

    # ── 报告 ────────────────────────────────────────
    print("\n" + "="*70)
    print(f"Jaccard 分布 (k={args.k})")
    print("="*70)
    for name, vals in [
        ("Same cluster (sanity)", j_sc),
        ("Cross cluster + Same GT (★ target)", j_csg),
        ("Cross cluster + Diff GT (background)", j_cdg),
    ]:
        print(f"\n  {name}: n={len(vals):,}")
        if len(vals):
            print(f"     mean={vals.mean():.4f}  median={np.median(vals):.4f}  "
                  f"std={vals.std():.4f}")
            print(f"     P25={np.quantile(vals, 0.25):.4f}  "
                  f"P75={np.quantile(vals, 0.75):.4f}")

    # ── 关键判定 ────────────────────────────────────
    print("\n" + "="*70)
    print("🎯 关键指标")
    print("="*70)
    if not len(j_csg):
        decision = "❌ cross-cluster same-GT 无样本, Jaccard 没有保护对象"
        delta = None
        theta_recommend = None
    else:
        delta = float(np.median(j_csg) - np.median(j_cdg))
        theta_recommend = float(np.quantile(j_cdg, 0.90))
        same_gt_above = float((j_csg > theta_recommend).mean())
        print(f"\n  ⭐ Δ_median = {delta:+.4f}")
        print(f"     cross-cluster same-GT median: {np.median(j_csg):.4f}")
        print(f"     cross-cluster diff-GT median: {np.median(j_cdg):.4f}")
        print(f"\n  推荐 θ_j ≈ {theta_recommend:.3f} (P90 of background)")
        print(f"     → same-GT pair 的 {same_gt_above:.1%} 会被保护")
        print(f"     → bg pair 误保护率: 10%")

        if delta > 0.10:
            decision = (f"🟢 commit Patch A. Δ_median={delta:.3f} 显著. "
                        f"用 θ_j={theta_recommend:.3f} 跑 v20.A")
        elif delta > 0.05:
            decision = (f"🟡 区分度中等 (Δ={delta:.3f}). 可尝试, "
                        f"但效果不会很大, 考虑配合 Patch B")
        else:
            decision = (f"🔴 Jaccard 区分度不足 (Δ={delta:.3f}). "
                        f"试 k=4 或 k=6, 或换 anchor (Hamming dist on aligned)")

    # ── 画图 ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 60)
    ax.hist(j_sc, bins=bins, alpha=0.4, density=True,
            label=f'Same cluster (n={len(j_sc)})', color='steelblue')
    if len(j_csg):
        ax.hist(j_csg, bins=bins, alpha=0.5, density=True,
                label=f'Cross cluster + Same GT ★ (n={len(j_csg)})',
                color='darkorange')
        ax.axvline(np.median(j_csg), color='darkorange', ls='--', lw=2)
    ax.hist(j_cdg, bins=bins, alpha=0.4, density=True,
            label=f'Cross cluster + Diff GT (n={len(j_cdg)})', color='crimson')
    ax.axvline(np.median(j_cdg), color='crimson', ls='--', lw=2)
    if theta_recommend is not None:
        ax.axvline(theta_recommend, color='black', ls=':', lw=2,
                   label=f'recommended θ_j = {theta_recommend:.3f}')
    ax.set_xlabel(f'{args.k}-mer Jaccard similarity')
    ax.set_ylabel('density')
    ax.set_title(f'Jaccard distribution by pair type (k={args.k})')
    ax.legend(); ax.grid(alpha=0.3)
    fig_path = os.path.join(args.output_dir, 'jaccard_distribution.png')
    plt.tight_layout(); plt.savefig(fig_path, dpi=120); plt.close()
    print(f"\n📊 {fig_path}")

    # ── 输出决策 ────────────────────────────────────
    print("\n" + "="*70)
    print("🎯 v20.A (Jaccard mask) 可行性判定")
    print("="*70)
    print(f"\n  {decision}\n")

    summary = os.path.join(args.output_dir, 'jaccard_spike_summary.txt')
    with open(summary, 'w') as f:
        f.write(f"k = {args.k}\n")
        f.write(f"n_pairs_per_group = {args.n_pairs_per_group}\n\n")
        for name, vals in [("same_cluster", j_sc),
                            ("cross_cluster_same_GT", j_csg),
                            ("cross_cluster_diff_GT", j_cdg)]:
            if len(vals):
                f.write(f"{name}:\n")
                f.write(f"  n      = {len(vals)}\n")
                f.write(f"  mean   = {vals.mean():.6f}\n")
                f.write(f"  median = {np.median(vals):.6f}\n")
                f.write(f"  std    = {vals.std():.6f}\n\n")
        if delta is not None:
            f.write(f"Delta_median = {delta:.6f}\n")
            f.write(f"Recommended theta_j = {theta_recommend:.6f}\n")
        f.write(f"\nDecision: {decision}\n")
    print(f"💾 {summary}\n")


if __name__ == '__main__':
    main()