#!/usr/bin/env python
"""
diagnose_path_d.py —— Path D 可行性 spike (centroid-level, 无 GPU)

只读 v19 已生成的 centroids_*.pt + refined_labels_*.txt + GT tags.
不重跑推理. 全程 5-10 分钟.

D 路径假设: 当两个不同 cluster 的 centroid 余弦 > 0.95 时, 它们大概率应该合并
            (Clover 过分割的同一 GT 分子). 我们要把这种 cross-cluster 当 positive,
            让 contrastive 主动告诉 clustering "这俩该合一个".

Q4 (有空间): R3 centroids 中 cos > 0.95 的 cluster pair 有多少?
              如果 < 50 对, D 路径无显著合并空间, 价值低.
Q5 (合理性): 这些高相似 pair 中, 多少其主导 GT 实际是同一个?
              如果 < 50%, "高相似 ≠ 同源", D 会污染聚类.

决策:
  Q4 ≥ 200 + Q5 ≥ 70%   🟢 commit D
  Q4 ≥ 50  + Q5 ≥ 70%   🟡 D 收益边际, 考虑 A 更稳
  Q5 < 50%               🔴 退 A
"""
import os, sys, argparse, glob
from collections import Counter, defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def auto_find(experiment_dir, pattern):
    files = sorted(
        glob.glob(os.path.join(experiment_dir, '04_Iterative_Labels', pattern)),
        key=os.path.getmtime)
    return files[-1] if files else None


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


def cluster_gt_distribution(labels, gt):
    """每个 cluster 的主导 GT 及其占比"""
    cmap = defaultdict(list)
    for l, g in zip(labels, gt):
        if l >= 0 and g >= 0:
            cmap[int(l)].append(int(g))
    out = {}
    for c, gs in cmap.items():
        if not gs: continue
        c_top, n_top = Counter(gs).most_common(1)[0]
        out[c] = (c_top, n_top, len(gs))  # (gt_id, n_top, n_total)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment_dir', required=True)
    p.add_argument('--gt_tags_file', required=True)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--centroids_path', default=None,
                   help='不指定则自动用最新一轮')
    p.add_argument('--labels_path', default=None)
    args = p.parse_args()

    out_dir = args.output_dir or os.path.join(
        args.experiment_dir, 'results', 'spike_path_d')
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 输出目录: {out_dir}")

    cents_path = args.centroids_path or auto_find(
        args.experiment_dir, 'centroids_*.pt')
    labels_path = args.labels_path or auto_find(
        args.experiment_dir, 'refined_labels_*.txt')
    if not cents_path or not labels_path:
        print(f"❌ 未找到 centroids/labels"); sys.exit(1)
    print(f"📂 centroids: {os.path.basename(cents_path)}")
    print(f"📂 labels:    {os.path.basename(labels_path)}")

    # 加载 centroids
    cd = torch.load(cents_path, map_location='cpu')
    centroids = cd['centroids']
    cids = sorted(centroids.keys())
    K = len(cids)
    cm = torch.stack([centroids[c] for c in cids])  # (K, D)
    print(f"  K={K} clusters, D={cm.shape[1]} dims")

    # ── 算 centroid-centroid cosine similarity ──
    cm_norm = torch.nn.functional.normalize(cm, dim=-1)
    print(f"  计算 K×K cosine matrix ({K*K*4/1024**2:.0f} MB)...")
    cos_sim = (cm_norm @ cm_norm.T).numpy()
    np.fill_diagonal(cos_sim, -1.0)

    # 上三角
    iu, ju = np.triu_indices(K, k=1)
    sims = cos_sim[iu, ju]
    total_pairs = len(sims)

    # ── 加载 GT ──
    labels = np.loadtxt(labels_path, dtype=int)
    gt = load_gt_tags(args.gt_tags_file, len(labels))
    cluster_gt = cluster_gt_distribution(labels, gt)
    print(f"  簇主导 GT 已计算: {len(cluster_gt):,} 个簇")

    # ============================================================
    # Q4: 各阈值下 cluster pair 数
    # ============================================================
    print("\n" + "═" * 70)
    print("Q4: 高相似度 cluster pair 数量")
    print("═" * 70)
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
    print(f"\n  {'Threshold':<14}{'#Pairs':>12}{'% of all':>14}")
    print("  " + "─" * 40)
    pair_counts = {}
    for t in thresholds:
        n = int((sims > t).sum())
        pair_counts[t] = n
        print(f"  cos > {t:<8}{n:>12,}{100.0*n/total_pairs:>13.4f}%")

    # ============================================================
    # Q5: 各阈值下 GT 命中率
    # ============================================================
    print("\n" + "═" * 70)
    print("Q5: 高相似度 cluster pair 的 GT 主导一致率")
    print("═" * 70)
    results = []
    for t in thresholds:
        mask = sims > t
        if mask.sum() == 0:
            continue
        ip, jp = iu[mask], ju[mask]
        n_pair = len(ip)
        n_same, n_known = 0, 0
        for i, j in zip(ip, jp):
            ci, cj = cids[i], cids[j]
            if ci in cluster_gt and cj in cluster_gt:
                n_known += 1
                if cluster_gt[ci][0] == cluster_gt[cj][0]:
                    n_same += 1
        hit = n_same / max(n_known, 1)
        results.append({'t': t, 'n_pair': n_pair, 'n_known': n_known,
                        'n_same': n_same, 'hit': hit})
        print(f"  cos > {t}: {n_pair:>7,} pairs | "
              f"已知GT: {n_known:>7,} | 同GT: {n_same:>6,} ({hit:.2%})")

    # ============================================================
    # 画图
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: 相似度直方图
    axes[0].hist(sims, bins=100, color='steelblue', alpha=0.7,
                 edgecolor='none')
    for t, c in [(0.95, 'red'), (0.90, 'orange'), (0.85, 'gold')]:
        axes[0].axvline(t, color=c, linestyle='--', label=f'cos={t}')
    axes[0].set_xlabel('Centroid cosine similarity')
    axes[0].set_ylabel('# pairs (log)')
    axes[0].set_yscale('log')
    axes[0].set_title(f'Q4: K={K} clusters → {total_pairs:,} pairs')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # 右: hit rate
    if results:
        ts = [r['t'] for r in results]
        hrs = [r['hit'] for r in results]
        axes[1].plot(ts, hrs, 'o-', lw=2.5, ms=10, color='darkgreen')
        for r in results:
            axes[1].annotate(
                f"n={r['n_pair']}\n{r['hit']:.0%}",
                xy=(r['t'], r['hit']), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=9)
        axes[1].axhline(0.70, color='red', linestyle='--',
                        label='target 70%')
        axes[1].axhline(0.50, color='gray', linestyle=':',
                        label='floor 50%')
        axes[1].set_xlabel('Cosine threshold')
        axes[1].set_ylabel('Same-GT hit rate')
        axes[1].set_title('Q5: Hit rate (D 路径上限)')
        axes[1].set_ylim([0, 1.05])
        axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'q45_centroid_pair.png')
    plt.savefig(fig_path, dpi=120); plt.close()
    print(f"\n  📊 {fig_path}")

    # ============================================================
    # 决策
    # ============================================================
    print("\n" + "═" * 70)
    print("🎯 Path D 可行性判定 (主阈值 cos=0.95)")
    print("═" * 70)
    main = next((r for r in results if abs(r['t'] - 0.95) < 1e-6), None)
    if main:
        n_pair, hit = main['n_pair'], main['hit']
        print(f"  Q4 (pair ≥ 200): "
              f"{'✅' if n_pair >= 200 else ('🟡' if n_pair >= 50 else '❌')} "
              f"(实际 {n_pair:,})")
        print(f"  Q5 (hit ≥ 70%):  "
              f"{'✅' if hit >= 0.70 else ('🟡' if hit >= 0.50 else '❌')} "
              f"({hit:.2%})")

        if hit < 0.50:
            decision = ("🔴 退 Path A. cos>0.95 cluster pair 大多 GT 不一致, "
                        "D 路径会引入大量 false positive 污染 contrastive.")
        elif n_pair >= 200 and hit >= 0.70:
            decision = ("🟢 commit Path D. 真有大量合并空间, hit rate 高, "
                        "cross-cluster positive 是 SSI-EC 的 novel mechanism.")
        elif n_pair >= 50 and hit >= 0.70:
            decision = ("🟡 D 可做但收益边际 (n<200). "
                        "建议: Path A 主线 + D 作为附加 ablation 探索.")
        elif hit >= 0.50 and n_pair >= 200:
            decision = ("🟡 hit rate 中等 (50-70%). "
                        "D 路径需配合更严阈值 (cos>0.97), 看那个阈值能否达 70%.")
        else:
            decision = "🔴 D 路径不强, 建议 Path A."
        print(f"\n  推荐: {decision}\n")
    else:
        decision = "无法判定 (cos=0.95 阈值未在结果中)"
        print(f"  {decision}\n")

    # 保存
    summary_path = os.path.join(out_dir, 'spike_d_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"K={K}, total pairs={total_pairs:,}\n\n")
        f.write("Q4 pair count by threshold:\n")
        for t, n in pair_counts.items():
            f.write(f"  cos>{t}: {n:,}\n")
        f.write("\nQ5 hit rate by threshold:\n")
        for r in results:
            f.write(f"  cos>{r['t']}: {r['n_pair']:,} pairs, "
                    f"hit={r['hit']:.4f} "
                    f"({r['n_same']:,}/{r['n_known']:,})\n")
        f.write(f"\nDecision: {decision}\n")
    print(f"  💾 总结: {summary_path}")


if __name__ == '__main__':
    main()