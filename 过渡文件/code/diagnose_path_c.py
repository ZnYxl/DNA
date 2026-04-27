#!/usr/bin/env python
"""
diagnose_path_c.py —— Path C 可行性 minimal spike

只读 v19 已生成的 read_state_*.pt + refined_labels_*.txt + GT tags,
不重跑 GPU 推理. 半小时内出结论.

Q1 (核心假设): purity 是否随 U_epi 单调下降?
              ── 如果不单调, C 路径的"U_epi 高 → label 不可信"假设破产
Q2 (训得动): high_conf 约束下 batch 内 positive pair 存活率 ≥ 30%?
              ── 如果不达标, C 会因 pair 塌缩导致 loss=0
Q3 (稳定性): U_epi/U_ale 在 R1→R2→R3 的 median 漂移 < 30%?
              ── 如果飘大, threshold 必须 round-adaptive (实现复杂)

决策:
   Q1 ✅ + Q2 ✅           → 🟢 commit Path C
   Q1 ✅ + Q2 ❌           → 🟡 改 soft 版本 (不删 pair, 加激进权重)
   Q1 ❌                  → 🔴 假设破产, 退 Path A

注意: GT proxy 用 tag-as-id (保守估计). 真实 GT-level purity 应更低.
       如果 tag 级 purity 都不单调, GT 级更不会.
"""
import os, sys, argparse, glob
from collections import Counter, defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_state(path):
    s = torch.load(path, map_location='cpu')
    return {k: np.asarray(s[k]) if k in s else None
            for k in ['u_epi', 'u_ale', 'zone_ids', 'strength']}


def auto_discover(experiment_dir):
    """自动找最新的 R1/R2/R3 state + labels"""
    labels_dir = os.path.join(experiment_dir, '04_Iterative_Labels')
    state_files = sorted(glob.glob(os.path.join(labels_dir, 'read_state_*.pt')),
                         key=os.path.getmtime)
    label_files = sorted(glob.glob(os.path.join(labels_dir, 'refined_labels_*.txt')),
                         key=os.path.getmtime)
    paired = []
    for sf in state_files:
        ts = os.path.basename(sf).replace('read_state_', '').replace('.pt', '')
        lf = os.path.join(labels_dir, f'refined_labels_{ts}.txt')
        if os.path.exists(lf):
            paired.append((ts, sf, lf))
    return paired[-3:]  # 最后三轮


def load_gt_tags(path, n_reads):
    """读 tag 文件, 返回 (read_idx → tag_proxy_id) 数组"""
    tags = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) >= 2:
                tags.append(parts[0])
            else:
                tags.append(parts[0])
    # 补齐 / 截断
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
    return gt, len(uniq)


def cluster_purity_dict(labels, gt):
    """每个 cluster 的 max-class fraction"""
    cmap = defaultdict(list)
    for l, g in zip(labels, gt):
        if l >= 0 and g >= 0:
            cmap[int(l)].append(int(g))
    return {c: Counter(gs).most_common(1)[0][1] / len(gs)
            for c, gs in cmap.items() if gs}


# ============================================================================
# Q1: Purity vs U_epi
# ============================================================================
def q1(state, labels, gt, out_dir, n_bins=5):
    print("\n" + "═" * 70)
    print("Q1: Purity vs U_epi 分箱 (Path C 核心假设)")
    print("═" * 70)
    valid = (labels >= 0) & (gt >= 0)
    n_valid = int(valid.sum())
    print(f"  有效 reads (label≥0 ∧ gt≥0): {n_valid:,}")
    if n_valid < 1000:
        print("  ⚠️ 有效样本太少, Q1 不可信")
        return None, False

    u_epi = state['u_epi']
    cp = cluster_purity_dict(labels, gt)
    print(f"  簇 purity 计算完毕: {len(cp):,} 个簇")

    # 分箱
    qs = np.quantile(u_epi[valid], np.linspace(0, 1, n_bins + 1))
    purities = []
    for b in range(n_bins):
        lo, hi = qs[b], qs[b + 1]
        m = valid & (u_epi >= lo) & (u_epi <= (hi + 1e-9 if b == n_bins - 1 else hi))
        ps = [cp.get(int(l), 0.0) for l in labels[m]]
        purity = float(np.mean(ps)) if ps else 0.0
        purities.append(purity)
        print(f"  Bin {b+1} U_epi∈({lo:.5f}, {hi:.5f}): "
              f"n={int(m.sum()):,}, mean_purity={purity:.4f}")

    # 单调性: 允许 0.005 容差
    monotone = all(purities[i] >= purities[i+1] - 0.005 for i in range(n_bins-1))
    delta = purities[0] - purities[-1]
    print(f"\n  单调下降: {'✅ YES' if monotone else '❌ NO'}  "
          f"(P[bin1] - P[bin5] = {delta:+.4f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(range(1, n_bins + 1))
    ax.plot(xs, purities, 'o-', lw=2.5, ms=10, color='steelblue')
    for i, p in enumerate(purities):
        ax.annotate(f'{p:.3f}', xy=(i+1, p), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=10)
    ax.set_xlabel('U_epi quintile (1=lowest, 5=highest)')
    ax.set_ylabel('Mean cluster purity')
    ax.set_title(f"Q1: Purity vs U_epi  (Δ={delta:+.4f}, "
                 f"{'monotone' if monotone else 'NON-monotone'})")
    ax.set_ylim([min(purities) - 0.02, 1.02])
    ax.grid(alpha=0.3)
    fig_path = os.path.join(out_dir, 'q1_purity_vs_uepi.png')
    plt.tight_layout(); plt.savefig(fig_path, dpi=120); plt.close()
    print(f"  📊 {fig_path}")
    return purities, monotone


# ============================================================================
# Q2: Pair 存活率
# ============================================================================
def q2(state, labels, out_dir, batch_size=256, n_sims=100):
    print("\n" + "═" * 70)
    print("Q2: high_conf 约束下 pair 存活率")
    print("═" * 70)
    valid_idx = np.where(labels >= 0)[0]
    u_epi, u_ale = state['u_epi'], state['u_ale']
    print(f"  采样池: {len(valid_idx):,} reads, batch_size={batch_size}, "
          f"模拟 {n_sims} batches")

    # 三档阈值: P50/P70/P85 (双向)
    combos = [
        ('严格', np.quantile(u_epi[valid_idx], 0.50),
                  np.quantile(u_ale[valid_idx], 0.50)),
        ('中等', np.quantile(u_epi[valid_idx], 0.70),
                  np.quantile(u_ale[valid_idx], 0.70)),
        ('宽松', np.quantile(u_epi[valid_idx], 0.85),
                  np.quantile(u_ale[valid_idx], 0.85)),
    ]
    rng = np.random.default_rng(42)
    self_mask = np.eye(batch_size, dtype=bool)
    results = []

    for name, te, ta in combos:
        pf, pn, nf, nn, zero_pos = 0, 0, 0, 0, 0
        for _ in range(n_sims):
            sample = rng.choice(valid_idx, size=batch_size, replace=False)
            l = labels[sample]
            hc = (u_epi[sample] < te) & (u_ale[sample] < ta)
            l_eq = l[:, None] == l[None, :]
            hc_pair = hc[:, None] & hc[None, :]
            pf += int((l_eq & ~self_mask).sum() / 2)
            nf += int(((~l_eq) & ~self_mask).sum() / 2)
            p_hc = int((l_eq & hc_pair & ~self_mask).sum() / 2)
            pn += p_hc
            nn += int(((~l_eq) & hc_pair & ~self_mask).sum() / 2)
            if p_hc == 0:
                zero_pos += 1
        ps = pn / max(pf, 1)
        ns = nn / max(nf, 1)
        print(f"\n  [{name}] θ_e={te:.4f}, θ_a={ta:.4f}")
        print(f"     Pos pair 存活: {ps:.2%} "
              f"({pn/n_sims:.0f}/{pf/n_sims:.0f} per batch)")
        print(f"     Neg pair 存活: {ns:.2%}")
        print(f"     Zero-pos batches: {zero_pos}/{n_sims} "
              f"({'⚠️ 危险' if zero_pos > n_sims * 0.1 else 'OK'})")
        results.append({'name': name, 'te': te, 'ta': ta,
                        'pos_survival': ps, 'neg_survival': ns,
                        'zero_pos': zero_pos})
    return results


# ============================================================================
# Q3: U 漂移
# ============================================================================
def q3(states, out_dir):
    if len(states) < 2:
        print("\n  ⚠️ 只有 1 轮 state, 跳过 Q3")
        return None
    print("\n" + "═" * 70)
    print("Q3: U 分布在 R1→R3 漂移")
    print("═" * 70)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {1: 'steelblue', 2: 'darkorange', 3: 'crimson'}
    rd = {}
    for r, st in states.items():
        ue = st['u_epi'][st['u_epi'] > 0]
        ua = st['u_ale'][st['u_ale'] > 0]
        rd[r] = (np.median(ue), np.median(ua))
        print(f"  R{r}: U_epi median={rd[r][0]:.4f}, "
              f"U_ale median={rd[r][1]:.4f}, n={len(ue):,}")
        axes[0].hist(ue, bins=80, alpha=0.4, density=True,
                     label=f'R{r}', color=colors.get(r, 'gray'))
        axes[1].hist(ua, bins=80, alpha=0.4, density=True,
                     label=f'R{r}', color=colors.get(r, 'gray'))
    for ax, name in zip(axes, ['U_epi', 'U_ale']):
        ax.set_xlabel(name); ax.set_ylabel('density')
        ax.set_title(f'Q3: {name} drift')
        ax.legend(); ax.grid(alpha=0.3)
    fig_path = os.path.join(out_dir, 'q3_u_drift.png')
    plt.tight_layout(); plt.savefig(fig_path, dpi=120); plt.close()
    print(f"  📊 {fig_path}")

    me = [rd[r][0] for r in sorted(rd.keys())]
    ma = [rd[r][1] for r in sorted(rd.keys())]
    de = (max(me) - min(me)) / np.mean(me) * 100
    da = (max(ma) - min(ma)) / np.mean(ma) * 100
    print(f"\n  U_epi median 相对漂移: {de:.1f}%")
    print(f"  U_ale median 相对漂移: {da:.1f}%")
    return de, da


# ============================================================================
# 主入口
# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment_dir', required=True)
    p.add_argument('--gt_tags_file', required=True)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--batch_size', type=int, default=256)
    args = p.parse_args()

    out_dir = args.output_dir or os.path.join(
        args.experiment_dir, 'results', 'spike_path_c')
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 输出目录: {out_dir}")

    paired = auto_discover(args.experiment_dir)
    if not paired:
        print("❌ 未找到 read_state_*.pt + refined_labels_*.txt 配对")
        sys.exit(1)
    print(f"\n🔍 发现 {len(paired)} 轮:")
    for ts, sf, lf in paired:
        print(f"   ts={ts}  state={os.path.basename(sf)}  "
              f"labels={os.path.basename(lf)}")

    # 用最后一轮做 Q1/Q2 (最完整训练后的 U)
    ts_last, sf_last, lf_last = paired[-1]
    print(f"\n📂 主分析使用 R{len(paired)} (ts={ts_last})")
    state_main = load_state(sf_last)
    labels_main = np.loadtxt(lf_last, dtype=int)
    print(f"  reads: {len(labels_main):,}, "
          f"label≥0: {int((labels_main>=0).sum()):,}")

    # GT proxy
    gt, n_uniq = load_gt_tags(args.gt_tags_file, len(labels_main))
    print(f"  GT proxy (tag-as-id): {n_uniq:,} unique tags, "
          f"valid: {int((gt>=0).sum()):,}")

    # Q1
    purities, q1_ok = q1(state_main, labels_main, gt, out_dir)

    # Q2
    q2_results = q2(state_main, labels_main, out_dir,
                    batch_size=args.batch_size)

    # Q3
    states_per_round = {}
    for i, (ts, sf, _) in enumerate(paired, start=1):
        states_per_round[i] = load_state(sf)
    q3_result = q3(states_per_round, out_dir)

    # 决策
    print("\n" + "═" * 70)
    print("🎯 Path C 可行性判定")
    print("═" * 70)
    pos_mid = q2_results[1]['pos_survival']
    q2_ok = pos_mid >= 0.30
    print(f"  Q1 (purity 单调下降):                     "
          f"{'✅' if q1_ok else '❌'}")
    print(f"  Q2 (中等 θ 下 pos pair 存活 ≥ 30%):      "
          f"{'✅' if q2_ok else '❌'}  ({pos_mid:.2%})")
    if q3_result:
        de, da = q3_result
        q3_ok = de < 30 and da < 30
        print(f"  Q3 (R1→R{len(paired)} U median 漂移 < 30%):     "
              f"{'✅' if q3_ok else '❌'}  "
              f"(epi {de:.1f}%, ale {da:.1f}%)")
    print()
    if q1_ok and q2_ok:
        decision = "🟢 commit Path C (用 U 重新定义 pair, 不只是加权)"
    elif q1_ok and not q2_ok:
        decision = "🟡 Path C soft 版本 (不删 pair, 加激进权重 r→16)"
    else:
        decision = "🔴 退回 Path A (rebirth_mode=bounded + 训练靶子混合)"
    print(f"  推荐: {decision}\n")

    # 保存
    summary = os.path.join(out_dir, 'spike_summary.txt')
    with open(summary, 'w') as f:
        f.write(f"Q1 monotone: {q1_ok}\n")
        f.write(f"  purities (bin1→bin5): {purities}\n\n")
        f.write(f"Q2 (batch_size={args.batch_size}):\n")
        for r in q2_results:
            f.write(f"  {r}\n")
        if q3_result:
            f.write(f"\nQ3: drift_epi={q3_result[0]:.1f}%, "
                    f"drift_ale={q3_result[1]:.1f}%\n")
        f.write(f"\nDecision: {decision}\n")
    print(f"  💾 总结: {summary}")


if __name__ == '__main__':
    main()