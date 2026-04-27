#!/usr/bin/env python
"""
diagnose_path_e_light.py —— Path E 轻量 spike

E 的赌注: R1 训完的 encoder 已经"够好", R2/R3 重训反而过度调整、
         把同源 cluster 推得更开. 如果这赌注成立, R1→R3 encoder 应该
         变化"有方向但量不大".

衡量: 对每个 R1 cluster centroid, 找它在 R3 centroids 中的 nearest neighbor,
      看 cosine 分布.

  - 大部分 R1 → R3 nearest cos > 0.95:   encoder 几乎没变, E 可行性高 ✅
  - 中位数 0.85-0.95:                    encoder 中度调整, 需重量验证 🟡
  - 大部分 nearest cos < 0.85:           encoder 大改, frozen 损失大 🔴

不需要 GPU. 5 分钟内出结果.
"""
import os, sys, argparse, glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def auto_find_all(experiment_dir, pattern):
    files = sorted(
        glob.glob(os.path.join(experiment_dir, '04_Iterative_Labels', pattern)),
        key=os.path.getmtime)
    return files


def load_centroids_normalized(path):
    cd = torch.load(path, map_location='cpu')
    cents = cd['centroids']
    cids = sorted(cents.keys())
    mat = torch.stack([cents[c] for c in cids])
    return torch.nn.functional.normalize(mat, dim=-1).numpy(), cids


def nearest_cos(src_mat, tgt_mat):
    """对 src 中每行, 找 tgt 中余弦最高的, 返回 (n_src,) 的最大相似度数组"""
    # 分块算, 防大矩阵 OOM
    n_src = len(src_mat)
    chunk = 2000
    out = np.zeros(n_src, dtype=np.float32)
    for s in range(0, n_src, chunk):
        e = min(s + chunk, n_src)
        sim = src_mat[s:e] @ tgt_mat.T  # (chunk, n_tgt)
        out[s:e] = sim.max(axis=1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment_dir', required=True)
    p.add_argument('--output_dir', default=None)
    args = p.parse_args()

    out_dir = args.output_dir or os.path.join(
        args.experiment_dir, 'results', 'spike_path_e_light')
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 输出目录: {out_dir}")

    # 找所有 centroids
    cent_files = auto_find_all(args.experiment_dir, 'centroids_*.pt')
    if len(cent_files) < 2:
        print(f"❌ centroids 文件 < 2 ({len(cent_files)}), 无法对比")
        sys.exit(1)
    print(f"\n🔍 发现 {len(cent_files)} 轮 centroids:")
    for i, f in enumerate(cent_files, 1):
        print(f"   R{i}: {os.path.basename(f)}")

    # 加载所有
    all_cents = []
    for f in cent_files:
        mat, cids = load_centroids_normalized(f)
        all_cents.append((mat, cids))
        print(f"   loaded {os.path.basename(f)}: K={len(cids)}, D={mat.shape[1]}")

    R = len(all_cents)

    # ============================================================
    # 关键分析: 对每个 R1 centroid, 找它在 R{i} 中的 nearest neighbor
    # ============================================================
    print("\n" + "═" * 70)
    print("R1 → R{i} centroid 漂移分析")
    print("═" * 70)

    R1_mat = all_cents[0][0]
    drift_results = []

    fig, axes = plt.subplots(1, R - 1, figsize=(6 * (R - 1), 5), squeeze=False)
    axes = axes[0]

    for i in range(1, R):
        Ri_mat = all_cents[i][0]
        cos_max = nearest_cos(R1_mat, Ri_mat)
        median = np.median(cos_max)
        p25, p75 = np.quantile(cos_max, [0.25, 0.75])
        frac_high = (cos_max > 0.95).mean()
        frac_mid = ((cos_max > 0.85) & (cos_max <= 0.95)).mean()
        frac_low = (cos_max <= 0.85).mean()

        print(f"\n  R1 → R{i+1}:")
        print(f"     median nearest cos: {median:.4f}  "
              f"(P25={p25:.4f}, P75={p75:.4f})")
        print(f"     cos > 0.95:  {frac_high:.2%}  ({int(frac_high*len(R1_mat)):,} / "
              f"{len(R1_mat):,})")
        print(f"     0.85-0.95:   {frac_mid:.2%}")
        print(f"     cos < 0.85:  {frac_low:.2%}")

        drift_results.append({
            'round': i + 1, 'median': median, 'p25': p25, 'p75': p75,
            'frac_high': frac_high, 'frac_mid': frac_mid, 'frac_low': frac_low,
        })

        ax = axes[i - 1]
        ax.hist(cos_max, bins=80, color='steelblue', alpha=0.7,
                edgecolor='none')
        ax.axvline(0.95, color='red', linestyle='--', label='0.95')
        ax.axvline(0.85, color='orange', linestyle='--', label='0.85')
        ax.axvline(median, color='black', linestyle='-',
                   label=f'median={median:.3f}')
        ax.set_xlabel(f'max cos(R1, R{i+1}) per R1 centroid')
        ax.set_ylabel('# R1 centroids')
        ax.set_title(f'R1 → R{i+1} drift (K_R1={len(R1_mat):,})')
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'r1_to_rN_drift.png')
    plt.savefig(fig_path, dpi=120); plt.close()
    print(f"\n  📊 {fig_path}")

    # ============================================================
    # 决策 (基于 R1→R3 漂移)
    # ============================================================
    print("\n" + "═" * 70)
    print("🎯 Path E 轻量 spike 判定")
    print("═" * 70)

    # 用最后一轮 (R3) 做主要判定
    final = drift_results[-1]
    target_round = final['round']
    median = final['median']
    high = final['frac_high']
    low = final['frac_low']

    print(f"\n  R1 → R{target_round} encoder 漂移:")
    print(f"     median nearest cos = {median:.4f}")
    print(f"     fraction with cos > 0.95: {high:.2%}")
    print(f"     fraction with cos < 0.85: {low:.2%}")

    if high >= 0.70 or median >= 0.95:
        verdict = "✅ encoder 几乎没变"
        decision = ("🟢 E 可行性高. R1 encoder 已收敛, R2/R3 重训仅小幅调整. "
                    "建议: 跑重量 spike (R2' step2) 验证 SR 是否上升.")
    elif median >= 0.85 and low <= 0.30:
        verdict = "🟡 encoder 中度调整"
        decision = ("🟡 E 不确定, 必须跑重量 spike. "
                    "encoder 在 R2/R3 有非平凡调整, frozen 是否损失需实测.")
    else:
        verdict = "❌ encoder 大幅改变"
        decision = ("🔴 E 风险高. encoder 在 R2/R3 重训中显著调整, "
                    "frozen R1 会丢失后续学到的表示. 建议直接走 Path A 或 G.")

    print(f"\n  Verdict: {verdict}")
    print(f"  Recommendation: {decision}\n")

    # 保存
    summary_path = os.path.join(out_dir, 'spike_e_light_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Path E light spike summary\n")
        f.write(f"="*60 + "\n\n")
        for r in drift_results:
            f.write(f"R1 → R{r['round']}: median nearest cos = {r['median']:.4f}\n")
            f.write(f"  cos > 0.95: {r['frac_high']:.2%}\n")
            f.write(f"  0.85-0.95:  {r['frac_mid']:.2%}\n")
            f.write(f"  cos < 0.85: {r['frac_low']:.2%}\n\n")
        f.write(f"\nVerdict: {verdict}\n")
        f.write(f"Decision: {decision}\n")
    print(f"  💾 {summary_path}")


if __name__ == '__main__':
    main()