#!/usr/bin/env python
"""
spike_g_R1_vs_R3.py —— 原 spike_g_isometric.py 的最小改动版

唯一改动: 增加 --iter 参数选择用哪一轮的 ckpt (默认 R1 = iter_1)。
          原脚本写死 ckpts[-1] (R3)。其余逻辑/口径/采样/AUROC 完全不变,
          保证 R1 和 R3 的 AUROC 同口径可比。

用法:
  # 测 R1 (默认)
  python spike_g_R1_vs_R3.py --experiment_dir <dir> --gt_tags <output.txt>
  # 测 R3 (复现原 0.77, 做 sanity check)
  python spike_g_R1_vs_R3.py --experiment_dir <dir> --gt_tags <output.txt> --iter 3

判读:
  - R1 AUROC 也低 (~0.77) -> 表征从一开始就处理不了被切开的 GT,
                             与迭代无关. v21 = 识别跨簇同源当正样本 (用 edit).
  - R1 AUROC 高 (~0.9), R3 才低 -> 是迭代把表征搞坏的.
                             v21 = 迭代稳定性 (防 encoder 在 R2/R3 退化).
"""
import os, sys, argparse, glob
from collections import defaultdict
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


def load_gt(path, n_reads):
    tags = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            tags.append(line.split('\t')[0] if '\t' in line else line)
    if len(tags) < n_reads:
        tags = tags + ['__PAD__'] * (n_reads - len(tags))
    elif len(tags) > n_reads:
        tags = tags[:n_reads]
    uniq = sorted(set(tags))
    tag_id = {t: i for i, t in enumerate(uniq)}
    gt = np.array([tag_id[t] for t in tags], dtype=int)
    pad = tag_id.get('__PAD__', -1)
    if pad >= 0: gt[gt == pad] = -1
    return gt


@torch.no_grad()
def infer_embeddings(model, data_loader, indices, device, model_max_len, batch_size=512):
    from models.step1_data import seq_to_onehot
    model.eval(); model.to(device)
    N = len(indices)
    embs = torch.zeros(N, model.dim, dtype=torch.float32)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        encs = [seq_to_onehot(data_loader.reads[r], model_max_len)
                for r in indices[s:e]]
        enc_t = torch.stack(encs).to(device)
        if enc_t.shape[1] != model_max_len:
            if enc_t.shape[1] < model_max_len:
                pad = torch.zeros(enc_t.shape[0],
                                   model_max_len - enc_t.shape[1], 4,
                                   device=device)
                enc_t = torch.cat([enc_t, pad], dim=1)
            else:
                enc_t = enc_t[:, :model_max_len, :]
        _, pooled = model.encode_reads(enc_t)
        embs[s:e] = pooled.cpu().float()
    return embs


def kmer_bitvec(seq, k=5):
    base = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    vec = np.zeros(4 ** k, dtype=np.float32)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        idx, valid = 0, True
        for c in kmer:
            if c not in base: valid = False; break
            idx = idx * 4 + base[c]
        if valid: vec[idx] = 1.0
    return vec


def jaccard(v1, v2):
    inter = (v1 * v2).sum()
    union = ((v1 + v2) > 0).sum()
    return float(inter / max(union, 1))


def edit_sim(s1, s2):
    try:
        import edlib
        ed = edlib.align(s1, s2, task='distance')['editDistance']
        return 1.0 - ed / max(len(s1), len(s2))
    except ImportError:
        return None


def auroc(same, diff):
    s = np.concatenate([same, diff])
    y = np.concatenate([np.ones(len(same)), np.zeros(len(diff))])
    order = np.argsort(-s)
    y = y[order]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0: return 0.5
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    return float(np.trapz(tp / n_pos, fp / n_neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment_dir', required=True)
    p.add_argument('--gt_tags', required=True)
    p.add_argument('--code_dir', default='/mnt/st_data/liangxinyi/code')
    p.add_argument('--n_pairs', type=int, default=3000)
    p.add_argument('--n_pairs_edit', type=int, default=500)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--iter', type=int, default=1,
                   help='用哪一轮的 ckpt: 1=R1(默认), 3=R3. 这是唯一新增的开关')
    p.add_argument('--output_dir', default=None)
    args = p.parse_args()

    out = args.output_dir or os.path.join(
        args.experiment_dir, 'results', f'spike_g_iter{args.iter}')
    os.makedirs(out, exist_ok=True)
    print(f"📁 输出: {out}")
    print(f"🔧 测试轮次: R{args.iter}\n")

    if args.code_dir not in sys.path:
        sys.path.insert(0, args.code_dir)

    # ── labels: 仍用最新 (R3) 的 refined_labels 做 Clover/GT 对齐采样 ──
    # 注意: pair 采样依赖的是 Clover 簇结构与 GT, 与 ckpt 轮次无关,
    #       保持和原脚本一致, 只换推理用的 ckpt. 这样 R1/R3 采样的 pair
    #       集合完全相同, embedding 是唯一变量, 严格单变量对比.
    r3_labels = auto_find(args.experiment_dir, 'refined_labels_*.txt')
    if not r3_labels:
        print("❌ 找不到 refined_labels"); sys.exit(1)
    print(f"📂 labels (采样用, 固定): {os.path.basename(r3_labels)}")

    ckpts = sorted(glob.glob(os.path.join(
        args.experiment_dir, 'results', 'iter_*_step1', 'models',
        'step1_final_model.pth')), key=os.path.getmtime)
    if not ckpts:
        print("❌ 找不到 step1 ckpt"); sys.exit(1)
    # ★ 唯一改动: 按 --iter 选 ckpt (原脚本写死 ckpts[-1])
    if args.iter < 1 or args.iter > len(ckpts):
        print(f"❌ --iter={args.iter} 超出范围 (共 {len(ckpts)} 个 ckpt)"); sys.exit(1)
    sel_ckpt = ckpts[args.iter - 1]
    print(f"📂 选中 ckpt (R{args.iter}): {sel_ckpt}")
    print(f"   (所有可用 ckpt: {[os.path.relpath(c, args.experiment_dir) for c in ckpts]})\n")

    from models.step1_data import CloverDataLoader
    from models.step1_model import Step1EvidentialModel

    dl = CloverDataLoader(args.experiment_dir, labels_path=r3_labels)
    labels = np.array(dl.clover_labels)
    print(f"📊 reads: {len(dl.reads):,}, label≥0: {(labels>=0).sum():,}")
    # [FIX] 用序列做 key 的官方 GT 加载 (今天确认 100% 匹配),
    #       取代原 load_gt 的按行号对齐 (output.txt 280万行与 reads 行序不对齐 -> 错).
    dl.load_gt_tags(args.gt_tags)
    gt = np.array(dl.gt_labels)
    print(f"📊 GT valid: {(gt>=0).sum():,}  (序列做key, 应=330788)\n")

    print("📐 采样 pair...")
    cluster_to_idx = defaultdict(list)
    gt_to_clusters = defaultdict(lambda: defaultdict(list))
    for i, (l, g) in enumerate(zip(labels, gt)):
        if l >= 0 and g >= 0:
            cluster_to_idx[int(l)].append(i)
            gt_to_clusters[int(g)][int(l)].append(i)
    multi_clusters = [c for c, rs in cluster_to_idx.items() if len(rs) >= 2]
    fragmented_gts = [g for g, cs in gt_to_clusters.items() if len(cs) >= 2]
    print(f"   multi-read clusters: {len(multi_clusters):,}")
    print(f"   fragmented GTs:      {len(fragmented_gts):,}")

    # ★ 固定随机种子, 保证 R1/R3 采到完全相同的 pair (单变量)
    rng = np.random.default_rng(42)
    pairs = {'same_cluster': [], 'cross_same_gt': [], 'cross_diff_gt': []}

    for _ in range(args.n_pairs):
        c = rng.choice(multi_clusters)
        i, j = rng.choice(cluster_to_idx[c], size=2, replace=False)
        pairs['same_cluster'].append((int(i), int(j)))

    attempts = 0
    while len(pairs['cross_same_gt']) < args.n_pairs and attempts < args.n_pairs * 10:
        attempts += 1
        g = rng.choice(fragmented_gts)
        cs = list(gt_to_clusters[g].keys())
        c1, c2 = rng.choice(cs, size=2, replace=False)
        i = rng.choice(gt_to_clusters[g][int(c1)])
        j = rng.choice(gt_to_clusters[g][int(c2)])
        pairs['cross_same_gt'].append((int(i), int(j)))

    valid_idx = np.where((labels >= 0) & (gt >= 0))[0]
    while len(pairs['cross_diff_gt']) < args.n_pairs:
        i, j = rng.choice(valid_idx, size=2, replace=False)
        if labels[i] != labels[j] and gt[i] != gt[j]:
            pairs['cross_diff_gt'].append((int(i), int(j)))

    for k, v in pairs.items(): print(f"   {k}: {len(v):,}")
    print()

    needed = sorted(set(idx for grp in pairs.values() for pair in grp for idx in pair))
    print(f"🔮 推理 R{args.iter} embeddings ({len(needed):,} reads)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(sel_ckpt, map_location=device)
    sa = ckpt.get('args', {})
    mdim = sa.get('dim', 256) if isinstance(sa, dict) else 256
    mlen = sa.get('max_length', 201) if isinstance(sa, dict) else 201
    model = Step1EvidentialModel(dim=mdim, max_length=mlen,
                                  num_clusters=max(50, len(cluster_to_idx)),
                                  device=str(device)).to(device)
    sd = ckpt['model_state_dict']
    if 'length_adapter.weight' in sd:
        sh = sd['length_adapter.weight'].shape
        if sh[1] == mlen and sh[0] == mlen:
            import torch.nn as nn
            model.length_adapter = nn.Linear(sh[1], sh[0]).to(device)
    model.load_state_dict(sd, strict=False)

    embs = infer_embeddings(model, dl, needed, device, mlen)
    pos = {ridx: i for i, ridx in enumerate(needed)}
    print(f"✅ embeddings: {tuple(embs.shape)}\n")
    del model; torch.cuda.empty_cache()

    print(f"📏 计算 4 个距离 (k={args.k})...")
    kcache = {r: kmer_bitvec(dl.reads[r], args.k) for r in needed}

    results = {}
    for grp, grp_pairs in pairs.items():
        d_l2, d_cos, d_jac, d_ed = [], [], [], []
        n_edit_sub = min(args.n_pairs_edit, len(grp_pairs))
        for k_p, (i, j) in enumerate(grp_pairs):
            ei, ej = embs[pos[i]], embs[pos[j]]
            d_l2.append(-float(torch.norm(ei - ej).item()))
            d_cos.append(float(torch.nn.functional.cosine_similarity(
                ei.unsqueeze(0), ej.unsqueeze(0)).item()))
            d_jac.append(jaccard(kcache[i], kcache[j]))
            if k_p < n_edit_sub:
                e = edit_sim(dl.reads[i], dl.reads[j])
                if e is not None: d_ed.append(e)
        results[grp] = {
            'L2': np.array(d_l2), 'cosine': np.array(d_cos),
            'jaccard': np.array(d_jac),
            'edit': np.array(d_ed) if d_ed else None,
        }
        ed_str = f"{np.mean(d_ed):.3f}" if d_ed else "N/A"
        print(f"   {grp}: L2={np.mean(d_l2):.3f}  cos={np.mean(d_cos):.3f}  "
              f"jac={np.mean(d_jac):.3f}  edit={ed_str}")

    print("\n" + "=" * 70)
    print(f"🎯 保距性诊断 R{args.iter} (cross-same-GT vs cross-diff-GT)")
    print("=" * 70)
    print(f"\n  {'Distance':<12}{'Δ_median':>14}{'AUROC':>10}{'Clean sep?':>14}")
    print("  " + "-" * 50)

    scores = {}
    for d in ['L2', 'cosine', 'jaccard', 'edit']:
        same = results['cross_same_gt'][d]
        diff = results['cross_diff_gt'][d]
        if same is None or diff is None: continue
        dmed = np.median(same) - np.median(diff)
        au = auroc(same, diff)
        clean = np.quantile(same, 0.25) > np.quantile(diff, 0.75)
        print(f"  {d:<12}{dmed:>+14.4f}{au:>10.4f}{'✓' if clean else '✗':>14}")
        scores[d] = (dmed, au, clean)

    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = {'same_cluster': 'steelblue', 'cross_same_gt': 'darkorange',
              'cross_diff_gt': 'crimson'}
    for ax, d in zip(axes, ['L2', 'cosine', 'jaccard', 'edit']):
        for grp, vals in results.items():
            v = vals[d]
            if v is None or len(v) == 0: continue
            ax.hist(v, bins=50, alpha=0.4, density=True, label=grp, color=colors[grp])
        ax.set_xlabel(d); ax.set_ylabel('density')
        if d in scores:
            dm, au, _ = scores[d]
            ax.set_title(f"{d}  Δ_med={dm:+.3f}  AUROC={au:.3f}")
        else:
            ax.set_title(f"{d}  (N/A)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.suptitle(f"G spike R{args.iter}: distance comparison "
                 f"(n_pairs={args.n_pairs}/group)", fontsize=12)
    plt.tight_layout()
    fp = os.path.join(out, f'g_spike_R{args.iter}.png')
    plt.savefig(fp, dpi=120); plt.close()
    print(f"\n📊 {fp}")

    emb_au = [scores[d][1] for d in ['L2', 'cosine'] if d in scores]
    seq_au = [scores[d][1] for d in ['jaccard', 'edit'] if d in scores]
    eb = max(emb_au) if emb_au else 0
    sb = max(seq_au) if seq_au else 0
    print("\n" + "=" * 70)
    print(f"  R{args.iter} Embedding best AUROC: {eb:.4f}")
    print(f"  R{args.iter} Sequence  best AUROC: {sb:.4f}")
    print("=" * 70)

    sp = os.path.join(out, f'g_spike_R{args.iter}_summary.txt')
    with open(sp, 'w') as f:
        f.write(f"G spike R{args.iter} summary\n{'='*60}\n\n")
        f.write(f"ckpt: {sel_ckpt}\n")
        f.write(f"n_pairs/group: {args.n_pairs}, k={args.k}\n\n")
        for d, (dm, au, cl) in scores.items():
            f.write(f"{d}:  Δ_med={dm:+.4f}  AUROC={au:.4f}  clean_sep={cl}\n")
        f.write(f"\nR{args.iter} embedding best AUROC: {eb:.4f}\n")
        f.write(f"R{args.iter} sequence  best AUROC: {sb:.4f}\n")
    print(f"💾 {sp}")


if __name__ == '__main__':
    main()