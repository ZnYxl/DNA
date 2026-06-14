#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spike_split_separability.py
===========================
诊断目标: 上一个 spike 证明 "拆分混合簇" 上界 +581 success(+4.96pt)。
         但那是上帝视角(用GT把次要GT的read精确抠出来)。
         真做拆分时无GT, 只能靠簇内信号。
         本 spike 问: 这 581 个可救次要GT, 用无监督信号能挑出多少?

只读。加载 R1 模型跑 encoder(需GPU)。不改任何文件。

对每个 "可救次要GT"(在某混合簇里被主GT压制、但单独MV能success、现状未success),
刻画三个无监督可分信号:

  信号1 size占比 = 次要GT read数 / 主GT read数
     大(>~0.2) -> 不是一条尾巴, size 能定位; 小 -> 淹没在噪声里认不出
  信号2 seq_dist = 次要GT consensus 与 主GT consensus 的 levenshtein
     大(>~10) -> 簇内有两个明显序列团, 可序列聚类劈开; 小 -> 近似重复, 劈不开
  信号3 emb_sep = 簇内对 read embedding 做 2-means,
     次要GT的read 是否干净落入与主GT不同的那一团 (homogeneity)
     高 -> embedding 能在簇内二分时把它分出来(你encoder AUROC0.999, 最可能发力)

输出: 把 581 按 "三信号下是否可分" 分桶,
     给出 "无监督拆分实际可达上界"(信号判定可分 ∩ GT判定可救)。
     这个数 = v21 拆分机制的真实天花板, 不是 581。
"""
import argparse
import os
import sys
import numpy as np
from collections import defaultdict, Counter


# ── 自动定位 code 根, 复用 eval_reconstruction 口径 ──────────────────────────
def _add_code_root():
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(8):
        cand = os.path.join(d, 'models', 'eval_reconstruction.py')
        if os.path.exists(cand):
            if d not in sys.path:
                sys.path.insert(0, d)
            mdir = os.path.join(d, 'models')
            if mdir not in sys.path:
                sys.path.insert(0, mdir)
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    if here not in sys.path:
        sys.path.insert(0, here)
    return None

_code_root = _add_code_root()
if _code_root:
    print(f"[path] code 根: {_code_root}")

try:
    from eval_reconstruction import (
        levenshtein,
        load_reads_from_readtxt,
        load_gt_tags_file,
        load_gt_refs_fasta,
        build_tag_to_ref_mapping,
        match_reads_to_gt,
        find_read_txt,
    )
    from models.step1_model import Step1EvidentialModel
    from models.step1_data import seq_to_onehot
except ImportError as e:
    print("❌ import 失败。确认 spike 在 code 树下, models/ 里有 eval_reconstruction/step1_model/step1_data。")
    print(f"   detail: {e}")
    sys.exit(1)

import torch
import torch.nn.functional as F


# ── MV (与上个spike/compute_mv_consensus同口径) ────────────────────────────────
def mv_consensus(read_seqs, ref_length):
    N = len(read_seqs)
    if N == 0:
        return ""
    thresh = max(N * 0.5, 1)
    out = []
    for pos in range(ref_length):
        counter = Counter(); valid = 0
        for s in read_seqs:
            if pos < len(s):
                valid += 1
                b = s[pos]
                if b in 'ACGT':
                    counter[b] += 1
        if valid >= thresh and counter:
            out.append(counter.most_common(1)[0][0])
    return ''.join(out)


def kmeans2(X, iters=20, seed=0):
    """极简 2-means(欧式), 返回每点的 0/1 簇标签。X: (N,D) np.float32。"""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N < 2:
        return np.zeros(N, dtype=int)
    # 初始化: 取距离最远的两点附近
    i0 = rng.integers(N)
    d0 = ((X - X[i0]) ** 2).sum(1)
    i1 = int(d0.argmax())
    cent = np.stack([X[i0], X[i1]])
    labels = np.zeros(N, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(2)  # (N,2)
        new = d.argmin(1)
        if (new == labels).all():
            labels = new; break
        labels = new
        for k in range(2):
            if (labels == k).any():
                cent[k] = X[labels == k].mean(0)
    return labels


@torch.no_grad()
def embed_reads(model, seqs, max_len, device, batch=512):
    """返回 (N, D) pooled embedding。"""
    embs = []
    for start in range(0, len(seqs), batch):
        chunk = seqs[start:start+batch]
        encs = torch.stack([seq_to_onehot(s, max_len) for s in chunk])  # (b,L,4)
        if encs.shape[1] != max_len:
            if encs.shape[1] < max_len:
                encs = F.pad(encs, (0, 0, 0, max_len - encs.shape[1]))
            else:
                encs = encs[:, :max_len, :]
        encs = encs.to(device)
        _, pooled = model.encode_reads(encs)
        embs.append(pooled.cpu().numpy())
    return np.concatenate(embs, 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="可救次要GT的无监督可分性刻画 spike")
    ap.add_argument('--experiment_dir', required=True)
    ap.add_argument('--gt_refs', required=True)
    ap.add_argument('--gt_tags', required=True)
    ap.add_argument('--read_txt', default=None)
    ap.add_argument('--model_ckpt', required=True,
                    help='R1 模型 step1_final_model.pth (用于 embedding 信号)')
    ap.add_argument('--ref_length', type=int, default=196)
    ap.add_argument('--max_length', type=int, default=201,
                    help='模型输入长度(默认201, 会从ckpt args覆盖)')
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--device', default='cuda')
    # 信号判"可分"的阈值(只用于分桶统计, 可调)
    ap.add_argument('--size_ratio_th', type=float, default=0.20)
    ap.add_argument('--seq_dist_th', type=int, default=10)
    ap.add_argument('--emb_homo_th', type=float, default=0.80)
    args = ap.parse_args()

    print("=" * 72)
    print("  可救次要GT 的无监督可分性刻画 spike")
    print("=" * 72)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── GT 链路 ───────────────────────────────────────────────────────────────
    read_txt = args.read_txt or find_read_txt(args.experiment_dir)
    print(f"\n[1] reads: {read_txt}")
    reads, clover_labels = load_reads_from_readtxt(read_txt)
    print(f"\n[2] GT tags: {args.gt_tags}")
    seq_to_tag, tag_to_reads = load_gt_tags_file(args.gt_tags)
    print(f"\n[3] GT refs: {args.gt_refs}")
    gt_refs = load_gt_refs_fasta(args.gt_refs)
    ref_len_med = int(np.median([len(s) for s in gt_refs.values()]))
    tag_to_ref = build_tag_to_ref_mapping(tag_to_reads, gt_refs, ref_len=ref_len_med)
    print(f"\n[4] reads -> GT ref id")
    gt_ref_ids = match_reads_to_gt(reads, seq_to_tag, tag_to_ref)

    # ── 簇结构 ────────────────────────────────────────────────────────────────
    cl_to_ridx = defaultdict(list)
    for i, c in enumerate(clover_labels):
        cl_to_ridx[int(c)].append(i)
    cl_gt_counts = {}; cl_majgt = {}
    for c, ridxs in cl_to_ridx.items():
        cnt = Counter()
        for ri in ridxs:
            g = int(gt_ref_ids[ri])
            if g >= 0:
                cnt[g] += 1
        cl_gt_counts[c] = cnt
        if cnt:
            cl_majgt[c] = cnt.most_common(1)[0][0]
    all_gt = set(int(g) for g in gt_ref_ids[gt_ref_ids >= 0].tolist())
    n_gt = len(all_gt)

    def cluster_seqs(c):
        return [reads[ri] for ri in cl_to_ridx[c]]

    # 现状 success(同上个spike): 每GT在以它为主GT的簇里, 任一簇MV==ref 即success
    print(f"\n[5] 计算现状 success ...")
    majgt_to_clusters = defaultdict(list)
    for c, mg in cl_majgt.items():
        majgt_to_clusters[mg].append(c)
    gt_success_now = set()
    for g in all_gt:
        for c in majgt_to_clusters.get(g, []):
            cons = mv_consensus(cluster_seqs(c), args.ref_length)
            if cons and levenshtein(cons, gt_refs[g]) == 0:
                gt_success_now.add(g); break
    print(f"    现状 success: {len(gt_success_now):,}/{n_gt:,} (SR={len(gt_success_now)/n_gt:.4f})")

    # ── 找出所有 "可救次要GT" 实例 (簇,次要GT) ─────────────────────────────────
    # 与上个spike一致: 混合簇里非主GT, 单独MV success, 且该GT现状未success。
    # 这里保留 (cluster, 次要GT, 它的局部read idx) 以便算三信号。
    print(f"\n[6] 定位可救次要GT实例 ...")
    rescuable = []   # list of dict: {cluster, sec_gt, sec_ridx, maj_gt, maj_ridx}
    seen_gt = set()
    for c, cnt in cl_gt_counts.items():
        if len(cnt) < 2:
            continue
        mg = cl_majgt[c]
        gt_to_local = defaultdict(list)
        for ri in cl_to_ridx[c]:
            g = int(gt_ref_ids[ri])
            if g >= 0:
                gt_to_local[g].append(ri)
        maj_ridx = gt_to_local[mg]
        for g, ris in gt_to_local.items():
            if g == mg:
                continue
            if g in gt_success_now or g in seen_gt:
                continue
            cons = mv_consensus([reads[ri] for ri in ris], args.ref_length)
            if cons and levenshtein(cons, gt_refs[g]) == 0:
                rescuable.append({
                    'cluster': c, 'sec_gt': g, 'sec_ridx': ris,
                    'maj_gt': mg, 'maj_ridx': maj_ridx,
                })
                seen_gt.add(g)   # 同上个spike: 一个GT只算一次
    print(f"    可救次要GT实例: {len(rescuable)} (应≈581)")

    if not rescuable:
        print("    无可救实例, 退出。"); return

    # ── 加载模型(信号3) ───────────────────────────────────────────────────────
    print(f"\n[7] 加载 R1 模型: {args.model_ckpt}")
    ckpt = torch.load(args.model_ckpt, map_location=device)
    cargs = ckpt.get('args', {})
    mdim = cargs.get('dim', args.dim)
    mmax = cargs.get('max_length', args.max_length)
    print(f"    dim={mdim}, max_length={mmax}")
    model = Step1EvidentialModel(dim=mdim, max_length=mmax,
                                 num_clusters=50, device=str(device)).to(device)
    import torch.nn as nn
    sd = ckpt['model_state_dict']
    if 'length_adapter.weight' in sd:
        sh = sd['length_adapter.weight'].shape
        if sh[1] == mmax and sh[0] == mmax:
            model.length_adapter = nn.Linear(sh[1], sh[0]).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"    ✅ 模型就绪")

    # ── 三信号刻画 ────────────────────────────────────────────────────────────
    print(f"\n[8] 刻画三信号 (含 embedding 簇内二分, 跑 encoder) ...")
    size_ratios = []; seq_dists = []; emb_homos = []
    sig1_ok = sig2_ok = sig3_ok = 0

    for k, item in enumerate(rescuable):
        c = item['cluster']; mg = item['maj_gt']
        sec_ridx = item['sec_ridx']; maj_ridx = item['maj_ridx']
        n_sec = len(sec_ridx); n_maj = len(maj_ridx)

        # 信号1: size 占比
        ratio = n_sec / max(n_maj, 1)
        size_ratios.append(ratio)
        if ratio >= args.size_ratio_th:
            sig1_ok += 1

        # 信号2: 次要GT consensus vs 主GT consensus 的 edit
        sec_cons = mv_consensus([reads[ri] for ri in sec_ridx], args.ref_length)
        maj_cons = mv_consensus([reads[ri] for ri in maj_ridx], args.ref_length)
        sdist = levenshtein(sec_cons, maj_cons) if (sec_cons and maj_cons) else 0
        seq_dists.append(sdist)
        if sdist >= args.seq_dist_th:
            sig2_ok += 1

        # 信号3: 簇内全部read embedding 做 2-means, 看次要GT的read是否聚成纯团
        all_ridx = cl_to_ridx[c]
        seqs = [reads[ri] for ri in all_ridx]
        emb = embed_reads(model, seqs, mmax, device)
        km = kmeans2(emb)
        # 次要GT的read落在哪一团? 取多数团, 算 homogeneity = 该团里次要GT占比
        local_pos = {ri: idx for idx, ri in enumerate(all_ridx)}
        sec_km = [km[local_pos[ri]] for ri in sec_ridx]
        dom_team = Counter(sec_km).most_common(1)[0][0]
        # homo = 落在 dom_team 的次要read数 / dom_team 总read数
        team_size = int((km == dom_team).sum())
        sec_in_team = sum(1 for t in sec_km if t == dom_team)
        homo = sec_in_team / max(team_size, 1)
        # 同时要求次要GT自己别被劈散: 次要read进dom_team的比例
        sec_purity = sec_in_team / max(n_sec, 1)
        emb_homos.append((homo, sec_purity))
        if homo >= args.emb_homo_th and sec_purity >= 0.5:
            sig3_ok += 1

        if (k + 1) % 100 == 0:
            print(f"    ... {k+1}/{len(rescuable)}")

    size_ratios = np.array(size_ratios)
    seq_dists = np.array(seq_dists)
    homos = np.array([h for h, _ in emb_homos])
    secpur = np.array([p for _, p in emb_homos])

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    N = len(rescuable)
    print(f"\n{'='*72}")
    print(f"  581 可救次要GT 的无监督可分性")
    print(f"{'='*72}")
    print(f"  样本数: {N}")
    print(f"\n  信号1 size占比 (次要/主):")
    print(f"    median={np.median(size_ratios):.3f}  "
          f"P25={np.percentile(size_ratios,25):.3f}  P75={np.percentile(size_ratios,75):.3f}")
    print(f"    >= {args.size_ratio_th}: {sig1_ok}/{N} ({sig1_ok/N*100:.1f}%)  "
          f"[size能定位]")
    print(f"    次要GT read数: median={np.median([len(x['sec_ridx']) for x in rescuable]):.0f}")

    print(f"\n  信号2 次要vs主 consensus edit:")
    print(f"    median={np.median(seq_dists):.0f}  "
          f"P25={np.percentile(seq_dists,25):.0f}  P75={np.percentile(seq_dists,75):.0f}")
    print(f"    >= {args.seq_dist_th}: {sig2_ok}/{N} ({sig2_ok/N*100:.1f}%)  "
          f"[序列团可劈开]")
    near_dup = int((seq_dists <= 2).sum())
    print(f"    <= 2 (近似重复, 劈不开): {near_dup}/{N} ({near_dup/N*100:.1f}%)")

    print(f"\n  信号3 embedding簇内2-means:")
    print(f"    homogeneity median={np.median(homos):.3f}  "
          f"次要GT纯度 median={np.median(secpur):.3f}")
    print(f"    homo>={args.emb_homo_th} 且 纯度>=0.5: {sig3_ok}/{N} ({sig3_ok/N*100:.1f}%)  "
          f"[embedding可二分]")

    # 任一信号可分 / 全信号可分
    arr1 = size_ratios >= args.size_ratio_th
    arr2 = seq_dists >= args.seq_dist_th
    arr3 = (homos >= args.emb_homo_th) & (secpur >= 0.5)
    any_ok = int((arr1 | arr2 | arr3).sum())
    all3 = int((arr1 & arr2 & arr3).sum())
    emb_or_seq = int((arr2 | arr3).sum())
    print(f"\n  ── 综合可分性(无监督真实可达上界) ──")
    print(f"    任一信号可分      : {any_ok}/{N} ({any_ok/N*100:.1f}%)  -> SR 至多 +{any_ok/n_gt*100:.2f}pt")
    print(f"    序列或embedding   : {emb_or_seq}/{N} ({emb_or_seq/N*100:.1f}%)  "
          f"-> 最可能的可达上界")
    print(f"    三信号全可分(最稳): {all3}/{N} ({all3/N*100:.1f}%)")
    print(f"\n  现状SR {len(gt_success_now)/n_gt:.4f} -> 若拿到'序列或embedding'可分部分: "
          f"{(len(gt_success_now)+emb_or_seq)/n_gt:.4f}")
    print(f"  (上帝视角上界 0.9546 / FedDNA完美簇 0.9726)")
    print("=" * 72)


if __name__ == "__main__":
    main()