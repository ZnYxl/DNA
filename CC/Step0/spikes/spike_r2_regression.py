#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spike_r2_regression.py
======================
排查 R2 回退根因 + 评估对比学习(encoder)健康度。

背景:
  R1 拆分后 SR=0.9539(巅峰), R2 暴跌到 0.9164, R3 没回来(0.9166)。
  簇数稳定(15663→15742→15988, 没过拆)。
  SR 掉444个success但covered只掉94 → 主要是"covered里success→fail", consensus质量被破坏。

两个嫌疑:
  A. encoder重训退化: R1拆出的干净簇consensus当靶→重训→encoder变差→R2基于坏embedding搞乱簇
  B. 训练靶污染: R1拆出的4015新簇里有"拆过头"的残缺半簇→consensus残缺→当靶教坏encoder

本spike两部分交叉定位:

Part A — R2崩的是哪批簇:
  找"R1 success → R2 fail"的GT, 看它们R1所在簇是 新拆簇(ID>=11648) 还是 老簇(ID<11648)。
    主要是新拆簇 → 指向B(拆过头的簇崩)
    老簇也大量崩 → 指向A(encoder退化波及全局)

Part B — 对比学习健康度(R1/R2/R3 encoder的embedding区分力):
  每轮encoder测 同GT对 vs 异GT对 的 cosine AUROC。
    三轮稳在~0.99 → encoder没退化, R2崩不是encoder的锅(指向B)
    R2/R3下降      → encoder重训退化了(指向A), 对比学习在反向

只读。Part B 需GPU(加载三个encoder跑推理)。
"""
import argparse, os, sys
import numpy as np
from collections import defaultdict, Counter


def _add_code_root():
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(8):
        if os.path.exists(os.path.join(d, 'models', 'eval_reconstruction.py')):
            if d not in sys.path: sys.path.insert(0, d)
            m = os.path.join(d, 'models')
            if m not in sys.path: sys.path.insert(0, m)
            return d
        p = os.path.dirname(d)
        if p == d: break
        d = p
    if here not in sys.path: sys.path.insert(0, here)
    return None

_root = _add_code_root()
if _root: print(f"[path] code 根: {_root}")

try:
    from eval_reconstruction import (
        levenshtein, load_reads_from_readtxt, load_gt_tags_file,
        load_gt_refs_fasta, build_tag_to_ref_mapping, match_reads_to_gt, find_read_txt,
    )
    from models.step1_model import Step1EvidentialModel
    from models.step1_data import seq_to_onehot
except ImportError as e:
    print(f"❌ import 失败: {e}"); sys.exit(1)

import torch
import torch.nn.functional as F


def mv_consensus(read_seqs, ref_length):
    N = len(read_seqs)
    if N == 0: return ""
    thresh = max(N*0.5, 1)
    out = []
    for pos in range(ref_length):
        cnt = Counter(); valid = 0
        for s in read_seqs:
            if pos < len(s):
                valid += 1
                if s[pos] in 'ACGT': cnt[s[pos]] += 1
        if valid >= thresh and cnt:
            out.append(cnt.most_common(1)[0][0])
    return ''.join(out)


def success_map(labels, reads, gt_ref_ids, gt_refs, ref_length):
    """
    返回 {gt_id: (is_success, cluster_id_that_succeeded_or_majclu)}。
    每个GT在以它为主GT的簇里, 任一簇MV==ref(ED==0) 即success;
    记录该GT对应的(优先success的)簇id, 用于追溯它是新簇还是老簇。
    """
    cl_to_didx = defaultdict(list)
    for i, c in enumerate(labels):
        if c >= 0: cl_to_didx[int(c)].append(i)
    cl_majgt = {}
    for c, ridxs in cl_to_didx.items():
        cnt = Counter(int(gt_ref_ids[ri]) for ri in ridxs if gt_ref_ids[ri] >= 0)
        if cnt: cl_majgt[c] = cnt.most_common(1)[0][0]
    majgt_to_clusters = defaultdict(list)
    for c, mg in cl_majgt.items():
        majgt_to_clusters[mg].append(c)

    out = {}
    all_gt = set(int(g) for g in gt_ref_ids[gt_ref_ids >= 0].tolist())
    for g in all_gt:
        cs = majgt_to_clusters.get(g, [])
        succ = False; succ_c = cs[0] if cs else -1
        for c in cs:
            cons = mv_consensus([reads[ri] for ri in cl_to_didx[c]], ref_length)
            if cons and levenshtein(cons, gt_refs[g]) == 0:
                succ = True; succ_c = c; break
        out[g] = (succ, succ_c)
    return out


@torch.no_grad()
def embed_auroc(model_ckpt, reads, gt_ref_ids, device, max_len, n_pairs=20000, seed=0):
    """加载一个encoder, 抽同GT对/异GT对, 算cosine AUROC(区分不同GT的能力)。"""
    ckpt = torch.load(model_ckpt, map_location=device)
    cargs = ckpt.get('args', {})
    mdim = cargs.get('dim', 256); mmax = cargs.get('max_length', max_len)
    model = Step1EvidentialModel(dim=mdim, max_length=mmax, num_clusters=50,
                                 device=str(device)).to(device)
    import torch.nn as nn
    sd = ckpt['model_state_dict']
    if 'length_adapter.weight' in sd:
        sh = sd['length_adapter.weight'].shape
        if sh[1] == mmax and sh[0] == mmax:
            model.length_adapter = nn.Linear(sh[1], sh[0]).to(device)
    model.load_state_dict(sd, strict=False); model.eval()

    rng = np.random.default_rng(seed)
    # 按GT分组(只取有GT的read)
    gt_to_idx = defaultdict(list)
    for i, g in enumerate(gt_ref_ids):
        if g >= 0: gt_to_idx[int(g)].append(i)
    gts_multi = [g for g, v in gt_to_idx.items() if len(v) >= 2]

    # 采样 read 索引集合(去重, 一次性编码)
    pos_pairs = []; neg_pairs = []
    for _ in range(n_pairs):
        g = gts_multi[rng.integers(len(gts_multi))]
        a, b = rng.choice(gt_to_idx[g], 2, replace=False)
        pos_pairs.append((a, b))
    all_gts = list(gt_to_idx.keys())
    for _ in range(n_pairs):
        g1, g2 = rng.choice(all_gts, 2, replace=False)
        a = gt_to_idx[g1][rng.integers(len(gt_to_idx[g1]))]
        b = gt_to_idx[g2][rng.integers(len(gt_to_idx[g2]))]
        neg_pairs.append((a, b))

    need = sorted(set([i for p in pos_pairs+neg_pairs for i in p]))
    pos_in = {ri: k for k, ri in enumerate(need)}
    embs = np.zeros((len(need), mdim), dtype=np.float32)
    B = 512
    for s in range(0, len(need), B):
        chunk = need[s:s+B]
        encs = torch.stack([seq_to_onehot(reads[ri], mmax) for ri in chunk])
        if encs.shape[1] != mmax:
            encs = F.pad(encs, (0,0,0,mmax-encs.shape[1])) if encs.shape[1]<mmax else encs[:,:mmax,:]
        _, pooled = model.encode_reads(encs.to(device))
        embs[s:s+len(chunk)] = pooled.cpu().numpy()
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    def cos(p):
        a = embs[pos_in[p[0]]]; b = embs[pos_in[p[1]]]
        return float((a*b).sum())
    pos_s = np.array([cos(p) for p in pos_pairs])
    neg_s = np.array([cos(p) for p in neg_pairs])
    # AUROC = P(pos_sim > neg_sim)
    from numpy import concatenate
    scores = concatenate([pos_s, neg_s])
    labels = concatenate([np.ones(len(pos_s)), np.zeros(len(neg_s))])
    order = np.argsort(-scores)
    labels = labels[order]
    tps = np.cumsum(labels); fps = np.cumsum(1-labels)
    tpr = tps/tps[-1]; fpr = fps/fps[-1]
    auroc = float(np.trapz(tpr, fpr))
    del model; torch.cuda.empty_cache()
    return auroc, float(pos_s.mean()), float(neg_s.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment_dir', required=True)
    ap.add_argument('--gt_refs', required=True)
    ap.add_argument('--gt_tags', required=True)
    ap.add_argument('--read_txt', default=None)
    ap.add_argument('--labels_dir', required=True,
                    help='04_Iterative_Labels 目录')
    ap.add_argument('--r1_labels', required=True)  # refined_labels_131658.txt
    ap.add_argument('--r2_labels', required=True)  # refined_labels_135922.txt
    ap.add_argument('--r1_model', required=True)
    ap.add_argument('--r2_model', required=True)
    ap.add_argument('--r3_model', default=None)
    ap.add_argument('--ref_length', type=int, default=196)
    ap.add_argument('--max_length', type=int, default=201)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--skip_partB', action='store_true', help='只跑Part A(不加载encoder)')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("="*72)
    print("  Spike: R2 回退根因 + 对比学习健康度")
    print("="*72)

    # GT 链路
    read_txt = args.read_txt or find_read_txt(args.experiment_dir)
    print(f"\n[1] reads: {read_txt}")
    reads, _ = load_reads_from_readtxt(read_txt)
    print(f"\n[2] GT tags"); seq_to_tag, tag_to_reads = load_gt_tags_file(args.gt_tags)
    print(f"\n[3] GT refs"); gt_refs = load_gt_refs_fasta(args.gt_refs)
    ref_len_med = int(np.median([len(s) for s in gt_refs.values()]))
    tag_to_ref = build_tag_to_ref_mapping(tag_to_reads, gt_refs, ref_len=ref_len_med)
    print(f"\n[4] reads->GT"); gt_ref_ids = match_reads_to_gt(reads, seq_to_tag, tag_to_ref)

    # ── Part A ──
    print(f"\n{'='*72}\n  Part A — R2 崩的是哪批簇\n{'='*72}")
    r1_labels = np.loadtxt(args.r1_labels, dtype=int)
    r2_labels = np.loadtxt(args.r2_labels, dtype=int)
    print(f"   R1 labels: {len(r1_labels)}, 簇 {len(set(r1_labels[r1_labels>=0]))}")
    print(f"   R2 labels: {len(r2_labels)}, 簇 {len(set(r2_labels[r2_labels>=0]))}")

    print(f"   计算 R1 success map ...")
    r1_succ = success_map(r1_labels, reads, gt_ref_ids, gt_refs, args.ref_length)
    print(f"   计算 R2 success map ...")
    r2_succ = success_map(r2_labels, reads, gt_ref_ids, gt_refs, args.ref_length)

    # R1 success -> R2 fail 的 GT
    regressed = [g for g in r1_succ
                 if r1_succ[g][0] and (g in r2_succ) and not r2_succ[g][0]]
    print(f"\n   R1 success 数: {sum(1 for g in r1_succ if r1_succ[g][0])}")
    print(f"   R2 success 数: {sum(1 for g in r2_succ if r2_succ[g][0])}")
    print(f"   回退 GT (R1✓→R2✗): {len(regressed)}")

    # 这些回退GT, 在R1所在簇是新拆簇(>=11648)还是老簇(<11648)?
    SPLIT_ID_THRESHOLD = 11648  # R1拆分新ID从11648起(Clover原簇0..11647)
    new_clu = sum(1 for g in regressed if r1_succ[g][1] >= SPLIT_ID_THRESHOLD)
    old_clu = len(regressed) - new_clu
    print(f"\n   回退GT在R1的簇归属:")
    print(f"     新拆簇(ID>={SPLIT_ID_THRESHOLD}): {new_clu}  ({new_clu/max(len(regressed),1)*100:.1f}%)")
    print(f"     老簇  (ID< {SPLIT_ID_THRESHOLD}): {old_clu}  ({old_clu/max(len(regressed),1)*100:.1f}%)")
    print(f"\n   判定:")
    if len(regressed) == 0:
        print(f"     (无回退, 异常, 检查labels对应轮次)")
    elif new_clu > old_clu * 1.5:
        print(f"     ➜ 主要是新拆簇崩 → 指向 B(拆过头的残缺簇污染R2训练靶)")
    elif old_clu > new_clu * 1.5:
        print(f"     ➜ 主要是老簇崩 → 指向 A(encoder重训退化, 波及未拆的老簇)")
    else:
        print(f"     ➜ 新老簇都崩 → A/B 共同作用, 需Part B进一步分清")

    # ── Part B ──
    if not args.skip_partB:
        print(f"\n{'='*72}\n  Part B — 对比学习健康度(encoder embedding AUROC)\n{'='*72}")
        models = [('R1', args.r1_model), ('R2', args.r2_model)]
        if args.r3_model: models.append(('R3', args.r3_model))
        print(f"   {'轮次':>4} {'AUROC':>8} {'同GT cos':>10} {'异GT cos':>10}")
        aurocs = {}
        for name, ck in models:
            au, pmean, nmean = embed_auroc(ck, reads, gt_ref_ids, device, args.max_length)
            aurocs[name] = au
            print(f"   {name:>4} {au:>8.4f} {pmean:>10.4f} {nmean:>10.4f}")
        print(f"\n   判定:")
        if 'R2' in aurocs and aurocs['R2'] < aurocs.get('R1', 1) - 0.01:
            print(f"     ➜ R2 encoder AUROC 下降 → encoder重训退化(指向A), 对比学习在反向")
        else:
            print(f"     ➜ encoder AUROC 三轮稳定 → encoder没退化(指向B), R2崩不是encoder的锅")

    print(f"\n{'='*72}")
    print(f"  综合结论: Part A 看簇归属, Part B 看encoder健康, 交叉定位A/B。")
    print(f"  A(encoder退化) → 解法: 冻结encoder")
    print(f"  B(训练靶污染) → 解法: 拆分后过滤残缺/小簇再当训练靶")
    print("="*72)


if __name__ == "__main__":
    main()