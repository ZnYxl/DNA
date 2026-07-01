#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# spike_uncertainty_weighted_consensus.py
# 诊断: 偶然不确定性 u_ale 加权 consensus 能否在 R3 (SR=0.9699) 基础上涨 SR?
# -----------------------------------------------------------------------------
# 单变量: 同一套 R3 refined_labels, 同 ref_length 截断, 同 SR 评估口径,
#         唯一区别 = 逐位投票时每条 read 的票权:
#           等权 (baseline):  weight = 1
#           u_ale 加权 (exp):  weight = exp(-alpha * u_ale)
#                              (u_ale 高=噪声大=票轻; alpha 控制衰减强度)
#         扫多个 alpha, 看加权能否提升 SR。
#
# 只读: 不改任何生产代码, 不落盘标签。consensus 与 SR 评估均在本脚本内自洽实现,
#       与 compute_mv_consensus / eval_reconstruction 同口径 (50% has_vote 门限 +
#       ref_length 截断 + success = consensus==reference)。
#
# 用法:
#   python spike_uncertainty_weighted_consensus.py
# =============================================================================
import os
import sys
import numpy as np
import torch
from collections import defaultdict, Counter

# ── 配置 ──
EXP_DIR    = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d"
LABELS     = os.path.join(EXP_DIR, "04_Iterative_Labels", "refined_labels_214640.txt")  # R3
READ_STATE = os.path.join(EXP_DIR, "04_Iterative_Labels", "read_state_214640.pt")        # R3
GT_REFS    = "/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/reads.fasta"
GT_TAGS    = os.path.join(EXP_DIR, "seq1d_tags_reads.txt")
REF_LENGTH = 196
ALPHAS     = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]   # 0.0 == 等权基线
N_GT_DENOM = 11826                             # 与 FedDNA / eval_reconstruction 同口径

CODE_ROOT = "/mnt/st_data/liangxinyi/code"
for p in (CODE_ROOT, os.path.join(CODE_ROOT, "models")):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.step1_data import CloverDataLoader
from models.eval_reconstruction import levenshtein

BASES = 'ACGT'
B2I = {b: i for i, b in enumerate(BASES)}


def load_gt_refs(path):
    """reads.fasta -> {gt_id: ref_seq}。GT id 取 header 里的数字。"""
    refs = {}
    cur_id = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur_id is not None:
                    refs[cur_id] = ''.join(seq_parts).upper()
                # header 提取 id (取第一段数字)
                import re
                m = re.search(r'(\d+)', line)
                cur_id = int(m.group(1)) if m else line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if cur_id is not None:
        refs[cur_id] = ''.join(seq_parts).upper()
    return refs


def load_gt_tags(gt_path, reads):
    seq_to_gt = {}
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                seq_to_gt[parts[1].upper()] = int(parts[0])
    gt = np.full(len(reads), -1, dtype=np.int64)
    for i, r in enumerate(reads):
        g = seq_to_gt.get(r.upper())
        if g is not None:
            gt[i] = g
    return gt


def weighted_consensus(seqs, weights, ref_length):
    """
    加权逐位多数投票。weights 与 seqs 等长, 每条 read 一个标量票权。
    has_vote 门限: 该位有效(权重和 >= 50% 总权重) 才输出碱基。
    与等权口径一致(weights 全 1 时退化为普通 MV)。
    """
    n = len(seqs)
    if n == 0:
        return ""
    total_w = float(sum(weights))
    thresh = max(total_w * 0.5, 1e-9)
    out = []
    for pos in range(ref_length):
        acc = np.zeros(4, dtype=np.float64)
        valid_w = 0.0
        for s, w in zip(seqs, weights):
            if pos < len(s):
                b = s[pos]
                if b in B2I:
                    acc[B2I[b]] += w
                    valid_w += w
        if valid_w >= thresh and acc.sum() > 0:
            out.append(BASES[int(acc.argmax())])
    return ''.join(out)


def evaluate_sr(consensus_by_cluster, cluster_to_gt, gt_refs, n_gt_denom):
    """
    每个 GT ref 只评一次(多 cluster 映同一 ref 取 ED 最小)。
    SR = #{完全匹配} / n_gt_denom (与 FedDNA 同口径)。
    """
    gt_to_best_ed = {}
    for cid, cons in consensus_by_cluster.items():
        gid = cluster_to_gt.get(cid)
        if gid is None or gid not in gt_refs:
            continue
        ref = gt_refs[gid]
        ed = levenshtein(cons, ref)
        if gid not in gt_to_best_ed or ed < gt_to_best_ed[gid]:
            gt_to_best_ed[gid] = ed
    success = sum(1 for ed in gt_to_best_ed.values() if ed == 0)
    covered = len(gt_to_best_ed)
    eer_vals = [ed / REF_LENGTH for ed in gt_to_best_ed.values()]
    return {
        'SR':     success / max(n_gt_denom, 1),
        'success': success,
        'recall': covered / max(n_gt_denom, 1),
        'eer':    float(np.mean(eer_vals)) if eer_vals else 0.0,
    }


def main():
    print("=" * 72)
    print("  🔬 不确定性加权 consensus 诊断 (R3, baseline SR=0.9699)")
    print("=" * 72)

    dl = CloverDataLoader(EXP_DIR)
    reads = dl.reads
    print(f"   Reads: {len(reads)}")

    labels = np.loadtxt(LABELS, dtype=np.int64)
    print(f"   标签: {LABELS.split('/')[-1]}, 簇数={len(set(labels[labels>=0]))}")

    st = torch.load(READ_STATE, map_location='cpu')
    u_ale = np.asarray(st['u_ale'], dtype=np.float64)   # (TOTAL_READS,) 按 real_idx
    print(f"   u_ale: min={u_ale.min():.3f} max={u_ale.max():.3f} mean={u_ale.mean():.3f}")

    gt_refs = load_gt_refs(GT_REFS)
    gt = load_gt_tags(GT_TAGS, reads)
    print(f"   GT refs: {len(gt_refs)}, GT 匹配: {(gt>=0).sum()}/{len(reads)}")

    # ── 按 cluster 分组 (real_idx 空间) ──
    cl_to_ridx = defaultdict(list)
    for ridx, lab in enumerate(labels):
        if lab >= 0:
            cl_to_ridx[int(lab)].append(ridx)

    # 每个 cluster 的 GT 映射 (majority vote)
    cluster_to_gt = {}
    for cid, ridxs in cl_to_ridx.items():
        gts = [gt[r] for r in ridxs if gt[r] >= 0]
        if gts:
            cluster_to_gt[cid] = Counter(gts).most_common(1)[0][0]

    print(f"\n   {'alpha':>6}  {'SR':>8}  {'success':>8}  {'recall':>8}  {'EER':>9}   说明")
    print(f"   {'-'*62}")

    baseline_sr = None
    for alpha in ALPHAS:
        consensus_by_cluster = {}
        for cid, ridxs in cl_to_ridx.items():
            seqs = [reads[r] for r in ridxs]
            if alpha == 0.0:
                weights = [1.0] * len(ridxs)            # 等权基线
            else:
                weights = [float(np.exp(-alpha * u_ale[r])) for r in ridxs]
            cons = weighted_consensus(seqs, weights, REF_LENGTH)
            if cons:
                consensus_by_cluster[cid] = cons

        res = evaluate_sr(consensus_by_cluster, cluster_to_gt, gt_refs, N_GT_DENOM)
        tag = "等权基线" if alpha == 0.0 else f"exp(-{alpha}·u_ale)"
        if alpha == 0.0:
            baseline_sr = res['SR']
            delta = ""
        else:
            d = res['SR'] - baseline_sr
            delta = f"  Δ={d:+.4f}" + ("  ✅涨" if d > 0 else ("  ❌跌" if d < 0 else "  =持平"))
        print(f"   {alpha:>6.1f}  {res['SR']:>8.4f}  {res['success']:>8d}  "
              f"{res['recall']:>8.4f}  {res['eer']:>9.6f}   {tag}{delta}")

    print(f"\n   ── 判读 ──")
    print(f"   alpha=0 的 SR 应 ≈ 0.9699 (验证口径对). 若某 alpha 的 SR 明显 > 基线,")
    print(f"   则 u_ale 加权 consensus 有真实增益, evidential 可在此位置耦合进 step2。")
    print(f"   若所有 alpha 都 ≤ 基线, 说明 MV 等权已足够, 换方向 B/C/D。")
    print("=" * 72)


if __name__ == "__main__":
    main()