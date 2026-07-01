# spike_evidential_split.py
# -*- coding: utf-8 -*-
"""
Spike: evidential 不确定性融入 step2 拆分判据 (乘法版) —— 只读验证
====================================================================
目标:
  在 R1 标签上, 对比两套拆分判据下的 SR 与拆分行为:
    (A) 纯 edit (基线):   dAB >= tau            (tau=5)
    (B) 乘法 (evidential): dAB * min(S_A,S_B)/S_ref >= tau'   (tau' 扫 {3,4,5,6,7})
  S_ref = 全局平均 strength。

判据唯一差异: 最后那个门控。簇内 edit 二分 (_split_two) 完全不变 ->
单变量, 二分质量不受污染 (符合 Spike 1 "加权不破坏拆分质量")。

铁律遵守:
  - GT 对齐: 序列做 key (seq_to_tag), 不按 id/行号查 GT。
  - SR 评估: 直接 import evaluate_reconstruction, 不自己实现 SR。
  - 只读: 不写任何持久文件, 不改主代码。

数据对齐 (已 grep 确认):
  - refined_labels_204939.txt: 330788 行 = TOTAL_READS, 每行第 i 个是 real_idx i 的标签。
  - read_state_204939.pt: strength shape=(330788,), 同 real_idx 空间。
  - reads.fasta: 第 i 条 (按文件顺序) = real_idx i。
  => labels[i], strength[i], reads[i] 天然同索引, 无需 didx 中转。

用法:
  cd /mnt/st_data/liangxinyi/code
  python spike_evidential_split.py
"""
import os
import sys
import numpy as np
from collections import defaultdict, Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import torch

# ---------------------------------------------------------------------------
# 路径 (按 grep 确认的真实路径)
# ---------------------------------------------------------------------------
CODE_ROOT   = "/mnt/st_data/liangxinyi/code"
EXP_DIR     = os.path.join(CODE_ROOT, "CC/Step0/Experiments/seq_1d")
LABEL_DIR   = os.path.join(EXP_DIR, "04_Iterative_Labels")
LABELS_PATH = os.path.join(LABEL_DIR, "refined_labels_204939.txt")   # R1
STATE_PATH  = os.path.join(LABEL_DIR, "read_state_204939.pt")        # R1
READS_FASTA = os.path.join(CODE_ROOT, "CC/Step0/Sequencing_data_first_dimension/reads.fasta")
TAGS_PATH   = os.path.join(EXP_DIR, "seq1d_tags_reads.txt")
REFS_PATH   = os.path.join(EXP_DIR, "seq1d_refs.txt")

REF_LENGTH    = 196
TAU_BASE      = 5                  # 纯 edit 基线门控
TAU_PRIME_LST = [3, 4, 5, 6, 7]    # 乘法版扫的 tau'
MIN_SPLIT     = 6
MAX_PAIRWISE  = 80
LOW_S_FACTOR  = 0.6                # 判定 "低 strength 子簇": min(S) < LOW_S_FACTOR * S_ref

sys.path.insert(0, CODE_ROOT)

# ---------------------------------------------------------------------------
# 复用 eval_reconstruction 的 levenshtein + SR 口径 (不自己实现)
# ---------------------------------------------------------------------------
from models.eval_reconstruction import (
    levenshtein,
    load_reads_from_readtxt,      # 真 reads 来源 (330788, real_idx 空间, 已 thinning)
    load_gt_tags_file,
    build_tag_to_ref_mapping,
    match_reads_to_gt,
    load_gt_refs_fasta,
    build_cluster_to_gt,
    evaluate_reconstruction,
)

# read.txt: 330788 条真 reads, load_reads_from_readtxt 输出即 real_idx 空间
READ_TXT = os.path.join(EXP_DIR, "03_FedDNA_In", "read.txt")


# ---------------------------------------------------------------------------
# reads.fasta 读取 (按文件顺序 = real_idx)
# ---------------------------------------------------------------------------
def load_reads_fasta_ordered(path):
    reads = []
    cur = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur:
                    reads.append("".join(cur).upper())
                    cur = []
            elif line:
                cur.append(line)
    if cur:
        reads.append("".join(cur).upper())
    return reads


# ---------------------------------------------------------------------------
# consensus (与 cluster_split._mv_consensus 同口径)
# ---------------------------------------------------------------------------
def mv_consensus(read_seqs, ref_length):
    N = len(read_seqs)
    if N == 0:
        return ""
    thresh = max(N * 0.5, 1)
    out = []
    for pos in range(ref_length):
        cnt = Counter()
        valid = 0
        for s in read_seqs:
            if pos < len(s):
                valid += 1
                b = s[pos]
                if b in "ACGT":
                    cnt[b] += 1
        if valid >= thresh and cnt:
            out.append(cnt.most_common(1)[0][0])
    return "".join(out)


def split_two(seqs, max_pairwise=80, seed=0):
    """与 cluster_split._split_two 同逻辑 (edit 二分, 不变)。"""
    n = len(seqs)
    if n < 2:
        return list(range(n)), []
    if n <= max_pairwise:
        idxs = list(range(n))
        sub = seqs
    else:
        rng = np.random.default_rng(seed)
        idxs = sorted(rng.choice(n, max_pairwise, replace=False).tolist())
        sub = [seqs[i] for i in idxs]
    m = len(sub)
    D = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(i + 1, m):
            d = levenshtein(sub[i], sub[j])
            D[i, j] = D[j, i] = d
    Z = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Z, t=2, criterion="maxclust")
    a_local = [idxs[i] for i in range(m) if lab[i] == 1]
    b_local = [idxs[i] for i in range(m) if lab[i] == 2]
    if n > max_pairwise:
        ca = mv_consensus([seqs[i] for i in a_local],
                          max((len(seqs[i]) for i in a_local), default=0)) if a_local else ""
        cb = mv_consensus([seqs[i] for i in b_local],
                          max((len(seqs[i]) for i in b_local), default=0)) if b_local else ""
        assigned = set(idxs)
        for i in range(n):
            if i in assigned:
                continue
            da = levenshtein(seqs[i], ca) if ca else 1e9
            db = levenshtein(seqs[i], cb) if cb else 1e9
            (a_local if da <= db else b_local).append(i)
    return a_local, b_local


# ---------------------------------------------------------------------------
# 一趟拆分: 对每个候选簇算 (dAB, S_A, S_B), 缓存下来, 多个判据复用
# ---------------------------------------------------------------------------
def precompute_split_candidates(labels, strength, reads):
    """
    返回 candidates: list of dict, 每个候选簇含:
      cid, a_real(list real_idx), b_real(list real_idx), dAB, SA, SB
    只对 size>=MIN_SPLIT 的簇做一次 edit 二分 (最贵的步骤, 只跑一次)。
    """
    cl_to_real = defaultdict(list)
    for ridx, lab in enumerate(labels):
        if lab >= 0:
            cl_to_real[int(lab)].append(ridx)

    candidates = []
    n_examined = 0
    for cid, real_list in cl_to_real.items():
        if len(real_list) < MIN_SPLIT:
            continue
        n_examined += 1
        seqs = [reads[r] for r in real_list]
        a_loc, b_loc = split_two(seqs, max_pairwise=MAX_PAIRWISE)
        if len(a_loc) < 1 or len(b_loc) < 1:
            continue
        consA = mv_consensus([seqs[i] for i in a_loc], REF_LENGTH)
        consB = mv_consensus([seqs[i] for i in b_loc], REF_LENGTH)
        if not consA or not consB:
            continue
        dAB = levenshtein(consA, consB)
        a_real = [real_list[i] for i in a_loc]
        b_real = [real_list[i] for i in b_loc]
        SA = float(np.mean([strength[r] for r in a_real]))
        SB = float(np.mean([strength[r] for r in b_real]))
        candidates.append({
            "cid": cid, "a_real": a_real, "b_real": b_real,
            "dAB": dAB, "SA": SA, "SB": SB,
        })
    return candidates, n_examined, cl_to_real


# ---------------------------------------------------------------------------
# 给定判据, 把候选应用到 labels, 返回新 labels + 决策集合
# ---------------------------------------------------------------------------
def apply_criterion(labels, candidates, mode, tau, s_ref):
    """
    mode='edit':  split if dAB >= tau
    mode='mult':  split if dAB * min(SA,SB)/s_ref >= tau
    返回: new_labels, split_cids(set), n_split
    """
    new_labels = labels.copy()
    next_id = int(labels.max()) + 1 if (labels >= 0).any() else 0
    split_cids = set()
    for c in candidates:
        if mode == "edit":
            do_split = c["dAB"] >= tau
        else:
            g = min(c["SA"], c["SB"]) / s_ref
            do_split = (c["dAB"] * g) >= tau
        if do_split:
            new_labels[c["b_real"]] = next_id
            next_id += 1
            split_cids.add(c["cid"])
    n_split = len(split_cids)
    return new_labels, split_cids, n_split


# ---------------------------------------------------------------------------
# 用 evaluate_reconstruction 算 SR (复用口径)
# ---------------------------------------------------------------------------
def compute_sr(labels, reads, gt_ref_ids, gt_refs, name):
    """
    labels: (N,) real_idx 空间; reads: list; gt_ref_ids: (N,) 每条 read 的 GT ref id。
    流程: 对每簇做 MV consensus -> build_cluster_to_gt -> evaluate_reconstruction。
    """
    # 1. 每簇 consensus
    cl_to_real = defaultdict(list)
    for ridx, lab in enumerate(labels):
        if lab >= 0:
            cl_to_real[int(lab)].append(ridx)
    consensus = {}
    for cid, real_list in cl_to_real.items():
        seqs = [reads[r] for r in real_list]
        consensus[cid] = mv_consensus(seqs, REF_LENGTH)

    # 2. cluster -> gt (复用官方口径, 用 gt_ref_ids 当 gt_tags)
    cluster_to_gt, _purity = build_cluster_to_gt(labels, gt_ref_ids)

    # 3. SR
    r = evaluate_reconstruction(consensus, cluster_to_gt, gt_refs, name=name)
    return r["success_rate"], r


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SPIKE: evidential 乘法拆分判据 vs 纯 edit (只读)")
    print("=" * 70)

    # --- 加载 ---
    print("\n[1/5] 加载 labels / strength / reads ...")
    labels = np.loadtxt(LABELS_PATH, dtype=np.int64)
    state = torch.load(STATE_PATH, map_location="cpu")
    strength = np.asarray(state["strength"], dtype=np.float32)
    reads, _clover = load_reads_from_readtxt(READ_TXT)   # 330788, real_idx 空间
    print(f"   labels:   {labels.shape}, 非-1: {(labels>=0).sum():,}")
    print(f"   strength: {strength.shape}, mean={strength[labels>=0].mean():.2f}")
    print(f"   reads:    {len(reads):,}")
    assert len(labels) == len(strength) == len(reads), \
        f"长度不一致! labels={len(labels)} strength={len(strength)} reads={len(reads)}"

    s_ref = float(strength[labels >= 0].mean())
    print(f"   S_ref (全局均值 strength) = {s_ref:.2f}")

    # --- GT 对齐 (序列做 key, 铁律) ---
    print("\n[2/5] GT 对齐 (序列做 key) ...")
    # load_gt_tags_file 直接返回 (seq_to_tag, tag_to_reads), 复用官方链路, 不自建反向映射
    seq_to_tag, tag_to_reads = load_gt_tags_file(TAGS_PATH)
    gt_refs = load_gt_refs_fasta(REFS_PATH)                          # {ref_id: seq}
    tag_to_ref = build_tag_to_ref_mapping(tag_to_reads, gt_refs, ref_len=REF_LENGTH)
    gt_ref_ids = match_reads_to_gt(reads, seq_to_tag, tag_to_ref)    # (N,) real_idx 空间

    # --- baseline: 不拆 (Clover 原始) 的 SR, 作 sanity check ---
    print("\n[3/5] Sanity: 不拆 SR (应 ~0.9107) ...")
    sr_nosplit, _ = compute_sr(labels, reads, gt_ref_ids, gt_refs, "no_split")
    print(f"   不拆 SR = {sr_nosplit:.4f}")

    # --- 预计算候选 (最贵步骤, 只跑一次) ---
    print("\n[4/5] 预计算拆分候选 (edit 二分, 只跑一次) ...")
    candidates, n_examined, _ = precompute_split_candidates(labels, strength, reads)
    print(f"   考察簇(size>={MIN_SPLIT}): {n_examined:,}")
    print(f"   产生候选(二分成功):       {len(candidates):,}")

    # --- 判据 A: 纯 edit ---
    print("\n[5/5] 跑判据对比 ...")
    lab_edit, cids_edit, n_edit = apply_criterion(labels, candidates, "edit", TAU_BASE, s_ref)
    sr_edit, _ = compute_sr(lab_edit, reads, gt_ref_ids, gt_refs, f"edit_tau{TAU_BASE}")

    # --- 判据 B: 乘法, 扫 tau' ---
    results = []
    for tp in TAU_PRIME_LST:
        lab_m, cids_m, n_m = apply_criterion(labels, candidates, "mult", tp, s_ref)
        sr_m, _ = compute_sr(lab_m, reads, gt_ref_ids, gt_refs, f"mult_tau{tp}")
        # 行为差异
        only_edit = cids_edit - cids_m      # 纯edit拆、乘法不拆 (被保护)
        only_mult = cids_m - cids_edit      # 乘法拆、纯edit不拆
        # 被保护的低 strength 簇: only_edit 中 min(SA,SB) < LOW_S_FACTOR*s_ref
        cand_by_cid = {c["cid"]: c for c in candidates}
        protected_lowS = sum(
            1 for cid in only_edit
            if min(cand_by_cid[cid]["SA"], cand_by_cid[cid]["SB"]) < LOW_S_FACTOR * s_ref
        )
        diff = len(only_edit) + len(only_mult)
        results.append({
            "tau": tp, "sr": sr_m, "n_split": n_m,
            "diff": diff, "only_edit": len(only_edit),
            "only_mult": len(only_mult), "protected_lowS": protected_lowS,
        })

    # --- 输出对比表 ---
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)
    print(f"  不拆基线 SR        : {sr_nosplit:.4f}")
    print(f"  纯 edit (tau={TAU_BASE}) SR : {sr_edit:.4f}  (拆 {n_edit:,} 簇)  <- 守 0.9699 看这个")
    print()
    print(f"  {'判据':<18}{'tau':>4}{'SR':>9}{'拆分簇':>8}"
          f"{'vs_edit差异':>11}{'被保护':>8}{'多拆':>6}{'低S保护':>9}")
    print("  " + "-" * 70)
    print(f"  {'纯edit(基线)':<18}{TAU_BASE:>4}{sr_edit:>9.4f}{n_edit:>8,}"
          f"{0:>11}{0:>8}{0:>6}{0:>9}")
    for r in results:
        flag = "  <= 守住" if r["sr"] >= 0.9699 else ("  (略降)" if r["sr"] >= sr_edit - 0.003 else "  (掉)")
        print(f"  {'乘法evidential':<18}{r['tau']:>4}{r['sr']:>9.4f}{r['n_split']:>8,}"
              f"{r['diff']:>11}{r['only_edit']:>8}{r['only_mult']:>6}{r['protected_lowS']:>9}{flag}")

    print()
    print("  列含义:")
    print("    被保护  = 纯edit会拆但乘法不拆的簇数 (only_edit)")
    print("    多拆    = 乘法拆但纯edit不拆的簇数   (only_mult)")
    print(f"    低S保护 = 被保护簇中 min(S)<{LOW_S_FACTOR}*S_ref 的数量 (对应论文证据)")
    print()
    print("  论文叙事用: 低S保护>0 且 SR守住 -> 不确定性确实在调节拆分决策")
    print("=" * 70)


if __name__ == "__main__":
    main()