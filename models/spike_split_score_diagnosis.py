#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# spike_split_score_diagnosis.py
# 诊断: 拆分判定式里"strength 该乘还是该除"—— 用 GT 当裁判, 数据说话。
# -----------------------------------------------------------------------------
# 对 R1 每个被考察拆分的簇 (size >= min_size):
#   1. 复用生产代码 _split_two 分出 A/B 子簇 (保证与真实拆分行为一致)
#   2. 算 cons_A/cons_B、edit(cons_A,cons_B)、S_A/S_B (子簇平均 strength)、
#      u_epi_A/u_epi_B (子簇平均认知不确定性)
#   3. 用 GT 判定该簇"是否真该拆": 簇内存在第二个 GT 分子且其 read 数
#      >= max(GT_MIN_ABS, GT_MIN_FRAC * 簇大小) → 该拆(label=1); 否则不该拆(0)
#   4. 对 4 个候选 score 算区分"该拆/不该拆"的 AUROC:
#        score_edit  = edit                          (现版基线)
#        score_div   = edit / min(S_A, S_B)          (老师版: strength 当分母)
#        score_mul   = edit * min(S_A, S_B) / S_ref  (乘法版: strength 当门控)
#        score_uepi  = edit * (1 - mean(u_epi_B))    (u_epi 门控版, 参考)
#   AUROC 最高者 = 区分力最强 = 最该用的公式。
#
# 只读, 不改任何生产代码 / 不落盘标签。复用 cluster_split 的 _split_two/_mv_consensus。
#
# 用法:
#   python spike_split_score_diagnosis.py
#   (路径默认 Seq_1D R1, 可改下方 CONFIG)
# =============================================================================
import os
import sys
import numpy as np
import torch
from collections import defaultdict, Counter

# ── 路径配置 ──
EXP_DIR     = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d"
READ_STATE  = os.path.join(EXP_DIR, "04_Iterative_Labels", "read_state_204939.pt")  # R1
GT_TAGS     = os.path.join(EXP_DIR, "seq1d_tags_reads.txt")
REF_LENGTH  = 196
MIN_SIZE    = 6        # 与生产 split_min_size 一致
MAX_PAIRWISE= 80       # 与生产一致
# GT "该拆"判定阈值
GT_MIN_ABS  = 3        # 第二 GT 分子至少 3 条 read
GT_MIN_FRAC = 0.05     # 且至少占簇 5%

# 让 import models.* 生效
CODE_ROOT = "/mnt/st_data/liangxinyi/code"
for p in (CODE_ROOT, os.path.join(CODE_ROOT, "models")):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.cluster_split import _split_two, _mv_consensus
from models.step1_data import CloverDataLoader
from models.eval_reconstruction import levenshtein


def load_gt_tags(gt_path, reads):
    """sequence -> GT id;返回每条 read 的 GT(-1=未匹配)。"""
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
    matched = 0
    for i, r in enumerate(reads):
        g = seq_to_gt.get(r.upper())
        if g is not None:
            gt[i] = g; matched += 1
    print(f"   GT 匹配: {matched}/{len(reads)} ({100*matched/len(reads):.1f}%)")
    return gt


def gt_should_split(gt_in_cluster):
    """簇内 GT 列表 -> 是否真该拆(存在 read 数达标的第二分子)。"""
    if len(gt_in_cluster) == 0:
        return 0
    cnt = Counter(g for g in gt_in_cluster if g >= 0)
    if len(cnt) < 2:
        return 0
    n = len(gt_in_cluster)
    top2 = cnt.most_common(2)
    second_count = top2[1][1]
    if second_count >= max(GT_MIN_ABS, GT_MIN_FRAC * n):
        return 1
    return 0


def auroc(scores, labels):
    """二分类 AUROC(无 sklearn 依赖, 用秩统计)。labels: 1=正类(该拆)。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 处理并列: 同分取平均秩
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_rank = np.zeros(len(counts))
    np.add.at(sum_rank, inv, ranks)
    avg_rank_per_val = sum_rank / counts
    ranks = avg_rank_per_val[inv]
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    print("=" * 72)
    print("  🔬 拆分判定式诊断: strength 该乘还是该除? (GT 当裁判)")
    print("=" * 72)

    # ── 加载 ──
    dl = CloverDataLoader(EXP_DIR)
    reads = dl.reads
    clover = np.array(dl.clover_labels, dtype=np.int64)
    print(f"   Reads: {len(reads)}, Clover 簇: {len(set(clover[clover>=0]))}")

    st = torch.load(READ_STATE, map_location='cpu')
    strength = np.asarray(st['strength'], dtype=np.float64)  # (TOTAL_READS,) 按 real_idx
    u_epi    = np.asarray(st['u_epi'],    dtype=np.float64)
    S_ref = float(strength[strength > 0].mean())
    print(f"   全局平均 strength S_ref = {S_ref:.3f}")

    gt = load_gt_tags(GT_TAGS, reads)

    # ── 按 Clover 簇分组 (real_idx 空间) ──
    cl_to_idx = defaultdict(list)
    for ridx, lab in enumerate(clover):
        if lab >= 0:
            cl_to_idx[int(lab)].append(ridx)

    rows = []   # 每个被考察簇一行
    n_examined = 0
    for cid, ridxs in cl_to_idx.items():
        if len(ridxs) < MIN_SIZE:
            continue
        n_examined += 1

        seqs = [reads[r] for r in ridxs]
        a_loc, b_loc = _split_two(seqs, levenshtein, max_pairwise=MAX_PAIRWISE)
        if len(a_loc) < 1 or len(b_loc) < 1:
            continue
        consA = _mv_consensus([seqs[i] for i in a_loc], REF_LENGTH)
        consB = _mv_consensus([seqs[i] for i in b_loc], REF_LENGTH)
        if not consA or not consB:
            continue
        edit = levenshtein(consA, consB)

        a_ridx = [ridxs[i] for i in a_loc]
        b_ridx = [ridxs[i] for i in b_loc]
        S_A = float(strength[a_ridx].mean())
        S_B = float(strength[b_ridx].mean())
        ue_B = float(u_epi[b_ridx].mean())
        Smin = min(S_A, S_B)

        # GT 真相: 整个簇是否真混了两个分子
        gt_label = gt_should_split([gt[r] for r in ridxs])

        rows.append({
            'cid': cid, 'size': len(ridxs),
            'edit': edit, 'S_A': S_A, 'S_B': S_B, 'Smin': Smin, 'ue_B': ue_B,
            'gt': gt_label,
        })

    print(f"   被考察簇 (size>={MIN_SIZE}): {n_examined}, 成功分A/B: {len(rows)}")

    if not rows:
        print("   ❌ 无可用簇"); return

    # ── 4 个候选 score ──
    edit_arr = np.array([r['edit'] for r in rows], dtype=np.float64)
    Smin_arr = np.array([r['Smin'] for r in rows], dtype=np.float64)
    ueB_arr  = np.array([r['ue_B'] for r in rows], dtype=np.float64)
    gt_arr   = np.array([r['gt']   for r in rows], dtype=np.int64)

    score_edit = edit_arr
    score_div  = edit_arr / np.clip(Smin_arr, 1e-6, None)        # 老师版
    score_mul  = edit_arr * Smin_arr / S_ref                     # 乘法版
    score_uepi = edit_arr * (1.0 - ueB_arr)                      # u_epi 门控版

    n_pos = int(gt_arr.sum()); n_neg = len(gt_arr) - n_pos
    print(f"\n   GT 裁判: 该拆={n_pos}, 不该拆={n_neg} "
          f"(第二分子 >= max({GT_MIN_ABS}, {GT_MIN_FRAC:.0%}×size))")

    print(f"\n   {'score 公式':<34}{'AUROC':>8}")
    print(f"   {'-'*44}")
    results = [
        ("score_edit  = edit (基线)",             auroc(score_edit, gt_arr)),
        ("score_div   = edit / min(S) (老师版)",  auroc(score_div,  gt_arr)),
        ("score_mul   = edit×min(S)/Sref (乘法)", auroc(score_mul,  gt_arr)),
        ("score_uepi  = edit×(1-u_epi_B)",        auroc(score_uepi, gt_arr)),
    ]
    for name, a in results:
        print(f"   {name:<34}{a:>8.4f}")

    best = max(results, key=lambda x: (x[1] if not np.isnan(x[1]) else -1))
    print(f"\n   🏆 区分力最强: {best[0]}  (AUROC={best[1]:.4f})")
    print(f"   AUROC 越高 = 越能把'真该拆'和'不该拆'分开 = 越该用这个公式。")

    # ── 典型误判实例: 看老师除法 vs 乘法在哪些簇上判反 ──
    # 找"不该拆但 score_div 很高"(老师版会误拆) 和 "该拆但 score_div 低"(老师版会漏拆)
    print(f"\n   ── 老师除法版的风险实例 ──")
    div_rank = np.argsort(-score_div)  # score 高在前
    print(f"   不该拆(gt=0)却被 score_div 排在前列(易误拆), Top 5:")
    cnt = 0
    for i in div_rank:
        if rows[i]['gt'] == 0:
            r = rows[i]
            print(f"     cid={r['cid']:>6} size={r['size']:>4} edit={r['edit']:>3} "
                  f"S_A={r['S_A']:.1f} S_B={r['S_B']:.1f} Smin={r['Smin']:.1f} "
                  f"| div={score_div[i]:.3f} mul={score_mul[i]:.3f}")
            cnt += 1
            if cnt >= 5: break

    # ── 同时报: edit 单独已经多好(看 strength 到底加不加分) ──
    print(f"\n   ── 结论提示 ──")
    a_edit = auroc(score_edit, gt_arr)
    a_div  = auroc(score_div,  gt_arr)
    a_mul  = auroc(score_mul,  gt_arr)
    if a_mul > a_edit + 0.01 and a_mul >= a_div:
        print(f"   乘法版 AUROC({a_mul:.3f}) > 基线({a_edit:.3f}) 且 >= 老师除法({a_div:.3f})")
        print(f"   → strength 当门控(乘)有正贡献, 方向应为乘。")
    elif a_div > a_edit + 0.01 and a_div > a_mul:
        print(f"   老师除法 AUROC({a_div:.3f}) 最高 → 老师是对的, 用除法。")
    else:
        print(f"   三者接近(edit={a_edit:.3f} div={a_div:.3f} mul={a_mul:.3f})")
        print(f"   → strength 耦合增益不明显, 纯 edit 可能已足够。需谨慎评估是否值得做。")

    print("\n" + "=" * 72)
    print("  ✅ 诊断完成")
    print("=" * 72)


if __name__ == "__main__":
    main()