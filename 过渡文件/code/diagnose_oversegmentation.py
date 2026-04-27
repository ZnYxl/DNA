#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  diagnose_oversegmentation.py — Clover 过分割碎片簇诊断
================================================================================

目的:
  R0 (Clover) 把 1,696 个 GT 分子切成 2-5 个碎片簇. 这些碎片簇的 consensus
  (从 ref.txt 取出) 是否彼此一致? 这决定了 MNN 合并是否有效.

输出分类:
  A (一致):  碎片 consensus 两两 ED ≤ ed_threshold_A (默认 2)
            → MNN 合并必然有效, 值得开启
  B (分散):  存在两两 ED ≥ ed_threshold_B (默认 5)
            → MNN 会被 ED≤5 防线拒绝, 合并无效, 需要其他方案
  C (中间):  max 两两 ED 在 (A, B) 之间
            → 合并可能提 Recall 但不一定提 SR

关键辅助信息:
  - 每个碎片簇 size 分布 (1-read 孤儿 vs 有一定基础的小簇)
  - 每个 GT 的真值序列 vs 碎片 consensus 的 ED (看这些碎片里有没有一个是对的)
  - Jaccard 相似度 (与 MNN 的 seq_jaccard_threshold=0.15 对齐)
  - 如果提供 consensus_path, 可以用实际的 v17 R1 consensus 代替 ref.txt
    (--consensus_path /path/to/iter_1_step2/consensus/consensus_XXX.fasta)

用法:
  cd /mnt/st_data/liangxinyi/code/
  python diagnose_oversegmentation.py \\
      --experiment_dir /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/ \\
      --gt_refs /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/reads.fasta \\
      --gt_tags /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/seq1d_tags_reads.txt

可选:
  --consensus_path       替代 ref.txt, 用指定轮次的 consensus FASTA
  --primer_prefix 20     Jaccard 剔除引物区 (与 MNN 对齐)
  --primer_suffix 20
  --max_examples 20      每类别列前 N 个示例
  --output_dir ./diag/   保存 .tsv 明细
"""
import os
import sys
import argparse
import time
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np


# ───────────────────────────────────────────────────────────────────
# ED 引擎 (edlib 优先, fallback 到 rapidfuzz, 再 fallback 到手写 DP)
# ───────────────────────────────────────────────────────────────────
def _resolve_ed_engine():
    try:
        import edlib
        def _ed(a, b):
            return edlib.align(a, b, mode="NW", task="distance")['editDistance']
        return _ed, "edlib"
    except ImportError:
        pass
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance, "rapidfuzz"
    except ImportError:
        pass
    def _ed_dp(a, b):
        m, n = len(a), len(b)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i-1] == b[j-1] else 1
                curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
            prev = curr
        return prev[n]
    return _ed_dp, "dp_fallback"


ED_FUNC, ED_ENGINE = _resolve_ed_engine()


# ───────────────────────────────────────────────────────────────────
# Jaccard (与 merge_clusters.py 对齐)
# ───────────────────────────────────────────────────────────────────
def _kmer_jaccard(seq_a: str, seq_b: str, k: int = 8,
                  primer_prefix: int = 0, primer_suffix: int = 0) -> float:
    if primer_prefix > 0 or primer_suffix > 0:
        end_a = len(seq_a) - primer_suffix if primer_suffix > 0 else len(seq_a)
        end_b = len(seq_b) - primer_suffix if primer_suffix > 0 else len(seq_b)
        seq_a = seq_a[primer_prefix:end_a]
        seq_b = seq_b[primer_prefix:end_b]
    if len(seq_a) < k or len(seq_b) < k:
        return 0.0
    set_a = set(seq_a[i:i+k] for i in range(len(seq_a) - k + 1))
    set_b = set(seq_b[i:i+k] for i in range(len(seq_b) - k + 1))
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


# ───────────────────────────────────────────────────────────────────
# 数据加载
# ───────────────────────────────────────────────────────────────────
def load_reads_and_clover_labels(read_path):
    """
    read.txt 格式: 每个 Clover 簇用 '=====分隔符=====' 分隔,
                  块索引 = cluster_id (从 0 起).
    第一块在第一个分隔符之前, 直接从 cid=0 开始.

    Returns:
        reads:  list[str]
        labels: list[int], 每条 read 对应的 cluster_id
    """
    reads = []
    labels = []
    cid = 0
    with open(read_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if '分隔符' in s or s.startswith('====='):
                cid += 1
                continue
            reads.append(s)
            labels.append(cid)
    return reads, labels


def load_ref_txt(ref_path):
    """每行一条 consensus, 行号即 cluster_id."""
    ref = {}
    with open(ref_path, 'r') as f:
        for cid, line in enumerate(f):
            s = line.strip()
            if s:
                ref[cid] = s
    return ref


def load_fasta(path):
    """FASTA: >cluster_<id>\\n<seq>\\n"""
    ref = {}
    cur_cid = None
    cur_seq = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if s.startswith('>'):
                if cur_cid is not None:
                    ref[cur_cid] = ''.join(cur_seq)
                header = s[1:]
                try:
                    cur_cid = int(header.split('_')[-1])
                except ValueError:
                    cur_cid = None
                cur_seq = []
            elif cur_cid is not None:
                cur_seq.append(s)
        if cur_cid is not None:
            ref[cur_cid] = ''.join(cur_seq)
    return ref


def load_gt_tags_OLD_WRONG(tags_path, reads):
    """[DELETED — 错误地把 tag 当成 gt_id. 用 build_tag_to_ref 代替]"""
    raise NotImplementedError("使用 load_gt_tags_detailed + build_tag_to_ref")


def load_gt_refs_fasta(path):
    """
    加载 GT reference FASTA. header 为数字，作为 gt_id.
    同 diagnose_sr_regression.py 的 load_gt_refs 逻辑 —— 按出现顺序从 0 编号.
    (gt_id 最终由 build_tag_to_ref 通过 pseudo-ref 投票得到, 与这里的编号一致)
    """
    refs = {}
    name = None
    seq_lines = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if s.startswith('>'):
                if name is not None:
                    refs[len(refs)] = ''.join(seq_lines).upper()
                name = s[1:]
                seq_lines = []
            else:
                seq_lines.append(s)
        if name is not None:
            refs[len(refs)] = ''.join(seq_lines).upper()
    return refs


def load_gt_tags_detailed(tags_path, reads):
    """
    加载 tags 文件 并建立 seq→tag 和 tag→[reads] 映射.
    同 diagnose_sr_regression.py 的 load_gt_tags 逻辑.
    Returns:
        seq_to_tag: dict[str, int]
        tag_to_reads: dict[int, list[str]]
    """
    from collections import defaultdict
    seq_to_tag = {}
    tag_to_reads = defaultdict(list)
    with open(tags_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    tag = int(parts[0])
                    seq = parts[1].strip().upper()
                    seq_to_tag[seq] = tag
                    tag_to_reads[tag].append(seq)
                except ValueError:
                    pass
    return seq_to_tag, tag_to_reads


def build_tag_to_ref(tag_to_reads, gt_refs):
    """
    Tag → Reference ID 映射 (完全照抄 diagnose_sr_regression.py).
    先 pseudo-ref 精确匹配, 剩余用 ED ≤ 20 近似搜索 (长度 ±3 预筛).
    """
    from collections import Counter, defaultdict
    # 每个 tag 的 majority vote → pseudo-reference
    print(f"   构建 pseudo-references ({len(tag_to_reads)} tags)...")
    tag_pseudo = {}
    for tag, seqs in tag_to_reads.items():
        if not seqs:
            continue
        max_len = max(len(s) for s in seqs)
        consensus = []
        for pos in range(max_len):
            counts = Counter()
            for s in seqs:
                if pos < len(s):
                    counts[s[pos]] += 1
            consensus.append(counts.most_common(1)[0][0])
        tag_pseudo[tag] = ''.join(consensus)

    # 精确匹配
    ref_seq_to_id = {seq: rid for rid, seq in gt_refs.items()}
    tag_to_ref = {}
    exact = 0
    unmatched = []
    for tag, pseudo in tag_pseudo.items():
        rid = ref_seq_to_id.get(pseudo)
        if rid is not None:
            tag_to_ref[tag] = rid
            exact += 1
        else:
            unmatched.append(tag)
    print(f"   精确匹配: {exact}/{len(tag_pseudo)} tags")

    # ED 近似匹配
    ed_match = 0
    if unmatched:
        print(f"   ED 近似匹配剩余 {len(unmatched)} tags...")
        ref_by_len = defaultdict(list)
        for rid, seq in gt_refs.items():
            ref_by_len[len(seq)].append((rid, seq))
        for idx, tag in enumerate(unmatched):
            if (idx + 1) % 500 == 0:
                print(f"      进度: {idx+1}/{len(unmatched)}", end='\r')
            pseudo = tag_pseudo[tag]
            plen = len(pseudo)
            best_ed = 999
            best_rid = -1
            for dl in range(4):
                for sign in [0, 1, -1]:
                    check_len = plen + sign * dl
                    for rid, ref_seq in ref_by_len.get(check_len, []):
                        ed = ED_FUNC(pseudo, ref_seq)
                        if ed < best_ed:
                            best_ed = ed; best_rid = rid
                        if ed == 0: break
                    if best_ed == 0: break
                if best_ed == 0: break
            if best_ed <= 20:
                tag_to_ref[tag] = best_rid
                ed_match += 1
        print()
    print(f"   总映射: {len(tag_to_ref)}/{len(tag_pseudo)} "
          f"(精确={exact}, ED近似={ed_match})")
    return tag_to_ref


def match_reads_to_gt(reads, seq_to_tag, tag_to_ref):
    """每条 read → GT reference ID (同 diagnose_sr_regression.py)"""
    n = len(reads)
    gt_ref_ids = [-1] * n
    matched = 0
    for i, seq in enumerate(reads):
        tag = seq_to_tag.get(seq.upper(), -1)
        if tag >= 0:
            rid = tag_to_ref.get(tag, -1)
            gt_ref_ids[i] = rid
            if rid >= 0:
                matched += 1
    print(f"   Read→Ref 匹配: {matched}/{n}")
    return gt_ref_ids


# ───────────────────────────────────────────────────────────────────
# 主诊断
# ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment_dir', required=True,
                    help='e.g. .../Experiments/seq_1d/')
    ap.add_argument('--gt_refs', required=True,
                    help='GT reference FASTA (reads.fasta)')
    ap.add_argument('--gt_tags', required=True,
                    help='GT tags file (seq1d_tags_reads.txt)')
    ap.add_argument('--consensus_path', default=None,
                    help='(可选) 用某轮次的 FASTA 替代 ref.txt, '
                         '例如 iter_1_step2/consensus/consensus_XXX.fasta')
    ap.add_argument('--ed_threshold_A', type=int, default=2,
                    help='A 类判据: 两两 ED 全部 ≤ 此值 (默认 2)')
    ap.add_argument('--ed_threshold_B', type=int, default=5,
                    help='B 类判据: 存在两两 ED > 此值 (默认 5)')
    ap.add_argument('--primer_prefix', type=int, default=20,
                    help='Jaccard 剔除前端引物长度 (默认 20, 与 merge 对齐)')
    ap.add_argument('--primer_suffix', type=int, default=20,
                    help='Jaccard 剔除后端引物长度 (默认 20, 与 merge 对齐)')
    ap.add_argument('--max_examples', type=int, default=20,
                    help='每类别列前 N 个示例 (默认 20)')
    ap.add_argument('--output_dir', default=None,
                    help='(可选) 保存 .tsv 明细的目录')
    args = ap.parse_args()

    t_start = time.time()
    print("=" * 70)
    print("  🔬 Clover 过分割碎片簇诊断")
    print("=" * 70)
    print(f"  ED engine: {ED_ENGINE}")
    print()

    # ------------------------------------------------------------------
    # 1. 加载
    # ------------------------------------------------------------------
    print("📂 加载数据...")
    feddna_dir = os.path.join(args.experiment_dir, "03_FedDNA_In")
    read_path = os.path.join(feddna_dir, "read.txt")
    ref_path = os.path.join(feddna_dir, "ref.txt")

    if not os.path.exists(read_path):
        print(f"❌ read.txt 未找到: {read_path}")
        sys.exit(1)

    reads, clover_labels = load_reads_and_clover_labels(read_path)
    n_clover = len(set(l for l in clover_labels if l >= 0))
    print(f"   ✅ Reads: {len(reads):,}")
    print(f"   ✅ Clover clusters: {n_clover}")

    # Consensus 源: consensus_path (指定轮次) 或 ref.txt (Clover MV)
    if args.consensus_path:
        if not os.path.exists(args.consensus_path):
            print(f"❌ consensus_path 未找到: {args.consensus_path}")
            sys.exit(1)
        consensus = load_fasta(args.consensus_path)
        print(f"   ✅ Consensus 源: {args.consensus_path}")
        print(f"      {len(consensus)} 条 consensus")
    else:
        if not os.path.exists(ref_path):
            print(f"❌ ref.txt 未找到: {ref_path}")
            sys.exit(1)
        consensus = load_ref_txt(ref_path)
        print(f"   ✅ Consensus 源: ref.txt (Clover MV)")
        print(f"      {len(consensus)} 条 consensus")

    gt_refs = load_gt_refs_fasta(args.gt_refs)
    print(f"   ✅ GT references: {len(gt_refs)}")

    # ──────────────────────────────────────────────────────────────
    # 关键改动: tags 文件的第 1 列是 TAG id, 不是 reference id.
    # 正确流程 (同 diagnose_sr_regression.py):
    #   tag → pseudo-reference (majority vote) → 精确/ED 匹配 reads.fasta → gt_id
    # ──────────────────────────────────────────────────────────────
    seq_to_tag, tag_to_reads = load_gt_tags_detailed(args.gt_tags, reads)
    print(f"   ✅ GT tags: {len(seq_to_tag):,} 唯一序列, {len(tag_to_reads):,} 唯一 tags")

    tag_to_ref = build_tag_to_ref(tag_to_reads, gt_refs)
    gt_labels = match_reads_to_gt(reads, seq_to_tag, tag_to_ref)
    n_gt = len(set(g for g in gt_labels if g >= 0))
    print(f"   ✅ 唯一 GT: {n_gt}")

    # ------------------------------------------------------------------
    # 2. 建立 GT → Clover 碎片簇 的映射
    #    对每条 read 同时有 gt_id 和 clover_cid, 投票给该 GT 的主导簇
    # ------------------------------------------------------------------
    print()
    print("🔍 构建 GT → Clover 碎片簇映射...")
    gt_to_cluster_count: Dict[int, Counter] = defaultdict(Counter)
    cluster_size: Counter = Counter()
    for i, (g, c) in enumerate(zip(gt_labels, clover_labels)):
        if g >= 0 and c >= 0:
            gt_to_cluster_count[g][c] += 1
        if c >= 0:
            cluster_size[c] += 1

    print(f"   GT 分子数: {len(gt_to_cluster_count)}")

    # 统计每个 GT 被切成多少片
    n_frag_hist = Counter()
    for g, cnt in gt_to_cluster_count.items():
        n_frag_hist[len(cnt)] += 1
    print(f"   片数分布:")
    for n, c in sorted(n_frag_hist.items()):
        print(f"      {n} 片: {c}")

    # ------------------------------------------------------------------
    # 3. 筛出过分割的 GT (>=2 片)
    # ------------------------------------------------------------------
    fragmented = {g: cnt for g, cnt in gt_to_cluster_count.items() if len(cnt) >= 2}
    print(f"\n   被过分割的 GT: {len(fragmented)}")

    # ------------------------------------------------------------------
    # 4. 对每个过分割 GT 做两两 ED + GT-vs-consensus ED
    #    → 分类到 A / B / C
    # ------------------------------------------------------------------
    print()
    print("🧪 两两 ED 分析...")
    print("   A: 碎片两两 ED ≤ {} (一致, MNN 可合)".format(args.ed_threshold_A))
    print("   B: 碎片两两 max_ED > {} (分散, MNN 会被拒)".format(args.ed_threshold_B))
    print("   C: 介于 A 和 B 之间")
    print()

    results = []   # list of dicts
    missing_consensus = 0
    t0 = time.time()

    for idx, (g, cnt) in enumerate(fragmented.items()):
        if (idx + 1) % 500 == 0:
            print(f"      进度: {idx+1}/{len(fragmented)}, "
                  f"{(time.time()-t0):.1f}s")

        cids = sorted(cnt.keys(), key=lambda c: -cnt[c])   # 按主导程度排
        seqs = [consensus.get(c) for c in cids]
        if any(s is None for s in seqs):
            missing_consensus += 1
            continue

        # 两两 ED + Jaccard
        n = len(seqs)
        max_ed, min_ed = 0, 10**9
        sum_ed, cnt_ed = 0, 0
        max_jac, min_jac = 0.0, 1.0
        sum_jac, cnt_jac = 0.0, 0
        pair_list = []
        for i in range(n):
            for j in range(i+1, n):
                ed = ED_FUNC(seqs[i], seqs[j])
                jac = _kmer_jaccard(seqs[i], seqs[j], k=8,
                                    primer_prefix=args.primer_prefix,
                                    primer_suffix=args.primer_suffix)
                pair_list.append((cids[i], cids[j], ed, jac))
                max_ed = max(max_ed, ed)
                min_ed = min(min_ed, ed)
                sum_ed += ed; cnt_ed += 1
                max_jac = max(max_jac, jac)
                min_jac = min(min_jac, jac)
                sum_jac += jac; cnt_jac += 1

        avg_ed = sum_ed / cnt_ed
        avg_jac = sum_jac / cnt_jac

        # GT 真值 ED
        gt_ref = gt_refs.get(g, "")
        gt_ed_list = [ED_FUNC(gt_ref, s) for s in seqs] if gt_ref else []
        min_gt_ed = min(gt_ed_list) if gt_ed_list else -1

        # 类别
        if max_ed <= args.ed_threshold_A:
            cat = 'A'
        elif max_ed > args.ed_threshold_B:
            cat = 'B'
        else:
            cat = 'C'

        sizes = [cnt[c] for c in cids]
        results.append({
            'gt_id': g,
            'n_frag': n,
            'cids': cids,
            'sizes': sizes,
            'size_min': min(sizes),
            'size_max': max(sizes),
            'max_ed': max_ed,
            'min_ed': min_ed,
            'avg_ed': avg_ed,
            'min_jaccard': min_jac,
            'max_jaccard': max_jac,
            'avg_jaccard': avg_jac,
            'min_gt_ed': min_gt_ed,
            'cat': cat,
            'pairs': pair_list,
        })

    if missing_consensus > 0:
        print(f"   ⚠️ {missing_consensus} 个过分割 GT 的 consensus 不完整, 已跳过")

    # ------------------------------------------------------------------
    # 5. 总结统计
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  📊 分类总体分布")
    print("=" * 70)
    cat_hist = Counter(r['cat'] for r in results)
    total = len(results)
    for cat in ['A', 'B', 'C']:
        c = cat_hist.get(cat, 0)
        pct = c / total * 100 if total else 0
        print(f"   {cat}: {c:>6}  ({pct:5.1f}%)")
    print(f"   总: {total}")

    # 按 n_frag 分层
    print()
    print("   分类 × 碎片数:")
    print(f"   {'n_frag':>8} {'A':>8} {'B':>8} {'C':>8}")
    by_nfrag = defaultdict(Counter)
    for r in results:
        by_nfrag[r['n_frag']][r['cat']] += 1
    for n in sorted(by_nfrag.keys()):
        row = by_nfrag[n]
        print(f"   {n:>8} {row.get('A', 0):>8} {row.get('B', 0):>8} "
              f"{row.get('C', 0):>8}")

    # ------------------------------------------------------------------
    # 6. MNN 模拟: 当前 merge_clusters.py 的阈值下, A 里有多少会被合并?
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  🔗 MNN 实际可合并性模拟")
    print("  (与 merge_clusters.py 阈值对齐: Jaccard ≥ 0.15, ED ≤ 5)")
    print("=" * 70)

    would_merge_A = 0; would_merge_B = 0; would_merge_C = 0
    blocked_by_ed = 0
    blocked_by_jac = 0
    mergeable_examples = defaultdict(list)
    blocked_examples = defaultdict(list)

    for r in results:
        # 这里用"所有两两对都满足 ED≤5 且 Jac≥0.15"作为可合并的保守近似
        # (真实 MNN 还要求是互近邻, 这里只看序列防线是否放行)
        all_ok = True
        failure_reason = None
        for (ca, cb, ed, jac) in r['pairs']:
            if ed > 5:
                all_ok = False
                failure_reason = 'ED>5'
                break
            if jac < 0.15:
                all_ok = False
                failure_reason = 'Jac<0.15'
                break
        if all_ok:
            if r['cat'] == 'A': would_merge_A += 1
            elif r['cat'] == 'B': would_merge_B += 1
            else: would_merge_C += 1
            if len(mergeable_examples[r['cat']]) < args.max_examples:
                mergeable_examples[r['cat']].append(r)
        else:
            if failure_reason == 'ED>5':
                blocked_by_ed += 1
            else:
                blocked_by_jac += 1
            if len(blocked_examples[r['cat']]) < args.max_examples:
                blocked_examples[r['cat']].append((r, failure_reason))

    print(f"   序列防线放行 (A/B/C): {would_merge_A}/{would_merge_B}/"
          f"{would_merge_C}")
    print(f"   被 ED>5 拒绝:    {blocked_by_ed}")
    print(f"   被 Jaccard<0.15 拒绝: {blocked_by_jac}")

    pass_A_pct = would_merge_A / max(cat_hist.get('A', 0), 1) * 100
    pass_B_pct = would_merge_B / max(cat_hist.get('B', 0), 1) * 100
    pass_C_pct = would_merge_C / max(cat_hist.get('C', 0), 1) * 100
    print(f"   A 类放行率: {pass_A_pct:.1f}%")
    print(f"   B 类放行率: {pass_B_pct:.1f}%  ← 理论上应该低")
    print(f"   C 类放行率: {pass_C_pct:.1f}%")

    # ------------------------------------------------------------------
    # 7. GT 真值 ED 分布 (看有没有"反正碎片里有一个已经对了"的情况)
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  🎯 GT 真值对齐: min(ED(consensus_i, GT))")
    print("  (若很多 GT 的 min_gt_ed=0, 说明碎片簇里已有正确重建,"
          " 合并后失败的概率低)")
    print("=" * 70)
    gt_ed_bins = Counter()
    valid_results = [r for r in results if r['min_gt_ed'] >= 0]
    for r in valid_results:
        e = r['min_gt_ed']
        if e == 0: gt_ed_bins['=0'] += 1
        elif e == 1: gt_ed_bins['=1'] += 1
        elif e == 2: gt_ed_bins['=2'] += 1
        elif e <= 5: gt_ed_bins['3-5'] += 1
        else: gt_ed_bins['>5'] += 1
    n_v = len(valid_results)
    for k in ['=0', '=1', '=2', '3-5', '>5']:
        c = gt_ed_bins.get(k, 0)
        pct = c / n_v * 100 if n_v else 0
        print(f"      min_gt_ed {k:>3}: {c:>5} ({pct:5.1f}%)")

    # ------------------------------------------------------------------
    # 8. 示例输出 (每类前 N 个)
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"  📋 示例: A 类 (前 {min(args.max_examples, len(mergeable_examples['A']))} 个, "
          f"序列防线放行, 合并后必对)")
    print("=" * 70)
    for r in mergeable_examples['A'][:args.max_examples]:
        sz = ','.join(str(s) for s in r['sizes'])
        print(f"   GT#{r['gt_id']:>5} n_frag={r['n_frag']} "
              f"sizes=[{sz}] max_ed={r['max_ed']} "
              f"min_jac={r['min_jaccard']:.3f} "
              f"min_gt_ed={r['min_gt_ed']}")

    print()
    print("=" * 70)
    print(f"  📋 示例: B 类 (前 {min(args.max_examples, len(blocked_examples['B']))} 个, "
          f"碎片 consensus 彼此分散)")
    print("=" * 70)
    for r, reason in blocked_examples['B'][:args.max_examples]:
        sz = ','.join(str(s) for s in r['sizes'])
        print(f"   GT#{r['gt_id']:>5} n_frag={r['n_frag']} "
              f"sizes=[{sz}] max_ed={r['max_ed']} "
              f"min_jac={r['min_jaccard']:.3f} "
              f"min_gt_ed={r['min_gt_ed']}  ❌{reason}")

    print()
    print("=" * 70)
    print(f"  📋 示例: C 类 (前 {min(args.max_examples, args.max_examples)} 个)")
    print("=" * 70)
    c_samples = mergeable_examples['C'][:args.max_examples//2] + \
                [x[0] for x in blocked_examples['C'][:args.max_examples//2]]
    for r in c_samples:
        sz = ','.join(str(s) for s in r['sizes'])
        print(f"   GT#{r['gt_id']:>5} n_frag={r['n_frag']} "
              f"sizes=[{sz}] max_ed={r['max_ed']} "
              f"min_jac={r['min_jaccard']:.3f} "
              f"min_gt_ed={r['min_gt_ed']}")

    # ------------------------------------------------------------------
    # 9. TSV 输出
    # ------------------------------------------------------------------
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        tsv_path = os.path.join(args.output_dir, "oversegmentation_diag.tsv")
        with open(tsv_path, 'w') as f:
            f.write("gt_id\tn_frag\tsizes\tcids\tmax_ed\tmin_ed\tavg_ed\t"
                    "min_jaccard\tmax_jaccard\tavg_jaccard\tmin_gt_ed\tcat\n")
            for r in results:
                f.write(f"{r['gt_id']}\t{r['n_frag']}\t"
                        f"{','.join(str(s) for s in r['sizes'])}\t"
                        f"{','.join(str(c) for c in r['cids'])}\t"
                        f"{r['max_ed']}\t{r['min_ed']}\t{r['avg_ed']:.2f}\t"
                        f"{r['min_jaccard']:.4f}\t{r['max_jaccard']:.4f}\t"
                        f"{r['avg_jaccard']:.4f}\t"
                        f"{r['min_gt_ed']}\t{r['cat']}\n")
        print(f"\n   💾 明细 TSV: {tsv_path}")

    # ------------------------------------------------------------------
    # 10. 总结性建议
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  🎯 诊断结论 (读我!)")
    print("=" * 70)
    pct_A = cat_hist.get('A', 0) / max(total, 1) * 100
    pct_B = cat_hist.get('B', 0) / max(total, 1) * 100
    pct_C = cat_hist.get('C', 0) / max(total, 1) * 100

    if pct_A >= 60:
        verdict = "✅ MNN 合并极可能有效"
        action = (f"建议直接开启 --disable_merge=False 跑 v18. "
                  f"预期能挽回 ~{would_merge_A} 个 A 类 GT (若合并后 fusion 正确).")
    elif pct_B >= 50:
        verdict = "⚠️ MNN 合并很可能被序列防线挡住"
        action = ("不建议直接开 MNN. 考虑: (1) 降 ED 防线到 ≤8; "
                  "(2) 放宽 Jaccard 到 0.10; "
                  "(3) 或者根本上换方向, 从 Step1 encoder 入手让碎片 embedding 靠拢.")
    else:
        verdict = "🟡 混合场景, 开启 MNN 效果不确定"
        action = (f"A={pct_A:.1f}% (放行 {would_merge_A}), "
                  f"B={pct_B:.1f}% (拒绝), C={pct_C:.1f}%. "
                  f"可以试跑 v18, 但收益上限 ~{would_merge_A} 个 GT.")

    print(f"  分布:  A={pct_A:.1f}%  B={pct_B:.1f}%  C={pct_C:.1f}%")
    print(f"  结论:  {verdict}")
    print(f"  行动:  {action}")
    print()
    print(f"  Clover 过分割潜在上限: {len(fragmented)} 个 GT")
    print(f"  MNN 实际可合并 (A类放行): {would_merge_A} 个")
    print(f"  Recall 改进上限估计:       +{would_merge_A / 11826 * 100:.2f}pp")
    print()
    print(f"  总耗时: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()