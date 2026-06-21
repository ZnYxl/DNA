#!/usr/bin/env python3
"""
diagnose_sr_regression.py
=========================
诊断 SR 退化: 精确定位哪些 GT 分子从"成功"变"失败", 分析原因。

输出:
  1. R0 成功但 Rn 失败的 GT 分子列表 (regression)
  2. 每个 regression 分子的: 簇大小, purity, ED, 错在哪几个位置
  3. 按簇大小分层 (大簇 vs 小簇)
  4. 按 purity 分层 (纯簇 vs 杂簇) — 区分"decoder 解错"和"聚类不纯"
  5. 输出 v5 vs v13 的 EER 趋势对比分析

用法:
  python diagnose_sr_regression.py \
      --experiment_dir /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/ \
      --gt_refs /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/reads.fasta \
      --gt_tags /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/seq1d_tags_reads.txt
"""

import argparse
import os
import sys
import glob
import numpy as np
from collections import Counter, defaultdict

# ═══════════════════════════════════════════════════════════════
# Edit Distance
# ═══════════════════════════════════════════════════════════════
try:
    import edlib
    def levenshtein(a, b):
        return edlib.align(a, b, mode="NW", task="path")['editDistance']
    def align_detail(a, b):
        """返回 (ed, cigar) 用于定位错误位置"""
        r = edlib.align(a, b, mode="NW", task="path")
        return r['editDistance'], r.get('cigar', '')
    ED_ENGINE = "edlib"
except ImportError:
    def levenshtein(a, b):
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                tmp = dp[j]
                dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = tmp
        return dp[n]
    def align_detail(a, b):
        return levenshtein(a, b), ''
    ED_ENGINE = "pure-python"


# ═══════════════════════════════════════════════════════════════
# 数据加载 (与 eval_reconstruction 兼容)
# ═══════════════════════════════════════════════════════════════
def load_reads(exp_dir):
    """加载 read.txt (与 step1_data.py CloverDataLoader 一致)"""
    read_path = os.path.join(exp_dir, "03_FedDNA_In", "read.txt")
    reads = []
    clover_labels = []
    current_cluster = 0
    with open(read_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("====="):
                current_cluster += 1
            elif line[0] in "ACGTacgt":
                reads.append(line.upper())
                clover_labels.append(current_cluster)
    return reads, np.array(clover_labels)


def load_gt_refs(path):
    """加载 GT references (FASTA)"""
    refs = {}
    name = None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    refs[len(refs)] = ''.join(seq_lines)
                name = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line.upper())
    if name is not None:
        refs[len(refs)] = ''.join(seq_lines)
    return refs


def load_gt_tags(path):
    """加载 seq → tag 映射。文件格式: tag_id<TAB>sequence"""
    seq_to_tag = {}
    tag_to_reads = defaultdict(list)
    total = 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                total += 1
                try:
                    tag = int(parts[0])
                    seq = parts[1].strip().upper()
                    seq_to_tag[seq] = tag
                    tag_to_reads[tag].append(seq)
                except ValueError:
                    pass
    print(f"   GT tags: {total:,} 行, {len(seq_to_tag):,} 唯一序列, "
          f"{len(tag_to_reads):,} 唯一 tags")
    return seq_to_tag, tag_to_reads


def build_tag_to_ref(tag_to_reads, gt_refs):
    """Tag → Reference ID 映射 (精确匹配优先 + 长度预筛 ED)"""

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

    # 建立 ref_sequence → ref_id 反向索引 (精确匹配)
    ref_seq_to_id = {}
    for rid, seq in gt_refs.items():
        ref_seq_to_id[seq] = rid

    tag_to_ref = {}
    exact = 0
    ed_match = 0

    # 第一轮：精确匹配
    unmatched_tags = []
    for tag, pseudo in tag_pseudo.items():
        rid = ref_seq_to_id.get(pseudo)
        if rid is not None:
            tag_to_ref[tag] = rid
            exact += 1
        else:
            unmatched_tags.append(tag)

    print(f"   精确匹配: {exact}/{len(tag_pseudo)} tags")

    # 第二轮：未匹配的用 ED (长度预筛 ±3, 前缀预筛)
    if unmatched_tags:
        print(f"   ED 近似匹配剩余 {len(unmatched_tags)} tags...")
        # 按长度分桶加速
        ref_by_len = defaultdict(list)
        for rid, seq in gt_refs.items():
            ref_by_len[len(seq)].append((rid, seq))

        for idx, tag in enumerate(unmatched_tags):
            if (idx + 1) % 200 == 0:
                print(f"      进度: {idx+1}/{len(unmatched_tags)}", end='\r')
            pseudo = tag_pseudo[tag]
            plen = len(pseudo)
            best_ed = 999
            best_rid = -1
            # 只搜长度 ±3 的 refs
            for dl in range(4):
                for sign in [0, 1, -1]:
                    check_len = plen + sign * dl
                    for rid, ref_seq in ref_by_len.get(check_len, []):
                        ed = levenshtein(pseudo, ref_seq)
                        if ed < best_ed:
                            best_ed = ed
                            best_rid = rid
                        if ed == 0:
                            break
                    if best_ed == 0:
                        break
                if best_ed == 0:
                    break
            if best_ed <= 20:
                tag_to_ref[tag] = best_rid
                ed_match += 1
        print()  # newline after \r

    print(f"   总映射: {len(tag_to_ref)}/{len(tag_pseudo)} "
          f"(精确={exact}, ED近似={ed_match})")
    return tag_to_ref


def match_reads_to_gt(reads, seq_to_tag, tag_to_ref):
    """每条 read → GT reference ID"""
    gt_ref_ids = np.full(len(reads), -1, dtype=int)
    for i, seq in enumerate(reads):
        tag = seq_to_tag.get(seq, -1)
        if tag >= 0:
            ref_id = tag_to_ref.get(tag, -1)
            gt_ref_ids[i] = ref_id
    matched = (gt_ref_ids >= 0).sum()
    print(f"   Read→Ref 匹配: {matched}/{len(reads)}")
    return gt_ref_ids


def parse_consensus_fasta(path):
    """解析 consensus FASTA"""
    consensus = {}
    name = None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    cid = int(name.replace("cluster_", ""))
                    consensus[cid] = ''.join(seq_lines)
                name = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line.upper())
    if name is not None:
        cid = int(name.replace("cluster_", ""))
        consensus[cid] = ''.join(seq_lines)
    return consensus


def load_ref_txt(path):
    """加载 Clover MV baseline 的 ref.txt (每行一条 consensus, 无分隔符)"""
    consensus = {}
    cid = 0
    with open(path) as f:
        for line in f:
            seq = line.strip()
            if seq:
                consensus[cid] = seq.upper()
                cid += 1
    return consensus


def discover_rounds(exp_dir):
    """自动发现各轮次的 consensus + labels"""
    results = []
    iter_dirs = sorted(glob.glob(os.path.join(exp_dir, "results", "iter_*_step2")))
    for d in iter_dirs:
        # 提取轮次号
        base = os.path.basename(d)
        try:
            round_num = int(base.split("_")[1])
        except (IndexError, ValueError):
            continue

        # 找 consensus FASTA
        fasta_files = sorted(glob.glob(os.path.join(d, "consensus", "*.fasta")),
                             key=os.path.getmtime)
        if not fasta_files:
            continue

        # 找 labels
        label_dir = os.path.join(exp_dir, "04_Iterative_Labels")
        label_files = sorted(glob.glob(os.path.join(label_dir, "refined_labels_*.txt")),
                             key=os.path.getmtime)

        # 按轮次匹配 (第 round_num 个 label 文件)
        if round_num - 1 < len(label_files):
            label_path = label_files[round_num - 1]
        elif label_files:
            label_path = label_files[-1]
        else:
            continue

        results.append((round_num, fasta_files[-1], label_path))

    return results


# ═══════════════════════════════════════════════════════════════
# 核心: 按 GT 分子粒度做 SR 归因
# ═══════════════════════════════════════════════════════════════
def analyze_per_molecule(consensus, labels, gt_ref_ids, gt_refs, reads, round_name):
    """
    对每个 GT 分子:
      - 找到对应的 cluster (majority vote)
      - 计算 ED
      - 记录簇大小、purity、错误位置

    Returns:
        {ref_id: {ed, cluster_size, purity, consensus_seq, ref_seq, ...}}
    """
    # 建立 cluster → reads 映射
    cluster_reads = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            cluster_reads[int(label)].append(i)

    # 每个 cluster 的 majority GT ref
    cluster_to_gt = {}
    cluster_purity = {}
    cluster_size = {}
    for cid, ridx_list in cluster_reads.items():
        gt_counts = Counter()
        for ri in ridx_list:
            gid = gt_ref_ids[ri]
            if gid >= 0:
                gt_counts[gid] += 1
        if gt_counts:
            maj_gt = gt_counts.most_common(1)[0][0]
            maj_count = gt_counts.most_common(1)[0][1]
            cluster_to_gt[cid] = maj_gt
            cluster_purity[cid] = maj_count / len(ridx_list)
            cluster_size[cid] = len(ridx_list)

    # 反向: GT ref → best cluster
    gt_to_clusters = defaultdict(list)
    for cid, gid in cluster_to_gt.items():
        gt_to_clusters[gid].append(cid)

    results = {}
    for ref_id, ref_seq in gt_refs.items():
        cids = gt_to_clusters.get(ref_id, [])
        if not cids:
            results[ref_id] = {
                'ed': -1, 'status': 'uncovered',
                'cluster_size': 0, 'purity': 0, 'n_clusters': 0
            }
            continue

        # 取 purity 最高的 cluster (如果有多个映到同一 GT)
        best_cid = max(cids, key=lambda c: (cluster_purity.get(c, 0), cluster_size.get(c, 0)))

        cons_seq = consensus.get(best_cid, '')
        if not cons_seq:
            results[ref_id] = {
                'ed': -1, 'status': 'no_consensus',
                'cluster_size': cluster_size.get(best_cid, 0),
                'purity': cluster_purity.get(best_cid, 0),
                'n_clusters': len(cids)
            }
            continue

        ed, cigar = align_detail(cons_seq, ref_seq)
        results[ref_id] = {
            'ed': ed,
            'status': 'success' if ed == 0 else 'fail',
            'cluster_size': cluster_size.get(best_cid, 0),
            'purity': cluster_purity.get(best_cid, 0),
            'n_clusters': len(cids),
            'cons_len': len(cons_seq),
            'ref_len': len(ref_seq),
            'len_diff': len(cons_seq) - len(ref_seq),
            'cigar': cigar,
        }

    return results


def print_regression_analysis(r0_results, rn_results, round_name, gt_refs):
    """对比 R0 和 Rn, 找出 regression 和 improvement"""
    regressions = []      # R0 成功, Rn 失败
    improvements = []     # R0 失败, Rn 成功
    persistent_fail = []  # 都失败
    persistent_ok = []    # 都成功

    for ref_id in gt_refs:
        r0 = r0_results.get(ref_id, {})
        rn = rn_results.get(ref_id, {})
        r0_ok = r0.get('ed', -1) == 0
        rn_ok = rn.get('ed', -1) == 0

        if r0_ok and not rn_ok:
            regressions.append((ref_id, r0, rn))
        elif not r0_ok and rn_ok:
            improvements.append((ref_id, r0, rn))
        elif not r0_ok and not rn_ok:
            persistent_fail.append((ref_id, r0, rn))
        else:
            persistent_ok.append((ref_id, r0, rn))

    total = len(gt_refs)
    print(f"\n{'='*70}")
    print(f"  📊 {round_name} vs R0 归因分析")
    print(f"{'='*70}")
    print(f"  R0 成功 & Rn 成功 (持续成功): {len(persistent_ok):,}")
    print(f"  R0 成功 & Rn 失败 (退化):     {len(regressions):,}  ← 重点分析")
    print(f"  R0 失败 & Rn 成功 (改善):     {len(improvements):,}")
    print(f"  R0 失败 & Rn 失败 (持续失败): {len(persistent_fail):,}")
    print(f"  SR 变化: {len(persistent_ok)+len(improvements)}/{total} → "
          f"ΔSR = {len(improvements)-len(regressions):+d}")

    if not regressions:
        print(f"  ✅ 无退化分子")
        return

    # ── 退化分子分析 ──
    print(f"\n  ── 退化分子详细分析 ({len(regressions)} 个) ──")

    # 按原因分类
    uncovered = [x for x in regressions if x[2].get('status') == 'uncovered']
    no_cons   = [x for x in regressions if x[2].get('status') == 'no_consensus']
    decoder_fail = [x for x in regressions if x[2].get('status') == 'fail']

    print(f"\n  原因分布:")
    print(f"    簇消失 (uncovered):     {len(uncovered):,}")
    print(f"    无 consensus:           {len(no_cons):,}")
    print(f"    Decoder 解错:           {len(decoder_fail):,}")

    if decoder_fail:
        # 按 purity 分层
        pure_fail = [x for x in decoder_fail if x[2].get('purity', 0) >= 0.99]
        impure_fail = [x for x in decoder_fail if x[2].get('purity', 0) < 0.99]

        print(f"\n  Decoder 解错分层 ({len(decoder_fail)} 个):")
        print(f"    纯簇失败 (purity≥0.99): {len(pure_fail):,}  ← encoder 信息丢失")
        print(f"    杂簇失败 (purity<0.99): {len(impure_fail):,}  ← 聚类不纯导致")

        # 按簇大小分层
        sizes = [x[2].get('cluster_size', 0) for x in decoder_fail]
        small = sum(1 for s in sizes if s < 5)
        medium = sum(1 for s in sizes if 5 <= s < 20)
        large = sum(1 for s in sizes if s >= 20)
        print(f"\n  Decoder 解错按簇大小:")
        print(f"    小簇 (<5 reads):   {small:,}")
        print(f"    中簇 (5-19):       {medium:,}")
        print(f"    大簇 (≥20):        {large:,}")

        # ED 分布
        eds = [x[2].get('ed', 0) for x in decoder_fail if x[2].get('ed', 0) > 0]
        if eds:
            print(f"\n  Decoder 解错的 ED 分布:")
            print(f"    Mean:   {np.mean(eds):.2f}")
            print(f"    Median: {np.median(eds):.0f}")
            print(f"    ED=1:   {sum(1 for e in eds if e == 1):,} ({sum(1 for e in eds if e==1)/len(eds)*100:.1f}%)")
            print(f"    ED=2:   {sum(1 for e in eds if e == 2):,}")
            print(f"    ED 3-5: {sum(1 for e in eds if 3 <= e <= 5):,}")
            print(f"    ED>5:   {sum(1 for e in eds if e > 5):,}")

        # 长度差分析
        len_diffs = [x[2].get('len_diff', 0) for x in decoder_fail if 'len_diff' in x[2]]
        if len_diffs:
            print(f"\n  长度差 (consensus_len - ref_len):")
            print(f"    Mean:   {np.mean(len_diffs):+.2f}")
            print(f"    =0:     {sum(1 for d in len_diffs if d == 0):,} (长度正确)")
            print(f"    +1:     {sum(1 for d in len_diffs if d == 1):,} (多 1bp)")
            print(f"    -1:     {sum(1 for d in len_diffs if d == -1):,} (少 1bp)")
            print(f"    |d|>1:  {sum(1 for d in len_diffs if abs(d) > 1):,} (严重偏差)")

        # 打印前 20 个纯簇失败案例
        if pure_fail:
            print(f"\n  ── 前 20 个纯簇失败案例 (encoder 信息丢失的直接证据) ──")
            for i, (ref_id, r0, rn) in enumerate(sorted(pure_fail, key=lambda x: x[2].get('ed', 0))[:20]):
                ref_seq = gt_refs.get(ref_id, '')
                cons_seq_info = f"len={rn.get('cons_len', '?')}"
                print(f"    [{i+1}] GT#{ref_id}: ED={rn['ed']}, "
                      f"size={rn['cluster_size']}, purity={rn['purity']:.3f}, "
                      f"ref_len={len(ref_seq)}, {cons_seq_info}, "
                      f"Δlen={rn.get('len_diff', '?')}")

    # ── 改善分子分析 ──
    if improvements:
        print(f"\n  ── 改善分子 ({len(improvements)} 个) ──")
        imp_sizes = [x[2].get('cluster_size', 0) for x in improvements]
        imp_purs  = [x[2].get('purity', 0) for x in improvements]
        print(f"    平均簇大小: {np.mean(imp_sizes):.1f}")
        print(f"    平均 purity: {np.mean(imp_purs):.3f}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--gt_refs", required=True)
    parser.add_argument("--gt_tags", required=True)
    parser.add_argument("--max_regression_detail", type=int, default=20)
    args = parser.parse_args()

    exp_dir = args.experiment_dir

    print("=" * 70)
    print("  🔬 SSI-EC SR 退化诊断")
    print("=" * 70)
    print(f"  ED engine: {ED_ENGINE}")

    # ── 1. 加载数据 ──
    print(f"\n📂 加载数据...")
    reads, clover_labels = load_reads(exp_dir)
    print(f"   Reads: {len(reads):,}, Clover 簇: {len(set(clover_labels.tolist())):,}")

    gt_refs = load_gt_refs(args.gt_refs)
    print(f"   GT references: {len(gt_refs):,}")

    seq_to_tag, tag_to_reads = load_gt_tags(args.gt_tags)

    tag_to_ref = build_tag_to_ref(tag_to_reads, gt_refs)
    gt_ref_ids = match_reads_to_gt(reads, seq_to_tag, tag_to_ref)

    # ── 2. R0 基线 ──
    print(f"\n📊 R0 (Clover+MV) 分析...")
    ref_txt_path = os.path.join(exp_dir, "03_FedDNA_In", "ref.txt")
    if not os.path.exists(ref_txt_path):
        print(f"  ❌ 找不到 ref.txt: {ref_txt_path}")
        return
    r0_consensus = load_ref_txt(ref_txt_path)
    print(f"   R0 consensus: {len(r0_consensus):,} 个簇")

    r0_results = analyze_per_molecule(
        r0_consensus, clover_labels, gt_ref_ids, gt_refs, reads, "R0"
    )
    r0_success = sum(1 for r in r0_results.values() if r.get('ed', -1) == 0)
    r0_covered = sum(1 for r in r0_results.values() if r.get('ed', -1) >= 0)
    print(f"   R0: SR={r0_success}/{len(gt_refs)} ({r0_success/len(gt_refs)*100:.2f}%), "
          f"覆盖={r0_covered}")

    # ── 3. 各轮次分析 ──
    rounds = discover_rounds(exp_dir)
    print(f"\n🔍 发现 {len(rounds)} 个轮次")

    for round_num, fasta_path, label_path in rounds:
        print(f"\n{'─'*70}")
        print(f"  📦 Round {round_num}")
        print(f"     Consensus: {os.path.basename(fasta_path)}")
        print(f"     Labels:    {os.path.basename(label_path)}")

        labels = np.loadtxt(label_path, dtype=int)
        if len(labels) != len(reads):
            print(f"  ⚠️ labels 长度不匹配, 跳过")
            continue

        consensus = parse_consensus_fasta(fasta_path)
        print(f"     Consensus 簇数: {len(consensus):,}")

        rn_results = analyze_per_molecule(
            consensus, labels, gt_ref_ids, gt_refs, reads, f"Round {round_num}"
        )
        rn_success = sum(1 for r in rn_results.values() if r.get('ed', -1) == 0)
        rn_covered = sum(1 for r in rn_results.values() if r.get('ed', -1) >= 0)
        print(f"     SR={rn_success}/{len(gt_refs)} ({rn_success/len(gt_refs)*100:.2f}%), "
              f"覆盖={rn_covered}")

        # ── EER 分析 ──
        covered_eds = [r['ed'] for r in rn_results.values()
                       if r.get('ed', -1) >= 0]
        if covered_eds:
            covered_refs = [gt_refs[rid] for rid, r in rn_results.items()
                           if r.get('ed', -1) >= 0]
            eer_values = [ed / max(len(ref), 1)
                          for ed, ref in zip(covered_eds, covered_refs)]
            print(f"     EER={np.mean(eer_values):.6f}, "
                  f"ED_mean={np.mean(covered_eds):.2f}")

        # ── 核心: 与 R0 对比 ──
        print_regression_analysis(r0_results, rn_results, f"Round {round_num}", gt_refs)

    # ── 4. 总结 ──
    print(f"\n{'='*70}")
    print(f"  📋 诊断总结")
    print(f"{'='*70}")
    print(f"""
  如果"纯簇失败"占主导 → encoder 的位置级碱基信息被对比学习破坏
  如果"杂簇失败"占主导 → 聚类不纯, evidence fusion 被异源 reads 污染
  如果"簇消失"占主导   → Zone III 隔离过度, 小簇丢失
  如果 ED=1 且 Δlen=+1 占主导 → 截断 Bug 复发 (检查 save_consensus_fasta)

  v5 参考: EER 逐轮下降 (0.0046→0.0027), 说明 decoder 在改善
  v13 参考: EER 逐轮上升 (0.0046→0.0099), 说明 decoder 在退化
  如果"纯簇失败"在 v13 远多于 v5, 就是 v5→v13 代码改动引入的
""")


if __name__ == "__main__":
    main()