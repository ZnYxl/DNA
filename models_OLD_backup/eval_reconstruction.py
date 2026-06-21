#!/usr/bin/env python3
"""
eval_reconstruction_v2.py
=========================
Tag-based reconstruction evaluation for SSI-EC.

核心思路（Option A: Tag-based mapping）:
  1. 从 read.txt 加载所有 reads（保持 DataLoader 顺序）
  2. 用 seq1d_tags_reads.txt 的 "sequence → GT tag" 映射，找到每条 read 的 GT reference ID
  3. 对每个 cluster，用 labels 文件确定哪些 reads 在该 cluster，
     majority vote 确定该 cluster 对应的 GT reference
  4. 比较 consensus vs GT reference，计算 Success Rate / Edit Error Rate

支持功能:
  - 自动发现实验目录下所有轮次（Round 0 = Clover MV baseline）
  - 多轮对比表
  - 每个 GT reference 只评估一次（多 cluster 映到同一 ref 时取最优）

指标定义（与 FedDNA 论文完全对齐）:
  Success Rate   = #{consensus == reference} / #{all GT references}
  Edit Error Rate= mean( ED(consensus, reference) / len(reference) )

用法:
  python eval_reconstruction_v2.py \\
      --experiment_dir /path/to/seq_1d/ \\
      --gt_refs  /path/to/reads.fasta \\
      --gt_tags  /path/to/seq1d_tags_reads.txt

  # 也可以手动指定 read.txt 路径:
  python eval_reconstruction_v2.py \\
      --experiment_dir /path/to/seq_1d/ \\
      --gt_refs  /path/to/reads.fasta \\
      --gt_tags  /path/to/seq1d_tags_reads.txt \\
      --read_txt /path/to/read.txt

依赖:
  pip install edlib numpy tqdm
"""

import argparse
import os
import sys
import re
import glob
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# Edit Distance
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import edlib
    def levenshtein(a: str, b: str) -> int:
        return edlib.align(a, b, mode="NW", task="distance")['editDistance']
    _ED_ENGINE = "edlib"
except ImportError:
    try:
        import editdistance as _ed
        def levenshtein(a: str, b: str) -> int:
            return int(_ed.eval(a, b))
        _ED_ENGINE = "editdistance"
    except ImportError:
        def levenshtein(a: str, b: str) -> int:
            m, n = len(a), len(b)
            if m == 0: return n
            if n == 0: return m
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                prev, dp[0] = dp[0], i
                for j in range(1, n + 1):
                    tmp = dp[j]
                    dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
                    prev = tmp
            return dp[n]
        _ED_ENGINE = "pure-python (slow!)"


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════════
def load_reads_from_readtxt(path: str):
    """
    从 read.txt 加载所有 reads，返回 (reads_list, clover_labels_array)。
    clover_labels[i] 是 read i 的初始 Clover 簇编号 (0, 1, 2, ...)。
    """
    reads = []
    clover_labels = []
    cid = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("====="):
                cid += 1
            else:
                reads.append(line.upper())
                clover_labels.append(cid)
    n_clusters = cid  # 分隔符数 == 簇数
    print(f"   Reads:    {len(reads):,}")
    print(f"   Clusters: {n_clusters:,}")
    return reads, np.array(clover_labels, dtype=np.int64)


def load_gt_tags_file(gt_tags_file: str):
    """
    从 GT tags 文件 (tag\\tread) 加载:
      - seq_to_tag: {read_sequence → tag_id}  (用于匹配 reads)
      - tag_to_reads: {tag_id → [read_sequences]}  (用于 majority vote 建 tag→ref 映射)
    """
    seq_to_tags = defaultdict(list)
    tag_to_reads = defaultdict(list)
    total = 0
    with open(gt_tags_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                total += 1
                try:
                    tag = int(parts[0])
                    seq = parts[1].strip().upper()
                    seq_to_tags[seq].append(tag)
                    tag_to_reads[tag].append(seq)
                except ValueError:
                    pass

    seq_to_tag = {}
    for seq, tags in seq_to_tags.items():
        seq_to_tag[seq] = Counter(tags).most_common(1)[0][0]

    print(f"   GT tags 文件: {total:,} 行, {len(seq_to_tag):,} 唯一序列, "
          f"{len(tag_to_reads):,} 唯一 tags")
    return seq_to_tag, tag_to_reads


def _majority_vote_simple(reads: list, ref_len: int) -> str:
    """逐位多数投票，生成 pseudo-reference。"""
    vote = [Counter() for _ in range(ref_len)]
    for read in reads:
        for pos in range(min(len(read), ref_len)):
            b = read[pos].upper()
            if b in 'ACGT':
                vote[pos][b] += 1
    result = []
    for pos in range(ref_len):
        if vote[pos]:
            result.append(vote[pos].most_common(1)[0][0])
        else:
            result.append('A')
    return ''.join(result)


def build_tag_to_ref_mapping(tag_to_reads: dict, gt_refs: dict, ref_len: int = 196):
    """
    构建 tag_id → ref_id 映射。

    策略:
      1. 对每个 tag 的 reads 做 majority vote → pseudo-reference
      2. 建立 ref_sequence → ref_id 反向索引
      3. 精确匹配: pseudo-ref 完全等于某 ref → 直接映射
      4. 近似匹配: 精确失败时，找 ED 最小的 ref（兜底）

    大部分 tag 应该精确匹配成功（reads 足够多时 MV 精度极高）。
    """
    print(f"\n{'─' * 68}")
    print(f"  🔗  构建 Tag → Reference 映射")
    print(f"{'─' * 68}")

    # 反向索引: sequence → ref_id
    ref_seq_to_id = {}
    for ref_id, ref_seq in gt_refs.items():
        ref_seq_to_id[ref_seq] = ref_id

    # 准备 ref 序列列表 (用于 ED 兜底)
    ref_ids_list = sorted(gt_refs.keys())
    ref_seqs_list = [gt_refs[rid] for rid in ref_ids_list]

    tag_to_ref = {}
    exact_match = 0
    ed_match = 0
    failed = 0

    tags = sorted(tag_to_reads.keys())
    for tag_id in tqdm(tags, desc="  Tag→Ref mapping", leave=False):
        reads = tag_to_reads[tag_id]

        # Majority vote
        pseudo_ref = _majority_vote_simple(reads, ref_len)

        # 策略1: 精确匹配
        if pseudo_ref in ref_seq_to_id:
            tag_to_ref[tag_id] = ref_seq_to_id[pseudo_ref]
            exact_match += 1
            continue

        # 策略2: 最小 ED 匹配
        best_ed = float('inf')
        best_ref_id = None
        for rid, rseq in zip(ref_ids_list, ref_seqs_list):
            ed = levenshtein(pseudo_ref, rseq)
            if ed < best_ed:
                best_ed = ed
                best_ref_id = rid
                if ed == 0:
                    break

        if best_ref_id is not None and best_ed <= ref_len * 0.3:
            tag_to_ref[tag_id] = best_ref_id
            ed_match += 1
        else:
            failed += 1

    print(f"   唯一 tags:       {len(tags):,}")
    print(f"   精确匹配:        {exact_match:,}  ({exact_match/len(tags)*100:.1f}%)")
    print(f"   ED 近似匹配:     {ed_match:,}")
    print(f"   映射失败:        {failed:,}")
    print(f"   总映射成功:      {len(tag_to_ref):,}")

    return tag_to_ref


def match_reads_to_gt(reads: list, seq_to_tag: dict, tag_to_ref: dict) -> np.ndarray:
    """
    为每条 read 查找对应的 GT reference ID。

    链路: read_sequence → tag_id (via seq_to_tag) → ref_id (via tag_to_ref)
    """
    gt_ref_ids = np.full(len(reads), -1, dtype=np.int64)
    matched = 0
    tag_miss = 0
    ref_miss = 0
    for i, read in enumerate(reads):
        tag = seq_to_tag.get(read)
        if tag is None:
            tag_miss += 1
            continue
        ref_id = tag_to_ref.get(tag)
        if ref_id is None:
            ref_miss += 1
            continue
        gt_ref_ids[i] = ref_id
        matched += 1
    print(f"   Read → Ref 匹配: {matched:,} / {len(reads):,} ({matched/len(reads)*100:.1f}%)")
    if tag_miss > 0:
        print(f"   tag 查找失败:    {tag_miss:,}")
    if ref_miss > 0:
        print(f"   tag→ref 失败:   {ref_miss:,}")
    return gt_ref_ids


def load_gt_refs_fasta(path: str) -> dict:
    # [GTREFS-BARESEQ] 自动兼容两种格式:
    #   1. 标准 FASTA(含 '>' 头): 按头解析 ID
    #   2. 裸序列(无 '>' 头, 每行一条序列): 行号(从1)当 ID
    # eval 用序列做 key 对齐, ID 仅作标识, 故裸序列按行号编号不影响 SR/EER。
    """从 FASTA 或裸序列文件加载 GT references: {int_id: sequence}。"""
    # 先探测是否含 '>' 头
    has_header = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('>'):
                has_header = True
            break  # 只看第一条非空行

    refs = {}
    if has_header:
        # ---- 标准 FASTA 解析(原逻辑) ----
        cur_id = None
        cur_seq = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if cur_id is not None:
                        refs[cur_id] = ''.join(cur_seq).upper()
                    try:
                        cur_id = int(line[1:].split()[0])
                    except ValueError:
                        cur_id = line[1:].strip()
                    cur_seq = []
                elif line:
                    cur_seq.append(line)
        if cur_id is not None:
            refs[cur_id] = ''.join(cur_seq).upper()
        print(f"   GT references: {len(refs):,}  (FASTA 格式)")
    else:
        # ---- 裸序列解析: 每行一条, 行号(从1)当 ID ----
        with open(path) as f:
            idx = 1
            for line in f:
                seq = line.strip().upper()
                if seq:
                    refs[idx] = seq
                    idx += 1
        print(f"   GT references: {len(refs):,}  (裸序列格式, 行号当 ID)")
    return refs


def load_ref_txt(path: str) -> dict:
    """从 ref.txt（每行一条序列）加载 Clover MV pseudo-ref: {cluster_id: sequence}。"""
    refs = {}
    with open(path) as f:
        for i, line in enumerate(f):
            seq = line.strip().upper()
            if seq:
                refs[i] = seq
    return refs


def parse_consensus_fasta(path: str) -> dict:
    """解析 consensus FASTA: {cluster_id_int: sequence}。"""
    seqs = {}
    cur_id = None
    cur_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cur_id is not None:
                    seqs[cur_id] = ''.join(cur_seq).upper()
                header = line[1:].strip()
                m = re.match(r'cluster_(\d+)', header)
                if m:
                    cur_id = int(m.group(1))
                else:
                    try:
                        cur_id = int(header)
                    except ValueError:
                        cur_id = header
                cur_seq = []
            elif line:
                cur_seq.append(line)
    if cur_id is not None:
        seqs[cur_id] = ''.join(cur_seq).upper()
    return seqs


# ═══════════════════════════════════════════════════════════════════════════════
# 核心: Cluster → GT Reference 映射
# ═══════════════════════════════════════════════════════════════════════════════
def build_cluster_to_gt(labels: np.ndarray, gt_tags: np.ndarray):
    """
    对每个 cluster，majority vote 确定对应的 GT reference。

    Returns:
        cluster_to_gt:  {cluster_id: gt_ref_id}
        cluster_purity: {cluster_id: float}  (majority count / total)
    """
    cluster_tags = defaultdict(list)
    for i in range(len(labels)):
        if labels[i] >= 0 and gt_tags[i] >= 0:
            cluster_tags[int(labels[i])].append(int(gt_tags[i]))

    cluster_to_gt = {}
    cluster_purity = {}
    for cid, tags in cluster_tags.items():
        counter = Counter(tags)
        majority_tag, majority_count = counter.most_common(1)[0]
        cluster_to_gt[cid] = majority_tag
        cluster_purity[cid] = majority_count / len(tags)

    return cluster_to_gt, cluster_purity


# ═══════════════════════════════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_reconstruction(
    consensus: dict,        # {cluster_id: consensus_seq}
    cluster_to_gt: dict,    # {cluster_id: gt_ref_id}
    gt_refs: dict,          # {gt_ref_id: gt_sequence}
    name: str = "Method",
) -> dict:
    """
    Per-GT-reference 评估。

    对每个 GT reference:
      - 找到所有映射到它的 cluster
      - 在这些 cluster 的 consensus 中选 ED 最小的
      - 若无 cluster 映射，ED = len(reference)（完全丢失）

    这保证每个 GT reference 恰好被评估一次，与 FedDNA 的评估语义对齐。
    """
    n_gt = len(gt_refs)

    # 反转映射: gt_ref_id → [cluster_ids that have consensus]
    gt_to_clusters = defaultdict(list)
    for cid, gt_id in cluster_to_gt.items():
        if cid in consensus:
            gt_to_clusters[gt_id].append(cid)

    ed_list = []
    eer_list = []
    success = 0
    covered = 0

    for gt_id in tqdm(sorted(gt_refs.keys()), desc=f"  [{name}] ED", leave=False):
        gt_seq = gt_refs[gt_id]
        gt_len = max(len(gt_seq), 1)
        clusters = gt_to_clusters.get(gt_id, [])

        if not clusters:
            # GT reference 未被任何 cluster 覆盖
            ed_list.append(gt_len)
            eer_list.append(1.0)
            continue

        covered += 1

        # 多个 cluster 映到同一 GT ref → 取最优
        best_ed = float('inf')
        for cid in clusters:
            ed = levenshtein(consensus[cid], gt_seq)
            best_ed = min(best_ed, ed)

        ed_list.append(best_ed)
        eer_list.append(best_ed / gt_len)
        if best_ed == 0:
            success += 1

    ed_arr = np.array(ed_list, dtype=np.float32)
    eer_arr = np.array(eer_list, dtype=np.float32)

    # 只对 covered 的 ref 统计 EER（与 FedDNA 一致：它们的 cluster 预先对齐，100% covered）
    covered_mask = np.array([1 if gt_to_clusters.get(gt_id) else 0
                              for gt_id in sorted(gt_refs.keys())], dtype=bool)
    eer_covered = eer_arr[covered_mask]

    results = {
        'name':          name,
        'n_gt':          n_gt,
        'n_clusters':    len(consensus),
        'n_covered':     covered,
        'success':       success,
        'success_rate':  success / max(n_gt, 1),
        'recall':        covered / max(n_gt, 1),
        'eer_mean':      float(eer_covered.mean()) if len(eer_covered) > 0 else 0.0,
        'ed_mean':       float(ed_arr[covered_mask].mean()) if covered > 0 else 0.0,
        'ed_median':     float(np.median(ed_arr[covered_mask])) if covered > 0 else 0.0,
        'ed_p90':        float(np.percentile(ed_arr[covered_mask], 90)) if covered > 0 else 0.0,
        'ed_p95':        float(np.percentile(ed_arr[covered_mask], 95)) if covered > 0 else 0.0,
        'ed_max':        float(ed_arr[covered_mask].max()) if covered > 0 else 0.0,
    }

    _print_result(results)
    return results


def _print_result(r: dict):
    sep = "═" * 68
    print(f"\n{sep}")
    print(f"  📊  {r['name']}")
    print(f"{sep}")
    print(f"  GT 分子总数      : {r['n_gt']:>10,}")
    print(f"  Consensus 簇数   : {r['n_clusters']:>10,}")
    print(f"  覆盖 GT 分子数   : {r['n_covered']:>10,}")
    print()
    print(f"  ✅ Success Rate   : {r['success_rate']:.6f}  "
          f"({r['success']:,}/{r['n_gt']:,}  完全匹配)")
    print(f"  📡 Recall         : {r['recall']:.6f}  "
          f"({r['n_covered']:,}/{r['n_gt']:,}  被覆盖)")
    print()
    print(f"  Edit Error Rate   : {r['eer_mean']:.6f}  "
          f"(covered 均值, 与 FedDNA 对齐)")
    print(f"  Edit Distance (covered only):")
    print(f"    Mean   : {r['ed_mean']:>8.2f}")
    print(f"    Median : {r['ed_median']:>8.2f}")
    print(f"    P90    : {r['ed_p90']:>8.2f}")
    print(f"    P95    : {r['ed_p95']:>8.2f}")
    print(f"    Max    : {r['ed_max']:>8.2f}")
    print(sep)


def _print_comparison_table(all_results: list):
    """打印多方法横向对比表。"""
    if not all_results:
        return

    header = (f"{'Method':<18} {'#Clusters':>9} {'Recall':>8} "
              f"{'SR':>10} {'EER':>10} {'ED_mean':>8} {'ED_med':>8} {'ED_P90':>8}")
    sep_line = "─" * len(header)
    print(f"\n{'═' * len(header)}")
    print("  📋  横向对比表 (Reconstruction Quality)")
    print(f"{'═' * len(header)}")
    print(f"  {header}")
    print(f"  {sep_line}")
    for r in all_results:
        sr_str = f"{r['success_rate']:.4f} ({r['success']:>5})"
        print(f"  {r['name']:<18} "
              f"{r['n_clusters']:>9,} "
              f"{r['recall']:>8.4f} "
              f"{sr_str:>10} "
              f"{r['eer_mean']:>10.6f} "
              f"{r['ed_mean']:>8.2f} "
              f"{r['ed_median']:>8.2f} "
              f"{r['ed_p90']:>8.2f}")
    print(f"{'═' * len(header)}")

    # FedDNA 参考线
    print(f"\n  📌 FedDNA 参考 (Seq_1D): SR=97.26%, EER=0.15%")
    print(f"  📌 Iter. Recon. 参考:    SR=97.45%")


# ═══════════════════════════════════════════════════════════════════════════════
# 自动发现轮次
# ═══════════════════════════════════════════════════════════════════════════════
def find_read_txt(experiment_dir: str) -> str:
    """在多个可能的位置查找 read.txt。"""
    candidates = [
        os.path.join(experiment_dir, "03_FedDNA_In", "read.txt"),
        os.path.join(experiment_dir, "04_FedDNA_In", "read.txt"),
        os.path.join(experiment_dir, "read.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # 尝试往上一级的 clover_out 找
    parent = os.path.dirname(experiment_dir.rstrip('/'))
    extra = [
        os.path.join(parent, "clover_out", "04_FedDNA_In", "read.txt"),
        os.path.join(parent, "Sequencing_data_first_dimension", "clover_out", "04_FedDNA_In", "read.txt"),
    ]
    for p in extra:
        if os.path.exists(p):
            return p

    return None


def find_ref_txt(experiment_dir: str) -> str:
    """查找 ref.txt (Clover MV pseudo-references)。"""
    candidates = [
        os.path.join(experiment_dir, "03_FedDNA_In", "ref.txt"),
        os.path.join(experiment_dir, "04_FedDNA_In", "ref.txt"),
        os.path.join(experiment_dir, "ref.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def discover_rounds(experiment_dir: str) -> list:
    """
    自动发现所有轮次的 (labels_path, consensus_path, round_name)。

    匹配策略: consensus 和 labels 文件名共享时间戳。
    例如: consensus_162641.fasta ↔ refined_labels_162641.txt
    """
    rounds = []

    # 扫描 consensus 文件
    consensus_pattern = os.path.join(
        experiment_dir, "results", "iter_*_step2", "consensus", "consensus_*.fasta"
    )
    consensus_files = sorted(glob.glob(consensus_pattern))

    labels_dir = os.path.join(experiment_dir, "04_Iterative_Labels")

    for cons_path in consensus_files:
        # 从路径提取轮次号: iter_3_step2 → 3
        m_iter = re.search(r'iter_(\d+)_step2', cons_path)
        if not m_iter:
            continue
        round_idx = int(m_iter.group(1))

        # 从文件名提取时间戳: consensus_162641.fasta → 162641
        m_ts = re.search(r'consensus_(\d+)\.fasta', os.path.basename(cons_path))
        if not m_ts:
            continue
        timestamp = m_ts.group(1)

        # 找对应的 labels 文件
        labels_path = os.path.join(labels_dir, f"refined_labels_{timestamp}.txt")
        if not os.path.exists(labels_path):
            print(f"  ⚠️  Round {round_idx}: 找不到 labels {labels_path}, 跳过")
            continue

        rounds.append((labels_path, cons_path, f"Round {round_idx}"))

    return rounds


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="SSI-EC 重建质量评估 (Tag-based mapping, FedDNA 指标)"
    )
    parser.add_argument('--experiment_dir', required=True,
                        help='SSI-EC 实验目录')
    parser.add_argument('--gt_refs', required=True,
                        help='GT reference FASTA (e.g. reads.fasta)')
    parser.add_argument('--gt_tags', required=True,
                        help='GT tags 文件 (tag\\tread)')
    parser.add_argument('--read_txt', default=None,
                        help='read.txt 路径 (自动发现, 可手动指定)')
    parser.add_argument('--out', default=None,
                        help='输出 TSV 文件 (可选)')
    parser.add_argument('--skip_round0', action='store_true',
                        help='跳过 Round 0 (Clover MV baseline)')
    parser.add_argument('--consensus_override', nargs='+', default=None,
                        help='额外评估的 baseline consensus fasta(可多个), 复用 Clover labels, 同口径')
    args = parser.parse_args()

    exp_dir = args.experiment_dir

    print(f"\n{'═' * 68}")
    print(f"  🧬  SSI-EC Reconstruction Evaluation (Tag-based)")
    print(f"{'═' * 68}")
    print(f"  ED engine: {_ED_ENGINE}")
    print(f"  Exp dir:   {exp_dir}")

    # ── 1. 加载 read.txt ─────────────────────────────────────────────────────
    print(f"\n{'─' * 68}")
    print(f"  📂  加载 reads")
    print(f"{'─' * 68}")

    read_txt = args.read_txt or find_read_txt(exp_dir)
    if read_txt is None:
        print("  ❌ 找不到 read.txt! 请用 --read_txt 手动指定。")
        sys.exit(1)
    print(f"  路径: {read_txt}")
    reads, clover_labels = load_reads_from_readtxt(read_txt)

    # ── 2. 加载 GT tags 文件 ─────────────────────────────────────────────────
    print(f"\n{'─' * 68}")
    print(f"  📂  加载 GT tags")
    print(f"{'─' * 68}")
    print(f"  路径: {args.gt_tags}")
    seq_to_tag, tag_to_reads = load_gt_tags_file(args.gt_tags)

    # ── 3. 加载 GT references ────────────────────────────────────────────────
    print(f"\n{'─' * 68}")
    print(f"  📂  加载 GT references")
    print(f"{'─' * 68}")
    print(f"  路径: {args.gt_refs}")
    gt_refs = load_gt_refs_fasta(args.gt_refs)

    # ── 4. 构建 Tag → Reference 映射 ────────────────────────────────────────
    #    Tag ID (13411, 111259, ...) ≠ reads.fasta header (1, 2, 3, ...)
    #    需要通过 majority vote + 序列匹配建立对应关系
    ref_len = int(np.median([len(s) for s in gt_refs.values()]))
    print(f"   参考序列中位长度: {ref_len}bp")
    tag_to_ref = build_tag_to_ref_mapping(tag_to_reads, gt_refs, ref_len=ref_len)

    # ── 5. 匹配 reads → GT reference ID ─────────────────────────────────────
    print(f"\n{'─' * 68}")
    print(f"  📂  匹配 Reads → GT Reference")
    print(f"{'─' * 68}")
    gt_ref_ids = match_reads_to_gt(reads, seq_to_tag, tag_to_ref)

    # 覆盖率检查
    unique_refs_in_reads = set(gt_ref_ids[gt_ref_ids >= 0].tolist())
    print(f"   Reads 中覆盖的 GT refs: {len(unique_refs_in_reads):,} / {len(gt_refs):,}")

    # ── 6. 发现各轮次 ────────────────────────────────────────────────────────
    print(f"\n{'─' * 68}")
    print(f"  🔍  发现轮次")
    print(f"{'─' * 68}")

    all_results = []

    # Round 0: Clover MV baseline
    if not args.skip_round0:
        ref_txt = find_ref_txt(exp_dir)
        if ref_txt:
            print(f"\n  📦 Round 0 (Clover MV baseline)")
            print(f"     ref.txt: {ref_txt}")
            clover_consensus = load_ref_txt(ref_txt)
            print(f"     Consensus: {len(clover_consensus):,} 条")

            c2g_r0, pur_r0 = build_cluster_to_gt(clover_labels, gt_ref_ids)
            avg_pur = np.mean(list(pur_r0.values())) if pur_r0 else 0
            print(f"     簇 → GT 映射: {len(c2g_r0):,} 簇, 平均 purity={avg_pur:.4f}")

            r0 = evaluate_reconstruction(clover_consensus, c2g_r0, gt_refs, name="R0 Clover+MV")
            all_results.append(r0)
        else:
            print(f"  ⚠️  找不到 ref.txt, 跳过 Round 0")

    # Baseline consensus override (Iter/Div/BMA), 复用 Clover labels
    if args.consensus_override:
        c2g_bl, _ = build_cluster_to_gt(clover_labels, gt_ref_ids)
        for bl_path in args.consensus_override:
            import os as _os
            tag = _os.path.basename(bl_path).replace('consensus_','').replace('.fasta','')
            print(f"\n  📦 Baseline: {tag}")
            print(f"     Consensus: {bl_path}")
            bl_cons = parse_consensus_fasta(bl_path)
            print(f"     Consensus 簇数: {len(bl_cons):,}")
            r_bl = evaluate_reconstruction(bl_cons, c2g_bl, gt_refs, name=f'BL:{tag}')
            all_results.append(r_bl)

    # Round 1, 2, 3, ...
    discovered = discover_rounds(exp_dir)
    print(f"\n  发现 {len(discovered)} 个 SSI-EC 轮次:")
    for labels_path, cons_path, name in discovered:
        print(f"    {name}: {os.path.basename(cons_path)} + {os.path.basename(labels_path)}")

    for labels_path, cons_path, round_name in discovered:
        print(f"\n  📦 {round_name}")
        print(f"     Consensus: {cons_path}")
        print(f"     Labels:    {labels_path}")

        labels = np.loadtxt(labels_path, dtype=int)
        if len(labels) != len(reads):
            print(f"     ⚠️  labels 长度 {len(labels)} ≠ reads 长度 {len(reads)}, 跳过")
            continue

        consensus = parse_consensus_fasta(cons_path)
        print(f"     Consensus 簇数: {len(consensus):,}")

        c2g, pur = build_cluster_to_gt(labels, gt_ref_ids)
        avg_pur = np.mean(list(pur.values())) if pur else 0
        print(f"     簇 → GT 映射: {len(c2g):,} 簇, 平均 purity={avg_pur:.4f}")

        r = evaluate_reconstruction(consensus, c2g, gt_refs, name=round_name)
        all_results.append(r)

    # ── 5. 横向对比 ──────────────────────────────────────────────────────────
    _print_comparison_table(all_results)

    # ── 6. 可选: 保存 TSV ────────────────────────────────────────────────────
    if args.out and all_results:
        keys = ['name', 'n_gt', 'n_clusters', 'n_covered', 'success',
                'success_rate', 'recall', 'eer_mean', 'ed_mean',
                'ed_median', 'ed_p90', 'ed_p95', 'ed_max']
        with open(args.out, 'w') as f:
            f.write('\t'.join(keys) + '\n')
            for r in all_results:
                f.write('\t'.join(str(r.get(k, '')) for k in keys) + '\n')
        print(f"\n  💾  结果已保存: {args.out}")


if __name__ == '__main__':
    main()