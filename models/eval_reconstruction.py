#!/usr/bin/env python3
"""
eval_reconstruction.py
======================
对 SSI-EC 各轮输出的 consensus FASTA 进行重建质量评估。

评估指标（与 FedDNA 论文完全对齐）：
  - Success Rate   : consensus 与 GT reference 完全匹配的比例
  - Edit Error Rate: 归一化 Levenshtein 距离 = ED / len(reference)
  - Recall         : 被至少一条 consensus 覆盖的 GT 分子比例
  - 分布统计       : ED 的 P50 / P90 / P95 / max

用法：
  python eval_reconstruction.py \
      --refs  /path/to/exp1_refs.fasta \
      --fasta /path/to/consensus_R3.fasta \
      --name  "SSI-EC R3"

  # 或者一次性对比多个轮次（自动对齐）：
  python eval_reconstruction.py \
      --refs  /path/to/exp1_refs.fasta \
      --fasta consensus_R1.fasta consensus_R2.fasta consensus_R3.fasta \
      --name  "R1" "R2" "R3"

  # 同时评估 Clover 直接 majority-vote 的结果（作为 baseline）：
  python eval_reconstruction.py \
      --refs  /path/to/exp1_refs.fasta \
      --fasta clover_mv.fasta ssi_ec_r3.fasta \
      --name  "Clover+MV" "SSI-EC R3"

依赖：
  pip install editdistance tqdm   （均为轻量级，无需 GPU）
"""

import argparse
import sys
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# ── 尝试导入 editdistance，否则用纯 Python 实现 ──────────────────────────────
try:
    import editdistance as _ed
    def levenshtein(a: str, b: str) -> int:
        return int(_ed.eval(a, b))
except ImportError:
    # 纯 Python fallback（慢，但功能正确）
    def levenshtein(a: str, b: str) -> int:
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                tmp = dp[j]
                dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = tmp
        return dp[n]


# ── FASTA 解析 ────────────────────────────────────────────────────────────────
def parse_fasta(path: str) -> dict:
    """
    返回 {header_str: sequence_str}。
    header 是 '>' 之后的完整行（去除首尾空白）。
    """
    seqs = {}
    cur_header = None
    cur_seq = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith('>'):
                if cur_header is not None:
                    seqs[cur_header] = ''.join(cur_seq).upper()
                cur_header = line[1:].strip()
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_header is not None:
        seqs[cur_header] = ''.join(cur_seq).upper()
    return seqs


# ── 核心评估函数 ──────────────────────────────────────────────────────────────
def evaluate(
    gt_seqs: dict,          # {header: seq}，来自 exp1_refs.fasta
    pred_seqs: dict,        # {header: seq}，来自 consensus FASTA
    name: str = "Method",
    verbose: bool = True,
) -> dict:
    """
    将 pred_seqs 与 gt_seqs 做最优匹配，计算重建质量指标。

    匹配策略：
      1. 先尝试 header 完全匹配（pred header == gt header）
      2. 若 pred header 形如 'cluster_123'，提取 '123' 再匹配 gt header
      3. 剩余未匹配的 pred，用序列完全相等做兜底（应对 Clover 直接 MV）

    这样设计是为了兼容：
      - SSI-EC 输出的 >cluster_0, >cluster_1, ...
      - 直接编号的 >0, >1, ...（与 exp1_refs.fasta 格式一致）
    """

    # ── 建立 GT 查找表 ────────────────────────────────────────────────────────
    gt_by_header = {h: s for h, s in gt_seqs.items()}    # header → seq
    gt_by_seq    = {s: h for h, s in gt_seqs.items()}    # seq → header（用于兜底）
    n_gt = len(gt_seqs)

    # ── 匹配 pred → gt ────────────────────────────────────────────────────────
    matched_pairs = []      # [(pred_seq, gt_seq, gt_header)]
    covered_gt = set()

    for pred_header, pred_seq in pred_seqs.items():
        gt_seq = None

        # 策略1：header 完全匹配
        if pred_header in gt_by_header:
            gt_seq = gt_by_header[pred_header]
            covered_gt.add(pred_header)

        # 策略2：'cluster_N' → 'N'
        elif pred_header.startswith('cluster_'):
            idx_str = pred_header[len('cluster_'):]
            if idx_str in gt_by_header:
                gt_seq = gt_by_header[idx_str]
                covered_gt.add(idx_str)

        # 策略3：序列完全匹配（兜底）
        elif pred_seq in gt_by_seq:
            gt_header = gt_by_seq[pred_seq]
            gt_seq = gt_by_header[gt_header]
            covered_gt.add(gt_header)

        if gt_seq is not None:
            matched_pairs.append((pred_seq, gt_seq))

    n_matched = len(matched_pairs)

    if n_matched == 0:
        print(f"  ⚠️  [{name}] 没有任何 pred 能匹配到 GT，请检查 header 格式！")
        print(f"       pred headers 示例: {list(pred_seqs.keys())[:3]}")
        print(f"       gt   headers 示例: {list(gt_seqs.keys())[:3]}")
        return {}

    # ── 逐对计算 ED ───────────────────────────────────────────────────────────
    ed_list = []
    eer_list = []
    success = 0

    for pred_seq, gt_seq in tqdm(matched_pairs, desc=f"  [{name}] 计算 ED", leave=False):
        ed = levenshtein(pred_seq, gt_seq)
        eer = ed / max(len(gt_seq), 1)
        ed_list.append(ed)
        eer_list.append(eer)
        if ed == 0:
            success += 1

    ed_arr  = np.array(ed_list,  dtype=np.float32)
    eer_arr = np.array(eer_list, dtype=np.float32)

    success_rate   = success / n_gt          # 分母是 GT 总数（严格标准）
    recall         = len(covered_gt) / n_gt  # 被覆盖的 GT 分子比例

    results = {
        'name':          name,
        'n_gt':          n_gt,
        'n_pred':        len(pred_seqs),
        'n_matched':     n_matched,
        'success':       success,
        'success_rate':  success_rate,
        'recall':        recall,
        'ed_mean':       float(ed_arr.mean()),
        'ed_median':     float(np.median(ed_arr)),
        'ed_p90':        float(np.percentile(ed_arr, 90)),
        'ed_p95':        float(np.percentile(ed_arr, 95)),
        'ed_max':        float(ed_arr.max()),
        'eer_mean':      float(eer_arr.mean()),
        'eer_median':    float(np.median(eer_arr)),
    }

    if verbose:
        _print_result(results)

    return results


def _print_result(r: dict):
    sep = "═" * 64
    print(f"\n{sep}")
    print(f"  📊  {r['name']}")
    print(sep)
    print(f"  GT 分子总数      : {r['n_gt']:>10,}")
    print(f"  Pred consensus 数: {r['n_pred']:>10,}")
    print(f"  成功匹配 GT 数   : {r['n_matched']:>10,}")
    print()
    print(f"  ✅ Success Rate   : {r['success_rate']:.6f}  "
          f"({r['success']}/{r['n_gt']}  完全匹配)")
    print(f"  📡 Recall         : {r['recall']:.6f}  "
          f"(被覆盖的 GT 分子)")
    print()
    print(f"  Edit Error Rate   : {r['eer_mean']:.6f}  (均值)")
    print(f"  Edit Distance     :")
    print(f"    Mean   : {r['ed_mean']:>8.2f}")
    print(f"    Median : {r['ed_median']:>8.2f}")
    print(f"    P90    : {r['ed_p90']:>8.2f}")
    print(f"    P95    : {r['ed_p95']:>8.2f}")
    print(f"    Max    : {r['ed_max']:>8.2f}")


def _print_comparison_table(all_results: list):
    """打印多方法横向对比表。"""
    if len(all_results) < 2:
        return

    header = f"{'Method':<20} {'SR':>8} {'Recall':>8} {'EER':>8} {'ED_mean':>8} {'ED_med':>8} {'ED_P90':>8}"
    sep    = "─" * len(header)
    print(f"\n{'═'*len(header)}")
    print("  📋  横向对比表")
    print(f"{'═'*len(header)}")
    print("  " + header)
    print("  " + sep)
    for r in all_results:
        print(f"  {r['name']:<20} "
              f"{r['success_rate']:>8.4f} "
              f"{r['recall']:>8.4f} "
              f"{r['eer_mean']:>8.4f} "
              f"{r['ed_mean']:>8.2f} "
              f"{r['ed_median']:>8.2f} "
              f"{r['ed_p90']:>8.2f}")
    print(f"{'═'*len(header)}\n")


# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SSI-EC 重建质量评估（Success Rate / Edit Distance）"
    )
    parser.add_argument(
        '--refs', required=True,
        help='GT reference FASTA，例如 exp1_refs.fasta'
    )
    parser.add_argument(
        '--fasta', nargs='+', required=True,
        help='一个或多个 consensus FASTA 文件'
    )
    parser.add_argument(
        '--name', nargs='+', default=None,
        help='与 --fasta 一一对应的方法名（可选，默认用文件名）'
    )
    parser.add_argument(
        '--out', default=None,
        help='将结果保存为 TSV 文件（可选）'
    )
    args = parser.parse_args()

    # 校验参数
    if args.name and len(args.name) != len(args.fasta):
        print("错误: --name 的数量必须与 --fasta 一致", file=sys.stderr)
        sys.exit(1)

    names = args.name or [os.path.basename(f) for f in args.fasta]

    # 加载 GT
    print(f"\n{'═'*64}")
    print(f"  📂  加载 GT references: {args.refs}")
    gt_seqs = parse_fasta(args.refs)
    print(f"  GT 分子总数: {len(gt_seqs):,}")
    print(f"  示例 header: {list(gt_seqs.keys())[:3]}")

    # 逐个 FASTA 评估
    all_results = []
    for fasta_path, method_name in zip(args.fasta, names):
        print(f"\n{'─'*64}")
        print(f"  📄  加载 {method_name}: {fasta_path}")
        if not os.path.exists(fasta_path):
            print(f"  ⚠️  文件不存在，跳过: {fasta_path}")
            continue
        pred_seqs = parse_fasta(fasta_path)
        print(f"  Pred consensus 数: {len(pred_seqs):,}")
        r = evaluate(gt_seqs, pred_seqs, name=method_name)
        if r:
            all_results.append(r)

    # 横向对比表
    _print_comparison_table(all_results)

    # 可选：保存 TSV
    if args.out and all_results:
        keys = ['name', 'n_gt', 'n_pred', 'n_matched', 'success',
                'success_rate', 'recall', 'eer_mean', 'ed_mean',
                'ed_median', 'ed_p90', 'ed_p95', 'ed_max']
        with open(args.out, 'w') as f:
            f.write('\t'.join(keys) + '\n')
            for r in all_results:
                f.write('\t'.join(str(r.get(k, '')) for k in keys) + '\n')
        print(f"  💾  结果已保存: {args.out}")


if __name__ == '__main__':
    main()