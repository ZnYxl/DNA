"""
eval_reconstruction.py - 每轮 Consensus 重建质量评估 (exp_1)

将每轮 consensus_sequences.fasta 与 GT references 对比:
  - Exact Match Rate
  - Base-level Accuracy
  - Edit Distance 分布

使用 CloverDataLoader 建立 cluster→GT ref 映射, 保证索引一致。

用法:
  cd /mnt/st_data/liangxinyi/code
  python eval_reconstruction.py
"""
import re
import os
import sys
import numpy as np
from collections import Counter, defaultdict

# 确保能 import models
CODE_DIR = "/mnt/st_data/liangxinyi/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.step1_data import CloverDataLoader

# ================= 路径配置 =================
EXP_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_Real_last"
GT_REFS_FILE = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1/exp1_refs.fasta"
GT_TAGS_FILE = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1/exp1_tags_reads.txt"

# 自动扫描 (只保留存在 consensus fasta 的)
CANDIDATE_ROUNDS = [
    ("R1",       os.path.join(EXP_DIR, "results/iter_1_step2")),
    ("R1_fixed", os.path.join(EXP_DIR, "results/iter_1_step2_fixed")),
    ("R2",       os.path.join(EXP_DIR, "results/iter_2_step2")),
    ("R3",       os.path.join(EXP_DIR, "results/iter_3_step2")),
]
# ============================================


def load_fasta(path):
    """返回 {header: sequence}"""
    seqs = {}
    name = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                name = line[1:]
                seqs[name] = ''
            elif name:
                seqs[name] += line
    return seqs


def edit_distance(s1, s2):
    """Levenshtein"""
    n, m = len(s1), len(s2)
    if n > m:
        s1, s2 = s2, s1
        n, m = m, n
    prev = list(range(n + 1))
    for j in range(1, m + 1):
        curr = [j] + [0] * n
        for i in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            curr[i] = min(curr[i-1] + 1, prev[i] + 1, prev[i-1] + cost)
        prev = curr
    return prev[n]


def main():
    # =================================================================
    # 1. 用 CloverDataLoader 建立 cluster→GT ref 映射
    # =================================================================
    print("[1] 加载数据 (CloverDataLoader)...")
    data_loader = CloverDataLoader(EXP_DIR)
    TOTAL = len(data_loader.reads)
    clover_labels = np.array(data_loader.clover_labels, dtype=int)
    print(f"    data_loader: {TOTAL:,} reads, {len(set(clover_labels[clover_labels>=0])):,} 簇")

    # 加载 GT tags
    print("[2] 加载 GT tags...")
    data_loader.load_gt_tags(GT_TAGS_FILE)
    gt_labels = np.array(data_loader.gt_labels, dtype=int)
    print(f"    GT 有效: {(gt_labels >= 0).sum():,}")

    # 加载 GT references (tag_str → sequence)
    print("[3] 加载 GT references...")
    gt_refs_raw = load_fasta(GT_REFS_FILE)
    # 建 tag_id(int) → sequence 映射
    # gt_tags_file 的 tag 是字符串, refs.fasta 的 header 也是字符串
    # 用 data_loader.gt_labels 的整数 ID → 需要知道 tag_str→tag_int 映射
    # 最简单: 直接从 gt_tags_file 建 tag_str → tag_int, 然后 gt_refs[tag_int] = seq
    tag_str_to_int = {}
    with open(GT_TAGS_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                tag_str = parts[0]
                if tag_str not in tag_str_to_int:
                    tag_str_to_int[tag_str] = int(tag_str) if tag_str.isdigit() else len(tag_str_to_int)

    gt_refs = {}  # tag_int → sequence
    for header, seq in gt_refs_raw.items():
        # header 可能是 tag_str 本身
        if header in tag_str_to_int:
            gt_refs[tag_str_to_int[header]] = seq
        # 也试纯数字
        if header.isdigit():
            gt_refs[int(header)] = seq
    print(f"    GT refs: {len(gt_refs):,} 条")
    if gt_refs:
        first_k = next(iter(gt_refs))
        print(f"    示例: tag={first_k}, len={len(gt_refs[first_k])}bp")

    # 建立 cluster_id → 多数投票 GT_tag_int
    print("[4] 建立 cluster → GT ref 映射 (多数投票)...")
    cluster_gt_votes = defaultdict(Counter)
    for i in range(TOTAL):
        cid = int(clover_labels[i])
        gt = int(gt_labels[i])
        if cid >= 0 and gt >= 0:
            cluster_gt_votes[cid][gt] += 1

    cluster_to_gt_int = {}
    for cid, counter in cluster_gt_votes.items():
        cluster_to_gt_int[cid] = counter.most_common(1)[0][0]
    print(f"    {len(cluster_to_gt_int):,} 簇映射到 GT refs")

    # 释放内存
    data_loader.reads = []
    del clover_labels, gt_labels, cluster_gt_votes

    # =================================================================
    # 2. 每轮评估
    # =================================================================
    print("\n" + "=" * 80)
    print("📊 Consensus 重建质量评估 (exp_1)")
    print("=" * 80)

    for round_name, rdir in CANDIDATE_ROUNDS:
        fasta_path = os.path.join(rdir, "consensus_sequences.fasta")
        if not os.path.exists(fasta_path):
            continue

        print(f"\n{'─' * 60}")
        print(f"🔬 {round_name}")
        print(f"{'─' * 60}")

        consensus_seqs = load_fasta(fasta_path)
        print(f"    Consensus 簇数: {len(consensus_seqs)}")

        exact_matches = 0
        total_compared = 0
        base_acc_list = []
        edit_dist_list = []
        no_gt = 0

        for header, pred_seq in consensus_seqs.items():
            m = re.match(r'cluster_(\d+)', header)
            if not m:
                continue
            cid = int(m.group(1))

            if cid not in cluster_to_gt_int:
                no_gt += 1
                continue
            gt_tag_int = cluster_to_gt_int[cid]
            if gt_tag_int not in gt_refs:
                no_gt += 1
                continue

            gt_seq = gt_refs[gt_tag_int]
            total_compared += 1

            if pred_seq == gt_seq:
                exact_matches += 1

            min_len = min(len(pred_seq), len(gt_seq))
            max_len_val = max(len(pred_seq), len(gt_seq))
            matches = sum(p == g for p, g in zip(pred_seq[:min_len], gt_seq[:min_len]))
            base_acc_list.append(matches / max_len_val)

            ed = edit_distance(pred_seq, gt_seq)
            edit_dist_list.append(ed)

        if total_compared == 0:
            print(f"    ⚠️ 无法匹配任何 GT reference")
            continue

        ba = np.array(base_acc_list)
        ed_arr = np.array(edit_dist_list)

        print(f"    已比对: {total_compared:,} 簇 (无GT: {no_gt})")
        print(f"    ────────────────────────────────────────")
        print(f"    Exact Match Rate:    {exact_matches:,}/{total_compared:,} = {exact_matches/total_compared*100:.2f}%")
        print(f"    Base-level Accuracy: mean={ba.mean():.6f}, median={np.median(ba):.6f}")
        print(f"    Edit Distance:       mean={ed_arr.mean():.2f}, median={np.median(ed_arr):.1f}, max={ed_arr.max()}")
        print(f"    ──── Edit Distance 分布 ────")
        for thresh in [0, 1, 2, 3, 5, 10]:
            cnt = int((ed_arr <= thresh).sum())
            print(f"      ED≤{thresh:2d}: {cnt:>7,} ({cnt/total_compared*100:6.2f}%)")
        cnt_high = int((ed_arr > 10).sum())
        print(f"      ED>10: {cnt_high:>7,} ({cnt_high/total_compared*100:6.2f}%)")

    print(f"\n{'=' * 80}")
    print("✅ 重建评估完成")


if __name__ == "__main__":
    main()