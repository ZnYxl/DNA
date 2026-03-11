#!/usr/bin/env python3
"""
SSI-EC Zone I 分层评估脚本
用法: python eval_zone1.py 2>&1 | tee zone1_eval.txt

评估每一轮的 Zone I / Zone II / Zone III 各自的 Purity 和 PCR
"""
import numpy as np
import torch
import os
import glob
from collections import Counter, defaultdict

# ═══════════════════════════════════════════════════
# 路径配置 (根据你的服务器修改)
# ═══════════════════════════════════════════════════
EXP_DIR   = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_NoPrimer"
GT_FILE   = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1_NoPrimer/exp1_tags_reads.txt"
READ_FILE = os.path.join(EXP_DIR, "03_FedDNA_In", "read.txt")
LABELS_DIR = os.path.join(EXP_DIR, "04_Iterative_Labels")


def load_reads():
    reads = []
    with open(READ_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("====="):
                continue
            reads.append(line)
    return reads


def load_gt(reads):
    seq_to_gt = {}
    with open(GT_FILE) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    seq_to_gt[parts[1].strip().upper()] = int(parts[0])
                except ValueError:
                    pass
    gt = np.full(len(reads), -1, dtype=np.int64)
    for i, r in enumerate(reads):
        g = seq_to_gt.get(r.upper())
        if g is not None:
            gt[i] = g
    matched = (gt >= 0).sum()
    print(f"   GT 匹配: {matched:,} / {len(reads):,} ({matched/len(reads)*100:.1f}%)")
    return gt


def eval_zone(labels, gt, zone_ids, zone_val, zone_name):
    """评估特定 zone 的 Purity 和 PCR"""
    mask = (zone_ids == zone_val) & (labels >= 0) & (gt >= 0)
    n = mask.sum()
    if n == 0:
        print(f"   {zone_name}: 无有效 reads")
        return None

    pred = labels[mask]
    gold = gt[mask]

    c2g = defaultdict(list)
    for p, g in zip(pred, gold):
        c2g[p].append(g)

    correct = sum(Counter(gs).most_common(1)[0][1] for gs in c2g.values())
    purity = correct / n
    perfect = sum(1 for gs in c2g.values() if len(set(gs)) == 1)
    n_clusters = len(c2g)
    pcr = perfect / n_clusters if n_clusters > 0 else 0

    print(f"   {zone_name}:")
    print(f"      Reads:   {n:>10,}  ({n/len(labels)*100:.1f}%)")
    print(f"      簇数:    {n_clusters:>10,}")
    print(f"      Purity:  {purity:.4f}")
    print(f"      PCR:     {perfect}/{n_clusters} ({pcr:.4f})")
    return {'reads': int(n), 'clusters': n_clusters, 'purity': purity, 'pcr': pcr, 'perfect': perfect}


def main():
    print("=" * 70)
    print("  SSI-EC Zone 分层评估")
    print("=" * 70)

    reads = load_reads()
    gt = load_gt(reads)
    N = len(reads)

    # 找所有 round 的文件
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "refined_labels_*.txt")))
    state_files = sorted(glob.glob(os.path.join(LABELS_DIR, "read_state_*.pt")))

    print(f"\n   找到 {len(label_files)} 轮结果")

    # Clover 基线
    print(f"\n{'='*70}")
    print(f"  Clover 基线 (全量)")
    print(f"{'='*70}")
    clover_labels = []
    cid = 0
    with open(READ_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("====="):
                cid += 1
            else:
                clover_labels.append(cid)
    clover_labels = np.array(clover_labels, dtype=np.int64)

    clover_mask = (clover_labels >= 0) & (gt >= 0)
    c2g_clover = defaultdict(list)
    for p, g in zip(clover_labels[clover_mask], gt[clover_mask]):
        c2g_clover[p].append(g)
    clover_correct = sum(Counter(gs).most_common(1)[0][1] for gs in c2g_clover.values())
    clover_purity = clover_correct / clover_mask.sum()
    clover_perfect = sum(1 for gs in c2g_clover.values() if len(set(gs)) == 1)
    clover_pcr = clover_perfect / len(c2g_clover)
    print(f"   Reads:   {clover_mask.sum():,}")
    print(f"   簇数:    {len(c2g_clover):,}")
    print(f"   Purity:  {clover_purity:.4f}")
    print(f"   PCR:     {clover_perfect}/{len(c2g_clover)} ({clover_pcr:.4f})")

    # 每一轮评估
    all_results = []
    for r_idx in range(min(len(label_files), len(state_files))):
        round_num = r_idx + 1
        print(f"\n{'='*70}")
        print(f"  Round {round_num}")
        print(f"{'='*70}")
        print(f"   文件: {os.path.basename(label_files[r_idx])}")

        labels = np.loadtxt(label_files[r_idx], dtype=np.int64)
        state = torch.load(state_files[r_idx], map_location='cpu')
        zone_ids = np.array(state['zone_ids'], dtype=np.int64)

        # 全量评估
        print(f"\n   --- 全量 (不含 noise) ---")
        r_all = eval_zone(labels, gt, np.ones(N, dtype=np.int64), 1, "全量")

        # 分 Zone 评估
        print(f"\n   --- 分 Zone ---")
        r_z1 = eval_zone(labels, gt, zone_ids, 1, "Zone I  (Safe)")
        r_z2 = eval_zone(labels, gt, zone_ids, 2, "Zone II (Hard)")
        r_z3 = eval_zone(labels, gt, zone_ids, 3, "Zone III(Dirty)")

        all_results.append({
            'round': round_num,
            'all': r_all, 'z1': r_z1, 'z2': r_z2, 'z3': r_z3
        })

    # 汇总表
    if all_results:
        print(f"\n{'='*90}")
        print(f"{'汇总表':^90}")
        print(f"{'='*90}")
        print(f"  {'':15} {'Clover':>10} ", end="")
        for r in all_results:
            print(f"  {'R'+str(r['round'])+' 全量':>10}  {'R'+str(r['round'])+' ZoneI':>10}", end="")
        print()
        print(f"  {'-'*85}")

        # Purity 行
        print(f"  {'Purity':15} {clover_purity:>10.4f} ", end="")
        for r in all_results:
            p_all = r['all']['purity'] if r['all'] else 0
            p_z1 = r['z1']['purity'] if r['z1'] else 0
            print(f"  {p_all:>10.4f}  {p_z1:>10.4f}", end="")
        print()

        # PCR 行
        print(f"  {'PCR':15} {clover_pcr:>10.4f} ", end="")
        for r in all_results:
            pcr_all = r['all']['pcr'] if r['all'] else 0
            pcr_z1 = r['z1']['pcr'] if r['z1'] else 0
            print(f"  {pcr_all:>10.4f}  {pcr_z1:>10.4f}", end="")
        print()

        # Zone I 覆盖率
        print(f"  {'Zone I %':15} {'100%':>10} ", end="")
        for r in all_results:
            z1_pct = r['z1']['reads'] / N * 100 if r['z1'] else 0
            print(f"  {'':>10}  {z1_pct:>9.1f}%", end="")
        print()

        print(f"{'='*90}")

    print(f"\n✅ 评估完成")


if __name__ == '__main__':
    main()