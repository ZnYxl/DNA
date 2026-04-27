#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  diagnose_negone_zone_fate.py
    "上轮的 -1 reads, 本轮被判到了哪个 Zone?"
================================================================================

背景 (v18 后的认识):
  zone_include_noise=True 让 -1 reads 参与了 Zone 判定.
  但 Zone 判定本身不会给 -1 reads 赋 label (Zone I/II 只是"保持原 label", 
  所以 -1 仍然是 -1). 真正给 -1 reads 重获 label 的是"死数据复活".

  v18 为什么相比 v17 还是涨了一点? 因为 -1 reads 参与后, Zone III 阈值
  (P95 of u_ale) 在全量分布上计算, 拉得更高, 间接减少 Zone III 隔离量.

  但 -1 reads 本身没重获 label. 如果我们想在 v19 里让 -1 reads 也能重获
  label, 需要知道: 在 v18 的 Zone 判定下, -1 reads 被判到了哪个 Zone?

    - 如果大部分进了 Zone III → 它们 u_ale 高, 是"真脏", v19 赋标签收益低
    - 如果大部分进了 Zone I/II → 它们 u_ale 低, 是"好 reads 被误弃", 
      v19 赋标签收益大

输入:
  --experiment_dir 指向 v18 跑完的实验目录
  (需要 read_state_HHMMSS.pt, refined_labels_HHMMSS.txt)

用法:
  cd /mnt/st_data/liangxinyi/code/
  python diagnose_negone_zone_fate.py \\
      --experiment_dir /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/

无 embedding 依赖, 5 秒跑完.
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch


def discover_rounds(exp_dir):
    label_dir = os.path.join(exp_dir, "04_Iterative_Labels")
    labels = sorted(glob.glob(os.path.join(label_dir, "refined_labels_*.txt")),
                    key=os.path.getmtime)
    states = sorted(glob.glob(os.path.join(label_dir, "read_state_*.pt")),
                    key=os.path.getmtime)
    rounds = []
    for r, (l, s) in enumerate(zip(labels, states), 1):
        rounds.append((r, l, s))
    return rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment_dir', required=True)
    args = ap.parse_args()

    rounds = discover_rounds(args.experiment_dir)
    print("=" * 70)
    print("  🔬 -1 reads 在下一轮的 Zone 命运")
    print("=" * 70)
    print(f"\n  发现 {len(rounds)} 轮次")
    for r, l, s in rounds:
        print(f"    R{r}: {os.path.basename(l)}")

    if len(rounds) < 2:
        print("\n❌ 至少需要 2 轮数据才能分析")
        sys.exit(1)

    # 加载每轮数据
    all_data = []
    for r, labels_path, state_path in rounds:
        labels = np.loadtxt(labels_path, dtype=int)
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        zone_ids = state['zone_ids']
        if isinstance(zone_ids, torch.Tensor):
            zone_ids = zone_ids.numpy()
        all_data.append({
            'round': r,
            'labels': labels,
            'zone_ids': zone_ids,
        })
        n_negone = int((labels == -1).sum())
        n_z1 = int((zone_ids == 1).sum())
        n_z2 = int((zone_ids == 2).sum())
        n_z3 = int((zone_ids == 3).sum())
        print(f"    R{r}: labels=-1:{n_negone:,}  Zone 分布: "
              f"Z1={n_z1:,} Z2={n_z2:,} Z3={n_z3:,}")

    # 核心分析: 对每一对相邻轮, 看上轮 -1 reads 在本轮被判到哪
    print(f"\n{'='*70}")
    print(f"  📊 上轮 -1 reads 在本轮的 Zone 命运")
    print(f"{'='*70}")

    for i in range(1, len(all_data)):
        prev = all_data[i-1]
        curr = all_data[i]

        # 上轮是 -1 的 reads 的索引
        negone_mask = (prev['labels'] == -1)
        negone_indices = np.where(negone_mask)[0]
        n_negone = len(negone_indices)

        if n_negone == 0:
            print(f"\n  R{prev['round']} → R{curr['round']}: 上轮无 -1 reads")
            continue

        # 本轮 Zone 命运
        curr_zones = curr['zone_ids'][negone_indices]
        n_z0 = int((curr_zones == 0).sum())   # 未评估
        n_z1 = int((curr_zones == 1).sum())
        n_z2 = int((curr_zones == 2).sum())
        n_z3 = int((curr_zones == 3).sum())

        # 看这些 -1 reads 本轮最终 label 是什么 (0=复活, -1=仍然是噪声)
        curr_labels = curr['labels'][negone_indices]
        n_revived = int((curr_labels >= 0).sum())
        n_stayed_negone = int((curr_labels == -1).sum())

        print(f"\n  ── R{prev['round']} → R{curr['round']} ──")
        print(f"  上轮 -1 reads: {n_negone:,}")
        print(f"  本轮 Zone 分布:")
        print(f"     Zone 0 (未评估):  {n_z0:,}  ({n_z0/n_negone*100:.1f}%)")
        print(f"     Zone I  (安全):  {n_z1:,}  ({n_z1/n_negone*100:.1f}%)")
        print(f"     Zone II (中间):  {n_z2:,}  ({n_z2/n_negone*100:.1f}%)")
        print(f"     Zone III(脏):    {n_z3:,}  ({n_z3/n_negone*100:.1f}%)")
        print(f"  本轮最终命运:")
        print(f"     重获 label:     {n_revived:,}  ({n_revived/n_negone*100:.1f}%)")
        print(f"     仍是 -1:        {n_stayed_negone:,}  ({n_stayed_negone/n_negone*100:.1f}%)")

        # 深入: Zone I/II 的 -1 reads 有多少被救回来了?
        z12_mask = (curr_zones == 1) | (curr_zones == 2)
        z12_indices_abs = negone_indices[z12_mask]
        if len(z12_indices_abs) > 0:
            z12_labels = curr['labels'][z12_indices_abs]
            z12_revived = int((z12_labels >= 0).sum())
            z12_total = len(z12_indices_abs)
            print(f"\n  📌 关键诊断: 上轮 -1 + 本轮 Zone I/II 的 reads:")
            print(f"     总数:      {z12_total:,}")
            print(f"     其中重获 label: {z12_revived:,}  ({z12_revived/z12_total*100:.1f}%)")
            print(f"     其中仍是 -1:    {z12_total - z12_revived:,}  "
                  f"({(z12_total-z12_revived)/z12_total*100:.1f}%)")
            print(f"  → 如果【仍是-1】占比高, v19 直接赋标签能救它们")
            print(f"  → 如果【重获 label】占比已高, 说明死数据复活已经在救了, v19 边际小")

    # 总结
    print(f"\n{'='*70}")
    print(f"  🎯 结论建议")
    print(f"{'='*70}")
    if len(all_data) >= 2:
        prev = all_data[-2]
        curr = all_data[-1]
        negone_indices = np.where(prev['labels'] == -1)[0]
        if len(negone_indices) > 0:
            curr_zones = curr['zone_ids'][negone_indices]
            curr_labels = curr['labels'][negone_indices]
            z12_mask = (curr_zones == 1) | (curr_zones == 2)
            z12_idx_abs = negone_indices[z12_mask]

            z12_in_z12 = len(z12_idx_abs)
            z12_stayed_negone = int((curr['labels'][z12_idx_abs] == -1).sum())

            # 潜在可救数 = Zone I/II 的 -1 reads 且仍是 -1 的
            rescue_potential = z12_stayed_negone
            total_still_negone = int((curr['labels'][negone_indices] == -1).sum())

            print(f"\n  最后一轮 ({prev['round']} → {curr['round']}) 分析:")
            print(f"  - 上轮遗留 -1: {len(negone_indices):,} reads")
            print(f"  - 本轮仍是 -1: {total_still_negone:,}")
            print(f"  - 其中在 Zone I/II 里仍是 -1: {rescue_potential:,}")
            print(f"    (v19 如果在 Zone I/II 里对 -1 reads 赋 label, 最多能救这些)")

            if rescue_potential > 3000:
                print(f"\n  ✅ 可救数 {rescue_potential:,} >> 3000, v19 值得做")
                print(f"  → 预期 Recall 再涨 ~{rescue_potential/11826*100:.1f}pp")
            elif rescue_potential > 1000:
                print(f"\n  🟡 可救数 {rescue_potential:,} 在 1-3k, v19 边际中等")
            else:
                print(f"\n  ❌ 可救数 {rescue_potential:,} < 1000, v19 收益极小")
                print(f"  → -1 reads 大多数进了 Zone III, 它们是真脏 reads")


if __name__ == "__main__":
    main()