#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spike: 假负样本率验证 (False-Negative Rate)
==========================================
新根因假设 (假设 A):
  embedding 不保距 (Spike G AUROC 0.77) 的真正原因不是"正样本脏"
  (上个 spike 证明 Clover 簇正样本纯度 99.79%, 很干净),
  而是"负样本脏" —— 同一个真 GT 分子被 Clover 过分割成多个簇,
  这些同源簇互为负样本, InfoNCE 被迫把真同源对推开 = 假负样本.

本 spike 测三件事:
  1. 过分割倍率: 平均每个真 GT 被切成几个 Clover 簇
  2. 假负样本率: 随机跨簇负样本对里, 真同 GT 的比例 (= InfoNCE 在错误推开的比例)
  3. edit 救援力: 这些假负样本对的归一化 edit 距离是否 < 0.05
                  (edit 能否把假负样本识别出来 -> 决定 v21 方向 X 是否可行)

判读:
  - 假负样本率显著 (>5%) 且 假负样本 edit 普遍 <0.05
    -> 假设 A 成立, 且 edit 可救 -> v21 方向 X (edit 屏蔽/转正假负样本)
  - 假负样本率低 (<2%)
    -> 假设 A 不成立, 负样本其实也干净, 根因另在他处, 需重新诊断
  - 假负样本率高但 edit 也救不了 (edit 普遍 >0.1)
    -> IDS 噪声太重, 序列空间也分不开同源, 需更强手段

只读, 纯 CPU.
"""
import sys, os, time, random
sys.path.insert(0, '/mnt/st_data/liangxinyi/code')
import numpy as np
from collections import defaultdict

from models.step1_data import CloverDataLoader

# ----------------------------- 配置 -----------------------------
EXP_DIR    = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d"
OUTPUT_TXT = "/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/output.txt"
OUT_DIR    = "/mnt/st_data/liangxinyi/code/CC/Step0/spike_purity_out"

N_NEG_PAIRS = 200000   # 随机抽多少跨簇负样本对来估假负样本率
EDIT_CHECK  = 2000     # 对多少个假负样本对算 edit (太慢, 抽样)
EDIT_NEG_CHECK = 2000  # 对多少个真负样本对算 edit (做对照, 看 edit 是否真能区分)
SEED        = 42
# ---------------------------------------------------------------

random.seed(SEED); np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    print("=" * 60)
    print("📂 加载数据 + GT")
    print("=" * 60)
    dl = CloverDataLoader(EXP_DIR)
    dl.load_gt_tags(OUTPUT_TXT)
    reads  = dl.reads
    gt     = np.array(dl.gt_labels)
    clover = np.array(dl.clover_labels)
    N = len(reads)
    print(f"   总 read: {N}, 不同 GT: {len(set(gt[gt>=0].tolist()))}, "
          f"不同 Clover 簇: {len(set(clover[clover>=0].tolist()))}")

    # ---------- 1. 过分割倍率 ----------
    print("\n" + "=" * 60)
    print("📐 1. Clover 过分割倍率")
    print("=" * 60)
    gt_to_clovers = defaultdict(set)
    for i in range(N):
        if gt[i] >= 0 and clover[i] >= 0:
            gt_to_clovers[int(gt[i])].add(int(clover[i]))
    splits = np.array([len(s) for s in gt_to_clovers.values()])
    print(f"   真 GT 数: {len(splits)}")
    print(f"   平均每个 GT 被切成 {splits.mean():.2f} 个 Clover 簇")
    print(f"   中位数: {int(np.median(splits))}, 最大: {splits.max()}, "
          f"只占1簇(未被切)的 GT: {(splits==1).sum()} ({(splits==1).mean()*100:.1f}%)")
    print(f"   被切成 >=2 簇的 GT: {(splits>=2).sum()} ({(splits>=2).mean()*100:.1f}%)")

    # ---------- 2. 假负样本率 ----------
    print("\n" + "=" * 60)
    print("🔬 2. 假负样本率 (随机跨簇负样本对里真同 GT 的比例)")
    print("=" * 60)
    # 随机抽对, 保留"跨 Clover 簇"的对 (= 训练时的负样本), 统计其中同 GT 的
    valid_idx = np.where((gt >= 0) & (clover >= 0))[0]
    a = np.random.choice(valid_idx, size=N_NEG_PAIRS, replace=True)
    b = np.random.choice(valid_idx, size=N_NEG_PAIRS, replace=True)
    keep = (a != b) & (clover[a] != clover[b])   # 跨簇 = 负样本
    a, b = a[keep], b[keep]
    n_neg = len(a)
    same_gt = (gt[a] == gt[b])
    n_false_neg = int(same_gt.sum())
    fn_rate = n_false_neg / max(n_neg, 1)
    print(f"   采样跨簇负样本对: {n_neg}")
    print(f"   其中真同 GT (假负样本): {n_false_neg} ({fn_rate*100:.3f}%)")
    print(f"   -> InfoNCE 每个 batch 约有 {fn_rate*100:.2f}% 的负样本在错误推开真同源对")

    # 保存假负样本对的索引供 edit 检查
    fn_a = a[same_gt]; fn_b = b[same_gt]
    tn_a = a[~same_gt]; tn_b = b[~same_gt]  # 真负样本 (对照)

    # ---------- 3. edit 救援力 ----------
    print("\n" + "=" * 60)
    print("🐢 3. edit 救援力: 假负样本 vs 真负样本 的归一化 edit 距离分布")
    print("=" * 60)
    try:
        import editdistance as _ed
    except ImportError:
        print("   ⚠️ 无 editdistance 库, 跳过. (pip install editdistance 启用)")
        _ed = None

    if _ed is not None and len(fn_a) > 0:
        def norm_ed_stats(idx_a, idx_b, n_check, tag):
            m = min(n_check, len(idx_a))
            sel = np.random.choice(len(idx_a), size=m, replace=False)
            ds = []
            for k in sel:
                ia, ib = idx_a[k], idx_b[k]
                d = _ed.eval(reads[ia], reads[ib]) / max(len(reads[ia]), 1)
                ds.append(d)
            ds = np.array(ds)
            print(f"   [{tag}] n={m}: "
                  f"median={np.median(ds):.4f}, mean={ds.mean():.4f}, "
                  f"<0.05 的比例={ (ds<0.05).mean()*100:.1f}%, "
                  f"<0.10 的比例={ (ds<0.10).mean()*100:.1f}%")
            return ds

        print("   假负样本 (跨簇但真同 GT, 应该 edit 很小):")
        fn_ds = norm_ed_stats(fn_a, fn_b, EDIT_CHECK, "假负样本")
        print("   真负样本 (跨簇且异 GT, 应该 edit 很大):")
        tn_ds = norm_ed_stats(tn_a, tn_b, EDIT_NEG_CHECK, "真负样本")

        # 用一个阈值看能否分开
        TH = 0.08
        fn_caught = (fn_ds < TH).mean()
        tn_falsely = (tn_ds < TH).mean()
        print(f"\n   若用 norm_edit < {TH} 识别假负样本:")
        print(f"      能救回的假负样本: {fn_caught*100:.1f}%")
        print(f"      误伤的真负样本:   {tn_falsely*100:.1f}% (越低越好)")

    # ---------- 判读 ----------
    print("\n" + "=" * 60)
    print("📊 判读")
    print("=" * 60)
    print(f"   过分割: {(splits>=2).mean()*100:.0f}% 的 GT 被切成多簇 (倍率 {splits.mean():.1f}x)")
    print(f"   假负样本率: {fn_rate*100:.2f}%")
    if fn_rate >= 0.05:
        print(f"   🟢 假负样本率显著 -> 假设 A 成立: 负样本被假同源污染")
        if _ed is not None:
            if fn_caught >= 0.8 and tn_falsely <= 0.05:
                print(f"   ✅ edit 能干净识别假负样本 -> v21 方向 X (edit 屏蔽/转正) 可行")
            else:
                print(f"   ⚠️ edit 区分力不足 (救回{fn_caught*100:.0f}%/误伤{tn_falsely*100:.0f}%) -> 需调阈值或换策略")
    elif fn_rate >= 0.02:
        print(f"   🟡 假负样本率中等 -> 假设 A 部分成立, 收益有限, 需权衡")
    else:
        print(f"   🔴 假负样本率低 -> 假设 A 不成立! 负样本也干净,")
        print(f"      embedding 不保距的根因另在他处 (encoder 容量? recon/contrastive 冲突?), 需重新诊断")

    print(f"\n⏱️ 总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()