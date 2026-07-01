#!/usr/bin/env python3
"""
nips17_calibrate_spike.py
=========================
NIPS'17 (Rashtchian et al. 2017) 复现的【参数标定 spike】—— 只读，不改任何数据。

目的
----
原文 §5.1 参数（r=25, θ_low=40, θ_high=60, blocking=22）是针对 m=150、p≈4%
的 Organick 数据标定的。Seq_1D 是 m=196、测序平台/错误率未知，必须从【实际要跑的
那份打薄后数据】反推参数，否则套用原文常数会让 baseline 不公平（偏紧→簇被切碎）。

本 spike 抽样量三类分布，给出 Seq_1D 上的推荐 r / θ_low / θ_high：

  1. 簇内编辑距离  d_E(同 tag 内 read 对)        → 定 r（原文 r 略大于簇直径）
  2. 簇内 q-gram 距离 d_H(σ, 同 tag 内 read 对)  → 定 θ_low（簇内绝大多数应 ≤ θ_low）
  3. 簇间 q-gram 距离 d_H(σ, 不同 tag 的 read 对) → 定 θ_high（簇间应 > θ_high，防误并）

blocked signature 两种块长都量（22 严格照原文 / 28 按 m=196 比例），定稿再选。

数据源：打薄后的 seq1d_tags_reads.txt（tag<TAB>read），与 baseline 实跑同源。
全程：只读、抽样、不写任何业务文件（仅可选存一张分布 PNG 供留档）。

用法
----
    python nips17_calibrate_spike.py
    python nips17_calibrate_spike.py --tags_file /path/seq1d_tags_reads.txt
    python nips17_calibrate_spike.py --n_intra_pairs 20000 --n_inter_pairs 20000
"""

import os
import sys
import argparse
import random
from collections import defaultdict

import numpy as np

# ── edlib 自检（kunyu conda env dna 应已具备）──
try:
    import edlib
except ImportError:
    print("✗ 缺少 edlib。请先 `pip install edlib` 或在 conda env dna 中运行。")
    sys.exit(1)


DEFAULT_TAGS_FILE = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d_p0.2/seq1d_tags_reads.txt"
RANDOM_SEED = 42
PRIMER_PREFIX = 20   # Seq_1D 引物前缀（与 pipeline --primer_prefix 一致）
PRIMER_SUFFIX = 20   # Seq_1D 引物后缀


def edit_distance(a, b):
    return edlib.align(a, b, task="distance")["editDistance"]


def blocked_signature(seq, block_len, q=3):
    """
    原文 blocked binary signature：把 seq 切成 block_len 字符的块，
    每块算 q-gram 指示集（σ_q），返回所有块的 q-gram 集合之并（带块偏移，
    使不同块的同一 q-gram 互不混淆）。

    q-gram 距离 = 两个 signature 集合的对称差大小（= Hamming distance of
    indicator vectors）。这里用 set 表示稀疏指示向量，等价且省内存。
    """
    sig = set()
    for bstart in range(0, len(seq), block_len):
        block = seq[bstart:bstart + block_len]
        boff = bstart // block_len
        for i in range(len(block) - q + 1):
            sig.add((boff, block[i:i + q]))
    return sig


def qgram_distance(sig_a, sig_b):
    """对称差大小 = Hamming distance of indicator vectors。"""
    return len(sig_a ^ sig_b)


def strip_primer(seq):
    """去引物区，只留 payload（与 SSI-EC 口径一致）。引物区高度同源会污染距离分布。"""
    if PRIMER_PREFIX + PRIMER_SUFFIX >= len(seq):
        return seq
    return seq[PRIMER_PREFIX: len(seq) - PRIMER_SUFFIX]


def load_groups(tags_file):
    """读 tag<TAB>read，按 tag 分组。返回 {tag: [read,...]}。"""
    groups = defaultdict(list)
    n = 0
    with open(tags_file) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            tag, read = parts
            groups[tag].append(read)
            n += 1
    return groups, n


def pct(arr, ps=(50, 90, 95, 99, 99.9, 100)):
    arr = np.asarray(arr, dtype=float)
    return {p: float(np.percentile(arr, p)) for p in ps}


def fmt_pct(d):
    return "  ".join(f"P{p}={v:.1f}" for p, v in d.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags_file", default=DEFAULT_TAGS_FILE)
    ap.add_argument("--n_intra_pairs", type=int, default=20000,
                    help="簇内 read 对抽样数（编辑距离 + q-gram 距离）")
    ap.add_argument("--n_inter_pairs", type=int, default=20000,
                    help="簇间 read 对抽样数（q-gram 距离）")
    ap.add_argument("--block_lens", type=int, nargs="+", default=[22, 28],
                    help="blocked signature 块长（两种都量）")
    ap.add_argument("--strip_primer", action="store_true",
                    help="去引物区只留 payload（默认关，两种都看可分别跑）")
    ap.add_argument("--png", default=None, help="可选：分布 PNG 输出路径")
    args = ap.parse_args()

    rng = random.Random(RANDOM_SEED)

    print("=" * 64)
    print("  NIPS'17 参数标定 spike（只读）")
    print("=" * 64)
    print(f"  tags_file     : {args.tags_file}")
    print(f"  edlib         : ✓")
    print(f"  strip_primer  : {args.strip_primer} (prefix={PRIMER_PREFIX}, suffix={PRIMER_SUFFIX})")
    print(f"  block_lens    : {args.block_lens}")
    print(f"  抽样           : 簇内 {args.n_intra_pairs:,} 对 / 簇间 {args.n_inter_pairs:,} 对")
    print()

    if not os.path.exists(args.tags_file):
        print(f"✗ 文件不存在: {args.tags_file}")
        sys.exit(1)

    groups, n_reads = load_groups(args.tags_file)
    multi = {t: rs for t, rs in groups.items() if len(rs) >= 2}
    tags_multi = list(multi.keys())
    all_tags = list(groups.keys())

    print(f"  读到 reads     : {n_reads:,}")
    print(f"  tag 总数       : {len(groups):,}")
    print(f"  size≥2 的 tag  : {len(tags_multi):,}（可抽簇内对）")
    print()

    _xf = strip_primer if args.strip_primer else (lambda s: s)

    # ── 1. 簇内编辑距离 → r ──
    intra_ed = []
    tries = 0
    while len(intra_ed) < args.n_intra_pairs and tries < args.n_intra_pairs * 5:
        tries += 1
        t = rng.choice(tags_multi)
        rs = multi[t]
        a, b = rng.sample(rs, 2)
        intra_ed.append(edit_distance(_xf(a), _xf(b)))

    # ── 2. 簇内 q-gram 距离（每个块长）→ θ_low ──
    # ── 3. 簇间 q-gram 距离（每个块长）→ θ_high ──
    intra_qg = {bl: [] for bl in args.block_lens}
    inter_qg = {bl: [] for bl in args.block_lens}

    # 簇内 q-gram：复用上面采的同 tag 对
    intra_pairs = []
    tries = 0
    while len(intra_pairs) < args.n_intra_pairs and tries < args.n_intra_pairs * 5:
        tries += 1
        t = rng.choice(tags_multi)
        a, b = rng.sample(multi[t], 2)
        intra_pairs.append((_xf(a), _xf(b)))

    # 簇间 q-gram：随机两个不同 tag 各取一条
    inter_pairs = []
    tries = 0
    while len(inter_pairs) < args.n_inter_pairs and tries < args.n_inter_pairs * 5:
        tries += 1
        t1, t2 = rng.sample(all_tags, 2)
        a = rng.choice(groups[t1])
        b = rng.choice(groups[t2])
        inter_pairs.append((_xf(a), _xf(b)))

    for bl in args.block_lens:
        for a, b in intra_pairs:
            intra_qg[bl].append(qgram_distance(
                blocked_signature(a, bl), blocked_signature(b, bl)))
        for a, b in inter_pairs:
            inter_qg[bl].append(qgram_distance(
                blocked_signature(a, bl), blocked_signature(b, bl)))

    # ── 结果 ──
    print("─" * 64)
    print("  【1】簇内编辑距离 d_E（定 r）")
    print("─" * 64)
    ed_p = pct(intra_ed)
    print(f"    n={len(intra_ed):,}   {fmt_pct(ed_p)}")
    print(f"    均值={np.mean(intra_ed):.2f}  最大={max(intra_ed)}")
    r_reco = int(np.ceil(ed_p[99.9]))
    print(f"    → 推荐 r ≈ P99.9 = {r_reco}（覆盖几乎所有簇内对，略大于簇直径）")
    print(f"      对比原文 r=25 (m=150,p=4%)；理论 4pm，反推 p≈{np.mean(intra_ed)/(2*196)*100:.1f}%")
    print()

    for bl in args.block_lens:
        print("─" * 64)
        print(f"  【2&3】q-gram 距离 d_H（block_len={bl}）")
        print("─" * 64)
        ia = pct(intra_qg[bl])
        ie = pct(inter_qg[bl])
        print(f"    簇内 n={len(intra_qg[bl]):,}  {fmt_pct(ia)}")
        print(f"    簇间 n={len(inter_qg[bl]):,}  {fmt_pct(ie)}")
        # θ_low：簇内绝大多数 ≤ 它（取簇内 P95~P99）
        # θ_high：簇间几乎都 > 它，且 > θ_low（取簇内 P99.9 与簇间 P1 之间）
        theta_low = int(np.ceil(ia[95]))
        inter_p1 = float(np.percentile(inter_qg[bl], 1))
        theta_high = int(np.ceil(min(ia[99.9], inter_p1)))
        gap = ie[50] - ia[50]
        print(f"    簇内/簇间中位数间隔 = {gap:.1f}  "
              f"({'✓ 可分' if gap > 5 else '⚠ 间隔小,q-gram 区分力弱'})")
        print(f"    → 推荐 θ_low ≈ 簇内P95 = {theta_low}")
        print(f"    → 推荐 θ_high ≈ {theta_high}（簇内P99.9={ia[99.9]:.0f} 与簇间P1={inter_p1:.0f} 取小）")
        if theta_high <= theta_low:
            print(f"    ⚠ θ_high≤θ_low：该块长下两级阈值无法拉开，建议换块长或仅用编辑距离判据")
        print()

    # ── 可选 PNG ──
    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ncol = 1 + len(args.block_lens)
            fig, axes = plt.subplots(1, ncol, figsize=(5 * ncol, 4))
            if ncol == 1:
                axes = [axes]
            axes[0].hist(intra_ed, bins=40, color="#4C72B0")
            axes[0].axvline(r_reco, color="red", ls="--", label=f"r={r_reco}")
            axes[0].set_title("Intra-cluster edit distance"); axes[0].legend()
            for k, bl in enumerate(args.block_lens, start=1):
                axes[k].hist(intra_qg[bl], bins=40, alpha=0.6, color="#55A868", label="intra")
                axes[k].hist(inter_qg[bl], bins=40, alpha=0.6, color="#C44E52", label="inter")
                axes[k].set_title(f"q-gram dist (block={bl})"); axes[k].legend()
            fig.tight_layout(); fig.savefig(args.png, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✅ 分布 PNG: {args.png}")
        except Exception as e:
            print(f"  ⚠ PNG 失败: {e}")

    print("=" * 64)
    print("  spike 完成。把上面【推荐 r / θ_low / θ_high】贴回，即可定稿正式复现脚本。")
    print("=" * 64)


if __name__ == "__main__":
    main()