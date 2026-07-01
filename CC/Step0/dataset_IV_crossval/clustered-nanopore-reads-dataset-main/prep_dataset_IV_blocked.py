#!/usr/bin/env python3
"""
prep_dataset_IV_blocked.py
==========================
GradHC 交叉验证 —— dataset IV —— 【喂法 A: 按 GT 分块】(README 标准输入格式)

与单块版的关键区别:
    单块版 (错误): 全部 reads 塞进 1 个 ***** 块 → GradHC 以为只有1个真簇 → 塌缩
    本版  (正确): 每个 GT 簇一个 ***** 块, rep = Centers 对应链
                  → 完全对齐 GradHC README / 论文喂法
                  → GradHC 内部按块建立初始结构 + GT参照, 再 shuffle 重新聚类

为什么"按GT分块"不是作弊:
    GradHC process_input 读完所有块后执行 random.shuffle(self.all_reads)，
    打乱所有 reads，聚类决策不看原始块归属。分块只提供:
      (1) 初始 cluster 结构 (算法起点，会被重新聚类)
      (2) original_strand_dict (GT参照，仅用于它内部精度统计)
    这正是论文跑 dataset I-IV 的标准方式。

输入:
    Centers.txt   第 i 行 = 第 i 簇的真值 rep
    Clusters.txt  第 i 个 ==== 块 = Centers 第 i 行的噪声拷贝 (顺序对应已验证)

输出:
    01_gradhc_input_blocked.txt   每簇: <Center链> + ***** + <reads> + 双空行
    01_gt_seq_to_tag.txt          GT映射 tag\tread (评估用, 序列做key)

用法:
    python prep_dataset_IV_blocked.py \
        --centers  /abs/path/Centers.txt \
        --clusters /abs/path/Clusters.txt \
        --outdir   /abs/path/prep
"""

import os
import argparse
from collections import Counter

SEP = '*' * 29   # 与 README 示例一致 (任意长度的*行都被 GradHC 识别为分隔)


def banner(t):
    print(f"\n{'─'*60}\n  {t}\n{'─'*60}")


def load_centers(path):
    centers = []
    with open(path) as f:
        for line in f:
            s = line.strip().upper()
            if s:
                centers.append(s)
    return centers


def parse_clusters(path):
    """解析 Clusters.txt → list[list[read]], 外层index=簇序号。'='开头行为分隔符。"""
    clusters = []
    cur = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line[0] == '=':
                if cur is not None:
                    clusters.append(cur)
                cur = []
                continue
            if cur is None:
                cur = []
            cur.append(line.upper())
    if cur is not None and len(cur) > 0:
        clusters.append(cur)
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--centers',  required=True)
    ap.add_argument('--clusters', required=True)
    ap.add_argument('--outdir',   required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    banner("加载 Centers / Clusters")
    centers = load_centers(args.centers)
    clusters = parse_clusters(args.clusters)
    print(f"  Centers 链数:   {len(centers):,}")
    print(f"  Clusters 簇数:  {len(clusters):,}")

    if len(centers) != len(clusters):
        print(f"  ⚠️  数量不一致 ({len(centers)} vs {len(clusters)})！按 min 截断对齐。")
    n = min(len(centers), len(clusters))

    # ── 体检 ──
    banner("数据体检")
    sizes = [len(c) for c in clusters]
    total_reads = sum(sizes)
    ss = sorted(sizes, reverse=True)
    n_empty = sum(1 for s in sizes if s == 0)
    print(f"  总 reads:       {total_reads:,}")
    print(f"  簇大小: max={ss[0]}, med={ss[len(ss)//2]}, min={ss[-1]}")
    print(f"  空簇: {n_empty},  成簇(>=1): {sum(1 for s in sizes if s>=1):,}")

    # rep 一致性抽查: Center[i] 是否与 cluster[i] 的 reads 同源 (前缀比对)
    banner("rep↔簇 对应抽查 (确认顺序对应)")
    for i in [0, 1, 100, n//2, n-1]:
        if i < n and len(clusters[i]) > 0:
            c_pref = centers[i][:25]
            r_pref = clusters[i][0][:25]
            # 简单前缀重叠度
            match = sum(1 for a, b in zip(c_pref, r_pref) if a == b)
            flag = "✓" if match >= 15 else "⚠️ 低重叠!"
            print(f"  簇{i:5d}: Center前缀={c_pref}")
            print(f"           read前缀  ={r_pref}  (匹配{match}/25 {flag})")

    # ── 写按GT分块的 GradHC 输入 ──
    banner("写 GradHC 输入 (按GT分块, README标准格式)")
    gradhc_input = os.path.join(args.outdir, '01_gradhc_input_blocked.txt')
    n_written_reads = 0
    n_written_blocks = 0
    with open(gradhc_input, 'w', newline='\n') as f:
        for i in range(n):
            reads = clusters[i]
            if len(reads) == 0:
                continue   # 跳过空簇
            f.write(centers[i] + '\n')   # rep = Center 链
            f.write(SEP + '\n')
            for r in reads:
                f.write(r + '\n')
                n_written_reads += 1
            f.write('\n\n')   # 双空行分块
            n_written_blocks += 1
    print(f"  ✅ {gradhc_input}")
    print(f"     {n_written_blocks:,} 块 (=GT簇),  {n_written_reads:,} reads")

    # ── 写 GT 映射 (seq→tag, 序列做key) ──
    banner("写 GT 映射 (seq→tag)")
    gt_path = os.path.join(args.outdir, '01_gt_seq_to_tag.txt')
    seq_first = {}
    n_cross = 0
    with open(gt_path, 'w') as f:
        for tag in range(n):
            for r in clusters[tag]:
                f.write(f"{tag}\t{r}\n")
                if r in seq_first and seq_first[r] != tag:
                    n_cross += 1
                else:
                    seq_first[r] = tag
    print(f"  ✅ {gt_path}")
    print(f"     唯一 read: {len(seq_first):,},  跨簇重复实例: {n_cross:,}")

    banner("完成 — 下一步")
    print(f"  python run_gradhc_default.py \\")
    print(f"    --input      {gradhc_input} \\")
    print(f"    --gt         {gt_path} \\")
    print(f"    --gradhc_dir <GradHC仓库>")
    print()
    print(f"  ⚠️ 注意: run 脚本会再读 input 文件喂给 GradHC。")
    print(f"     GradHC 内部 shuffle 后重新聚类，分块只是初始结构+GT参照，非作弊。")


if __name__ == '__main__':
    main()