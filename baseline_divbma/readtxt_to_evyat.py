#!/usr/bin/env python3
"""
readtxt_to_evyat.py
===================
把打薄 Clover 的 read.txt 转成 Sabary Iterative Reconstruction 需要的 evyat.txt。

read.txt 格式:
    <read>
    <read>
    =====分隔符=====
    <read>
    ...

evyat.txt 格式（每簇）:
    <original>      ← 占位，用该簇 MV 结果（算法不依赖它，仅供 C++ 内部 ED 直方图）
    ****
    <copy>
    <copy>
    <空行>
    <空行>

铁律:
  - 簇顺序与 read.txt 严格一致
  - 不跳过任何簇（含单 read 簇）→ 保证 cluster_i 对齐 eval 的 labels 簇 i
"""
import argparse
from collections import Counter

SEP_PREFIX = "====="   # read.txt 分隔符以此开头

def majority_vote(reads, ref_len):
    vote = [Counter() for _ in range(ref_len)]
    for r in reads:
        for pos in range(min(len(r), ref_len)):
            b = r[pos].upper()
            if b in "ACGT":
                vote[pos][b] += 1
    out, last = [], "A"
    for pos in range(ref_len):
        if vote[pos]:
            last = vote[pos].most_common(1)[0][0]
        out.append(last)
    return "".join(out)

def load_clusters(read_txt):
    """按 read.txt 顺序返回 [[reads of cluster0], [reads of cluster1], ...]"""
    clusters, cur = [], []
    with open(read_txt) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(SEP_PREFIX):
                clusters.append(cur)
                cur = []
            else:
                cur.append(s.upper())
    if cur:               # 文件末尾若无分隔符，补最后一簇
        clusters.append(cur)
    return clusters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read_txt", required=True)
    ap.add_argument("--out", default="evyat.txt")
    ap.add_argument("--ref_len", type=int, default=196)
    args = ap.parse_args()

    clusters = load_clusters(args.read_txt)
    n_single = sum(1 for c in clusters if len(c) == 1)
    n_empty  = sum(1 for c in clusters if len(c) == 0)

    with open(args.out, "w") as f:
        for reads in clusters:
            # 空簇也要占位，绝不跳过，否则编号错位
            original = majority_vote(reads, args.ref_len) if reads else "A" * args.ref_len
            f.write(original + "\n")
            f.write("****\n")
            for r in reads:
                f.write(r + "\n")
            f.write("\n\n")   # 两个空行 = 簇结束

    print(f"簇数:        {len(clusters):,}")
    print(f"  单 read 簇: {n_single:,}")
    print(f"  空簇:       {n_empty:,}  (已占位, 不影响编号)")
    print(f"输出:        {args.out}")
    print(f"\n⚠️ 校验: 此簇数必须 == eval 脚本里 read.txt 解析出的簇数")

if __name__ == "__main__":
    main()