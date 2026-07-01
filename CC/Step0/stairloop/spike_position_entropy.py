#!/usr/bin/env python3
"""
spike_position_entropy.py
=========================
只读：扫 test_encode.fasta 全部 ref，算每个位置的碱基熵(区分度)。
熵高 = 该位置碱基多样、区分度高，适合放 Clover 树锚点。
熵低 = 重复 motif 区(ACAAAC/AGTATG)，所有序列雷同，hash 会撞 → 欠分割。

输出每位置熵 + 推荐 thd_tree_loc / four_tree_loc。
用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/stairloop
    python spike_position_entropy.py
"""
import math
from collections import Counter

REF = '/mnt/st_data/liangxinyi/code/CC/Step0/stairloop/test_encode.fasta'
REF_LEN = 130
WIN = 15           # Clover other_tree_len，树窗口宽度

def main():
    pos = [Counter() for _ in range(REF_LEN)]
    n = 0
    with open(REF) as f:
        for line in f:
            if line.startswith('>'): continue
            s = line.strip()
            n += 1
            for i in range(min(len(s), REF_LEN)):
                pos[i][s[i].upper()] += 1
    print(f"ref 条数: {n:,}\n")

    ent = []
    for i in range(REF_LEN):
        tot = sum(pos[i].values())
        e = 0.0
        for b in 'ACGT':
            p = pos[i].get(b, 0) / tot if tot else 0
            if p > 0: e -= p * math.log2(p)
        ent.append(e)

    # 打印每 5 位的熵（条形）
    print("位置  熵(0-2)  " + "区分度")
    for i in range(0, REF_LEN, 5):
        seg = ent[i:i+5]
        avg = sum(seg)/len(seg)
        bar = '█' * int(avg*20)
        print(f"  {i:3d}  {avg:.2f}  {bar}")

    # 滑窗找平均熵最高的 WIN 宽窗口（前段、后段各一）
    def best_window(lo, hi):
        best_i, best_e = lo, -1
        for i in range(lo, hi - WIN):
            we = sum(ent[i:i+WIN]) / WIN
            if we > best_e:
                best_e, best_i = we, i
        return best_i, best_e

    front_i, front_e = best_window(5, REF_LEN//2)
    back_i,  back_e  = best_window(REF_LEN//2, REF_LEN-5)

    print(f"\n=== 高区分度窗口 (宽{WIN}) ===")
    print(f"  前半最佳: 位置 {front_i}  平均熵 {front_e:.2f}")
    print(f"  后半最佳: 位置 {back_i}   平均熵 {back_e:.2f}")

    # 换算 Clover 参数：
    #  thd_tree_loc = front_i  (前段树从开头数)
    #  four_tree_loc: 后段树锚点 = read_len-2-four_tree_loc = back_i
    #    => four_tree_loc = read_len-2-back_i
    thd = front_i
    four = REF_LEN - 2 - back_i
    print(f"\n=== 推荐 Clover 参数 ===")
    print(f"  thd_tree_loc  = {thd}")
    print(f"  four_tree_loc = {four}")
    print(f"  (当前 44/86，若差异大说明锚点切在了重复区)")

    # 额外：全局低熵区警告
    low = [i for i, e in enumerate(ent) if e < 0.3]
    print(f"\n  低熵位置(熵<0.3, 重复motif): {len(low)} / {REF_LEN}")
    if low:
        print(f"  低熵区间示例: {low[:20]}")

if __name__ == '__main__':
    main()