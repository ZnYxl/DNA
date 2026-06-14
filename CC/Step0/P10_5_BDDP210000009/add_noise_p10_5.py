#!/usr/bin/env python3
"""
add_noise_p10_5.py
==================
P10_5 加噪：在原始 output.txt 上为每条 read 注入 sub/del/ins 错误，
                生成 output_noisy.txt，模拟更高错误率的测序场景。

加噪策略：每个位置独立采样 4 个事件之一
  - sub: 2.0%   随机替换为其他 3 个碱基之一
  - del: 1.0%   删除该位置
  - ins: 1.0%   在该位置后插入一个随机碱基
  - keep: 96.0% 保持不变

总错误率 = 4%，模拟比 Twist Bioscience 默认更高的错误环境。

注意：
  ‣ reads 长度会有微小波动（del/ins 概率相同时长度期望不变，但方差增加）
  ‣ tag (BWA 比对结果) 不变——加噪不破坏 ground truth label
  ‣ 不动 ref.fasta，GT 标准保持不变
  ‣ 用确定性随机种子保证可复现

用法:
    python add_noise_p10_5.py
    python add_noise_p10_5.py --p_sub 0.03 --p_del 0.015 --p_ins 0.015  # 6% 总错误率
"""

import os
import random
import argparse
import time
from collections import Counter

# ============================================================
# 配置
# ============================================================
BASE_DIR    = '/mnt/st_data/liangxinyi/code/CC/Step0/P10_5_BDDP210000009'
INPUT_FILE  = os.path.join(BASE_DIR, 'output.txt')
OUTPUT_FILE = os.path.join(BASE_DIR, 'output_noisy.txt')

# 默认加噪率（与 PE_AYB 难度匹配）
DEFAULT_P_SUB = 0.020
DEFAULT_P_DEL = 0.010
DEFAULT_P_INS = 0.010

RANDOM_SEED = 42
BASES = 'ACGT'


def add_noise_to_read(seq, p_sub, p_del, p_ins, rng):
    """
    对单条 read 注入 sub/del/ins 错误。

    每个位置独立采样：
      - sub (p_sub):    替换为其他 3 个碱基之一
      - del (p_del):    跳过该位置
      - ins (p_ins):    保留该位置 + 在其后插入随机碱基
      - keep (1-others): 保持不变

    Args:
        seq:   原始序列
        p_sub, p_del, p_ins: 各事件概率
        rng:   random.Random 实例（线程安全 + 可复现）

    Returns:
        加噪后的序列
    """
    out = []
    for base in seq:
        r = rng.random()
        if r < p_sub:
            # sub: 替换为 BASES \ {base} 中随机一个
            choices = [b for b in BASES if b != base.upper()]
            out.append(rng.choice(choices))
        elif r < p_sub + p_del:
            # del: 不输出
            pass
        elif r < p_sub + p_del + p_ins:
            # ins: 先输出原碱基，再插入一个随机碱基
            out.append(base)
            out.append(rng.choice(BASES))
        else:
            # keep
            out.append(base)
    return ''.join(out)


def main():
    parser = argparse.ArgumentParser(description='P10_5 加噪脚本')
    parser.add_argument('--input',  type=str, default=INPUT_FILE)
    parser.add_argument('--output', type=str, default=OUTPUT_FILE)
    parser.add_argument('--p_sub',  type=float, default=DEFAULT_P_SUB)
    parser.add_argument('--p_del',  type=float, default=DEFAULT_P_DEL)
    parser.add_argument('--p_ins',  type=float, default=DEFAULT_P_INS)
    parser.add_argument('--seed',   type=int,   default=RANDOM_SEED)
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  🧬  P10_5 加噪")
    print("=" * 60)
    print()
    print(f"  输入:    {args.input}")
    print(f"  输出:    {args.output}")
    print(f"  p_sub:   {args.p_sub*100:.1f}%")
    print(f"  p_del:   {args.p_del*100:.1f}%")
    print(f"  p_ins:   {args.p_ins*100:.1f}%")
    print(f"  总错误:  {(args.p_sub+args.p_del+args.p_ins)*100:.1f}%")
    print(f"  seed:    {args.seed}")
    print()

    rng = random.Random(args.seed)

    n_reads = 0
    len_before_total = 0
    len_after_total = 0
    len_diff_counter = Counter()  # 跟踪长度变化分布
    t0 = time.time()

    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        for line in fin:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            tag, seq = parts

            noisy_seq = add_noise_to_read(seq, args.p_sub, args.p_del, args.p_ins, rng)

            fout.write(f"{tag}\t{noisy_seq}\n")

            n_reads += 1
            len_before_total += len(seq)
            len_after_total += len(noisy_seq)
            len_diff_counter[len(noisy_seq) - len(seq)] += 1

            if n_reads % 1000000 == 0:
                elapsed = time.time() - t0
                print(f"  已处理: {n_reads:>10,} reads ({elapsed:.1f}s)")

    elapsed = time.time() - t0

    print()
    print(f"  ✅ 完成: {n_reads:,} reads, 耗时 {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print()
    print(f"  长度统计:")
    print(f"    加噪前 avg len: {len_before_total / max(n_reads, 1):.2f}bp")
    print(f"    加噪后 avg len: {len_after_total / max(n_reads, 1):.2f}bp")
    print(f"    长度变化分布 (top 7):")
    for diff, cnt in sorted(len_diff_counter.most_common(7),
                            key=lambda x: x[0]):
        sign = '+' if diff > 0 else ''
        pct = cnt / n_reads * 100
        print(f"      Δlen = {sign}{diff:>3d}:  {cnt:>10,}  ({pct:5.2f}%)")
    print()
    print(f"  下一步:")
    print(f"     python pipeline_p10_5_thin_noisy.py")
    print()


if __name__ == '__main__':
    main()