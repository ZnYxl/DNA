#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_i16.py  ——  I16 (Antkowiak 2020) reads 预处理
Clover / GradHC 共用的清洗步骤,不涉及 GT。

做三件事(全部温和,避免误伤 deletion 主导的 payload):
  1) 切 3' 端 Illumina 接头 (AGATCGGAAGAGC / CTCTTCCGATCT 反向部分)
  2) 切 3' 末尾的 poly-G 尾 (库制备 Swift kit 加的 G 富集尾, 论文明说 80% G)
  3) 长度过滤: 保留 [LEN_MIN, LEN_MAX] nt 的 read

输入:  I16_S2_R1_001.fastq  (29M reads, 流式读, 不全载入内存)
输出:  i16_clean.fasta       (每条带原始 read 序号作 id)

设计要点:
  - 不切 GC 混合区(避免误伤真实 payload, spike 验证过强切只把中位ED 15->13,收益极小)
  - 不做反向互补(spike 确认正向, motif 正向命中远多于反向)
  - 纯标准库, 零外部依赖(cutadapt 未安装)
"""
import sys, os

# ============ 配置 ============
FQ_PATH   = "I16_S2_R1_001.fastq"
OUT_FASTA = "i16_clean.fasta"
LEN_MIN   = 40          # 短于此判为空读/接头二聚体, 丢弃
LEN_MAX   = 75          # 长于此判为接头读穿严重, 丢弃
ILLUMINA  = "AGATCGGAAGAGC"   # Illumina 通用接头(3'端读穿)
ILLUMINA2 = "CTCTTCCGATCT"    # 接头另一段(spike 见 read 前缀高频出现)
MIN_POLYG = 4           # 3'末尾连续 >=4 个 G 才剥(避免误伤真实G)
PROBE_LENS = (10, 8)    # 接头前缀匹配探针长度(容忍接头自身有错)
# =============================


def trim_illumina(s):
    """切 3' 端 Illumina 接头: 接头前缀出现即从那里截断。"""
    cut = len(s)
    for adp in (ILLUMINA, ILLUMINA2):
        for pl in PROBE_LENS:
            i = s.find(adp[:pl])
            if i >= 0:
                cut = min(cut, i)
                break
    return s[:cut]


def trim_polyg_tail(s):
    """剥 3' 末尾连续的 poly-G (>=MIN_POLYG 才剥)。"""
    j = len(s)
    while j > 0 and s[j-1] == 'G':
        j -= 1
    run = len(s) - j
    if run >= MIN_POLYG:
        return s[:j]
    return s


def main():
    if not os.path.exists(FQ_PATH):
        sys.exit(f"[ERR] 找不到输入 {FQ_PATH}")

    n_total = n_kept = n_short = n_long = 0
    with open(FQ_PATH) as fin, open(OUT_FASTA, "w") as fout:
        while True:
            h = fin.readline()
            if not h:
                break
            seq = fin.readline().strip()
            fin.readline()        # '+'
            fin.readline()        # qual
            n_total += 1

            seq = trim_illumina(seq)
            seq = trim_polyg_tail(seq)

            L = len(seq)
            if L < LEN_MIN:
                n_short += 1
                continue
            if L > LEN_MAX:
                n_long += 1
                continue

            fout.write(f">r{n_total}\n{seq}\n")
            n_kept += 1

            if n_total % 2_000_000 == 0:
                print(f"  ...已处理 {n_total:,}  保留 {n_kept:,}", flush=True)

    print("\n========== 预处理完成 ==========")
    print(f"总 reads      : {n_total:,}")
    print(f"保留          : {n_kept:,}  ({n_kept/n_total:.1%})")
    print(f"丢弃(过短)    : {n_short:,}")
    print(f"丢弃(过长)    : {n_long:,}")
    print(f"输出          : {OUT_FASTA}")


if __name__ == "__main__":
    main()