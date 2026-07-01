#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_i16_gradhc.py  ——  I16 上跑 GradHC (实证 MinHash 锚点对 deletion 鲁棒)

基于 Seq_1D 的 pipeline_seq1d_gradhc.py 模板改造。差异:
  - 预处理/打薄/GT标签已由 preprocess_i16.py + build_gt_labels.py 完成,
    本脚本直接吃 i16_labeled_sampled.fasta (已采样, 每簇<=30条)
  - q=4 (60nt 实测最优: 同源/异源 q-gram Jaccard 间隙 0.234 最大; Seq_1D 的 q=8 在60nt会塌缩)
  - REF_LEN=60, 长度过滤放宽到 [40,75] (deletion 拉短)
  - 输出 pred_gradhc.txt (seq<TAB>cid) 供 eval_clustering_i16.py 统一评估

GradHC 调用方式与 Seq_1D 版完全一致:
  - import GradHCBasedCluster, 子类多态覆盖 sd_high (不改源码)
  - os.chdir(GRADHC_DIR) 必须在 import 之前 (WORKING_DIR 在 import 时固定)
"""
import os, re, sys, glob, time, argparse
from collections import defaultdict, Counter

# ============ 配置 ============
BASE_DIR    = "/mnt/st_data/liangxinyi/code/CC/Step0/I16_S2_R1_001"
# GradHC 仓库根目录: 复用 Seq_1D 那份(含 GradHC_clustering.py)
GRADHC_DIR  = "/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/GradHC"
SAMPLED_FASTA = os.path.join(BASE_DIR, "i16_labeled_sampled.fasta")
OUT_DIR     = os.path.join(BASE_DIR, "gradhc_out")

REF_LEN     = 60
LEN_MIN     = 40        # deletion 拉短, 放宽
LEN_MAX     = 75

# GradHC 参数 (60nt 适配)
GRADHC_Q    = 4         # ← 60nt 实测最优(非 Seq_1D 的 8); 同源/异源间隙 0.234 最大
GRADHC_K    = 3
GRADHC_M    = 40
GRADHC_L    = 32
GRADHC_DIST = 12
GRADHC_SD_HIGH = None   # None=用GradHC默认(final=0.25/chunk=0.32); 若过度合并再调
# =============================


def banner(t):
    print(f"\n{'-'*60}\n  {t}\n{'-'*60}\n")


def step1_read_fasta(fasta):
    """读已采样 fasta -> [(gt_id, seq), ...], 同时建 seq->gt_id (评估用,但评估走独立GT文件)"""
    banner("Step 1  读已采样 fasta")
    reads = []
    with open(fasta) as f:
        gid = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                m = re.search(r"gt=(\d+)", line)
                gid = m.group(1) if m else "-1"
            elif line and gid is not None:
                if LEN_MIN <= len(line) <= LEN_MAX:
                    reads.append((gid, line))
                gid = None
    n_gt = len(set(g for g, _ in reads))
    print(f"  reads(长度{LEN_MIN}-{LEN_MAX}): {len(reads):,}")
    print(f"  GT簇数: {n_gt:,}")
    return reads


def step2_write_gradhc_input(reads, gradhc_input):
    """
    GradHC 输入格式(分块): rep行 + ***** + reads + 双空行。
    无监督: 全部 reads 放进同一块, rep 用占位串(不泄露GT)。
    """
    banner("Step 2  写 GradHC 输入(单块,无监督)")
    placeholder_rep = "A" * REF_LEN
    with open(gradhc_input, "w", newline="\n") as f:
        f.write(placeholder_rep + "\n")
        f.write("*" * 29 + "\n")
        for gid, read in reads:
            f.write(read + "\n")
        f.write("\n\n")
    print(f"  GradHC输入: {gradhc_input}  ({len(reads):,} reads)")


def step3_run_gradhc(gradhc_input):
    banner("Step 3  运行 GradHC")
    results_dir = os.path.join(GRADHC_DIR, "Results")
    os.makedirs(results_dir, exist_ok=True)
    input_base = os.path.basename(gradhc_input)
    old_pattern = os.path.join(results_dir, input_base + "_*.clustering_results")
    for old in glob.glob(old_pattern):
        os.remove(old)

    # WORKING_DIR 在 import 时固定 -> 必须先 chdir 再 import
    prev_cwd = os.getcwd()
    os.chdir(GRADHC_DIR)
    if GRADHC_DIR not in sys.path:
        sys.path.insert(0, GRADHC_DIR)
    from GradHC_clustering import GradHCBasedCluster

    # 子类多态覆盖 sd_high(不改源码); SD_HIGH=None 时走 GradHC 默认
    SD = GRADHC_SD_HIGH
    class GradHCTuned(GradHCBasedCluster):
        def clustering_given_chunk(self, chunk_rep, sd_high=None, sd_low=0.28):
            if SD is not None and sd_high is None:
                sd_high = SD
            return super().clustering_given_chunk(chunk_rep, sd_high=sd_high, sd_low=sd_low)
        def final_clustering(self, sd_high=None, sd_low=0.22, low_work_rate=0.005,
                             high_work_rate=0.03, rounds_before_refresh=8, min_rounds=300):
            if SD is not None and sd_high is None:
                sd_high = SD
            return super().final_clustering(
                sd_high=sd_high, sd_low=sd_low,
                low_work_rate=low_work_rate, high_work_rate=high_work_rate,
                rounds_before_refresh=rounds_before_refresh, min_rounds=min_rounds)

    print(f"  GradHC: q={GRADHC_Q} k={GRADHC_K} m={GRADHC_M} L={GRADHC_L} dist={GRADHC_DIST} "
          f"sd_high={'默认' if SD is None else SD}")
    print(f"  运行中(数十万read,较慢,请耐心)...\n")
    t0 = time.time()
    try:
        cluster = GradHCTuned(
            gradhc_input,
            q=GRADHC_Q, k=GRADHC_K, m=GRADHC_M, L=GRADHC_L,
            distance_threshold=GRADHC_DIST,
            serial=True, export=True,
        )
        cluster.run()
    finally:
        os.chdir(prev_cwd)
    print(f"\n  GradHC完成, 耗时 {time.time()-t0:.1f}s")

    matches = glob.glob(old_pattern)
    if not matches:
        raise FileNotFoundError(f"GradHC输出未找到: {old_pattern}")
    return max(matches, key=os.path.getmtime)


def step4_parse_and_write_pred(result_path, pred_path):
    """
    GradHC 输出每块: rep行 / ***** / reads... / 双空行。
    每块 reads 构成一个簇。写 pred: seq<TAB>cid。
    """
    banner("Step 4  解析 GradHC 输出 -> 写 pred")
    clusters = []
    cur = None
    expect_rep = True
    with open(result_path) as f:
        for raw in f:
            line = raw.strip()
            if line == "":
                if cur:
                    clusters.append(cur)
                cur = None
                expect_rep = True
                continue
            if line[0] == "*":
                expect_rep = False
                cur = []
                continue
            if expect_rep:
                cur = None
                expect_rep = True
                continue
            if cur is None:
                cur = []
            cur.append(line)
    if cur:
        clusters.append(cur)

    with open(pred_path, "w") as f:
        for cid, reads in enumerate(clusters):
            for seq in reads:
                f.write(f"{seq}\t{cid}\n")

    sizes = [len(c) for c in clusters]
    print(f"  预测簇数: {len(clusters):,}")
    if sizes:
        print(f"  簇大小 max={max(sizes)} med={sorted(sizes)[len(sizes)//2]} min={min(sizes)}")
    print(f"  pred文件: {pred_path}")
    return pred_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=SAMPLED_FASTA)
    ap.add_argument("--q", type=int, default=None, help="覆盖GRADHC_Q")
    ap.add_argument("--sd_high", type=float, default=None, help="覆盖sd_high(过度合并时调)")
    ap.add_argument("--out_suffix", default="")
    ap.add_argument("--skip_gradhc", action="store_true", help="复用已有结果只重新评估")
    args = ap.parse_args()

    global GRADHC_Q, GRADHC_SD_HIGH
    if args.q is not None: GRADHC_Q = args.q
    if args.sd_high is not None: GRADHC_SD_HIGH = args.sd_high

    os.makedirs(OUT_DIR, exist_ok=True)
    gradhc_input = os.path.join(OUT_DIR, "01_gradhc_input.txt")
    pred_path    = os.path.join(OUT_DIR, f"pred_gradhc{args.out_suffix}.txt")

    print("="*60)
    print("  I16 x GradHC  (实证 MinHash 锚点对 deletion 鲁棒)")
    print("="*60)
    print(f"  fasta: {args.fasta}")
    print(f"  q={GRADHC_Q} (60nt最优, 非Seq_1D的8)")

    reads = step1_read_fasta(args.fasta)

    if not args.skip_gradhc:
        step2_write_gradhc_input(reads, gradhc_input)
        result = step3_run_gradhc(gradhc_input)
    else:
        banner("跳过 GradHC, 复用已有结果")
        results_dir = os.path.join(GRADHC_DIR, "Results")
        pattern = os.path.join(results_dir, os.path.basename(gradhc_input) + "_*.clustering_results")
        matches = glob.glob(pattern)
        if not matches:
            sys.exit(f"未找到已有结果: {pattern}")
        result = max(matches, key=os.path.getmtime)
        print(f"  使用: {result}")

    pred = step4_parse_and_write_pred(result, pred_path)

    banner("完成")
    print("  跑统一评估(对齐GradHC论文口径):")
    print(f"    python eval_clustering_i16.py --pred {pred} \\")
    print(f"      --gt {os.path.join(BASE_DIR,'i16_gt_labels.txt')} --name GradHC")


if __name__ == "__main__":
    main()