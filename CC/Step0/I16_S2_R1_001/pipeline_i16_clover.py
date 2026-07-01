#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_i16_clover.py  ——  I16 上跑 Clover (实证前缀树在 deletion-dominant 短序列上崩溃)

与 Seq_1D 版的差异: 预处理/打薄/GT标签已由 preprocess_i16.py + build_gt_labels.py 完成,
本脚本直接吃 i16_labeled_sampled.fasta (已采样, 每簇<=30条), 不再做 BWA tag / 打薄。

流程:
  Step1 读 fasta -> 建 idx->(gt_id, seq) -> 写 Clover tag模式输入 (每行 "gt_id seq")
  Step2 按 60nt 真实结构 patch load_config.py -> 跑 Clover
  Step3 解析 Clover 输出 (idx, cid) -> 写 pred 文件 (seq <TAB> cid) 供 eval_clustering_i16.py
  Step4 报告 Clover 自报数 + 提示用统一评估脚本看公平指标

I16 Clover 配置 (按 60nt 真实结构, 给它最好的机会):
  -L 60  -D 8  h_index=e_index=0 (无合成primer)  thd/four_tree_loc=15/30  read_len_min=40
"""
import os, re, sys, subprocess, time, argparse
from collections import defaultdict, Counter

# ============ 配置 ============
BASE_DIR   = "/mnt/st_data/liangxinyi/code/CC/Step0/I16_S2_R1_001"
CLOVER_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/Clover"
SAMPLED_FASTA = os.path.join(BASE_DIR, "i16_labeled_sampled.fasta")
OUT_DIR    = os.path.join(BASE_DIR, "clover_out")

REF_LEN    = 60
# Clover 参数 (60nt 适配)
CLOVER_D       = 8     # end_tree_len: 树深度, 不能超过可用前缀
CLOVER_V       = 3     # 纵向漂移(给deletion漂移容忍)
CLOVER_H       = 3     # 横向漂移阈值
CLOVER_H_INDEX = 0     # I16 无合成 primer
CLOVER_E_INDEX = 0
CLOVER_THD_LOC = 15    # 中段树位置(按60nt比例)
CLOVER_FOUR_LOC= 30
CLOVER_RLMIN   = 40    # read_len_min: deletion拉短, 放宽
# =============================


def banner(t):
    print(f"\n{'-'*60}\n  {t}\n{'-'*60}\n")


def step1_write_input(fasta, clover_input, idx_map_out, no_tag):
    banner("Step 1  读已采样 fasta -> 写 Clover 输入")
    idx_map = {}     # idx(int) -> (gt_id, seq)
    n = 0
    with open(fasta) as fin, open(clover_input, "w") as fout:
        gid = None
        for line in fin:
            line = line.rstrip("\n")
            if line.startswith(">"):
                m = re.search(r"gt=(\d+)", line)
                gid = m.group(1) if m else "-1"
            elif line and gid is not None:
                n += 1
                idx_map[n] = (gid, line)
                if no_tag:
                    # --no-tag(Virtual_mode=False): 第一列=唯一行号idx,
                    # Clover才会把(idx,cid)写进index_list -> 逐read簇分配
                    fout.write(f"{n} {line}\n")
                else:
                    # tag模式: 第一列=gt_id, Clover自报coverage/accuracy
                    fout.write(f"{gid} {line}\n")
                gid = None
    print(f"  reads: {n:,}  模式: {'--no-tag(逐read簇分配)' if no_tag else 'tag(自报统计)'}")
    print(f"  Clover输入: {clover_input}")
    # 存 idx_map 供解析阶段(seq回查)
    import pickle
    with open(idx_map_out, "wb") as f:
        pickle.dump(idx_map, f)
    n_gt = len(set(v[0] for v in idx_map.values()))
    print(f"  GT簇(tag)数: {n_gt:,}")
    return idx_map, n_gt


def step2_run_clover(clover_input, out_base, n_gt, no_tag):
    banner("Step 2  配置并运行 Clover (60nt)")
    cfg = os.path.join(CLOVER_DIR, "clover", "load_config.py")
    with open(cfg) as f:
        content = f.read()
    patches = {
        "h_index_nums":  CLOVER_H_INDEX,
        "e_index_nums":  CLOVER_E_INDEX,
        "thd_tree_loc":  CLOVER_THD_LOC,
        "four_tree_loc": CLOVER_FOUR_LOC,
        "read_len_min":  CLOVER_RLMIN,
    }
    for k, v in patches.items():
        content = re.sub(rf'"{k}"\s*:\s*\d+', f'"{k}" : {v}', content)
    with open(cfg, "w") as f:
        f.write(content)
    print("  已patch load_config.py:")
    for k, v in patches.items():
        print(f"    {k:16s} = {v}")

    cmd = [sys.executable, "-m", "clover.main",
           "-I", clover_input, "-O", out_base,
           "-L", str(REF_LEN), "-P", "0",
           "-D", str(CLOVER_D), "-V", str(CLOVER_V), "-H", str(CLOVER_H)]
    if no_tag:
        cmd.append("--no-tag")   # Virtual_mode=False -> 输出(idx,cid)到index_list
    else:
        cmd += ["-T", str(n_gt)] # tag模式: 自报coverage/redundancy
    env = os.environ.copy()
    env["PYTHONPATH"] = CLOVER_DIR + os.pathsep + env.get("PYTHONPATH", "")
    mode = "--no-tag(逐read簇分配)" if no_tag else f"-T {n_gt}(自报统计)"
    print(f"\n  cmd: -L {REF_LEN} -D {CLOVER_D} -V {CLOVER_V} -H {CLOVER_H} {mode}")
    print("  运行中...\n")
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env, cwd=CLOVER_DIR)
    print(f"\n  Clover完成, 耗时 {time.time()-t0:.1f}s")

    # tag模式下输出文件: 找 -O 指定的
    out_txt = out_base + ".txt"
    if not os.path.exists(out_txt):
        alt = os.path.join(CLOVER_DIR, os.path.basename(out_base) + ".txt")
        if os.path.exists(alt):
            os.rename(alt, out_txt)
    return out_txt if os.path.exists(out_txt) else None


def step3_parse_and_write_pred(out_txt, idx_map, pred_path):
    banner("Step 3  解析 Clover 输出 -> 写 pred (seq<TAB>cid)")
    if not out_txt or not os.path.exists(out_txt):
        print("  [警告] 未找到 Clover 簇分配输出文件。")
        print("  tag模式下 Clover 主要打印自报统计; 若需逐read簇分配,")
        print("  需用 --no-tag 模式输出 index_list。见下方备注。")
        return None
    with open(out_txt) as f:
        content = f.read()
    pairs = re.findall(r"\('?(\d+)'?,\s*'?(\d+)'?\)", content)
    cid_of = {}
    for idx_str, cid_str in pairs:
        idx = int(idx_str)
        if idx in idx_map:
            cid_of[idx] = cid_str
    # 写 pred: seq <TAB> cid
    with open(pred_path, "w") as f:
        for idx, (gid, seq) in idx_map.items():
            cid = cid_of.get(idx)
            if cid is not None:
                f.write(f"{seq}\t{cid}\n")
    n_clust = len(set(cid_of.values()))
    print(f"  解析 (idx,cid): {len(pairs):,}")
    print(f"  预测簇数: {n_clust:,}  (GT={len(set(v[0] for v in idx_map.values())):,})")
    print(f"  pred文件: {pred_path}")
    return pred_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=SAMPLED_FASTA)
    ap.add_argument("--no_tag", action="store_true",
                    help="用--no-tag模式跑(输出逐read簇分配, 便于公平评估)")
    ap.add_argument("--D", type=int, default=None, help="覆盖树深度CLOVER_D")
    ap.add_argument("--V", type=int, default=None, help="覆盖纵向漂移CLOVER_V")
    ap.add_argument("--H", type=int, default=None, help="覆盖横向漂移CLOVER_H")
    ap.add_argument("--out_suffix", default="", help="pred文件后缀,扫参数时区分")
    args = ap.parse_args()

    global CLOVER_D, CLOVER_V, CLOVER_H
    if args.D is not None: CLOVER_D = args.D
    if args.V is not None: CLOVER_V = args.V
    if args.H is not None: CLOVER_H = args.H

    os.makedirs(OUT_DIR, exist_ok=True)
    clover_input = os.path.join(OUT_DIR, "01_clover_input.txt")
    idx_map_pkl  = os.path.join(OUT_DIR, "01_idx_map.pkl")
    out_base     = os.path.join(OUT_DIR, "02_clover_result")
    pred_path    = os.path.join(OUT_DIR, f"pred_clover{args.out_suffix}.txt")

    print("="*60)
    print("  I16 x Clover  (实证前缀树崩溃)")
    print("="*60)
    print(f"  fasta: {args.fasta}")
    print(f"  配置: L={REF_LEN} D={CLOVER_D} h/e_index=0 (无primer)")

    idx_map, n_gt = step1_write_input(args.fasta, clover_input, idx_map_pkl, args.no_tag)
    out_txt = step2_run_clover(clover_input, out_base, n_gt, args.no_tag)
    pred = step3_parse_and_write_pred(out_txt, idx_map, pred_path)

    banner("完成")
    print("  Clover 自报的 Accuracy 只看簇内纯度, 不看过度切分, 会虚高!")
    print("  跑统一评估看公平指标:")
    if pred:
        print(f"    python eval_clustering_i16.py --pred {pred_path} \\")
        print(f"      --gt {os.path.join(BASE_DIR,'i16_gt_labels.txt')} --name Clover")
    else:
        print("    [需逐read簇分配] 重跑加 --no_tag, 或检查 Clover -O 输出。")
        print("    Clover tag模式默认只打印统计、不落 index_list 文件;")
        print("    若 pred 为空, 用 --no_tag 重跑本脚本。")


if __name__ == "__main__":
    main()