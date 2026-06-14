#!/usr/bin/env python3
"""
spike_rqs_calibration.py —— 无监督指标 RQS 的可靠性校准
========================================================
背景: 导师指出现有 SR 公式用 #{全部GT} 当分母, 偷看了 GT 个数, 不符合
      无监督/自监督 setting。需要一个只用 reads+labels+consensus 的指标,
      并用 GT 证明它可靠 (GT 只做校准, 不进指标公式)。

无监督指标 RQS (Reconstruction Quality Score):
  对每个簇 (纯无监督, 不看 GT):
    size          = 簇内 read 数
    support_ratio = 簇内 read 中, 到该簇 consensus 归一化 edit < 0.05 的比例
    可信重建      ⟺ size >= 5 且 support_ratio >= 0.5
  RQS              = 可信簇数 / 总簇数
  read_util        = 非 -1 read 数 / 总 read 数        (覆盖轴)
  RQS_combined     = read_util × (可信簇数 / 总簇数)    (综合: 覆盖×质量)

GT 校准 (只为验证 RQS, 不进公式):
  对每个簇用 GT 看它"真的重建对了吗" (consensus == 该簇多数GT的reference)。
  输出混淆矩阵: RQS可信/不可信 × GT正确/错误。
  若 RQS可信 高度集中在 GT正确 -> RQS 是可靠无监督代理。

逐轮跑 R1/R2/R3, 看:
  1. RQS / read_util / RQS_combined 逐轮趋势 (应能复现覆盖率塌陷)
  2. RQS 与真实 SR 的吻合度 (校准混淆矩阵)

依赖: edlib, numpy
"""
import os, sys, glob, re, argparse
from collections import defaultdict, Counter
import numpy as np

import edlib
def ned(a, b):
    if not a or not b: return 1.0
    return edlib.align(a, b, mode="NW", task="distance")['editDistance'] / max(len(a), len(b))

SUPPORT_ED = 0.05
MIN_SIZE   = 5
MIN_SUPPORT= 0.5

# ---------- 复用已验证的对齐工具 ----------
def read_fasta_ordered(path):
    pairs = []; cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                cur = int(line[1:].split()[0])
            else:
                if cur is not None:
                    pairs.append((cur, line.upper())); cur = None
    pairs.sort(key=lambda x: x[0])
    return [s for _, s in pairs]

def build_tag_to_ref(sam_path, ref_fasta):
    sns = []
    with open(sam_path) as f:
        for line in f:
            if line.startswith('@SQ'):
                for fld in line.strip().split('\t'):
                    if fld.startswith('SN:'):
                        sns.append(int(fld[3:])); break
            elif not line.startswith('@'):
                break
    sns.sort()
    refs = read_fasta_ordered(ref_fasta)
    n = min(len(sns), len(refs))
    return {sns[k]: refs[k] for k in range(n)}

def parse_consensus_fasta(path):
    seqs = {}; cur = None; buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cur is not None: seqs[cur] = ''.join(buf).upper()
                m = re.match(r'cluster_(\d+)', line[1:].strip())
                cur = int(m.group(1)) if m else line[1:].strip()
                buf = []
            elif line: buf.append(line)
    if cur is not None: seqs[cur] = ''.join(buf).upper()
    return seqs

def discover_rounds(exp):
    cons = sorted(glob.glob(os.path.join(exp,"results","iter_*_step2","consensus","consensus_*.fasta")))
    out = []
    for c in cons:
        mi = re.search(r'iter_(\d+)_step2', c); mt = re.search(r'consensus_(\d+)\.fasta', c)
        if not mi or not mt: continue
        lab = os.path.join(exp,"04_Iterative_Labels", f"refined_labels_{mt.group(1)}.txt")
        if os.path.exists(lab): out.append((int(mi.group(1)), lab, c))
    out.sort()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment_dir', required=True)
    p.add_argument('--gt_tags', required=True)      # output.txt (序列做key拿GT)
    p.add_argument('--ref_fasta', required=True)    # reads.fasta
    p.add_argument('--sam', required=True)          # mem-se.sam (排序SN对齐)
    p.add_argument('--code_dir', default='/mnt/st_data/liangxinyi/code')
    p.add_argument('--max_reads_per_cluster', type=int, default=40,
                   help='每簇最多抽多少read算support(防大簇拖慢), 0=全用')
    args = p.parse_args()

    if args.code_dir not in sys.path: sys.path.insert(0, args.code_dir)
    from models.step1_data import CloverDataLoader

    print("="*64); print("📂 加载 reads + GT + ref"); print("="*64)
    rounds = discover_rounds(args.experiment_dir)
    dl = CloverDataLoader(args.experiment_dir, labels_path=rounds[-1][1])
    dl.load_gt_tags(args.gt_tags)
    reads = dl.reads
    gt = np.array(dl.gt_labels)
    tag_to_ref = build_tag_to_ref(args.sam, args.ref_fasta)
    print(f"   reads={len(reads)}, GT匹配={int((gt>=0).sum())}, tag->ref={len(tag_to_ref)}")

    print("\n" + "="*64)
    print(f"🔬 逐轮 RQS (无监督) + GT校准   [support_ed<{SUPPORT_ED}, size>={MIN_SIZE}, support>={MIN_SUPPORT}]")
    print("="*64)

    rng = np.random.default_rng(42)
    summary = []
    for ridx, lab_path, cons_path in rounds:
        labels = np.loadtxt(lab_path, dtype=int)
        consensus = parse_consensus_fasta(cons_path)
        # 簇 -> read 索引
        cl_to_reads = defaultdict(list)
        for i, l in enumerate(labels):
            if l >= 0: cl_to_reads[int(l)].append(i)

        n_clu = 0
        n_trust = 0
        n_noise = int((labels < 0).sum())
        read_util = (labels >= 0).mean()
        # 校准混淆: (rqs_trust, gt_correct)
        conf = {('T','C'):0, ('T','W'):0, ('U','C'):0, ('U','W'):0}

        for cid, ridxs in cl_to_reads.items():
            if cid not in consensus:
                continue
            n_clu += 1
            cons_seq = consensus[cid]
            # --- 无监督: support_ratio ---
            use = ridxs
            if args.max_reads_per_cluster and len(ridxs) > args.max_reads_per_cluster:
                use = [ridxs[k] for k in rng.choice(len(ridxs), args.max_reads_per_cluster, replace=False)]
            sup = sum(1 for i in use if ned(reads[i], cons_seq) < SUPPORT_ED)
            support_ratio = sup / max(len(use), 1)
            size = len(ridxs)
            rqs_trust = (size >= MIN_SIZE and support_ratio >= MIN_SUPPORT)
            if rqs_trust: n_trust += 1

            # --- GT校准: 这个簇真的重建对了吗 ---
            gts = [int(gt[i]) for i in ridxs if gt[i] >= 0]
            gt_correct = False
            if gts:
                maj = Counter(gts).most_common(1)[0][0]
                if maj in tag_to_ref:
                    gt_correct = (ned(cons_seq, tag_to_ref[maj]) < 1e-9)
            key = ('T' if rqs_trust else 'U', 'C' if gt_correct else 'W')
            conf[key] += 1

        rqs = n_trust / max(n_clu, 1)
        rqs_combined = read_util * rqs
        # 校准指标
        TC, TW, UC, UW = conf[('T','C')], conf[('T','W')], conf[('U','C')], conf[('U','W')]
        precision = TC / max(TC+TW, 1)   # RQS判可信的, 真对的比例
        recall_q  = TC / max(TC+UC, 1)   # 真对的簇里, 被RQS判可信的比例

        print(f"\n── Round {ridx} ──")
        print(f"   [无监督] 总簇={n_clu}, 可信簇={n_trust}, RQS={rqs:.4f}")
        print(f"            read_util={read_util:.4f} (噪声-1={n_noise}), RQS_combined={rqs_combined:.4f}")
        print(f"   [GT校准] 混淆矩阵 (簇数):")
        print(f"            RQS可信&GT对={TC}   RQS可信&GT错={TW}")
        print(f"            RQS不可信&GT对={UC}  RQS不可信&GT错={UW}")
        print(f"            RQS精度(可信里真对)={precision:.4f}  RQS召回(真对里被判可信)={recall_q:.4f}")
        summary.append((ridx, rqs, read_util, rqs_combined, precision))

    print("\n" + "="*64); print("📊 逐轮总结"); print("="*64)
    print(f"   {'Round':>6}{'RQS':>9}{'read_util':>11}{'RQS_comb':>10}{'RQS精度':>9}")
    for ridx, rqs, ru, rc, prec in summary:
        print(f"   {ridx:>6}{rqs:>9.4f}{ru:>11.4f}{rc:>10.4f}{prec:>9.4f}")
    print("\n   判读:")
    print("   - RQS精度若三轮都≥~0.95 -> RQS是可靠无监督代理, 可当主指标")
    print("   - read_util / RQS_combined 若逐轮降 -> 复现覆盖率塌陷(无需GT即可见)")


if __name__ == '__main__':
    main()