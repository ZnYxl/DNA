#!/usr/bin/env python3
"""
pipeline_stairloop_gradhc.py
============================
StairLoop CA GradHC baseline 版（无先验二向 canonical 归一化）

与之前差异：
  - Step0 不再用 BWA 先验救回、不和 ref 比。每条 read 取
    canonical = min(seq, revcomp(seq)) 字典序，纯无监督方向归一化。
  - GT (read2ref_q60.tsv) 仅用于评测 Purity/Coverage，不进聚类决策。
  - 评测用 read_id 关联 GT；canonical 重复用 pool.pop() 消费机制。

用法:
    python pipeline_stairloop_gradhc.py --min_cluster_size 5 --tag v5
    python pipeline_stairloop_gradhc.py --min_cluster_size 3 --tag v3
    python pipeline_stairloop_gradhc.py --min_cluster_size 5 --tag v5 --skip_gradhc
"""
import os, sys, glob, gzip, time, argparse
from collections import defaultdict, Counter

BASE       = '/mnt/st_data/liangxinyi/code/CC/Step0/stairloop'
FASTQ      = os.path.join(BASE, 'sequencing reads', 'filter_sequences3.fastq')
GT_TSV     = os.path.join(BASE, 'read2ref_q60.tsv')
REF        = os.path.join(BASE, 'test_encode.fasta')
GRADHC_DIR = '/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/GradHC'

REF_LEN   = 130
LEN_MIN, LEN_MAX = 122, 138
N_GT_TAGS = 45360

GRADHC_Q, GRADHC_K, GRADHC_M, GRADHC_L = 8, 3, 40, 32
GRADHC_DIST = 12
GRADHC_SD_HIGH = 0.40

_COMP = str.maketrans('ACGTacgt', 'TGCATGCA')
def revcomp(s): return s.translate(_COMP)[::-1]
def canonical(s):
    """无先验二向归一化：正向 vs 反向互补，取字典序最小。"""
    rc = revcomp(s)
    return s if s <= rc else rc
def banner(t): print(f"\n{'-'*60}\n  {t}\n{'-'*60}")


def load_gt():
    g = {}
    with open(GT_TSV) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2: g[p[0]] = p[1]
    return g


# ============================================================ Step0 无先验 canonical
def step0_process(gt):
    banner("Step 0  读 fastq + 无先验二向 canonical 归一化")
    op = gzip.open(FASTQ, 'rt') if FASTQ.endswith('.gz') else open(FASTQ)
    # 按 GT ref 分组（仅用于过滤 size 和评测，不进聚类）
    by_ref = defaultdict(list)   # ref_id -> [canonical_seq, ...]
    total = no_gt = len_drop = 0
    with op as f:
        while True:
            h = f.readline()
            if not h: break
            seq = f.readline().strip(); f.readline(); f.readline()
            total += 1
            if not (LEN_MIN <= len(seq) <= LEN_MAX) or 'N' in seq.upper():
                len_drop += 1; continue
            rf = gt.get(h[1:].split()[0])
            if rf is None:
                no_gt += 1; continue
            by_ref[rf].append(canonical(seq))   # canonical，无先验
    print(f"  总 reads:     {total:,}")
    print(f"  长度/N 剔除:  {len_drop:,}")
    print(f"  无 GT 跳过:   {no_gt:,}")
    print(f"  有效 reads:   {sum(len(v) for v in by_ref.values()):,}")
    print(f"  覆盖 ref:     {len(by_ref):,} / {N_GT_TAGS:,}")
    return by_ref


# ============================================================ Step1 过滤+写GradHC输入
def step1_write_gradhc_input(by_ref, min_size, out_dir):
    banner(f"Step 1  按 GT 簇过滤 (size>={min_size}) + 写 GradHC 输入(单块无监督)")
    kept = {rf: s for rf, s in by_ref.items() if len(s) >= min_size}
    gradhc_in  = os.path.join(out_dir, '01_gradhc_input.txt')
    gradhc_tag = os.path.join(out_dir, '01_gradhc_tag_input.txt')
    cleaned = [(rf, s) for rf, seqs in kept.items() for s in seqs]

    with open(gradhc_in, 'w', newline='\n') as f:
        f.write('A' * REF_LEN + '\n')
        f.write('*' * 29 + '\n')
        for rf, read in cleaned:
            f.write(read + '\n')
        f.write('\n\n')

    read_to_tags = defaultdict(list)
    for rf, read in cleaned:
        read_to_tags[read].append(rf)
    with open(gradhc_tag, 'w') as f:
        for rf, read in cleaned:
            f.write(f"{rf}\t{read}\n")

    print(f"  保留簇: {len(kept):,}   reads: {len(cleaned):,}")
    print(f"  GradHC 输入: {gradhc_in}")
    return gradhc_in, gradhc_tag, read_to_tags


# ============================================================ Step2 GradHC
def step2_run_gradhc(gradhc_in):
    banner("Step 2  运行 GradHC")
    results_dir = os.path.join(GRADHC_DIR, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    pattern = os.path.join(results_dir, os.path.basename(gradhc_in) + '_*.clustering_results')
    for old in glob.glob(pattern):
        os.remove(old); print(f"  清除旧结果: {os.path.basename(old)}")

    prev = os.getcwd()
    os.chdir(GRADHC_DIR)
    if GRADHC_DIR not in sys.path: sys.path.insert(0, GRADHC_DIR)
    from GradHC_clustering import GradHCBasedCluster

    class GradHCSdHigh(GradHCBasedCluster):
        _SD_HIGH = GRADHC_SD_HIGH
        def clustering_given_chunk(self, chunk_rep, sd_high=None, sd_low=0.28):
            if sd_high is None: sd_high = self._SD_HIGH
            return super().clustering_given_chunk(chunk_rep, sd_high=sd_high, sd_low=sd_low)
        def final_clustering(self, sd_high=None, sd_low=0.22, low_work_rate=0.005,
                             high_work_rate=0.03, rounds_before_refresh=8, min_rounds=300):
            if sd_high is None: sd_high = self._SD_HIGH
            return super().final_clustering(sd_high=sd_high, sd_low=sd_low,
                low_work_rate=low_work_rate, high_work_rate=high_work_rate,
                rounds_before_refresh=rounds_before_refresh, min_rounds=min_rounds)

    print(f"  q={GRADHC_Q} k={GRADHC_K} m={GRADHC_M} L={GRADHC_L} dist={GRADHC_DIST} sd_high={GRADHC_SD_HIGH}")
    print(f"  运行中（较慢，请耐心）...\n")
    t0 = time.time()
    try:
        c = GradHCSdHigh(gradhc_in, q=GRADHC_Q, k=GRADHC_K, m=GRADHC_M, L=GRADHC_L,
                         distance_threshold=GRADHC_DIST, serial=True, export=True)
        c.run()
    finally:
        os.chdir(prev)
    print(f"  耗时 {time.time()-t0:.1f}s")
    m = glob.glob(pattern)
    if not m: raise FileNotFoundError(pattern)
    return max(m, key=os.path.getmtime)


# ============================================================ Step3 解析
def step3_parse_gradhc(out_path):
    banner("Step 3  解析 GradHC 输出")
    clusters, cur, expect_rep = [], None, True
    with open(out_path) as f:
        for raw in f:
            line = raw.strip()
            if line == '':
                if cur: clusters.append(cur)
                cur, expect_rep = None, True
                continue
            if line[0] == '*':
                expect_rep = False; cur = []
                continue
            if expect_rep:
                cur = None; expect_rep = True
                continue
            if cur is None: cur = []
            cur.append(line)
    if cur: clusters.append(cur)
    cid = {i: r for i, r in enumerate(clusters)}
    sizes = [len(v) for v in cid.values()]
    print(f"  簇数: {len(cid):,}   reads: {sum(sizes):,}")
    if sizes: print(f"  max={max(sizes)}, med={sorted(sizes)[len(sizes)//2]}, min={min(sizes)}")
    return cid

def step3_5_filter(cid, min_reads):
    before = len(cid)
    cid = {c: r for c, r in cid.items() if len(r) >= min_reads}
    print(f"  小簇过滤(<{min_reads}): {before:,} -> {len(cid):,}")
    return cid


# ============================================================ Step4 评测
def step4_stats(cid, read_to_tags, out_dir):
    banner("Step 4  Purity / Coverage")
    pool = {r: list(t) for r, t in read_to_tags.items()}
    total = pure = 0
    covered = set(); sizes = []
    for reads in cid.values():
        tags = []
        for r in reads:
            cand = pool.get(r)
            if cand: tags.append(cand.pop())
        if not tags: continue
        sizes.append(len(reads)); total += len(reads)
        m, c = Counter(tags).most_common(1)[0]
        pure += c; covered.add(m)
    sizes.sort(reverse=True)
    purity = pure / max(total, 1); coverage = len(covered) / N_GT_TAGS
    print(f"  簇数:     {len(cid):,}")
    print(f"  reads:    {total:,}")
    print(f"  Purity:   {purity*100:.2f}%")
    print(f"  Coverage: {coverage*100:.2f}%  ({len(covered)}/{N_GT_TAGS})")
    if sizes: print(f"  max={sizes[0]}, median={sizes[len(sizes)//2]}")
    with open(os.path.join(out_dir, '03_stats.txt'), 'w') as f:
        f.write(f"clusters: {len(cid)}\nreads: {total}\nPurity: {purity*100:.2f}%\nCoverage: {coverage*100:.2f}%\n")
    return purity, coverage


# ============================================================ Step5+6 MV
def majority_vote(reads, ref_len):
    vote = [Counter() for _ in range(ref_len)]
    for r in reads:
        for p in range(min(len(r), ref_len)):
            b = r[p].upper()
            if b in 'ACGT': vote[p][b] += 1
    res, last = [], 'A'
    for p in range(ref_len):
        if vote[p]: last = vote[p].most_common(1)[0][0]
        res.append(last)
    return ''.join(res)

def step56_write(cid, out_dir):
    banner("Step 5+6  MV -> read.txt + ref.txt")
    fed = os.path.join(out_dir, '04_FedDNA_In'); os.makedirs(fed, exist_ok=True)
    read_p, ref_p = os.path.join(fed, 'read.txt'), os.path.join(fed, 'ref.txt')
    SEP = "=====分隔符=====\n"; nc = nr = 0
    with open(read_p, 'w') as fr, open(ref_p, 'w') as ff:
        for c in sorted(cid.keys()):
            reads = cid[c]
            if not reads: continue
            for r in reads: fr.write(r + '\n')
            fr.write(SEP)
            ff.write(majority_vote(reads, REF_LEN) + '\n')
            nc += 1; nr += len(reads)
    print(f"  簇 {nc:,}, reads {nr:,}")


# ============================================================ Step7 部署
def step7_deploy(gradhc_tag, out_dir, tag):
    banner("Step 7  部署 SSI-EC")
    exp = os.path.join(out_dir, 'exp')
    os.makedirs(os.path.join(exp, '03_FedDNA_In'), exist_ok=True)
    import shutil
    for fn in ['read.txt', 'ref.txt']:
        shutil.copy2(os.path.join(out_dir, '04_FedDNA_In', fn),
                     os.path.join(exp, '03_FedDNA_In', fn))
    gt_tags = os.path.join(exp, 'stairloop_tags_reads.txt')
    shutil.copy2(gradhc_tag, gt_tags)
    gt_refs = os.path.join(exp, 'stairloop_refs.txt')
    with open(REF) as fin, open(gt_refs, 'w') as fout:
        for line in fin:
            if not line.startswith('>'): fout.write(line)
    print(f"  GT tags: {gt_tags}\n  GT refs: {gt_refs}")
    print(f"\n  >> 运行 SSI-EC:")
    print(f"     cd /mnt/st_data/liangxinyi/code/models")
    print(f"     python main_loop.py --experiment_dir {exp}/ \\")
    print(f"       --feddna_checkpoint /mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth \\")
    print(f"       --max_iterations 3 --max_length {REF_LEN} --cl_mode ours \\")
    print(f"       --gt_tags_file {gt_tags} --gt_refs_file {gt_refs} \\")
    print(f"       2>&1 | tee stairloop_gradhc_{tag}.log")


# ============================================================ Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min_cluster_size', type=int, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--skip_gradhc', action='store_true')
    ap.add_argument('--gradhc_result', default=None)
    ap.add_argument('--min_reads', type=int, default=2)
    args = ap.parse_args()

    out_dir = os.path.join(BASE, f'out_gradhc_{args.tag}')
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print("="*60)
    print(f"  StairLoop CA GradHC [{args.tag}] size>={args.min_cluster_size} (无先验canonical)")
    print("="*60)

    gt = load_gt()
    print(f"  gt={len(gt):,}")
    by_ref = step0_process(gt)
    gradhc_in, gradhc_tag, read_to_tags = step1_write_gradhc_input(by_ref, args.min_cluster_size, out_dir)
    del by_ref

    if not args.skip_gradhc:
        result = step2_run_gradhc(gradhc_in)
    else:
        if args.gradhc_result:
            result = args.gradhc_result
        else:
            pat = os.path.join(GRADHC_DIR, 'Results', os.path.basename(gradhc_in) + '_*.clustering_results')
            m = glob.glob(pat)
            if not m: raise FileNotFoundError(pat)
            result = max(m, key=os.path.getmtime)
        banner("Step 2  跳过 GradHC"); print(f"  使用: {result}")

    cid = step3_parse_gradhc(result)
    cid = step3_5_filter(cid, args.min_reads)
    step4_stats(cid, read_to_tags, out_dir)
    step56_write(cid, out_dir)
    step7_deploy(gradhc_tag, out_dir, args.tag)
    print(f"\n  总耗时 {time.time()-t0:.1f}s")
    print("="*60)


if __name__ == '__main__':
    main()