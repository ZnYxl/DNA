#!/usr/bin/env python3
"""
pipeline_stairloop.py
=====================
StairLoop CA 全流程（模仿 Seq_1D pipeline）：
  reads → 方向归一化(翻转救回离群read) → 按GT簇过滤 → Clover
        → 解析 → Purity/Coverage → MV写FedDNA输入 → 部署SSI-EC

两版实验(按 GT 簇过滤):
    python pipeline_stairloop.py --min_cluster_size 5 --tag v5
    python pipeline_stairloop.py --min_cluster_size 3 --tag v3
    python pipeline_stairloop.py --min_cluster_size 5 --tag v5 --skip_clover
"""
import os, re, sys, gzip, time, subprocess, argparse
from collections import defaultdict, Counter
import edlib

# ============================================================ 配置
BASE       = '/mnt/st_data/liangxinyi/code/CC/Step0/stairloop'
FASTQ      = os.path.join(BASE, 'sequencing reads', 'filter_sequences3.fastq')
GT_TSV     = os.path.join(BASE, 'read2ref_q60.tsv')
REF        = os.path.join(BASE, 'test_encode.fasta')
CLOVER_DIR = '/mnt/st_data/liangxinyi/code/CC/Step0/Clover'

REF_LEN   = 130
LEN_MIN, LEN_MAX = 122, 138
N_GT_TAGS = 45360
NOISE_ED  = 0.10

CLOVER_TREE_DEPTH, CLOVER_V_DRIFT, CLOVER_H_DRIFT = 20, 3, 3
CLOVER_H_INDEX, CLOVER_E_INDEX = 0, 0

_COMP = str.maketrans('ACGTacgt', 'TGCATGCA')
def revcomp(s): return s.translate(_COMP)[::-1]
def ned(a, b): return edlib.align(a, b, task='distance')['editDistance'] / len(b)
def banner(t): print(f"\n{'-'*60}\n  {t}\n{'-'*60}")


# ============================================================ 加载
def load_ref():
    d, rid = {}, None
    with open(REF) as f:
        for line in f:
            if line.startswith('>'): rid = line[1:].strip()
            elif rid: d[rid] = line.strip(); rid = None
    return d

def load_gt():
    g = {}
    with open(GT_TSV) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2: g[p[0]] = p[1]
    return g


# ============================================================ Step0 方向归一化
def step0_process(ref, gt):
    banner("Step 0  读 fastq + 方向归一化(翻转救回离群 read)")
    op = gzip.open(FASTQ, 'rt') if FASTQ.endswith('.gz') else open(FASTQ)
    by_ref = defaultdict(list)
    total = no_gt = len_drop = flipped = noise_drop = 0
    with op as f:
        while True:
            h = f.readline()
            if not h: break
            seq = f.readline().strip(); f.readline(); f.readline()
            total += 1
            if not (LEN_MIN <= len(seq) <= LEN_MAX) or 'N' in seq.upper():
                len_drop += 1; continue
            rf = gt.get(h[1:].split()[0])
            if rf is None or rf not in ref:
                no_gt += 1; continue
            rseq = ref[rf]
            if ned(seq, rseq) <= NOISE_ED:
                by_ref[rf].append(seq)
            else:
                rc = revcomp(seq)
                if ned(rc, rseq) <= NOISE_ED:
                    by_ref[rf].append(rc); flipped += 1
                else:
                    noise_drop += 1
    print(f"  总 reads:        {total:,}")
    print(f"  长度/N 剔除:     {len_drop:,}")
    print(f"  无 GT 跳过:      {no_gt:,}")
    print(f"  翻转救回:        {flipped:,}")
    print(f"  真噪声剔除:      {noise_drop:,}")
    print(f"  有效 reads:      {sum(len(v) for v in by_ref.values()):,}")
    print(f"  覆盖 ref:        {len(by_ref):,} / {N_GT_TAGS:,}")
    return by_ref


# ============================================================ Step1 过滤+写Clover输入
def step1_filter_write(by_ref, min_size, out_dir):
    banner(f"Step 1  按 GT 簇过滤 (size>={min_size}) + 写 Clover 输入")
    kept = {rf: s for rf, s in by_ref.items() if len(s) >= min_size}
    clover_in = os.path.join(out_dir, '01_clover_input.txt')
    idx_map = {}  # idx -> (ref_id, seq)
    with open(clover_in, 'w') as f:
        i = 0
        for rf, seqs in kept.items():
            for s in seqs:
                i += 1
                f.write(f"{i} {s}\n")
                idx_map[i] = (rf, s)
    print(f"  保留簇: {len(kept):,}   reads: {len(idx_map):,}")
    print(f"  Clover 输入: {clover_in}")
    return clover_in, idx_map


# ============================================================ Step2 Clover
def step2_run_clover(clover_in, out_base):
    banner("Step 2  运行 Clover")
    cfg = os.path.join(CLOVER_DIR, 'clover', 'load_config.py')
    with open(cfg) as f: content = f.read()
    patches = {'h_index_nums': CLOVER_H_INDEX, 'e_index_nums': CLOVER_E_INDEX,
               'thd_tree_loc': 21, 'four_tree_loc': 56}   # spike熵分析推荐(高区分度窗口)
    for k, v in patches.items():
        content = re.sub(rf'"{k}"\s*:\s*\d+', f'"{k}" : {v}', content)
    with open(cfg, 'w') as f: f.write(content)

    cmd = [sys.executable, "-m", "clover.main", "-I", clover_in, "-O", out_base,
           "-L", str(REF_LEN), "-P", "0", "-D", str(CLOVER_TREE_DEPTH),
           "-V", str(CLOVER_V_DRIFT), "-H", str(CLOVER_H_DRIFT), "--no-tag"]
    env = os.environ.copy()
    env["PYTHONPATH"] = CLOVER_DIR + os.pathsep + env.get("PYTHONPATH", "")
    print(f"  Clover -L {REF_LEN} -D {CLOVER_TREE_DEPTH} -V {CLOVER_V_DRIFT} -H {CLOVER_H_DRIFT}\n")
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env, cwd=CLOVER_DIR)
    print(f"  耗时 {time.time()-t0:.1f}s")
    result_path = out_base + '.txt'
    if not os.path.exists(result_path):
        alt = os.path.join(CLOVER_DIR, os.path.basename(out_base) + '.txt')
        if os.path.exists(alt):
            os.rename(alt, result_path)
    return result_path


# ============================================================ Step3 解析Clover结果
def step3_parse_clover(out_txt):
    banner("Step 3  解析 Clover 结果")
    with open(out_txt) as f:
        content = f.read()
    pairs = re.findall(r"\('?(\d+)'?,\s*'?(\d+)'?\)", content)
    cid_to_idxs = defaultdict(list)
    for idx_str, cid_str in pairs:
        cid_to_idxs[int(cid_str)].append(int(idx_str))
    sizes = [len(v) for v in cid_to_idxs.values()]
    print(f"  (idx,cid): {len(pairs):,} 条   簇数: {len(cid_to_idxs):,}")
    if sizes:
        print(f"  max={max(sizes)}, med={sorted(sizes)[len(sizes)//2]}, min={min(sizes)}")
    return dict(cid_to_idxs)


def step3_5_filter(cid_to_idxs, min_reads):
    before = len(cid_to_idxs)
    cid_to_idxs = {c: i for c, i in cid_to_idxs.items() if len(i) >= min_reads}
    print(f"  小簇过滤(<{min_reads}): {before:,} -> {len(cid_to_idxs):,}")
    return cid_to_idxs


# ============================================================ Step4 评测
def step4_stats(cid_to_idxs, idx_map, out_dir):
    banner("Step 4  Purity / Coverage")
    total_reads = total_pure = 0
    covered = set()
    sizes = []
    for idxs in cid_to_idxs.values():
        refs = [idx_map[i][0] for i in idxs if i in idx_map]
        if not refs: continue
        sizes.append(len(refs))
        total_reads += len(refs)
        maj, cnt = Counter(refs).most_common(1)[0]
        total_pure += cnt
        covered.add(maj)
    sizes.sort(reverse=True)
    purity = total_pure / max(total_reads, 1)
    coverage = len(covered) / N_GT_TAGS
    print(f"  簇数:      {len(cid_to_idxs):,}")
    print(f"  reads:     {total_reads:,}")
    print(f"  Purity:    {purity*100:.2f}%")
    print(f"  Coverage:  {coverage*100:.2f}%  ({len(covered)}/{N_GT_TAGS})")
    if sizes:
        print(f"  size: max={sizes[0]}, median={sizes[len(sizes)//2]}")
    with open(os.path.join(out_dir, '03_stats.txt'), 'w') as f:
        f.write(f"clusters: {len(cid_to_idxs)}\nreads: {total_reads}\n")
        f.write(f"Purity: {purity*100:.2f}%\nCoverage: {coverage*100:.2f}%\n")
    return purity, coverage


# ============================================================ Step5+6 MV写FedDNA输入
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

def step56_write(cid_to_idxs, idx_map, out_dir):
    banner("Step 5+6  MV -> read.txt + ref.txt")
    fed_dir = os.path.join(out_dir, '04_FedDNA_In')
    os.makedirs(fed_dir, exist_ok=True)
    read_p = os.path.join(fed_dir, 'read.txt')
    ref_p  = os.path.join(fed_dir, 'ref.txt')
    SEP = "=====分隔符=====\n"
    nc = nr = 0
    with open(read_p, 'w') as fr, open(ref_p, 'w') as ff:
        for cid in sorted(cid_to_idxs.keys()):
            reads = [idx_map[i][1] for i in cid_to_idxs[cid] if i in idx_map]
            if not reads: continue
            for r in reads: fr.write(r + '\n')
            fr.write(SEP)
            ff.write(majority_vote(reads, REF_LEN) + '\n')
            nc += 1; nr += len(reads)
    print(f"  簇 {nc:,}, reads {nr:,}")
    print(f"  {read_p}\n  {ref_p}")
    return read_p, ref_p


# ============================================================ Step7 部署
def step7_deploy(idx_map, out_dir, tag):
    banner("Step 7  部署 SSI-EC")
    exp_dir = os.path.join(out_dir, 'exp')
    os.makedirs(os.path.join(exp_dir, '03_FedDNA_In'), exist_ok=True)
    import shutil
    for fn in ['read.txt', 'ref.txt']:
        shutil.copy2(os.path.join(out_dir, '04_FedDNA_In', fn),
                     os.path.join(exp_dir, '03_FedDNA_In', fn))
    # GT tags
    gt_tags = os.path.join(exp_dir, 'stairloop_tags_reads.txt')
    with open(gt_tags, 'w') as f:
        for ref, seq in idx_map.values():
            f.write(f"{ref}\t{seq}\n")
    # GT refs
    gt_refs = os.path.join(exp_dir, 'stairloop_refs.txt')
    with open(REF) as fin, open(gt_refs, 'w') as fout:
        for line in fin:
            if not line.startswith('>'): fout.write(line)
    print(f"  GT tags: {gt_tags}")
    print(f"  GT refs: {gt_refs}")
    print(f"\n  >> 运行 SSI-EC:")
    print(f"     cd /mnt/st_data/liangxinyi/code/models")
    print(f"     python main_loop.py \\")
    print(f"       --experiment_dir {exp_dir}/ \\")
    print(f"       --feddna_checkpoint /mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth \\")
    print(f"       --max_iterations 3 --max_length {REF_LEN} --cl_mode ours \\")
    print(f"       --gt_tags_file {gt_tags} --gt_refs_file {gt_refs} \\")
    print(f"       2>&1 | tee stairloop_{tag}.log")


# ============================================================ Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min_cluster_size', type=int, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--skip_clover', action='store_true')
    ap.add_argument('--min_reads', type=int, default=2)
    args = ap.parse_args()

    out_dir = os.path.join(BASE, f'out_{args.tag}')
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, '02_clover_result')

    t0 = time.time()
    print("="*60)
    print(f"  StairLoop CA  [{args.tag}]  size>={args.min_cluster_size}")
    print("="*60)

    ref = load_ref(); gt = load_gt()
    print(f"  ref={len(ref):,}  gt={len(gt):,}")

    by_ref = step0_process(ref, gt)
    clover_in, idx_map = step1_filter_write(by_ref, args.min_cluster_size, out_dir)
    del by_ref

    if not args.skip_clover:
        out_txt = step2_run_clover(clover_in, out_base)
    else:
        out_txt = out_base + '.txt'
        banner("Step 2  跳过 Clover")
        if not os.path.exists(out_txt):
            raise FileNotFoundError(out_txt)

    cid_to_idxs = step3_parse_clover(out_txt)
    cid_to_idxs = step3_5_filter(cid_to_idxs, args.min_reads)
    step4_stats(cid_to_idxs, idx_map, out_dir)
    step56_write(cid_to_idxs, idx_map, out_dir)
    step7_deploy(idx_map, out_dir, args.tag)

    print(f"\n  总耗时 {time.time()-t0:.1f}s")
    print("="*60)


if __name__ == '__main__':
    main()