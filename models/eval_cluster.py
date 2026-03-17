#!/usr/bin/env python3
"""
SSI-EC 全口径评估 (双轨 Purity + 100% 包含噪声版)
====================
"""
import os, sys, glob, time, re
import numpy as np
from collections import Counter, defaultdict

# ═══════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════
EXP_DIR      = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_NoPrimer"
GT_TAGS_FILE = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1_NoPrimer/exp1_tags_reads.txt"
GT_REFS_FILE = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1_NoPrimer/exp1_refs.fasta"
MAX_ROUNDS   = 3

# ═══════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════
def load_reads():
    read_path = os.path.join(EXP_DIR, "03_FedDNA_In", "read.txt")
    reads, clover_labels = [], []
    cid = 0
    with open(read_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("====="):
                cid += 1
            else:
                reads.append(line)
                clover_labels.append(cid)
    return reads, np.array(clover_labels, dtype=np.int64)

def load_gt(reads):
    print(f"\n📋 加载 GT: {os.path.basename(GT_TAGS_FILE)}")
    seq_to_gt = {}
    total = 0
    with open(GT_TAGS_FILE) as f:
        for line in f:
            total += 1
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    seq_to_gt[parts[1].strip().upper()] = int(parts[0])
                except ValueError:
                    pass

    gt = np.full(len(reads), -1, dtype=np.int64)
    matched = 0
    for i, r in enumerate(reads):
        g = seq_to_gt.get(r.upper())
        if g is not None:
            gt[i] = g
            matched += 1

    print(f"   匹配:         {matched:,} / {len(reads):,} ({matched/len(reads)*100:.1f}%)")
    return gt

def load_refs():
    if not os.path.exists(GT_REFS_FILE): return None
    refs = {}
    cid, seq = None, []
    with open(GT_REFS_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cid is not None: refs[cid] = ''.join(seq)
                cid = int(line[1:].split()[0])
                seq = []
            elif line:
                seq.append(line)
    if cid is not None: refs[cid] = ''.join(seq)
    return refs

def load_consensus_fasta(path):
    seqs = {}
    cid, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cid is not None: seqs[cid] = ''.join(seq)
                nums = re.findall(r'\d+', line[1:].split()[0])
                cid = int(nums[0]) if nums else None
                seq = []
            elif line:
                seq.append(line)
    if cid is not None: seqs[cid] = ''.join(seq)
    return seqs

def _edit_distance(s1, s2):
    n, m = len(s1), len(s2)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev, curr = curr, prev
    return prev[m]

# ═══════════════════════════════════════════════════
# 核心指标计算 (包含所有噪声 -1)
# ═══════════════════════════════════════════════════
def compute_metrics(pred, gt, name):
    # 【核心修改】只要求有真实的 GT，不再排除 pred == -1 的数据！
    # 所有被模型丢弃为 -1 的噪声也会被强制算入 Purity 分母！
    valid = (gt >= 0)
    n_valid = valid.sum()
    p, g = pred[valid], gt[valid]

    if n_valid == 0:
        return None

    # 按预测簇归类真实标签 (预测为 -1 的也会被归入簇 -1)
    c2g = defaultdict(list)
    for pi, gi in zip(p, g):
        c2g[pi].append(gi)

    # 1. 严格 Purity (不过滤任何簇)
    correct_strict = sum(Counter(gs).most_common(1)[0][1] for gs in c2g.values())
    pur_strict = correct_strict / n_valid

    # 2. Clover 对齐 Purity (过滤 size <= 1 的孤立簇)
    clover_total_reads = 0
    clover_correct_reads = 0
    clover_valid_clusters = 0

    for pi, gs in c2g.items():
        if len(gs) > 1:
            clover_total_reads += len(gs)
            clover_correct_reads += Counter(gs).most_common(1)[0][1]
            clover_valid_clusters += 1

    pur_clover = clover_correct_reads / clover_total_reads if clover_total_reads > 0 else 0.0

    # 3. PCR 及其他指标 (排除簇 -1 参与 PCR 计算，因为垃圾桶不是一个真实的分子簇)
    perfect = 0
    for pi, gs in c2g.items():
        if pi >= 0 and len(set(gs)) == 1:
            perfect += 1
    
    # 有效预测簇数（不包含垃圾桶 -1）
    n_pred_valid = len([pi for pi in c2g.keys() if pi >= 0])
    pcr = perfect / max(n_pred_valid, 1)

    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(g, p)
        nmi = normalized_mutual_info_score(g, p)
    except ImportError:
        ari, nmi = None, None

    n_gt = len(set(g.tolist()))

    return {'name': name, 'pur': pur_strict, 'pur_clover': pur_clover, 'pcr': pcr, 'ari': ari, 'nmi': nmi,
            'n_pred': n_pred_valid, 'n_gt': n_gt, 'n_reads': int(n_valid),
            'coverage': n_valid / len(pred)}

def over_segmentation(pred, gt):
    valid = (pred >= 0) & (gt >= 0)
    gt2pred = defaultdict(set)
    for p, g in zip(pred[valid], gt[valid]):
        gt2pred[g].add(p)
    frags = [len(s) for s in gt2pred.values()]
    if not frags: return
    print(f"\n  📐 过分割统计:")
    print(f"     GT 分子数:        {len(gt2pred):,}")
    print(f"     平均碎片/分子:    {np.mean(frags):.2f}")
    print(f"     最大碎片数:       {max(frags)}")

def eval_reconstruction(cons_path, pred, gt, refs, name, max_eval=5000):
    if refs is None: return
    consensus = load_consensus_fasta(cons_path)
    if not consensus: return

    c2g = defaultdict(list)
    for p, g in zip(pred, gt):
        if p >= 0 and g >= 0: c2g[p].append(g)
    c_maj = {c: Counter(gs).most_common(1)[0][0] for c, gs in c2g.items()}

    cids = sorted(set(consensus.keys()) & set(c_maj.keys()))
    if len(cids) > max_eval:
        rng = np.random.RandomState(42)
        cids = sorted(rng.choice(cids, max_eval, replace=False))

    eds = []
    for c in cids:
        gid = c_maj[c]
        if gid in refs:
            eds.append(_edit_distance(consensus[c], refs[gid]))

    if eds:
        print(f"\n  🧬 重建质量 [{name}] - Mean ED: {np.mean(eds):.2f}, Success Rate: {sum(1 for e in eds if e==0)/len(eds):.4f}")

# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("  SSI-EC 聚类效果全口径评估 (100% 全局对比 Baseline)")
    print("=" * 80)

    reads, clover_labels = load_reads()
    N = len(reads)
    gt = load_gt(reads)
    refs = load_refs()
    all_results = []

    # 1. Clover 基线
    r = compute_metrics(clover_labels, gt, "Clover (原始基线)")
    if r: all_results.append(r)
    over_segmentation(clover_labels, gt)

    # 2. SSI-EC 每一轮迭代
    labels_dir = os.path.join(EXP_DIR, "04_Iterative_Labels")
    results_dir = os.path.join(EXP_DIR, "results")
    label_files = sorted(glob.glob(os.path.join(labels_dir, "refined_labels_*.txt")))

    for idx in range(min(MAX_ROUNDS, len(label_files))):
        r_idx = idx + 1
        lpath = label_files[idx]
        labels_r = np.loadtxt(lpath, dtype=np.int64)
        # 此时包含所有标签（包括被标记为 -1 的）
        r = compute_metrics(labels_r, gt, f"SSI-EC Round {r_idx} (包含全部噪声)")
        if r: all_results.append(r)
        
        step2_dir = os.path.join(results_dir, f"iter_{r_idx}_step2")
        cons_files = sorted(glob.glob(os.path.join(step2_dir, "consensus", "consensus_*.fasta")))
        if cons_files: eval_reconstruction(cons_files[-1], labels_r, gt, refs, f"Round {r_idx}")

    # 3. Post-process 结果
    final_path = os.path.join(results_dir, "final", "final_labels.txt")
    if os.path.exists(final_path):
        final = np.loadtxt(final_path, dtype=np.int64)
        r = compute_metrics(final, gt, "SSI-EC Final (强制收编)")
        if r: all_results.append(r)

    # 4. 终极汇总表打印
    if all_results:
        print(f"\n{'='*110}")
        print(f"{'📋 SSI-EC vs Clover 聚类指标全局对比表 (无任何数据过滤)':^110s}")
        print(f"{'='*110}")
        print(f"  {'Method':<32s} | {'Strict Pur':>10s} | {'Clover Pur':>10s} | {'PCR':>8s} | {'ARI':>8s} | {'NMI':>7s} | {'Cover':>7s}")
        print(f"  {'-'*105}")
        for r in all_results:
            a = f"{r['ari']:.4f}" if r['ari'] is not None else "  N/A"
            n = f"{r['nmi']:.4f}" if r['nmi'] is not None else " N/A"
            c = f"{r['coverage']*100:.1f}%"
            print(f"  {r['name']:<32s} | {r['pur']:>10.4f} | {r['pur_clover']:>10.4f} | {r['pcr']:>8.4f} | {a:>8s} | {n:>7s} | {c:>7s}")
        print(f"{'='*110}")
        print("\n📝 提示: Cover 列现在应该全部是 100.0% (代表对所有有GT的 Reads 进行了无差别评估)。")

if __name__ == '__main__':
    main()