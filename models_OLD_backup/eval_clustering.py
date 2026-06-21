#!/usr/bin/env python3
"""
SSI-EC 全口径评估 v3
====================
1. 验证 GT 匹配 + refs.fasta ID 对应关系
2. Clover 基线 (Purity / PCR / ARI / NMI)
3. SSI-EC 每轮评估
4. Post-process 后评估
5. 过分割量化
6. 重建质量 (consensus vs GT ref Edit Distance)

运行:
  cd /mnt/st_data/liangxinyi/code/models/
  python eval_clustering.py 2>&1 | tee eval_results.txt
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
    """从 read.txt 解析 reads + Clover 标签（分隔符法）"""
    read_path = os.path.join(EXP_DIR, "03_FedDNA_In", "read.txt")
    reads, clover_labels = [], []
    cid = 0
    with open(read_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("====="):
                cid += 1
            else:
                reads.append(line)
                clover_labels.append(cid)
    return reads, np.array(clover_labels, dtype=np.int64)


def load_gt(reads):
    """
    GT 加载: 与 CloverDataLoader.load_gt_tags 完全一致。
    文件格式: GT_cluster_id<TAB>trimmed_sequence
    策略: sequence→cluster_id 字典 + 精确匹配
    """
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

    print(f"   GT 文件行数:  {total:,}")
    print(f"   唯一序列:     {len(seq_to_gt):,}")
    print(f"   匹配:         {matched:,} / {len(reads):,} ({matched/len(reads)*100:.1f}%)")
    return gt


def load_refs():
    """加载 GT reference FASTA → {ref_id: sequence}"""
    if not os.path.exists(GT_REFS_FILE):
        print(f"   ⚠️ refs 文件不存在")
        return None
    refs = {}
    cid, seq = None, []
    with open(GT_REFS_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cid is not None:
                    refs[cid] = ''.join(seq)
                cid = int(line[1:].split()[0])
                seq = []
            elif line:
                seq.append(line)
    if cid is not None:
        refs[cid] = ''.join(seq)
    print(f"   GT refs:      {len(refs)} 条")
    return refs


def load_consensus_fasta(path):
    seqs = {}
    cid, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cid is not None:
                    seqs[cid] = ''.join(seq)
                nums = re.findall(r'\d+', line[1:].split()[0])
                cid = int(nums[0]) if nums else None
                seq = []
            elif line:
                seq.append(line)
    if cid is not None:
        seqs[cid] = ''.join(seq)
    return seqs


# ═══════════════════════════════════════════════════
# Edit Distance
# ═══════════════════════════════════════════════════

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
# 替换原有的 compute_metrics 函数
# ═══════════════════════════════════════════════════
def compute_metrics(pred, gt, name):
    valid = (pred >= 0) & (gt >= 0)
    n_valid = valid.sum()
    p, g = pred[valid], gt[valid]

    if n_valid == 0:
        print(f"  ⚠️ 无有效评估样本")
        return None

    # 将预测结果按簇分组
    c2g = defaultdict(list)
    for pi, gi in zip(p, g):
        c2g[pi].append(gi)

    # --------------------------------------------------
    # 1. 严格 Purity (SSI-EC 标准：不过滤任何簇)
    # --------------------------------------------------
    correct_strict = sum(Counter(gs).most_common(1)[0][1] for gs in c2g.values())
    pur_strict = correct_strict / n_valid

    # --------------------------------------------------
    # 2. 对齐 Clover 的 Purity (核心修改：丢弃 size <= 1 的簇)
    # --------------------------------------------------
    clover_total_reads = 0
    clover_correct_reads = 0
    clover_valid_clusters = 0

    for pi, gs in c2g.items():
        if len(gs) > 1:  # 完全对齐 Clover: tag_len > Cluster_size_threshold (1)
            clover_total_reads += len(gs)
            clover_correct_reads += Counter(gs).most_common(1)[0][1]
            clover_valid_clusters += 1

    pur_clover = clover_correct_reads / clover_total_reads if clover_total_reads > 0 else 0.0

    # --------------------------------------------------
    # 3. 其他指标
    # --------------------------------------------------
    # Perfect Cluster Rate
    perfect = sum(1 for gs in c2g.values() if len(set(gs)) == 1)
    pcr = perfect / len(c2g)

    # ARI / NMI
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(g, p)
        nmi = normalized_mutual_info_score(g, p)
    except ImportError:
        ari, nmi = None, None

    n_pred = len(c2g)
    n_gt = len(set(g.tolist()))

    print(f"\n{'─'*70}")
    print(f"  📊 {name}")
    print(f"{'─'*70}")
    print(f"  Reads 评估: {n_valid:>12,} / {len(pred):,} ({n_valid/len(pred)*100:.1f}%)")
    print(f"  预测簇数:   {n_pred:>12,}")
    print(f"  GT 簇数:    {n_gt:>12,}")
    print(f"  Purity (严格标准):     {pur_strict:.4f}  ({correct_strict:,}/{n_valid:,})")
    print(f"  Purity (Clover对齐):   {pur_clover:.4f}  ({clover_correct_reads:,}/{clover_total_reads:,}) [已过滤 {n_pred - clover_valid_clusters:,} 个孤立簇]")
    print(f"  Perfect Cluster Rate: {pcr:.4f}  ({perfect:,}/{n_pred:,})")
    if ari is not None:
        print(f"  ARI:                 {ari:.6f}")
        print(f"  NMI:                 {nmi:.4f}")

    return {'name': name, 'pur': pur_strict, 'pur_clover': pur_clover, 'pcr': pcr, 'ari': ari, 'nmi': nmi,
            'n_pred': n_pred, 'n_gt': n_gt, 'n_reads': int(n_valid),
            'coverage': n_valid / len(pred)}


def over_segmentation(pred, gt):
    valid = (pred >= 0) & (gt >= 0)
    gt2pred = defaultdict(set)
    for p, g in zip(pred[valid], gt[valid]):
        gt2pred[g].add(p)
    frags = [len(s) for s in gt2pred.values()]
    if not frags:
        return
    print(f"\n  📐 过分割统计:")
    print(f"     GT 分子数:        {len(gt2pred):,}")
    print(f"     平均碎片/分子:    {np.mean(frags):.2f}")
    print(f"     中位碎片/分子:    {np.median(frags):.0f}")
    print(f"     最大碎片数:       {max(frags)}")
    print(f"     1片={sum(1 for f in frags if f==1):,}  "
          f"2-5片={sum(1 for f in frags if 2<=f<=5):,}  "
          f"6-20片={sum(1 for f in frags if 6<=f<=20):,}  "
          f">20片={sum(1 for f in frags if f>20):,}")


# ═══════════════════════════════════════════════════
# 验证 refs.fasta ID 对应关系
# ═══════════════════════════════════════════════════

def verify_refs(reads, gt, refs):
    if refs is None:
        return
    print(f"\n{'='*65}")
    print(f"🔍 验证: refs.fasta ID 与 GT tag ID 对应关系")
    print(f"{'='*65}")

    gt2reads = defaultdict(list)
    for i, g in enumerate(gt):
        if g >= 0:
            gt2reads[g].append(reads[i])

    # 抽样 100 个 GT cluster
    sample_gts = sorted(gt2reads.keys())[:100]
    ok, total_ed, tested = 0, 0, 0
    for gid in sample_gts:
        if gid not in refs:
            continue
        tested += 1
        ed = _edit_distance(gt2reads[gid][0], refs[gid])
        total_ed += ed
        if ed <= 10:
            ok += 1

    print(f"   测试 {tested} 个 GT cluster (read vs ref ED):")
    print(f"   ED ≤ 10: {ok}/{tested} ({ok/max(tested,1)*100:.1f}%)")
    print(f"   平均 ED: {total_ed/max(tested,1):.2f}")
    if ok / max(tested, 1) > 0.9:
        print(f"   ✅ refs.fasta ID = GT tag ID，对应正确")
    else:
        print(f"   ⚠️ 对应关系可能有问题!")


# ═══════════════════════════════════════════════════
# 重建质量
# ═══════════════════════════════════════════════════

def eval_reconstruction(cons_path, pred, gt, refs, name, max_eval=5000):
    if refs is None:
        return
    consensus = load_consensus_fasta(cons_path)
    if not consensus:
        return

    c2g = defaultdict(list)
    for p, g in zip(pred, gt):
        if p >= 0 and g >= 0:
            c2g[p].append(g)
    c_maj = {c: Counter(gs).most_common(1)[0][0] for c, gs in c2g.items()}
    c_pur = {c: Counter(gs).most_common(1)[0][1] / len(gs) for c, gs in c2g.items()}

    cids = sorted(set(consensus.keys()) & set(c_maj.keys()))
    if len(cids) > max_eval:
        rng = np.random.RandomState(42)
        cids = sorted(rng.choice(cids, max_eval, replace=False))

    eds, skip = [], 0
    pure_eds, impure_eds = [], []
    for c in cids:
        gid = c_maj[c]
        if gid not in refs:
            skip += 1
            continue
        ed = _edit_distance(consensus[c], refs[gid])
        eds.append(ed)
        (pure_eds if c_pur[c] == 1.0 else impure_eds).append(ed)

    if not eds:
        print(f"  ⚠️ 无有效重建评估对")
        return

    sr = sum(1 for e in eds if e == 0) / len(eds)
    print(f"\n  🧬 重建质量 [{name}]")
    print(f"     评估簇: {len(eds):,}  (跳过: {skip})")
    print(f"     Mean ED:        {np.mean(eds):.2f}")
    print(f"     Median ED:      {np.median(eds):.0f}")
    print(f"     Success Rate:   {sr:.4f}  ({sum(1 for e in eds if e==0)}/{len(eds)})")
    print(f"     ED ≤ 3:         {sum(1 for e in eds if e<=3)/len(eds):.4f}")
    print(f"     ED ≤ 5:         {sum(1 for e in eds if e<=5)/len(eds):.4f}")
    if pure_eds:
        print(f"     [纯簇 {len(pure_eds)}] SR={sum(1 for e in pure_eds if e==0)/len(pure_eds):.4f}"
              f"  MeanED={np.mean(pure_eds):.2f}")
    if impure_eds:
        print(f"     [杂簇 {len(impure_eds)}] SR={sum(1 for e in impure_eds if e==0)/len(impure_eds):.4f}"
              f"  MeanED={np.mean(impure_eds):.2f}")


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  SSI-EC 全口径评估")
    print("=" * 65)

    reads, clover_labels = load_reads()
    N = len(reads)
    print(f"\n📂 Reads: {N:,}, Clover 簇: {len(set(clover_labels.tolist())):,}")

    gt = load_gt(reads)
    refs = load_refs()

    all_results = []

    # 0. 验证 refs ID
    verify_refs(reads, gt, refs)

    # 1. Clover 基线
    print(f"\n{'='*65}")
    print(f"📊 [1] Clover 基线")
    print(f"{'='*65}")
    r = compute_metrics(clover_labels, gt, "Clover (原始)")
    if r: all_results.append(r)
    over_segmentation(clover_labels, gt)

    # 2. SSI-EC 每轮
    labels_dir = os.path.join(EXP_DIR, "04_Iterative_Labels")
    results_dir = os.path.join(EXP_DIR, "results")
    label_files = sorted(glob.glob(os.path.join(labels_dir, "refined_labels_*.txt")))
    print(f"\n找到 {len(label_files)} 个 refined_labels 文件")

    for idx in range(min(MAX_ROUNDS, len(label_files))):
        r_idx = idx + 1
        lpath = label_files[idx]
        print(f"\n{'='*65}")
        print(f"📊 [2] SSI-EC Round {r_idx}")
        print(f"{'='*65}")
        print(f"   文件: {os.path.basename(lpath)}")

        labels_r = np.loadtxt(lpath, dtype=np.int64)
        assert len(labels_r) == N, f"长度不匹配: {len(labels_r)} vs {N}"

        r = compute_metrics(labels_r, gt, f"SSI-EC R{r_idx} (不含噪声)")
        if r: all_results.append(r)
        over_segmentation(labels_r, gt)

        step2_dir = os.path.join(results_dir, f"iter_{r_idx}_step2")
        cons_files = sorted(glob.glob(os.path.join(step2_dir, "consensus", "consensus_*.fasta")))
        if cons_files:
            eval_reconstruction(cons_files[-1], labels_r, gt, refs, f"Round {r_idx}")

    # 3. Post-process
    final_path = os.path.join(results_dir, "final", "final_labels.txt")
    if os.path.exists(final_path):
        print(f"\n{'='*65}")
        print(f"📊 [3] Post-process 后")
        print(f"{'='*65}")
        final = np.loadtxt(final_path, dtype=np.int64)
        assert len(final) == N
        r = compute_metrics(final, gt, "SSI-EC Final (含 post-process)")
        if r: all_results.append(r)

        last_round = min(MAX_ROUNDS, len(label_files))
        last_cons = sorted(glob.glob(os.path.join(
            results_dir, f"iter_{last_round}_step2", "consensus", "consensus_*.fasta")))
        if last_cons:
            eval_reconstruction(last_cons[-1], final, gt, refs, "Final")

    # 4. 汇总表
    if all_results:
        print(f"\n{'='*105}")
        print(f"{'📋 SSI-EC vs Clover 全局评估汇总':^105s}")
        print(f"{'='*105}")
        print(f"  {'Method':<30s} {'Strict Pur':>10s} {'Clover Pur':>10s} {'PCR':>8s} {'ARI':>9s} "
              f"{'NMI':>7s} {'Pred':>7s} {'GT':>6s} {'Cover':>7s}")
        print(f"  {'─'*100}")
        for r in all_results:
            a = f"{r['ari']:.5f}" if r['ari'] is not None else "   N/A"
            n = f"{r['nmi']:.4f}" if r['nmi'] is not None else " N/A"
            c = f"{r['coverage']*100:.1f}%"
            # 这里同时打印出 Strict Purity 和 Clover Purity
            print(f"  {r['name']:<30s} {r['pur']:>10.4f} {r['pur_clover']:>10.4f} {r['pcr']:>8.4f} "
                  f"{a:>9s} {n:>7s} {r['n_pred']:>7,} {r['n_gt']:>6,} {c:>7s}")
        print(f"{'='*105}")

    print(f"\n✅ 评估完成")


if __name__ == '__main__':
    main()