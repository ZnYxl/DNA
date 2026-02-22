"""
eval_reconstruction.py - Per-Pred-Cluster 重建评估

核心思想:
  SSI-EC 贡献: 聚类 + zone/strength 质量信号
  FedDNA 贡献: read→reference 的 evidence 重建能力
  两者结合: zone-aware evidence fusion

三组对比 (同一个 FedDNA 模型做推理, 只换聚类标签和 fusion 策略):
  A. Clover 聚类 + FedDNA 等权 fusion         → baseline
  B. SSI-EC 聚类 + FedDNA 等权 fusion          → 只体现聚类改善
  C. SSI-EC 聚类 + FedDNA zone-aware 加权 fusion → 完整 SSI-EC

+ D. Raw majority vote (不用模型, 纯碱基投票)   → 最简 baseline

per-pred-cluster 评估:
  1. 对每个 pred 簇, 用簇内 reads 做 evidence fusion → 一条重建序列
  2. Majority voting 映射 pred 簇 → GT ref
  3. 与 GT ref 比: Success Rate (ED=0), Edit Error Rate

用法:
  cd /mnt/st_data/liangxinyi/code
  CUDA_VISIBLE_DEVICES=0 python eval_reconstruction.py
"""
import os
import re
import sys
import gc
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

CODE_DIR = "/mnt/st_data/liangxinyi/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.step1_model import Step1EvidentialModel
from models.step1_data import CloverDataLoader, seq_to_onehot

# ================= 路径配置 =================
EXP_DIR    = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_Real_last"
GT_REFS    = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1/exp1_refs.fasta"
GT_TAGS    = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1/exp1_tags_reads.txt"
FEDDNA_CKP = "/mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN    = 150
DIM        = 256
BATCH_SIZE = 512     # 每批推理的 reads 数
# ============================================


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def load_fasta(path):
    seqs = {}
    name = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                name = line[1:]
                seqs[name] = ''
            elif name:
                seqs[name] += line
    return seqs


def parse_gt_refs(path):
    raw = load_fasta(path)
    refs = {}
    for header, seq in raw.items():
        nums = re.findall(r'\d+', header)
        if nums:
            refs[int(nums[0])] = seq
        elif header.isdigit():
            refs[int(header)] = seq
    return refs


def edit_distance(s1, s2):
    n, m = len(s1), len(s2)
    if n > m:
        s1, s2 = s2, s1
        n, m = m, n
    prev = list(range(n + 1))
    for j in range(1, m + 1):
        curr = [j] + [0] * n
        for i in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            curr[i] = min(curr[i-1] + 1, prev[i] + 1, prev[i-1] + cost)
        prev = curr
    return prev[n]


def majority_vote_raw(reads_list, out_len):
    """纯碱基多数投票 (不用模型)"""
    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    char_idx = {'A':0,'C':1,'G':2,'T':3,'a':0,'c':1,'g':2,'t':3,'N':0,'n':0}
    votes = np.zeros((out_len, 4), dtype=np.int32)
    for read in reads_list:
        L = min(len(read), out_len)
        for pos in range(L):
            votes[pos, char_idx.get(read[pos], 0)] += 1
    return ''.join([base_map[i] for i in np.argmax(votes, axis=1)])


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_feddna_model(ckpt_path, device):
    """加载 FedDNA 原始模型 (read→reference 重建能力)"""
    checkpoint = torch.load(ckpt_path, map_location=device)

    # FedDNA checkpoint 可能是不同格式, 尝试兼容
    if 'model_state_dict' in checkpoint:
        sd = checkpoint['model_state_dict']
        step1_args = checkpoint.get('args', {})
    elif 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
        step1_args = {}
    else:
        sd = checkpoint
        step1_args = {}

    model_max_len = step1_args.get('max_length', MAX_LEN)

    # 用 Step1EvidentialModel 加载 (共享 encoder + rnnblock)
    model = Step1EvidentialModel(
        dim=DIM, max_length=model_max_len,
        num_clusters=80000, device=str(device)
    ).to(device)

    # 处理 length_adapter
    if 'length_adapter.weight' in sd:
        sh = sd['length_adapter.weight'].shape
        model.length_adapter = nn.Linear(sh[1], sh[0]).to(device)
    else:
        # FedDNA 原始模型的 key 可能不带 prefix
        sample_key = list(sd.keys())[0]
        print(f"    ℹ️ checkpoint 首个 key: {sample_key}")

    model.load_state_dict(sd, strict=False)
    model.eval()

    # 统计成功加载的 key 数量
    model_keys = set(dict(model.state_dict()).keys())
    loaded = len([k for k in sd if k in model_keys])
    print(f"    ✅ FedDNA 模型加载 (max_len={model_max_len})")
    print(f"       Loaded keys: {loaded}/{len(sd)}")

    return model, model_max_len


# ---------------------------------------------------------------------------
# Evidence 推理 (per cluster)
# ---------------------------------------------------------------------------
@torch.no_grad()
def infer_cluster_evidence(model, reads_seqs, max_len, device):
    """
    对一个簇的 reads 逐条推理 FedDNA evidence

    Returns:
        evidence: (n_reads, max_len, 4) tensor on CPU
    """
    n = len(reads_seqs)
    all_evidence = []

    for batch_start in range(0, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch_seqs = reads_seqs[batch_start:batch_end]

        reads_tensor = torch.stack([
            seq_to_onehot(s, max_len) for s in batch_seqs
        ]).to(device)

        # Padding/Truncation
        if reads_tensor.shape[1] != max_len:
            if reads_tensor.shape[1] < max_len:
                reads_tensor = F.pad(reads_tensor,
                    (0, 0, 0, max_len - reads_tensor.shape[1]))
            else:
                reads_tensor = reads_tensor[:, :max_len, :]

        emb, _ = model.encode_reads(reads_tensor)
        evid, strength, alpha = model.decode_to_evidence(emb)
        # evid: (batch, max_len, 4)
        all_evidence.append(evid.cpu())

    return torch.cat(all_evidence, dim=0)  # (n, max_len, 4)


# ---------------------------------------------------------------------------
# Fusion 策略
# ---------------------------------------------------------------------------
def fuse_equal_weight(evidence):
    """FedDNA 原始: ē_l = (1/n) Σ e^(i)_l"""
    fused = evidence.mean(dim=0)  # (max_len, 4)
    return fused


def fuse_zone_aware(evidence, strengths, zones):
    """
    SSI-EC zone-aware fusion:
      - Zone III (zone=3) 排除
      - Zone I+II 参与, softmax(strength) 加权
    """
    keep_mask = torch.tensor([z in (1, 2) for z in zones])
    if keep_mask.sum() == 0:
        # fallback: 全部参与等权
        return fuse_equal_weight(evidence)

    kept_evidence = evidence[keep_mask]
    kept_strength = torch.tensor([s for s, z in zip(strengths, zones) if z in (1, 2)])

    weights = F.softmax(kept_strength, dim=0)
    fused = (kept_evidence * weights.view(-1, 1, 1)).sum(dim=0)
    return fused


def decode_evidence(fused_evidence):
    """evidence → sequence"""
    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    alpha = fused_evidence + 1.0
    pred = torch.argmax(alpha, dim=-1)
    return ''.join([base_map[i.item()] for i in pred])


# ---------------------------------------------------------------------------
# 评估核心
# ---------------------------------------------------------------------------
def build_cluster_gt_map(pred_labels, gt_labels):
    """Majority voting: pred_cluster → GT ref id"""
    cluster_gt_counts = defaultdict(Counter)
    for i in range(len(pred_labels)):
        pid = int(pred_labels[i])
        gid = int(gt_labels[i])
        if pid >= 0 and gid >= 0:
            cluster_gt_counts[pid][gid] += 1

    cluster_to_gt = {}
    for cid, counter in cluster_gt_counts.items():
        cluster_to_gt[cid] = counter.most_common(1)[0][0]
    return cluster_to_gt


def evaluate_scheme(consensus_seqs, cluster_to_gt, gt_refs, scheme_name):
    """评估一组重建结果"""
    exact = 0
    evaluated = 0
    eds = []
    eers = []

    for cid, consensus in consensus_seqs.items():
        gt_id = cluster_to_gt.get(cid)
        if gt_id is None or gt_id not in gt_refs:
            continue
        gt_seq = gt_refs[gt_id]
        ed = edit_distance(consensus, gt_seq)
        eds.append(ed)
        eers.append(ed / max(len(gt_seq), 1))
        if ed == 0:
            exact += 1
        evaluated += 1

    if evaluated == 0:
        return None

    ed_arr = np.array(eds)
    eer_arr = np.array(eers)
    return {
        'name': scheme_name,
        'evaluated': evaluated,
        'exact': exact,
        'success_rate': exact / evaluated,
        'mean_ed': ed_arr.mean(),
        'median_ed': np.median(ed_arr),
        'mean_eer': eer_arr.mean(),
        'ed_arr': ed_arr,
    }


def print_scheme(m):
    if m is None:
        return
    print(f"\n    [{m['name']}]")
    print(f"      Clusters 评估: {m['evaluated']:,}")
    print(f"      ✅ Success Rate (ED=0):  {m['exact']:,}/{m['evaluated']:,} = {m['success_rate']*100:.2f}%")
    print(f"      📏 Edit Error Rate:      {m['mean_eer']*100:.4f}%")
    print(f"      📏 Mean ED: {m['mean_ed']:.2f}  |  Median: {m['median_ed']:.1f}")
    ed = m['ed_arr']
    n = m['evaluated']
    for t in [0, 1, 2, 3, 5, 10]:
        c = int((ed <= t).sum())
        print(f"        ED≤{t:2d}: {c:>7,} ({c/n*100:6.2f}%)")
    c_high = int((ed > 10).sum())
    print(f"        ED>10: {c_high:>7,} ({c_high/n*100:6.2f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device(DEVICE)

    # =================================================================
    # 1. 加载数据
    # =================================================================
    print("[1] 加载数据...")
    data_loader = CloverDataLoader(EXP_DIR)
    TOTAL = len(data_loader.reads)
    clover_labels = np.array(data_loader.clover_labels, dtype=int)
    print(f"    Reads: {TOTAL:,}")

    # GT
    print("[2] 加载 GT...")
    data_loader.load_gt_tags(GT_TAGS)
    gt_labels_arr = np.array(data_loader.gt_labels, dtype=int)

    gt_refs = parse_gt_refs(GT_REFS)
    ref_len_sample = len(next(iter(gt_refs.values()))) if gt_refs else MAX_LEN
    print(f"    GT refs: {len(gt_refs):,}, ref 长度: {ref_len_sample}bp")

    # =================================================================
    # 2. 加载 SSI-EC 最后一轮 labels + state
    # =================================================================
    print("[3] 加载 SSI-EC labels...")
    labels_dir = os.path.join(EXP_DIR, "04_Iterative_Labels")
    label_files = sorted(glob.glob(os.path.join(labels_dir, "refined_labels_*.txt")))
    state_files = sorted(glob.glob(os.path.join(labels_dir, "read_state_*.pt")))

    # ⚠️ 时间戳跨天, 字母序 ≠ 轮次序!
    #    231829 = Round 1 (前一天 23:18)
    #    115135 = Round 2 (次日 11:51)
    #    201006 = Round 3 (同日 20:10)
    # 手动指定 Round 3 (最后一轮) 的时间戳
    ROUND3_TS = "201006"

    r3_label = os.path.join(labels_dir, f"refined_labels_{ROUND3_TS}.txt")
    r3_state = os.path.join(labels_dir, f"read_state_{ROUND3_TS}.pt")

    if not os.path.exists(r3_label):
        print(f"    ⚠️ 未找到 Round 3 labels ({r3_label}), 只评估 Clover baseline")
        ssi_labels = None
        ssi_state = None
    else:
        ssi_labels = np.loadtxt(r3_label, dtype=int)
        ssi_state = torch.load(r3_state, map_location='cpu')
        n_assigned = (ssi_labels >= 0).sum()
        print(f"    SSI-EC Round 3: refined_labels_{ROUND3_TS}.txt")
        print(f"    已分配: {n_assigned:,} ({n_assigned/TOTAL*100:.1f}%)")

    # Post-proc labels
    final_path = os.path.join(EXP_DIR, "results/final/final_labels.txt")
    pp_labels = None
    if os.path.exists(final_path):
        pp_labels = np.loadtxt(final_path, dtype=int)
        print(f"    Post-proc labels: {(pp_labels>=0).sum():,} 已分配")

    # =================================================================
    # 3. 加载 FedDNA 模型
    # =================================================================
    print("[4] 加载 FedDNA 模型...")
    model, model_max_len = load_feddna_model(FEDDNA_CKP, device)

    # =================================================================
    # 4. 建立 cluster→GT 映射
    # =================================================================
    print("[5] 建立 cluster→GT 映射...")
    clover_c2g = build_cluster_gt_map(clover_labels, gt_labels_arr)
    print(f"    Clover: {len(clover_c2g):,} 簇有 GT 映射")

    ssi_c2g = None
    if ssi_labels is not None:
        ssi_c2g = build_cluster_gt_map(ssi_labels, gt_labels_arr)
        print(f"    SSI-EC: {len(ssi_c2g):,} 簇有 GT 映射")

    pp_c2g = None
    if pp_labels is not None:
        pp_c2g = build_cluster_gt_map(pp_labels, gt_labels_arr)
        print(f"    Post-proc: {len(pp_c2g):,} 簇有 GT 映射")

    # =================================================================
    # 5. 建立 cluster → read_indices
    # =================================================================
    def build_cluster_index(labels_arr):
        idx = defaultdict(list)
        for i in range(len(labels_arr)):
            cid = int(labels_arr[i])
            if cid >= 0:
                idx[cid].append(i)
        return idx

    clover_clusters = build_cluster_index(clover_labels)
    ssi_clusters = build_cluster_index(ssi_labels) if ssi_labels is not None else {}
    pp_clusters = build_cluster_index(pp_labels) if pp_labels is not None else {}

    # =================================================================
    # 6. Clover: 方案 A (FedDNA eq) + D (raw vote)
    # =================================================================
    print(f"\n{'=' * 80}")
    print(f"📊 Per-Pred-Cluster 重建评估")
    print(f"{'=' * 80}")

    print(f"\n{'─'*65}")
    print(f"📌 处理 Clover 簇 (方案 A + D)")
    print(f"   总簇数: {len(clover_clusters):,}")
    print(f"{'─'*65}")

    consensus_A = {}
    consensus_D = {}
    processed = 0
    for cid, read_indices in clover_clusters.items():
        if len(read_indices) < 2:
            continue

        reads_seqs = [data_loader.reads[i] for i in read_indices]

        # D: raw vote
        consensus_D[cid] = majority_vote_raw(reads_seqs, ref_len_sample)

        # A: FedDNA eq
        evidence = infer_cluster_evidence(model, reads_seqs, model_max_len, device)
        fused = fuse_equal_weight(evidence)
        consensus_A[cid] = decode_evidence(fused)

        processed += 1
        if processed % 10000 == 0:
            print(f"      进度: {processed:,}", flush=True)

    print(f"    ✅ 完成: {processed:,} 簇")

    # =================================================================
    # 7. SSI-EC: 方案 B (eq) + C (zone-aware)
    # =================================================================
    consensus_B = {}
    consensus_C = {}
    if ssi_labels is not None:
        print(f"\n{'─'*65}")
        print(f"📌 处理 SSI-EC 簇 (方案 B + C)")
        print(f"   总簇数: {len(ssi_clusters):,}")
        print(f"{'─'*65}")

        zone_full = ssi_state['zone_ids']
        str_full = ssi_state['strength']

        processed = 0
        for cid, read_indices in ssi_clusters.items():
            if len(read_indices) < 2:
                continue

            reads_seqs = [data_loader.reads[i] for i in read_indices]
            zones = [int(zone_full[i]) for i in read_indices]
            strengths = [float(str_full[i]) for i in read_indices]

            evidence = infer_cluster_evidence(model, reads_seqs, model_max_len, device)

            fused_b = fuse_equal_weight(evidence)
            consensus_B[cid] = decode_evidence(fused_b)

            fused_c = fuse_zone_aware(evidence, strengths, zones)
            consensus_C[cid] = decode_evidence(fused_c)

            processed += 1
            if processed % 10000 == 0:
                print(f"      进度: {processed:,}", flush=True)

        print(f"    ✅ 完成: {processed:,} 簇")

    # =================================================================
    # 8. Post-proc: 方案 B' + C'
    # =================================================================
    consensus_B2 = {}
    consensus_C2 = {}
    if pp_labels is not None and ssi_state is not None:
        print(f"\n{'─'*65}")
        print(f"📌 处理 Post-proc 簇 (方案 B' + C')")
        print(f"   总簇数: {len(pp_clusters):,}")
        print(f"{'─'*65}")

        pp_zone = ssi_state['zone_ids'].copy()
        pp_str = ssi_state['strength'].copy()
        if ssi_labels is not None:
            for i in range(TOTAL):
                if int(ssi_labels[i]) == -1 and int(pp_labels[i]) >= 0:
                    pp_zone[i] = 2
                    pp_str[i] = 0.1

        processed = 0
        for cid, read_indices in pp_clusters.items():
            if len(read_indices) < 2:
                continue

            reads_seqs = [data_loader.reads[i] for i in read_indices]
            zones = [int(pp_zone[i]) for i in read_indices]
            strengths = [float(pp_str[i]) for i in read_indices]

            evidence = infer_cluster_evidence(model, reads_seqs, model_max_len, device)

            fused_b = fuse_equal_weight(evidence)
            consensus_B2[cid] = decode_evidence(fused_b)

            fused_c = fuse_zone_aware(evidence, strengths, zones)
            consensus_C2[cid] = decode_evidence(fused_c)

            processed += 1
            if processed % 10000 == 0:
                print(f"      进度: {processed:,}", flush=True)

        print(f"    ✅ 完成: {processed:,} 簇")

    # =================================================================
    # 9. 评估 + 汇总
    # =================================================================
    print(f"\n{'=' * 80}")
    print("📊 评估结果")
    print(f"{'=' * 80}")

    all_metrics = []

    m = evaluate_scheme(consensus_D, clover_c2g, gt_refs, "D: Clover+Raw vote")
    print_scheme(m); all_metrics.append(m) if m else None

    m = evaluate_scheme(consensus_A, clover_c2g, gt_refs, "A: Clover+FedDNA eq")
    print_scheme(m); all_metrics.append(m) if m else None

    if ssi_c2g:
        m = evaluate_scheme(consensus_B, ssi_c2g, gt_refs, "B: SSI-EC+FedDNA eq")
        print_scheme(m); all_metrics.append(m) if m else None

        m = evaluate_scheme(consensus_C, ssi_c2g, gt_refs, "C: SSI-EC+FedDNA zone")
        print_scheme(m); all_metrics.append(m) if m else None

    if pp_c2g:
        m = evaluate_scheme(consensus_B2, pp_c2g, gt_refs, "B': PostProc+FedDNA eq")
        print_scheme(m); all_metrics.append(m) if m else None

        m = evaluate_scheme(consensus_C2, pp_c2g, gt_refs, "C': PostProc+FedDNA zone")
        print_scheme(m); all_metrics.append(m) if m else None

    # 汇总表
    if all_metrics:
        print(f"\n{'=' * 100}")
        print("📋 汇总表")
        print(f"{'=' * 100}")
        print(f"  {'方案':<28s} {'Clusters':>9s} {'Success%':>10s} {'EER%':>10s} "
              f"{'MeanED':>8s} {'ED≤3':>7s} {'ED≤10':>7s}")
        print("  " + "─" * 90)
        for m in all_metrics:
            ed = m['ed_arr']
            n = m['evaluated']
            ed3 = (ed <= 3).sum() / n * 100
            ed10 = (ed <= 10).sum() / n * 100
            print(f"  {m['name']:<28s} {n:>9,} "
                  f"{m['success_rate']*100:>9.2f}% {m['mean_eer']*100:>9.4f}% "
                  f"{m['mean_ed']:>8.2f} {ed3:>6.2f}% {ed10:>6.2f}%")

    print(f"\n{'=' * 100}")
    print("💡 解读:")
    print("  B vs A → SSI-EC 聚类改善的贡献")
    print("  C vs B → Zone-aware fusion 的贡献")
    print("  C vs A → SSI-EC 整体贡献 (聚类+重建)")
    print("  D vs A → FedDNA evidence vs 纯碱基投票")
    print("✅ 完成")


if __name__ == "__main__":
    main()