"""
eval_reconstruction.py - Per-Pred-Cluster 重建评估 (极速 + 省存版)

核心思想:
  SSI-EC 贡献: 聚类 + zone/strength 质量信号
  FedDNA 贡献: read→reference 的 evidence 重建能力
  两者结合: zone-aware evidence fusion

v2 修复:
  [FIX] 超大簇采样上限 MAX_READS_PER_CLUSTER=5000
  [FIX] majority_vote_raw 用 numpy 向量化加速
  [FIX] 进度打印每 5000 簇一次
"""
import os
import re
import sys
import gc
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

CODE_DIR = "/mnt/st_data/liangxinyi/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.Model import Encoder, Model as FedDNAModel
from models.step1_data import CloverDataLoader, seq_to_onehot

# ================= 路径配置 =================
EXP_DIR    = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_Real_last"
GT_REFS    = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1/exp1_refs.fasta"
GT_TAGS    = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1/exp1_tags_reads.txt"
FEDDNA_CKP = "/mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
PAD_LEN    = 155     
OUT_LEN    = 150     
DIM        = 256
BATCH_INFER = 1024
MAX_READS_PER_CLUSTER = 5000   # 超大簇采样上限
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
            if not line: continue
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
    """纯碱基多数投票 - numpy 向量化版"""
    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    char_to_idx = {'A':0,'C':1,'G':2,'T':3,'a':0,'c':1,'g':2,'t':3,'N':0,'n':0}
    votes = np.zeros((out_len, 4), dtype=np.int32)
    for read in reads_list:
        L = min(len(read), out_len)
        for pos in range(L):
            votes[pos, char_to_idx.get(read[pos], 0)] += 1
    return ''.join([base_map[i] for i in np.argmax(votes, axis=1)])


def cap_indices(read_indices, max_n=MAX_READS_PER_CLUSTER):
    """超大簇随机采样"""
    if len(read_indices) <= max_n:
        return read_indices
    return random.sample(read_indices, max_n)


def load_feddna_model(ckpt_path, device):
    encoder = Encoder(dim=DIM).to(device)
    model = FedDNAModel(encoder, dim=DIM, noise_length=PAD_LEN, label_length=OUT_LEN).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    sd = checkpoint.get('state_dict', checkpoint)
    if 'model_state_dict' in checkpoint:
        sd = checkpoint['model_state_dict']
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"    ✅ 原生 FedDNA 模型加载完毕 (pad={PAD_LEN}, out={OUT_LEN})")
    return model


# ---------------------------------------------------------------------------
# Fusion 策略
# ---------------------------------------------------------------------------
def fuse_equal_weight(evidence):
    return evidence.mean(dim=0)


def fuse_zone_aware(evidence, strengths, zones):
    keep_mask = torch.tensor([z in (1, 2) for z in zones])
    if keep_mask.sum() == 0:
        return fuse_equal_weight(evidence)

    kept_evidence = evidence[keep_mask]
    kept_strength = torch.tensor([s for s, z in zip(strengths, zones) if z in (1, 2)])
    
    weights = F.softmax(torch.log(kept_strength + 1), dim=0)
    fused = (kept_evidence * weights.view(-1, 1, 1)).sum(dim=0)
    return fused


def decode_evidence(fused_evidence):
    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    alpha = fused_evidence + 1.0
    pred = torch.argmax(alpha, dim=-1)
    return ''.join([base_map[i.item()] for i in pred])


# ---------------------------------------------------------------------------
# 评估核心
# ---------------------------------------------------------------------------
def build_cluster_gt_map(pred_labels, gt_labels):
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
    exact = 0
    evaluated = 0
    eds = []
    eers = []
    for cid, consensus in consensus_seqs.items():
        gt_id = cluster_to_gt.get(cid)
        if gt_id is None or gt_id not in gt_refs: continue
        gt_seq = gt_refs[gt_id]
        ed = edit_distance(consensus, gt_seq)
        eds.append(ed)
        eers.append(ed / max(len(gt_seq), 1))
        if ed == 0: exact += 1
        evaluated += 1

    if evaluated == 0: return None
    ed_arr = np.array(eds)
    eer_arr = np.array(eers)
    return {
        'name': scheme_name, 'evaluated': evaluated, 'exact': exact,
        'success_rate': exact / evaluated, 'mean_ed': ed_arr.mean(),
        'median_ed': np.median(ed_arr), 'mean_eer': eer_arr.mean(), 'ed_arr': ed_arr,
    }


def print_scheme(m):
    if m is None: return
    print(f"\n    [{m['name']}]")
    print(f"      Clusters 评估: {m['evaluated']:,}")
    print(f"      ✅ Success Rate (ED=0):  {m['exact']:,}/{m['evaluated']:,} = {m['success_rate']*100:.2f}%")
    print(f"      📏 Edit Error Rate:      {m['mean_eer']*100:.4f}%")
    print(f"      📏 Mean ED: {m['mean_ed']:.2f}  |  Median: {m['median_ed']:.1f}")
    ed, n = m['ed_arr'], m['evaluated']
    for t in [0, 1, 2, 3, 5, 10]:
        c = int((ed <= t).sum())
        print(f"        ED≤{t:2d}: {c:>7,} ({c/n*100:6.2f}%)")
    c_high = int((ed > 10).sum())
    print(f"        ED>10: {c_high:>7,} ({c_high/n*100:6.2f}%)")


# ---------------------------------------------------------------------------
# 通用簇处理函数
# ---------------------------------------------------------------------------
def process_clusters(clusters, evidence_all, data_loader, ref_len,
                     zone_full=None, str_full=None, do_raw_vote=False):
    """
    统一处理一组簇的重建

    Returns:
        consensus_eq:   dict {cid: seq}  等权 fusion
        consensus_zone: dict {cid: seq}  zone-aware fusion (如果 zone_full 提供)
        consensus_raw:  dict {cid: seq}  raw vote (如果 do_raw_vote=True)
    """
    consensus_eq = {}
    consensus_zone = {} if zone_full is not None else None
    consensus_raw = {} if do_raw_vote else None

    total = len(clusters)
    processed = 0

    for cid, read_indices in clusters.items():
        if len(read_indices) < 2:
            continue

        # 采样上限
        sampled = cap_indices(read_indices)

        # FedDNA evidence fusion (等权)
        evidence = evidence_all[sampled].float()
        consensus_eq[cid] = decode_evidence(fuse_equal_weight(evidence))

        # Zone-aware fusion
        if zone_full is not None and str_full is not None:
            zones = [int(zone_full[i]) for i in sampled]
            strengths = [float(str_full[i]) for i in sampled]
            consensus_zone[cid] = decode_evidence(
                fuse_zone_aware(evidence, strengths, zones))

        # Raw majority vote
        if do_raw_vote:
            vote_indices = cap_indices(read_indices, 2000)  # vote 更慢，cap 更小
            reads_seqs = [data_loader.reads[i] for i in vote_indices]
            consensus_raw[cid] = majority_vote_raw(reads_seqs, ref_len)

        processed += 1
        if processed % 5000 == 0:
            print(f"      进度: {processed:,}/{total:,} 簇", flush=True)

    print(f"      ✅ 完成: {processed:,}/{total:,} 簇")
    return consensus_eq, consensus_zone, consensus_raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(42)
    device = torch.device(DEVICE)

    print("[1] 加载数据...")
    data_loader = CloverDataLoader(EXP_DIR)
    TOTAL = len(data_loader.reads)
    clover_labels = np.array(data_loader.clover_labels, dtype=int)
    print(f"    Reads: {TOTAL:,}")

    print("[2] 加载 GT...")
    data_loader.load_gt_tags(GT_TAGS)
    gt_labels_arr = np.array(data_loader.gt_labels, dtype=int)
    gt_refs = parse_gt_refs(GT_REFS)
    ref_len_sample = len(next(iter(gt_refs.values()))) if gt_refs else OUT_LEN
    print(f"    GT refs: {len(gt_refs):,}, ref 长度: {ref_len_sample}bp")

    print("[3] 加载 SSI-EC labels...")
    labels_dir = os.path.join(EXP_DIR, "04_Iterative_Labels")
    ROUND3_TS = "201006"
    r3_label = os.path.join(labels_dir, f"refined_labels_{ROUND3_TS}.txt")
    r3_state = os.path.join(labels_dir, f"read_state_{ROUND3_TS}.pt")

    ssi_labels, ssi_state = None, None
    if os.path.exists(r3_label):
        ssi_labels = np.loadtxt(r3_label, dtype=int)
        ssi_state = torch.load(r3_state, map_location='cpu')
        print(f"    SSI-EC Round 3: refined_labels_{ROUND3_TS}.txt")
        print(f"    已分配: {(ssi_labels>=0).sum():,}")

    final_path = os.path.join(EXP_DIR, "results/final/final_labels.txt")
    pp_labels = None
    if os.path.exists(final_path):
        pp_labels = np.loadtxt(final_path, dtype=int)
        print(f"    Post-proc: {(pp_labels>=0).sum():,}")

    print("[4] 加载 FedDNA 模型...")
    model = load_feddna_model(FEDDNA_CKP, device)

    # =================================================================
    # 全量预计算 Evidence
    # =================================================================
    print(f"\n[4.5] 🚀 全局预计算 Evidence (Float16 省存)...")
    evidence_all = torch.zeros((TOTAL, OUT_LEN, 4), dtype=torch.float16)

    model.eval()
    with torch.no_grad():
        for start in range(0, TOTAL, BATCH_INFER):
            end = min(start + BATCH_INFER, TOTAL)
            batch_seqs = data_loader.reads[start:end]

            reads_tensor = torch.stack([
                seq_to_onehot(s, PAD_LEN) for s in batch_seqs
            ]).to(device)
            reads_tensor = reads_tensor.unsqueeze(1)

            fused_evid = model(reads_tensor)
            evidence_all[start:end] = fused_evid.cpu().half()

            if (start // BATCH_INFER) % 500 == 0:
                print(f"      推理进度: {start:>9,} / {TOTAL:>9,} "
                      f"({start/TOTAL*100:.1f}%)", flush=True)

    del model
    torch.cuda.empty_cache()
    mem_gb = evidence_all.element_size() * evidence_all.nelement() / 1e9
    print(f"      ✅ Evidence 完成 ({mem_gb:.2f} GB), GPU 已释放")

    # =================================================================
    # 建立映射
    # =================================================================
    print("\n[5] 建立 cluster→GT 映射...")
    clover_c2g = build_cluster_gt_map(clover_labels, gt_labels_arr)
    ssi_c2g = build_cluster_gt_map(ssi_labels, gt_labels_arr) if ssi_labels is not None else None
    # Post-proc 跳过
    pp_c2g = None

    def build_cluster_index(labels_arr):
        idx = defaultdict(list)
        for i in range(len(labels_arr)):
            if int(labels_arr[i]) >= 0:
                idx[int(labels_arr[i])].append(i)
        return idx

    clover_clusters = build_cluster_index(clover_labels)
    ssi_clusters = build_cluster_index(ssi_labels) if ssi_labels is not None else {}
    pp_clusters = build_cluster_index(pp_labels) if pp_labels is not None else {}

    print(f"    Clover: {len(clover_clusters):,} 簇")
    if ssi_labels is not None:
        print(f"    SSI-EC: {len(ssi_clusters):,} 簇")
    if pp_labels is not None:
        print(f"    Post-proc: {len(pp_clusters):,} 簇")

    # =================================================================
    # 逐方案处理
    # =================================================================
    print(f"\n{'=' * 80}")
    print(f"📊 Per-Pred-Cluster 重建评估")
    print(f"{'=' * 80}")

    # --- Clover (A + D) ---
    print(f"\n📌 Clover 簇 (方案 A: FedDNA eq + D: Raw vote)")
    cons_A, _, cons_D = process_clusters(
        clover_clusters, evidence_all, data_loader, ref_len_sample,
        do_raw_vote=True)

    # --- SSI-EC (B + C) ---
    cons_B, cons_C = {}, {}
    if ssi_labels is not None:
        print(f"\n📌 SSI-EC 簇 (方案 B: eq + C: zone-aware)")
        cons_B, cons_C, _ = process_clusters(
            ssi_clusters, evidence_all, data_loader, ref_len_sample,
            zone_full=ssi_state['zone_ids'], str_full=ssi_state['strength'])

    # --- Post-proc (B' + C') --- 跳过，77K簇太慢
    cons_B2, cons_C2 = {}, {}

    # =================================================================
    # 评估汇总
    # =================================================================
    all_metrics = []
    print(f"\n{'=' * 80}\n📊 评估结果\n{'=' * 80}")

    schemes = [
        (cons_D,  clover_c2g, "D: Clover+Raw vote"),
        (cons_A,  clover_c2g, "A: Clover+FedDNA eq"),
    ]
    if ssi_c2g:
        schemes += [
            (cons_B, ssi_c2g, "B: SSI-EC+FedDNA eq"),
            (cons_C, ssi_c2g, "C: SSI-EC+FedDNA zone"),
        ]
    if pp_c2g:
        schemes += [
            (cons_B2, pp_c2g, "B': PostProc+FedDNA eq"),
            (cons_C2, pp_c2g, "C': PostProc+FedDNA zone"),
        ]

    for cons, c2g, name in schemes:
        m = evaluate_scheme(cons, c2g, gt_refs, name)
        print_scheme(m)
        if m: all_metrics.append(m)

    # 汇总表
    if all_metrics:
        print(f"\n{'=' * 100}\n📋 汇总表\n{'=' * 100}")
        print(f"  {'方案':<28s} {'Clusters':>9s} {'Success%':>10s} "
              f"{'EER%':>10s} {'MeanED':>8s} {'ED≤3':>7s} {'ED≤10':>7s}")
        print("  " + "─" * 90)
        for m in all_metrics:
            ed, n = m['ed_arr'], m['evaluated']
            print(f"  {m['name']:<28s} {n:>9,} "
                  f"{m['success_rate']*100:>9.2f}% "
                  f"{m['mean_eer']*100:>9.4f}% "
                  f"{m['mean_ed']:>8.2f} "
                  f"{(ed<=3).sum()/n*100:>6.2f}% "
                  f"{(ed<=10).sum()/n*100:>6.2f}%")

    print(f"\n💡 解读:")
    print(f"  B vs A → SSI-EC 聚类改善")
    print(f"  C vs B → Zone-aware fusion 贡献")
    print(f"  C vs A → SSI-EC 整体贡献")
    print(f"  D vs A → FedDNA evidence vs 纯投票")
    print(f"✅ 完成")


if __name__ == "__main__":
    main()