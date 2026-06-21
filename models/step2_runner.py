# models/step2_runner.py
"""
Step2: Clustering Refinement & Consensus Decoding

终态流程 (一条线, 无旁支):
    加载模型与标签
      → 全量推理 (embedding + evidential uncertainty u_epi/u_ale/strength)
      → 簇内拆分 (intra-cluster split, 唯一改变 label 的迭代引擎)
      → MV consensus (训练靶子 + FASTA 评估)
      → 落盘 (labels / u_epi,u_ale,strength / consensus_dict / cluster_change_info)

簇内拆分是本框架唯一的迭代机制: 对每个簇按 edit 距离层次二分, 两子簇各自
MV consensus, 若两 consensus 的 edit >= tau 则判定 "两个分子被错并" 并拆开。
纯簇二分后两半 consensus 几乎一致, 过不了 tau 门控, 天然受保护。

不确定性 (u_epi 认知 / u_ale 偶然) 来自 Dirichlet evidence, 用于 Step1 的
不确定性加权对比损失, 并随结果落盘供分析。
"""
import torch
import torch.nn as nn
import numpy as np
import os
import gc
import sys
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.step1_model import Step1EvidentialModel, decompose_uncertainty
from models.step1_data  import CloverDataLoader, Step1Dataset
from models.step2_decode import (run_feddna_decode,
                                  compute_mv_consensus,
                                  save_consensus_fasta)
from models.cluster_split import split_clusters
from models.eval_reconstruction import levenshtein as _edit_distance


# ===========================================================================
# 簇困难度 (CV of strength): 供 Step1 动态采样区分难/易簇
# ===========================================================================
def compute_cluster_difficulty(new_labels_np, strength_np) -> Dict[int, float]:
    """
    用簇内 strength 的变异系数 (CV = std/mean) 衡量簇困难度。
      - 纯净簇: reads 同源, strength 高且集中 -> 低 CV
      - 混合簇: reads 异源, strength 分散     -> 高 CV
    """
    cluster_strengths: Dict[int, list] = defaultdict(list)
    for didx, label in enumerate(new_labels_np):
        if label >= 0:
            cluster_strengths[int(label)].append(float(strength_np[didx]))

    difficulty: Dict[int, float] = {}
    for cid, strengths in cluster_strengths.items():
        if len(strengths) < 2:
            difficulty[cid] = 0.0
            continue
        arr = np.array(strengths)
        difficulty[cid] = float(arr.std() / (arr.mean() + 1e-6))
    return difficulty


# ===========================================================================
# GT 聚类评估 (Purity / Perfect Cluster Rate) — 仅观测, 不参与迭代
# ===========================================================================
def _evaluate_with_gt(gt_labels_list, new_labels, flat_real_indices, output_dir):
    try:
        gt_arr  = np.array(gt_labels_list)
        new_arr = new_labels.cpu().numpy() if isinstance(new_labels, torch.Tensor) else new_labels

        cluster_gt_counter = defaultdict(Counter)
        for new_l, real_idx in zip(new_arr, flat_real_indices):
            if new_l < 0:
                continue
            gt = int(gt_arr[real_idx]) if real_idx < len(gt_arr) else -1
            if gt >= 0:
                cluster_gt_counter[int(new_l)][gt] += 1

        total_weighted = total_reads_eval = perfect_clusters = 0
        total_clusters = len(cluster_gt_counter)
        for counter in cluster_gt_counter.values():
            size = sum(counter.values())
            majority = counter.most_common(1)[0][1]
            total_weighted   += majority
            total_reads_eval += size
            if majority == size:
                perfect_clusters += 1

        purity       = total_weighted / max(total_reads_eval, 1)
        perfect_rate = perfect_clusters / max(total_clusters, 1)
        print(f"\n   🎯 GT 聚类评估 (基于 {total_reads_eval:,} reads, {total_clusters:,} 簇):")
        print(f"      Cluster Purity:       {purity:.4f}  ({total_weighted:,}/{total_reads_eval:,})")
        print(f"      Perfect Cluster Rate: {perfect_clusters}/{total_clusters} ({perfect_rate:.4f})  ← 重建成功率上界")

        try:
            log_dir = os.path.join(output_dir, "paper_logs")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "gt_eval.txt"), 'a') as f:
                f.write(f"purity={purity:.4f}, perfect_rate={perfect_rate:.4f}, "
                        f"perfect={perfect_clusters}, total={total_clusters}\n")
        except Exception:
            pass
    except Exception as e:
        print(f"   ⚠️ GT 评估失败: {e}")


def _record_paper_log(args, total_reads, consensus_dict, avg_strength):
    try:
        log_dir = os.path.join(args.output_dir, "paper_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"round{args.round_idx}_stats.txt")
        with open(log_path, 'w') as f:
            f.write(f"Round: {args.round_idx}\n")
            f.write(f"Total Reads: {total_reads}\n")
            f.write(f"Avg Strength: {avg_strength:.4f}\n")
            f.write(f"Consensus Clusters: {len(consensus_dict)}\n")
        print(f"   📝 论文数据: {log_path}")
    except Exception as e:
        print(f"   ⚠️ 论文数据记录失败: {e}")


# ===========================================================================
# 主流程
# ===========================================================================
def run_step2(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    round_idx = getattr(args, 'round_idx', 1)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 加载模型与数据
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📦 加载模型与数据")
    print("=" * 60)

    try:
        checkpoint = torch.load(args.step1_checkpoint, map_location=device)
        step1_args = checkpoint.get('args', {})
        model_dim     = step1_args.get('dim', args.dim)
        model_max_len = step1_args.get('max_length', args.max_length)
        print(f"   ✅ 模型参数: Dim={model_dim}, MaxLen={model_max_len}")
    except Exception as e:
        print(f"   ❌ Checkpoint 加载失败: {e}"); return None

    try:
        labels_path = getattr(args, 'refined_labels', None)
        data_loader = CloverDataLoader(args.experiment_dir, labels_path=labels_path)
        TOTAL_READS = len(data_loader.reads)
        current_clusters = set(l for l in data_loader.clover_labels if l >= 0)
        num_clusters = max(50, len(current_clusters))
        print(f"   📊 数据: {TOTAL_READS} Reads, {len(current_clusters)} 有效簇")
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}"); return None

    gt_tags_file = getattr(args, 'gt_tags_file', None)
    if gt_tags_file and os.path.exists(gt_tags_file):
        data_loader.load_gt_tags(gt_tags_file)

    model = Step1EvidentialModel(
        dim=model_dim, max_length=model_max_len,
        num_clusters=num_clusters, device=str(device)
    ).to(device)
    sd = checkpoint['model_state_dict']
    if 'length_adapter.weight' in sd:
        sh = sd['length_adapter.weight'].shape
        if sh[1] == model_max_len and sh[0] == model_max_len:
            model.length_adapter = nn.Linear(sh[1], sh[0]).to(device)
            print(f"   🔧 恢复 length_adapter: Linear({sh[1]}, {sh[0]})")
        else:
            print(f"   ⚠️ checkpoint 的 length_adapter 维度 {sh} "
                  f"与 max_length={model_max_len} 不兼容，已跳过")
    model.load_state_dict(sd, strict=False)
    model.eval()
    del checkpoint, sd
    gc.collect()

    # ------------------------------------------------------------------
    # 2. 全量推理: embedding + evidential uncertainty
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🔮 推理 (提取 Embeddings + 不确定性)")
    print("=" * 60)

    dataset = Step1Dataset(data_loader, max_len=model_max_len, inference_mode=True)
    print(f"   🔮 全量推理: {TOTAL_READS} reads")

    inference_loader = torch.utils.data.DataLoader(
        dataset, batch_size=getattr(args, 'batch_size', 1024),
        shuffle=False, num_workers=0, pin_memory=False
    )

    N = len(dataset)
    D = model_dim
    # 只保留 pooled emb (N, D); 序列级 emb 太大 (~210GB), consensus 解码时重跑 encoder。
    strength = torch.zeros(N)
    u_epi    = torch.zeros(N)
    u_ale    = torch.zeros(N)
    flat_real_indices = [dataset.valid_indices[i] for i in range(N)]

    ptr = 0
    with torch.no_grad():
        for batch_data in inference_loader:
            reads_batch = batch_data['encoding'].to(device)
            bs = reads_batch.shape[0]
            emb, pooled = model.encode_reads(reads_batch)
            ev, stre, alpha = model.decode_to_evidence(emb)
            u_e, u_a = decompose_uncertainty(alpha)
            strength[ptr:ptr+bs] = stre.mean(dim=-1).cpu()
            u_epi[ptr:ptr+bs]    = u_e.cpu()
            u_ale[ptr:ptr+bs]    = u_a.cpu()
            ptr += bs
            del reads_batch, emb, pooled, ev, stre, alpha, u_e, u_a
            if ptr % 100000 < 1024:
                torch.cuda.empty_cache()

    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"   ✅ 推理完成: {N} samples")

    _np_u_epi    = u_epi.numpy().copy()
    _np_u_ale    = u_ale.numpy().copy()
    _np_strength = strength.numpy().copy()
    _avg_strength = float(_np_strength.mean())

    # 当前标签 (Clover 初始或上一轮 refined)
    labels_tensor = torch.tensor(
        [data_loader.clover_labels[flat_real_indices[i]] for i in range(N)],
        dtype=torch.long
    )
    new_labels_np = labels_tensor.cpu().numpy().copy()

    # ------------------------------------------------------------------
    # 3. 簇内拆分 (唯一的迭代引擎)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("✂️  簇内拆分 (唯一迭代引擎)")
    print("=" * 60)

    _split_tau      = getattr(args, 'split_tau', 5)
    _split_min_size = getattr(args, 'split_min_size', 6)
    _split_ref_len  = getattr(args, 'ref_length', None) or 196
    new_labels_np, _split_stats = split_clusters(
        new_labels_np=new_labels_np,
        flat_real_indices=flat_real_indices,
        data_loader=data_loader,
        levenshtein=_edit_distance,
        ref_length=_split_ref_len,
        tau=_split_tau,
        min_split_size=_split_min_size,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 4. GT 聚类评估 (仅观测)
    # ------------------------------------------------------------------
    if gt_tags_file and os.path.exists(gt_tags_file):
        _evaluate_with_gt(data_loader.gt_labels, new_labels_np,
                          flat_real_indices, args.output_dir)

    # ------------------------------------------------------------------
    # 5. Consensus 计算
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🧬 Consensus 计算")
    print("=" * 60)

    # MV consensus: 训练靶子 (打破 encoder 自污染闭环) + FASTA 评估同源
    consensus_dict = compute_mv_consensus(
        data_loader=data_loader,
        new_labels_np=new_labels_np,
        flat_real_indices=flat_real_indices,
        model_max_len=model_max_len,
        ref_length=getattr(args, 'ref_length', None),
    )

    # 簇困难度 (供 Step1 动态采样)
    cluster_change_info = compute_cluster_difficulty(new_labels_np, _np_strength)
    cv_threshold  = getattr(args, 'cv_threshold', 0.3)
    hard_clusters = sum(1 for v in cluster_change_info.values() if v >= cv_threshold)
    cv_values     = list(cluster_change_info.values())
    cv_median     = float(np.median(cv_values)) if cv_values else 0.0
    print(f"   ✅ cluster_difficulty: 困难簇(≥{cv_threshold})={hard_clusters}, "
          f"完美簇={len(cluster_change_info)-hard_clusters}, 中位CV={cv_median:.3f}")

    # ------------------------------------------------------------------
    # 6. 落盘
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("💾 保存状态")
    print("=" * 60)

    next_round_dir = os.path.join(args.experiment_dir, "04_Iterative_Labels")
    os.makedirs(next_round_dir, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")

    full_labels = np.full(TOTAL_READS, -1, dtype=int)
    full_labels[flat_real_indices] = new_labels_np
    label_path = os.path.join(next_round_dir, f"refined_labels_{ts}.txt")
    np.savetxt(label_path, full_labels, fmt='%d')
    print(f"   💾 标签: {label_path}")

    # 不确定性落盘 (evidential 核心输出, 供分析)
    full_u_epi    = np.zeros(TOTAL_READS, dtype=np.float32)
    full_u_ale    = np.zeros(TOTAL_READS, dtype=np.float32)
    full_strength = np.zeros(TOTAL_READS, dtype=np.float32)
    full_u_epi[flat_real_indices]    = _np_u_epi
    full_u_ale[flat_real_indices]    = _np_u_ale
    full_strength[flat_real_indices] = _np_strength
    state_path = os.path.join(next_round_dir, f"read_state_{ts}.pt")
    torch.save({'u_epi': full_u_epi, 'u_ale': full_u_ale,
                'strength': full_strength, 'round_idx': round_idx}, state_path)
    print(f"   💾 状态 (u_epi/u_ale/strength): {state_path}")
    del full_u_epi, full_u_ale, full_strength
    gc.collect()

    consensus_path = os.path.join(next_round_dir, f"consensus_dict_{ts}.pt")
    torch.save(consensus_dict, consensus_path)
    print(f"   💾 Consensus Dict: {consensus_path}")

    change_info_path = os.path.join(next_round_dir, f"cluster_change_info_{ts}.pt")
    torch.save(cluster_change_info, change_info_path)
    print(f"   💾 Cluster Change Info: {change_info_path}")

    # FASTA: MV consensus on 严格 labels (评估 SR 用, 与训练靶子同源)
    fasta_dir = os.path.join(args.output_dir, "consensus")
    os.makedirs(fasta_dir, exist_ok=True)
    fasta_path = os.path.join(fasta_dir, f"consensus_{ts}.fasta")
    try:
        save_consensus_fasta(
            consensus_dict, new_labels_np, flat_real_indices,
            data_loader, model_max_len, fasta_path,
            ref_length=getattr(args, 'ref_length', None),
        )
        print(f"   💾 Fasta: {fasta_path}")
    except Exception as e:
        print(f"   ⚠️ Fasta 保存失败: {e}")
        fasta_path = None

    # ------------------------------------------------------------------
    # 7. 论文数据
    # ------------------------------------------------------------------
    _record_paper_log(args, TOTAL_READS, consensus_dict, _avg_strength)

    del _np_u_epi, _np_u_ale, _np_strength
    gc.collect()

    return {
        'next_round_files': {
            'labels':              label_path,
            'state':               state_path,
            'reference':           fasta_path,
            'consensus':           consensus_path,
            'cluster_change_info': change_info_path,
        }
    }