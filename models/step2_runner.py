# models/step2_runner.py
"""
Step2: Evidence-Guided Clustering Refinement

修复清单:
  [FIX-EMA]    删除动量更新 Strength
  [FIX-FASTA]  修复 Fasta 输出尾部填充 A 的 Bug
  [FIX-P0]     next_round_files 增加 'consensus' 和 'cluster_change_info' 键
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import gc
import sys
from datetime import datetime
from collections import defaultdict
from typing import Dict, Optional
from models.merge_clusters import merge_close_centroids

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.step1_model import Step1EvidentialModel, decompose_uncertainty
from models.step1_data  import CloverDataLoader, Step1Dataset, _BASE_LUT
from models.step1_visualizer import Step1Visualizer
from models.step2_refine import (split_confidence_by_zone,
                                 compute_centroids_weighted,
                                 compute_global_delta,
                                 refine_reads)


def compute_consensus_from_memory(
    reads_list, labels_np, strength_np, flat_real_indices, max_len, round_idx=1
) -> Dict[int, torch.Tensor]:
    cluster_to_didx: Dict[int, list] = defaultdict(list)
    for didx, label in enumerate(labels_np):
        if label >= 0:
            cluster_to_didx[int(label)].append(didx)

    consensus_dict: Dict[int, torch.Tensor] = {}
    for cluster_id, didx_list in cluster_to_didx.items():
        if len(didx_list) < 1:
            continue
        # [FIX-N] 第5列(index 4)静默吸收 'N'，避免 np.add.at IndexError
        vote_matrix = np.zeros((max_len, 5), dtype=np.float64)
        for didx in didx_list:
            real_idx = flat_real_indices[didx]
            read = reads_list[real_idx] if real_idx < len(reads_list) else ''
            s = float(strength_np[didx]) if round_idx >= 2 else 1.0
            L = min(len(read), max_len)
            if L > 0:
                # ★ 向量化：numpy LUT 替代 Python 字典循环（~105x 加速）
                byte_arr = np.frombuffer(read[:L].encode('ascii'), dtype=np.uint8)
                indices = _BASE_LUT[byte_arr]
                np.add.at(vote_matrix[:L], (np.arange(L), indices), s)
        # ★ 修复 padding：argmax([0,0,0,0])=0 → 'A' one-hot，
        #   会让 masked_bayes_risk 把 padding 位置误判为有效碱基，污染 Loss
        # [FIX-N] 仅对 ACGT 4列计算 has_vote 和 argmax，第5列('N')自然截断
        has_vote = vote_matrix[:, :4].sum(axis=1) > 0
        consensus_indices = vote_matrix[:, :4].argmax(axis=1)
        one_hot = np.eye(4, dtype=np.float32)[consensus_indices]
        one_hot[~has_vote] = 0.0                       # padding → all zeros
        consensus_dict[cluster_id] = torch.from_numpy(one_hot)
    return consensus_dict


def compute_cluster_difficulty(new_labels_np, flat_real_indices, strength_np) -> Dict[int, float]:
    """
    基于合并后新簇内 reads 的 strength 分布，衡量簇的困难程度。

    原 compute_cluster_change_info 的问题:
      MNN 合并后 67K→15K，几乎所有旧 cluster_id 都发生变化，
      导致 change_frac ≈ 1.0 对绝大多数簇成立，采样退化为全量训练。

    新指标: 变异系数 CV = std(strength) / mean(strength)
      - 纯净簇: reads 来自同一分子，strength 高且集中 → 低 CV → 完美簇
      - 混簇:   reads 来自多个分子，strength 低且分散 → 高 CV → 困难簇
      - 阈值 CV_THRESHOLD = 0.3（可通过 args.cv_threshold 覆盖）

    Returns:
        {cluster_id: cv_value}  cv 越高越困难
    """
    cluster_strengths: Dict[int, list] = defaultdict(list)
    for didx, label in enumerate(new_labels_np):
        if label >= 0:
            cluster_strengths[int(label)].append(float(strength_np[didx]))

    difficulty: Dict[int, float] = {}
    for cid, strengths in cluster_strengths.items():
        if len(strengths) < 2:
            difficulty[cid] = 0.0   # 单 read 簇，无法判断，视为完美
            continue
        arr = np.array(strengths)
        mean_s = arr.mean()
        cv = arr.std() / (mean_s + 1e-6)
        difficulty[cid] = float(cv)

    return difficulty


def _evaluate_with_gt_numpy(consensus_dict, gt_labels_list, new_labels,
                            flat_real_indices, output_dir):
    """
    正确的聚类质量评估：GT cluster_id 与 Clover cluster_id 是两套独立编号，
    不能直接比较数值。正确做法：评估每个预测簇的 Purity 和 Perfect Cluster Rate。

    Purity：对每个预测簇，找簇内最多的 GT label，计算该比例的读数加权均值。
    Perfect Cluster Rate：簇内所有 reads 来自同一 GT 分子的簇 / 总簇数，是重建成功率上界。
    """
    try:
        from collections import Counter
        gt_arr  = np.array(gt_labels_list)
        new_arr = new_labels.cpu().numpy() if isinstance(new_labels, torch.Tensor) else new_labels

        cluster_gt_counter = defaultdict(Counter)
        for didx, (new_l, real_idx) in enumerate(zip(new_arr, flat_real_indices)):
            if new_l < 0:
                continue
            gt = int(gt_arr[real_idx]) if real_idx < len(gt_arr) else -1
            if gt >= 0:
                cluster_gt_counter[int(new_l)][gt] += 1

        total_weighted   = 0
        total_reads_eval = 0
        perfect_clusters = 0
        total_clusters   = len(cluster_gt_counter)

        for cid, counter in cluster_gt_counter.items():
            cluster_size = sum(counter.values())
            majority     = counter.most_common(1)[0][1]
            total_weighted   += majority
            total_reads_eval += cluster_size
            if majority == cluster_size:
                perfect_clusters += 1

        purity       = total_weighted   / max(total_reads_eval, 1)
        perfect_rate = perfect_clusters / max(total_clusters,   1)

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


def _record_paper_log_safe(args, total_reads, refine_stats, consensus_dict,
                           avg_strength, delta):
    try:
        log_dir = os.path.join(args.output_dir, "paper_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"round{args.round_idx}_stats.txt")
        with open(log_path, 'w') as f:
            f.write(f"Round: {args.round_idx}\n")
            f.write(f"Total Reads: {total_reads}\n")
            f.write(f"Delta: {delta:.4f}\n")
            f.write(f"Avg Strength: {avg_strength:.4f}\n")
            f.write(f"Consensus Clusters: {len(consensus_dict)}\n")
            if refine_stats:
                for k, v in refine_stats.items():
                    f.write(f"{k}: {v}\n")
        print(f"   📝 论文数据: {log_path}")
    except Exception as e:
        print(f"   ⚠️ 论文数据记录失败: {e}")


def run_step2(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    round_idx = getattr(args, 'round_idx', 1)
    os.makedirs(args.output_dir, exist_ok=True)

    # =====================================================================
    # 1. 加载模型与数据
    # =====================================================================
    print("\n" + "=" * 60)
    print("📦 加载模型与数据")
    print("=" * 60)

    try:
        checkpoint = torch.load(args.step1_checkpoint, map_location=device)
        step1_args = checkpoint.get('args', {})
        model_dim = step1_args.get('dim', args.dim)
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
        if sh[0] == model_max_len:
            model.length_adapter = nn.Linear(sh[1], sh[0]).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    del checkpoint, sd
    gc.collect()

    # =====================================================================
    # 2. 推理（全量）
    # =====================================================================
    print("\n" + "=" * 60)
    print("🔮 推理 (提取 Embeddings)")
    print("=" * 60)

    dataset = Step1Dataset(data_loader, max_len=model_max_len, inference_mode=True)
    print(f"   🔮 全量推理: {TOTAL_READS} reads (label >= 0)")

    inference_loader = torch.utils.data.DataLoader(
        dataset, batch_size=getattr(args, 'batch_size', 1024),
        shuffle=False, num_workers=0, pin_memory=False
    )

    N = len(dataset)
    D = model_dim
    print(f"   📦 预分配: {N} samples × {D} dim (float16)", flush=True)

    embeddings  = torch.zeros(N, D, dtype=torch.float16)
    strength    = torch.zeros(N)
    u_epi       = torch.zeros(N)
    u_ale       = torch.zeros(N)
    flat_real_indices = [dataset.valid_indices[i] for i in range(N)]

    ptr = 0
    with torch.no_grad():
        for batch_data in inference_loader:
            reads_batch  = batch_data['encoding'].to(device)
            bs = reads_batch.shape[0]
            emb, pooled = model.encode_reads(reads_batch)
            ev, stre, alpha = model.decode_to_evidence(emb)
            u_e, u_a = decompose_uncertainty(alpha)
            embeddings[ptr:ptr+bs] = pooled.cpu().half()
            strength[ptr:ptr+bs]   = stre.mean(dim=-1).cpu()
            u_epi[ptr:ptr+bs]      = u_e.cpu()
            u_ale[ptr:ptr+bs]      = u_a.cpu()
            ptr += bs
            del reads_batch, emb, pooled, ev, stre, alpha, u_e, u_a
            if ptr % 100000 < 1024:
                torch.cuda.empty_cache()

    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"   ✅ 推理完成: {N} samples")

    # [FIX-EMA] 动量更新已删除。原 EMA (0.7/0.3) 污染 U_ale 分布，
    # 导致 Zone III 从 10% 雪崩到 24%+，直接使用本轮原始 strength。

    embeddings_f32 = embeddings.float()
    del embeddings
    gc.collect()

    _np_u_epi    = u_epi.numpy().copy()
    _np_u_ale    = u_ale.numpy().copy()
    _np_strength = strength.numpy().copy()

    # =====================================================================
    # 3. 三区制划分
    # =====================================================================
    print("\n" + "=" * 60)
    print("🔪 三区制划分")
    print("=" * 60)

    labels_tensor = torch.tensor(
        [data_loader.clover_labels[flat_real_indices[i]] for i in range(N)],
        dtype=torch.long
    )

    zone_ids, zone_stats = split_confidence_by_zone(u_epi, u_ale, labels_tensor)
    _np_zone_ids = zone_ids.numpy().copy()

    # =====================================================================
    # 4. 质心计算 + 标签修正
    # =====================================================================
    print("\n" + "=" * 60)
    print("🎯 质心 + 标签修正")
    print("=" * 60)

    old_labels_np = labels_tensor.numpy().copy()

    centroids, cluster_sizes = compute_centroids_weighted(
        embeddings_f32, labels_tensor, strength, zone_ids
    )

    # ★ 安全簇合并 (MNN + 序列双重校验 + 最大簇大小约束)
    # 序列校验用上一轮的 consensus_dict（当轮的在 Section 6 才生成）
    prev_consensus_dict = None
    prev_consensus_path = getattr(args, 'consensus_path', None)
    if prev_consensus_path and os.path.exists(prev_consensus_path):
        try:
            prev_consensus_dict = torch.load(prev_consensus_path, map_location='cpu')
            prev_consensus_dict = {int(k): v for k, v in prev_consensus_dict.items()}
            print(f"   📂 加载上一轮 consensus 用于序列校验: {len(prev_consensus_dict)} 个簇")
        except Exception as e:
            print(f"   ⚠️ 加载上一轮 consensus 失败: {e}，跳过序列校验")
            prev_consensus_dict = None

    centroids, labels_tensor, merge_stats = merge_close_centroids(
        centroids, labels_tensor, cluster_sizes,
        embeddings_f32, zone_ids, strength,
        threshold=0.98,
        max_cluster_size=2000,
        max_rounds=60,
        consensus_dict=prev_consensus_dict,
        seq_jaccard_threshold=0.5,
    )

    delta = compute_global_delta(embeddings_f32, labels_tensor, zone_ids, centroids)
    # [FIX-ZONE2] 不再做 Zone II 重分配：Zone I 可能无法覆盖全部 GT，
    #             强制重分配会丢失 Zone II 中的有效 reads。
    new_labels = labels_tensor          # 合并后的标签直接作为最终标签
    noise_mask = (labels_tensor < 0)
    refine_stats = {'zone2_reassigned': 0, 'zone2_noise': 0}
    # 注意: delta 保留用于 _record_paper_log_safe 日志，不影响标签。

    _avg_strength = float(_np_strength.mean())
    del embeddings_f32
    gc.collect()

    # =====================================================================
    # 5. GT 评估
    # =====================================================================
    if gt_tags_file and os.path.exists(gt_tags_file):
        _evaluate_with_gt_numpy(
            {}, data_loader.gt_labels, new_labels, flat_real_indices, args.output_dir
        )

    # =====================================================================
    # 6. Consensus 计算（CPU only）
    # =====================================================================
    print("\n" + "=" * 60)
    print("🧬 [FIX] Consensus 计算（CPU only，利用已有 strength）")
    print("=" * 60)

    new_labels_np = new_labels.cpu().numpy()

    # [FIX-DECODE] 用 FedDNA ds_fusion 替换 majority-vote consensus
    from models.step2_decode import run_feddna_decode
    # 模型已在 model.cpu() 后，需要重新上 GPU
    model.to(device)
    consensus_dict = run_feddna_decode(
        model=model,
        data_loader=data_loader,
        new_labels_np=new_labels_np,
        flat_real_indices=flat_real_indices,
        model_max_len=model_max_len,
        device=device,
        batch_size=getattr(args, 'batch_size', 512),
    )
    model.cpu()
    torch.cuda.empty_cache()
    print(f"   ✅ 生成 {len(consensus_dict)} 个簇的 consensus")

    cv_values = list(cluster_change_info.values()) if cluster_change_info else []
    median_cv = float(np.median(cv_values)) if cv_values else 0.05
    cv_threshold = max(0.05, median_cv * 2.5)
    print(f"   📊 动态 CV 阈值: 中位CV={median_cv:.4f} × 2.5 = {cv_threshold:.4f}")
    cluster_change_info = compute_cluster_difficulty(
        new_labels_np, flat_real_indices, _np_strength
    )
    hard_clusters = sum(1 for v in cluster_change_info.values() if v >= cv_threshold)
    easy_clusters = len(cluster_change_info) - hard_clusters
    cv_values = list(cluster_change_info.values())
    cv_median  = float(np.median(cv_values)) if cv_values else 0.0
    print(f"   ✅ cluster_difficulty (CV): 困难簇(≥{cv_threshold})={hard_clusters}, "
          f"完美簇={easy_clusters}, 中位CV={cv_median:.3f}")

    # =====================================================================
    # 7. 保存输出
    # =====================================================================
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

    full_u_epi    = np.zeros(TOTAL_READS, dtype=np.float32)
    full_u_ale    = np.zeros(TOTAL_READS, dtype=np.float32)
    full_strength = np.zeros(TOTAL_READS, dtype=np.float32)
    full_zone_ids = np.zeros(TOTAL_READS, dtype=np.int64)
    full_u_epi[flat_real_indices]    = _np_u_epi
    full_u_ale[flat_real_indices]    = _np_u_ale
    full_strength[flat_real_indices] = _np_strength
    full_zone_ids[flat_real_indices] = _np_zone_ids

    state_path = os.path.join(next_round_dir, f"read_state_{ts}.pt")
    torch.save({'u_epi': full_u_epi, 'u_ale': full_u_ale,
                'strength': full_strength, 'zone_ids': full_zone_ids,
                'round_idx': round_idx}, state_path)
    print(f"   💾 状态: {state_path}")
    del full_u_epi, full_u_ale, full_strength, full_zone_ids
    gc.collect()

    centroids_path = os.path.join(next_round_dir, f"centroids_{ts}.pt")
    torch.save({'centroids': centroids, 'cluster_sizes': cluster_sizes,
                'delta': delta, 'round_idx': round_idx}, centroids_path)
    print(f"   💾 质心: {centroids_path}")

    consensus_path = os.path.join(next_round_dir, f"consensus_dict_{ts}.pt")
    torch.save(consensus_dict, consensus_path)
    print(f"   💾 Consensus Dict: {consensus_path}")

    change_info_path = os.path.join(next_round_dir, f"cluster_change_info_{ts}.pt")
    torch.save(cluster_change_info, change_info_path)
    print(f"   💾 Cluster Change Info: {change_info_path}")

    # [FIX-DECODE] 用 save_consensus_fasta 替换原 FASTA 保存逻辑
    from models.step2_decode import save_consensus_fasta
    fasta_dir = os.path.join(args.output_dir, "consensus")
    os.makedirs(fasta_dir, exist_ok=True)
    fasta_path = os.path.join(fasta_dir, f"consensus_{ts}.fasta")
    try:
        save_consensus_fasta(
            consensus_dict, new_labels_np, flat_real_indices,
            data_loader, model_max_len, fasta_path
        )
        print(f"   💾 Fasta: {fasta_path}")
    except Exception as e:
        print(f"   ⚠️ Fasta 保存失败: {e}")
        fasta_path = None

    # =====================================================================
    # 8. 论文数据 + 可视化
    # =====================================================================
    _record_paper_log_safe(args, TOTAL_READS, refine_stats, consensus_dict,
                           _avg_strength, delta)

    try:
        viz = Step1Visualizer(args.output_dir)
        viz.plot_uncertainty_distribution(
            torch.tensor(_np_u_epi), torch.tensor(_np_u_ale),
            torch.tensor(_np_zone_ids)
        )
    except Exception as e:
        print(f"   ⚠️ 可视化跳过: {e}")

    del _np_u_epi, _np_u_ale, _np_strength, _np_zone_ids
    gc.collect()

    return {
        'next_round_files': {
            'labels':              label_path,
            'state':               state_path,
            'centroids':           centroids_path,
            'reference':           fasta_path,
            'consensus':           consensus_path,
            'cluster_change_info': change_info_path,
        }
    }