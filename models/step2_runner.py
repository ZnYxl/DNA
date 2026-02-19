# models/step2_runner.py
"""
Step2 主入口: Evidence-Guided Refinement & Decoding (v2 精简版)

v2 变更:
  [DEL] 动量更新 Strength (MOMENTUM_CURR/PREV) — 无理论必要性
  [DEL] 噪声复活机制 (RESURRECTION_SENTINEL/哨兵标签) — 由 Post-processing 替代
  [NEW] 保存质心到磁盘 (供 Post-processing 使用)
  [NEW] 每轮评估指标 (ARI/NMI 等)

保留:
  [FIX-#1]  inference_mode=True → Step2 推理全量数据
  [FIX-MEM] 推理阶段不积累 evidence/alpha (省内存)
  [FIX-OOM] 推理后立即释放 GPU, 后续 refine 全在 CPU
  [FIX]     num_workers=0 避免 fork 大量 reads 卡死
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.step1_model import Step1EvidentialModel, decompose_uncertainty
from models.step1_data import CloverDataLoader, Step1Dataset
from models.step2_refine import (
    split_confidence_by_zone,
    compute_centroids_weighted,
    compute_global_delta,
    refine_reads
)
from models.step2_decode import (
    decode_cluster_consensus,
    save_consensus_sequences
)
from models.step1_visualizer import Step1Visualizer


@torch.no_grad()
def run_step2(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Step2 启动 | 设备: {device} | 轮次: {args.round_idx}")

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
        print(f"   ❌ Checkpoint 加载失败: {e}")
        return None

    try:
        labels_path = getattr(args, 'refined_labels', None)
        data_loader = CloverDataLoader(args.experiment_dir, labels_path=labels_path)
        TOTAL_READS = len(data_loader.reads)

        current_clusters = set(data_loader.clover_labels)
        if -1 in current_clusters:
            current_clusters.remove(-1)
        num_clusters = max(50, len(current_clusters))

        print(f"   📊 数据: {TOTAL_READS} Reads, {len(current_clusters)} 有效簇")
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return None

    # GT 标签加载 (可选)
    gt_tags_file = getattr(args, 'gt_tags_file', None)
    if gt_tags_file and os.path.exists(gt_tags_file):
        data_loader.load_gt_tags(gt_tags_file)

    # 模型初始化
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

    # =====================================================================
    # 2. 推理 (全量, 只保留标量: pooled_emb, strength, u_epi, u_ale)
    #    [v2] 不再有噪声复活, 直接推理 label >= 0 的 reads
    # =====================================================================
    print("\n" + "=" * 60)
    print("🔮 推理 (提取 Embeddings)")
    print("=" * 60)

    cap = getattr(args, 'training_cap', TOTAL_READS)
    use_full = (cap >= TOTAL_READS)

    if use_full:
        dataset = Step1Dataset(
            data_loader,
            max_len=model_max_len,
            inference_mode=True
        )
        print(f"   🔮 全量推理: {TOTAL_READS} reads (label >= 0)")
    else:
        dataset = Step1Dataset(
            data_loader,
            max_len=model_max_len,
            training_cap=cap,
            round_idx=args.round_idx,
            inference_mode=False
        )
        print(f"   🧪 测试模式: {cap} reads")

    inference_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1024, shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 预分配张量
    N = len(dataset)
    D = step1_args.get('dim', args.dim)
    print(f"   📦 预分配: {N} samples × {D} dim", flush=True)

    embeddings = torch.zeros(N, D)
    strength = torch.zeros(N)
    u_epi = torch.zeros(N)
    u_ale = torch.zeros(N)
    labels = torch.zeros(N, dtype=torch.long)
    flat_real_indices = torch.zeros(N, dtype=torch.long)

    offset = 0
    total_batches = len(inference_loader)
    for batch_idx, batch in enumerate(inference_loader):
        reads = batch['encoding'].to(device)
        lbls = batch['clover_label']
        idxs = batch['read_idx']
        B = reads.shape[0]

        # Padding / Truncation
        if reads.shape[1] != model_max_len:
            if reads.shape[1] < model_max_len:
                reads = F.pad(reads, (0, 0, 0, model_max_len - reads.shape[1]))
            else:
                reads = reads[:, :model_max_len, :]

        emb, pooled = model.encode_reads(reads)
        evid, stre, alph = model.decode_to_evidence(emb)
        epi, ale = decompose_uncertainty(alph)

        embeddings[offset:offset+B] = pooled.cpu()
        strength[offset:offset+B] = stre.mean(dim=1).cpu()
        u_epi[offset:offset+B] = epi.cpu()
        u_ale[offset:offset+B] = ale.cpu()
        flat_real_indices[offset:offset+B] = idxs

        if isinstance(lbls, torch.Tensor):
            labels[offset:offset+B] = lbls
        else:
            labels[offset:offset+B] = torch.tensor(lbls)

        offset += B

        if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == total_batches:
            print(f"      进度: {batch_idx + 1}/{total_batches}", flush=True)

    # 截断
    embeddings = embeddings[:offset]
    strength = strength[:offset]
    u_epi = u_epi[:offset]
    u_ale = u_ale[:offset]
    labels = labels[:offset]
    flat_real_indices = flat_real_indices[:offset].numpy()

    print(f"   ✅ 推理完成: {offset} samples, embeddings {embeddings.shape}", flush=True)

    # 释放 GPU
    print(f"   🗑️ 释放 GPU 显存...", flush=True)
    del model
    torch.cuda.empty_cache()
    print(f"   ✅ GPU 显存已释放", flush=True)

    # =====================================================================
    # 3. 三区制划分 & 质心 & Delta
    #    [v2] 不再有动量更新, 不再有哨兵标签
    # =====================================================================
    zone_ids, zone_stats = split_confidence_by_zone(u_epi, u_ale, labels)

    centroids, cluster_sizes = compute_centroids_weighted(
        embeddings, labels, strength, zone_ids
    )
    delta = compute_global_delta(embeddings, labels, zone_ids, centroids)

    # =====================================================================
    # 4. Zone-aware 修正
    # =====================================================================
    new_labels, noise_mask, refine_stats = refine_reads(
        embeddings, labels, zone_ids, centroids, delta, round_idx=args.round_idx
    )

    # =====================================================================
    # 5. Consensus 解码 (按簇重新推理 evidence)
    # =====================================================================
    print("\n" + "=" * 60)
    print("🧬 Consensus 解码 (按簇重新推理)")
    print("=" * 60)

    device_consensus = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model_consensus = Step1EvidentialModel(
        dim=step1_args.get('dim', args.dim),
        max_length=model_max_len,
        num_clusters=num_clusters,
        device=str(device_consensus)
    ).to(device_consensus)

    sd = checkpoint['model_state_dict']
    if 'length_adapter.weight' in sd:
        sh = sd['length_adapter.weight'].shape
        if sh[0] == model_max_len:
            model_consensus.length_adapter = nn.Linear(sh[1], sh[0]).to(device_consensus)
    model_consensus.load_state_dict(sd, strict=False)
    model_consensus.eval()

    consensus_dict = _consensus_with_reinference(
        model_consensus, dataset, new_labels, zone_ids, flat_real_indices,
        model_max_len, device_consensus
    )

    del model_consensus
    torch.cuda.empty_cache()

    fasta_path = os.path.join(args.output_dir, "consensus_sequences.fasta")
    save_consensus_sequences(consensus_dict, fasta_path)

    # =====================================================================
    # 5b. GT 评估 (可选)
    # =====================================================================
    if gt_tags_file and os.path.exists(gt_tags_file):
        _evaluate_with_gt(consensus_dict, data_loader, new_labels,
                          flat_real_indices, args.output_dir)

    # =====================================================================
    # 5c. 每轮评估指标 (ARI/NMI 等)
    # =====================================================================
    if gt_tags_file and os.path.exists(gt_tags_file):
        try:
            from models.eval_metrics import compute_all_metrics, save_metrics_report
            gt_arr = np.array(data_loader.gt_labels)

            # 构建全长预测标签
            round_pred = np.full(TOTAL_READS, -1, dtype=int)
            round_pred[flat_real_indices] = new_labels.cpu().numpy()

            metrics = compute_all_metrics(round_pred, gt_arr, verbose=True)
            report_path = os.path.join(args.output_dir, f"eval_round{args.round_idx}.txt")
            save_metrics_report(metrics, report_path,
                                round_info=f"Round {args.round_idx} (before post-processing)")
        except Exception as e:
            print(f"   ⚠️ 评估指标计算跳过: {e}")

    # =====================================================================
    # 6. 保存全长状态
    #    [v2] 额外保存质心 (供 post-processing 使用)
    # =====================================================================
    next_round_dir = os.path.join(args.experiment_dir, "04_Iterative_Labels")
    os.makedirs(next_round_dir, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")

    # 保存标签
    full_labels = np.full(TOTAL_READS, -1, dtype=int)
    full_labels[flat_real_indices] = new_labels.cpu().numpy()
    label_path = os.path.join(next_round_dir, f"refined_labels_{ts}.txt")
    np.savetxt(label_path, full_labels, fmt='%d')

    # 保存状态
    full_u_epi = np.zeros(TOTAL_READS, dtype=np.float32)
    full_u_ale = np.zeros(TOTAL_READS, dtype=np.float32)
    full_strength = np.zeros(TOTAL_READS, dtype=np.float32)
    full_zone_ids = np.zeros(TOTAL_READS, dtype=np.int64)

    full_u_epi[flat_real_indices] = u_epi.cpu().numpy()
    full_u_ale[flat_real_indices] = u_ale.cpu().numpy()
    full_strength[flat_real_indices] = strength.cpu().numpy()
    full_zone_ids[flat_real_indices] = zone_ids.cpu().numpy()

    state_path = os.path.join(next_round_dir, f"read_state_{ts}.pt")
    torch.save({
        'u_epi': full_u_epi, 'u_ale': full_u_ale,
        'strength': full_strength, 'zone_ids': full_zone_ids,
        'round_idx': args.round_idx
    }, state_path)

    # [v2] 保存质心
    centroids_path = os.path.join(next_round_dir, f"centroids_{ts}.pt")
    torch.save({
        'centroids': centroids,
        'cluster_sizes': cluster_sizes,
        'delta': delta,
        'round_idx': args.round_idx
    }, centroids_path)
    print(f"   💾 质心已保存: {centroids_path}")

    # =====================================================================
    # 7. 论文数据埋点
    # =====================================================================
    _record_paper_log(args, TOTAL_READS, refine_stats, consensus_dict,
                      new_labels, strength, delta)

    # 可视化
    try:
        viz = Step1Visualizer(args.output_dir)
        viz.plot_uncertainty_distribution(u_epi, u_ale, zone_ids)
    except Exception as e:
        print(f"   ⚠️ 可视化跳过: {e}")

    return {
        'next_round_files': {
            'labels': label_path,
            'state': state_path,
            'centroids': centroids_path,
            'reference': fasta_path
        }
    }


# ===========================================================================
# 按簇重新推理 evidence 做 consensus (省内存)
# ===========================================================================
def _consensus_with_reinference(model, dataset, new_labels, zone_ids,
                                flat_real_indices, max_len, device):
    from models.step1_data import seq_to_onehot

    cluster_to_didx = defaultdict(list)
    labels_np = new_labels.cpu().numpy()

    for didx in range(len(dataset)):
        real_idx = dataset.valid_indices[didx]
        label = int(labels_np[didx]) if didx < len(labels_np) else -1
        if label >= 0:
            cluster_to_didx[label].append(didx)

    zone_np = zone_ids.cpu().numpy()
    consensus_dict = {}
    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    processed = 0
    total_clusters = len(cluster_to_didx)

    for label, didx_list in cluster_to_didx.items():
        if len(didx_list) < 2:
            continue

        reads_list = []
        hc_flags = []
        for didx in didx_list:
            real_idx = dataset.valid_indices[didx]
            seq = dataset.data_loader.reads[real_idx]
            reads_list.append(seq_to_onehot(seq, max_len))
            hc_flags.append(zone_np[didx] == 1)

        reads_tensor = torch.stack(reads_list).to(device)
        hc_mask = torch.tensor(hc_flags, device=device)

        with torch.no_grad():
            emb, pooled = model.encode_reads(reads_tensor)
            evid, stre, alph = model.decode_to_evidence(emb)
        strength_seq = stre.mean(dim=1)

        high_conf_count = int(hc_mask.sum().item())
        count = len(didx_list)

        if high_conf_count == 0:
            continue

        if high_conf_count >= 2:
            fused_alpha = alph[hc_mask].mean(dim=0)
        else:
            conf_weights = torch.where(hc_mask, 2.0, 0.5)
            str_weights = F.softmax(strength_seq, dim=0)
            combined = conf_weights * str_weights
            combined = (combined / combined.sum()).view(-1, 1, 1)
            fused_alpha = torch.sum(alph * combined, dim=0)

        consensus_prob = fused_alpha / fused_alpha.sum(dim=-1, keepdim=True)
        consensus_indices = torch.argmax(consensus_prob, dim=-1)
        consensus_seq = ''.join([base_map[idx.item()] for idx in consensus_indices])

        consensus_dict[int(label)] = {
            'consensus_prob': consensus_prob.cpu(),
            'consensus_seq': consensus_seq,
            'num_reads': count,
            'num_high_conf': high_conf_count,
            'avg_strength': strength_seq.mean().item()
        }

        processed += 1
        if processed % 5000 == 0:
            print(f"      Consensus: {processed}/{total_clusters}", flush=True)

    print(f"\n   🧬 共识序列: {len(consensus_dict)} 个")
    return consensus_dict


# ===========================================================================
# GT 评估
# ===========================================================================
def _evaluate_with_gt(consensus_dict, data_loader, new_labels,
                      flat_real_indices, output_dir):
    print("\n" + "=" * 60)
    print("📋 GT 评估")
    print("=" * 60)

    labels_np = new_labels.cpu().numpy()
    gt_labels = data_loader.gt_labels

    cluster_gt_counts = defaultdict(Counter)
    total_assigned = 0

    for i, didx_real in enumerate(flat_real_indices):
        label = int(labels_np[i])
        if label < 0:
            continue
        gt = gt_labels[didx_real]
        if gt >= 0:
            cluster_gt_counts[label][gt] += 1
            total_assigned += 1

    if total_assigned == 0:
        print("   ⚠️ 无 GT 匹配")
        return

    # Purity
    total_correct = 0
    for cid, gt_counter in cluster_gt_counts.items():
        total_correct += gt_counter.most_common(1)[0][1]
    purity = total_correct / max(total_assigned, 1)

    # Recovery
    all_gt = set()
    recovered_gt = set()
    for cid, gt_counter in cluster_gt_counts.items():
        for gt_id in gt_counter:
            all_gt.add(gt_id)
        recovered_gt.add(gt_counter.most_common(1)[0][0])
    recovery = len(recovered_gt) / max(len(all_gt), 1)

    print(f"   Purity:    {purity:.4f}  ({purity*100:.2f}%)")
    print(f"   Recovery:  {recovery:.4f}  ({recovery*100:.2f}%)")
    print(f"   Assigned:  {total_assigned:,}")
    print(f"   GT Clusters: {len(all_gt):,}")
    print(f"   Pred Clusters with GT: {len(cluster_gt_counts):,}")

    # 保存
    try:
        report_path = os.path.join(output_dir, "gt_evaluation.txt")
        with open(report_path, 'w') as f:
            f.write(f"Purity: {purity:.6f}\n")
            f.write(f"Recovery: {recovery:.6f}\n")
            f.write(f"Assigned: {total_assigned}\n")
            f.write(f"GT Clusters: {len(all_gt)}\n")
            f.write(f"Pred Clusters: {len(cluster_gt_counts)}\n")
        print(f"   💾 GT 评估: {report_path}")
    except Exception as e:
        print(f"   ⚠️ 保存 GT 评估失败: {e}")


# ===========================================================================
# 论文数据埋点 [v2] 移除 resurrection 相关字段
# ===========================================================================
def _record_paper_log(args, total_reads, refine_stats, consensus_dict,
                      new_labels, strength, delta):
    try:
        log_file = os.path.join(args.output_dir, "paper_log.csv")
        with open(log_file, 'w') as f:
            f.write("Round,Total_Reads,Zone1_Safe,Zone2_Reassigned,Zone3_Dirty,"
                    "Final_Clusters,Avg_Strength,Avg_HC_Ratio,Delta\n")

            z1 = refine_stats.get('zone1_kept', 0)
            z2_fix = refine_stats.get('zone2_reassigned', 0)
            z3 = refine_stats.get('zone3_dirty', 0)
            final_clusters = len(consensus_dict)
            avg_s = strength.mean().item()

            hc_ratios = [c['num_high_conf'] / c['num_reads']
                         for c in consensus_dict.values() if c['num_reads'] > 0]
            avg_hc = sum(hc_ratios) / len(hc_ratios) if hc_ratios else 0.0

            f.write(f"{args.round_idx},{total_reads},{z1},{z2_fix},{z3},"
                    f"{final_clusters},{avg_s:.4f},{avg_hc:.4f},{delta:.4f}\n")

        print(f"   ✅ 数据: {log_file}")
    except Exception as e:
        print(f"   ⚠️ 写日志失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step2: Refinement & Decoding')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--step1_checkpoint', type=str, required=True)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--refined_labels', type=str, default=None)
    parser.add_argument('--prev_state', type=str, default=None)
    parser.add_argument('--round_idx', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./step2_results')
    parser.add_argument('--gt_tags_file', type=str, default=None)
    parser.add_argument('--gt_refs_file', type=str, default=None)
    parser.add_argument('--training_cap', type=int, default=2000000)

    args = parser.parse_args()
    try:
        run_step2(args)
    except Exception as e:
        print(f"❌ Step2 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)