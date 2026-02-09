# models/step2_runner.py
"""
Step2 主入口：Evidence-Guided Refinement & Decoding (Goldman Edition)

功能升级:
  1. 噪声复活 (Resurrection): 哨兵标签机制，给“误杀”的 reads 第二次机会
  2. 论文埋点 (Paper Recorder): 自动记录每一轮的关键指标到 CSV
  3. 状态传递: 生成 read_state.pt 供下一轮动态采样
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
import numpy as np
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.step1_model  import Step1EvidentialModel, decompose_uncertainty
from models.step1_data   import CloverDataLoader, Step1Dataset
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

# ---------------------------------------------------------------------------
# 全局超参
# ---------------------------------------------------------------------------
MOMENTUM_CURR = 0.7          # 当前轮 strength 权重
MOMENTUM_PREV = 0.3          # 上一轮 strength 权重
RESURRECTION_SENTINEL = 999999 # 哨兵值

@torch.no_grad()
def run_step2(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Step2 启动 | 设备: {device} | 轮次: {args.round_idx}")

    # =====================================================================
    # 1. 加载 Step1 模型与数据
    # =====================================================================
    print("\n" + "=" * 60)
    print("📦 加载模型与数据")
    print("=" * 60)

    # 加载 Checkpoint
    try:
        checkpoint = torch.load(args.step1_checkpoint, map_location=device)
        step1_args = checkpoint.get('args', {})
        model_dim = step1_args.get('dim', args.dim)
        model_max_len = step1_args.get('max_length', args.max_length)
        print(f"   ✅ 模型参数: Dim={model_dim}, MaxLen={model_max_len}")
    except Exception as e:
        print(f"   ❌ Checkpoint 加载失败: {e}")
        return None

    # 加载数据
    try:
        labels_path = getattr(args, 'refined_labels', None)
        data_loader = CloverDataLoader(args.experiment_dir, labels_path=labels_path)
        TOTAL_READS = len(data_loader.reads)
        
        # 确定簇数量
        current_clusters = set(data_loader.clover_labels)
        if -1 in current_clusters: current_clusters.remove(-1)
        num_clusters = max(50, len(current_clusters))
        
        print(f"   📊 数据统计: {TOTAL_READS} Reads, {len(current_clusters)} 有效簇")
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return None

    # 初始化模型
    model = Step1EvidentialModel(
        dim=model_dim, 
        max_length=model_max_len, 
        num_clusters=num_clusters, 
        device=device
    ).to(device)
    
    # 预加载 length_adapter
    sd = checkpoint['model_state_dict']
    if 'length_adapter.weight' in sd:
        sh = sd['length_adapter.weight'].shape
        if sh[0] == model_max_len:
            model.length_adapter = nn.Linear(sh[1], sh[0]).to(device)
        else:
            print(f"   ⚠️ Adapter 维度不匹配 ({sh[0]} vs {model_max_len})，跳过")
    
    model.load_state_dict(sd, strict=False)
    model.eval()

    # =====================================================================
    # 2. 噪声复活预处理
    # =====================================================================
    print("\n" + "=" * 60)
    print("🔄 噪声复活检测")
    print("=" * 60)

    original_labels = list(data_loader.clover_labels) # 备份
    labels_np = np.array(original_labels)
    resurrection_mask = (labels_np == -1)
    n_resurrect = resurrection_mask.sum()

    if n_resurrect > 0:
        print(f"   🔙 尝试复活 {n_resurrect} 条噪声 Reads...")
        indices_to_resurrect = np.where(resurrection_mask)[0]
        for idx in indices_to_resurrect:
            data_loader.clover_labels[idx] = RESURRECTION_SENTINEL
    else:
        print("   ✅ 无噪声 Reads 需要复活")

    # =====================================================================
    # 3. 批量推理
    # =====================================================================
    print("\n" + "=" * 60)
    print("🔮 全量推理 (提取 Evidence)")
    print("=" * 60)

    dataset = Step1Dataset(data_loader, max_len=model_max_len)
    inference_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True
    )

    all_embeddings, all_strength, all_evidence, all_alpha = [], [], [], []
    all_u_epi, all_u_ale = [], []
    all_labels, all_indices = [], []

    for batch_idx, batch in enumerate(inference_loader):
        reads = batch['encoding'].to(device)
        lbls = batch['clover_label']
        idxs = batch['read_idx']

        # 强制 Padding
        if reads.shape[1] != model_max_len:
            if reads.shape[1] < model_max_len:
                reads = F.pad(reads, (0, 0, 0, model_max_len - reads.shape[1]))
            else:
                reads = reads[:, :model_max_len, :]

        # Forward
        emb, pooled = model.encode_reads(reads)
        evid, stre, alph = model.decode_to_evidence(emb)
        epi, ale = decompose_uncertainty(alph)

        all_embeddings.append(pooled.cpu())
        all_strength.append(stre.mean(dim=1).cpu())
        all_alpha.append(alph.cpu())
        all_evidence.append(evid.cpu())
        all_u_epi.append(epi.cpu())
        all_u_ale.append(ale.cpu())
        all_indices.append(idxs)
        
        if isinstance(lbls, torch.Tensor):
            all_labels.extend(lbls.tolist())
        else:
            all_labels.extend(lbls)
            
        if (batch_idx + 1) % 100 == 0:
            print(f"      进度: {batch_idx + 1}/{len(inference_loader)}", end='\r')

    # 拼接
    embeddings = torch.cat(all_embeddings).to(device)
    strength = torch.cat(all_strength).to(device)
    alpha = torch.cat(all_alpha).to(device)
    evidence = torch.cat(all_evidence).to(device)
    u_epi = torch.cat(all_u_epi).to(device)
    u_ale = torch.cat(all_u_ale).to(device)
    labels = torch.tensor(all_labels, device=device)
    flat_real_indices = torch.cat(all_indices).numpy()

    # 恢复 Loader 状态
    data_loader.clover_labels = original_labels

    # 识别哨兵
    sentinel_tensor_mask = (labels == RESURRECTION_SENTINEL)
    labels[sentinel_tensor_mask] = -1 

    print(f"\n   ✅ 推理完成，样本数: {len(labels)}")

    # =====================================================================
    # 4. 动量更新
    # =====================================================================
    if getattr(args, 'prev_state', None) and os.path.exists(args.prev_state):
        try:
            print(f"   📊 执行动量更新...")
            prev_state = torch.load(args.prev_state, map_location='cpu')
            prev_str_full = prev_state['strength']
            prev_str_sub = torch.tensor(prev_str_full[flat_real_indices], device=device)
            strength = MOMENTUM_CURR * strength + MOMENTUM_PREV * prev_str_sub
        except Exception as e:
            print(f"   ⚠️ 动量更新跳过: {e}")

    # =====================================================================
    # 5. 三区制划分 & 质心
    # =====================================================================
    labels_for_zone = labels.clone()
    labels_for_zone[sentinel_tensor_mask] = 0 
    
    zone_ids, zone_stats = split_confidence_by_zone(u_epi, u_ale, labels_for_zone)
    zone_ids[sentinel_tensor_mask] = 2 # 复活强制进 Zone II
    
    centroids, _ = compute_centroids_weighted(embeddings, labels, strength, zone_ids)
    delta = compute_global_delta(embeddings, labels, zone_ids, centroids)

    # =====================================================================
    # 6. Zone-aware 修正
    # =====================================================================
    new_labels, noise_mask, refine_stats = refine_reads(
        embeddings, labels, zone_ids, centroids, delta, round_idx=args.round_idx
    )

    # =====================================================================
    # 7. Consensus 解码
    # =====================================================================
    print("\n" + "=" * 60)
    print("🧬 Consensus 解码")
    print("=" * 60)
    
    consensus_dict = decode_cluster_consensus(
        evidence, alpha, new_labels, strength, (zone_ids == 1)
    )
    save_consensus_sequences(consensus_dict, os.path.join(args.output_dir, "consensus_sequences.fasta"))

    # =====================================================================
    # 8. 保存全长状态
    # =====================================================================
    next_round_dir = os.path.join(args.experiment_dir, "04_Iterative_Labels")
    os.makedirs(next_round_dir, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")

    # Labels
    full_labels = np.full(TOTAL_READS, -1, dtype=int)
    full_labels[flat_real_indices] = new_labels.cpu().numpy()
    label_path = os.path.join(next_round_dir, f"refined_labels_{ts}.txt")
    np.savetxt(label_path, full_labels, fmt='%d')

    # State
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
        'u_epi': full_u_epi,
        'u_ale': full_u_ale,
        'strength': full_strength,
        'zone_ids': full_zone_ids,
        'round_idx': args.round_idx
    }, state_path)

    # =====================================================================
    # 9. 论文数据埋点 (Paper Recorder) - [NEW]
    # =====================================================================
    print("\n" + "📊" * 30)
    print("📝 正在记录论文实验数据...")
    
    log_file = os.path.join(args.experiment_dir, "goldman_experiment_log.csv")
    file_exists = os.path.exists(log_file)
    
    try:
        with open(log_file, 'a') as f:
            if not file_exists:
                # 写入表头
                f.write("Round,Total_Reads,Zone1_Safe,Zone2_Reassigned,Zone3_Dirty,Resurrected_Count,Final_Clusters,Avg_Strength,Avg_HighConf_Ratio,Delta\n")
            
            # 提取数据
            z1 = refine_stats.get('zone1_kept', 0)
            z2_fix = refine_stats.get('zone2_reassigned', 0)
            z3 = refine_stats.get('zone3_dirty', 0)
            resurrect_cnt = (sentinel_tensor_mask & (new_labels >= 0)).sum().item()
            final_clusters = len(consensus_dict)
            avg_s = strength.mean().item()
            
            # 计算 Avg High Conf Ratio
            hc_ratios = []
            for c in consensus_dict.values():
                if c['num_reads'] > 0:
                    hc_ratios.append(c['num_high_conf'] / c['num_reads'])
            avg_hc = sum(hc_ratios) / len(hc_ratios) if hc_ratios else 0.0
            
            # 写入一行
            line = f"{args.round_idx},{TOTAL_READS},{z1},{z2_fix},{z3},{resurrect_cnt},{final_clusters},{avg_s:.4f},{avg_hc:.4f},{delta:.4f}\n"
            f.write(line)
            
        print(f"   ✅ [Paper] 数据已追加至: {log_file}")
        print(f"      Round: {args.round_idx} | Clusters: {final_clusters} | Resurrected: {resurrect_cnt}")
    except Exception as e:
        print(f"   ⚠️ 写入日志失败: {e}")
    
    print("📊" * 30)

    # 可视化
    try:
        viz = Step1Visualizer(args.output_dir)
        viz.plot_uncertainty_distribution(u_epi, u_ale, zone_ids)
    except: pass

    return {
        'next_round_files': {
            'labels': label_path,
            'state': state_path,
            'reference': os.path.join(args.output_dir, "consensus_sequences.fasta")
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step2: Refinement & Decoding (Goldman)')
    parser.add_argument('--experiment_dir',   type=str, required=True)
    parser.add_argument('--step1_checkpoint', type=str, required=True)
    parser.add_argument('--dim',              type=int, default=256)
    parser.add_argument('--max_length',       type=int, default=117)
    parser.add_argument('--device',           type=str, default='cuda')
    parser.add_argument('--refined_labels',   type=str, default=None)
    parser.add_argument('--prev_state',       type=str, default=None)
    parser.add_argument('--round_idx',        type=int, default=1)
    parser.add_argument('--output_dir',       type=str, default=f'./step2_results')

    args = parser.parse_args()
    try:
        run_step2(args)
    except Exception as e:
        print(f"❌ Step2 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)