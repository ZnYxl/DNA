# models/step2_runner.py
"""
Step2 主入口：Evidence-Guided Refinement & Decoding (顶刊增强版)
功能：
1. 噪声复活 (Resurrection): 通过哨兵标签技术召回上一轮误删的 reads
2. 状态传递: 保存 strength, u_epi, u_ale 供下一轮动量更新和采样
3. 聚类纠错: 基于证据的距离判决，解决 Clover 过聚类问题
4. 顶刊数据监控: 实时计算簇合并数量与 Micro Accuracy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
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
RESURRECTION_SENTINEL = 999999 # 哨兵值：让 Step1Dataset 放行，但 Step1 训练应忽略

@torch.no_grad()
def run_step2(args):
    """
    Step 2 执行流程：
    1. 加载模型与数据
    2. 注入哨兵标签，激活噪声复活机制
    3. 批量推理获取 Evidence (Alpha)
    4. 动量更新 Evidence Strength
    5. 三区制划分 (Zone Splitting)
    6. 计算加权质心与自适应 Delta
    7. Zone-aware 修正 (核心纠错)
    8. 顶刊指标统计 (簇数量、准确率)
    9. Consensus 解码与状态保存
    """
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
        # 优先使用 checkpoint 中的配置，确保维度匹配
        model_dim = step1_args.get('dim', args.dim)
        model_max_len = step1_args.get('max_length', args.max_length)
        print(f"   ✅ 模型参数已加载: Dim={model_dim}, MaxLen={model_max_len}")
    except Exception as e:
        print(f"   ❌ Checkpoint 加载失败: {e}")
        return None

    # 加载数据 (传入上一轮的 refined_labels 以继承状态)
    try:
        labels_path = getattr(args, 'refined_labels', None)
        data_loader = CloverDataLoader(args.experiment_dir, labels_path=labels_path)
        TOTAL_READS = len(data_loader.reads)
        
        # 确定簇数量 (用于初始化分类头，虽然 Step2 不用分类头，但模型结构需要对齐)
        current_clusters = set(data_loader.clover_labels)
        if -1 in current_clusters: current_clusters.remove(-1)
        if RESURRECTION_SENTINEL in current_clusters: current_clusters.remove(RESURRECTION_SENTINEL)
        num_clusters = max(50, len(current_clusters))
        
        print(f"   📊 数据统计: {TOTAL_READS} Reads, {len(current_clusters)} 初始有效簇")
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
    
    # 预加载 length_adapter (处理维度不匹配问题)
    sd = checkpoint['model_state_dict']
    if 'length_adapter.weight' in sd:
        sh = sd['length_adapter.weight'].shape
        model.length_adapter = nn.Linear(sh[1], sh[0]).to(device)
    
    model.load_state_dict(sd, strict=False)
    model.eval()

    # =====================================================================
    # 2. 噪声复活预处理 (Sentinel Injection)
    # =====================================================================
    print("\n" + "=" * 60)
    print("🔄 噪声复活检测")
    print("=" * 60)

    original_labels = list(data_loader.clover_labels) # 备份
    labels_np = np.array(original_labels)
    resurrection_mask = (labels_np == -1)
    n_resurrect = resurrection_mask.sum()

    if n_resurrect > 0:
        print(f"   🔙 发现 {n_resurrect} 条噪声 Reads，尝试复活...")
        # 将 -1 修改为哨兵值，使其通过 Step1Dataset 的 label >= 0 过滤
        indices_to_resurrect = np.where(resurrection_mask)[0]
        for idx in indices_to_resurrect:
            data_loader.clover_labels[idx] = RESURRECTION_SENTINEL
    else:
        print("   ✅ 无噪声 Reads 需要复活")

    # =====================================================================
    # 3. 批量推理 (Inference)
    # =====================================================================
    print("\n" + "=" * 60)
    print("🔮 全量推理 (提取 Evidence)")
    print("=" * 60)

    dataset = Step1Dataset(data_loader, max_len=model_max_len)
    inference_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True
    )

    # 存储容器
    all_embeddings, all_strength, all_evidence, all_alpha = [], [], [], []
    all_u_epi, all_u_ale = [], []
    all_labels, all_indices = [], []

    for batch_idx, batch in enumerate(inference_loader):
        reads = batch['encoding'].to(device)
        lbls = batch['clover_label']
        idxs = batch['read_idx']

        # 强制 Padding 对齐
        if reads.shape[1] != model_max_len:
            if reads.shape[1] < model_max_len:
                reads = F.pad(reads, (0, 0, 0, model_max_len - reads.shape[1]))
            else:
                reads = reads[:, :model_max_len, :]

        # Forward
        emb, pooled = model.encode_reads(reads)
        evid, stre, alph = model.decode_to_evidence(emb)
        epi, ale = decompose_uncertainty(alph) #

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
            
        if (batch_idx + 1) % 50 == 0:
            print(f"      处理进度: {batch_idx + 1}/{len(inference_loader)}", end='\r')

    # 拼接全量数据
    embeddings = torch.cat(all_embeddings).to(device)
    strength = torch.cat(all_strength).to(device)
    alpha = torch.cat(all_alpha).to(device)
    evidence = torch.cat(all_evidence).to(device)
    u_epi = torch.cat(all_u_epi).to(device)
    u_ale = torch.cat(all_u_ale).to(device)
    labels = torch.tensor(all_labels, device=device)
    flat_real_indices = torch.cat(all_indices).numpy()

    # 恢复数据加载器状态 (移除哨兵)
    data_loader.clover_labels = original_labels

    # 识别哨兵位置
    sentinel_tensor_mask = (labels == RESURRECTION_SENTINEL)
    labels[sentinel_tensor_mask] = -1 # 恢复为 -1 以进行正常的质心计算过滤

    print(f"\n   ✅ 推理完成，处理样本数: {len(labels)}")

    # =====================================================================
    # 4. 动量更新 (Momentum Update)
    # =====================================================================
    print("\n" + "=" * 60)
    print("📊 动量更新 Strength")
    print("=" * 60)

    if getattr(args, 'prev_state', None) and os.path.exists(args.prev_state):
        try:
            prev_state = torch.load(args.prev_state, map_location='cpu')
            prev_str_full = prev_state['strength']
            # 映射回当前推理的子集
            prev_str_sub = torch.tensor(prev_str_full[flat_real_indices], device=device)
            
            strength = MOMENTUM_CURR * strength + MOMENTUM_PREV * prev_str_sub
            print(f"   ✅ 动量融合完成 (Curr: {MOMENTUM_CURR}, Prev: {MOMENTUM_PREV})")
        except Exception as e:
            print(f"   ⚠️ 动量更新失败，使用当前 Strength: {e}")
    else:
        print("   ℹ️ 无上一轮状态，跳过动量更新")

    # =====================================================================
    # 5. 三区制划分 & 质心计算
    # =====================================================================
    print("\n" + "=" * 60)
    print("🔍 三区制划分与质心计算")
    print("=" * 60)

    # 为 Zone 划分准备标签 (哨兵设为 0 以参与计算，但不影响质心)
    labels_for_zone = labels.clone()
    labels_for_zone[sentinel_tensor_mask] = 0 
    
    zone_ids, zone_stats = split_confidence_by_zone(u_epi, u_ale, labels_for_zone)
    
    # [核心] 复活 Reads 强制进入 Zone II (Hard)，必须经过距离判决才能生存
    zone_ids[sentinel_tensor_mask] = 2
    
    # 计算质心 (仅使用非负标签)
    centroids, cluster_sizes = compute_centroids_weighted(embeddings, labels, strength, zone_ids)
    
    # 计算自适应 Delta
    delta = compute_global_delta(embeddings, labels, zone_ids, centroids)

    # =====================================================================
    # 6. Zone-aware 修正 (核心纠错)
    # =====================================================================
    print("\n" + "=" * 60)
    print("🔄 Zone-aware 聚类修正")
    print("=" * 60)

    new_labels, noise_mask, refine_stats = refine_reads(
        embeddings, labels, zone_ids, centroids, delta, round_idx=args.round_idx
    )

    # =====================================================================
    # 7. 顶刊数据监控 (核心：过聚类纠错与 Micro Accuracy)
    # =====================================================================
    print("\n" + "📊" * 20)
    print("📈 SSI-EC 顶刊数据监控 (ERR036 验证)")
    
    # A. 簇数量演变
    initial_valid_mask = (labels >= 0)
    final_valid_mask = (new_labels >= 0)
    
    initial_clusters_cnt = len(torch.unique(labels[initial_valid_mask]))
    final_clusters_cnt = len(torch.unique(new_labels[final_valid_mask]))
    
    print(f"   🔹 簇数量演变: {initial_clusters_cnt} -> {final_clusters_cnt}")
    print(f"      (Clover初始过聚类 -> SSI-EC合并修正，目标值: ~72000)")

    # B. 复活统计
    resurrected_cnt = (sentinel_tensor_mask & (new_labels >= 0)).sum().item()
    print(f"   🔹 噪声复活数: {resurrected_cnt} / {sentinel_tensor_mask.sum().item()}")

    # C. Micro Accuracy
    if hasattr(data_loader, 'gt_labels') and len(data_loader.gt_labels) > 0:
        # 映射 GT 到当前推理样本
        gt_full = np.array(data_loader.gt_labels)
        # 确保索引不过界
        valid_map_mask = flat_real_indices < len(gt_full)
        
        if valid_map_mask.all():
            gt_subset = gt_full[flat_real_indices]
            pred_subset = new_labels.cpu().numpy()
            
            # 只评估非噪声预测的准确性
            eval_mask = (pred_subset >= 0)
            if eval_mask.any():
                correct = (gt_subset[eval_mask] == pred_subset[eval_mask]).sum()
                total_eval = eval_mask.sum()
                acc = correct / total_eval
                print(f"   🔹 修正后 Micro Accuracy: {acc:.4%}")
                print(f"      (基于 {total_eval} 条有效预测)")
            else:
                print("   ⚠️ 无有效预测，无法计算 Accuracy")
        else:
            print("   ⚠️ 索引越界，无法对齐 GT Labels")
    
    print("📊" * 20)

    # 调用 Visualizer 画不确定性分布图
    viz = Step1Visualizer(args.output_dir)
    viz.plot_uncertainty_distribution(u_epi, u_ale, zone_ids)

    # =====================================================================
    # 8. 解码与保存
    # =====================================================================
    print("\n" + "=" * 60)
    print("💾 解码与状态保存")
    print("=" * 60)

    # Consensus 解码
    high_conf_mask = (zone_ids == 1)
    consensus_dict = decode_cluster_consensus(
        all_evidence, # 使用 list 以避免不必要的 cat 开销，如果 decode 支持 list
        alpha,        # 这里 alpha 已经是 cat 过的 tensor
        new_labels, 
        strength, 
        high_conf_mask
    )
    
    # 修正：decode_cluster_consensus 内部需要 tensor 类型的 evidence
    # 上面已经 cat 成了 evidence 变量，直接传 evidence
    consensus_dict = decode_cluster_consensus(
        evidence, 
        alpha, 
        new_labels, 
        strength, 
        high_conf_mask
    )

    save_consensus_sequences(consensus_dict, os.path.join(args.output_dir, "consensus_sequences.fasta"))

    # 保存全长状态 (用于下一轮)
    next_round_dir = os.path.join(args.experiment_dir, "04_Iterative_Labels")
    os.makedirs(next_round_dir, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")

    # 保存 Labels
    full_labels = np.full(TOTAL_READS, -1, dtype=int)
    full_labels[flat_real_indices] = new_labels.cpu().numpy()
    label_path = os.path.join(next_round_dir, f"refined_labels_{ts}.txt")
    np.savetxt(label_path, full_labels, fmt='%d')

    # 保存 State
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

    print(f"   ✅ 状态已保存: {state_path}")

    return {
        'next_round_files': {
            'labels': label_path,
            'state': state_path,
            'reference': os.path.join(args.output_dir, "consensus_sequences.fasta")
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step2: Refinement & Decoding')
    parser.add_argument('--experiment_dir',   type=str, required=True)
    parser.add_argument('--step1_checkpoint', type=str, required=True)
    parser.add_argument('--dim',              type=int, default=256)
    parser.add_argument('--max_length',       type=int, default=150)
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