# models/step2_runner.py
"""
Step2 主入口：Evidence-Guided Refinement & Decoding
关键：不训练，只推理+决策
✅ 相对不确定性原则：簇内比较，偏向确定性
✅ 修复版 v3 (最终版)：
   1. 🌟 数据对齐修复：输出标签数量严格等于原始 Reads 数量 (解决 Mismatch 报错)
   2. 包含 length_adapter 权重预加载修复
   3. 包含 强制 Padding 逻辑
   4. 包含 DataLoader 多进程批量推理加速
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
import numpy as np
from datetime import datetime

# ✅ 添加路径处理
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.step1_model import Step1EvidentialModel
from models.step1_data import CloverDataLoader, Step1Dataset
from models.step2_refine import (
    split_confidence_by_percentile,
    compute_cluster_centroids,
    refine_low_confidence_reads,
    compute_adaptive_delta
)
from models.step2_decode import (
    decode_cluster_consensus,
    save_consensus_sequences
)


@torch.no_grad()
def run_step2(args):
    """
    Step2主流程：推理 -> 修正 -> 解码 -> 对齐保存
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")

    # ========== 1️⃣ 加载Step1模型 ==========
    print("\n" + "=" * 60)
    print("📦 加载Step1训练好的模型")
    print("=" * 60)

    try:
        checkpoint = torch.load(args.step1_checkpoint, map_location=device)
        print(f"   ✅ checkpoint加载成功")
    except Exception as e:
        print(f"   ❌ checkpoint加载失败: {e}")
        return None

    # ✅ 获取模型参数
    if 'args' in checkpoint:
        step1_args = checkpoint['args']
        model_dim = step1_args.get('dim', args.dim)
        model_max_length = step1_args.get('max_length', args.max_length)
    else:
        model_dim = args.dim
        model_max_length = args.max_length

    # ✅ 重建数据加载器 & 获取总数 (关键修复)
    try:
        data_loader = CloverDataLoader(args.experiment_dir)
        TOTAL_READS_COUNT = len(data_loader.reads)  # 🌟 必须获取原始总数
        num_clusters = len(set(data_loader.clover_labels))
        print(f"   📊 数据统计: {TOTAL_READS_COUNT} 总Reads, {num_clusters} 簇")
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return None

    # ✅ 重建模型
    try:
        model = Step1EvidentialModel(
            dim=model_dim,
            max_length=model_max_length,
            num_clusters=num_clusters,
            device=device
        ).to(device)
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        return None

    # ✅ 修复: 权重预加载 (length_adapter)
    try:
        state_dict = checkpoint['model_state_dict']
        if 'length_adapter.weight' in state_dict:
            weight_shape = state_dict['length_adapter.weight'].shape
            model.length_adapter = nn.Linear(weight_shape[1], weight_shape[0]).to(device)
            print(f"   🔧 预初始化 length_adapter: {weight_shape}")
    except Exception:
        pass

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # ========== 2️⃣ 批量推理 (带索引记录) ==========
    print("\n" + "=" * 60)
    print("🔮 Step1模型批量推理")
    print("=" * 60)

    try:
        # 注意：Step1Dataset 可能会过滤掉 -1 的数据，所以 len(dataset) <= TOTAL_READS_COUNT
        dataset = Step1Dataset(data_loader, max_len=model_max_length)
        
        # 使用 DataLoader 加速
        inference_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True
        )
        print(f"   📊 有效推理数据: {len(dataset)} (Batch Size: 1024)")
    except Exception as e:
        print(f"   ❌ 数据集创建失败: {e}")
        return None

    all_embeddings = []
    all_strength = []
    all_alpha = []
    all_evidence = []
    all_labels = []
    all_real_indices = []  # 🌟 记录真实索引

    print(f"   🔄 开始推理...")

    for batch_idx, batch_data in enumerate(inference_loader):
        reads = batch_data['encoding'].to(device) # (B, L, 4)
        labels = batch_data['clover_label']
        read_indices = batch_data['read_idx']     # (B,) 获取原始索引

        # 🌟 强制 Padding 逻辑
        curr_len = reads.shape[1]
        target_len = model_max_length
        if curr_len > target_len:
            reads = reads[:, :target_len, :]
        elif curr_len < target_len:
            pad_len = target_len - curr_len
            reads = F.pad(reads, (0, 0, 0, pad_len), "constant", 0)

        # 推理
        with torch.no_grad():
            embeddings, pooled_emb = model.encode_reads(reads)
            evidence, strength, alpha = model.decode_to_evidence(embeddings)

        # 收集结果
        all_embeddings.append(pooled_emb.cpu())
        all_strength.append(strength.mean(dim=1).cpu())
        all_alpha.append(alpha.cpu())
        all_evidence.append(evidence.cpu())
        all_real_indices.append(read_indices) # 记录索引

        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.tolist())
        else:
            all_labels.extend(labels)

        if (batch_idx + 1) % 50 == 0:
            print(f"      已处理 Batch: {batch_idx + 1}/{len(inference_loader)}", end='\r')

    if len(all_embeddings) == 0:
        print(f"\n   ❌ 没有成功推理的reads！")
        return None

    # 拼接张量
    embeddings = torch.cat(all_embeddings, dim=0).to(device)
    strength = torch.cat(all_strength, dim=0).to(device)
    alpha = torch.cat(all_alpha, dim=0).to(device)
    evidence = torch.cat(all_evidence, dim=0).to(device)
    labels = torch.tensor(all_labels, device=device)
    
    # 🌟 拼接所有索引
    flat_real_indices = torch.cat(all_real_indices).numpy()

    print(f"\n   ✅ 推理完成. 张量形状: {embeddings.shape}")

    # ========== 3️⃣ Phase A: 相对证据筛选 ==========
    print("\n" + "=" * 60)
    print("🔍 Phase A: 相对证据筛选")
    print("=" * 60)

    low_conf_mask, conf_stats = split_confidence_by_percentile(
        strength, labels, p=args.uncertainty_percentile
    )
    high_conf_mask = ~low_conf_mask

    # ========== 4️⃣ Phase B: 簇修正 ==========
    print("\n" + "=" * 60)
    print("🔄 Phase B: 簇修正")
    print("=" * 60)

    centroids, cluster_sizes = compute_cluster_centroids(
        embeddings, labels, high_conf_mask
    )

    if args.delta is None:
        delta = compute_adaptive_delta(
            embeddings, centroids, percentile=args.delta_percentile
        )
    else:
        delta = args.delta
    
    # new_labels 的长度 = len(dataset) (即有效reads的数量)
    new_labels, noise_mask, refine_stats = refine_low_confidence_reads(
        embeddings, labels, low_conf_mask, centroids, delta
    )

    # ========== 5️⃣ Phase C: Consensus ==========
    print("\n" + "=" * 60)
    print("🧬 Phase C: Consensus解码")
    print("=" * 60)

    consensus_dict = decode_cluster_consensus(
        evidence, alpha, new_labels, strength, high_conf_mask
    )

    os.makedirs(args.output_dir, exist_ok=True)
    consensus_path = os.path.join(args.output_dir, "consensus_sequences.fasta")
    save_consensus_sequences(consensus_dict, consensus_path)

    # ========== 6️⃣ 准备下一轮迭代的数据 (🌟核心修复) ==========
    print("\n" + "=" * 60)
    print("🔄 准备下一轮 (Next Round) 数据 - 对齐修复")
    print("=" * 60)

    next_round_dir = os.path.join(args.experiment_dir, "04_Iterative_Labels")
    os.makedirs(next_round_dir, exist_ok=True)
    timestamp_id = datetime.now().strftime("%H%M%S")
    label_save_path = os.path.join(next_round_dir, f"refined_labels_{timestamp_id}.txt")

    # 🌟 关键逻辑：还原到全长数组
    # 1. 创建全长数组，默认填 -1 (噪声)
    full_refined_labels = np.full(TOTAL_READS_COUNT, -1, dtype=int)
    
    # 2. 将修正后的 new_labels 填入对应的原始位置
    # new_labels 是 Tensor (N_valid,), flat_real_indices 是 numpy (N_valid,)
    current_refined_labels = new_labels.cpu().numpy()
    
    # 安全检查
    if len(flat_real_indices) != len(current_refined_labels):
        print(f"   ❌ 严重错误: 索引数量与标签数量不一致!")
        return None
        
    full_refined_labels[flat_real_indices] = current_refined_labels
    
    # 3. 保存全长文件
    np.savetxt(label_save_path, full_refined_labels, fmt='%d')
    
    print(f"   📝 修正标签已保存: {label_save_path}")
    print(f"      - 原始Reads总数: {TOTAL_READS_COUNT}")
    print(f"      - 保存标签总数: {len(full_refined_labels)} (必须一致)")
    print(f"      - 有效修正数: {len(current_refined_labels)}")
    print(f"      - 自动标记噪声(-1): {TOTAL_READS_COUNT - len(current_refined_labels)}")

    # 保存结果 dict
    try:
        results = {
            'new_labels': new_labels.cpu(),
            'noise_mask': noise_mask.cpu(),
            'strength': strength.cpu(),
            'consensus_dict': consensus_dict,
            'next_round_files': {
                'labels': label_save_path,
                'reference': consensus_path
            },
            'args': vars(args)
        }
        results_path = os.path.join(args.output_dir, "step2_results.pth")
        torch.save(results, results_path)
        print(f"\n   💾 完整结果已保存: {results_path}")
    except Exception as e:
        print(f"   ⚠️ 结果保存失败: {e}")
        results = None

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step2: Evidence-Guided Refinement & Decoding')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--step1_checkpoint', type=str, required=True)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--uncertainty_percentile', type=float, default=0.2)
    parser.add_argument('--delta', type=float, default=None)
    parser.add_argument('--delta_percentile', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default=f'./step2_results_{datetime.now().strftime("%H%M%S")}')

    args = parser.parse_args()

    try:
        run_step2(args)
    except Exception as e:
        print(f"❌ Step2执行异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)