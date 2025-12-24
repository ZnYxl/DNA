# models/step2_runner.py
"""
Step2 主入口：Evidence-Guided Refinement & Decoding
关键：不训练，只推理+决策
✅ 相对不确定性原则：簇内比较，偏向确定性
✅ 修复版 v2：
   1. 包含 length_adapter 权重预加载修复
   2. 包含 迭代接口 (next_round_files)
   3. 🔥 新增：强制 Padding 到 max_length，避开未训练的 adapter
"""
import torch
import torch.nn as nn
import torch.nn.functional as F  # ✅ 需要用到 F.pad
import os
import sys
import argparse
import numpy as np
from datetime import datetime

# ✅ 添加路径处理（与step1_train.py相同）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 现在可以正常导入
from models.step1_model import Step1EvidentialModel, load_pretrained_feddna
from models.step1_data import CloverDataLoader, Step1Dataset
from models.step2_refine import (
    split_confidence_by_percentile,
    compute_cluster_centroids,
    refine_low_confidence_reads,
    compute_adaptive_delta
)
from models.step2_decode import (
    decode_cluster_consensus,
    save_consensus_sequences,
    compute_consensus_quality_metrics
)


@torch.no_grad()
def run_step2(args):
    """
    Step2主流程：
    1. 加载Step1模型（freeze）
    2. 推理得到embeddings + evidence
    3. 相对证据筛选（簇内比较）
    4. 簇修正（只修正低置信度）
    5. 偏向确定性的consensus解码
    6. 准备下一轮迭代数据
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

    # ✅ 使用Step1训练时保存的参数
    if 'args' in checkpoint:
        step1_args = checkpoint['args']
        print(f"   📋 Step1训练参数:")
        print(f"      dim: {step1_args.get('dim', args.dim)}")
        print(f"      max_length: {step1_args.get('max_length', args.max_length)}")

        # 使用Step1的参数
        model_dim = step1_args.get('dim', args.dim)
        model_max_length = step1_args.get('max_length', args.max_length)
    else:
        print(f"   ⚠️ checkpoint中没有保存args，使用当前参数")
        model_dim = args.dim
        model_max_length = args.max_length

    # 重建数据加载器
    try:
        data_loader = CloverDataLoader(args.experiment_dir)
        num_clusters = len(set(data_loader.clover_labels))
        print(f"   📊 数据统计: {len(data_loader.reads)} reads, {num_clusters} 簇")
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return None

    # ✅ 使用Step1相同的参数重建模型
    try:
        model = Step1EvidentialModel(
            dim=model_dim,
            max_length=model_max_length,
            num_clusters=num_clusters,
            device=device
        ).to(device)
        print(f"   ✅ 模型结构创建成功")
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        return None

    # ================= 🔴 核心修复 1: 权重预加载 🔴 =================
    # 手动检查并初始化 length_adapter，防止权重加载被跳过
    try:
        state_dict = checkpoint['model_state_dict']
        if 'length_adapter.weight' in state_dict:
            print(f"   🔧 检测到 checkpoint 包含 length_adapter，正在预初始化...")
            weight_shape = state_dict['length_adapter.weight'].shape
            in_features = weight_shape[1]
            out_features = weight_shape[0]
            
            # 手动初始化层
            model.length_adapter = nn.Linear(in_features, out_features).to(device)
            print(f"      已初始化: Linear({in_features} -> {out_features})")
    except Exception as e:
        print(f"   ⚠️ 预初始化 length_adapter 时出错 (非致命): {e}")

    # ✅ 尝试加载
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"   ✅ 模型完全匹配加载")
    except RuntimeError as e:
        print(f"   ⚠️ 模型结构不完全匹配: {e}")
        # ... (省略之前的过滤加载代码，保持简洁，逻辑一样) ...
        # 如果需要完整的过滤代码，可以保留之前的写法，这里简化展示核心逻辑
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"   ✅ 尝试强制加载 (strict=False)")

    model.eval()
    print(f"   📊 模型参数总数: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 2️⃣ 推理全部数据 (批量优化版) ==========
    print("\n" + "=" * 60)
    print("🔮 Step1模型推理（提取embeddings + evidence）")
    print("=" * 60)

    try:
        dataset = Step1Dataset(data_loader, max_len=model_max_length)
        # ✅ 使用 DataLoader 进行批量推理
        # num_workers=4 可以利用你的多核CPU加速数据加载
        inference_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True
        )
        print(f"   📊 数据集大小: {len(dataset)} | Batch Size: 1024")
    except Exception as e:
        print(f"   ❌ 数据集创建失败: {e}")
        return None

    all_embeddings = []
    all_strength = []
    all_alpha = []
    all_evidence = []
    all_labels = []
    
    # 只需要存 read_idx 用于后续对齐，或者直接按顺序
    print(f"   🔄 开始批量推理...")

    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(inference_loader):
            # 获取数据
            reads = batch_data['encoding'].to(device) # (B, L, 4)
            labels = batch_data['clover_label']       # list or tensor
            
            # ================= 强制 Padding (你的核心修复) =================
            curr_len = reads.shape[1]
            target_len = model_max_length
            if curr_len > target_len:
                reads = reads[:, :target_len, :]
            elif curr_len < target_len:
                pad_len = target_len - curr_len
                reads = F.pad(reads, (0, 0, 0, pad_len), "constant", 0)
            # ================================================================

            # 批量推理
            embeddings, pooled_emb = model.encode_reads(reads)
            evidence, strength, alpha = model.decode_to_evidence(embeddings)

            # 收集结果 (转到CPU以节省显存)
            all_embeddings.append(pooled_emb.cpu())
            all_strength.append(strength.mean(dim=1).cpu())
            all_alpha.append(alpha.cpu())
            all_evidence.append(evidence.cpu())
            
            # 处理 labels (如果是tensor转list，或者直接extend)
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.tolist())
            else:
                all_labels.extend(labels)

            if (batch_idx + 1) % 100 == 0:
                print(f"      已处理 Batch: {batch_idx + 1}/{len(inference_loader)}")

    if len(all_embeddings) == 0:
        print(f"   ❌ 没有成功推理的reads！")
        return None

    # 拼接结果
    try:
        embeddings = torch.cat(all_embeddings, dim=0).to(device)
        strength = torch.cat(all_strength, dim=0).to(device)
        alpha = torch.cat(all_alpha, dim=0).to(device)
        evidence = torch.cat(all_evidence, dim=0).to(device)
        labels = torch.tensor(all_labels, device=device)

        print(f"   ✅ 推理完成:")
        print(f"      Total: {len(labels)} reads")
        print(f"      Embeddings: {embeddings.shape}")
    except Exception as e:
        print(f"   ❌ 拼接张量失败 (可能显存不足): {e}")
        return None
    # ========== 3️⃣ Phase A: 相对证据筛选 ==========
    print("\n" + "=" * 60)
    print("🔍 Phase A: 相对证据筛选（簇内比较）")
    print("=" * 60)

    try:
        low_conf_mask, conf_stats = split_confidence_by_percentile(
            strength, labels, p=args.uncertainty_percentile
        )
        high_conf_mask = ~low_conf_mask

    except Exception as e:
        print(f"   ❌ 相对证据筛选失败: {e}")
        return None

    # ========== 4️⃣ Phase B: 簇修正 ==========
    print("\n" + "=" * 60)
    print("🔄 Phase B: 簇修正（只修正低置信度reads）")
    print("=" * 60)

    try:
        centroids, cluster_sizes = compute_cluster_centroids(
            embeddings, labels, high_conf_mask
        )

        if args.delta is None:
            delta = compute_adaptive_delta(
                embeddings, centroids, percentile=args.delta_percentile
            )
        else:
            delta = args.delta
            print(f"   🎯 使用固定delta: {delta:.4f}")

        new_labels, noise_mask, refine_stats = refine_low_confidence_reads(
            embeddings, labels, low_conf_mask, centroids, delta
        )

    except Exception as e:
        print(f"   ❌ 簇修正失败: {e}")
        return None

    # ========== 5️⃣ Phase C: 偏向确定性的Consensus解码 ==========
    print("\n" + "=" * 60)
    print("🧬 Phase C: 偏向确定性的Consensus解码")
    print("=" * 60)

    try:
        consensus_dict = decode_cluster_consensus(
            evidence, alpha, new_labels, strength, high_conf_mask
        )

        os.makedirs(args.output_dir, exist_ok=True)
        consensus_path = os.path.join(args.output_dir, "consensus_sequences.fasta")
        save_consensus_sequences(consensus_dict, consensus_path)

    except Exception as e:
        print(f"   ❌ Consensus解码失败: {e}")
        return None

    # ========== 6️⃣ 生成报告 ==========
    print("\n" + "=" * 60)
    print("📊 Step2 最终统计")
    print("=" * 60)

    print(f"\n   📈 簇修正效果:")
    print(f"      原始簇数: {len(torch.unique(labels))}")
    print(f"      修正后簇数: {len(consensus_dict)}")
    print(f"      总噪声reads: {noise_mask.sum()}/{len(labels)} ({noise_mask.float().mean() * 100:.1f}%)")
    print(f"      重新分配: {refine_stats['reassigned']}")
    print(f"      新增噪声: {refine_stats['marked_noise']}")

    print(f"\n   🧬 Consensus质量:")
    for label in sorted(list(consensus_dict.keys())[:5]):
        info = consensus_dict[label]
        print(f"      簇{label}: {info['num_reads']} reads ({info['num_high_conf']} 高置信度), "
              f"strength={info['avg_strength']:.3f}, "
              f"len={len(info['consensus_seq'])}")

    # ========== 7️⃣ 准备下一轮迭代的数据 (缝合接口) ==========
    print("\n" + "=" * 60)
    print("🔄 准备下一轮 (Next Round) 数据")
    print("=" * 60)
    
    # 1. 保存新的伪标签 (Refined Labels)
    next_round_dir = os.path.join(args.experiment_dir, "04_Iterative_Labels")
    os.makedirs(next_round_dir, exist_ok=True)
    
    timestamp_id = datetime.now().strftime("%H%M%S")
    label_save_path = os.path.join(next_round_dir, f"refined_labels_{timestamp_id}.txt")
    
    np.savetxt(label_save_path, new_labels.cpu().numpy(), fmt='%d')
    print(f"   📝 修正标签已保存: {label_save_path}")

    # 保存完整结果
    try:
        results = {
            'new_labels': new_labels.cpu(),
            'noise_mask': noise_mask.cpu(),
            'high_conf_mask': high_conf_mask.cpu(),
            'low_conf_mask': low_conf_mask.cpu(),
            'strength': strength.cpu(),
            'consensus_dict': consensus_dict,
            'refine_stats': refine_stats,
            'conf_stats': conf_stats,
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

    print(f"\n🎉 Step2完成！相对不确定性原则生效！")
    print(f"📁 输出目录: {args.output_dir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step2: Evidence-Guided Refinement & Decoding')

    parser.add_argument('--experiment_dir', type=str, required=True, help='实验目录')
    parser.add_argument('--step1_checkpoint', type=str, required=True, help='Step1 checkpoint')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--uncertainty_percentile', type=float, default=0.2, help='低置信度百分比')
    parser.add_argument('--delta', type=float, default=None, help='距离阈值')
    parser.add_argument('--delta_percentile', type=int, default=10, help='delta百分位数')
    parser.add_argument('--output_dir', type=str,
                        default=f'./step2_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        help='输出目录')

    args = parser.parse_args()

    try:
        run_step2(args)
    except Exception as e:
        print(f"❌ Step2执行异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)