# models/step2_runner.py
"""
Step2 主入口：Evidence-Guided Refinement & Decoding
关键：不训练，只推理+决策
"""
import torch
import os
import sys
import argparse
from datetime import datetime

# ✅ 添加路径处理（与step1_train.py相同）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 现在可以正常导入
from models.step1_model import Step1EvidentialModel, load_pretrained_feddna
from models.step1_data import CloverDataLoader, Step1Dataset
from models.step2_refine import (
    select_high_confidence_reads,
    compute_cluster_centroids,
    refine_low_confidence_reads,
    compute_adaptive_delta
)
from models.step2_decode import (
    decode_cluster_consensus,
    save_consensus_sequences,
    compute_consensus_quality_metrics
)

# models/step2_runner.py - 修复模型加载部分

@torch.no_grad()
def run_step2(args):
    """
    Step2主流程：
    1. 加载Step1模型（freeze）
    2. 推理得到embeddings + evidence
    3. 证据筛选
    4. 簇修正
    5. consensus解码
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # ========== 1️⃣ 加载Step1模型 ==========
    print("\n" + "=" * 60)
    print("📦 加载Step1训练好的模型")
    print("=" * 60)
    
    checkpoint = torch.load(args.step1_checkpoint, map_location=device)
    
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
    data_loader = CloverDataLoader(args.experiment_dir)
    num_clusters = len(set(data_loader.clover_labels))
    
    # ✅ 使用Step1相同的参数重建模型
    model = Step1EvidentialModel(
        dim=model_dim,
        max_length=model_max_length,
        num_clusters=num_clusters,
        device=device
    ).to(device)
    
    # ✅ 尝试加载，如果有不匹配的参数就忽略
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"   ✅ 模型完全匹配加载")
    except RuntimeError as e:
        print(f"   ⚠️ 模型结构不完全匹配: {e}")
        print(f"   🔄 尝试忽略不匹配的参数...")
        
        # 获取当前模型的参数名
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint['model_state_dict'].keys())
        
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        print(f"      缺失参数: {missing_keys}")
        print(f"      多余参数: {unexpected_keys}")
        
        # 只加载匹配的参数
        filtered_state_dict = {
            k: v for k, v in checkpoint['model_state_dict'].items() 
            if k in model_keys
        }
        
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"   ✅ 已加载匹配的参数，忽略不匹配部分")
    
    model.eval()  # ✅ 评估模式，freeze参数
    
    print(f"   📊 最终模型参数: {sum(p.numel() for p in model.parameters()):,}")    
    print(f"   ✅ 模型已加载: {args.step1_checkpoint}")
    print(f"   📊 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== 2️⃣ 推理全部数据 ==========
    print("\n" + "=" * 60)
    print("🔮 Step1模型推理（提取embeddings + evidence）")
    print("=" * 60)
    
    dataset = Step1Dataset(data_loader, max_len=args.max_length)
    
    all_embeddings = []
    all_strength = []
    all_alpha = []
    all_evidence = []
    all_labels = []
    all_read_ids = []
    
    print(f"   处理 {len(dataset)} 条reads...")
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        reads = item['encoding'].unsqueeze(0).to(device)  # (1, L, 4)
        
        # Step1推理
        embeddings, pooled_emb = model.encode_reads(reads)
        evidence, strength, alpha = model.decode_to_evidence(embeddings)
        
        all_embeddings.append(pooled_emb.squeeze(0))
        all_strength.append(strength.mean())  # 平均strength
        all_alpha.append(alpha.squeeze(0))
        all_evidence.append(evidence.squeeze(0))
        all_labels.append(item['clover_label'])
        all_read_ids.append(idx)
        
        if (idx + 1) % 1000 == 0:
            print(f"      已处理: {idx+1}/{len(dataset)}")
    
    # 转换为张量
    embeddings = torch.stack(all_embeddings)  # (N, D)
    strength = torch.stack(all_strength)      # (N,)
    alpha = torch.stack(all_alpha)            # (N, L, 4)
    evidence = torch.stack(all_evidence)      # (N, L, 4)
    labels = torch.tensor(all_labels, device=device)  # (N,)
    
    print(f"   ✅ 推理完成:")
    print(f"      Embeddings: {embeddings.shape}")
    print(f"      Evidence: {evidence.shape}")
    print(f"      平均strength: {strength.mean():.4f}")
    
    # ========== 3️⃣ Phase A: 证据筛选 ==========
    print("\n" + "=" * 60)
    print("🔍 Phase A: 证据筛选")
    print("=" * 60)
    
    high_conf_mask, tau_used = select_high_confidence_reads(
        strength, 
        tau=args.tau,
        quantile=args.quantile
    )
    
    # ========== 4️⃣ Phase B: 簇修正 ==========
    print("\n" + "=" * 60)
    print("🔄 Phase B: 簇修正")
    print("=" * 60)
    
    # 计算簇中心（只用高置信度）
    centroids, cluster_sizes = compute_cluster_centroids(
        embeddings, labels, high_conf_mask
    )
    
    # 自适应计算delta
    if args.delta is None:
        delta = compute_adaptive_delta(
            embeddings, centroids, percentile=args.delta_percentile
        )
    else:
        delta = args.delta
        print(f"   🎯 使用固定delta: {delta:.4f}")
    
    # 修正低置信度reads
    new_labels, noise_mask, refine_stats = refine_low_confidence_reads(
        embeddings, labels, high_conf_mask, centroids, delta
    )
    
    # ========== 5️⃣ Phase C: Consensus解码 ==========
    print("\n" + "=" * 60)
    print("🧬 Phase C: Consensus解码")
    print("=" * 60)
    
    consensus_dict = decode_cluster_consensus(
        evidence, alpha, new_labels, strength
    )
    
    # 保存共识序列
    os.makedirs(args.output_dir, exist_ok=True)
    consensus_path = os.path.join(args.output_dir, "consensus_sequences.fasta")
    save_consensus_sequences(consensus_dict, consensus_path)
    
    # ========== 6️⃣ 生成报告 ==========
    print("\n" + "=" * 60)
    print("📊 Step2 最终统计")
    print("=" * 60)
    
    print(f"\n   📈 簇修正效果:")
    print(f"      原始簇数: {len(torch.unique(labels))}")
    print(f"      修正后簇数: {len(consensus_dict)}")
    print(f"      总噪声reads: {noise_mask.sum()}/{len(labels)} ({noise_mask.float().mean()*100:.1f}%)")
    print(f"      重新分配: {refine_stats['reassigned']}")
    print(f"      新增噪声: {refine_stats['marked_noise']}")
    
    print(f"\n   🧬 Consensus质量:")
    for label in sorted(list(consensus_dict.keys())[:5]):  # 显示前5个
        info = consensus_dict[label]
        print(f"      簇{label}: {info['num_reads']} reads, "
              f"strength={info['avg_strength']:.3f}, "
              f"len={len(info['consensus_seq'])}")
    
    # 保存完整结果
    results = {
        'new_labels': new_labels.cpu(),
        'noise_mask': noise_mask.cpu(),
        'high_conf_mask': high_conf_mask.cpu(),
        'strength': strength.cpu(),
        'consensus_dict': consensus_dict,
        'refine_stats': refine_stats,
        'args': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, "step2_results.pth")
    torch.save(results, results_path)
    print(f"\n   💾 完整结果已保存: {results_path}")
    
    print(f"\n🎉 Step2完成！")
    print(f"📁 输出目录: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step2: Evidence-Guided Refinement & Decoding')
    
    # 输入参数
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='实验目录（与Step1相同）')
    parser.add_argument('--step1_checkpoint', type=str, required=True,
                       help='Step1训练好的模型checkpoint')
    
    # 模型参数（需与Step1一致）
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Step2参数
    parser.add_argument('--tau', type=float, default=None,
                       help='置信度阈值（None=自动）')
    parser.add_argument('--quantile', type=float, default=0.5,
                       help='tau的分位数（当tau=None时）')
    parser.add_argument('--delta', type=float, default=None,
                       help='距离阈值（None=自适应）')
    parser.add_argument('--delta_percentile', type=int, default=10,
                       help='delta的百分位数（接收最近X%）')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str,
                       default=f'./step2_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 运行Step2
    results = run_step2(args)
