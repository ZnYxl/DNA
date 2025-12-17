# models/step1_train.py
import torch
import torch.optim as optim
import argparse
import os
import sys
from datetime import datetime

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.step1_model import Step1EvidentialModel, load_pretrained_feddna
from models.step1_data import CloverDataLoader, Step1Dataset, create_cluster_balanced_sampler, seq_to_onehot
from models.step1_visualizer import Step1Visualizer

def evaluate_with_gt(outputs, data_loader, batch_gt_labels, device):
    """
    ✅ GT只用于评估，不参与训练
    计算ARI、NMI等指标
    """
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        return {
            'gt_available': True,
            'note': 'GT evaluation metrics can be added here'
        }
    except ImportError:
        return {'gt_available': False}

def train_step1(args):
    """步骤一训练主函数（严格自监督版本）"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # 1️⃣ 加载数据
    print("\n" + "=" * 60)
    print("📂 数据加载")
    print("=" * 60)
    
    data_loader = CloverDataLoader(args.experiment_dir)
    dataset = Step1Dataset(data_loader, max_len=args.max_length)
    
    # 2️⃣ 创建模型
    print("\n" + "=" * 60)
    print("🧠 模型创建")
    print("=" * 60)
    
    num_clover_clusters = len(set(data_loader.clover_labels))
    num_gt_clusters = len(data_loader.gt_cluster_seqs)
    
    model = Step1EvidentialModel(
        dim=args.dim,
        max_length=args.max_length,
        num_clusters=max(num_clover_clusters, num_gt_clusters, args.min_clusters),
        device=device
    ).to(device)
    
    print(f"   模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Clover簇数: {num_clover_clusters}")
    print(f"   GT簇数: {num_gt_clusters}")
    
    # 3️⃣ 加载FedDNA预训练权重
    if args.feddna_checkpoint and os.path.exists(args.feddna_checkpoint):
        model = load_pretrained_feddna(model, args.feddna_checkpoint, device)
    else:
        print(f"⚠️ FedDNA权重文件不存在: {args.feddna_checkpoint}")
        print("   使用随机初始化权重")
    
    # ✅ 4️⃣ GT只用于评估，不参与训练
    print(f"\n📊 GT数据状态:")
    print(f"   - GT簇数: {len(data_loader.gt_cluster_seqs)}")
    print(f"   - GT用途: 仅用于评估和监控")
    print(f"   - ❗ GT不参与任何训练loss")
    
    # 5️⃣ 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 6️⃣ 训练循环
    print("\n" + "=" * 60)
    print("🚀 开始训练（严格自监督模式）")
    print("=" * 60)
    
    model.train()
    training_history = {
        'total_loss': [],
        'contrastive_loss': [],
        'reconstruction_loss': [],
        'kl_loss': [],
        'strength_incentive_loss': [],
        'avg_strength': [],
        'high_conf_ratio': [],
        'annealing_coef': [] # ✅ 新增：记录退火系数
    }
    
    for epoch in range(args.epochs):
        # 创建cluster-balanced batch
        batch_indices_list = create_cluster_balanced_sampler(
            dataset, 
            batch_size=args.batch_size,
            max_clusters_per_batch=args.max_clusters_per_batch
        )
        
        epoch_losses = {
            'total': 0, 'contrastive': 0, 
            'reconstruction': 0, 'kl_divergence': 0
        }
        epoch_stats = {
            'avg_strength': 0,
            'high_conf_ratio': 0
        }
        
        current_annealing_coef = 0.0 # 用于记录当前epoch实际使用的系数
        num_batches = 0
        successful_batches = 0
        
        print(f"\n📦 生成 {len(batch_indices_list)} 个batch")
        
        for batch_idx, indices in enumerate(batch_indices_list):
            if len(indices) < 2:
                continue
            
            # 构建batch数据
            batch_reads = []
            batch_clover_labels = []
            batch_gt_labels = []
            
            for idx in indices:
                item = dataset[idx]
                batch_reads.append(item['encoding'])
                batch_clover_labels.append(item['clover_label'])
                batch_gt_labels.append(item['gt_label'])
            
            # 转换为张量
            reads_batch = torch.stack(batch_reads).to(device)
            clover_labels_batch = torch.tensor(batch_clover_labels, device=device)
            
            # ✅ 前向传播
            try:
                loss_dict, outputs = model(
                    reads_batch, 
                    clover_labels_batch,
                    epoch=epoch
                )
                
                # ✅ 获取真实的 Annealing Coef
                current_annealing_coef = loss_dict.get('annealing_coef', 0.0)
                
                # ✅ 检查loss有效性
                if torch.isnan(loss_dict['total']) or torch.isinf(loss_dict['total']):
                    print(f"   ⚠️ Batch {batch_idx}: 检测到异常loss，跳过")
                    continue
                
                # 反向传播
                optimizer.zero_grad()
                loss_dict['total'].backward()
                
                # ✅ 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"   ⚠️ Batch {batch_idx}: 梯度异常，跳过更新")
                    continue
                
                optimizer.step()
                successful_batches += 1
                
            except Exception as e:
                print(f"   ❌ Batch {batch_idx}: 训练异常 {e}，跳过")
                continue
            
            # 累计损失和统计
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key].item()
            
            epoch_stats['avg_strength'] += outputs['avg_strength']
            epoch_stats['high_conf_ratio'] += outputs['high_conf_ratio']
            num_batches += 1
            
            # 打印batch进度
            if batch_idx % 50 == 0:
                print(f"   Batch {batch_idx}/{len(batch_indices_list)}: "
                      f"Loss={loss_dict['total'].item():.4f}, "
                      f"Strength={outputs['avg_strength']:.3f}, "
                      f"KL_Coef={current_annealing_coef:.3f}") # 实时打印系数
        
        # ✅ Scheduler Step
        if successful_batches > 0:
            scheduler.step()
        
        # ✅ 记录历史
        if num_batches > 0:
            avg_losses = {k: v/num_batches for k, v in epoch_losses.items()}
            avg_stats = {k: v/num_batches for k, v in epoch_stats.items()}
            
            training_history['total_loss'].append(avg_losses.get('total', 0.0))
            training_history['contrastive_loss'].append(avg_losses.get('contrastive', 0.0))
            training_history['reconstruction_loss'].append(avg_losses.get('reconstruction', 0.0))
            training_history['kl_loss'].append(avg_losses.get('kl_divergence', 0.0))
            training_history['avg_strength'].append(avg_stats.get('avg_strength', 0.0))
            training_history['high_conf_ratio'].append(avg_stats.get('high_conf_ratio', 0.0))
            training_history['annealing_coef'].append(current_annealing_coef) # 记录系数
            
            # ✅ 详细的epoch报告
            print(f"\n📊 Epoch {epoch+1}/{args.epochs}:")
            print(f"   📉 损失:")
            print(f"      Total: {avg_losses['total']:.4f}")
            print(f"      Contrastive: {avg_losses['contrastive']:.4f}")
            print(f"      Reconstruction: {avg_losses['reconstruction']:.4f}")
            print(f"      KL Divergence: {avg_losses['kl_divergence']:.4f} (Raw)")
            print(f"   📊 Evidence统计:")
            print(f"      平均Strength: {avg_stats['avg_strength']:.3f}")
            print(f"      高置信度比例: {avg_stats['high_conf_ratio']*100:.1f}%")
            print(f"   ⚙️ 训练状态:")
            print(f"      Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print(f"      Annealing Coef: {current_annealing_coef:.3f}") # ✅ 使用真实系数
            print(f"      成功Batch: {successful_batches}/{len(batch_indices_list)}")
            
            # ✅ 智能状态提示
            if epoch < 5:
                print(f"   🔥 [Phase 1] 强制学习: 对比学习 Warm-up (无筛选), KL 系数=0")
            elif epoch < 10:
                print(f"   💪 [Phase 2] 积累信心: 对比学习 (开启筛选), KL 系数=0")
            else:
                print(f"   ✂️ [Phase 3] 证据修剪: KL 正则化介入 (系数={current_annealing_coef:.3f})")
                
        else:
            print(f"\n⚠️ Epoch {epoch+1}: 没有成功的batch，跳过")
            for k in training_history:
                training_history[k].append(0.0)
        
        # 保存checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_history': training_history,
                'args': vars(args)
            }
            checkpoint_path = os.path.join(args.output_dir, "models", f"step1_epoch_{epoch+1}.pth")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f"   💾 保存checkpoint: {checkpoint_path}")
    
    # ✅ 训练完成后打印历史记录统计
    print(f"\n📊 训练历史记录统计:")
    for key, values in training_history.items():
        if len(values) > 0:
            print(f"   {key}: {len(values)} 条记录, 最终值: {values[-1]:.6f}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "models", "step1_final_model.pth")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': training_history,
        'args': vars(args)
    }, final_model_path)
    
    # 生成可视化
    print(f"\n" + "=" * 60)
    print("📊 生成训练结果与可视化")
    print("=" * 60)
    
    try:
        visualizer = Step1Visualizer(args.output_dir)
        visualizer.generate_all_outputs(training_history, model, args)
    except Exception as e:
        print(f"⚠️ 可视化生成失败 (可能缺少依赖): {e}")
    
    print(f"\n🎉 步骤一训练完成！")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"💾 最终模型: {final_model_path}")
    
    return model, training_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='步骤一：Evidence-driven训练（严格自监督）')
    
    # 数据参数
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='实验目录路径')
    parser.add_argument('--feddna_checkpoint', type=str, 
                       default='result/FLDNA_I/I_1214234233/model/epoch1_I.pth',
                       help='FedDNA预训练权重路径')
    
    # 模型参数
    parser.add_argument('--dim', type=int, default=256, help='特征维度')
    parser.add_argument('--max_length', type=int, default=150, help='序列最大长度')
    parser.add_argument('--min_clusters', type=int, default=50, help='最小簇数量')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_clusters_per_batch', type=int, default=5, help='每个batch最大簇数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, 
                       default=f'./step1_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='输出目录')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始训练
    model, history = train_step1(args)