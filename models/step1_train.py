# models/step1_train.py
import torch
import torch.optim as optim
import argparse
import os
import sys
from datetime import datetime

# 添加路径，确保能导入 models 包
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.step1_model import Step1EvidentialModel, load_pretrained_feddna
from models.step1_data import CloverDataLoader, Step1Dataset, create_cluster_balanced_sampler
from models.step1_visualizer import Step1Visualizer


def train_step1(args):
    """步骤一训练主函数（支持迭代训练）"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")

    # 1️⃣ 加载数据
    print("\n" + "=" * 60)
    print("📂 数据加载")
    print("=" * 60)

    # ✅ 修复：安全获取 refined_labels 参数
    labels_path = getattr(args, 'refined_labels', None)
    
    data_loader = CloverDataLoader(args.experiment_dir, labels_path=labels_path)
    dataset = Step1Dataset(data_loader, max_len=args.max_length)

    # 2️⃣ 创建模型
    print("\n" + "=" * 60)
    print("🧠 模型创建")
    print("=" * 60)

    num_clover_clusters = len(set(data_loader.clover_labels))
    num_gt_clusters = len(data_loader.gt_cluster_seqs)
    
    # 确保簇数量足够
    num_clusters = max(num_clover_clusters, num_gt_clusters, args.min_clusters)

    model = Step1EvidentialModel(
        dim=args.dim,
        max_length=args.max_length,
        num_clusters=num_clusters,
        device=device
    ).to(device)

    print(f"   模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   当前簇数: {num_clover_clusters}")

    # 3️⃣ 加载预训练权重
    if args.feddna_checkpoint and os.path.exists(args.feddna_checkpoint):
        model = load_pretrained_feddna(model, args.feddna_checkpoint, device)
    else:
        print(f"⚠️ 预训练权重不存在或未指定: {getattr(args, 'feddna_checkpoint', 'None')}")
        print("   使用随机初始化权重")

    # 4️⃣ 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 5️⃣ 训练循环
    print("\n" + "=" * 60)
    print("🚀 开始训练")
    print("=" * 60)

    model.train()
    training_history = {'total_loss': [], 'avg_strength': [], 'high_conf_ratio': []}

    for epoch in range(args.epochs):
        batch_indices_list = create_cluster_balanced_sampler(
            dataset,
            batch_size=args.batch_size,
            max_clusters_per_batch=args.max_clusters_per_batch
        )

        epoch_loss = 0
        epoch_strength = 0
        epoch_high_conf = 0
        num_batches = 0

        for indices in batch_indices_list:
            if len(indices) < 2: continue

            # 构建batch
            batch_reads = [dataset[idx]['encoding'] for idx in indices]
            batch_labels = [dataset[idx]['clover_label'] for idx in indices]

            reads_batch = torch.stack(batch_reads).to(device)
            labels_batch = torch.tensor(batch_labels, device=device)

            # 前向传播 (传入 reads 用于重建输入)
            loss_dict, outputs = model(reads_batch, labels_batch, epoch=epoch)

            # 反向传播
            optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录
            epoch_loss += loss_dict['total'].item()
            epoch_strength += outputs['avg_strength']
            epoch_high_conf += outputs['high_conf_ratio']
            num_batches += 1

        if num_batches > 0:
            scheduler.step()
            avg_loss = epoch_loss / num_batches
            avg_strength = epoch_strength / num_batches
            avg_high_conf = epoch_high_conf / num_batches

            training_history['total_loss'].append(avg_loss)
            training_history['avg_strength'].append(avg_strength)
            training_history['high_conf_ratio'].append(avg_high_conf)

            print(f"   Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | Strength: {avg_strength:.2f} | HighConf: {avg_high_conf:.1%}")

    # 6️⃣ 保存最终模型
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    final_model_path = os.path.join(args.output_dir, "models", "step1_final_model.pth")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args)
    }, final_model_path)

    print(f"\n💾 模型已保存: {final_model_path}")
    
    # 生成可视化
    try:
        visualizer = Step1Visualizer(args.output_dir)
        visualizer.generate_all_outputs(training_history, model, args)
    except Exception as e:
        print(f"⚠️ 可视化生成跳过: {e}")

    # ✅ 返回模型路径，供外部调用
    return final_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 1 Training')

    # 必需参数
    parser.add_argument('--experiment_dir', type=str, required=True, help='实验目录')
    
    # 可选参数
    parser.add_argument('--feddna_checkpoint', type=str, default=None, help='预训练权重')
    # ✅ 修复：添加 refined_labels 参数定义
    parser.add_argument('--refined_labels', type=str, default=None, help='迭代修正后的标签文件')
    
    # 训练参数
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--min_clusters', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_clusters_per_batch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./step1_results', help='输出目录')
    parser.add_argument('--save_interval', type=int, default=10)

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train_step1(args)