import torch
import torch.optim as optim
import argparse
import os
import sys
import time  # 新增
from datetime import datetime
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.step1_model import Step1EvidentialModel, load_pretrained_feddna
from models.step1_data import CloverDataLoader, Step1Dataset, create_cluster_balanced_sampler
from models.step1_visualizer import Step1Visualizer

# 定义一个自定义 Sampler 适配 DataLoader
class ListBatchSampler:
    def __init__(self, batches):
        self.batches = batches
    def __iter__(self):
        return iter(self.batches)
    def __len__(self):
        return len(self.batches)

def train_step1(args):
    """步骤一训练主函数（高性能优化版）"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")

    # 1️⃣ 加载数据
    print("\n" + "=" * 60)
    print("📂 数据加载")
    print("=" * 60)
    
    labels_path = getattr(args, 'refined_labels', None)
    data_loader = CloverDataLoader(args.experiment_dir, labels_path=labels_path)
    dataset = Step1Dataset(data_loader, max_len=args.max_length)

    # 2️⃣ 创建模型
    print("\n" + "=" * 60)
    print("🧠 模型创建")
    print("=" * 60)

    num_clover_clusters = len(set(data_loader.clover_labels))
    num_gt_clusters = len(data_loader.gt_cluster_seqs)
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
        print(f"   使用随机初始化权重")

    # 4️⃣ 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 5️⃣ 训练循环
    print("\n" + "=" * 60)
    print("🚀 开始训练 (已开启多进程加速与实时日志)")
    print("=" * 60)

    model.train()

    training_history = {
        'total_loss': [], 'avg_strength': [], 'high_conf_ratio': [],
        'contrastive_loss': [], 'reconstruction_loss': [], 'kl_loss': []
    }

    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 1. 生成 Batch 索引 (你之前的优化成果，非常快)
        batch_indices_list = create_cluster_balanced_sampler(
            dataset,
            batch_size=args.batch_size,
            max_clusters_per_batch=args.max_clusters_per_batch
        )
        
        # 2. 🔥 核心优化：使用 DataLoader 进行多进程加载
        # 这将解决 "CPU单核处理100万条数据太慢" 的问题
        # num_workers=8 表示开启8个进程并行读取数据
        batch_sampler = ListBatchSampler(batch_indices_list)
        train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_sampler=batch_sampler, 
            num_workers=8,  # 你的服务器有64核，开16个非常稳
            pin_memory=True
        )

        epoch_loss = 0
        epoch_con_loss = 0
        epoch_rec_loss = 0
        epoch_kl_loss = 0
        epoch_strength = 0
        epoch_high_conf = 0
        num_batches = 0
        
        total_batches = len(batch_indices_list)
        print(f"\n🔄 Epoch {epoch + 1}/{args.epochs} 开始... (共 {total_batches} Batches)")

        # 3. 训练循环 (带进度打印)
        for i, batch_data in enumerate(train_loader):
            # 获取数据 (DataLoader 自动帮我们整理好了)
            reads_batch = batch_data['encoding'].to(device)
            labels_batch = batch_data['clover_label'].to(device)

            # 前向传播
            loss_dict, outputs = model(reads_batch, labels_batch, epoch=epoch)

            # 反向传播
            optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录数据
            loss_val = loss_dict['total'].item()
            epoch_loss += loss_val
            epoch_con_loss += loss_dict['contrastive'].item()
            epoch_rec_loss += loss_dict['reconstruction'].item()
            epoch_kl_loss += loss_dict['kl_divergence'].item()
            epoch_strength += outputs['avg_strength']
            epoch_high_conf += outputs['high_conf_ratio']
            num_batches += 1
            
            # ✅ 实时日志：每 10 个 Batch 打印一次，让你知道它在动！
            if (i + 1) % 10 == 0:
                print(f"   [Batch {i+1}/{total_batches}] Loss: {loss_val:.4f} | "
                      f"Str: {outputs['avg_strength']:.1f}", end='\r')

        # Epoch 结束处理
        scheduler.step()
        epoch_time = time.time() - start_time
        
        # 计算平均值
        avg_loss = epoch_loss / num_batches
        avg_con = epoch_con_loss / num_batches
        avg_rec = epoch_rec_loss / num_batches
        avg_kl = epoch_kl_loss / num_batches
        avg_strength = epoch_strength / num_batches
        avg_high_conf = epoch_high_conf / num_batches

        # 存入历史
        training_history['total_loss'].append(avg_loss)
        training_history['contrastive_loss'].append(avg_con)
        training_history['reconstruction_loss'].append(avg_rec)
        training_history['kl_loss'].append(avg_kl)
        training_history['avg_strength'].append(avg_strength)
        training_history['high_conf_ratio'].append(avg_high_conf)

        # 打印 Epoch 总结 (换行)
        print(f"\n   ✅ Epoch {epoch + 1} 完成 ({epoch_time:.1f}s) | "
              f"Avg Loss: {avg_loss:.4f} | Avg Str: {avg_strength:.1f}")

    # 6️⃣ 保存
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    final_model_path = os.path.join(args.output_dir, "models", "step1_final_model.pth")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args)
    }, final_model_path)
    
    print(f"\n💾 模型已保存: {final_model_path}")

    # 可视化
    try:
        visualizer = Step1Visualizer(args.output_dir)
        visualizer.generate_all_outputs(training_history, model, args)
    except Exception as e:
        print(f"⚠️ 可视化生成跳过: {e}")

    return final_model_path

if __name__ == "__main__":
    # 参数解析部分保持不变，复制你原来的即可
    parser = argparse.ArgumentParser(description='Step 1 Training')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--feddna_checkpoint', type=str, default=None)
    parser.add_argument('--refined_labels', type=str, default=None)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--min_clusters', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_clusters_per_batch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--output_dir', type=str, default='./step1_results')
    parser.add_argument('--save_interval', type=int, default=10)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_step1(args)