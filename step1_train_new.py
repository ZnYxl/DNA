# step1_train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from models.step1_model import Step1EvidentialModel, load_pretrained_feddna
from utils.step1_utils import CloverDataset, create_mini_batch_sampler, prepare_reference_sequences

def parse_clover_results(clover_output_file):
    """
    解析Clover聚类结果
    返回: reads_data, cluster_labels, reference_seqs
    """
    # 这里需要根据你的Clover输出格式来实现
    # 示例实现：
    reads_data = []
    cluster_labels = []
    reference_seqs = {}
    
    # TODO: 根据实际的Clover输出格式解析
    # 假设格式为：
    # >cluster_0_read_1
    # ATCGATCG...
    # >cluster_0_consensus
    # ATCGATCG...
    
    return reads_data, cluster_labels, reference_seqs

def train_step1(args):
    """
    步骤一训练主函数
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1️⃣ 解析Clover结果
    print("Loading Clover clustering results...")
    reads_data, cluster_labels, reference_seqs = parse_clover_results(args.clover_file)
    
    # 2️⃣ 创建数据集
    dataset = CloverDataset(reads_data, cluster_labels, reference_seqs)
    print(f"Dataset size: {len(dataset)} reads")
    print(f"Number of clusters: {len(set(cluster_labels))}")
    
    # 3️⃣ 创建模型
    model = Step1EvidentialModel(
        dim=args.dim,
        noise_length=dataset.max_length,
        label_length=dataset.max_length,
        num_clusters=len(set(cluster_labels)),
        device=device
    ).to(device)
    
    # 4️⃣ 加载FedDNA预训练权重
    model = load_pretrained_feddna(model, args.feddna_checkpoint, device)
    
    # 5️⃣ 准备参考序列
    reference_tensors = prepare_reference_sequences(reference_seqs, dataset.max_length, device)
    
    # 6️⃣ 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 7️⃣ 训练循环
    model.train()
    for epoch in range(args.epochs):
        # 创建mini-batch
        batch_indices_list = create_mini_batch_sampler(dataset, args.batch_size)
        
        epoch_losses = {'total': 0, 'contrastive': 0, 'reconstruction': 0, 'kl_divergence': 0}
        num_batches = 0
        
        for batch_indices in batch_indices_list:
            # 构建batch
            batch_reads = []
            batch_labels = []
            
            for idx in batch_indices:
                read_tensor, label = dataset[idx]
                batch_reads.append(read_tensor)
                batch_labels.append(label)
            
            if len(batch_reads) == 0:
                continue
            
            # 转换为张量
            reads_batch = torch.stack(batch_reads).unsqueeze(1).to(device)  # (B, 1, L, 4)
            labels_batch = torch.tensor(batch_labels, device=device)  # (B,)
            
            # 前向传播
            loss_dict, outputs = model(reads_batch, labels_batch, reference_tensors, epoch)
            
            # 反向传播
            optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累计损失
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key].item()
            num_batches += 1
        
        scheduler.step()
        
        # 打印epoch结果
        if num_batches > 0:
            avg_losses = {k: v/num_batches for k, v in epoch_losses.items()}
            print(f"Epoch {epoch+1}/{args.epochs}:")
            print(f"  Total Loss: {avg_losses['total']:.4f}")
            print(f"  Contrastive: {avg_losses['contrastive']:.4f}")
            print(f"  Reconstruction: {avg_losses['reconstruction']:.4f}")
            print(f"  KL Divergence: {avg_losses['kl_divergence']:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, f"{args.output_dir}/step1_epoch_{epoch+1}.pth")
    
    print("Step 1 training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 1: Evidence-driven Training')
    
    # 数据参数
    parser.add_argument('--clover_file', type=str, required=True, help='Clover聚类结果文件')
    parser.add_argument('--feddna_checkpoint', type=str, 
                       default='result/FLDNA_I/I_1214234233/model/epoch1_I.pth',
                       help='FedDNA预训练权重路径')
    
    # 模型参数
    parser.add_argument('--dim', type=int, default=256, help='特征维度')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./step1_output', help='输出目录')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始训练
    train_step1(args)
