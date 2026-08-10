import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import pathlib
import sys
import os
import numpy as np

# 假设你的 my_train.py 放在 code/ 目录下，和其他子文件夹同级
# 如果不在，请调整 sys.path 或移动文件
from data.DNA_data import MyDataset, collater, CustomSampler, CustomBatchSampler
from models.Model import Encoder, Model
from utils.Loss import CEBayesRiskLoss, KLDivergenceLoss

def main():
    # 1. 简单的参数配置
    parser = argparse.ArgumentParser(description="Single Client Training Baseline")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size (Clusters per batch)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--dim', type=int, default=256, help="Model dimension")
    # 默认使用 'I' 数据集，你可以改成 'B', 'P', 'S'
    parser.add_argument('--dataset_name', type=str, default='I', help="Dataset name: I, B, P, or S")
    args = parser.parse_args()

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用设备: {device}")

    # 2. 数据准备 (Data Loading)
    print(f"📥 正在加载数据: {args.dataset_name} ...")
    
    # 路径字典 (请确保 Dataset 文件夹在上一级或者路径正确)
    # 假设目录结构是 code/my_train.py 和 code/../Dataset
    path_dict = {
        'I': pathlib.Path('../Dataset/I'),
        'B': pathlib.Path('../Dataset/B'),
        'P': pathlib.Path('../Dataset/P'),
        'S': pathlib.Path('../Dataset/S')
    }
    
    # 检查路径是否存在
    if not path_dict[args.dataset_name].exists():
        # 如果找不到，尝试直接从当前目录找 (适应不同的运行位置)
        path_dict = {
            'I': pathlib.Path('Dataset/I'),
            'B': pathlib.Path('Dataset/B'),
            'P': pathlib.Path('Dataset/P'),
            'S': pathlib.Path('Dataset/S')
        }
    
    # 硬编码的长度参数 (来自 main_fl_dna.py)
    padding_length_dict = {'I': 155, 'B': 205, 'P': 188, 'S': 201}
    label_length_dict = {'I': 150, 'B': 200, 'P': 183, 'S': 196}
    
    if args.dataset_name not in padding_length_dict:
        raise ValueError(f"未知的数据集: {args.dataset_name}")

    padding_len = padding_length_dict[args.dataset_name]
    label_len = label_length_dict[args.dataset_name]

    # 实例化数据集
    train_set = MyDataset(path_dict, datasets=[args.dataset_name], mode='train')
    
    # --- 🔴 关键修复：使用定制采样器防止 IndexError ---
    # 1. Sampler: 对数据按长度进行分组排序
    train_sampler = CustomSampler(data=train_set)
    # 2. BatchSampler: 保证每个 Batch 里的数据长度一致
    train_batch_sampler = CustomBatchSampler(sampler=train_sampler, batch_size=args.batch_size, drop_last=True)
    # 3. Collater: 负责 Padding 和 One-Hot
    train_collate_fn = collater(padding_len)
    
    train_loader = DataLoader(
        dataset=train_set, 
        batch_sampler=train_batch_sampler, # 使用 batch_sampler
        collate_fn=train_collate_fn,
        num_workers=0 # 调试时设为0更安全
    )
    
    print(f"✅ 数据加载成功！训练集大小: {len(train_set)} 个簇")

    # 3. 模型初始化 (Model Initialization)
    print("🧠 正在初始化模型...")
    
    encoder = Encoder(dim=args.dim).to(device)
    model = Model(encoder, args.dim, padding_len, label_len).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # 损失函数 (DEL 标配: 贝叶斯风险 + KL散度)
    criterion_risk = CEBayesRiskLoss().to(device)
    criterion_kld = KLDivergenceLoss().to(device)

    # 4. 训练循环 (Training Loop)
    print("🔥 开始训练循环...")
    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        total_batches = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device).float() # Shape: (B, N, L, 4)
            labels = labels.to(device)         # Shape: (B, L)

            # --- 前向传播 ---
            # 这里的 outputs 是 fused_evidence
            outputs = model(inputs)

            # --- Label 处理 ---
            # 师姐的 Loss 需要 One-Hot 标签
            eye = torch.eye(4, dtype=torch.float32, device=device)
            labels_onehot = eye[labels.long()] 
            
            # --- 损失计算 (包含 Annealing) ---
            # KL 散度的权重随 epoch 增加而增加 (从 0 到 1)
            annealing_coef = min(1.0, (epoch + 0.1) / args.epochs) 
            
            loss_risk = criterion_risk(outputs, labels_onehot)
            loss_kld = criterion_kld(outputs, labels_onehot)
            
            loss = loss_risk + annealing_coef * loss_kld

            # --- 反向传播 ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_batches += 1

            if i % 10 == 0:
                print(f"   Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f} (Risk: {loss_risk.item():.4f}, KLD: {loss_kld.item():.4f})")
        
        avg_loss = epoch_loss / total_batches if total_batches > 0 else 0
        print(f"⭐️ Epoch {epoch+1} 完成! 平均 Loss: {avg_loss:.4f}")

    print("🎉 训练脚本运行结束！Baseline 验证成功。")

if __name__ == '__main__':
    main()