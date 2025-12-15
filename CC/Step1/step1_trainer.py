"""
Step1 训练器
负责基础训练循环的执行
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from step1_model import SimplifiedFedDNA
from step1_loss import ComprehensiveLoss
from step1_data import CloverClusterDataset

class BasicTrainer:
    """Step1基础训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = SimplifiedFedDNA(**config['model_params']).to(self.device)
        
        # 初始化优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training_params']['lr'])
        self.criterion = ComprehensiveLoss(**config['training_params']['loss_weights'])
        
        # 加载数据
        dataset = CloverClusterDataset(config['data_dir'])
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        print(f"🔧 Step1训练器初始化完成")
        print(f"   设备: {self.device}")
        print(f"   模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {'total_loss': 0, 'reconstruction_loss': 0, 'contrastive_loss': 0, 'kl_loss': 0}
        step_count = 0
        
        for i, (reads, ref) in enumerate(self.dataloader):
            reads = reads.to(self.device)
            ref = ref.squeeze(0).to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            evidence_batch, contrastive_features = self.model(reads)
            
            # 证据融合
            evidence_single_batch = evidence_batch.squeeze(0)
            fused_evidence, strengths = self.model.evidence_fusion(evidence_single_batch)
            
            # 计算损失
            contrastive_features_flat = contrastive_features.squeeze(0)
            losses = self.criterion(fused_evidence, ref, contrastive_features_flat)
            
            # 反向传播
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # 记录损失
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            step_count += 1
            
            # 打印进度
            if (i + 1) % 5 == 0:
                print(f"  📊 Step {i+1:3d} | Loss: {losses['total_loss'].item():.6f}")
        
        # 计算平均损失
        avg_losses = {key: val / max(1, step_count) for key, val in epoch_losses.items()}
        return avg_losses
    
    def train(self):
        """完整训练流程"""
        epochs = self.config['training_params']['epochs']
        history = []
        
        print(f"\n🚀 开始Step1训练 | Epochs: {epochs}")
        weights = self.config['training_params']['loss_weights']
        print(f"📊 损失函数权重: 重构={weights['alpha']}, 对比学习={weights['beta']}, KL散度={weights['gamma']}")
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch+1}/{epochs}")
            
            avg_losses = self.train_epoch(epoch)
            history.append(avg_losses)
            
            # 打印epoch总结
            print(f"📈 Epoch {epoch+1} 完成:")
            print(f"   Total Loss:    {avg_losses['total_loss']:.6f}")
            print(f"   Reconstruction: {avg_losses['reconstruction_loss']:.6f}")
            print(f"   Contrastive:   {avg_losses['contrastive_loss']:.6f}")
            print(f"   KL Divergence: {avg_losses['kl_loss']:.6f}")
            print("-" * 60)
        
        print("✅ Step1训练完成！")
        return history
    
    def save_model(self, filepath):
        """保存模型"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_class': 'SimplifiedFedDNA'
        }
        torch.save(save_dict, filepath)
        print(f"💾 Step1模型已保存到: {filepath}")
