"""
Step1 损失函数定义
包含：重构损失 + 对比学习损失 + KL散度损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ComprehensiveLoss(nn.Module):
    """综合损失函数"""
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1, temperature=0.5):
        super().__init__()
        self.alpha = alpha  # 重构损失权重
        self.beta = beta    # 对比学习损失权重
        self.gamma = gamma  # KL散度损失权重
        self.temperature = temperature
        
    def reconstruction_loss(self, fused_evidence, reference):
        """重构损失 (交叉熵)"""
        return F.cross_entropy(
            fused_evidence.view(-1, fused_evidence.size(-1)),
            reference.argmax(dim=-1).view(-1)
        )
    
    def contrastive_loss(self, features):
        """对比学习损失 (InfoNCE)"""
        if features.shape[0] <= 1:
            return torch.tensor(0.0, device=features.device)
            
        # 计算相似度矩阵
        features_norm = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        # InfoNCE损失
        labels = torch.arange(features.shape[0], device=features.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def kl_divergence_loss(self, fused_evidence, reference):
        """KL散度损失"""
        log_fused = F.log_softmax(fused_evidence, dim=-1)
        target_dist = F.softmax(reference, dim=-1)
        
        return F.kl_div(log_fused, target_dist, reduction='batchmean')
    
    def forward(self, fused_evidence, reference, contrastive_features):
        """计算总损失"""
        
        # 各项损失
        recon_loss = self.reconstruction_loss(fused_evidence, reference)
        contrast_loss = self.contrastive_loss(contrastive_features)
        kl_loss = self.kl_divergence_loss(fused_evidence, reference)
        
        # 总损失
        total_loss = (self.alpha * recon_loss + 
                     self.beta * contrast_loss + 
                     self.gamma * kl_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'contrastive_loss': contrast_loss,
            'kl_loss': kl_loss
        }
