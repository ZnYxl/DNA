# models/step1_model.py - 完整修复版
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Model import Encoder, RNNBlock
from utils.Loss import CEBayesRiskLoss, KLDivergenceLoss
import numpy as np

class Step1EvidentialModel(nn.Module):
    """
    步骤一：Evidence-driven训练模型（严格自监督版本）
    ❗ GT只用于评估，不参与训练loss
    """
    def __init__(self, 
                 dim=256, 
                 max_length=150,
                 num_clusters=50,
                 device='cuda'):
        super().__init__()
        
        # ===== FedDNA 核心组件 =====
        self.encoder = Encoder(dim=dim)
        
        # ✅ 修复：动态适配length_adapter
        self.length_adapter = None  # 延迟初始化
        
        self.rnnblock = RNNBlock(in_channels=dim, lstm_hidden_dim=256, rnn_dropout_p=0.1)
        
        # ===== 对比学习组件 =====
        self.projection_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 128)
        )
        
        self.dim = dim
        self.max_length = max_length
        self.num_clusters = num_clusters
        self.device = device
    
    def _init_length_adapter_if_needed(self, seq_len):
        """延迟初始化length_adapter，避免形状不匹配"""
        if self.length_adapter is None:
            self.length_adapter = nn.Linear(seq_len, self.max_length).to(self.device)
            print(f"   🔧 动态初始化length_adapter: {seq_len} -> {self.max_length}")
    
    def encode_reads(self, reads):
        """
        ✅ 使用FedDNA Encoder编码reads
        Args:
            reads: (B, L, 4) 批次中的reads
        Returns:
            embeddings: (B, L, dim) 编码后的特征
            pooled_emb: (B, dim) 池化后的全局特征
        """
        B, L, D = reads.shape
        
        # FedDNA Encoder: Conv2d + ConMamba
        embeddings = self.encoder(reads)  # (B, L, dim)
        
        # 全局池化用于对比学习
        pooled_emb = embeddings.mean(dim=1)  # (B, dim)
        
        return embeddings, pooled_emb
    
    def decode_to_evidence(self, embeddings):
        """
        ✅ 使用FedDNA Decoder生成evidence
        Args:
            embeddings: (B, L, dim)
        Returns:
            evidence: (B, L, 4) 每个位置的ACGT evidence
            strength: (B, L) 证据强度
            alpha: (B, L, 4) Dirichlet参数
        """
        B, L, D = embeddings.shape
        
        # 动态初始化length_adapter
        self._init_length_adapter_if_needed(L)
        
        # 长度适配
        if L != self.max_length:
            adapted = embeddings.permute(0, 2, 1)  # (B, dim, L)
            adapted = self.length_adapter(adapted)  # (B, dim, max_length)
            adapted = adapted.permute(0, 2, 1)     # (B, max_length, dim)
        else:
            adapted = embeddings
        
        # FedDNA RNN Decoder
        evidence = self.rnnblock(adapted)  # (B, L, 4)
        
        # ✅ 数值稳定的evidence处理
        evidence = torch.clamp(evidence, min=1e-8, max=1e8)  # 防止极值
        
        # 正确的evidence strength计算
        K = evidence.size(-1)  # 4
        alpha = evidence + 1.0
        strength = torch.sum(alpha, dim=-1)  # (B, L)
        
        return evidence, strength, alpha
    
    def contrastive_learning_with_evidence_filter(self, pooled_emb, cluster_labels, strength, 
                                                 temperature=0.1, epoch=0, warmup_epochs=5):
        """✅ 修复：数值稳定的对比学习"""
        if epoch < warmup_epochs:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if pooled_emb.size(0) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 投影到对比学习空间
        proj_emb = self.projection_head(pooled_emb)
        proj_emb = F.normalize(proj_emb, dim=-1)
        
        # ✅ 修复：数值稳定的confidence计算
        confidence = strength.mean(dim=1)  # (B,)
        
        # 检查NaN
        if torch.isnan(confidence).any():
            print(f"   ⚠️ 检测到confidence NaN，跳过对比学习")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # ✅ 改进：使用分位数而不是均值作为阈值
        conf_threshold = torch.quantile(confidence, 0.6)  # 更稳定的阈值
        conf_mask = confidence > conf_threshold
        
        if conf_mask.sum() < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(proj_emb, proj_emb.T) / temperature
        
        # 构建正样本mask
        labels_expanded = cluster_labels.unsqueeze(1)
        clover_positive_mask = (labels_expanded == labels_expanded.T).float()
        evidence_positive_mask = (conf_mask.unsqueeze(1) & conf_mask.unsqueeze(0)).float()
        
        positive_mask = clover_positive_mask * evidence_positive_mask
        positive_mask.fill_diagonal_(0)
        
        # ✅ 数值稳定的InfoNCE
        exp_sim = torch.exp(torch.clamp(sim_matrix, max=10))  # 防止exp爆炸
        
        numerator = torch.sum(exp_sim * positive_mask, dim=1) + 1e-8
        denominator = torch.sum(exp_sim, dim=1) - torch.diag(exp_sim) + 1e-8
        
        loss = -torch.log(numerator / denominator)
        
        # 只计算有正样本的loss
        valid_mask = (torch.sum(positive_mask, dim=1) > 0)
        if valid_mask.sum() > 0:
            final_loss = loss[valid_mask].mean()
            # 检查最终loss
            if torch.isnan(final_loss) or torch.isinf(final_loss):
                print(f"   ⚠️ 对比学习loss异常，返回0")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            return final_loss
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def self_reconstruction_loss(self, evidence, alpha, cluster_labels):
        """✅ 修复：数值稳定的重建损失"""
        bayes_risk = CEBayesRiskLoss().to(self.device)
        kld_loss_fn = KLDivergenceLoss().to(self.device)
        
        total_recon_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_kl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        fused_consensus = {}
        
        unique_labels = torch.unique(cluster_labels)
        processed_clusters = 0
        
        for label in unique_labels:
            if label < 0:
                continue
            
            cluster_mask = (cluster_labels == label)
            cluster_count = cluster_mask.sum()
            
            if cluster_count < 2:
                continue
            
            cluster_evidence = evidence[cluster_mask]
            cluster_alpha = alpha[cluster_mask]
            cluster_strength = torch.sum(cluster_alpha, dim=-1)
            
            # ✅ 检查NaN
            if torch.isnan(cluster_strength).any():
                print(f"   ⚠️ 簇{label}的strength包含NaN，跳过")
                continue
            
            # Evidence-weighted融合
            weights = F.softmax(cluster_strength.mean(dim=1), dim=0)
            weights = weights.unsqueeze(1).unsqueeze(2)
            
            fused_evidence = torch.sum(cluster_evidence * weights, dim=0, keepdim=True)
            fused_consensus[label.item()] = fused_evidence
            
            # 自重建损失
            cluster_recon_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            cluster_kl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            for i in range(cluster_count):
                read_evidence = cluster_evidence[i:i+1]
                
                try:
                    recon_loss_i = bayes_risk(fused_evidence, read_evidence)
                    kl_loss_i = kld_loss_fn(fused_evidence, read_evidence)
                    
                    # 检查loss有效性
                    if not (torch.isnan(recon_loss_i) or torch.isinf(recon_loss_i)):
                        cluster_recon_loss = cluster_recon_loss + recon_loss_i
                    if not (torch.isnan(kl_loss_i) or torch.isinf(kl_loss_i)):
                        cluster_kl_loss = cluster_kl_loss + kl_loss_i
                        
                except Exception as e:
                    print(f"   ⚠️ 簇{label}第{i}个read计算loss失败: {e}")
                    continue
            
            total_recon_loss = total_recon_loss + cluster_recon_loss
            total_kl_loss = total_kl_loss + cluster_kl_loss
            processed_clusters += 1
        
        # 归一化
        if processed_clusters > 0:
            total_recon_loss = total_recon_loss / processed_clusters
            total_kl_loss = total_kl_loss / processed_clusters
        
        return total_recon_loss, total_kl_loss, fused_consensus
    
    def forward(self, reads, cluster_labels, epoch=0):
        """
        ✅ 完整的前向传播
        Args:
            reads: (B, L, 4) mini-batch reads
            cluster_labels: (B,) Clover标签（仅用于组织对比学习）
            epoch: 当前epoch
        """
        # 1️⃣ FedDNA Encoder
        embeddings, pooled_emb = self.encode_reads(reads)
        
        # 2️⃣ FedDNA Decoder
        evidence, strength, alpha = self.decode_to_evidence(embeddings)
        
        # 3️⃣ Evidence-filtered对比学习
        contrastive_loss = self.contrastive_learning_with_evidence_filter(
            pooled_emb, cluster_labels, strength, epoch=epoch
        )
        
        # 4️⃣ 自重建损失（不使用GT）
        recon_loss, kl_loss, fused_consensus = self.self_reconstruction_loss(
            evidence, alpha, cluster_labels
        )
        
        # 5️⃣ 总损失（带annealing）
        annealing_coef = min(1.0, epoch / 10)
        total_loss = contrastive_loss + recon_loss + annealing_coef * kl_loss
        
        # 6️⃣ 统计信息（用于监控��
        avg_strength = strength.mean().item()
        high_conf_ratio = (strength.mean(dim=1) > strength.mean()).float().mean().item()
        
        loss_dict = {
            'total': total_loss,
            'contrastive': contrastive_loss,
            'reconstruction': recon_loss,
            'kl_divergence': kl_loss,
            'annealing_coef': annealing_coef
        }
        
        outputs = {
            'embeddings': embeddings,
            'evidence': evidence,
            'strength': strength,
            'alpha': alpha,
            'fused_consensus': fused_consensus,
            'avg_strength': avg_strength,
            'high_conf_ratio': high_conf_ratio
        }
        
        return loss_dict, outputs

def load_pretrained_feddna(model, checkpoint_path, device):
    """✅ 修复：更智能的权重加载"""
    print(f"🔄 加载FedDNA预训练权重: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                pretrained_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint
        else:
            pretrained_dict = checkpoint
        
        model_dict = model.state_dict()
        
        # 智能加载权重
        filtered_dict = {}
        skipped_keys = []
        
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                    print(f"   ✅ 加载: {k} {v.shape}")
                else:
                    skipped_keys.append(f"{k} (形状不匹配: 模型{model_dict[k].shape} vs 权重{v.shape})")
            else:
                # ✅ 跳过length_adapter相关权重，因为我们会动态初始化
                if 'length_adapter' in k:
                    print(f"   🔧 跳过length_adapter权重（将动态初始化）: {k}")
                else:
                    skipped_keys.append(f"{k} (模型中不存在)")
        
        # 更新模型参数
        if filtered_dict:
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict, strict=False)  # 允许部分加载
            print(f"✅ 成功加载 {len(filtered_dict)}/{len(pretrained_dict)} 个参数")
        
        if skipped_keys:
            print(f"⚠️ 跳过的权重:")
            for key in skipped_keys[:5]:  # 只显示前5个
                print(f"     {key}")
            if len(skipped_keys) > 5:
                print(f"     ... 还有 {len(skipped_keys) - 5} 个")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("   继续使用随机初始化权重")
    
    return model
