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
        """
        ✅ 修复版：打破死锁的对比学习
        策略：
        1. Warm-up期：无条件信任Clover标签，强制学习特征。
        2. 后期：利用Evidence筛选高质量样本进行微调。
        """
        # ---------------------------------------------------
        # 1️⃣ Warm-up 阶段：强制学习 (Bootstrap)
        # ---------------------------------------------------
        if epoch < warmup_epochs:
            # 直接使用所有样本，不进行筛选
            # 这一步至关重要！没有它，模型永远无法启动。
            valid_mask = torch.ones_like(strength.mean(dim=1), dtype=torch.bool)
            
            # 使用简单的 mask (所有非噪声样本)
            labels_expanded = cluster_labels.unsqueeze(1)
            positive_mask = (labels_expanded == labels_expanded.T).float()
            
            # 自身mask
            logits_mask = torch.scatter(
                torch.ones_like(positive_mask),
                1,
                torch.arange(pooled_emb.size(0)).view(-1, 1).to(self.device),
                0
            )
            positive_mask = positive_mask * logits_mask
            
            # 计算相似度
            proj_emb = self.projection_head(pooled_emb)
            proj_emb = F.normalize(proj_emb, dim=-1)
            sim_matrix = torch.matmul(proj_emb, proj_emb.T) / temperature
            
            # InfoNCE Loss
            exp_sim = torch.exp(torch.clamp(sim_matrix, max=10))
            numerator = torch.sum(exp_sim * positive_mask, dim=1) + 1e-8
            denominator = torch.sum(exp_sim, dim=1) - torch.diag(exp_sim) + 1e-8
            
            loss = -torch.log(numerator / denominator)
            return loss.mean()

        # ---------------------------------------------------
        # 2️⃣ Refinement 阶段：Evidence 驱动的筛选
        # ---------------------------------------------------
        if pooled_emb.size(0) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        confidence = strength.mean(dim=1)
        
        # 动态阈值：取当前batch的前60%高置信度样本
        # 注意：这里加了 detach() 防止梯度回传影响阈值计算，虽然 quantile 本身不可导
        conf_threshold = torch.quantile(confidence.detach(), 0.4) # 保留60%
        conf_mask = confidence >= conf_threshold
        
        if conf_mask.sum() < 2:
            # 如果batch内都没信心，回退到使用所有样本，避免梯度消失
            conf_mask = torch.ones_like(confidence, dtype=torch.bool)
            
        proj_emb = self.projection_head(pooled_emb)
        proj_emb = F.normalize(proj_emb, dim=-1)
        sim_matrix = torch.matmul(proj_emb, proj_emb.T) / temperature
        
        # 构建 Mask: (同簇) AND (两者都是高置信度)
        labels_expanded = cluster_labels.unsqueeze(1)
        clover_positive_mask = (labels_expanded == labels_expanded.T).float()
        evidence_positive_mask = (conf_mask.unsqueeze(1) & conf_mask.unsqueeze(0)).float()
        
        positive_mask = clover_positive_mask * evidence_positive_mask
        
        # 自身对角线设为0
        logits_mask = torch.scatter(
            torch.ones_like(positive_mask),
            1,
            torch.arange(pooled_emb.size(0)).view(-1, 1).to(self.device),
            0
        )
        positive_mask = positive_mask * logits_mask
        
        exp_sim = torch.exp(torch.clamp(sim_matrix, max=10))
        numerator = torch.sum(exp_sim * positive_mask, dim=1) + 1e-8
        denominator = torch.sum(exp_sim, dim=1) - torch.diag(exp_sim) + 1e-8
        
        loss = -torch.log(numerator / denominator)
        
        # 只对参与计算的样本求平均
        valid_indices = torch.where(torch.sum(positive_mask, dim=1) > 0)[0]
        if len(valid_indices) > 0:
            return loss[valid_indices].mean()
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def self_reconstruction_loss(self, evidence, alpha, cluster_labels, inputs):  # <--- 新增 inputs
        """
        ✅ 修复版：加入 Input Reconstruction，防止 Mode Collapse
        """
        bayes_risk = CEBayesRiskLoss().to(self.device)
        kld_loss_fn = KLDivergenceLoss().to(self.device)

        # 1. 核心修复：Input Reconstruction Loss (AE Loss)
        # 让模型必须学会重建输入的 ATCG 序列
        # inputs 是 one-hot 编码，可以直接作为 target
        input_recon_loss = bayes_risk(evidence, inputs)

        # --------------------------------------------------------
        # 下面是你原来的 Consensus Loss (可以保留作为正则项，但要防止主导)
        # --------------------------------------------------------
        total_consensus_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_kl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        fused_consensus = {}

        unique_labels = torch.unique(cluster_labels)
        processed_clusters = 0

        for label in unique_labels:
            if label < 0: continue
            cluster_mask = (cluster_labels == label)
            if cluster_mask.sum() < 2: continue

            # 获取数据
            cluster_evidence = evidence[cluster_mask]
            
            # 计算 Consensus (Target)
            # 注意：这里 detach() 很重要，防止梯度流向 Target 导致"作弊"
            weights = F.softmax(torch.sum(alpha[cluster_mask], dim=-1).mean(dim=1), dim=0).view(-1, 1, 1)
            fused_evidence_val = torch.sum(cluster_evidence * weights, dim=0, keepdim=True).detach() 
            fused_consensus[label.item()] = fused_evidence_val

            # Soft Target for Consistency
            fused_alpha_val = fused_evidence_val + 1.0
            target_prob = (fused_alpha_val / fused_alpha_val.sum(dim=-1, keepdim=True)).expand(cluster_mask.sum(), -1, -1)
            
            # Hard Target for KL
            target_one_hot = F.one_hot(fused_evidence_val.argmax(dim=-1), num_classes=4).float().expand(cluster_mask.sum(), -1, -1)

            # 计算簇内一致性损失
            cons_loss = bayes_risk(cluster_evidence, target_prob)
            kl = kld_loss_fn(cluster_evidence, target_one_hot)

            total_consensus_loss = total_consensus_loss + cons_loss
            total_kl_loss = total_kl_loss + kl
            processed_clusters += 1

        if processed_clusters > 0:
            total_consensus_loss /= processed_clusters
            total_kl_loss /= processed_clusters

        # 2. 组合 Loss
        # 💡 建议权重：90% 重建输入 (确保不瞎猜), 10% 逼近簇中心 (促进聚类)
        final_recon_loss = 1.0 * input_recon_loss + 0.1 * total_consensus_loss

        return final_recon_loss, total_kl_loss, fused_consensus
    
    def forward(self, reads, cluster_labels, epoch=0):
        # 1️⃣ FedDNA Encoder
        embeddings, pooled_emb = self.encode_reads(reads)
        
        # 2️⃣ FedDNA Decoder
        evidence, strength, alpha = self.decode_to_evidence(embeddings)
        
        # 3️⃣ 对比学习 (保持 Warmup=5)
        contrastive_loss = self.contrastive_learning_with_evidence_filter(
            pooled_emb, cluster_labels, strength, epoch=epoch, warmup_epochs=5
        )
        
        # 4️⃣ 自重建损失
        recon_loss, kl_loss, fused_consensus = self.self_reconstruction_loss(
            evidence, alpha, cluster_labels, reads
        )
        
        # 5️⃣ 总损失策略调整 (🚨 核心修改点)
        
        # 策略 A: 推迟 KL 介入 (让模型先从 Recon/Contrastive 学到自信)
        # 从 Epoch 10 开始介入，到 Epoch 40 达到最大值
        if epoch < 10:
            annealing_coef = 0.0
        else:
            annealing_coef = min(1.0, (epoch - 10) / 30)
            
        # 策略 B: 永久性缩小 KL 权重 (因为 KL 原始数值 500+ 太大了，必须缩放)
        # 我们希望 KL Loss 最终在 10-50 左右，而不是 500
        scaled_kl_loss = kl_loss * 0.05  # 缩小 20 倍
        
        # 策略 C: 激励项 (防止 Strength 归零)
        # 如果是 Warm-up 阶段，给一点点奖励让它产生 Strength
        if epoch < 10:
            l2_evidence = torch.mean(evidence ** 2) * 0.001
        else:
            l2_evidence = 0.0

        # 总损失
        total_loss = contrastive_loss + 10.0 * recon_loss + annealing_coef * scaled_kl_loss + l2_evidence
        
        # 统计信息
        avg_strength = strength.mean().item()
        high_conf_ratio = (strength.mean(dim=1) > 4.5).float().mean().item()
        
        loss_dict = {
            'total': total_loss,
            'contrastive': contrastive_loss,
            'reconstruction': recon_loss,
            'kl_divergence': kl_loss, # 记录原始值以便观察
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