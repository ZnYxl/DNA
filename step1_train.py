import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

# ==========================================
# 定义权重参数
# ==========================================
alpha = 1.0
beta = 0.01
gamma = 0.01

# ==========================================
# 导入基础组件
# ==========================================
try:
    from models.conmamba import ConmambaBlock
    print("✅ 成功导入 ConmambaBlock")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ==========================================
# 简化的核心组件
# ==========================================

class SimpleEncoder(nn.Module):
    """简化的编码器 - 只保留核心功能"""
    def __init__(self, input_dim=4, hidden_dim=64, seq_len=150):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 简单的特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # ConMamba块用于序列建模
        self.conmamba = ConmambaBlock(dim=hidden_dim)
        
    def forward(self, x):
        # x: [B*N, L, 4] 
        x = self.feature_extractor(x)  # [B*N, L, hidden_dim]
        x = self.conmamba(x)           # [B*N, L, hidden_dim]
        return x

class ContrastiveLearning(nn.Module):
    """对比学习模块"""
    def __init__(self, hidden_dim=64, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, embeddings):
        # embeddings: [B*N, L, hidden_dim]
        # 取平均池化作为序列表示
        seq_repr = torch.mean(embeddings, dim=1)  # [B*N, hidden_dim]
        projected = self.projection(seq_repr)     # [B*N, hidden_dim//2]
        return F.normalize(projected, dim=-1)

class EvidenceDecoder(nn.Module):
    """证据解码器 - 输出Evidence向量"""
    def __init__(self, hidden_dim=64, output_dim=4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # 确保输出为正值 (Evidence)
        )
        
    def forward(self, x):
        # x: [B*N, L, hidden_dim]
        evidence = self.decoder(x)  # [B*N, L, 4]
        return evidence

class EvidenceFusion(nn.Module):
    """证据融合模块"""
    def __init__(self):
        super().__init__()
        
    def calculate_strength(self, evidence):
        """计算证据强度作为融合权重"""
        # evidence: [N, L, 4]
        strength = torch.sum(evidence, dim=-1, keepdim=True)  # [N, L, 1]
        return strength
    
    def forward(self, evidence_batch):
        """
        evidence_batch: [N, L, 4] - N条reads的evidence
        """
        # 计算每条read的证据强度
        strengths = self.calculate_strength(evidence_batch)  # [N, L, 1]
        
        # 加权融合
        weighted_evidence = evidence_batch * strengths       # [N, L, 4]
        fused_evidence = torch.sum(weighted_evidence, dim=0) # [L, 4]
        total_weight = torch.sum(strengths, dim=0)           # [L, 1]
        
        # 避免除零
        fused_evidence = fused_evidence / (total_weight + 1e-8)
        
        return fused_evidence, strengths

class SimplifiedFedDNA(nn.Module):
    """简化版FedDNA - 只保留核心组件"""
    def __init__(self, input_dim=4, hidden_dim=64, seq_len=150):
        super().__init__()
        self.encoder = SimpleEncoder(input_dim, hidden_dim, seq_len)
        self.contrastive = ContrastiveLearning(hidden_dim)
        self.evidence_decoder = EvidenceDecoder(hidden_dim, input_dim)
        self.evidence_fusion = EvidenceFusion()
        
    def forward(self, reads_batch):
        """
        reads_batch: [B, N, L, 4] - B个batch，每个batch有N条reads
        """
        B, N, L, D = reads_batch.shape
        
        # 重塑为 [B*N, L, D]
        reads_flat = reads_batch.view(B * N, L, D)
        
        # 1. 编码
        embeddings = self.encoder(reads_flat)  # [B*N, L, hidden_dim]
        
        # 2. 对比学习特征
        contrastive_features = self.contrastive(embeddings)  # [B*N, hidden_dim//2]
        
        # 3. 证据解码
        evidence = self.evidence_decoder(embeddings)  # [B*N, L, 4]
        
        # 4. 重塑回batch形式
        evidence = evidence.view(B, N, L, D)  # [B, N, L, 4]
        contrastive_features = contrastive_features.view(B, N, -1)  # [B, N, hidden_dim//2]
        
        return evidence, contrastive_features

# ==========================================
# 综合损失函数
# ==========================================

class ComprehensiveLoss(nn.Module):
    """综合损失函数 - 包含重构、对比学习、KL散度"""
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01, temperature=0.1):
        super().__init__()
        self.alpha = alpha      # 重构损失权重
        self.beta = beta        # 对比学习损失权重  
        self.gamma = gamma      # KL散度损失权重
        self.temperature = temperature
        
    def contrastive_loss(self, features, cluster_labels=None):
        """
        对比学习损失 - 同簇内的reads应该相似，不同簇应该不同
        features: [N, feature_dim] - N条reads的对比学习特征
        """
        if features.shape[0] <= 1:
            return torch.tensor(0.0, device=features.device)
            
        # 简化版：假设同一个batch内的reads属于同一簇
        # 计算相似度矩阵
        features_norm = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        # 创建正样本mask (同一batch内为正样本)
        batch_size = features.shape[0]
        mask = torch.eye(batch_size, device=features.device).bool()
        
        # 计算InfoNCE损失
        exp_sim = torch.exp(similarity_matrix)
        exp_sim = exp_sim.masked_fill(mask, 0)  # 移除自己和自己的相似度
        
        # 正样本：同batch内其他样本的平均
        pos_sim = torch.sum(exp_sim, dim=1) / (batch_size - 1)
        # 负样本：这里简化处理，使用所有样本
        neg_sim = torch.sum(exp_sim, dim=1)
        
        # InfoNCE损失
        loss = -torch.log(pos_sim / (neg_sim + 1e-8))
        return torch.mean(loss)
    
    def kl_divergence_loss(self, evidence):
        """
        KL散度损失 - 衡量证据分布的不确定性
        evidence: [L, 4] - 融合后的证据向量
        """
        # 将evidence转换为Dirichlet分布参数
        alpha = evidence + 1  # [L, 4]
        
        # 计算与均匀分布的KL散度 (简化版本)
        # 使用证据的方差作为不确定性度量
        evidence_normalized = F.softmax(evidence, dim=-1)
        uniform_dist = torch.ones_like(evidence_normalized) / evidence_normalized.shape[-1]
        
        # KL散度: KL(P||Q) = sum(P * log(P/Q))
        kl_div = torch.sum(evidence_normalized * torch.log(evidence_normalized / (uniform_dist + 1e-8) + 1e-8), dim=-1)
        
        return torch.mean(kl_div)
    
    def forward(self, fused_evidence, ref, contrastive_features):
        """
        计算总损失
        fused_evidence: [L, 4] - 融合后的证据
        ref: [L, 4] - 参考序列
        contrastive_features: [N, feature_dim] - 对比学习特征
        """
        # 1. 重构损失
        reconstruction_loss = F.mse_loss(fused_evidence, ref)
        
        # 2. 对比学习损失
        contrastive_loss = self.contrastive_loss(contrastive_features)
        
        # 3. KL散度损失
        kl_loss = self.kl_divergence_loss(fused_evidence)
        
        # 4. 总损失
        total_loss = (self.alpha * reconstruction_loss + 
                     self.beta * contrastive_loss + 
                     self.gamma * kl_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'contrastive_loss': contrastive_loss,
            'kl_loss': kl_loss
        }

# ==========================================
# 数据集 (保持不变)
# ==========================================
class CloverClusterDataset(Dataset):
    def __init__(self, data_dir, seq_len=150):
        self.seq_len = seq_len
        self.clusters = [] 
        
        read_path = os.path.join(data_dir, "read.txt")
        ref_path = os.path.join(data_dir, "ref.txt")
        if not os.path.exists(ref_path):
            ref_path = os.path.join(data_dir, "reference.txt")
            
        if not os.path.exists(read_path):
            print(f"❌ 错误: 找不到数据文件 {read_path}")
            return
        
        print(f"📂 正在加载数据: {data_dir}")
        with open(ref_path, 'r') as f:
            refs = [line.strip() for line in f if line.strip()]
        with open(read_path, 'r') as f:
            content = f.read().strip()
        raw_clusters = content.split("===============================")
        
        for i, cluster_block in enumerate(raw_clusters):
            if not cluster_block.strip(): continue
            if i >= len(refs): break
            reads = [r.strip() for r in cluster_block.strip().split('\n') if r.strip()]
            if len(reads) > 0:
                self.clusters.append({'ref': refs[i], 'reads': reads})
                
        print(f"✅ 加载完成: {len(self.clusters)} 个簇")

    def one_hot(self, seq):
        mapping = {'A':0, 'C':1, 'G':2, 'T':3}
        arr = np.zeros((self.seq_len, 4), dtype=np.float32)
        l = min(len(seq), self.seq_len)
        for i in range(l):
            char = seq[i]
            if char in mapping: arr[i, mapping[char]] = 1.0
        return arr

    def __len__(self): return len(self.clusters)

    def __getitem__(self, idx):
        cluster = self.clusters[idx]
        reads_vec = np.array([self.one_hot(r) for r in cluster['reads']])
        ref_vec = self.one_hot(cluster['ref'])
        return torch.tensor(reads_vec), torch.tensor(ref_vec)

# ==========================================
# 步骤2: 困难样本修正模块
# ==========================================

class EvidenceRefinement(nn.Module):
    """证据驱动的困难样本修正模块"""
    
    def __init__(self, confidence_threshold=0.5, distance_threshold=2.0):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.distance_threshold = distance_threshold
        
    def calculate_confidence_score(self, evidence_strengths):
        """
        计算证据置信度分数
        evidence_strengths: [N, L, 1] - N条reads的证据强度
        返回: [N] - 每条read的置信度分数
        """
        # 方法1: 使用证据强度的平均值作为置信度
        confidence_scores = torch.mean(evidence_strengths.squeeze(-1), dim=1)  # [N]
        
        return confidence_scores
    
    def compute_cluster_centers(self, embeddings, labels, num_clusters):
        """
        计算簇中心
        embeddings: [N, feature_dim] - reads的嵌入表示
        labels: [N] - 当前标签
        num_clusters: int - 簇的数量
        返回: [K, feature_dim] - K个簇中心
        """
        device = embeddings.device
        feature_dim = embeddings.shape[1]
        centers = torch.zeros(num_clusters, feature_dim, device=device)
        
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                centers[k] = torch.mean(embeddings[mask], dim=0)
            else:
                # 如果某个簇为空，使用随机初始化
                centers[k] = torch.randn(feature_dim, device=device)
                
        return centers
    
    def reassign_hard_samples(self, hard_embeddings, cluster_centers):
        """
        重分配困难样本
        hard_embeddings: [M, feature_dim] - 困难样本的嵌入
        cluster_centers: [K, feature_dim] - 簇中心
        返回: [M] - 新的标签分配 (-1表示噪声)
        """
        if hard_embeddings.shape[0] == 0:
            return torch.tensor([], dtype=torch.long, device=hard_embeddings.device)
            
        # 计算到所有簇中心的距离
        distances = torch.cdist(hard_embeddings, cluster_centers)  # [M, K]
        
        # 找到最近的簇
        min_distances, nearest_clusters = torch.min(distances, dim=1)  # [M]
        
        # 判断是否为噪声（距离所有簇都太远）
        new_labels = nearest_clusters.clone()
        noise_mask = min_distances > self.distance_threshold
        new_labels[noise_mask] = -1  # 标记为噪声
        
        return new_labels, min_distances
    
    def forward(self, embeddings, evidence_strengths, current_labels, num_clusters):
        """
        执行完整的修正流程
        
        参数:
        - embeddings: [N, feature_dim] - reads的嵌入表示
        - evidence_strengths: [N, L, 1] - 证据强度
        - current_labels: [N] - 当前标签
        - num_clusters: int - 簇数量
        
        返回:
        - new_labels: [N] - 修正后的标签
        - refinement_stats: dict - 修正统计信息
        """
        N = embeddings.shape[0]
        device = embeddings.device
        
        # 1. 计算置信度分数
        confidence_scores = self.calculate_confidence_score(evidence_strengths)
        
        # 2. 阈值判断 - 识别困难样本
        high_confidence_mask = confidence_scores > self.confidence_threshold
        hard_sample_mask = ~high_confidence_mask
        
        # 3. 保留高置信度样本的标签
        new_labels = current_labels.clone()
        
        # 4. 处理困难样本
        if hard_sample_mask.sum() > 0:
            # 计算当前簇中心（基于高置信度样本）
            high_conf_embeddings = embeddings[high_confidence_mask]
            high_conf_labels = current_labels[high_confidence_mask]
            
            if high_conf_embeddings.shape[0] > 0:
                cluster_centers = self.compute_cluster_centers(
                    high_conf_embeddings, high_conf_labels, num_clusters
                )
                
                # 重分配困难样本
                hard_embeddings = embeddings[hard_sample_mask]
                reassigned_labels, distances = self.reassign_hard_samples(
                    hard_embeddings, cluster_centers
                )
                
                # 更新困难样本的标签
                new_labels[hard_sample_mask] = reassigned_labels
        
        # 5. 统计修正信息
        refinement_stats = {
            'total_samples': N,
            'high_confidence_count': high_confidence_mask.sum().item(),
            'hard_samples_count': hard_sample_mask.sum().item(),
            'noise_samples_count': (new_labels == -1).sum().item(),
            'label_changes': (new_labels != current_labels).sum().item(),
            'refinement_ratio': (new_labels != current_labels).float().mean().item(),
            'avg_confidence': confidence_scores.mean().item(),
            'min_confidence': confidence_scores.min().item(),
            'max_confidence': confidence_scores.max().item()
        }
        
        return new_labels, refinement_stats

# ==========================================
# 扩展的训练循环 - 包含修正阶段
# ==========================================

class RefinementTrainer:
    """包含修正阶段的训练器"""
    
    def __init__(self, model, criterion, optimizer, refinement_module, 
                 convergence_threshold=0.01, max_epochs=10):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.refinement = refinement_module
        self.convergence_threshold = convergence_threshold
        self.max_epochs = max_epochs
        
    def train_epoch_with_refinement(self, dataloader, device, epoch):
        """训练一个epoch，包含修正阶段"""
        
        self.model.train()
        epoch_losses = {'total_loss': 0, 'reconstruction_loss': 0, 'contrastive_loss': 0, 'kl_loss': 0}
        all_refinement_stats = []
        step_count = 0
        
        print(f"\n🔄 Epoch {epoch+1} - 开始训练+修正阶段...")
        
        for i, (reads, ref) in enumerate(dataloader):
            reads = reads.to(device)  # [1, N, 150, 4]
            ref = ref.squeeze(0).to(device)  # [150, 4]
            N = reads.shape[1]
            
            # === 步骤1: 正常训练 ===
            self.optimizer.zero_grad()
            
            # Forward pass
            evidence_batch, contrastive_features = self.model(reads)
            
            # 证据融合
            evidence_single_batch = evidence_batch.squeeze(0)  # [N, 150, 4]
            fused_evidence, strengths = self.model.evidence_fusion(evidence_single_batch)
            
            # 计算损失
            contrastive_features_flat = contrastive_features.squeeze(0)  # [N, feature_dim]
            losses = self.criterion(fused_evidence, ref, contrastive_features_flat)
            
            # 反向传播
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # === 步骤2: 困难样本修正 ===
            with torch.no_grad():
                # 假设初始标签都是0（同一个簇）
                current_labels = torch.zeros(N, dtype=torch.long, device=device)
                
                # 执行修正
                new_labels, refinement_stats = self.refinement(
                    embeddings=contrastive_features_flat,
                    evidence_strengths=strengths.unsqueeze(-1),  # [N, L] -> [N, L, 1]
                    current_labels=current_labels,
                    num_clusters=1  # 简化：假设每个batch是一个簇
                )
                
                all_refinement_stats.append(refinement_stats)
            
            # 记录损失
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            step_count += 1
            
            # 打印进度
            if (i + 1) % 5 == 0:
                print(f"  📊 Step {i+1:3d} | Loss: {losses['total_loss'].item():.4f} | "
                      f"修正率: {refinement_stats['refinement_ratio']:.3f} | "
                      f"置信度: {refinement_stats['avg_confidence']:.3f} | "
                      f"噪声: {refinement_stats['noise_samples_count']}")
        
        # 计算epoch统计
        avg_losses = {key: val / max(1, step_count) for key, val in epoch_losses.items()}
        
        # 汇总修正统计
        if all_refinement_stats:
            avg_refinement_ratio = np.mean([s['refinement_ratio'] for s in all_refinement_stats])
            avg_confidence = np.mean([s['avg_confidence'] for s in all_refinement_stats])
            total_noise = sum([s['noise_samples_count'] for s in all_refinement_stats])
        else:
            avg_refinement_ratio = 0.0
            avg_confidence = 0.0
            total_noise = 0
            
        return avg_losses, {
            'refinement_ratio': avg_refinement_ratio,
            'avg_confidence': avg_confidence,
            'total_noise_samples': total_noise
        }
    
    def train_with_refinement(self, dataloader, device):
        """完整的训练流程，包含收敛判断"""
        
        print("🚀 开始证据驱动的修正训练...")
        print(f"📋 配置: 收敛阈值={self.convergence_threshold}, 最大轮数={self.max_epochs}")
        
        training_history = {
            'losses': [],
            'refinement_ratios': [],
            'confidences': [],
            'noise_counts': []
        }
        
        for epoch in range(self.max_epochs):
            # 训练一个epoch
            avg_losses, refinement_stats = self.train_epoch_with_refinement(
                dataloader, device, epoch
            )
            
            # 记录历史
            training_history['losses'].append(avg_losses)
            training_history['refinement_ratios'].append(refinement_stats['refinement_ratio'])
            training_history['confidences'].append(refinement_stats['avg_confidence'])
            training_history['noise_counts'].append(refinement_stats['total_noise_samples'])
            
            # 打印epoch总结
            print(f"\n📈 Epoch {epoch+1} 完成:")
            print(f"   总损失: {avg_losses['total_loss']:.6f}")
            print(f"   修正比例: {refinement_stats['refinement_ratio']:.4f} ({refinement_stats['refinement_ratio']*100:.2f}%)")
            print(f"   平均置信度: {refinement_stats['avg_confidence']:.4f}")
            print(f"   噪声样本数: {refinement_stats['total_noise_samples']}")
            
            # 收敛判断
            if refinement_stats['refinement_ratio'] < self.convergence_threshold:
                print(f"\n✅ 收敛达成！修正比例 {refinement_stats['refinement_ratio']:.4f} < 阈值 {self.convergence_threshold}")
                print(f"🎯 训练在第 {epoch+1} 轮收敛")
                break
            else:
                print(f"   🔄 继续训练 (修正比例 {refinement_stats['refinement_ratio']:.4f} >= {self.convergence_threshold})")
            
            print("-" * 70)
        
        return training_history

# ==========================================
# 更新后的主训练函数 - 包含修正阶段
# ==========================================

def train_with_refinement():
    """包含修正阶段的完整训练函数"""
    
    DATA_DIR = "Dataset/CloverExp/train"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 训练参数
    input_dim = 4
    hidden_dim = 64
    seq_len = 150
    lr = 1e-3
    
    # 损失权重
    alpha = 1.0
    beta = 0.01  # 降低对比学习权重
    gamma = 0.01
    
    # 修正参数
    confidence_threshold = 0.3  # 置信度阈值
    distance_threshold = 1.5    # 距离阈值
    convergence_threshold = 0.01  # 收敛阈值 (1%)
    max_epochs = 8
    
    try:
        # 检查数据目录
        if not os.path.exists(DATA_DIR):
            print(f"❌ 目录不存在: {DATA_DIR}")
            return None
            
        # 加载数据
        dataset = CloverClusterDataset(DATA_DIR)
        if len(dataset) == 0:
            print(f"❌ 数据集为空，请检查数据文件")
            return None
            
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # 初始化模型
        model = SimplifiedFedDNA(input_dim, hidden_dim, seq_len).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = ComprehensiveLoss(alpha=alpha, beta=beta, gamma=gamma)
        
        # 初始化修正模块
        refinement_module = EvidenceRefinement(
            confidence_threshold=confidence_threshold,
            distance_threshold=distance_threshold
        ).to(DEVICE)
        
        # 创建训练器
        trainer = RefinementTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            refinement_module=refinement_module,
            convergence_threshold=convergence_threshold,
            max_epochs=max_epochs
        )
        
        print(f"🔧 模型配置:")
        print(f"   设备: {DEVICE}")
        print(f"   数据集大小: {len(dataset)} 个簇")
        print(f"   损失权重: 重构={alpha}, 对比学习={beta}, KL散度={gamma}")
        print(f"   修正参数: 置信度阈值={confidence_threshold}, 距离阈值={distance_threshold}")
        print(f"   收敛条件: 修正比例 < {convergence_threshold*100}%")
        
        # 开始训练
        training_history = trainer.train_with_refinement(dataloader, DEVICE)
        
        # 保存结果
        save_dict = {
            'model_state_dict': model.state_dict(),
            'refinement_state_dict': refinement_module.state_dict(),
            'training_history': training_history,
            'config': {
                'model_config': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'seq_len': seq_len,
                },
                'training_config': {
                    'learning_rate': lr,
                    'loss_weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
                    'max_epochs': max_epochs
                },
                'refinement_config': {
                    'confidence_threshold': confidence_threshold,
                    'distance_threshold': distance_threshold,
                    'convergence_threshold': convergence_threshold
                }
            }
        }
        
        torch.save(save_dict, "refined_model.pth")
        print(f"\n💾 完整模型已保存到: refined_model.pth")
        
        # 训练总结
        final_refinement_ratio = training_history['refinement_ratios'][-1]
        final_confidence = training_history['confidences'][-1]
        
        print(f"\n🎯 训练完成总结:")
        print(f"   最终修正比例: {final_refinement_ratio:.4f} ({final_refinement_ratio*100:.2f}%)")
        print(f"   最终平均置信度: {final_confidence:.4f}")
        print(f"   总训练轮数: {len(training_history['losses'])}")
        
        if final_refinement_ratio < convergence_threshold:
            print(f"   ✅ 成功收敛！")
        else:
            print(f"   ⚠️  未完全收敛，可考虑增加训练轮数")
            
        return training_history
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==========================================
# 原始训练函数 (保留作为备用)
# ==========================================
def train():
    """原始的训练函数 - 仅包含Step1"""
    DATA_DIR = "Dataset/CloverExp/train"
    EPOCHS = 5
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 参数配置
    input_dim = 4
    hidden_dim = 64
    seq_len = 150
    
    try:
        if not os.path.exists(DATA_DIR):
            print(f"❌ 目录不存在: {DATA_DIR}")
            return

        dataset = CloverClusterDataset(DATA_DIR)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        print(f"🔧 初始化简化版FedDNA模型...")
        model = SimplifiedFedDNA(input_dim, hidden_dim, seq_len).to(DEVICE)
        
        print("🎉 模型初始化成功！")

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 使用综合损失函数
    criterion = ComprehensiveLoss(alpha=alpha, beta=beta, gamma=gamma)

    print(f"\n🚀 开始训练 | Device: {DEVICE} | Epochs: {EPOCHS}")
    print(f"📊 损失函数权重: 重构={alpha}, 对比学习={beta}, KL散度={gamma}")
    
    # 记录训练历史
    train_history = {
        'total_loss': [],
        'reconstruction_loss': [],
        'contrastive_loss': [],
        'kl_loss': []
    }
    
    model.train()
    for epoch in range(EPOCHS):
        epoch_losses = {'total_loss': 0, 'reconstruction_loss': 0, 'contrastive_loss': 0, 'kl_loss': 0}
        step_count = 0
        
        for i, (reads, ref) in enumerate(dataloader):
            reads = reads.to(DEVICE)  # [1, N, 150, 4]
            ref = ref.squeeze(0).to(DEVICE)  # [150, 4]
            
            optimizer.zero_grad()
            
            # Forward pass
            evidence_batch, contrastive_features = model(reads)
            
            # 证据融合
            evidence_single_batch = evidence_batch.squeeze(0)  # [N, 150, 4]
            fused_evidence, strengths = model.evidence_fusion(evidence_single_batch)
            
            # 计算综合损失
            contrastive_features_flat = contrastive_features.squeeze(0)  # [N, feature_dim]
            losses = criterion(fused_evidence, ref, contrastive_features_flat)
            
            # 反向传播
            losses['total_loss'].backward()
            optimizer.step()
            
            # 记录损失
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            step_count += 1
            
            # 打印详细损失信息
            if (i + 1) % 10 == 0:
                print(f"  Step {i+1:3d} | Total: {losses['total_loss'].item():.6f} | "
                      f"Recon: {losses['reconstruction_loss'].item():.6f} | "
                      f"Contra: {losses['contrastive_loss'].item():.6f} | "
                      f"KL: {losses['kl_loss'].item():.6f}")
            
        # 计算平均损失
        for key in epoch_losses:
            avg_loss = epoch_losses[key] / max(1, step_count)
            train_history[key].append(avg_loss)
            
        print(f"\n📈 Epoch {epoch+1}/{EPOCHS} 完成:")
        print(f"   Total Loss:    {train_history['total_loss'][-1]:.6f}")
        print(f"   Reconstruction: {train_history['reconstruction_loss'][-1]:.6f}")
        print(f"   Contrastive:   {train_history['contrastive_loss'][-1]:.6f}")
        print(f"   KL Divergence: {train_history['kl_loss'][-1]:.6f}")
        print("-" * 60)
        
    print("\n✅ 训练完成！保存模型和训练历史...")
    
    # 保存完整的训练结果
    save_dict = {
        'model_state_dict': model.state_dict(),
        'train_history': train_history,
        'config': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'seq_len': seq_len,
            'epochs': EPOCHS,
            'learning_rate': LR,
            'loss_weights': {
                'alpha': alpha,
                'beta': beta, 
                'gamma': gamma
            }
        }
    }
    
    torch.save(save_dict, "comprehensive_model.pth")
    print("💾 模型已保存到: comprehensive_model.pth")
    
    # 打印训练总结
    print(f"\n📊 训练总结:")
    print(f"   最终总损失: {train_history['total_loss'][-1]:.6f}")
    print(f"   损失下降: {train_history['total_loss'][0]:.6f} → {train_history['total_loss'][-1]:.6f}")
    print(f"   改善幅度: {((train_history['total_loss'][0] - train_history['total_loss'][-1]) / train_history['total_loss'][0] * 100):.2f}%")
    
    return train_history

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    print("🎯 选择训练模式:")
    print("1. 完整训练 (Step1 + Step2 困难样本修正)")
    print("2. 基础训练 (仅Step1)")
    
    # 默认使用完整训练
    print("🚀 启动完整训练模式 (包含困难样本修正)...")
    train_history = train_with_refinement()
    
    # 如果需要使用基础训练，取消下面的注释
    # train_history = train()
