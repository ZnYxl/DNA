import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import json
import pandas as pd
from datetime import datetime
import logging
import shutil

# ==========================================
# 输出管理系统
# ==========================================

class ExperimentManager:
    """实验输出管理器"""
    
    def __init__(self, experiment_name="DNA_Clustering", base_dir="outputs"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 创建实验文件夹
        self.exp_dir = os.path.join(base_dir, f"{self.timestamp}_{experiment_name}")
        self.create_directories()
        
        # 设置日志
        self.setup_logging()
        
        print(f"📁 实验目录已创建: {self.exp_dir}")
        
    def create_directories(self):
        """创建所有必要的目录"""
        directories = [
            self.exp_dir,
            os.path.join(self.exp_dir, "model"),
            os.path.join(self.exp_dir, "visualizations"), 
            os.path.join(self.exp_dir, "results"),
            os.path.join(self.exp_dir, "logs")
        ]
        
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
            
    def setup_logging(self):
        """设置日志系统"""
        log_file = os.path.join(self.exp_dir, "logs", "training.log")
        
        # 创建logger
        self.logger = logging.getLogger('experiment')
        self.logger.setLevel(logging.INFO)
        
        # 清除已有的handlers
        self.logger.handlers.clear()
        
        # 文件handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def log(self, message, level="info"):
        """记录日志"""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        print(message)  # 同时打印到控制台
        
    def save_config(self, config):
        """保存实验配置"""
        config_file = os.path.join(self.exp_dir, "config.json")
        
        # 添加实验元信息
        full_config = {
            "experiment_info": {
                "name": self.experiment_name,
                "timestamp": self.timestamp,
                "directory": self.exp_dir
            },
            "model_config": config.get("model_config", {}),
            "training_config": config.get("training_config", {}),
            "refinement_config": config.get("refinement_config", {}),
            "data_config": config.get("data_config", {})
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=2, ensure_ascii=False)
            
        self.log(f"💾 配置已保存: {config_file}")
        
    def save_model(self, model_dict, filename="refined_model.pth"):
        """保存模型"""
        model_path = os.path.join(self.exp_dir, "model", filename)
        torch.save(model_dict, model_path)
        self.log(f"🤖 模型已保存: {model_path}")
        
        # 保存模型摘要
        summary_path = os.path.join(self.exp_dir, "model", "model_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"模型保存时间: {datetime.now()}\n")
            f.write(f"模型文件: {filename}\n")
            f.write(f"模型大小: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB\n")
            if 'config' in model_dict:
                f.write(f"模型配置: {model_dict['config']}\n")
                
    def save_visualization(self, fig, filename, title=""):
        """保存可视化图片"""
        viz_path = os.path.join(self.exp_dir, "visualizations", filename)
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # 关闭图片释放内存
        self.log(f"📊 图片已保存: {viz_path} - {title}")
        
    def save_metrics(self, metrics, filename="metrics_summary.json"):
        """保存评估指标"""
        metrics_path = os.path.join(self.exp_dir, "results", filename)
        
        # 确保所有值都是可序列化的
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
                
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
            
        self.log(f"📈 指标已保存: {metrics_path}")
        
    def save_cluster_assignments(self, true_labels, predicted_labels, confidences):
        """保存聚类分配结果"""
        results_df = pd.DataFrame({
            'sample_id': range(len(true_labels)),
            'true_cluster': true_labels,
            'predicted_cluster': predicted_labels,
            'confidence': confidences
        })
        
        csv_path = os.path.join(self.exp_dir, "results", "cluster_assignments.csv")
        results_df.to_csv(csv_path, index=False)
        self.log(f"📋 聚类结果已保存: {csv_path}")
        
    def generate_report(self, training_history, analysis_results):
        """生成实验报告"""
        report_path = os.path.join(self.exp_dir, "results", "analysis_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"实验报告: {self.experiment_name}\n")
            f.write(f"时间: {self.timestamp}\n")
            f.write("="*60 + "\n\n")
            
            # 训练总结
            f.write("🚀 训练总结:\n")
            f.write(f"   训练轮数: {len(training_history['losses'])}\n")
            if training_history['losses']:
                f.write(f"   最终损失: {training_history['losses'][-1]['total_loss']:.6f}\n")
                f.write(f"   最终修正比例: {training_history['refinement_ratios'][-1]:.4f}\n")
                f.write(f"   最终置信度: {training_history['confidences'][-1]:.4f}\n")
            f.write("\n")
            
            # 聚类评估
            if analysis_results and 'metrics' in analysis_results:
                metrics = analysis_results['metrics']
                f.write("📊 聚类评估:\n")
                f.write(f"   ARI: {metrics.get('ARI', 0):.4f}\n")
                f.write(f"   NMI: {metrics.get('NMI', 0):.4f}\n")
                f.write(f"   Silhouette: {metrics.get('Silhouette', 0):.4f}\n")
                f.write(f"   真实簇数: {metrics.get('True_Clusters', 0)}\n")
                f.write(f"   预测簇数: {metrics.get('Predicted_Clusters', 0)}\n")
                f.write("\n")
            
            # 置信度统计
            if analysis_results and 'confidences' in analysis_results:
                confidences = analysis_results['confidences']
                f.write("🎲 置信度统计:\n")
                f.write(f"   平均置信度: {np.mean(confidences):.4f}\n")
                f.write(f"   置信度标准差: {np.std(confidences):.4f}\n")
                f.write(f"   置信度范围: [{np.min(confidences):.4f}, {np.max(confidences):.4f}]\n")
                f.write("\n")
            
            f.write("📁 输出文件:\n")
            f.write(f"   模型文件: model/refined_model.pth\n")
            f.write(f"   配置文件: config.json\n")
            f.write(f"   可视化: visualizations/\n")
            f.write(f"   详细结果: results/\n")
            
        self.log(f"📄 实验报告已生成: {report_path}")
        
    def create_readme(self, description=""):
        """创建README文件"""
        readme_path = os.path.join(self.exp_dir, "README.md")
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.experiment_name}\n\n")
            f.write(f"**实验时间:** {self.timestamp}\n\n")
            f.write(f"**描述:** {description}\n\n")
            f.write("## 文件结构\n\n")
            f.write("```\n")
            f.write("├── config.json                 # 实验配置\n")
            f.write("├── model/                      # 模型文件\n")
            f.write("│   ├── refined_model.pth      # 训练好的模型\n")
            f.write("│   └── model_summary.txt      # 模型摘要\n")
            f.write("├── visualizations/            # 可视化结果\n")
            f.write("│   ├── clustering_analysis.png\n")
            f.write("│   ├── confidence_distribution.png\n")
            f.write("│   └── training_curves.png\n")
            f.write("├── results/                   # 分析结果\n")
            f.write("│   ├── metrics_summary.json\n")
            f.write("│   ├── cluster_assignments.csv\n")
            f.write("│   └── analysis_report.txt\n")
            f.write("├── logs/                      # 日志文件\n")
            f.write("│   └── training.log\n")
            f.write("└── README.md                  # 本文件\n")
            f.write("```\n\n")
            f.write("## 快速查看结果\n\n")
            f.write("1. 查看训练日志: `logs/training.log`\n")
            f.write("2. 查看聚类效果: `visualizations/clustering_analysis.png`\n")
            f.write("3. 查看详细报告: `results/analysis_report.txt`\n")
            f.write("4. 加载模型: `torch.load('model/refined_model.pth')`\n")
            
        self.log(f"📖 README已创建: {readme_path}")

# ==========================================
# 原有的所有类保持不变 (SimpleEncoder, ContrastiveLearning等)
# ==========================================

# 定义权重参数
alpha = 1.0
beta = 0.01
gamma = 0.01

# 导入基础组件
try:
    from models.conmamba import ConmambaBlock
    print("✅ 成功导入 ConmambaBlock")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# [这里插入之前的所有类定义，保持不变]
# SimpleEncoder, ContrastiveLearning, EvidenceDecoder, EvidenceFusion, 
# SimplifiedFedDNA, ComprehensiveLoss, CloverClusterDataset, 
# EvidenceRefinement, RefinementTrainer

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

class RefinementTrainer:
    """包含修正阶段的训练器"""
    
    def __init__(self, model, criterion, optimizer, refinement_module, 
                 convergence_threshold=0.01, max_epochs=10, exp_manager=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.refinement = refinement_module
        self.convergence_threshold = convergence_threshold
        self.max_epochs = max_epochs
        self.exp_manager = exp_manager  # 添加实验管理器
        
    def train_epoch_with_refinement(self, dataloader, device, epoch):
        """训练一个epoch，包含修正阶段"""
        
        self.model.train()
        epoch_losses = {'total_loss': 0, 'reconstruction_loss': 0, 'contrastive_loss': 0, 'kl_loss': 0}
        all_refinement_stats = []
        step_count = 0
        
        if self.exp_manager:
            self.exp_manager.log(f"🔄 Epoch {epoch+1} - 开始训练+修正阶段...")
        
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
                msg = (f"  📊 Step {i+1:3d} | Loss: {losses['total_loss'].item():.4f} | "
                      f"修正率: {refinement_stats['refinement_ratio']:.3f} | "
                      f"置信度: {refinement_stats['avg_confidence']:.3f} | "
                      f"噪声: {refinement_stats['noise_samples_count']}")
                if self.exp_manager:
                    self.exp_manager.log(msg)
                else:
                    print(msg)
        
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
        
        if self.exp_manager:
            self.exp_manager.log("🚀 开始证据驱动的修正训练...")
            self.exp_manager.log(f"📋 配置: 收敛阈值={self.convergence_threshold}, 最大轮数={self.max_epochs}")
        
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
            summary = (f"📈 Epoch {epoch+1} 完成:\n"
                      f"   总损失: {avg_losses['total_loss']:.6f}\n"
                      f"   修正比例: {refinement_stats['refinement_ratio']:.4f} ({refinement_stats['refinement_ratio']*100:.2f}%)\n"
                      f"   平均置信度: {refinement_stats['avg_confidence']:.4f}\n"
                      f"   噪声样本数: {refinement_stats['total_noise_samples']}")
            
            if self.exp_manager:
                self.exp_manager.log(summary)
            else:
                print(summary)
            
            # 收敛判断
            if refinement_stats['refinement_ratio'] < self.convergence_threshold:
                convergence_msg = (f"✅ 收敛达成！修正比例 {refinement_stats['refinement_ratio']:.4f} < 阈值 {self.convergence_threshold}\n"
                                 f"🎯 训练在第 {epoch+1} 轮收敛")
                if self.exp_manager:
                    self.exp_manager.log(convergence_msg)
                else:
                    print(convergence_msg)
                break
            else:
                continue_msg = f"   🔄 继续训练 (修正比例 {refinement_stats['refinement_ratio']:.4f} >= {self.convergence_threshold})"
                if self.exp_manager:
                    self.exp_manager.log(continue_msg)
                else:
                    print(continue_msg)
            
            if self.exp_manager:
                self.exp_manager.log("-" * 70)
        
        return training_history

# ==========================================
# 增强的聚类分析器 - 集成输出管理
# ==========================================

class EnhancedClusteringAnalyzer:
    """增强的聚类结果分析器 - 集成输出管理"""
    
    def __init__(self, model, refinement_module, device, exp_manager):
        self.model = model
        self.refinement_module = refinement_module
        self.device = device
        self.exp_manager = exp_manager
        
    def extract_features_and_labels(self, dataloader):
        """提取所有样本的特征和标签"""
        self.model.eval()
        
        all_features = []
        all_cluster_ids = []
        all_confidences = []
        all_refined_labels = []
        
        with torch.no_grad():
            for cluster_id, (reads, ref) in enumerate(dataloader):
                reads = reads.to(self.device)
                N = reads.shape[1]
                
                # 前向传播
                evidence_batch, contrastive_features = self.model(reads)
                evidence_single_batch = evidence_batch.squeeze(0)
                fused_evidence, strengths = self.model.evidence_fusion(evidence_single_batch)
                
                # 提取特征
                contrastive_features_flat = contrastive_features.squeeze(0)
                all_features.append(contrastive_features_flat.cpu())
                
                # 真实标签（簇ID）
                true_labels = torch.full((N,), cluster_id, dtype=torch.long)
                all_cluster_ids.append(true_labels)
                
                # 计算置信度和修正标签
                current_labels = torch.zeros(N, dtype=torch.long, device=self.device)
                refined_labels, refinement_stats = self.refinement_module(
                    embeddings=contrastive_features_flat,
                    evidence_strengths=strengths.unsqueeze(-1),
                    current_labels=current_labels,
                    num_clusters=1
                )
                
                all_confidences.extend([refinement_stats['avg_confidence']] * N)
                all_refined_labels.append(refined_labels.cpu())
        
        # 合并所有结果
        features = torch.cat(all_features, dim=0).numpy()  # [total_samples, feature_dim]
        true_labels = torch.cat(all_cluster_ids, dim=0).numpy()  # [total_samples]
        refined_labels = torch.cat(all_refined_labels, dim=0).numpy()  # [total_samples]
        
        return features, true_labels, refined_labels, all_confidences
    
    def perform_clustering(self, features, n_clusters):
        """使用K-means进行聚类"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(features)
        return predicted_labels, kmeans.cluster_centers_
    
    def calculate_metrics(self, true_labels, predicted_labels, features):
        """计算聚类评估指标"""
        metrics = {}
        
        # ARI (Adjusted Rand Index) - 越接近1越好
        metrics['ARI'] = adjusted_rand_score(true_labels, predicted_labels)
        
        # NMI (Normalized Mutual Information) - 越接近1越好  
        metrics['NMI'] = normalized_mutual_info_score(true_labels, predicted_labels)
        
        # Silhouette Score - 越接近1越好
        if len(np.unique(predicted_labels)) > 1:
            metrics['Silhouette'] = silhouette_score(features, predicted_labels)
        else:
            metrics['Silhouette'] = -1
            
        # 簇数量对比
        metrics['True_Clusters'] = len(np.unique(true_labels))
        metrics['Predicted_Clusters'] = len(np.unique(predicted_labels))
        
        return metrics
    
    def visualize_clustering(self, features, true_labels, predicted_labels):
        """可视化聚类结果"""
        
        self.exp_manager.log("🔄 正在进行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features)
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 真实标签
        scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                  c=true_labels, cmap='tab20', alpha=0.7, s=20)
        axes[0].set_title(f'真实标签 ({len(np.unique(true_labels))} 个簇)', fontsize=14)
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # 2. 预测标签  
        scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                  c=predicted_labels, cmap='tab20', alpha=0.7, s=20)
        axes[1].set_title(f'预测标签 ({len(np.unique(predicted_labels))} 个簇)', fontsize=14)
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        plt.colorbar(scatter2, ax=axes[1])
        
        # 3. 标签一致性（绿色=一致，红色=不一致）
        consistency = (true_labels == predicted_labels).astype(int)
        scatter3 = axes[2].scatter(features_2d[:, 0], features_2d[:, 1], 
                                  c=consistency, cmap='RdYlGn', alpha=0.7, s=20)
        axes[2].set_title(f'标签一致性 ({np.mean(consistency)*100:.1f}% 一致)', fontsize=14)
        axes[2].set_xlabel('t-SNE 1')
        axes[2].set_ylabel('t-SNE 2')
        plt.colorbar(scatter3, ax=axes[2])
        
        plt.tight_layout()
        
        # 保存图片
        self.exp_manager.save_visualization(fig, "clustering_analysis.png", "聚类分析结果")
    
    def visualize_confidence_distribution(self, confidences):
        """分析置信度分布"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(confidences, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(np.mean(confidences), color='red', linestyle='--', 
                       label=f'平均值: {np.mean(confidences):.3f}')
        axes[0].axvline(self.refinement_module.confidence_threshold, color='orange', linestyle='--',
                       label=f'阈值: {self.refinement_module.confidence_threshold}')
        axes[0].set_xlabel('置信度')
        axes[0].set_ylabel('频次')
        axes[0].set_title('置信度分布')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].boxplot(confidences)
        axes[1].set_ylabel('置信度')
        axes[1].set_title('置信度箱线图')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        self.exp_manager.save_visualization(fig, "confidence_distribution.png", "置信度分布分析")
    
    def plot_training_curves(self, training_history):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        epochs = range(1, len(training_history['losses']) + 1)
        total_losses = [loss['total_loss'] for loss in training_history['losses']]
        recon_losses = [loss['reconstruction_loss'] for loss in training_history['losses']]
        
        axes[0, 0].plot(epochs, total_losses, 'b-', label='总损失', linewidth=2)
        axes[0, 0].plot(epochs, recon_losses, 'r--', label='重构损失', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('训练损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 修正比例曲线
        axes[0, 1].plot(epochs, training_history['refinement_ratios'], 'g-', linewidth=2)
        axes[0, 1].axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='收敛阈值')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('修正比例')
        axes[0, 1].set_title('修正比例变化')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 置信度曲线
        axes[1, 0].plot(epochs, training_history['confidences'], 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('平均置信度')
        axes[1, 0].set_title('置信度变化')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 噪声样本数曲线
        axes[1, 1].plot(epochs, training_history['noise_counts'], 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('噪声样本数')
        axes[1, 1].set_title('噪声样本数变化')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        self.exp_manager.save_visualization(fig, "training_curves.png", "训练过程曲线")
    
    def full_analysis(self, dataloader, training_history, expected_clusters=50):
        """完整的聚类分析"""
        self.exp_manager.log("🔍 开始聚类结果分析...")
        self.exp_manager.log("=" * 60)
        
        # 1. 提取特征和标签
        self.exp_manager.log("📊 提取特征和标签...")
        features, true_labels, refined_labels, confidences = self.extract_features_and_labels(dataloader)
        
        self.exp_manager.log(f"   总样本数: {len(features)}")
        self.exp_manager.log(f"   特征维度: {features.shape[1]}")
        self.exp_manager.log(f"   真实簇数: {len(np.unique(true_labels))}")
        
        # 2. 使用K-means重新聚类
        self.exp_manager.log(f"🎯 使用K-means进行 {expected_clusters} 簇聚类...")
        predicted_labels, cluster_centers = self.perform_clustering(features, expected_clusters)
        
        # 3. 计算评估指标
        self.exp_manager.log("📈 计算聚类评估指标...")
        metrics = self.calculate_metrics(true_labels, predicted_labels, features)
        
        metrics_summary = (f"📊 聚类质量评估:\n"
                          f"   ARI (调整兰德指数):     {metrics['ARI']:.4f}\n"
                          f"   NMI (标准化互信息):     {metrics['NMI']:.4f}\n"
                          f"   Silhouette Score:      {metrics['Silhouette']:.4f}\n"
                          f"   真实簇数:              {metrics['True_Clusters']}\n"
                          f"   预测簇数:              {metrics['Predicted_Clusters']}")
        self.exp_manager.log(metrics_summary)
        
        # 4. 分析置信度分布
        confidence_stats = (f"🎲 置信度统计:\n"
                           f"   平均置信度:            {np.mean(confidences):.4f}\n"
                           f"   置信度标准差:          {np.std(confidences):.4f}\n"
                           f"   最小置信度:            {np.min(confidences):.4f}\n"
                           f"   最大置信度:            {np.max(confidences):.4f}\n"
                           f"   低于阈值的样本比例:     {np.mean(np.array(confidences) < self.refinement_module.confidence_threshold)*100:.2f}%")
        self.exp_manager.log(confidence_stats)
        
        # 5. 生成可视化
        self.exp_manager.log("🎨 生成可视化结果...")
        self.visualize_clustering(features, true_labels, predicted_labels)
        self.visualize_confidence_distribution(confidences)
        self.plot_training_curves(training_history)
        
        # 6. 保存数据
        self.exp_manager.save_metrics(metrics)
        self.exp_manager.save_cluster_assignments(true_labels, predicted_labels, confidences)
        
        # 7. 给出改进建议
        suggestions = []
        if metrics['ARI'] < 0.3:
            suggestions.append("⚠️  ARI过低，模型聚类效果差")
            suggestions.append("建议: 降低置信度阈值，增加训练轮数")
        if np.mean(confidences) > 1.0:
            suggestions.append("⚠️  置信度过高，可能过拟合")
            suggestions.append("建议: 调整证据强度计算方法")
        if np.mean(np.array(confidences) < self.refinement_module.confidence_threshold) < 0.1:
            suggestions.append("⚠️  几乎没有困难样本，修正模块未发挥作用")
            suggestions.append("建议: 降低置信度阈值到0.1-0.2")
            
        if suggestions:
            self.exp_manager.log("💡 改进建议:")
            for suggestion in suggestions:
                self.exp_manager.log(f"   {suggestion}")
            
        return {
            'features': features,
            'true_labels': true_labels, 
            'predicted_labels': predicted_labels,
            'metrics': metrics,
            'confidences': confidences
        }

# ==========================================
# 主训练函数 - 完整的实验管理版本
# ==========================================

def train_with_full_management():
    """完整的实验管理版本训练函数"""
    
    # 实验配置
    experiment_name = "HighIndel_DNA_Clustering"
    DATA_DIR = "Dataset/CloverExp/train"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建实验管理器
    exp_manager = ExperimentManager(experiment_name)
    
    # 训练参数
    input_dim = 4
    hidden_dim = 64
    seq_len = 150
    lr = 1e-3
    
    # 损失权重
    alpha = 1.0
    beta = 0.01
    gamma = 0.01
    
    # 修正参数
    confidence_threshold = 0.15
    distance_threshold = 1.0
    convergence_threshold = 0.05
    max_epochs = 10
    
    # 保存配置
    config = {
        "model_config": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "seq_len": seq_len,
        },
        "training_config": {
            "learning_rate": lr,
            "loss_weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
            "max_epochs": max_epochs,
            "device": str(DEVICE)
        },
        "refinement_config": {
            "confidence_threshold": confidence_threshold,
            "distance_threshold": distance_threshold,
            "convergence_threshold": convergence_threshold
        },
        "data_config": {
            "data_dir": DATA_DIR,
            "seq_length": seq_len
        }
    }
    
    exp_manager.save_config(config)
    
    try:
        # 检查数据目录
        if not os.path.exists(DATA_DIR):
            exp_manager.log(f"❌ 目录不存在: {DATA_DIR}", "error")
            return None, None
            
        # 加载数据
        dataset = CloverClusterDataset(DATA_DIR)
        if len(dataset) == 0:
            exp_manager.log("❌ 数据集为空，请检查数据文件", "error")
            return None, None
            
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
            max_epochs=max_epochs,
            exp_manager=exp_manager
        )
        
        exp_manager.log(f"🔧 模型配置:")
        exp_manager.log(f"   设备: {DEVICE}")
        exp_manager.log(f"   数据集大小: {len(dataset)} 个簇")
        exp_manager.log(f"   损失权重: 重构={alpha}, 对比学习={beta}, KL散度={gamma}")
        exp_manager.log(f"   修正参数: 置信度阈值={confidence_threshold}, 距离阈值={distance_threshold}")
        exp_manager.log(f"   收敛条件: 修正比例 < {convergence_threshold*100}%")
        
        # 开始训练
        training_history = trainer.train_with_refinement(dataloader, DEVICE)
        
        # 聚类结果分析
        exp_manager.log("\n" + "="*60)
        analyzer = EnhancedClusteringAnalyzer(model, refinement_module, DEVICE, exp_manager)
        analysis_results = analyzer.full_analysis(dataloader, training_history, expected_clusters=50)
        
        # 保存模型
        save_dict = {
            'model_state_dict': model.state_dict(),
            'refinement_state_dict': refinement_module.state_dict(),
            'training_history': training_history,
            'analysis_results': analysis_results,
            'config': config
        }
        
        exp_manager.save_model(save_dict)
        
        # 生成报告
        exp_manager.generate_report(training_history, analysis_results)
        exp_manager.create_readme(f"高插入缺失DNA聚类实验 - {len(dataset)}个簇，{sum(len(cluster['reads']) for cluster in dataset.clusters)}条reads")
        
        exp_manager.log(f"🎉 实验完成！所有结果已保存到: {exp_manager.exp_dir}")
        exp_manager.log("📋 快速查看结果:")
        exp_manager.log(f"   - 训练日志: logs/training.log")
        exp_manager.log(f"   - 聚类效果: visualizations/clustering_analysis.png")
        exp_manager.log(f"   - 详细报告: results/analysis_report.txt")
        exp_manager.log(f"   - 模型文件: model/refined_model.pth")
        
        return training_history, analysis_results, exp_manager.exp_dir
        
    except Exception as e:
        exp_manager.log(f"❌ 训练失败: {e}", "error")
        import traceback
        error_details = traceback.format_exc()
        exp_manager.log(f"错误详情:\n{error_details}", "error")
        
        # 保存错误信息到文件
        error_file = os.path.join(exp_manager.exp_dir, "error_log.txt")
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"实验失败时间: {datetime.now()}\n")
            f.write(f"错误信息: {e}\n\n")
            f.write("详细错误堆栈:\n")
            f.write(error_details)
        
        return None, None, exp_manager.exp_dir

# ==========================================
# 实验结果查看器
# ==========================================

class ExperimentViewer:
    """实验结果查看器"""
    
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        
    def list_experiments(self, base_dir="outputs"):
        """列出所有实验"""
        if not os.path.exists(base_dir):
            print(f"❌ 输出目录不存在: {base_dir}")
            return []
            
        experiments = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                config_file = os.path.join(item_path, "config.json")
                if os.path.exists(config_file):
                    experiments.append(item)
                    
        experiments.sort(reverse=True)  # 按时间倒序
        return experiments
    
    def show_experiment_summary(self):
        """显示实验摘要"""
        config_file = os.path.join(self.exp_dir, "config.json")
        report_file = os.path.join(self.exp_dir, "results", "analysis_report.txt")
        
        print(f"📁 实验目录: {self.exp_dir}")
        print("=" * 60)
        
        # 读取配置
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            exp_info = config.get("experiment_info", {})
            print(f"🏷️  实验名称: {exp_info.get('name', 'Unknown')}")
            print(f"⏰ 实验时间: {exp_info.get('timestamp', 'Unknown')}")
            
            model_config = config.get("model_config", {})
            print(f"🤖 模型配置: 隐藏维度={model_config.get('hidden_dim', 'Unknown')}")
            
            training_config = config.get("training_config", {})
            print(f"🚀 训练配置: 学习率={training_config.get('learning_rate', 'Unknown')}, 最大轮数={training_config.get('max_epochs', 'Unknown')}")
        
        # 读取报告
        if os.path.exists(report_file):
            print("\n📊 实验结果:")
            with open(report_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                in_metrics = False
                for line in lines:
                    line = line.strip()
                    if "聚类评估:" in line:
                        in_metrics = True
                        continue
                    elif in_metrics and line.startswith("   "):
                        print(f"  {line}")
                    elif in_metrics and not line.startswith("   "):
                        break
        
        # 检查文件存在性
        print(f"\n📂 输出文件:")
        files_to_check = [
            ("模型文件", "model/refined_model.pth"),
            ("聚类分析图", "visualizations/clustering_analysis.png"),
            ("置信度分布图", "visualizations/confidence_distribution.png"),
            ("训练曲线图", "visualizations/training_curves.png"),
            ("详细报告", "results/analysis_report.txt"),
            ("聚类结果", "results/cluster_assignments.csv")
        ]
        
        for name, path in files_to_check:
            full_path = os.path.join(self.exp_dir, path)
            status = "✅" if os.path.exists(full_path) else "❌"
            print(f"  {status} {name}: {path}")
    
    def load_model(self):
        """加载训练好的模型"""
        model_file = os.path.join(self.exp_dir, "model", "refined_model.pth")
        if not os.path.exists(model_file):
            print(f"❌ 模型文件不存在: {model_file}")
            return None
            
        try:
            model_dict = torch.load(model_file, map_location='cpu')
            print(f"✅ 模型加载成功: {model_file}")
            return model_dict
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None
    
    def show_metrics(self):
        """显示详细指标"""
        metrics_file = os.path.join(self.exp_dir, "results", "metrics_summary.json")
        if not os.path.exists(metrics_file):
            print(f"❌ 指标文件不存在: {metrics_file}")
            return
            
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
            
        print("📈 详细评估指标:")
        print("=" * 40)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

def list_all_experiments():
    """列出所有实验"""
    base_dir = "outputs"
    if not os.path.exists(base_dir):
        print(f"❌ 输出目录不存在: {base_dir}")
        print("💡 请先运行实验生成结果")
        return
        
    experiments = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            config_file = os.path.join(item_path, "config.json")
            if os.path.exists(config_file):
                experiments.append(item)
                
    if not experiments:
        print("📭 没有找到任何实验结果")
        return
        
    experiments.sort(reverse=True)  # 按时间倒序
    
    print(f"📋 找到 {len(experiments)} 个实验:")
    print("=" * 80)
    
    for i, exp in enumerate(experiments, 1):
        exp_path = os.path.join(base_dir, exp)
        config_file = os.path.join(exp_path, "config.json")
        
        # 读取基本信息
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            exp_info = config.get("experiment_info", {})
            name = exp_info.get("name", "Unknown")
            timestamp = exp_info.get("timestamp", "Unknown")
        except:
            name = "Unknown"
            timestamp = "Unknown"
            
        print(f"{i:2d}. {exp}")
        print(f"    名称: {name}")
        print(f"    时间: {timestamp}")
        print(f"    路径: {exp_path}")
        print()

def view_experiment(exp_name=None):
    """查看指定实验"""
    if exp_name is None:
        # 显示所有实验让用户选择
        list_all_experiments()
        return
        
    exp_path = os.path.join("outputs", exp_name)
    if not os.path.exists(exp_path):
        print(f"❌ 实验不存在: {exp_path}")
        return
        
    viewer = ExperimentViewer(exp_path)
    viewer.show_experiment_summary()
    
    print(f"\n💡 查看详细结果:")
    print(f"   viewer = ExperimentViewer('{exp_path}')")
    print(f"   viewer.show_metrics()  # 查看详细指标")
    print(f"   viewer.load_model()    # 加载模型")

# ==========================================
# 主程序入口
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DNA聚类实验管理系统")
    parser.add_argument("--mode", choices=["train", "list", "view"], default="train",
                       help="运行模式: train=训练, list=列出实验, view=查看实验")
    parser.add_argument("--exp_name", type=str, help="实验名称 (用于view模式)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("🎯 启动完整训练+分析模式...")
        train_history, analysis_results, exp_dir = train_with_full_management()
        
        if train_history is not None:
            print(f"\n🎉 训练完成！实验结果保存在: {exp_dir}")
            print(f"💡 查看结果: python {__file__} --mode view --exp_name {os.path.basename(exp_dir)}")
        else:
            print("❌ 训练失败，请查看错误日志")
            
    elif args.mode == "list":
        list_all_experiments()
        
    elif args.mode == "view":
        view_experiment(args.exp_name)
        
    else:
        print("❌ 未知模式，请使用 --help 查看帮助")
