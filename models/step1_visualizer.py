# models/step1_visualizer.py
"""
Step1训练结果可视化与报告生成 - 顶刊增强版
功能：
1. 生成训练曲线 (Loss, Evidence Strength)
2. 生成三区制不确定性散点图 (U_epi vs U_ale)
3. 生成训练总结报告与配置文件
"""
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns
import json
import os
from datetime import datetime
import torch
import numpy as np

# 设置绘图样式
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")

class Step1Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.reports_dir = os.path.join(output_dir, "reports")
        self.models_dir = os.path.join(output_dir, "models")
        
        # 创建子目录
        for dir_path in [self.plots_dir, self.logs_dir, self.reports_dir, self.models_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def _check_history_data(self, history):
        """检查并清理历史数据，防止 NaN 导致的绘图崩溃"""
        cleaned_history = {}
        
        for key, values in history.items():
            if isinstance(values, list) and len(values) > 0:
                # 过滤掉NaN和inf值
                clean_values = []
                for v in values:
                    if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v)):
                        clean_values.append(v)
                    else:
                        clean_values.append(0.0)  # 用0替换异常值
                
                cleaned_history[key] = clean_values
            else:
                # 如果列表为空，创建一个默认值
                cleaned_history[key] = [0.0]
                # print(f"   ⚠️ 历史记录 '{key}' 为空，使用默认值") # 减少日志噪音
        
        return cleaned_history
    
    def plot_training_losses(self, history):
        """绘制训练损失曲线"""
        history = self._check_history_data(history)
        
        if len(history.get('total_loss', [])) == 0:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Step1 Training Losses (Evidential Learning)', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['total_loss']) + 1)
        
        # 总损失
        if 'total_loss' in history:
            axes[0, 0].plot(epochs, history['total_loss'], 'b-', linewidth=2, label='Total Loss')
            axes[0, 0].set_title('Total Loss', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # 对比学习损失
        if 'contrastive_loss' in history:
            axes[0, 1].plot(epochs, history['contrastive_loss'], 'r-', linewidth=2, label='Contrastive Loss')
            axes[0, 1].set_title('Contrastive Loss', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # 重建损失 (核心红线权重 10.0)
        if 'reconstruction_loss' in history:
            axes[1, 0].plot(epochs, history['reconstruction_loss'], 'g-', linewidth=2, label='Reconstruction (wt:10.0)')
            axes[1, 0].set_title('Reconstruction Loss', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # KL散度
        if 'kl_loss' in history:
            axes[1, 1].plot(epochs, history['kl_loss'], 'm-', linewidth=2, label='KL Divergence')
            axes[1, 1].set_title('KL Divergence', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        loss_plot_path = os.path.join(self.plots_dir, "training_losses.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return loss_plot_path
    
    def plot_evidence_stats(self, history):
        """绘制Evidence统计图"""
        history = self._check_history_data(history)
        
        if len(history.get('avg_strength', [])) == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Evidence Statistics', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['avg_strength']) + 1)
        
        # 平均Evidence强度
        if 'avg_strength' in history:
            axes[0].plot(epochs, history['avg_strength'], 'orange', linewidth=3, label='Average Strength')
            axes[0].set_title('Average Evidence Strength (S)', fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        
        # 高置信度比例
        if 'high_conf_ratio' in history:
            high_conf_percent = [x * 100 for x in history['high_conf_ratio']]
            axes[1].plot(epochs, high_conf_percent, 'purple', linewidth=3, label='High Confidence %')
            axes[1].set_title('High Confidence Ratio (>10)', fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Percentage (%)')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        plt.tight_layout()
        evidence_plot_path = os.path.join(self.plots_dir, "evidence_stats.png")
        plt.savefig(evidence_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return evidence_plot_path
    
    def plot_learning_curves(self, history):
        """绘制归一化综合学习曲线，用于趋势对比"""
        history = self._check_history_data(history)
        required_keys = ['total_loss', 'avg_strength', 'high_conf_ratio']
        available_keys = [k for k in required_keys if k in history and len(history[k]) > 0]
        
        if len(available_keys) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        max_length = max(len(history[k]) for k in available_keys)
        epochs = range(1, max_length + 1)
        
        def safe_normalize(data):
            data = np.array(data)
            if len(data) == 0: return np.array([])
            d_min, d_max = np.min(data), np.max(data)
            if d_max == d_min: return np.zeros_like(data)
            return (data - d_min) / (d_max - d_min)
        
        if 'total_loss' in available_keys:
            norm_loss = safe_normalize(history['total_loss'])
            ax.plot(epochs, norm_loss, 'b-', linewidth=2, label='Total Loss (norm)')
        
        if 'avg_strength' in available_keys:
            norm_str = safe_normalize(history['avg_strength'])
            ax.plot(epochs, norm_str, 'orange', linewidth=2, label='Avg Strength (norm)')
        
        if 'high_conf_ratio' in available_keys:
            ax.plot(epochs, history['high_conf_ratio'], 'purple', linewidth=2, label='High Conf Ratio (raw)')
        
        ax.set_title('Learning Trends Overview', fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Normalized Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        curves_plot_path = os.path.join(self.plots_dir, "learning_curves.png")
        plt.savefig(curves_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return curves_plot_path

    # =========================================================================
    # 🆕 新增：不确定性散点图 (用于 Step2 调用)
    # =========================================================================
    def plot_uncertainty_distribution(self, u_epi, u_ale, zone_ids):
        """
        绘制 Epistemic vs Aleatoric 不确定性散点图
        zone_ids: 1=Safe(Green), 2=Hard(Orange), 3=Dirty(Red)
        """
        print(f"   📊 正在生成不确定性分布图 (Scatter)...")
        plt.figure(figsize=(10, 8))
        
        # 确保转为 numpy
        if isinstance(u_epi, torch.Tensor): u_epi = u_epi.cpu().numpy()
        if isinstance(u_ale, torch.Tensor): u_ale = u_ale.cpu().numpy()
        if isinstance(zone_ids, torch.Tensor): zone_ids = zone_ids.cpu().numpy()
        
        # 只绘制有效样本 (排除 label=-1 的 zone=0 或复活中的样本)
        mask = (zone_ids > 0)
        if not mask.any():
            print("   ⚠️ 无有效 Zone 数据，跳过绘图")
            return

        # 随机采样以避免点太多导致绘图极慢 (上限 50000 点)
        if mask.sum() > 50000:
            indices = np.where(mask)[0]
            sampled_indices = np.random.choice(indices, 50000, replace=False)
            plot_mask = np.zeros_like(mask, dtype=bool)
            plot_mask[sampled_indices] = True
            mask = plot_mask

        sns.scatterplot(
            x=u_epi[mask], 
            y=u_ale[mask], 
            hue=zone_ids[mask], 
            palette={1: '#2ecc71', 2: '#f39c12', 3: '#e74c3c'}, # Green, Orange, Red
            style=zone_ids[mask],
            alpha=0.6,
            s=15
        )
        
        plt.title('Uncertainty Distribution & Three-Zone Partitioning', fontsize=14, fontweight='bold')
        plt.xlabel('Epistemic Uncertainty (U_epi) - Model Ignorance')
        plt.ylabel('Aleatoric Uncertainty (U_ale) - Data Noise')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 自定义图例
        handles, labels = plt.gca().get_legend_handles_labels()
        zone_labels = {'1': 'Zone I (Safe)', '2': 'Zone II (Hard)', '3': 'Zone III (Dirty)'}
        new_labels = [zone_labels.get(l, l) for l in labels]
        plt.legend(handles, new_labels, title='Zones')

        save_path = os.path.join(self.plots_dir, "uncertainty_scatter.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 散点图已保存: {save_path}")

    def save_config(self, args):
        """保存训练配置"""
        config = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'args': vars(args)
        }
        config_path = os.path.join(self.logs_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return config_path
    
    def save_training_summary(self, history, model, args):
        """生成训练总结报告"""
        history = self._check_history_data(history)
        summary_path = os.path.join(self.reports_dir, "training_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SSI-EC Step1 Training Summary\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"实验目录: {args.experiment_dir}\n")
            f.write(f"训练设备: {args.device}\n")
            f.write(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}\n\n")
            
            if len(history.get('total_loss', [])) > 0:
                f.write("最终指标:\n")
                f.write(f"   Total Loss:       {history['total_loss'][-1]:.6f}\n")
                f.write(f"   Reconstruction:   {history.get('reconstruction_loss', [0])[-1]:.6f}\n")
                f.write(f"   Avg Strength:     {history.get('avg_strength', [0])[-1]:.4f}\n")
                f.write(f"   High Conf Ratio:  {history.get('high_conf_ratio', [0])[-1]*100:.2f}%\n")
            
            f.write("\n模型结构:\n")
            f.write(f"   Dim: {args.dim}, MaxLen: {args.max_length}\n")
            f.write(f"   Total Params: {sum(p.numel() for p in model.parameters()):,}\n")
            
        print(f"   📄 训练报告已保存: {summary_path}")
        return summary_path
    
    def save_model_info(self, model):
        """保存详细模型结构"""
        info_path = os.path.join(self.reports_dir, "model_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(str(model))
        return info_path
    
    def generate_all_outputs(self, history, model, args):
        """生成所有输出文件 (Loss, Config, Report)"""
        print(f"\n📊 [Visualizer] 生成可视化报告...")
        
        try:
            self.plot_training_losses(history)
            self.plot_evidence_stats(history)
            self.plot_learning_curves(history)
            self.save_config(args)
            self.save_training_summary(history, model, args)
            self.save_model_info(model)
        except Exception as e:
            print(f"   ❌ 可视化生成部分失败: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"   ✅ 可视化完成: {self.output_dir}")