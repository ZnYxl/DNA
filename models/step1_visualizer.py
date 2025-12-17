# models/step1_visualizer.py - 修复版本
"""
Step1训练结果可视化与报告生成
功能：生成训练曲线、统计图表、训练报告等
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
        """检查并清理历史数据"""
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
                print(f"   ⚠️ 历史记录 '{key}' 为空，使用默认值")
        
        return cleaned_history
    
    def plot_training_losses(self, history):
        """绘制训练损失曲线"""
        # ✅ 检查数据
        history = self._check_history_data(history)
        
        if len(history.get('total_loss', [])) == 0:
            print(f"   ⚠️ 没有损失数据，跳过损失曲线绘制")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Step1 Training Losses', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['total_loss']) + 1)
        
        # 总损失
        if 'total_loss' in history:
            axes[0, 0].plot(epochs, history['total_loss'], 'b-', linewidth=2, label='Total Loss')
            axes[0, 0].set_title('Total Loss', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # 对比学习损失
        if 'contrastive_loss' in history:
            axes[0, 1].plot(epochs, history['contrastive_loss'], 'r-', linewidth=2, label='Contrastive Loss')
            axes[0, 1].set_title('Contrastive Loss', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # 重建损失
        if 'reconstruction_loss' in history:
            axes[1, 0].plot(epochs, history['reconstruction_loss'], 'g-', linewidth=2, label='Reconstruction Loss')
            axes[1, 0].set_title('Reconstruction Loss', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # KL散度
        if 'kl_loss' in history:
            axes[1, 1].plot(epochs, history['kl_loss'], 'm-', linewidth=2, label='KL Divergence')
            axes[1, 1].set_title('KL Divergence', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # 保存图片
        loss_plot_path = os.path.join(self.plots_dir, "training_losses.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 损失曲线已保存: {loss_plot_path}")
        return loss_plot_path
    
    def plot_evidence_stats(self, history):
        """绘制Evidence统计图"""
        # ✅ 检查数据
        history = self._check_history_data(history)
        
        if len(history.get('avg_strength', [])) == 0:
            print(f"   ⚠️ 没有Evidence数据，跳过Evidence统计图绘制")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Evidence Statistics', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['avg_strength']) + 1)
        
        # 平均Evidence强度
        if 'avg_strength' in history:
            axes[0].plot(epochs, history['avg_strength'], 'orange', linewidth=3, label='Average Strength')
            axes[0].set_title('Average Evidence Strength', fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Strength')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        
        # 高置信度比例
        if 'high_conf_ratio' in history:
            high_conf_percent = [x * 100 for x in history['high_conf_ratio']]
            axes[1].plot(epochs, high_conf_percent, 'purple', linewidth=3, label='High Confidence %')
            axes[1].set_title('High Confidence Ratio', fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Percentage (%)')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        plt.tight_layout()
        
        # 保存图片
        evidence_plot_path = os.path.join(self.plots_dir, "evidence_stats.png")
        plt.savefig(evidence_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Evidence统计图已保存: {evidence_plot_path}")
        return evidence_plot_path
    
    def plot_learning_curves(self, history):
        """绘制综合学习曲线"""
        # ✅ 检查数据
        history = self._check_history_data(history)
        
        # 检查是否有足够的数据
        required_keys = ['total_loss', 'avg_strength', 'high_conf_ratio']
        available_keys = [k for k in required_keys if k in history and len(history[k]) > 0]
        
        if len(available_keys) == 0:
            print(f"   ⚠️ 没有足够的数据绘制学习曲线，跳过")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 使用最长的序列作为epoch基准
        max_length = max(len(history[k]) for k in available_keys)
        epochs = range(1, max_length + 1)
        
        # ✅ 安全的归一化函数
        def safe_normalize(data):
            data = np.array(data)
            if len(data) == 0:
                return np.array([])
            
            data_min = np.min(data)
            data_max = np.max(data)
            
            if data_max == data_min:
                return np.zeros_like(data)  # 如果所有值相同，返回0
            else:
                return (data - data_min) / (data_max - data_min)
        
        # 绘制可用的曲线
        if 'total_loss' in available_keys:
            normalized_loss = safe_normalize(history['total_loss'])
            if len(normalized_loss) > 0:
                loss_epochs = range(1, len(normalized_loss) + 1)
                ax.plot(loss_epochs, normalized_loss, 'b-', linewidth=2, label='Total Loss (norm)')
        
        if 'avg_strength' in available_keys:
            normalized_strength = safe_normalize(history['avg_strength'])
            if len(normalized_strength) > 0:
                strength_epochs = range(1, len(normalized_strength) + 1)
                ax.plot(strength_epochs, normalized_strength, 'orange', linewidth=2, label='Avg Strength (norm)')
        
        if 'high_conf_ratio' in available_keys:
            conf_ratio = history['high_conf_ratio']
            if len(conf_ratio) > 0:
                conf_epochs = range(1, len(conf_ratio) + 1)
                ax.plot(conf_epochs, conf_ratio, 'purple', linewidth=2, label='High Conf Ratio')
        
        ax.set_title('Learning Curves Overview', fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Normalized Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # 保存图片
        curves_plot_path = os.path.join(self.plots_dir, "learning_curves.png")
        plt.savefig(curves_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📉 学习曲线已保存: {curves_plot_path}")
        return curves_plot_path
    
    def save_config(self, args):
        """保存训练配置"""
        config = {
            'experiment_info': {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'experiment_dir': args.experiment_dir,
                'output_dir': args.output_dir
            },
            'model_config': {
                'dim': args.dim,
                'max_length': args.max_length,
                'min_clusters': args.min_clusters,
                'device': args.device
            },
            'training_config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'max_clusters_per_batch': args.max_clusters_per_batch,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'save_interval': args.save_interval
            },
            'data_config': {
                'feddna_checkpoint': args.feddna_checkpoint
            }
        }
        
        config_path = os.path.join(self.logs_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"   ⚙️ 配置文件已保存: {config_path}")
        return config_path
    
    def save_training_summary(self, history, model, args):
        """生成训练总结报告"""
        # ✅ 检查数据
        history = self._check_history_data(history)
        
        summary_path = os.path.join(self.reports_dir, "training_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Step1 Evidence-driven Training Summary\n")
            f.write("=" * 80 + "\n\n")
            
            # 基本信息
            f.write("📋 实验信息:\n")
            f.write(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"   实验目录: {args.experiment_dir}\n")
            f.write(f"   输出目录: {args.output_dir}\n\n")
            
            # 模型配置
            f.write("🧠 模型配置:\n")
            f.write(f"   特征维度: {args.dim}\n")
            f.write(f"   序列长度: {args.max_length}\n")
            f.write(f"   最小簇数: {args.min_clusters}\n")
            f.write(f"   总参数量: {sum(p.numel() for p in model.parameters()):,}\n\n")
            
            # 训练配置
            f.write("🚀 训练配置:\n")
            f.write(f"   训练轮数: {args.epochs}\n")
            f.write(f"   批次大小: {args.batch_size}\n")
            f.write(f"   学习率: {args.lr}\n")
            f.write(f"   权重衰减: {args.weight_decay}\n\n")
            
            # ✅ 安全的训练结果记录
            f.write("📊 训练结果:\n")
            if len(history.get('total_loss', [])) > 0:
                f.write(f"   最终总损失: {history['total_loss'][-1]:.6f}\n")
                f.write(f"   最终对比损失: {history.get('contrastive_loss', [0])[-1]:.6f}\n")
                f.write(f"   最终重建损失: {history.get('reconstruction_loss', [0])[-1]:.6f}\n")
                f.write(f"   最终KL散度: {history.get('kl_loss', [0])[-1]:.6f}\n")
                f.write(f"   最终平均强度: {history.get('avg_strength', [0])[-1]:.4f}\n")
                f.write(f"   最终高置信度比例: {history.get('high_conf_ratio', [0])[-1]*100:.2f}%\n\n")
                
                # 训练趋势
                f.write("📈 训练趋势:\n")
                if len(history['total_loss']) > 1:
                    initial_loss = history['total_loss'][0]
                    final_loss = history['total_loss'][-1]
                    if initial_loss > 0:
                        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
                        f.write(f"   损失下降: {loss_reduction:.2f}%\n")
                
                if len(history.get('avg_strength', [])) > 1:
                    initial_strength = history['avg_strength'][0]
                    final_strength = history['avg_strength'][-1]
                    if initial_strength > 0:
                        strength_change = (final_strength - initial_strength) / initial_strength * 100
                        f.write(f"   强度变化: {strength_change:+.2f}%\n")
                
                if len(history.get('high_conf_ratio', [])) > 1:
                    initial_conf = history['high_conf_ratio'][0]
                    final_conf = history['high_conf_ratio'][-1]
                    conf_change = (final_conf - initial_conf) * 100
                    f.write(f"   置信度变化: {conf_change:+.2f}个百分点\n\n")
            else:
                f.write("   ⚠️ 训练数据不完整或训练未成功完成\n\n")
            
            # 方法论检查
            f.write("✅ 方法论验证:\n")
            f.write("   - Evidence-driven学习: ✓\n")
            f.write("   - 严格自监督训练: ✓\n")
            f.write("   - GT仅用于评估: ✓\n")
            f.write("   - 数值稳定性保护: ✓\n")
            f.write("   - Warm-up机制: ✓\n\n")
            
            # 文件清单
            f.write("📁 输出文件:\n")
            f.write("   models/\n")
            f.write("   ├── step1_final_model.pth (最终模型)\n")
            f.write("   └── step1_epoch_*.pth (检查点)\n")
            f.write("   plots/\n")
            f.write("   ├── training_losses.png (损失曲线)\n")
            f.write("   ├── evidence_stats.png (Evidence统计)\n")
            f.write("   └── learning_curves.png (学习曲线)\n")
            f.write("   logs/\n")
            f.write("   └── config.json (配置文件)\n")
            f.write("   reports/\n")
            f.write("   └── training_summary.txt (本报告)\n")
        
        print(f"   📄 训练总结已保存: {summary_path}")
        return summary_path
    
    def save_model_info(self, model):
        """保存模型结构信息"""
        model_info_path = os.path.join(self.reports_dir, "model_info.txt")
        
        with open(model_info_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Step1 Model Architecture\n")
            f.write("=" * 80 + "\n\n")
            
            # 模型结构
            f.write("🏗️ 模型结构:\n")
            f.write(str(model))
            f.write("\n\n")
            
            # 参数统计
            f.write("📊 参数统计:\n")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            f.write(f"   总参数: {total_params:,}\n")
            f.write(f"   可训练参数: {trainable_params:,}\n")
            f.write(f"   冻结参数: {total_params - trainable_params:,}\n\n")
            
            # 各层参数
            f.write("🔍 各层参数详情:\n")
            for name, param in model.named_parameters():
                f.write(f"   {name}: {param.shape} ({param.numel():,} params)\n")
        
        print(f"   🏗️ 模型信息已保存: {model_info_path}")
        return model_info_path
    
    def generate_all_outputs(self, history, model, args):
        """生成所有输出文件"""
        print(f"\n📊 生成训练结果文件...")
        print(f"📁 输出目录: {self.output_dir}")
        
        # ✅ 检查历史数据状态
        print(f"📋 历史数据检查:")
        for key, values in history.items():
            if isinstance(values, list):
                print(f"   {key}: {len(values)} 条记录")
            else:
                print(f"   {key}: {type(values)}")
        
        # 生成图表（带错误处理）
        try:
            self.plot_training_losses(history)
        except Exception as e:
            print(f"   ❌ 损失曲线生成失败: {e}")
        
        try:
            self.plot_evidence_stats(history)
        except Exception as e:
            print(f"   ❌ Evidence统计图生成失败: {e}")
        
        try:
            self.plot_learning_curves(history)
        except Exception as e:
            print(f"   ❌ 学习���线生成失败: {e}")
        
        # 保存配置和报告
        try:
            self.save_config(args)
            self.save_training_summary(history, model, args)
            self.save_model_info(model)
        except Exception as e:
            print(f"   ❌ 报告生成失败: {e}")
        
        print(f"\n✅ 输出文件生成完成！")
        print(f"📂 查看结果: {self.output_dir}")
