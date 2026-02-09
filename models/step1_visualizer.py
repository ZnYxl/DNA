# models/step1_visualizer.py
"""
修复清单:
  [FIX-#6]  save_training_summary 用 getattr 安全获取 args.epochs / args.lr
"""
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import torch
import numpy as np

plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")


class Step1Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.plots_dir   = os.path.join(output_dir, "plots")
        self.logs_dir    = os.path.join(output_dir, "logs")
        self.reports_dir = os.path.join(output_dir, "reports")
        self.models_dir  = os.path.join(output_dir, "models")

        for d in [self.plots_dir, self.logs_dir, self.reports_dir, self.models_dir]:
            os.makedirs(d, exist_ok=True)

    def _check_history_data(self, history):
        cleaned = {}
        for key, values in history.items():
            if isinstance(values, list) and len(values) > 0:
                clean = []
                for v in values:
                    if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v)):
                        clean.append(v)
                    else:
                        clean.append(0.0)
                cleaned[key] = clean
            else:
                cleaned[key] = [0.0]
        return cleaned

    def plot_training_losses(self, history):
        history = self._check_history_data(history)
        if len(history.get('total_loss', [])) == 0:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Step1 Training Losses', fontsize=16, fontweight='bold')
        epochs = range(1, len(history['total_loss']) + 1)

        for ax, key, color, label in [
            (axes[0,0], 'total_loss',          'b', 'Total Loss'),
            (axes[0,1], 'contrastive_loss',    'r', 'Contrastive Loss'),
            (axes[1,0], 'reconstruction_loss', 'g', 'Reconstruction (wt:10.0)'),
            (axes[1,1], 'kl_loss',             'm', 'KL Divergence'),
        ]:
            if key in history:
                ax.plot(epochs, history[key], f'{color}-', linewidth=2, label=label)
                ax.set_title(label, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()

        plt.tight_layout()
        path = os.path.join(self.plots_dir, "training_losses.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    def plot_evidence_stats(self, history):
        history = self._check_history_data(history)
        if len(history.get('avg_strength', [])) == 0:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Evidence Statistics', fontsize=16, fontweight='bold')
        epochs = range(1, len(history['avg_strength']) + 1)

        if 'avg_strength' in history:
            axes[0].plot(epochs, history['avg_strength'], 'orange', linewidth=3)
            axes[0].set_title('Average Evidence Strength'); axes[0].grid(True, alpha=0.3)

        if 'high_conf_ratio' in history:
            axes[1].plot(epochs, [x*100 for x in history['high_conf_ratio']], 'purple', linewidth=3)
            axes[1].set_title('High Confidence %'); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.plots_dir, "evidence_stats.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    def plot_learning_curves(self, history):
        history = self._check_history_data(history)
        keys = [k for k in ['total_loss','avg_strength','high_conf_ratio']
                if k in history and len(history[k]) > 0]
        if not keys: return None

        fig, ax = plt.subplots(figsize=(12, 8))
        max_len = max(len(history[k]) for k in keys)
        epochs = range(1, max_len + 1)

        def norm(d):
            d = np.array(d)
            mn, mx = d.min(), d.max()
            return (d - mn) / (mx - mn) if mx > mn else np.zeros_like(d)

        for key, color, label in [
            ('total_loss', 'b', 'Total Loss (norm)'),
            ('avg_strength', 'orange', 'Avg Strength (norm)'),
            ('high_conf_ratio', 'purple', 'High Conf Ratio'),
        ]:
            if key in keys:
                data = norm(history[key]) if 'norm' in label else history[key]
                ax.plot(epochs, data, color=color, linewidth=2, label=label)

        ax.set_title('Learning Trends', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout()
        path = os.path.join(self.plots_dir, "learning_curves.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    def plot_uncertainty_distribution(self, u_epi, u_ale, zone_ids):
        print(f"   📊 生成不确定性分布图...")
        plt.figure(figsize=(10, 8))

        if isinstance(u_epi, torch.Tensor): u_epi = u_epi.cpu().numpy()
        if isinstance(u_ale, torch.Tensor): u_ale = u_ale.cpu().numpy()
        if isinstance(zone_ids, torch.Tensor): zone_ids = zone_ids.cpu().numpy()

        mask = (zone_ids > 0)
        if not mask.any():
            print("   ⚠️ 无有效 Zone 数据"); return

        if mask.sum() > 50000:
            idx = np.where(mask)[0]
            sampled = np.random.choice(idx, 50000, replace=False)
            new_mask = np.zeros_like(mask, dtype=bool)
            new_mask[sampled] = True
            mask = new_mask

        sns.scatterplot(
            x=u_epi[mask], y=u_ale[mask], hue=zone_ids[mask],
            palette={1: '#2ecc71', 2: '#f39c12', 3: '#e74c3c'},
            style=zone_ids[mask], alpha=0.6, s=15
        )
        plt.title('Uncertainty Distribution & Three-Zone Partitioning', fontsize=14, fontweight='bold')
        plt.xlabel('Epistemic Uncertainty (U_epi)')
        plt.ylabel('Aleatoric Uncertainty (U_ale)')
        plt.grid(True, alpha=0.3, linestyle='--')

        handles, labels = plt.gca().get_legend_handles_labels()
        zone_map = {'1': 'Zone I (Safe)', '2': 'Zone II (Hard)', '3': 'Zone III (Dirty)'}
        plt.legend(handles, [zone_map.get(l, l) for l in labels], title='Zones')

        path = os.path.join(self.plots_dir, "uncertainty_scatter.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 散点图: {path}")

    def save_config(self, args):
        config = {'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  'args': vars(args)}
        path = os.path.join(self.logs_dir, "config.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return path

    def save_training_summary(self, history, model, args):
        """[FIX-#6] 安全获取 args.epochs / args.lr"""
        history = self._check_history_data(history)
        path = os.path.join(self.reports_dir, "training_summary.txt")

        epochs = getattr(args, 'epochs', 'N/A')
        lr     = getattr(args, 'lr', 'N/A')

        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SSI-EC Step1 Training Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"实验目录: {args.experiment_dir}\n")
            f.write(f"设备: {args.device}\n")
            f.write(f"Epochs: {epochs} | Batch: {args.batch_size} | LR: {lr}\n\n")

            if len(history.get('total_loss', [])) > 0:
                f.write("最终指标:\n")
                f.write(f"   Total Loss:      {history['total_loss'][-1]:.6f}\n")
                f.write(f"   Reconstruction:  {history.get('reconstruction_loss',[0])[-1]:.6f}\n")
                f.write(f"   Avg Strength:    {history.get('avg_strength',[0])[-1]:.4f}\n")
                f.write(f"   High Conf Ratio: {history.get('high_conf_ratio',[0])[-1]*100:.2f}%\n")

            f.write(f"\nDim: {args.dim}, MaxLen: {args.max_length}\n")
            f.write(f"Total Params: {sum(p.numel() for p in model.parameters()):,}\n")

        print(f"   📄 训练报告: {path}")
        return path

    def save_model_info(self, model):
        path = os.path.join(self.reports_dir, "model_info.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(str(model))
        return path

    def generate_all_outputs(self, history, model, args):
        print(f"\n📊 [Visualizer] 生成可视化报告...")
        try:
            self.plot_training_losses(history)
            self.plot_evidence_stats(history)
            self.plot_learning_curves(history)
            self.save_config(args)
            self.save_training_summary(history, model, args)
            self.save_model_info(model)
        except Exception as e:
            print(f"   ❌ 可视化部分失败: {e}")
        print(f"   ✅ 可视化完成: {self.output_dir}")