# models/step1_visualizer.py
"""
Step1 training visualization: loss curves, evidence statistics, and config dump.
Only the plots consumed by step1_train are kept.
"""
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
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
        """Sanitize history: replace NaN/Inf with 0.0, empty series with [0.0]."""
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
            (axes[1,0], 'reconstruction_loss', 'g', 'Reconstruction Loss'),
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

    def save_config(self, args):
        config = {'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  'args': vars(args)}
        path = os.path.join(self.logs_dir, "config.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return path