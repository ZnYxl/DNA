"""
Step1: 基础训练循环 (DEL 证据驱动)
- Encoder + Decoder
- 对比学习
- 证据融合
- DEL损失计算
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1_model import SimplifiedFedDNA
from step1_loss import ComprehensiveLoss
from step1_data import CloverClusterDataset
from step1_trainer import BasicTrainer

def main():
    """Step1主函数：基础训练"""
    
    print("🚀 Step1: 基础训练循环 (DEL 证据驱动)")
    print("=" * 60)
    
    # 配置参数
    config = {
        'data_dir': "../../Dataset/CloverExp/train",
        'device': 'cuda',
        'model_params': {
            'input_dim': 4,
            'hidden_dim': 64,
            'seq_len': 150
        },
        'training_params': {
            'lr': 1e-3,
            'epochs': 5,
            'loss_weights': {
                'alpha': 1.0,    # 重构损失
                'beta': 0.01,    # 对比学习损失 (降低)
                'gamma': 0.01    # KL散度损失
            }
        }
    }
    
    # 创建训练器并开始训练
    trainer = BasicTrainer(config)
    history = trainer.train()
    
    # 保存结果
    trainer.save_model("step1_model.pth")
    print("\n✅ Step1 训练完成！")
    
    return history

if __name__ == "__main__":
    main()
