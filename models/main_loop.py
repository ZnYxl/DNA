# models/main_loop.py
import os
import argparse
import torch
import sys
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.step1_train import train_step1
from models.step2_runner import run_step2

def main_loop():
    parser = argparse.ArgumentParser(description="FedDNA Iterative Clustering & Reconstruction Master Loop")
    parser.add_argument('--experiment_dir', type=str, required=True, help="实验根目录")
    parser.add_argument('--max_iterations', type=int, default=3, help="最大迭代轮数")
    parser.add_argument('--device', type=str, default='cuda')
    
    # 初始超参数
    parser.add_argument('--step1_epochs', type=int, default=20)
    parser.add_argument('--step1_lr', type=float, default=1e-4)
    
    args = parser.parse_args()

    # 状态追踪
    current_labels_path = None  # 初始为None，使用Clover标签
    step1_checkpoint = None     # 每一轮更新

    print(f"🚀 开始闭环迭代训练 (Max Iterations: {args.max_iterations})")
    print(f"📂 实验目录: {args.experiment_dir}")

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n" + "="*80)
        print(f"🔄 Iteration {iteration} / {args.max_iterations}")
        print("="*80)

        # ==========================================
        # Step 1: 训练 (Training)
        # ==========================================
        print(f"\n[Step 1] Training Model (Iter {iteration})...")
        
        # 构造Step 1参数
        # 注意：你需要修改 step1_train.py 的 Step1Dataset 调用，
        # 让它支持传入 custom_labels_path (如果 current_labels_path 不为 None)
        # 但如果是第一轮，或者 step1_train 还没改好，它会默认用 Clover 标签
        
        step1_out_dir = os.path.join(args.experiment_dir, "results", f"iter_{iteration}_step1")
        
        step1_args = argparse.Namespace(
            experiment_dir=args.experiment_dir,
            output_dir=step1_out_dir,
            epochs=args.step1_epochs,
            batch_size=32,
            max_clusters_per_batch=5,
            lr=args.step1_lr,
            weight_decay=1e-5,
            dim=256,
            max_length=150,
            min_clusters=50,
            device=args.device,
            feddna_checkpoint='result/FLDNA_I/I_1214234233/model/epoch1_I.pth', # 确保路径正确
            save_interval=20
        )
        
        # 运行 Step 1
        model, history = train_step1(step1_args)
        step1_checkpoint = os.path.join(step1_out_dir, "models", "step1_final_model.pth")

        # ==========================================
        # Step 2: 修正与重建 (Refinement)
        # ==========================================
        print(f"\n[Step 2] Refining & Decoding (Iter {iteration})...")
        
        step2_out_dir = os.path.join(args.experiment_dir, "results", f"iter_{iteration}_step2")
        
        step2_args = argparse.Namespace(
            experiment_dir=args.experiment_dir,
            step1_checkpoint=step1_checkpoint,
            output_dir=step2_out_dir,
            dim=256,
            max_length=150,
            device=args.device,
            uncertainty_percentile=0.2,
            delta=None,
            delta_percentile=10
        )
        
        # 运行 Step 2
        results = run_step2(step2_args)
        
        # ==========================================
        # 更新状态 (Update)
        # ==========================================
        if results and 'next_round_files' in results:
            new_labels_path = results['next_round_files']['labels']
            consensus_path = results['next_round_files']['reference']
            
            print(f"\n✅ Iteration {iteration} 完成!")
            print(f"   📝 新标签: {new_labels_path}")
            print(f"   🧬 新序列: {consensus_path}")
            
            # 更新下一轮使用的标签
            current_labels_path = new_labels_path
            
            # TODO: 在 step1_data.py 中实现读取 current_labels_path 的逻辑
            # 目前如果不修改 step1_data，下一轮还是会用 Clover 标签，闭环效果会打折
        else:
            print("❌ Step 2 未返回有效结果，停止迭代。")
            break

    print("\n🎉 所有迭代完成！")

if __name__ == "__main__":
    main_loop()