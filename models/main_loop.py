# models/main_loop.py
"""
SSI-EC 闭环迭代总控

状态传递机制:
  每轮维护三个路径:
    current_checkpoint  — 上一轮 Step1 的模型权重
    current_labels      — 上一轮 Step2 输出的 refined_labels.txt
    current_state       — 上一轮 Step2 输出的 read_state.pt (含 u_epi/u_ale/zone_ids)

  Round 1:
    Step1: FedDNA 预训练权重 → 20 epoch, lr=1e-4
    Step2: 无 prev_state, round_idx=1 (delta 宽松 x1.5)

  Round 2+:
    Step1: 上一轮 checkpoint → 10 epoch, lr=1e-5
           动态采样器读 read_state.pt 做三区制采样
    Step2: 传入 prev_state 做动量更新, round_idx=N (delta 严格 x1.0)
"""
import os
import argparse
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.step1_train  import train_step1
from models.step2_runner import run_step2


def main_loop():
    parser = argparse.ArgumentParser(description="SSI-EC Iterative Clustering Master Loop")
    parser.add_argument('--experiment_dir',    type=str,  
                        default='code/CC/Step0/Experiments/ERR036',
                        help="实验根目录")
    parser.add_argument('--max_iterations',    type=int,  default=3,      help="最大迭代轮数")
    parser.add_argument('--device',            type=str,  default='cuda')
    parser.add_argument('--feddna_checkpoint', type=str,
                        default='result/FLDNA_I/I_1214234233/model/epoch1_I.pth',
                        help="FedDNA 预训练权重路径（Round 1 使用）")

    args = parser.parse_args()

    # =====================================================================
    # 状态变量
    # =====================================================================
    current_labels_path     = None   # Round 1 为 None → 用 Clover 原始标签
    current_checkpoint_path = None   # Round 1 为 None → 用 FedDNA 预训练
    current_state_path      = None   # Round 1 为 None → 无动量 / 无三区制采样

    print(f"🚀 开始 SSI-EC 闭环迭代训练")
    print(f"📂 实验目录: {args.experiment_dir}")
    print(f"🔁 最大轮数: {args.max_iterations}")

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'=' * 80}")
        print(f"🔄 Iteration {iteration} / {args.max_iterations}")
        print(f"{'=' * 80}")

        # ==================================================================
        # Step 1: 训练
        # ==================================================================
        print(f"\n[Step 1] Training (Round {iteration})...")

        step1_out_dir = os.path.join(
            args.experiment_dir, "results", f"iter_{iteration}_step1"
        )

        step1_args = argparse.Namespace(
            experiment_dir          = args.experiment_dir,
            output_dir              = step1_out_dir,
            batch_size              = 32,                   # 红线
            max_clusters_per_batch  = 5,
            weight_decay            = 1e-5,
            dim                     = 256,
            max_length              = 152,
            min_clusters            = 50,
            device                  = args.device,

            # Hot Start 参数
            round_idx               = iteration,
            feddna_checkpoint       = args.feddna_checkpoint,   # Round 1 用
            prev_checkpoint         = current_checkpoint_path,  # Round 2+ 用
            refined_labels          = current_labels_path,      # Round 2+ 用
            prev_state              = current_state_path,       # Round 2+ 用（动态采样）
        )

        # train_step1 返回 final_model_path (str)
        step1_checkpoint = train_step1(step1_args)

        if step1_checkpoint is None:
            print("❌ Step 1 未返回有效 checkpoint，停止迭代。")
            break

        # ==================================================================
        # Step 2: 修正与重建
        # ==================================================================
        print(f"\n[Step 2] Refining & Decoding (Round {iteration})...")

        step2_out_dir = os.path.join(
            args.experiment_dir, "results", f"iter_{iteration}_step2"
        )

        step2_args = argparse.Namespace(
            experiment_dir      = args.experiment_dir,
            step1_checkpoint    = step1_checkpoint,
            output_dir          = step2_out_dir,
            dim                 = 256,
            max_length          = 150,
            device              = args.device,

            # 三区制 + 噪声复活参数
            round_idx           = iteration,
            refined_labels      = current_labels_path,     # 上一轮的 labels（复活用）
            prev_state          = current_state_path,      # 上一轮的 state（动量用）
        )

        results = run_step2(step2_args)

        # ==================================================================
        # 状态更新
        # ==================================================================
        if results and 'next_round_files' in results:
            nrf = results['next_round_files']

            current_labels_path     = nrf['labels']
            current_state_path      = nrf.get('state', None)
            current_checkpoint_path = step1_checkpoint

            print(f"\n✅ Iteration {iteration} 完成!")
            print(f"   📝 新标签:      {current_labels_path}")
            print(f"   💾 新状态:      {current_state_path}")
            print(f"   🧬 新序列:      {nrf.get('reference', 'N/A')}")
            print(f"   🔋 下一轮权重:  {current_checkpoint_path}")
        else:
            print("❌ Step 2 未返回有效结果，停止迭代。")
            break

    print("\n🎉 所有迭代完成！")


if __name__ == "__main__":
    main_loop()
