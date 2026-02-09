# models/main_loop.py
"""
SSI-EC 闭环迭代总控 (Universal Edition)

修复清单:
  [FIX] 预训练权重路径修正
  [NEW] --gt_tags_file 支持 GT 评估
  [NEW] training_cap 可配置
  通用设计：通过命令行参数切换 Goldman / id20 / ERR036

用法:
  # id20
  python main_loop.py --experiment_dir .../id20_Real --max_length 150 --gt_tags_file .../id20_tags_reads.txt
  # Goldman
  python main_loop.py --experiment_dir .../Goldman_Real --max_length 117 --gt_tags_file None
"""
import os
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.step1_train  import train_step1
from models.step2_runner import run_step2


def main_loop():
    parser = argparse.ArgumentParser(description="SSI-EC Master Loop")

    # ===== 数据集配置 =====
    parser.add_argument('--experiment_dir', type=str,
                        default='/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/id20_Real',
                        help="实验根目录 (包含 03_FedDNA_In/read.txt)")
    parser.add_argument('--max_length', type=int, default=150,
                        help="id20=150, Goldman=117, ERR036=152")

    # ===== GT 评估 (可选) =====
    parser.add_argument('--gt_tags_file', type=str,
                        default='/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/id20/id20_tags_reads.txt',
                        help="GT 标签文件, 无GT时设为 None")
    parser.add_argument('--gt_refs_file', type=str, default=None,
                        help="GT 参考序列 FASTA (可选)")

    # ===== 迭代配置 =====
    parser.add_argument('--max_iterations', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')

    # ===== [FIX] 预训练权重 =====
    parser.add_argument('--feddna_checkpoint', type=str,
                        default='/mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth')

    # ===== 训练超参 =====
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_clusters_per_batch', type=int, default=8)
    parser.add_argument('--training_cap', type=int, default=2000000)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--min_clusters', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    args = parser.parse_args()
    if args.gt_tags_file and args.gt_tags_file.lower() == 'none':
        args.gt_tags_file = None

    current_labels_path     = None
    current_checkpoint_path = None
    current_state_path      = None

    print(f"🚀 SSI-EC 闭环迭代启动")
    print(f"📂 实验目录: {args.experiment_dir}")
    print(f"📏 序列长度: {args.max_length} bp")
    print(f"🔁 迭代轮数: {args.max_iterations}")
    print(f"🔋 预训练:   {os.path.basename(args.feddna_checkpoint)}")
    if args.gt_tags_file:
        print(f"📋 GT 评估:  {os.path.basename(args.gt_tags_file)}")

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'=' * 80}")
        print(f"🔄 Round {iteration} / {args.max_iterations}")
        print(f"{'=' * 80}\n")

        # ============== Step 1 ==============
        print(f"[Step 1] Evidence Learning...")
        step1_out = os.path.join(args.experiment_dir, "results", f"iter_{iteration}_step1")

        step1_args = argparse.Namespace(
            experiment_dir=args.experiment_dir, output_dir=step1_out,
            batch_size=args.batch_size, max_clusters_per_batch=args.max_clusters_per_batch,
            weight_decay=args.weight_decay, dim=args.dim, max_length=args.max_length,
            min_clusters=args.min_clusters, device=args.device, round_idx=iteration,
            feddna_checkpoint=args.feddna_checkpoint,
            prev_checkpoint=current_checkpoint_path,
            refined_labels=current_labels_path, prev_state=current_state_path,
            training_cap=args.training_cap,
        )
        step1_checkpoint = train_step1(step1_args)
        if step1_checkpoint is None:
            print("❌ Step 1 失败"); break

        # ============== Step 2 ==============
        print(f"\n[Step 2] Refine & Decode...")
        step2_out = os.path.join(args.experiment_dir, "results", f"iter_{iteration}_step2")

        step2_args = argparse.Namespace(
            experiment_dir=args.experiment_dir, step1_checkpoint=step1_checkpoint,
            output_dir=step2_out, dim=args.dim, max_length=args.max_length,
            device=args.device, round_idx=iteration,
            refined_labels=current_labels_path, prev_state=current_state_path,
            gt_tags_file=args.gt_tags_file, gt_refs_file=args.gt_refs_file,
        )
        results = run_step2(step2_args)

        # ============== 状态更新 ==============
        if results and 'next_round_files' in results:
            nrf = results['next_round_files']
            current_labels_path     = nrf['labels']
            current_state_path      = nrf.get('state')
            current_checkpoint_path = step1_checkpoint
            print(f"\n✅ Round {iteration} 完成. 标签: {os.path.basename(current_labels_path)}")
        else:
            print("❌ Step 2 失败"); break

    print(f"\n🎉 实验完成！结果: {args.experiment_dir}/results/")


if __name__ == "__main__":
    main_loop()