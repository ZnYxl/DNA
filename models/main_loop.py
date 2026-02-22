# models/main_loop.py
"""
SSI-EC 闭环迭代总控 (v2 精简版)

v2 变更:
  [NEW] 迭代结束后调用 Post-processing: 全量距离分配 (消除所有 -1)
  [NEW] 最终评估: 完整指标体系 (ARI/NMI/Purity/Recovery/Recall@cluster)
  [NEW] 收敛性追踪: 每轮标签变化率

保留:
  [FIX] 预训练权重路径修正
  [FIX] --gt_tags_file 支持 GT 评估
  [FIX] training_cap 可配置

用法:
  # exp_1
  python main_loop.py \\
    --experiment_dir .../exp_1_Real \\
    --max_length 150 \\
    --gt_tags_file .../exp1_bwa_tags_reads.txt

  # id20
  python main_loop.py \\
    --experiment_dir .../id20_Real \\
    --max_length 150 \\
    --gt_tags_file .../id20_tags_reads.txt
"""
import os
import argparse
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.step1_train import train_step1
from models.step2_runner import run_step2


def compute_label_change_rate(prev_labels_path, curr_labels_path):
    """
    收敛性追踪: 计算两轮之间的标签变化率
    = Hamming(labels_t, labels_{t-1}) / N
    """
    if prev_labels_path is None or not os.path.exists(prev_labels_path):
        return None
    if curr_labels_path is None or not os.path.exists(curr_labels_path):
        return None

    try:
        prev = np.loadtxt(prev_labels_path, dtype=int)
        curr = np.loadtxt(curr_labels_path, dtype=int)

        if len(prev) != len(curr):
            return None

        # 只比较两轮都有有效标签的 reads
        valid = (prev >= 0) & (curr >= 0)
        if valid.sum() == 0:
            return None

        changed = (prev[valid] != curr[valid]).sum()
        rate = changed / valid.sum()
        return rate
    except Exception:
        return None


def main_loop():
    parser = argparse.ArgumentParser(description="SSI-EC Master Loop (v2)")

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

    # ===== 预训练权重 =====
    parser.add_argument('--feddna_checkpoint', type=str,
                        default='/mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth')

    # ===== 训练超参 =====
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_clusters_per_batch', type=int, default=64)
    parser.add_argument('--training_cap', type=int, default=999999999,
                        help="Step1 训练采样上限, 默认全量")
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--min_clusters', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    args = parser.parse_args()
    if args.gt_tags_file and args.gt_tags_file.lower() == 'none':
        args.gt_tags_file = None

    current_labels_path = None
    current_checkpoint_path = None
    current_state_path = None
    current_centroids_path = None

    # =====================================================================
    # 自动恢复: 扫描已完成的轮次, 从断点继续
    # =====================================================================
    start_iteration = 1
    results_dir = os.path.join(args.experiment_dir, "results")
    labels_dir = os.path.join(args.experiment_dir, "04_Iterative_Labels")

    for check_round in range(1, args.max_iterations + 1):
        step1_dir = os.path.join(results_dir, f"iter_{check_round}_step1", "models")
        step2_dir = os.path.join(results_dir, f"iter_{check_round}_step2")

        # 检查 Step 1 checkpoint
        ckpt_path = os.path.join(step1_dir, "step1_final_model.pth")
        if not os.path.exists(ckpt_path):
            break

        # 检查 Step 2 输出 (找 labels 和 state)
        if not os.path.exists(step2_dir):
            break

        # 找到最新的 labels/state/centroids
        import glob
        label_files = sorted(glob.glob(os.path.join(labels_dir, "refined_labels_*.txt")))
        state_files = sorted(glob.glob(os.path.join(labels_dir, "read_state_*.pt")))
        centroid_files = sorted(glob.glob(os.path.join(labels_dir, "centroids_*.pt")))

        if not label_files:
            break

        # 这一轮完整, 更新状态
        current_checkpoint_path = ckpt_path
        current_labels_path = label_files[-1]
        current_state_path = state_files[-1] if state_files else None
        current_centroids_path = centroid_files[-1] if centroid_files else None
        start_iteration = check_round + 1
        print(f"   ✅ 检测到 Round {check_round} 已完成:")
        print(f"      checkpoint: {os.path.basename(ckpt_path)}")
        print(f"      labels:     {os.path.basename(current_labels_path)}")

    if start_iteration > 1:
        print(f"\n⏩ 从 Round {start_iteration} 继续 (跳过已完成的 {start_iteration - 1} 轮)")
    else:
        print(f"\n🆕 从 Round 1 开始 (无已有结果)")

    # 收敛性追踪
    convergence_log = []

    print(f"🚀 SSI-EC 闭环迭代启动 (v2)")
    print(f"📂 实验目录: {args.experiment_dir}")
    print(f"📏 序列长度: {args.max_length} bp")
    print(f"🔁 迭代轮数: {args.max_iterations}")
    print(f"🔋 预训练:   {os.path.basename(args.feddna_checkpoint)}")
    if args.gt_tags_file:
        print(f"📋 GT 评估:  {os.path.basename(args.gt_tags_file)}")

    for iteration in range(start_iteration, args.max_iterations + 1):
        print(f"\n{'=' * 80}")
        print(f"🔄 Round {iteration} / {args.max_iterations}")
        print(f"{'=' * 80}\n")

        prev_labels_path = current_labels_path  # 保留上一轮标签用于收敛性计算

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
            training_cap=args.training_cap,
        )
        results = run_step2(step2_args)

        # ============== 状态更新 ==============
        if results and 'next_round_files' in results:
            nrf = results['next_round_files']
            current_labels_path = nrf['labels']
            current_state_path = nrf.get('state')
            current_centroids_path = nrf.get('centroids')
            current_checkpoint_path = step1_checkpoint
            print(f"\n✅ Round {iteration} 完成. 标签: {os.path.basename(current_labels_path)}")

            # 收敛性追踪
            change_rate = compute_label_change_rate(prev_labels_path, current_labels_path)
            if change_rate is not None:
                convergence_log.append({
                    'round': iteration,
                    'label_change_rate': change_rate
                })
                print(f"   📈 标签变化率: {change_rate:.4f} ({change_rate*100:.2f}%)")
            else:
                print(f"   📈 标签变化率: N/A (首轮)")
        else:
            print("❌ Step 2 失败"); break

    # =====================================================================
    # Post-processing: 全量距离分配
    # =====================================================================
    if current_labels_path and current_centroids_path and current_checkpoint_path:
        print(f"\n{'=' * 80}")
        print(f"🔧 Post-processing: 全量距离分配")
        print(f"{'=' * 80}")

        try:
            from models.post_process import post_process_final_assignment

            pp_output_dir = os.path.join(args.experiment_dir, "results", "final")
            final_labels_path = post_process_final_assignment(
                experiment_dir=args.experiment_dir,
                final_checkpoint_path=current_checkpoint_path,
                final_labels_path=current_labels_path,
                centroids_path=current_centroids_path,
                output_dir=pp_output_dir,
                device=args.device,
                dim=args.dim,
                max_length=args.max_length,
                gt_tags_file=args.gt_tags_file,
            )
            print(f"\n✅ 最终标签: {final_labels_path}")
        except Exception as e:
            print(f"❌ Post-processing 失败: {e}")
            import traceback
            traceback.print_exc()

    # =====================================================================
    # 收敛性报告
    # =====================================================================
    if convergence_log:
        print(f"\n{'=' * 60}")
        print(f"📈 收敛性报告")
        print(f"{'=' * 60}")
        for entry in convergence_log:
            r = entry['round']
            cr = entry['label_change_rate']
            bar = '█' * int(cr * 100) + '░' * (50 - int(cr * 100))
            print(f"   Round {r}: {cr:.4f} ({cr*100:.2f}%) {bar}")

        # 保存收敛性日志
        try:
            conv_path = os.path.join(args.experiment_dir, "results", "convergence_log.txt")
            os.makedirs(os.path.dirname(conv_path), exist_ok=True)
            with open(conv_path, 'w') as f:
                f.write("Round,Label_Change_Rate\n")
                for entry in convergence_log:
                    f.write(f"{entry['round']},{entry['label_change_rate']:.6f}\n")
            print(f"   💾 收敛性日志: {conv_path}")
        except Exception as e:
            print(f"   ⚠️ 保存收敛性日志失败: {e}")

    print(f"\n🎉 实验完成！结果: {args.experiment_dir}/results/")


if __name__ == "__main__":
    main_loop()