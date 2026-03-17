# models/main_loop.py
"""
SSI-EC 主循环控制

修复清单:
  [FIX-P0]  在 Round 间传递 consensus_path 和 cluster_change_info
             - step1_args 新增 consensus_path, cluster_change_info
             - 从 results['next_round_files'] 提取两个新键
  [v2]      断点续跑支持
  [v2]      收敛性追踪
"""
import argparse
import os
import sys
import glob
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.step1_train import train_step1
from models.step2_runner import run_step2


def compute_label_change_rate(prev_labels_path, curr_labels_path):
    """计算两轮间的标签变化率"""
    if prev_labels_path is None or not os.path.exists(prev_labels_path):
        return None
    if not os.path.exists(curr_labels_path):
        return None
    try:
        prev = np.loadtxt(prev_labels_path, dtype=int)
        curr = np.loadtxt(curr_labels_path, dtype=int)
        valid = (prev >= 0) | (curr >= 0)
        if valid.sum() == 0:
            return 0.0
        changed = (prev[valid] != curr[valid]).sum()
        return float(changed) / float(valid.sum())
    except Exception as e:
        print(f"   ⚠️ 变化率计算失败: {e}")
        return None


def main_loop():
    parser = argparse.ArgumentParser(description='SSI-EC 主循环')
    parser.add_argument('--experiment_dir',     type=str, required=True)
    parser.add_argument('--feddna_checkpoint',  type=str, required=True)
    parser.add_argument('--max_iterations',     type=int, default=3)
    parser.add_argument('--dim',                type=int, default=256)
    parser.add_argument('--max_length',         type=int, default=150)
    parser.add_argument('--batch_size',         type=int, default=256)
    parser.add_argument('--max_clusters_per_batch', type=int, default=64)
    parser.add_argument('--weight_decay',       type=float, default=1e-4)
    parser.add_argument('--min_clusters',       type=int, default=50)
    parser.add_argument('--device',             type=str, default='cuda')
    parser.add_argument('--training_cap',       type=int, default=9999000000)
    parser.add_argument('--gt_tags_file',       type=str, default=None)
    parser.add_argument('--gt_refs_file',       type=str, default=None)
    parser.add_argument('--cv_threshold',       type=float, default=0.3,
                        help='困难簇 CV 阈值，默认 0.3，第一轮跑完后根据日志中位CV调整')
    parser.add_argument('--cl_mode',            type=str, default='ours',
                        choices=['standard', 'ale_only', 'epi_only', 'ours'],
                        help='对比学习消融模式: standard=标准InfoNCE, ale_only=只用U_ale, '
                             'epi_only=只用U_epi, ours=完整设计(默认)')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.experiment_dir, 'results'), exist_ok=True)

    # =====================================================================
    # 断点续跑检测
    # =====================================================================
    results_dir = os.path.join(args.experiment_dir, 'results')
    labels_dir  = os.path.join(args.experiment_dir, '04_Iterative_Labels')

    current_checkpoint_path    = None
    current_labels_path        = None
    current_state_path         = None
    current_centroids_path     = None
    current_consensus_path     = None        # [FIX-P0] 新增
    current_cluster_change_info = None       # [FIX-P0] 新增（内存中的 dict，不是路径）
    start_iteration = 1

    if os.path.exists(labels_dir):
        for check_round in range(1, args.max_iterations + 1):
            step1_dir = os.path.join(results_dir, f"iter_{check_round}_step1")
            step2_dir = os.path.join(results_dir, f"iter_{check_round}_step2")

            ckpt_path = os.path.join(step1_dir, "models", "step1_final_model.pth")
            if not os.path.exists(ckpt_path):
                # 兼容旧路径格式
                ckpt_path = os.path.join(step1_dir, "step1_final_model.pth")
                if not os.path.exists(ckpt_path):
                    break

            if not os.path.exists(step2_dir):
                break

            label_files   = sorted(glob.glob(os.path.join(labels_dir, "refined_labels_*.txt")))
            state_files   = sorted(glob.glob(os.path.join(labels_dir, "read_state_*.pt")))
            centroid_files= sorted(glob.glob(os.path.join(labels_dir, "centroids_*.pt")))
            consensus_files = sorted(glob.glob(os.path.join(labels_dir, "consensus_dict_*.pt")))  # [FIX-P0]
            change_files  = sorted(glob.glob(os.path.join(labels_dir, "cluster_change_info_*.pt")))  # [FIX-P0]

            if not label_files:
                break

            current_checkpoint_path = ckpt_path
            current_labels_path     = label_files[-1]
            current_state_path      = state_files[-1] if state_files else None
            current_centroids_path  = centroid_files[-1] if centroid_files else None

            # [FIX-P0] 加载 consensus_path 和 cluster_change_info
            if consensus_files:
                current_consensus_path = consensus_files[-1]
            if change_files:
                try:
                    current_cluster_change_info = torch.load(change_files[-1], map_location='cpu')
                except Exception:
                    current_cluster_change_info = None

            start_iteration = check_round + 1
            print(f"   ✅ 检测到 Round {check_round} 已完成")

    if start_iteration > 1:
        print(f"\n⏩ 从 Round {start_iteration} 继续 (跳过已完成的 {start_iteration - 1} 轮)")
    else:
        print(f"\n🆕 从 Round 1 开始 (无已有结果)")

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

        prev_labels_path = current_labels_path

        # ============== Step 1 ==============
        print(f"[Step 1] Evidence Learning...")
        step1_out = os.path.join(args.experiment_dir, "results", f"iter_{iteration}_step1")

        step1_args = argparse.Namespace(
            experiment_dir=args.experiment_dir,
            output_dir=step1_out,
            batch_size=args.batch_size,
            max_clusters_per_batch=args.max_clusters_per_batch,
            weight_decay=args.weight_decay,
            dim=args.dim,
            max_length=args.max_length,
            min_clusters=args.min_clusters,
            device=args.device,
            round_idx=iteration,
            feddna_checkpoint=args.feddna_checkpoint,
            prev_checkpoint=current_checkpoint_path,
            refined_labels=current_labels_path,
            prev_state=current_state_path,
            training_cap=args.training_cap,
            cl_mode=args.cl_mode,                   # 消融实验 flag
            cv_threshold=args.cv_threshold,         # 困难簇 CV 阈值
            # [FIX-P0] 新增：传递 consensus_path 和 cluster_change_info
            consensus_path=current_consensus_path,
            cluster_change_info=current_cluster_change_info,
        )
        step1_checkpoint = train_step1(step1_args)
        if step1_checkpoint is None:
            print("❌ Step 1 失败"); break

        # ============== Step 2 ==============
        print(f"\n[Step 2] Refine & Decode...")
        step2_out = os.path.join(args.experiment_dir, "results", f"iter_{iteration}_step2")

        step2_args = argparse.Namespace(
            experiment_dir=args.experiment_dir,
            step1_checkpoint=step1_checkpoint,
            output_dir=step2_out,
            dim=args.dim,
            max_length=args.max_length,
            batch_size=args.batch_size,          # step2 推理 DataLoader 需要
            device=args.device,
            round_idx=iteration,
            refined_labels=current_labels_path,
            prev_state=current_state_path,
            gt_tags_file=args.gt_tags_file,
            gt_refs_file=args.gt_refs_file,
            training_cap=args.training_cap,
            cv_threshold=args.cv_threshold,         # 困难簇 CV 阈值
        )
        results = run_step2(step2_args)

        # ============== 状态更新 ==============
        if results and 'next_round_files' in results:
            nrf = results['next_round_files']
            current_labels_path    = nrf['labels']
            current_state_path     = nrf.get('state')
            current_centroids_path = nrf.get('centroids')
            current_checkpoint_path = step1_checkpoint

            # [FIX-P0] 更新 consensus_path 和 cluster_change_info
            current_consensus_path = nrf.get('consensus')
            change_info_path = nrf.get('cluster_change_info')
            if change_info_path and os.path.exists(change_info_path):
                try:
                    current_cluster_change_info = torch.load(change_info_path, map_location='cpu')
                except Exception as e:
                    print(f"   ⚠️ 加载 cluster_change_info 失败: {e}")
                    current_cluster_change_info = None
            else:
                current_cluster_change_info = None

            print(f"\n✅ Round {iteration} 完成.")
            print(f"   标签: {os.path.basename(current_labels_path)}")
            if current_consensus_path:
                print(f"   Consensus: {os.path.basename(current_consensus_path)}")

            change_rate = compute_label_change_rate(prev_labels_path, current_labels_path)
            if change_rate is not None:
                convergence_log.append({'round': iteration, 'label_change_rate': change_rate})
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
            bar = '█' * min(50, int(cr * 100)) + '░' * max(0, 50 - int(cr * 100))
            print(f"   Round {r}: {cr:.4f} ({cr*100:.2f}%) {bar}")

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