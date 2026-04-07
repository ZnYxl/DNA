# models/main_loop.py
"""
SSI-EC 主循环控制

修复清单:
  [FIX-P0]  在 Round 间传递 consensus_path 和 cluster_change_info
             - step1_args 新增 consensus_path, cluster_change_info
             - 从 results['next_round_files'] 提取两个新键
  [v2]      断点续跑支持
  [v2]      收敛性追踪

G老师审查修复清单:
  [G老师-Bug1-FIX] 断点续跑时 Step1 未被真正跳过
    原来：break 前没有设置 args.skip_step1_round，注释说"会被跳过"但条件永远为 False，
          重启程序后 Step1 照常执行，checkpoint 被覆盖。
    修复：检测到 Step1 完成 Step2 未跑时，自动执行
          args.skip_step1_round = check_round，强制触发跳过逻辑。

  [G老师-Bug2-FIX] 跨天运行时文件排序错乱（Time Bomb）
    原来：sorted() 字典序排序，时间戳格式 "%H%M%S" 跨天后
          001000 < 235500，断点续跑拿到旧文件，EM 迭代回滚。
    修复：改用 key=os.path.getmtime 按实际修改时间排序。

  [G老师-Bug3-FIX] 断点续跑路径的异常静默吞掉
    原来：except Exception: current_cluster_change_info = None，
          文件损坏时无任何提示，Curriculum Learning 悄悄失效。
    修复：except Exception as e: print(...)，日志留下痕迹。
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
                        help='困难簇 CV 阈值，默认 0.3')
    parser.add_argument('--skip_step1_round',   type=int,   default=0,
                        help='跳过指定轮次的 Step 1，直接从 Step 2 开始')
    parser.add_argument('--prev_checkpoint',    type=str,   default=None,
                        help='手动指定第一轮跳过 Step 1 时的 checkpoint 路径')
    parser.add_argument('--cl_mode',            type=str,   default='ours',
                        choices=['standard', 'ale_only', 'epi_only', 'ours'],
                        help='对比学习消融模式: standard=标准InfoNCE, ale_only=只用U_ale, '
                             'epi_only=只用U_epi, ours=完整设计(默认)')
    parser.add_argument('--max_reads_per_cluster', type=int, default=30,
                        help='训练时每簇最多采样的 reads 数，0=不限制。'
                             '与 FedDNA 训练设计一致（FedDNA 用 5-30），'
                             '建议 30（与feddna保持一致）。'
                             '仅影响 Step1 训练，不影响 Step2 推理（全量）。')
    parser.add_argument('--target_clusters', type=int, default=None,
                        help='最终目标簇数（先验约束）。系统会自动做课程合并：'
                             'Round 1 目标 = (初始簇数+target)/2，'
                             'Round 2 目标 = (Round1结果+target)/2，'
                             '最后一轮直达 target。不设则无限制。')
    parser.add_argument('--primer_prefix', type=int, default=0,
                        help='前端引物长度 (bp)，Jaccard 校验时截掉。Seq_1D 设 20')
    parser.add_argument('--primer_suffix', type=int, default=0,
                        help='后端引物长度 (bp)，Jaccard 校验时截掉。Seq_1D 设 20')
    parser.add_argument('--freeze_consensus', action='store_true', default=False,
                        help='[实验2] 所有轮次的 Step1 训练目标始终用 ref.txt，'
                             '不用上一轮 Step2 产出的 consensus。用于诊断 B 层毒化。')
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

    if os.path.exists(results_dir):
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
                # step1 完成但 step2 未跑（典型：中断后重跑场景）
                current_checkpoint_path = ckpt_path
                start_iteration = check_round
                # [G老师-Bug1-FIX] 原注释写"step1 会被 skip_step1_round 跳过"，
                # 但 args.skip_step1_round 默认值是 0，不加命令行参数时
                # 条件 args.skip_step1_round == check_round 永远 False，
                # Step1 照常执行，辛苦跑好的 checkpoint 被无声覆盖。
                # 修复：断点续跑检测到此情况时直接修改 args，强制触发跳过逻辑。
                args.skip_step1_round = check_round
                print(f"   ✅ 检测到 Round {check_round} Step1 已完成，Step2 未跑")
                print(f"   ⚡ 自动设置 skip_step1_round={check_round}，将直接运行 Step2")
                break

            # [G老师-Bug2-FIX] 原来用 sorted() 字典序排序，但时间戳格式是 "%H%M%S"
            # （只有时分秒，无日期）。跨天运行时 001000 < 235500，导致断点续跑
            # 拿到的是上一轮的旧文件，EM 迭代回滚。
            # 修复：改用 os.path.getmtime 按文件实际修改时间排序，与文件名无关。
            label_files    = sorted(glob.glob(os.path.join(labels_dir, "refined_labels_*.txt")),    key=os.path.getmtime)
            state_files    = sorted(glob.glob(os.path.join(labels_dir, "read_state_*.pt")),         key=os.path.getmtime)
            centroid_files = sorted(glob.glob(os.path.join(labels_dir, "centroids_*.pt")),          key=os.path.getmtime)
            consensus_files= sorted(glob.glob(os.path.join(labels_dir, "consensus_dict_*.pt")),     key=os.path.getmtime)
            change_files   = sorted(glob.glob(os.path.join(labels_dir, "cluster_change_info_*.pt")),key=os.path.getmtime)

            if not label_files:
                break

            current_checkpoint_path = ckpt_path
            current_labels_path     = label_files[-1]
            current_state_path      = state_files[-1] if state_files else None
            current_centroids_path  = centroid_files[-1] if centroid_files else None

            # [FIX-P0] 加载 consensus_path 和 cluster_change_info
            if consensus_files and not getattr(args, 'freeze_consensus', False):
                current_consensus_path = consensus_files[-1]
            if change_files:
                try:
                    current_cluster_change_info = torch.load(change_files[-1], map_location='cpu')
                except Exception as e:
                    # [G老师-Bug3-FIX] 原来静默吞掉异常，文件损坏时程序不报错，
                    # 悄悄回退到全量采样，破坏 Curriculum Learning 策略。
                    # 修复：至少打印异常，让日志留下痕迹。
                    print(f"   ⚠️ 加载 cluster_change_info 失败 ({e})，将回退到无难度采样模式")
                    current_cluster_change_info = None

            start_iteration = check_round + 1
            print(f"   ✅ 检测到 Round {check_round} 已完成")

            if args.prev_checkpoint and os.path.exists(args.prev_checkpoint):
                current_checkpoint_path = args.prev_checkpoint
                print(f"   ⚡ 已捕获手动指定的 checkpoint: {current_checkpoint_path}")

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
    if args.freeze_consensus:
        print(f"🧊 [实验2] freeze_consensus=True: 所有轮次 Step1 训练目标固定为 ref.txt")

    for iteration in range(start_iteration, args.max_iterations + 1):
        print(f"\n{'=' * 80}")
        print(f"🔄 Round {iteration} / {args.max_iterations}")
        if args.target_clusters:
            print(f"   🎯 最终目标簇数: {args.target_clusters} (动态课程合并，每轮合半)")
        print(f"{'=' * 80}\n")

        prev_labels_path = current_labels_path

        # ============== Step 1 ==============
        if args.skip_step1_round == iteration:
            ckpt_name = os.path.basename(current_checkpoint_path) if current_checkpoint_path else "未找到模型(None)"
            print(f"[Step 1] 跳过 Round {iteration} Step 1（使用已有 checkpoint: {ckpt_name}）")
            
            step1_checkpoint = current_checkpoint_path
            if step1_checkpoint is None:
                print("❌ skip_step1_round 指定跳过，但 current_checkpoint_path 为空，请检查路径")
                break
        else:
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
                cl_mode=args.cl_mode,
                cv_threshold=getattr(args, 'cv_threshold', 0.3),
                consensus_path=current_consensus_path,
                cluster_change_info=current_cluster_change_info,
                max_reads_per_cluster=getattr(args, 'max_reads_per_cluster', 30),
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
            batch_size=args.batch_size,
            device=args.device,
            round_idx=iteration,
            refined_labels=current_labels_path,
            prev_state=current_state_path,
            gt_tags_file=args.gt_tags_file,
            gt_refs_file=args.gt_refs_file,
            training_cap=args.training_cap,
            consensus_path=current_consensus_path,
            cv_threshold=getattr(args, 'cv_threshold', 0.3),
            target_clusters=args.target_clusters,       # 最终目标，step2_runner 内部动态计算本轮目标
            max_iterations=args.max_iterations,          # step2_runner 需要知道是否为最后一轮
            primer_prefix=getattr(args, 'primer_prefix', 0),
            primer_suffix=getattr(args, 'primer_suffix', 0),
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
            if not getattr(args, 'freeze_consensus', False):
                current_consensus_path = nrf.get('consensus')
            else:
                # [实验2] freeze_consensus: Step1 始终用 ref.txt，不更新 consensus_path
                pass
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