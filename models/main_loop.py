# models/main_loop.py
"""
SSI-EC main loop controller.

Iterative flow: each round runs Step1 (evidential training) -> Step2 (intra-cluster
split + consensus); consensus and refined labels are fed back into the next round,
for max_iterations rounds.

Supports resume-from-checkpoint (locating the latest artifacts by file mtime) and
convergence tracking (label change rate).
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
    """Compute the label change rate between two rounds (convergence metric)."""
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
        print(f"   [warn] failed to compute change rate: {e}")
        return None


def main_loop():
    parser = argparse.ArgumentParser(description='SSI-EC main loop')
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
    parser.add_argument('--cv_threshold',       type=float, default=0.3,
                        help='hard-cluster CV threshold, default 0.3')
    parser.add_argument('--skip_step1_round',   type=int,   default=0,
                        help='skip Step 1 for the given round, start directly from Step 2')
    parser.add_argument('--prev_checkpoint',    type=str,   default=None,
                        help='manually specify the checkpoint path when skipping Step 1 in round 1')
    parser.add_argument('--cl_mode',            type=str,   default='ours',
                        choices=['standard', 'ale_only', 'epi_only', 'ours'],
                        help='contrastive-learning ablation mode: standard=plain InfoNCE, '
                             'ale_only=U_ale only, epi_only=U_epi only, ours=full design (default)')
    parser.add_argument('--max_reads_per_cluster', type=int, default=30,
                        help='max reads sampled per cluster during training (consistent with '
                             'FedDNA, recommended 30). Affects Step1 training only, not Step2 '
                             'inference (which is full-scale).')
    parser.add_argument('--ref_length', type=int, default=None,
                        help='reference sequence length (bp), for consensus truncation. Set 196 for Seq_1D')
    # Intra-cluster split engine parameters (the only iterative mechanism)
    parser.add_argument('--split_tau', type=int, default=5,
                        help='split gate threshold: split only when the two sub-cluster consensuses have edit distance >= tau.')
    parser.add_argument('--split_min_size', type=int, default=6,
                        help='do not attempt to split clusters with fewer reads than this.')
    # Evidential multiplicative split criterion (DEFAULT = ON; this is the paper's
    # main method). Use --no_split_evidential to fall back to the pure-edit baseline
    # (dAB>=tau), which is the ablation in the paper.
    parser.add_argument('--split_evidential', dest='split_evidential',
                        action='store_true', default=True,
                        help='[default] use the multiplicative evidential split '
                             'criterion dAB*min(S_A,S_B)/S_ref>=tau (paper main method).')
    parser.add_argument('--no_split_evidential', dest='split_evidential',
                        action='store_false',
                        help='fall back to the pure-edit split criterion dAB>=tau '
                             '(ablation baseline).')
    parser.add_argument('--split_tau_evidential', type=int, default=5,
                        help='gate threshold for the multiplicative evidential criterion (default 5).')

    args = parser.parse_args()

    os.makedirs(os.path.join(args.experiment_dir, 'results'), exist_ok=True)

    # =====================================================================
    # Resume-from-checkpoint detection
    # =====================================================================
    results_dir = os.path.join(args.experiment_dir, 'results')
    labels_dir  = os.path.join(args.experiment_dir, '04_Iterative_Labels')

    current_checkpoint_path     = None
    current_labels_path         = None
    current_state_path          = None
    current_consensus_path      = None
    current_cluster_change_info = None
    start_iteration = 1

    if os.path.exists(results_dir):
        for check_round in range(1, args.max_iterations + 1):
            step1_dir = os.path.join(results_dir, f"iter_{check_round}_step1")
            step2_dir = os.path.join(results_dir, f"iter_{check_round}_step2")

            ckpt_path = os.path.join(step1_dir, "models", "step1_final_model.pth")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(step1_dir, "step1_final_model.pth")
                if not os.path.exists(ckpt_path):
                    break

            if not os.path.exists(step2_dir):
                # Step1 done but Step2 not run: force-skip this round's Step1, resume at Step2
                current_checkpoint_path = ckpt_path
                start_iteration = check_round
                args.skip_step1_round = check_round
                print(f"   [resume] Round {check_round} Step1 complete, Step2 not yet run")
                print(f"   [resume] auto-setting skip_step1_round={check_round}, will run Step2 directly")
                break

            # Sort by file mtime (avoids %H%M%S lexicographic disorder across days)
            label_files     = sorted(glob.glob(os.path.join(labels_dir, "refined_labels_*.txt")),     key=os.path.getmtime)
            state_files     = sorted(glob.glob(os.path.join(labels_dir, "read_state_*.pt")),          key=os.path.getmtime)
            consensus_files = sorted(glob.glob(os.path.join(labels_dir, "consensus_dict_*.pt")),      key=os.path.getmtime)
            change_files    = sorted(glob.glob(os.path.join(labels_dir, "cluster_change_info_*.pt")), key=os.path.getmtime)

            if not label_files:
                break

            current_checkpoint_path = ckpt_path
            current_labels_path     = label_files[-1]
            current_state_path      = state_files[-1] if state_files else None
            current_consensus_path  = consensus_files[-1] if consensus_files else None
            if change_files:
                try:
                    current_cluster_change_info = torch.load(change_files[-1], map_location='cpu')
                except Exception as e:
                    print(f"   [warn] failed to load cluster_change_info ({e}), falling back to no-difficulty sampling")
                    current_cluster_change_info = None

            start_iteration = check_round + 1
            print(f"   [resume] Round {check_round} complete")

            if args.prev_checkpoint and os.path.exists(args.prev_checkpoint):
                current_checkpoint_path = args.prev_checkpoint
                print(f"   [resume] using manually specified checkpoint: {current_checkpoint_path}")

    if start_iteration > 1:
        print(f"\n[resume] continuing from Round {start_iteration} (skipping {start_iteration - 1} completed round(s))")
    else:
        print(f"\n[start] beginning from Round 1 (no existing results)")

    convergence_log = []

    print(f"SSI-EC closed-loop iteration starting")
    print(f"  experiment dir: {args.experiment_dir}")
    print(f"  sequence length: {args.max_length} bp")
    print(f"  iterations: {args.max_iterations}")
    print(f"  pretrained: {os.path.basename(args.feddna_checkpoint)}")
    if args.gt_tags_file:
        print(f"  GT eval: {os.path.basename(args.gt_tags_file)}")

    for iteration in range(start_iteration, args.max_iterations + 1):
        print(f"\n{'=' * 80}")
        print(f"Round {iteration} / {args.max_iterations}")
        print(f"{'=' * 80}\n")

        prev_labels_path = current_labels_path

        # ============== Step 1 ==============
        if args.skip_step1_round == iteration:
            ckpt_name = os.path.basename(current_checkpoint_path) if current_checkpoint_path else "model not found (None)"
            print(f"[Step 1] skipping Round {iteration} Step 1 (using existing checkpoint: {ckpt_name})")
            step1_checkpoint = current_checkpoint_path
            if step1_checkpoint is None:
                print("[error] skip_step1_round set but current_checkpoint_path is empty, check the path")
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
                print("[error] Step 1 failed"); break

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
            consensus_path=current_consensus_path,
            cv_threshold=getattr(args, 'cv_threshold', 0.3),
            ref_length=getattr(args, 'ref_length', None),
            split_tau=getattr(args, 'split_tau', 5),
            split_min_size=getattr(args, 'split_min_size', 6),
            # Evidential multiplicative criterion passthrough (default ON = paper main method)
            split_evidential=getattr(args, 'split_evidential', True),
            split_tau_evidential=getattr(args, 'split_tau_evidential', 5),
        )
        results = run_step2(step2_args)

        # ============== State update ==============
        if results and 'next_round_files' in results:
            nrf = results['next_round_files']
            current_labels_path     = nrf['labels']
            current_state_path      = nrf.get('state')
            current_consensus_path  = nrf.get('consensus')
            current_checkpoint_path = step1_checkpoint

            change_info_path = nrf.get('cluster_change_info')
            if change_info_path and os.path.exists(change_info_path):
                try:
                    current_cluster_change_info = torch.load(change_info_path, map_location='cpu')
                except Exception as e:
                    print(f"   [warn] failed to load cluster_change_info: {e}")
                    current_cluster_change_info = None
            else:
                current_cluster_change_info = None

            print(f"\n[done] Round {iteration} complete.")
            print(f"   labels: {os.path.basename(current_labels_path)}")
            if current_consensus_path:
                print(f"   consensus: {os.path.basename(current_consensus_path)}")

            change_rate = compute_label_change_rate(prev_labels_path, current_labels_path)
            if change_rate is not None:
                convergence_log.append({'round': iteration, 'label_change_rate': change_rate})
                print(f"   label change rate: {change_rate:.4f} ({change_rate*100:.2f}%)")
            else:
                print(f"   label change rate: N/A (first round)")
        else:
            print("[error] Step 2 failed"); break

    # =====================================================================
    # Convergence report
    # =====================================================================
    if convergence_log:
        print(f"\n{'=' * 60}")
        print(f"Convergence report")
        print(f"{'=' * 60}")
        for entry in convergence_log:
            r = entry['round']
            cr = entry['label_change_rate']
            bar = '#' * min(50, int(cr * 100)) + '.' * max(0, 50 - int(cr * 100))
            print(f"   Round {r}: {cr:.4f} ({cr*100:.2f}%) {bar}")
        try:
            conv_path = os.path.join(args.experiment_dir, "results", "convergence_log.txt")
            os.makedirs(os.path.dirname(conv_path), exist_ok=True)
            with open(conv_path, 'w') as f:
                f.write("Round,Label_Change_Rate\n")
                for entry in convergence_log:
                    f.write(f"{entry['round']},{entry['label_change_rate']:.6f}\n")
            print(f"   convergence log: {conv_path}")
        except Exception as e:
            print(f"   [warn] failed to save convergence log: {e}")

    print(f"\n[complete] experiment finished. results: {args.experiment_dir}/results/")


if __name__ == "__main__":
    main_loop()