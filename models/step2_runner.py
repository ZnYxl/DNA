# models/step2_runner.py
"""
Step2: clustering refinement & consensus decoding.

Terminal pipeline (single line, no branches):
    load model and labels
      -> full-scale inference (embedding + evidential uncertainty u_epi/u_ale/strength)
      -> intra-cluster split (the only label-changing iterative engine)
      -> MV consensus (training target + FASTA evaluation)
      -> dump (labels / u_epi,u_ale,strength / consensus_dict / cluster_change_info)

The intra-cluster split is the framework's sole iterative mechanism: for each
cluster, hierarchically bisect by edit distance, compute an MV consensus per
sub-cluster, and split when the two consensuses have edit distance >= tau. Pure
clusters bisect into near-identical halves and fail the tau gate, so they are
protected by construction.

The split criterion folds in evidential uncertainty by default (multiplicative):
with use_evidential=True a split requires dAB*min(S_A,S_B)/S_ref >= tau', suppressing
splits when evidence is weak. This is the paper's main method. Switch:
--no_split_evidential falls back to the pure-edit baseline (dAB>=tau, the ablation).

Uncertainty (u_epi epistemic / u_ale aleatoric) comes from the Dirichlet evidence;
it drives Step1's uncertainty-weighted contrastive loss and is dumped for analysis.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import gc
import sys
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.step1_model import Step1EvidentialModel, decompose_uncertainty
from models.step1_data  import CloverDataLoader, Step1Dataset
from models.step2_decode import compute_mv_consensus, save_consensus_fasta
from models.cluster_split import split_clusters
from models.eval_reconstruction import levenshtein as _edit_distance


# ===========================================================================
# Cluster difficulty (CV of strength): lets Step1 distinguish hard/easy clusters
# for dynamic sampling
# ===========================================================================
def compute_cluster_difficulty(new_labels_np, strength_np) -> Dict[int, float]:
    """Cluster difficulty = coefficient of variation (std/mean) of intra-cluster
    strength.
      - pure cluster: homologous reads, strength high and concentrated -> low CV
      - mixed cluster: heterologous reads, strength scattered          -> high CV
    """
    cluster_strengths: Dict[int, list] = defaultdict(list)
    for didx, label in enumerate(new_labels_np):
        if label >= 0:
            cluster_strengths[int(label)].append(float(strength_np[didx]))

    difficulty: Dict[int, float] = {}
    for cid, strengths in cluster_strengths.items():
        if len(strengths) < 2:
            difficulty[cid] = 0.0
            continue
        arr = np.array(strengths)
        difficulty[cid] = float(arr.std() / (arr.mean() + 1e-6))
    return difficulty


# ===========================================================================
# GT clustering evaluation (Purity / Perfect Cluster Rate) -- observation only,
# not part of the iteration
# ===========================================================================
def _evaluate_with_gt(gt_labels_list, new_labels, flat_real_indices, output_dir):
    try:
        gt_arr  = np.array(gt_labels_list)
        new_arr = new_labels.cpu().numpy() if isinstance(new_labels, torch.Tensor) else new_labels

        cluster_gt_counter = defaultdict(Counter)
        for new_l, real_idx in zip(new_arr, flat_real_indices):
            if new_l < 0:
                continue
            gt = int(gt_arr[real_idx]) if real_idx < len(gt_arr) else -1
            if gt >= 0:
                cluster_gt_counter[int(new_l)][gt] += 1

        total_weighted = total_reads_eval = perfect_clusters = 0
        total_clusters = len(cluster_gt_counter)
        for counter in cluster_gt_counter.values():
            size = sum(counter.values())
            majority = counter.most_common(1)[0][1]
            total_weighted   += majority
            total_reads_eval += size
            if majority == size:
                perfect_clusters += 1

        purity       = total_weighted / max(total_reads_eval, 1)
        perfect_rate = perfect_clusters / max(total_clusters, 1)
        print(f"\n   GT clustering eval (over {total_reads_eval:,} reads, {total_clusters:,} clusters):")
        print(f"      Cluster Purity:       {purity:.4f}  ({total_weighted:,}/{total_reads_eval:,})")
        print(f"      Perfect Cluster Rate: {perfect_clusters}/{total_clusters} ({perfect_rate:.4f})  (upper bound on reconstruction success)")

        try:
            log_dir = os.path.join(output_dir, "paper_logs")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "gt_eval.txt"), 'a') as f:
                f.write(f"purity={purity:.4f}, perfect_rate={perfect_rate:.4f}, "
                        f"perfect={perfect_clusters}, total={total_clusters}\n")
        except Exception:
            pass
    except Exception as e:
        print(f"   [warn] GT eval failed: {e}")


def _record_paper_log(args, total_reads, consensus_dict, avg_strength):
    try:
        log_dir = os.path.join(args.output_dir, "paper_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"round{args.round_idx}_stats.txt")
        with open(log_path, 'w') as f:
            f.write(f"Round: {args.round_idx}\n")
            f.write(f"Total Reads: {total_reads}\n")
            f.write(f"Avg Strength: {avg_strength:.4f}\n")
            f.write(f"Consensus Clusters: {len(consensus_dict)}\n")
        print(f"   paper log: {log_path}")
    except Exception as e:
        print(f"   [warn] paper log failed: {e}")


# ===========================================================================
# Main pipeline
# ===========================================================================
def run_step2(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    round_idx = getattr(args, 'round_idx', 1)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model and data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Loading model and data")
    print("=" * 60)

    try:
        checkpoint = torch.load(args.step1_checkpoint, map_location=device)
        step1_args = checkpoint.get('args', {})
        model_dim     = step1_args.get('dim', args.dim)
        model_max_len = step1_args.get('max_length', args.max_length)
        print(f"   model params: dim={model_dim}, max_len={model_max_len}")
    except Exception as e:
        print(f"   [error] checkpoint load failed: {e}"); return None

    try:
        labels_path = getattr(args, 'refined_labels', None)
        data_loader = CloverDataLoader(args.experiment_dir, labels_path=labels_path)
        TOTAL_READS = len(data_loader.reads)
        current_clusters = set(l for l in data_loader.clover_labels if l >= 0)
        num_clusters = max(50, len(current_clusters))
        print(f"   data: {TOTAL_READS} reads, {len(current_clusters)} valid clusters")
    except Exception as e:
        print(f"   [error] data load failed: {e}"); return None

    gt_tags_file = getattr(args, 'gt_tags_file', None)
    if gt_tags_file and os.path.exists(gt_tags_file):
        data_loader.load_gt_tags(gt_tags_file)

    model = Step1EvidentialModel(
        dim=model_dim, max_length=model_max_len,
        num_clusters=num_clusters, device=str(device)
    ).to(device)
    sd = checkpoint['model_state_dict']
    if 'length_adapter.weight' in sd:
        sh = sd['length_adapter.weight'].shape
        if sh[1] == model_max_len and sh[0] == model_max_len:
            model.length_adapter = nn.Linear(sh[1], sh[0]).to(device)
            print(f"   restored length_adapter: Linear({sh[1]}, {sh[0]})")
        else:
            print(f"   [warn] checkpoint length_adapter dim {sh} "
                  f"incompatible with max_length={model_max_len}, skipped")
    model.load_state_dict(sd, strict=False)
    model.eval()
    del checkpoint, sd
    gc.collect()

    # ------------------------------------------------------------------
    # 2. Full-scale inference: embedding + evidential uncertainty
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Inference (embeddings + uncertainty)")
    print("=" * 60)

    dataset = Step1Dataset(data_loader, max_len=model_max_len, inference_mode=True)
    print(f"   full-scale inference: {TOTAL_READS} reads")

    inference_loader = torch.utils.data.DataLoader(
        dataset, batch_size=getattr(args, 'batch_size', 1024),
        shuffle=False, num_workers=0, pin_memory=False
    )

    N = len(dataset)
    # Keep only pooled emb (N, dim); sequence-level emb is too large (~210GB), so the
    # encoder is re-run during consensus decoding.
    strength = torch.zeros(N)
    u_epi    = torch.zeros(N)
    u_ale    = torch.zeros(N)
    flat_real_indices = [dataset.valid_indices[i] for i in range(N)]

    ptr = 0
    with torch.no_grad():
        for batch_data in inference_loader:
            reads_batch = batch_data['encoding'].to(device)
            bs = reads_batch.shape[0]
            emb, pooled = model.encode_reads(reads_batch)
            ev, stre, alpha = model.decode_to_evidence(emb)
            u_e, u_a = decompose_uncertainty(alpha)
            strength[ptr:ptr+bs] = stre.mean(dim=-1).cpu()
            u_epi[ptr:ptr+bs]    = u_e.cpu()
            u_ale[ptr:ptr+bs]    = u_a.cpu()
            ptr += bs
            del reads_batch, emb, pooled, ev, stre, alpha, u_e, u_a
            if ptr % 100000 < 1024:
                torch.cuda.empty_cache()

    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"   inference done: {N} samples")

    _np_u_epi    = u_epi.numpy().copy()
    _np_u_ale    = u_ale.numpy().copy()
    _np_strength = strength.numpy().copy()
    _avg_strength = float(_np_strength.mean())

    # Current labels (Clover initial or previous round's refined)
    labels_tensor = torch.tensor(
        [data_loader.clover_labels[flat_real_indices[i]] for i in range(N)],
        dtype=torch.long
    )
    new_labels_np = labels_tensor.cpu().numpy().copy()

    # ------------------------------------------------------------------
    # 3. Intra-cluster split (the only iterative engine)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Intra-cluster split (sole iterative engine)")
    print("=" * 60)

    _split_tau      = getattr(args, 'split_tau', 5)
    _split_min_size = getattr(args, 'split_min_size', 6)
    _split_ref_len  = getattr(args, 'ref_length', None) or 196
    # Evidential multiplicative criterion switch (default ON = paper main method;
    # --no_split_evidential falls back to pure-edit for the ablation)
    _split_evidential = getattr(args, 'split_evidential', True)
    _split_tau_evid   = getattr(args, 'split_tau_evidential', 5)
    new_labels_np, _split_stats = split_clusters(
        new_labels_np=new_labels_np,
        flat_real_indices=flat_real_indices,
        data_loader=data_loader,
        levenshtein=_edit_distance,
        ref_length=_split_ref_len,
        tau=_split_tau,
        min_split_size=_split_min_size,
        verbose=True,
        # strength_np is in didx space, same index as new_labels_np
        strength_np=_np_strength,
        use_evidential=_split_evidential,
        tau_evidential=_split_tau_evid,
        s_ref=_avg_strength,
    )

    # ------------------------------------------------------------------
    # 4. GT clustering evaluation (observation only)
    # ------------------------------------------------------------------
    if gt_tags_file and os.path.exists(gt_tags_file):
        _evaluate_with_gt(data_loader.gt_labels, new_labels_np,
                          flat_real_indices, args.output_dir)

    # ------------------------------------------------------------------
    # 5. Consensus computation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Consensus computation")
    print("=" * 60)

    # MV consensus: training target (breaks the encoder self-pollution loop) +
    # homologous FASTA for evaluation
    consensus_dict = compute_mv_consensus(
        data_loader=data_loader,
        new_labels_np=new_labels_np,
        flat_real_indices=flat_real_indices,
        model_max_len=model_max_len,
        ref_length=getattr(args, 'ref_length', None),
    )

    # Cluster difficulty (for Step1 dynamic sampling)
    cluster_change_info = compute_cluster_difficulty(new_labels_np, _np_strength)
    cv_threshold  = getattr(args, 'cv_threshold', 0.3)
    hard_clusters = sum(1 for v in cluster_change_info.values() if v >= cv_threshold)
    cv_values     = list(cluster_change_info.values())
    cv_median     = float(np.median(cv_values)) if cv_values else 0.0
    print(f"   cluster_difficulty: hard(>={cv_threshold})={hard_clusters}, "
          f"easy={len(cluster_change_info)-hard_clusters}, median CV={cv_median:.3f}")

    # ------------------------------------------------------------------
    # 6. Dump
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Saving state")
    print("=" * 60)

    next_round_dir = os.path.join(args.experiment_dir, "04_Iterative_Labels")
    os.makedirs(next_round_dir, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")

    full_labels = np.full(TOTAL_READS, -1, dtype=int)
    full_labels[flat_real_indices] = new_labels_np
    label_path = os.path.join(next_round_dir, f"refined_labels_{ts}.txt")
    np.savetxt(label_path, full_labels, fmt='%d')
    print(f"   labels: {label_path}")

    # Uncertainty dump (core evidential output, for analysis)
    full_u_epi    = np.zeros(TOTAL_READS, dtype=np.float32)
    full_u_ale    = np.zeros(TOTAL_READS, dtype=np.float32)
    full_strength = np.zeros(TOTAL_READS, dtype=np.float32)
    full_u_epi[flat_real_indices]    = _np_u_epi
    full_u_ale[flat_real_indices]    = _np_u_ale
    full_strength[flat_real_indices] = _np_strength
    state_path = os.path.join(next_round_dir, f"read_state_{ts}.pt")
    torch.save({'u_epi': full_u_epi, 'u_ale': full_u_ale,
                'strength': full_strength, 'round_idx': round_idx}, state_path)
    print(f"   state (u_epi/u_ale/strength): {state_path}")
    del full_u_epi, full_u_ale, full_strength
    gc.collect()

    consensus_path = os.path.join(next_round_dir, f"consensus_dict_{ts}.pt")
    torch.save(consensus_dict, consensus_path)
    print(f"   consensus dict: {consensus_path}")

    change_info_path = os.path.join(next_round_dir, f"cluster_change_info_{ts}.pt")
    torch.save(cluster_change_info, change_info_path)
    print(f"   cluster change info: {change_info_path}")

    # FASTA: MV consensus on strict labels (for SR evaluation, homologous with the training target)
    fasta_dir = os.path.join(args.output_dir, "consensus")
    os.makedirs(fasta_dir, exist_ok=True)
    fasta_path = os.path.join(fasta_dir, f"consensus_{ts}.fasta")
    try:
        save_consensus_fasta(
            consensus_dict, new_labels_np, flat_real_indices,
            data_loader, model_max_len, fasta_path,
            ref_length=getattr(args, 'ref_length', None),
        )
        print(f"   fasta: {fasta_path}")
    except Exception as e:
        print(f"   [warn] fasta save failed: {e}")
        fasta_path = None

    # ------------------------------------------------------------------
    # 7. Paper log
    # ------------------------------------------------------------------
    _record_paper_log(args, TOTAL_READS, consensus_dict, _avg_strength)

    del _np_u_epi, _np_u_ale, _np_strength
    gc.collect()

    return {
        'next_round_files': {
            'labels':              label_path,
            'state':               state_path,
            'reference':           fasta_path,
            'consensus':           consensus_path,
            'cluster_change_info': change_info_path,
        }
    }