# models/step2_decode.py
"""
Consensus decoding for Step 2.

Terminal pipeline uses pure majority-vote (MV) consensus: for each cluster, count
one-hot bases per position, keep positions where at least 50% of reads have a base
(has_vote), and take the per-position argmax. This does not run the encoder, which
breaks the encoder self-pollution loop when the consensus is used as the next
round's training target.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict

BASE_MAP = ['A', 'C', 'G', 'T']


# ---------------------------------------------------------------------------
# MV consensus: pure majority vote, no encoder
# ---------------------------------------------------------------------------
def compute_mv_consensus(
    data_loader,
    new_labels_np,
    flat_real_indices,
    model_max_len: int,
    ref_length: int = None,
) -> Dict[int, torch.Tensor]:
    """Pure majority-vote consensus, used as the Round 2+ Step1 training target.

    Per cluster (label >= 0 reads only):
        counts[L, 4] = sum over reads of one_hot(seq)
        has_vote[L]  = (valid read count >= 50% * N)   # >=50% gate
        indices[L]   = counts.argmax(dim=-1)
        one_hot[L,4] = F.one_hot(indices, 4); one_hot[~has_vote] = 0

    Args:
        data_loader:       CloverDataLoader, provides data_loader.reads[real_idx]
        new_labels_np:     strict labels (numpy; -1 is skipped automatically)
        flat_real_indices: mapping to data_loader.reads real indices
        model_max_len:     one-hot length (typically 201)
        ref_length:        prior length, for logging only; the one-hot shape stays
                           model_max_len (truncation happens in save_consensus_fasta)

    Returns:
        consensus_dict: {cluster_id: Tensor(model_max_len, 4)} one-hot
    """
    from models.step1_data import seq_to_onehot

    # cluster_id -> real_idx list (label >= 0 only)
    cluster_to_ridx: Dict[int, list] = defaultdict(list)
    for didx, label in enumerate(new_labels_np):
        if label >= 0:
            real_idx = flat_real_indices[didx]
            cluster_to_ridx[int(label)].append(real_idx)

    print(f"\n   MV consensus: {len(cluster_to_ridx)} clusters")
    if ref_length is not None:
        print(f"   ref_length={ref_length} (one-hot shape stays L={model_max_len}, "
              f"truncation handled in save_consensus_fasta)")

    consensus_dict: Dict[int, torch.Tensor] = {}
    skipped = 0

    for cluster_id, ridx_list in cluster_to_ridx.items():
        n_reads = len(ridx_list)
        if n_reads < 1:
            skipped += 1
            continue

        # Collect one-hot encodings + padding masks
        encodings = []
        padding_masks = []
        for ridx in ridx_list:
            seq = data_loader.reads[ridx]
            enc = seq_to_onehot(seq, model_max_len)   # (L, 4)
            encodings.append(enc)
            pmask = enc.sum(dim=-1) > 0               # (L,) bool
            padding_masks.append(pmask)

        enc_tensor   = torch.stack(encodings)          # (N, L, 4)
        pmask_tensor = torch.stack(padding_masks)      # (N, L)

        # Tally votes
        counts   = enc_tensor.sum(dim=0)               # (L, 4) per-position base counts
        n_valid  = pmask_tensor.sum(dim=0).float()     # (L,)   per-position valid read count
        # has_vote: >=50% gate (prevents a single insertion read from extending the sequence)
        has_vote = (n_valid >= max(n_reads * 0.5, 1))  # (L,) bool

        # argmax -> one_hot, zero out padding positions
        indices = counts.argmax(dim=-1)                # (L,)
        one_hot = F.one_hot(indices, num_classes=4).float()  # (L, 4)
        one_hot[~has_vote] = 0.0

        consensus_dict[cluster_id] = one_hot

        del enc_tensor, pmask_tensor, counts, n_valid, has_vote, indices, one_hot

    print(f"   generated MV consensus for {len(consensus_dict)} clusters "
          f"(skipped {skipped} empty)")
    return consensus_dict


# ---------------------------------------------------------------------------
# Save consensus_dict to FASTA
# ---------------------------------------------------------------------------
def save_consensus_fasta(
    consensus_dict: Dict[int, torch.Tensor],
    new_labels_np: np.ndarray,
    flat_real_indices,
    data_loader,
    model_max_len: int,
    fasta_path: str,
    ref_length: int = None,
):
    """Write consensus_dict to FASTA.

    Truncation length: when ref_length is given (prior), use it uniformly;
    otherwise use the per-cluster mode of read lengths. The mode avoids a single
    insertion read (e.g. 197bp) extending the whole cluster, whose trailing zeros
    would argmax to 'A' and add a spurious base.
    """
    from collections import Counter as _Counter

    # Per-cluster read-length mode
    cluster_len_votes: Dict[int, list] = {}
    for didx, label in enumerate(new_labels_np):
        if label >= 0:
            real_idx = flat_real_indices[didx]
            rl = min(len(data_loader.reads[real_idx]), model_max_len)
            cid = int(label)
            if cid not in cluster_len_votes:
                cluster_len_votes[cid] = []
            cluster_len_votes[cid].append(rl)

    cluster_actual_len: Dict[int, int] = {}
    for cid, lens in cluster_len_votes.items():
        cluster_actual_len[cid] = _Counter(lens).most_common(1)[0][0]  # mode

    # With a prior, truncate uniformly to ref_length; otherwise fall back to the mode
    if ref_length is not None:
        print(f"   truncating to prior ref_length={ref_length} (overrides read mode)")
        mode_mismatch = sum(1 for cid, ml in cluster_actual_len.items() if ml != ref_length)
        if mode_mismatch > 0:
            print(f"   [warn] {mode_mismatch} clusters have read mode != ref_length (overridden by prior)")

    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    with open(fasta_path, 'w') as ff:
        for cluster_id, one_hot in sorted(consensus_dict.items()):
            if ref_length is not None:
                actual_len = ref_length
            else:
                actual_len = cluster_actual_len.get(cluster_id, model_max_len)
            indices = one_hot[:actual_len].argmax(dim=-1).numpy()
            seq = ''.join(BASE_MAP[i] for i in indices)
            ff.write(f">cluster_{cluster_id}\n{seq}\n")

    print(f"   consensus FASTA: {fasta_path}")