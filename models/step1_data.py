# models/step1_data.py
"""
Data loading and dynamic sampling for SSI-EC Step 1.

Provides:
  - seq_to_onehot: vectorized one-hot encoding with an 'N'-handling virtual channel
  - CloverDataLoader: loads reads / labels / references / GT tags
  - Step1Dataset: per-read samples with consensus targets
  - create_cluster_balanced_sampler: cluster-balanced greedy batching with
    per-epoch dynamic sampling for Round 2+
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from collections import defaultdict
import warnings

# ============================================================
# Vectorized one-hot encoding
# ============================================================
# 'N'-base handling via a virtual channel: 'A','C','G','T' map to indices 0-3;
# 'N' and any unexpected character map to index 4. After truncating to the first
# 4 channels, 'N' becomes [0,0,0,0], which the model's mask mechanism reads as a
# padded/absent position (mask = consensus.sum(dim=-1) > 0).
_BASE_LUT = np.full(256, 4, dtype=np.int64)  # default index 4 (virtual channel)
for _c, _i in [('A',0),('C',1),('G',2),('T',3),
               ('a',0),('c',1),('g',2),('t',3)]:
    _BASE_LUT[ord(_c)] = _i
# 'N', 'n' and any unexpected character keep index 4


def seq_to_onehot(seq: str, max_len: int) -> torch.Tensor:
    """Vectorized one-hot encoding (5-10x faster than a per-character loop).

    Uses a 5-channel trick to handle 'N':
      - A/C/G/T  -> indices 0-3
      - N or any unexpected char -> index 4 (virtual channel)
      - returns the first 4 channels, so 'N' positions become [0,0,0,0]
        and couple with the model's mask (mask = consensus.sum(dim=-1) > 0).

    Example:
        seq = "ACGNT" -> indices [0,1,2,4,3]
        first 4 channels: A=[1,0,0,0] C=[0,1,0,0] G=[0,0,1,0] N=[0,0,0,0] T=[0,0,0,1]
    """
    L = min(len(seq), max_len)
    byte_arr = np.frombuffer(seq[:L].encode('ascii'), dtype=np.uint8)
    indices = _BASE_LUT[byte_arr]

    # Allocate 5 channels (A, C, G, T, virtual)
    tensor = torch.zeros(max_len, 5)
    tensor[np.arange(L), indices] = 1.0

    # Return the first 4 channels: 'N's 1.0 sits in channel 4 and is dropped here
    return tensor[:, :4]  # shape: (max_len, 4)


# ============================================================
# Data loader
# ============================================================
class CloverDataLoader:
    def __init__(self, experiment_dir: str, labels_path: str = None):
        self.experiment_dir = experiment_dir

        feddna_subdir = os.path.join(experiment_dir, "03_FedDNA_In")
        self.feddna_dir = feddna_subdir if os.path.exists(feddna_subdir) else experiment_dir
        self.labels_path = labels_path

        self.reads: List[str] = []
        self.clover_labels: List[int] = []
        self.gt_labels: List[int] = []

        self._load_all_data()

    def _load_all_data(self):
        print("\n" + "=" * 60)
        print("[DataLoader] Loading data...")
        print("=" * 60)

        read_path = os.path.join(self.feddna_dir, "read.txt")
        if not os.path.exists(read_path):
            raise FileNotFoundError(f"read.txt not found: {read_path}")

        # read.txt format: reads, then a separator line after each cluster's reads.
        # The first cluster's reads have no preceding separator, so current_cluster
        # starts at 0 and advances each time a separator is seen.
        current_cluster = 0
        with open(read_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("====="):
                    current_cluster += 1   # separator = current cluster ends; next read is a new cluster
                else:
                    self.reads.append(line)
                    self.clover_labels.append(current_cluster)

        # One separator per cluster, so the separator count equals the cluster count.
        n_clusters = current_cluster
        print(f"   reads: {len(self.reads)}")
        print(f"   clusters (from read.txt): {n_clusters}")

        # Round 2+: override Clover's initial labels with refined_labels
        if self.labels_path and os.path.exists(self.labels_path):
            refined = np.loadtxt(self.labels_path, dtype=int).tolist()
            if len(refined) == len(self.reads):
                self.clover_labels = refined
                noise_cnt = sum(1 for x in refined if x < 0)
                print(f"   labels (refined): {self.labels_path}, noise={noise_cnt}")
            else:
                print(f"   [warn] refined_labels length mismatch ({len(refined)} vs {len(self.reads)}), keeping Clover labels")

        self.gt_labels = [-1] * len(self.reads)

        # Load ref.txt -> Round 1 consensus reference (tag majority vote, fully self-supervised)
        self.ref_seqs: Dict[int, str] = {}
        ref_path = os.path.join(self.feddna_dir, "ref.txt")
        if os.path.exists(ref_path):
            with open(ref_path, 'r') as rf:
                for cid, line in enumerate(rf):
                    seq = line.strip()
                    if seq:
                        self.ref_seqs[cid] = seq
            print(f"   ref.txt: {len(self.ref_seqs)} reference sequences")
        else:
            print(f"   [warn] ref.txt not found: {ref_path}")

        valid = sum(1 for l in self.clover_labels if l >= 0)
        unique = len(set(l for l in self.clover_labels if l >= 0))
        print(f"   valid reads: {valid}, unique clusters: {unique}")

    def load_gt_tags(self, gt_tags_file: str):
        """Load GT labels. File format: cluster_id<TAB>sequence.
        Builds a sequence->cluster_id map, then matches each read by exact sequence."""
        if not os.path.exists(gt_tags_file):
            print(f"   [warn] GT tags file not found: {gt_tags_file}")
            return
        print(f"   loading GT tags: {gt_tags_file}")

        # Build sequence -> cluster_id map
        seq_to_cluster: Dict[str, int] = {}
        total_lines = 0
        with open(gt_tags_file) as f:
            for line in f:
                total_lines += 1
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        cluster_id = int(parts[0])
                        seq = parts[1].strip().upper()
                        seq_to_cluster[seq] = cluster_id
                    except ValueError:
                        pass

        # Match each read against the map
        matched = 0
        for i, read in enumerate(self.reads):
            cluster_id = seq_to_cluster.get(read.upper())
            if cluster_id is not None:
                self.gt_labels[i] = cluster_id
                matched += 1

        rate = matched / max(len(self.reads), 1) * 100
        print(f"      GT entries: {total_lines}, matched: {matched}/{len(self.reads)} ({rate:.1f}%)")


# ============================================================
# Dataset
# ============================================================
class Step1Dataset(Dataset):
    def __init__(self, data_loader: CloverDataLoader, max_len: int = 150,
                 training_cap: int = 99999000000,
                 inference_mode: bool = False,
                 round_idx: int = 1,
                 consensus_dict: Optional[Dict[int, torch.Tensor]] = None,
                 cluster_change_info: Optional[Dict[int, float]] = None,
                 cv_threshold: float = 0.3,
                 max_reads_per_cluster: int = 0):
        """Per-read dataset with consensus targets.

        Args:
            training_cap:        sample cap during training (Round 2+ only; Round 1
                                 is forced full-scale)
            inference_mode:      ignore training_cap and use all valid samples
            round_idx:           round index, used to vary the sampling seed
            consensus_dict:      {cluster_id: Tensor(L, 4)} per-cluster pseudo-
                                 reference one-hot
            cluster_change_info: {cluster_id: CV} basis for Round 2+ cluster-level
                                 sampling. Hard clusters (CV >= cv_threshold): 100%
                                 of reads; easy clusters (CV < cv_threshold):
                                 R2=80%, R3+=50%
            cv_threshold:        hard/easy CV boundary, default 0.3
            max_reads_per_cluster: max reads sampled per cluster during training,
                                 0 = unlimited. Consistent with FedDNA's training
                                 design. Affects training sampling only, not Step2
                                 inference (full-scale).

        For Round 2+, the Dataset does not fix the sampled subset in __init__.
        self.valid_indices holds all valid reads; cluster_change_info / cv_threshold
        / easy_ratio are stored as instance attributes and consumed per-epoch by
        create_cluster_balanced_sampler, so each epoch sees a fresh random subset.
        """
        self.data_loader = data_loader
        self.max_len = max_len
        self.round_idx = round_idx
        self.consensus_dict = consensus_dict or {}
        self.cv_threshold = cv_threshold

        # Sampling metadata stored on the Dataset for the sampler to read per-epoch
        self.cluster_change_info = cluster_change_info
        self.easy_ratio = 0.80 if round_idx == 2 else 0.50  # R2=80%, R3+=50%
        self.max_reads_per_cluster = max_reads_per_cluster  # per-cluster cap, 0 = unlimited

        if self.consensus_dict:
            self._validate_consensus_dict()

        # Default consensus: all-zero (same as padding, triggers the mask mechanism)
        self._default_consensus = torch.zeros(max_len, 4)
        self._default_consensus_used = False

        full_valid_indices = [i for i, label in enumerate(data_loader.clover_labels) if label >= 0]
        total_valid = len(full_valid_indices)

        all_indices = list(range(len(data_loader.clover_labels)))

        if inference_mode:
            # Step 2 full-scale inference over all reads. In the terminal pipeline
            # labels carry no -1 (the split engine only regroups label>=0 reads and
            # never emits -1), so this is equivalent to the label>=0 set; all_indices
            # is kept as a safe superset in case upstream labels ever contain -1.
            self.valid_indices = all_indices
            n_negative = len(all_indices) - total_valid
            print(f"   inference mode (full-scale): {len(self.valid_indices)} samples "
                  f"({n_negative} reads with label=-1)")

        elif round_idx == 1:
            # Round 1: forced full-scale training (ignore training_cap) for a stable baseline
            self.valid_indices = full_valid_indices
            print(f"   Round 1 (forced full-scale): {len(self.valid_indices)} samples")

        elif cluster_change_info is not None and round_idx >= 2:
            # Round 2+ dynamic sampling: do not fix the subset here. valid_indices
            # holds the full pool and is overwritten per-epoch by the sampler.
            # full_valid_indices is the permanent pool the sampler draws from each
            # epoch, keeping __getitem__(idx) and the sampler's idx aligned to the
            # same read.
            self.valid_indices      = full_valid_indices  # initial = full; overwritten by sampler
            self.full_valid_indices = full_valid_indices  # permanent pool for per-epoch sampling
            print(f"   Round {round_idx} (dynamic sampling, full index retained): "
                  f"{len(self.valid_indices)} samples")
            print(f"      sampler runs per-epoch: "
                  f"hard clusters 100% / easy clusters {int(self.easy_ratio*100)}% (CV threshold={cv_threshold})")

        else:
            # Fallback: Round 2+ without cluster_change_info, use training_cap
            if total_valid > training_cap:
                seed = 42 + round_idx
                rng = np.random.RandomState(seed)
                self.valid_indices = rng.choice(
                    full_valid_indices, training_cap, replace=False
                ).tolist()
                print(f"   training (fallback): {total_valid} -> {training_cap} (seed={seed})")
            else:
                self.valid_indices = full_valid_indices

            print(f"   dataset size: {len(self.valid_indices)} samples")

    def _validate_consensus_dict(self):
        """Validate the consensus_dict format (shape/type/dim of the first few entries)."""
        issues = []
        for cluster_id, consensus in list(self.consensus_dict.items())[:10]:
            if not isinstance(consensus, torch.Tensor):
                issues.append(f"cluster {cluster_id}: not a Tensor")
            elif consensus.shape != (self.max_len, 4):
                issues.append(f"cluster {cluster_id}: shape {consensus.shape} != ({self.max_len}, 4)")
            elif consensus.dim() != 2:
                issues.append(f"cluster {cluster_id}: dim {consensus.dim()} != 2")

        if issues:
            warnings.warn(
                "consensus_dict format issues:\n" + "\n".join(issues[:5]),
                UserWarning
            )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        seq = self.data_loader.reads[real_idx]
        encoding = seq_to_onehot(seq, self.max_len)

        clover_label = self.data_loader.clover_labels[real_idx]
        gt_label = self.data_loader.gt_labels[real_idx]

        # Provide the consensus target for this read's cluster
        if clover_label in self.consensus_dict:
            consensus_target = self.consensus_dict[clover_label]
        else:
            # Warn once when falling back to the all-zero default
            if not self._default_consensus_used:
                warnings.warn(
                    f"cluster {clover_label} missing consensus, using default (all-zero)",
                    UserWarning
                )
                self._default_consensus_used = True
            consensus_target = self._default_consensus

        return {
            'encoding': encoding,
            'clover_label': clover_label,
            'gt_label': gt_label,
            'read_idx': real_idx,
            'consensus_target': consensus_target,   # (L, 4) one-hot of pseudo-reference
        }


# ============================================================
# Sampler (greedy fill)
# ============================================================
def create_cluster_balanced_sampler(dataset: Step1Dataset,
                                    batch_size: int = 256,
                                    max_clusters_per_batch: int = 64):
    """Cluster-balanced greedy batching.

    1. Round 2+ dynamic sampling: each call redraws an easy-cluster subset from the
       full pool (hard clusters 100%, easy clusters at dataset.easy_ratio), so each
       epoch trains on a fresh subset.
    2. Large-cluster remainders flow into current_batch rather than forming a
       single-cluster batch, ensuring every batch has negatives from multiple clusters.
    3. Singletons are spread evenly (round-robin into existing batches' free slots)
       instead of piling up at the epoch's end, so they contribute repulsive
       gradients as negatives throughout.
    """
    print("   building sampler (greedy fill)...")

    all_labels = dataset.data_loader.clover_labels

    # ==================================================================
    # Round 2+ dynamic sampling: redraw the easy-cluster subset each call
    # ==================================================================
    cluster_change_info = dataset.cluster_change_info
    if cluster_change_info is not None and dataset.round_idx >= 2:
        cv_threshold = dataset.cv_threshold
        easy_ratio   = dataset.easy_ratio
        rng = np.random.RandomState()   # unseeded -> different each epoch

        # Read the full pool from the permanent backup, not from valid_indices
        # (which the previous epoch overwrote with a subset).
        full_pool = getattr(dataset, 'full_valid_indices', dataset.valid_indices)

        # Group the full pool by cluster
        cluster_to_full: Dict[int, List[int]] = defaultdict(list)
        for i in full_pool:
            label = all_labels[i]
            if label >= 0:
                cluster_to_full[label].append(i)

        # Sample by hard/easy to form this epoch's training set
        epoch_indices = []
        n_hard = n_easy = n_hard_reads = n_easy_reads = 0
        for cid, idxs in cluster_to_full.items():
            cv = cluster_change_info.get(cid, 0.0)
            if cv >= cv_threshold:
                epoch_indices.extend(idxs)
                n_hard += 1
                n_hard_reads += len(idxs)
            else:
                k = max(1, int(len(idxs) * easy_ratio))
                chosen = rng.choice(idxs, k, replace=False).tolist()
                epoch_indices.extend(chosen)
                n_easy += 1
                n_easy_reads += len(chosen)

        print(f"   dynamic sampling this epoch: "
              f"hard {n_hard} x 100% = {n_hard_reads} reads | "
              f"easy {n_easy} x {int(easy_ratio*100)}% = {n_easy_reads} reads | "
              f"total {len(epoch_indices)}")

        # Sync Dataset's current indices to this epoch's subset. The DataLoader
        # passes batch idx (0,1,2...) to __getitem__, which does
        # self.valid_indices[idx], so valid_indices must match working_indices
        # exactly or idx=0 would fetch the full pool's element 0 instead of this
        # subset's element 0 (label/data misalignment).
        working_indices = epoch_indices
        dataset.valid_indices = working_indices
    else:
        # Round 1 / inference_mode / fallback: valid_indices is already correct
        working_indices = dataset.valid_indices

    # ==================================================================
    # Group by cluster (based on this epoch's working_indices)
    # ==================================================================
    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, real_idx in enumerate(working_indices):
        label = all_labels[real_idx]
        cluster_to_indices[label].append(idx)

    # ==================================================================
    # Per-cluster cap: at most max_reads_per_cluster reads per cluster, consistent
    # with FedDNA (which samples 5-30 reads/cluster in training, full at inference).
    # Affects the training sampler only; Step2 inference (inference_mode=True) does
    # not pass through here. Per-epoch random sampling approaches full coverage over
    # multiple epochs.
    # ==================================================================
    max_rpc = getattr(dataset, 'max_reads_per_cluster', 0)
    if max_rpc > 0:
        n_capped = 0
        for cid in list(cluster_to_indices.keys()):
            idxs = cluster_to_indices[cid]
            if len(idxs) > max_rpc:
                cluster_to_indices[cid] = np.random.choice(
                    idxs, max_rpc, replace=False
                ).tolist()
                n_capped += 1
        if n_capped > 0:
            print(f"   per-cluster cap: {max_rpc} reads/cluster, "
                  f"truncated {n_capped} large clusters")

    # Valid clusters (>=2 reads) and singleton clusters (1 read)
    valid_clusters = {cid: idxs for cid, idxs in cluster_to_indices.items()
                      if len(idxs) >= 2}
    singleton_indices = [idxs[0] for idxs in cluster_to_indices.values()
                         if len(idxs) == 1]

    cluster_ids = list(valid_clusters.keys())
    np.random.shuffle(cluster_ids)

    # ==================================================================
    # Greedy fill: pack clusters into current_batch.
    # Large-cluster remainders extend into current_batch rather than batching alone.
    # ==================================================================
    batches = []
    current_batch: List[int] = []
    current_n_clusters = 0

    for cid in cluster_ids:
        idxs = valid_clusters[cid]

        if len(idxs) > batch_size:
            # Seal the current batch first
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_n_clusters = 0

            # Cut into full blocks (exactly batch_size each)
            n_full = len(idxs) // batch_size
            for i in range(n_full):
                batches.append(idxs[i * batch_size:(i + 1) * batch_size])

            # Remainder extends into current_batch to mix with later small clusters
            remainder = idxs[n_full * batch_size:]
            if remainder:
                current_batch = list(remainder)
                current_n_clusters = 1
            continue

        # Normal cluster: seal then open a new batch if it would overflow
        if (len(current_batch) + len(idxs) > batch_size or
                current_n_clusters >= max_clusters_per_batch):
            if current_batch:
                batches.append(current_batch)
            current_batch = list(idxs)
            current_n_clusters = 1
        else:
            current_batch.extend(idxs)
            current_n_clusters += 1

    if current_batch:
        batches.append(current_batch)

    # ==================================================================
    # Spread singletons evenly: round-robin into existing batches' free slots,
    # so they contribute repulsive gradients as negatives throughout the epoch.
    # ==================================================================
    if singleton_indices:
        np.random.shuffle(singleton_indices)
        overflow = []

        if len(batches) == 0:
            # Edge case: the whole dataset is singletons
            for i in range(0, len(singleton_indices), batch_size):
                batches.append(singleton_indices[i:i + batch_size])
        else:
            n_batches = len(batches)
            ptr = 0   # round-robin pointer across all batches
            for s in singleton_indices:
                # From ptr, find the first batch with a free slot
                inserted = False
                for _ in range(n_batches):
                    b_idx = ptr % n_batches
                    ptr += 1
                    if len(batches[b_idx]) < batch_size:
                        batches[b_idx].append(s)
                        inserted = True
                        break
                if not inserted:
                    overflow.append(s)  # all batches full

            # Pack overflow singletons into small batches (count is tiny, negligible)
            for i in range(0, len(overflow), batch_size):
                batches.append(overflow[i:i + batch_size])

        n_distributed = len(singleton_indices) - len(overflow) if singleton_indices else 0
        print(f"   singletons: {len(singleton_indices)} spread "
              f"({n_distributed} into existing batches, {len(overflow) if singleton_indices else 0} overflow)")

    # Print stats
    sizes = [len(b) for b in batches]
    avg_size = sum(sizes) / max(len(sizes), 1)
    n_valid_c = len(valid_clusters)
    n_single = len(singleton_indices)
    print(f"   batches: {len(batches)} (avg {avg_size:.0f} samples/batch)")
    print(f"      valid clusters (>=2): {n_valid_c}, singleton clusters: {n_single}")

    return batches


def create_dynamic_sampler(dataset, batch_size=256, max_clusters_per_batch=64,
                           state_path=None, round_idx=1):
    return create_cluster_balanced_sampler(dataset, batch_size, max_clusters_per_batch)