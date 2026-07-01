#!/usr/bin/env python3
"""
Tag-based reconstruction evaluation for SSI-EC.

Approach (tag-based mapping):
  1. Load all reads from read.txt (preserving DataLoader order)
  2. Map each read to its GT reference id via the "sequence -> GT tag" file
  3. For each cluster, majority-vote its reads' GT tags to assign the cluster's GT
     reference
  4. Compare each consensus against its GT reference -> Success Rate / Edit Error Rate

Features:
  - Auto-discovers all rounds under the experiment dir (Round 0 = Clover MV baseline)
  - Multi-round comparison table
  - Each GT reference is evaluated once (when several clusters map to one ref, the
    best is taken)
  - Optional non-singleton evaluation via --min_cluster_size_eval 2. This keeps
    the default all-cluster protocol unchanged while allowing an auxiliary metric
    that excludes singleton clusters, which are pure by definition but unreliable
    as consensus reconstructions under sequencing noise.

Metric definitions:
  Success Rate is reported two ways:
    SR_reachable = #{consensus == reference} / #{GT refs reachable from reads}
                   (main metric; excludes GT molecules that have no read in this
                    experiment's read.txt, which no method could ever reconstruct)
    SR_total     = #{consensus == reference} / #{all GT references}
                   (reference metric, over the full GT set)
  Edit Error Rate = mean( ED(consensus, reference) / len(reference) ) over covered refs

  "Reachable" = GT references that at least one read maps to (via the
  read -> tag -> ref chain). GT molecules with no matching read are excluded from
  the reachable denominator, since they are not recoverable from the given reads.

Usage:
  python eval_reconstruction.py \\
      --experiment_dir /path/to/seq_1d/ \\
      --gt_refs  /path/to/reads.fasta \\
      --gt_tags  /path/to/seq1d_tags_reads.txt

Dependencies:
  pip install edlib numpy tqdm
"""

import argparse
import os
import sys
import re
import glob
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm

# ===========================================================================
# Edit distance
# ===========================================================================
try:
    import edlib
    def levenshtein(a: str, b: str) -> int:
        return edlib.align(a, b, mode="NW", task="distance")['editDistance']
    _ED_ENGINE = "edlib"
except ImportError:
    try:
        import editdistance as _ed
        def levenshtein(a: str, b: str) -> int:
            return int(_ed.eval(a, b))
        _ED_ENGINE = "editdistance"
    except ImportError:
        def levenshtein(a: str, b: str) -> int:
            m, n = len(a), len(b)
            if m == 0: return n
            if n == 0: return m
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                prev, dp[0] = dp[0], i
                for j in range(1, n + 1):
                    tmp = dp[j]
                    dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
                    prev = tmp
            return dp[n]
        _ED_ENGINE = "pure-python (slow!)"


# ===========================================================================
# Data loading
# ===========================================================================
def load_reads_from_readtxt(path: str):
    """Load all reads from read.txt -> (reads_list, clover_labels_array).
    clover_labels[i] is read i's initial Clover cluster id (0, 1, 2, ...)."""
    reads = []
    clover_labels = []
    cid = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("====="):
                cid += 1
            else:
                reads.append(line.upper())
                clover_labels.append(cid)
    n_clusters = cid  # separator count == cluster count
    print(f"   reads:    {len(reads):,}")
    print(f"   clusters: {n_clusters:,}")
    return reads, np.array(clover_labels, dtype=np.int64)


def load_gt_tags_file(gt_tags_file: str):
    """Load the GT tags file (tag<TAB>read):
      - seq_to_tag:  {read_sequence -> tag_id}   (to match reads)
      - tag_to_reads:{tag_id -> [read_sequences]} (to build tag->ref via majority vote)
    """
    seq_to_tags = defaultdict(list)
    tag_to_reads = defaultdict(list)
    total = 0
    with open(gt_tags_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                total += 1
                try:
                    tag = int(parts[0])
                    seq = parts[1].strip().upper()
                    seq_to_tags[seq].append(tag)
                    tag_to_reads[tag].append(seq)
                except ValueError:
                    pass

    seq_to_tag = {}
    for seq, tags in seq_to_tags.items():
        seq_to_tag[seq] = Counter(tags).most_common(1)[0][0]

    print(f"   GT tags file: {total:,} lines, {len(seq_to_tag):,} unique sequences, "
          f"{len(tag_to_reads):,} unique tags")
    return seq_to_tag, tag_to_reads


def _majority_vote_simple(reads: list, ref_len: int) -> str:
    """Per-position majority vote to build a pseudo-reference."""
    vote = [Counter() for _ in range(ref_len)]
    for read in reads:
        for pos in range(min(len(read), ref_len)):
            b = read[pos].upper()
            if b in 'ACGT':
                vote[pos][b] += 1
    result = []
    for pos in range(ref_len):
        if vote[pos]:
            result.append(vote[pos].most_common(1)[0][0])
        else:
            result.append('A')
    return ''.join(result)


def build_tag_to_ref_mapping(tag_to_reads: dict, gt_refs: dict, ref_len: int = 196):
    """Build a tag_id -> ref_id mapping.

    Strategy:
      1. majority-vote each tag's reads -> pseudo-reference
      2. build a ref_sequence -> ref_id reverse index
      3. exact match: pseudo-ref equals a ref -> map directly
      4. approximate match: on exact failure, take the ref with minimal ED (fallback)

    Most tags should match exactly (MV is highly accurate with enough reads).
    """
    print(f"\n{'-' * 68}")
    print(f"  Building tag -> reference mapping")
    print(f"{'-' * 68}")

    # Reverse index: sequence -> ref_id
    ref_seq_to_id = {}
    for ref_id, ref_seq in gt_refs.items():
        ref_seq_to_id[ref_seq] = ref_id

    # Ref sequence list (for ED fallback)
    ref_ids_list = sorted(gt_refs.keys())
    ref_seqs_list = [gt_refs[rid] for rid in ref_ids_list]

    tag_to_ref = {}
    exact_match = 0
    ed_match = 0
    failed = 0

    tags = sorted(tag_to_reads.keys())
    for tag_id in tqdm(tags, desc="  tag->ref mapping", leave=False):
        reads = tag_to_reads[tag_id]

        pseudo_ref = _majority_vote_simple(reads, ref_len)

        # Strategy 1: exact match
        if pseudo_ref in ref_seq_to_id:
            tag_to_ref[tag_id] = ref_seq_to_id[pseudo_ref]
            exact_match += 1
            continue

        # Strategy 2: minimal-ED match
        best_ed = float('inf')
        best_ref_id = None
        for rid, rseq in zip(ref_ids_list, ref_seqs_list):
            ed = levenshtein(pseudo_ref, rseq)
            if ed < best_ed:
                best_ed = ed
                best_ref_id = rid
                if ed == 0:
                    break

        if best_ref_id is not None and best_ed <= ref_len * 0.3:
            tag_to_ref[tag_id] = best_ref_id
            ed_match += 1
        else:
            failed += 1

    print(f"   unique tags:     {len(tags):,}")
    print(f"   exact matches:   {exact_match:,}  ({exact_match/len(tags)*100:.1f}%)")
    print(f"   ED matches:      {ed_match:,}")
    print(f"   failed:          {failed:,}")
    print(f"   total mapped:    {len(tag_to_ref):,}")

    return tag_to_ref


def match_reads_to_gt(reads: list, seq_to_tag: dict, tag_to_ref: dict) -> np.ndarray:
    """Find each read's GT reference id.
    Chain: read_sequence -> tag_id (seq_to_tag) -> ref_id (tag_to_ref)."""
    gt_ref_ids = np.full(len(reads), -1, dtype=np.int64)
    matched = 0
    tag_miss = 0
    ref_miss = 0
    for i, read in enumerate(reads):
        tag = seq_to_tag.get(read)
        if tag is None:
            tag_miss += 1
            continue
        ref_id = tag_to_ref.get(tag)
        if ref_id is None:
            ref_miss += 1
            continue
        gt_ref_ids[i] = ref_id
        matched += 1
    print(f"   read -> ref matched: {matched:,} / {len(reads):,} ({matched/len(reads)*100:.1f}%)")
    if tag_miss > 0:
        print(f"   tag lookup failed:   {tag_miss:,}")
    if ref_miss > 0:
        print(f"   tag->ref failed:     {ref_miss:,}")
    return gt_ref_ids


def load_gt_refs_fasta(path: str) -> dict:
    """Load GT references from FASTA or a bare-sequence file: {int_id: sequence}.
    Auto-detects format: standard FASTA (with '>' headers) uses header ids; a
    bare-sequence file (one sequence per line) uses the 1-based line number as id.
    Eval aligns by sequence, so the id is only an identifier and bare-sequence
    line numbering does not affect SR/EER.
    """
    # Probe whether the first non-empty line has a '>' header
    has_header = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('>'):
                has_header = True
            break

    refs = {}
    if has_header:
        cur_id = None
        cur_seq = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if cur_id is not None:
                        refs[cur_id] = ''.join(cur_seq).upper()
                    try:
                        cur_id = int(line[1:].split()[0])
                    except ValueError:
                        cur_id = line[1:].strip()
                    cur_seq = []
                elif line:
                    cur_seq.append(line)
        if cur_id is not None:
            refs[cur_id] = ''.join(cur_seq).upper()
        print(f"   GT references: {len(refs):,}  (FASTA format)")
    else:
        with open(path) as f:
            idx = 1
            for line in f:
                seq = line.strip().upper()
                if seq:
                    refs[idx] = seq
                    idx += 1
        print(f"   GT references: {len(refs):,}  (bare-sequence format, line number as id)")
    return refs


def load_ref_txt(path: str) -> dict:
    """Load Clover MV pseudo-references from ref.txt (one sequence per line):
    {cluster_id: sequence}."""
    refs = {}
    with open(path) as f:
        for i, line in enumerate(f):
            seq = line.strip().upper()
            if seq:
                refs[i] = seq
    return refs


def parse_consensus_fasta(path: str) -> dict:
    """Parse a consensus FASTA: {cluster_id_int: sequence}."""
    seqs = {}
    cur_id = None
    cur_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cur_id is not None:
                    seqs[cur_id] = ''.join(cur_seq).upper()
                header = line[1:].strip()
                m = re.match(r'cluster_(\d+)', header)
                if m:
                    cur_id = int(m.group(1))
                else:
                    try:
                        cur_id = int(header)
                    except ValueError:
                        cur_id = header
                cur_seq = []
            elif line:
                cur_seq.append(line)
    if cur_id is not None:
        seqs[cur_id] = ''.join(cur_seq).upper()
    return seqs


# ===========================================================================
# Cluster -> GT reference mapping
# ===========================================================================
def build_cluster_to_gt(labels: np.ndarray, gt_tags: np.ndarray, min_cluster_size: int = 1):
    """For each cluster, majority-vote to assign its GT reference.

    Args:
        labels: per-read cluster labels.
        gt_tags: per-read GT reference ids.
        min_cluster_size: auxiliary evaluation filter. Clusters with fewer reads
                          than this threshold are excluded from cluster->GT
                          mapping. Default 1 preserves the original protocol.

    Returns:
        cluster_to_gt:  {cluster_id: gt_ref_id}
        cluster_purity: {cluster_id: float}  (majority count / total)
    """
    cluster_tags = defaultdict(list)
    for i in range(len(labels)):
        if labels[i] >= 0 and gt_tags[i] >= 0:
            cluster_tags[int(labels[i])].append(int(gt_tags[i]))

    cluster_to_gt = {}
    cluster_purity = {}
    skipped_small = 0
    singleton_clusters = 0

    for cid, tags in cluster_tags.items():
        size = len(tags)
        if size == 1:
            singleton_clusters += 1
        if size < min_cluster_size:
            skipped_small += 1
            continue

        counter = Counter(tags)
        majority_tag, majority_count = counter.most_common(1)[0]
        cluster_to_gt[cid] = majority_tag
        cluster_purity[cid] = majority_count / size

    if min_cluster_size > 1:
        print(f"     cluster-size eval filter: keep size>={min_cluster_size}; "
              f"skipped {skipped_small:,}/{len(cluster_tags):,} clusters "
              f"(singletons={singleton_clusters:,})")
    else:
        print(f"     cluster-size eval filter: keep all clusters; "
              f"singletons={singleton_clusters:,}/{len(cluster_tags):,}")

    return cluster_to_gt, cluster_purity


# ===========================================================================
# Evaluation
# ===========================================================================
def evaluate_reconstruction(
    consensus: dict,        # {cluster_id: consensus_seq}
    cluster_to_gt: dict,    # {cluster_id: gt_ref_id}
    gt_refs: dict,          # {gt_ref_id: gt_sequence}
    reachable_refs: set,    # GT ref ids reachable from reads (main-metric denominator)
    name: str = "Method",
) -> dict:
    """Per-GT-reference evaluation.

    For each GT reference:
      - find all clusters mapping to it
      - take the minimal ED among those clusters' consensuses
      - if no cluster maps to it, ED = len(reference) (fully lost)

    Each GT reference is evaluated exactly once.

    Success Rate is reported two ways:
      SR_reachable = success / #{reachable GT refs}  (main; fair to the method,
                     excludes GT molecules with no read in this experiment)
      SR_total     = success / #{all GT refs}        (reference, over the full GT set)
    """
    n_gt = len(gt_refs)
    n_reachable = len(reachable_refs)

    # Reverse mapping: gt_ref_id -> [cluster_ids that have a consensus]
    gt_to_clusters = defaultdict(list)
    for cid, gt_id in cluster_to_gt.items():
        if cid in consensus:
            gt_to_clusters[gt_id].append(cid)

    ed_list = []
    eer_list = []
    success = 0
    covered = 0

    for gt_id in tqdm(sorted(gt_refs.keys()), desc=f"  [{name}] ED", leave=False):
        gt_seq = gt_refs[gt_id]
        gt_len = max(len(gt_seq), 1)
        clusters = gt_to_clusters.get(gt_id, [])

        if not clusters:
            # GT reference not covered by any cluster
            ed_list.append(gt_len)
            eer_list.append(1.0)
            continue

        covered += 1

        # Several clusters mapping to one GT ref -> take the best
        best_ed = float('inf')
        for cid in clusters:
            ed = levenshtein(consensus[cid], gt_seq)
            best_ed = min(best_ed, ed)

        ed_list.append(best_ed)
        eer_list.append(best_ed / gt_len)
        if best_ed == 0:
            success += 1

    ed_arr = np.array(ed_list, dtype=np.float32)
    eer_arr = np.array(eer_list, dtype=np.float32)

    # EER over covered refs only (their clusters are pre-aligned, 100% covered)
    covered_mask = np.array([1 if gt_to_clusters.get(gt_id) else 0
                              for gt_id in sorted(gt_refs.keys())], dtype=bool)
    eer_covered = eer_arr[covered_mask]

    results = {
        'name':                  name,
        'n_gt':                  n_gt,
        'n_reachable':           n_reachable,
        'n_clusters':            len(consensus),
        'n_eval_clusters':       len(cluster_to_gt),
        'n_covered':             covered,
        'success':               success,
        # Main metric: fair denominator (reachable refs)
        'success_rate':          success / max(n_reachable, 1),
        # Reference metric: full GT set
        'success_rate_total':    success / max(n_gt, 1),
        # Recall reported both ways for the same reason
        'recall':                covered / max(n_reachable, 1),
        'recall_total':          covered / max(n_gt, 1),
        'eer_mean':              float(eer_covered.mean()) if len(eer_covered) > 0 else 0.0,
        'ed_mean':               float(ed_arr[covered_mask].mean()) if covered > 0 else 0.0,
        'ed_median':             float(np.median(ed_arr[covered_mask])) if covered > 0 else 0.0,
        'ed_p90':                float(np.percentile(ed_arr[covered_mask], 90)) if covered > 0 else 0.0,
        'ed_p95':                float(np.percentile(ed_arr[covered_mask], 95)) if covered > 0 else 0.0,
        'ed_max':                float(ed_arr[covered_mask].max()) if covered > 0 else 0.0,
    }

    _print_result(results)
    return results


def _print_result(r: dict):
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  {r['name']}")
    print(f"{sep}")
    print(f"  GT molecules (total) : {r['n_gt']:>10,}")
    print(f"  GT reachable         : {r['n_reachable']:>10,}")
    print(f"  consensus clusters   : {r['n_clusters']:>10,}")
    print(f"  eval clusters        : {r.get('n_eval_clusters', r['n_clusters']):>10,}")
    print(f"  covered GT molecules : {r['n_covered']:>10,}")
    print()
    print(f"  Success Rate (reachable) : {r['success_rate']:.6f}  "
          f"({r['success']:,}/{r['n_reachable']:,})   [main]")
    print(f"  Success Rate (total)     : {r['success_rate_total']:.6f}  "
          f"({r['success']:,}/{r['n_gt']:,})")
    print(f"  Recall (reachable)       : {r['recall']:.6f}  "
          f"({r['n_covered']:,}/{r['n_reachable']:,})")
    print()
    print(f"  Edit Error Rate (covered mean): {r['eer_mean']:.6f}")
    print(f"  Edit Distance (covered only):")
    print(f"    Mean   : {r['ed_mean']:>8.2f}")
    print(f"    Median : {r['ed_median']:>8.2f}")
    print(f"    P90    : {r['ed_p90']:>8.2f}")
    print(f"    P95    : {r['ed_p95']:>8.2f}")
    print(f"    Max    : {r['ed_max']:>8.2f}")
    print(sep)


def _print_comparison_table(all_results: list):
    """Print the multi-method comparison table."""
    if not all_results:
        return

    header = (f"{'Method':<18} {'#Clusters':>9} {'#Eval':>8} {'Recall':>8} "
              f"{'SR_reach':>16} {'SR_total':>9} {'EER':>10} {'ED_mean':>8} {'ED_med':>8} {'ED_P90':>8}")
    sep_line = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print("  Comparison table (reconstruction quality)")
    print(f"{'=' * len(header)}")
    print(f"  {header}")
    print(f"  {sep_line}")
    for r in all_results:
        sr_str = f"{r['success_rate']:.4f} ({r['success']:>5})"
        print(f"  {r['name']:<18} "
              f"{r['n_clusters']:>9,} "
              f"{r.get('n_eval_clusters', r['n_clusters']):>8,} "
              f"{r['recall']:>8.4f} "
              f"{sr_str:>16} "
              f"{r['success_rate_total']:>9.4f} "
              f"{r['eer_mean']:>10.6f} "
              f"{r['ed_mean']:>8.2f} "
              f"{r['ed_median']:>8.2f} "
              f"{r['ed_p90']:>8.2f}")
    print(f"{'=' * len(header)}")
    print(f"\n  SR_reach = SR over reachable GT refs (main metric)")
    print(f"  SR_total = SR over all GT refs (reference)")


# ===========================================================================
# Round discovery
# ===========================================================================
def find_read_txt(experiment_dir: str) -> str:
    """Locate read.txt in several candidate paths."""
    candidates = [
        os.path.join(experiment_dir, "03_FedDNA_In", "read.txt"),
        os.path.join(experiment_dir, "04_FedDNA_In", "read.txt"),
        os.path.join(experiment_dir, "read.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    parent = os.path.dirname(experiment_dir.rstrip('/'))
    extra = [
        os.path.join(parent, "clover_out", "04_FedDNA_In", "read.txt"),
        os.path.join(parent, "Sequencing_data_first_dimension", "clover_out", "04_FedDNA_In", "read.txt"),
    ]
    for p in extra:
        if os.path.exists(p):
            return p

    return None


def find_ref_txt(experiment_dir: str) -> str:
    """Locate ref.txt (Clover MV pseudo-references)."""
    candidates = [
        os.path.join(experiment_dir, "03_FedDNA_In", "ref.txt"),
        os.path.join(experiment_dir, "04_FedDNA_In", "ref.txt"),
        os.path.join(experiment_dir, "ref.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def discover_rounds(experiment_dir: str) -> list:
    """Auto-discover (labels_path, consensus_path, round_name) for all rounds.
    Consensus and labels files share a timestamp, e.g.
    consensus_162641.fasta <-> refined_labels_162641.txt."""
    rounds = []

    consensus_pattern = os.path.join(
        experiment_dir, "results", "iter_*_step2", "consensus", "consensus_*.fasta"
    )
    consensus_files = sorted(glob.glob(consensus_pattern))

    labels_dir = os.path.join(experiment_dir, "04_Iterative_Labels")

    for cons_path in consensus_files:
        m_iter = re.search(r'iter_(\d+)_step2', cons_path)
        if not m_iter:
            continue
        round_idx = int(m_iter.group(1))

        m_ts = re.search(r'consensus_(\d+)\.fasta', os.path.basename(cons_path))
        if not m_ts:
            continue
        timestamp = m_ts.group(1)

        labels_path = os.path.join(labels_dir, f"refined_labels_{timestamp}.txt")
        if not os.path.exists(labels_path):
            print(f"  [warn] Round {round_idx}: labels not found {labels_path}, skipped")
            continue

        rounds.append((labels_path, cons_path, f"Round {round_idx}"))

    return rounds


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SSI-EC reconstruction evaluation (tag-based mapping)"
    )
    parser.add_argument('--experiment_dir', required=True,
                        help='SSI-EC experiment directory')
    parser.add_argument('--gt_refs', required=True,
                        help='GT reference FASTA (e.g. reads.fasta)')
    parser.add_argument('--gt_tags', required=True,
                        help='GT tags file (tag<TAB>read)')
    parser.add_argument('--read_txt', default=None,
                        help='read.txt path (auto-discovered; can be set manually)')
    parser.add_argument('--out', default=None,
                        help='output TSV file (optional)')
    parser.add_argument('--skip_round0', action='store_true',
                        help='skip Round 0 (Clover MV baseline)')
    parser.add_argument('--consensus_override', nargs='+', default=None,
                        help='extra baseline consensus fasta(s); reuse Clover labels, same protocol')
    parser.add_argument('--min_cluster_size_eval', type=int, default=1,
                        help='auxiliary evaluation filter: only clusters with at least this many reads are mapped/evaluated; default 1 keeps the original all-cluster protocol')
    args = parser.parse_args()

    exp_dir = args.experiment_dir

    print(f"\n{'=' * 68}")
    print(f"  SSI-EC reconstruction evaluation (tag-based)")
    print(f"{'=' * 68}")
    print(f"  ED engine: {_ED_ENGINE}")
    print(f"  exp dir:   {exp_dir}")
    print(f"  min_cluster_size_eval: {args.min_cluster_size_eval}")

    # 1. Load read.txt
    print(f"\n{'-' * 68}")
    print(f"  Loading reads")
    print(f"{'-' * 68}")

    read_txt = args.read_txt or find_read_txt(exp_dir)
    if read_txt is None:
        print("  [error] read.txt not found. Specify it with --read_txt.")
        sys.exit(1)
    print(f"  path: {read_txt}")
    reads, clover_labels = load_reads_from_readtxt(read_txt)

    # 2. Load GT tags
    print(f"\n{'-' * 68}")
    print(f"  Loading GT tags")
    print(f"{'-' * 68}")
    print(f"  path: {args.gt_tags}")
    seq_to_tag, tag_to_reads = load_gt_tags_file(args.gt_tags)

    # 3. Load GT references
    print(f"\n{'-' * 68}")
    print(f"  Loading GT references")
    print(f"{'-' * 68}")
    print(f"  path: {args.gt_refs}")
    gt_refs = load_gt_refs_fasta(args.gt_refs)

    # 4. Build tag -> reference mapping
    #    Tag IDs (13411, 111259, ...) differ from reads.fasta headers (1, 2, 3, ...),
    #    so build the correspondence via majority vote + sequence matching.
    ref_len = int(np.median([len(s) for s in gt_refs.values()]))
    print(f"   reference median length: {ref_len}bp")
    tag_to_ref = build_tag_to_ref_mapping(tag_to_reads, gt_refs, ref_len=ref_len)

    # 5. Match reads -> GT reference id
    print(f"\n{'-' * 68}")
    print(f"  Matching reads -> GT reference")
    print(f"{'-' * 68}")
    gt_ref_ids = match_reads_to_gt(reads, seq_to_tag, tag_to_ref)

    # Reachable GT refs = those at least one read maps to. This is the main-metric
    # denominator: GT molecules with no matching read cannot be reconstructed by any
    # method and are excluded from SR_reachable / recall.
    reachable_refs = set(gt_ref_ids[gt_ref_ids >= 0].tolist())
    print(f"   reachable GT refs: {len(reachable_refs):,} / {len(gt_refs):,} "
          f"({len(reachable_refs)/max(len(gt_refs),1)*100:.1f}% of full GT)")

    # 6. Discover rounds
    print(f"\n{'-' * 68}")
    print(f"  Discovering rounds")
    print(f"{'-' * 68}")

    all_results = []

    # Round 0: Clover MV baseline
    if not args.skip_round0:
        ref_txt = find_ref_txt(exp_dir)
        if ref_txt:
            print(f"\n  Round 0 (Clover MV baseline)")
            print(f"     ref.txt: {ref_txt}")
            clover_consensus = load_ref_txt(ref_txt)
            print(f"     consensus: {len(clover_consensus):,}")

            c2g_r0, pur_r0 = build_cluster_to_gt(
                clover_labels, gt_ref_ids, min_cluster_size=args.min_cluster_size_eval
            )
            avg_pur = np.mean(list(pur_r0.values())) if pur_r0 else 0
            print(f"     cluster -> GT: {len(c2g_r0):,} clusters, avg purity={avg_pur:.4f}")

            r0 = evaluate_reconstruction(clover_consensus, c2g_r0, gt_refs,
                                         reachable_refs, name="R0 Clover+MV")
            all_results.append(r0)
        else:
            print(f"  [warn] ref.txt not found, skipping Round 0")

    # Baseline consensus override (e.g. Iter/Div/BMA), reusing Clover labels
    if args.consensus_override:
        c2g_bl, _ = build_cluster_to_gt(
            clover_labels, gt_ref_ids, min_cluster_size=args.min_cluster_size_eval
        )
        for bl_path in args.consensus_override:
            tag = os.path.basename(bl_path).replace('consensus_', '').replace('.fasta', '')
            print(f"\n  Baseline: {tag}")
            print(f"     consensus: {bl_path}")
            bl_cons = parse_consensus_fasta(bl_path)
            print(f"     consensus clusters: {len(bl_cons):,}")
            r_bl = evaluate_reconstruction(bl_cons, c2g_bl, gt_refs,
                                           reachable_refs, name=f'BL:{tag}')
            all_results.append(r_bl)

    # Rounds 1, 2, 3, ...
    discovered = discover_rounds(exp_dir)
    print(f"\n  discovered {len(discovered)} SSI-EC rounds:")
    for labels_path, cons_path, name in discovered:
        print(f"    {name}: {os.path.basename(cons_path)} + {os.path.basename(labels_path)}")

    for labels_path, cons_path, round_name in discovered:
        print(f"\n  {round_name}")
        print(f"     consensus: {cons_path}")
        print(f"     labels:    {labels_path}")

        labels = np.loadtxt(labels_path, dtype=int)
        if len(labels) != len(reads):
            print(f"     [warn] labels length {len(labels)} != reads length {len(reads)}, skipped")
            continue

        consensus = parse_consensus_fasta(cons_path)
        print(f"     consensus clusters: {len(consensus):,}")

        c2g, pur = build_cluster_to_gt(
            labels, gt_ref_ids, min_cluster_size=args.min_cluster_size_eval
        )
        avg_pur = np.mean(list(pur.values())) if pur else 0
        print(f"     cluster -> GT: {len(c2g):,} clusters, avg purity={avg_pur:.4f}")

        r = evaluate_reconstruction(consensus, c2g, gt_refs,
                                    reachable_refs, name=round_name)
        all_results.append(r)

    # Comparison
    _print_comparison_table(all_results)

    # Optional: save TSV
    if args.out and all_results:
        keys = ['name', 'n_gt', 'n_reachable', 'n_clusters', 'n_eval_clusters', 'n_covered', 'success',
                'success_rate', 'success_rate_total', 'recall', 'recall_total',
                'eer_mean', 'ed_mean', 'ed_median', 'ed_p90', 'ed_p95', 'ed_max']
        with open(args.out, 'w') as f:
            f.write('\t'.join(keys) + '\n')
            for r in all_results:
                f.write('\t'.join(str(r.get(k, '')) for k in keys) + '\n')
        print(f"\n  results saved: {args.out}")


if __name__ == '__main__':
    main()