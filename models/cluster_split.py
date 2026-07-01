# models/cluster_split.py
# -*- coding: utf-8 -*-
"""
SSI-EC iterative engine: intra-cluster split.
====================================================
Rationale:
  - Clover's true loss is under-segmentation: two comparably-sized GT molecules
    are merged into one cluster, suppressing the minor GT so it cannot claim its
    own consensus. The merge direction rescues nothing; splitting is the fix.
  - Mechanism: hierarchically bisect a cluster's reads by edit distance, compute
    an MV consensus for each of the two sub-clusters, and if the edit distance
    between the two consensuses >= tau (default 5), judge that two molecules were
    wrongly merged and split; otherwise keep.
  - Pure clusters are protected by construction: homologous reads bisected in two
    yield near-identical consensuses (edit 0-1), which fail the tau gate.
  - tau=5 leaves a safety margin above the minimum edit distance between
    near-duplicate cross-origin reads (=2).

Evidential split criterion (DEFAULT ON = paper main method; switchable to pure-edit):
  evidential (multiply): dAB * min(S_A, S_B) / S_ref >= tau_evidential   [default]
  pure-edit (ablation):  dAB >= tau
  evidential (multiply):    dAB * min(S_A, S_B) / S_ref >= tau_evidential

  Physical meaning: an observed sequence difference is trusted as genuine
  molecular heterogeneity (and triggers a split) only when both sub-clusters have
  accumulated enough model evidence (high strength); if either sub-cluster has
  weak evidence (possibly noise), the split is suppressed, protecting noise from
  being split into spurious clusters.

  Direction note: putting strength in the denominator (dAB / min(S)) would make
  weaker evidence more likely to split, amplifying noise sub-clusters -> wrong
  splits. The multiplicative form (strength in the numerator) suppresses splits
  when evidence is weak, which is the correct direction.

  Switch: use_evidential=True (default) -> multiplicative criterion, requires
  strength_np + s_ref. use_evidential=False -> pure-edit dAB>=tau path (ablation
  baseline).

Why this is the engine that makes iteration useful:
  With Zone III disabled, step2 idles (labels are byte-for-byte unchanged). This
  split is the only mechanism that changes labels, and it merely regroups already-
  assigned reads -- it never produces -1, so it is neutral w.r.t. read_util (the
  coverage axis) and cannot trigger coverage collapse. Changing labels changes the
  next round's training consensus targets, forming a genuine iterative loop.

Insertion point (step2_runner.py):
  After new_labels_np is built, before the first consensus computation. Split
  new_labels_np in place; downstream consensus / dump / next-round training follow
  automatically.

Dependencies: edlib (already used by eval) + scipy. The levenshtein metric is
injected by the caller, reusing eval_reconstruction's definition.
"""
import numpy as np
from collections import defaultdict, Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def _mv_consensus(read_seqs, ref_length):
    """Per-position majority vote with a 50% has_vote threshold.
    Same convention as compute_mv_consensus."""
    N = len(read_seqs)
    if N == 0:
        return ""
    thresh = max(N * 0.5, 1)
    out = []
    for pos in range(ref_length):
        cnt = Counter()
        valid = 0
        for s in read_seqs:
            if pos < len(s):
                valid += 1
                b = s[pos]
                if b in 'ACGT':
                    cnt[b] += 1
        if valid >= thresh and cnt:
            out.append(cnt.most_common(1)[0][0])
    return ''.join(out)


def _split_two(seqs, levenshtein, max_pairwise=80, seed=0):
    """Hierarchically bisect a set of reads by edit distance.
    Returns two local index lists (a_local, b_local). When a cluster has more than
    max_pairwise reads, sample to build the distance matrix and assign the rest to
    the nearer of the two sub-cluster consensuses."""
    n = len(seqs)
    if n < 2:
        return list(range(n)), []

    if n <= max_pairwise:
        idxs = list(range(n))
        sub = seqs
    else:
        rng = np.random.default_rng(seed)
        idxs = sorted(rng.choice(n, max_pairwise, replace=False).tolist())
        sub = [seqs[i] for i in idxs]

    m = len(sub)
    D = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(i + 1, m):
            d = levenshtein(sub[i], sub[j])
            D[i, j] = D[j, i] = d
    Z = linkage(squareform(D, checks=False), method='average')
    lab = fcluster(Z, t=2, criterion='maxclust')  # values in {1, 2}
    a_local = [idxs[i] for i in range(m) if lab[i] == 1]
    b_local = [idxs[i] for i in range(m) if lab[i] == 2]

    if n > max_pairwise:
        ca = _mv_consensus([seqs[i] for i in a_local],
                           max((len(seqs[i]) for i in a_local), default=0)) if a_local else ""
        cb = _mv_consensus([seqs[i] for i in b_local],
                           max((len(seqs[i]) for i in b_local), default=0)) if b_local else ""
        assigned = set(idxs)
        for i in range(n):
            if i in assigned:
                continue
            da = levenshtein(seqs[i], ca) if ca else 1e9
            db = levenshtein(seqs[i], cb) if cb else 1e9
            (a_local if da <= db else b_local).append(i)

    return a_local, b_local


def split_clusters(
    new_labels_np,
    flat_real_indices,
    data_loader,
    levenshtein,
    ref_length=196,
    tau=5,
    min_split_size=6,
    max_pairwise=80,
    verbose=True,
    # ---- evidential multiplicative criterion (DEFAULT ON = paper main method) ----
    strength_np=None,
    use_evidential=True,
    tau_evidential=5,
    s_ref=None,
    low_s_factor=0.6,
):
    """Split clusters in place on new_labels_np (reads data_loader.reads only,
    writes no files).

    Args:
        new_labels_np:     (M,) int, labels in didx index space (-1 = noise,
                           excluded from splitting)
        flat_real_indices: didx -> data_loader.reads real index
        data_loader:       provides data_loader.reads[real_idx] -> sequence string
        levenshtein:       edit-distance function injected by the caller (edlib
                           version from eval_reconstruction)
        ref_length:        consensus truncation length (Seq_1D = 196)
        tau:               pure-edit gate; split only when the two sub-cluster
                           consensuses have edit distance >= tau (default 5)
        min_split_size:    do not attempt to split clusters with fewer reads
        max_pairwise:      max reads for building the distance matrix (sample above)

        --- evidential parameters ---
        strength_np:       (M,) float, per-read strength in didx space (same index
                           as new_labels_np; required when use_evidential=True)
        use_evidential:    True (default, paper main method) -> multiplicative
                           criterion dAB*min(S_A,S_B)/s_ref >= tau_evidential.
                           False -> pure-edit criterion dAB>=tau (ablation baseline).
        tau_evidential:    gate threshold for the multiplicative criterion (default
                           5, same value as edit tau for the cleanest ablation)
        s_ref:             global mean strength (required when use_evidential=True;
                           typically the average strength)
        low_s_factor:      factor defining a "low-evidence sub-cluster":
                           min(S) < low_s_factor * s_ref (statistics only, does not
                           affect the split decision)

    Returns:
        out_labels:  (M,) int, labels after splitting (new sub-clusters get new ids)
        stats:       dict, split statistics
    """
    labels = new_labels_np.copy()

    # Validate evidential inputs when enabled
    if use_evidential:
        if strength_np is None:
            raise ValueError("use_evidential=True requires strength_np (didx space, same index as new_labels_np)")
        if len(strength_np) != len(new_labels_np):
            raise ValueError(f"strength_np length {len(strength_np)} != new_labels_np length {len(new_labels_np)} "
                             f"(must share the didx index space)")
        if s_ref is None or s_ref <= 0:
            s_ref = float(np.asarray(strength_np)[labels >= 0].mean())  # fallback: mean over non-(-1)

    # Group didx by cluster (process label >= 0 only)
    cl_to_didx = defaultdict(list)
    for didx, lab in enumerate(labels):
        if lab >= 0:
            cl_to_didx[int(lab)].append(didx)

    # New ids start at max+1 to avoid collisions
    next_id = int(labels.max()) + 1 if (labels >= 0).any() else 0

    n_split = 0
    n_examined = 0
    split_dists = []
    # Evidence statistics for the paper
    n_protected_lowS = 0   # pure-edit would split but multiplicative suppresses, and min(S)<low_s_factor*s_ref
    n_protected_all  = 0   # pure-edit would split but multiplicative suppresses (regardless of strength)

    for cid, didxs in cl_to_didx.items():
        if len(didxs) < min_split_size:
            continue
        n_examined += 1

        seqs = [data_loader.reads[flat_real_indices[d]] for d in didxs]
        a_loc, b_loc = _split_two(seqs, levenshtein, max_pairwise=max_pairwise)
        if len(a_loc) < 1 or len(b_loc) < 1:
            continue

        consA = _mv_consensus([seqs[i] for i in a_loc], ref_length)
        consB = _mv_consensus([seqs[i] for i in b_loc], ref_length)
        if not consA or not consB:
            continue
        dAB = levenshtein(consA, consB)

        # ---- criterion ----
        if use_evidential:
            # Sub-cluster mean strength (didx-space direct indexing, no conversion)
            SA = float(np.mean([strength_np[didxs[i]] for i in a_loc]))
            SB = float(np.mean([strength_np[didxs[i]] for i in b_loc]))
            min_s = min(SA, SB)
            score = dAB * min_s / s_ref
            do_split = score >= tau_evidential

            # Evidence: pure-edit would split (dAB>=tau) but multiplicative suppresses
            if (dAB >= tau) and (not do_split):
                n_protected_all += 1
                if min_s < low_s_factor * s_ref:
                    n_protected_lowS += 1
        else:
            do_split = dAB >= tau

        if do_split:
            # Split: sub-cluster A keeps the original cid, B gets a new id
            b_didxs = [didxs[i] for i in b_loc]
            labels[b_didxs] = next_id
            next_id += 1
            n_split += 1
            split_dists.append(dAB)

    stats = {
        'use_evidential': use_evidential,
        'tau': tau_evidential if use_evidential else tau,
        'clusters_examined': n_examined,
        'clusters_split': n_split,
        'new_clusters_added': n_split,
        'n_clusters_before': len(cl_to_didx),
        'n_clusters_after': len(cl_to_didx) + n_split,
        'split_dist_median': float(np.median(split_dists)) if split_dists else 0.0,
        'clusters_protected_all':  n_protected_all,
        'clusters_protected_lowS': n_protected_lowS,
        's_ref': float(s_ref) if use_evidential else None,
    }

    if verbose:
        mode = f"evidential-multiply (tau'={tau_evidential}, S_ref={s_ref:.1f})" if use_evidential \
               else f"pure-edit (tau={tau})"
        print(f"\n   intra-cluster split [{mode}], min_size={min_split_size}")
        print(f"      clusters examined (size>={min_split_size}): {n_examined:,}")
        print(f"      clusters split:                            {n_split:,}")
        print(f"      cluster count: {stats['n_clusters_before']:,} -> {stats['n_clusters_after']:,}")
        if split_dists:
            print(f"      split-pair consensus edit median: {stats['split_dist_median']:.0f}")
        if use_evidential:
            print(f"      [evidence] suppressed (pure-edit would split): {n_protected_all:,} "
                  f"(of which low-evidence min(S)<{low_s_factor}*S_ref: {n_protected_lowS:,})")

    return labels, stats