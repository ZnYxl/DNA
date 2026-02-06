# models/step1_data.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter, deque  # ✅ 新增 deque


class CloverDataLoader:
    """
    适配你的数据格式的加载器
    ✅ 已修改支持迭代闭环：可以加载 refined_labels.txt
    """

    def __init__(self, experiment_dir: str, labels_path: str = None):
        self.experiment_dir = experiment_dir
        self.raw_dir = os.path.join(experiment_dir, "01_RawData")

        feddna_subdir = os.path.join(experiment_dir, "03_FedDNA_In")
        if os.path.exists(feddna_subdir):
            self.feddna_dir = feddna_subdir
        else:
            self.feddna_dir = experiment_dir
            print(f"   ℹ️ 使用非标准目录结构 (read.txt 直接在根目录)")

        self.labels_path = labels_path

        self.reads: List[str] = []
        self.clover_labels: List[int] = []
        self.gt_labels: List[int] = []
        self.gt_cluster_seqs: Dict[int, str] = {}

        self._load_all_data()

    def _load_feddna_format(self) -> Tuple[List[str], List[int]]:
        read_path = os.path.join(self.feddna_dir, "read.txt")

        if not os.path.exists(read_path):
            raise FileNotFoundError(f"找不到 read.txt: {read_path}")

        reads = []
        labels = []
        current_cluster = -1

        print(f"📂 加载FedDNA格式数据: {read_path}")

        with open(read_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("====="):
                    current_cluster += 1
                else:
                    reads.append(line)
                    labels.append(current_cluster)

        print(f"   ✅ 加载 {len(reads)} 条reads，{current_cluster + 1} 个Clover簇")
        return reads, labels

    def _load_raw_reads(self) -> Dict[str, str]:
        raw_path = os.path.join(self.raw_dir, "raw_reads.txt")
        if not os.path.exists(raw_path):
            print(f"   ⚠️ raw_reads.txt 不存在")
            return {}

        reads_dict = {}
        with open(raw_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    reads_dict[parts[0]] = parts[1]

        print(f"   ✅ 加载 {len(reads_dict)} 条原始reads")
        return reads_dict

    def _load_read_gt(self) -> Dict[str, Tuple[int, str, str]]:
        gt_path = os.path.join(self.raw_dir, "ground_truth_reads.txt")
        if not os.path.exists(gt_path):
            print(f"   ⚠️ ground_truth_reads.txt 不存在")
            return {}

        gt_dict = {}
        with open(gt_path, 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    gt_dict[parts[0]] = (int(parts[1]), parts[2], parts[3])

        print(f"   ✅ 加载 {len(gt_dict)} 条Read-Level GT")
        return gt_dict

    def _load_cluster_gt(self) -> Dict[int, str]:
        gt_path = os.path.join(self.raw_dir, "ground_truth_clusters.txt")
        if not os.path.exists(gt_path):
            print(f"   ⚠️ ground_truth_clusters.txt 不存在")
            return {}

        gt_dict = {}
        with open(gt_path, 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        gt_dict[int(parts[0])] = parts[1]
                    except ValueError:
                        continue

        print(f"   ✅ 加载 {len(gt_dict)} 个Cluster GT序列")
        return gt_dict

    def _build_gt_mapping(self, feddna_reads: List[str],
                          raw_reads: Dict[str, str],
                          read_gt: Dict[str, Tuple[int, str, str]]) -> List[int]:
        seq_to_id = {seq: rid for rid, seq in raw_reads.items()}

        gt_labels = []
        matched = 0

        for seq in feddna_reads:
            if seq in seq_to_id:
                read_id = seq_to_id[seq]
                if read_id in read_gt:
                    gt_labels.append(read_gt[read_id][0])
                    matched += 1
                else:
                    gt_labels.append(-1)
            else:
                gt_labels.append(-1)

        print(f"   ✅ GT标签匹配: {matched}/{len(feddna_reads)} ({matched / len(feddna_reads) * 100:.1f}%)")
        return gt_labels

    def _load_all_data(self):
        print("\n" + "=" * 60)
        print("📂 加载实验数据")
        print("=" * 60)

        self.reads, initial_labels = self._load_feddna_format()

        if self.labels_path and os.path.exists(self.labels_path):
            print(f"\n🔄 [Iterative] 正在加载 Refined Labels: {self.labels_path}")
            try:
                refined_labels = np.loadtxt(self.labels_path, dtype=int).tolist()

                if len(refined_labels) == len(self.reads):
                    self.clover_labels = refined_labels
                    print(f"   ✅ 成功覆盖标签: {len(self.clover_labels)} 条")

                    changes = sum(1 for x, y in zip(initial_labels, refined_labels) if x != y)
                    print(f"   📉 与初始Clover相比变化数: {changes} ({changes / len(initial_labels) * 100:.1f}%)")

                    noise_count = sum(1 for l in refined_labels if l == -1)
                    print(f"   🗑️ 当前噪声Reads数: {noise_count} ({noise_count / len(refined_labels) * 100:.1f}%)")
                else:
                    print(f"   ❌ 标签数量不匹配! Reads: {len(self.reads)}, Labels: {len(refined_labels)}")
                    print("   ⚠️ 回退使用初始 Clover 标签")
                    self.clover_labels = initial_labels
            except Exception as e:
                print(f"   ❌ Refined Labels 加载失败: {e}")
                print("   ⚠️ 回退使用初始 Clover 标签")
                self.clover_labels = initial_labels
        else:
            self.clover_labels = initial_labels
            if self.labels_path:
                print(f"   ⚠️ 指定的标签文件不存在: {self.labels_path}，已回退到默认")

        raw_reads = self._load_raw_reads()
        read_gt = self._load_read_gt()
        self.gt_cluster_seqs = self._load_cluster_gt()

        if raw_reads and read_gt:
            self.gt_labels = self._build_gt_mapping(self.reads, raw_reads, read_gt)
        else:
            self.gt_labels = [-1] * len(self.reads)
            print(f"   ⚠️ 无法建立GT映射，使用-1填充")

        print(f"\n📊 数据摘要:")
        print(f"   - 总reads: {len(self.reads)}")
        print(f"   - 当前使用簇数: {len(set(self.clover_labels))}")


def seq_to_onehot(seq: str, max_len: int = 150) -> torch.Tensor:
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    seq_padded = seq.ljust(max_len, 'N')[:max_len]
    indices = [base_to_idx.get(base.upper(), 0) for base in seq_padded]
    onehot = torch.zeros(max_len, 4)
    for i, idx in enumerate(indices):
        onehot[i, idx] = 1.0
    return onehot


class Step1Dataset(Dataset):
    def __init__(self, data_loader: CloverDataLoader, max_len: int = 150):
        self.data_loader = data_loader
        self.max_len = max_len
        self.valid_indices = [i for i, label in enumerate(data_loader.clover_labels) if label >= 0]

        print(f"📊 Dataset统计:")
        print(f"   - 有效reads (Label != -1): {len(self.valid_indices)}/{len(data_loader.reads)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        seq = self.data_loader.reads[real_idx]
        clover_label = self.data_loader.clover_labels[real_idx]
        gt_label = self.data_loader.gt_labels[real_idx]
        encoding = seq_to_onehot(seq, self.max_len)

        return {
            'encoding': encoding,
            'clover_label': clover_label,
            'gt_label': gt_label,
            'read_idx': real_idx,
            'sequence': seq
        }


# ===========================================================================
# 🚀 性能修复版采样器
# ===========================================================================
def create_cluster_balanced_sampler(dataset: Step1Dataset,
                                    batch_size: int = 32,
                                    max_clusters_per_batch: int = 5) -> List[List[int]]:
    """
    性能修复版：使用 deque 替代 list(set)
    解决大规模簇数量下死循环/卡死问题
    """
    print("   🔨 正在构建采样器 (Queue优化版)...")

    valid_indices = dataset.valid_indices
    all_labels = dataset.data_loader.clover_labels

    cluster_to_indices = defaultdict(list)
    for idx, real_idx in enumerate(valid_indices):
        label = all_labels[real_idx]
        cluster_to_indices[label].append(idx)

    for cid in cluster_to_indices:
        np.random.shuffle(cluster_to_indices[cid])

    cluster_ptrs = {cid: 0 for cid in cluster_to_indices}

    print(f"   📊 簇分布 (Top 5):")
    cluster_sizes = [(cid, len(indices)) for cid, indices in cluster_to_indices.items()]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    for i, (cid, size) in enumerate(cluster_sizes[:5]):
        print(f"      簇{cid}: {size}")

    batches = []

    # ✅ 优化点 1: 初始化一次列表并打乱
    cluster_ids = list(cluster_to_indices.keys())
    np.random.shuffle(cluster_ids)

    # ✅ 优化点 2: 使用双端队列 (Deque)
    active_queue = deque(cluster_ids)

    while active_queue:
        # 每次从队列头取 max_clusters_per_batch 个
        num_to_select = min(max_clusters_per_batch, len(active_queue))
        selected_clusters = []
        for _ in range(num_to_select):
            selected_clusters.append(active_queue.popleft())

        batch_indices = []
        reads_per_cluster = max(1, batch_size // num_to_select)

        # 记录还有剩余数据的簇，稍后放回队列尾部
        clusters_to_keep = []

        for cluster_id in selected_clusters:
            indices = cluster_to_indices[cluster_id]
            ptr = cluster_ptrs[cluster_id]
            remaining = len(indices) - ptr
            take = min(reads_per_cluster, remaining)

            if take > 0:
                batch_indices.extend(indices[ptr: ptr + take])
                cluster_ptrs[cluster_id] += take
                # 如果还有剩余，加入保留列表
                if cluster_ptrs[cluster_id] < len(indices):
                    clusters_to_keep.append(cluster_id)
            else:
                # 理论上不应进入这里，但为了保险
                pass

        # ✅ 将未消耗完的簇放回队列尾部 (Round Robin)
        for cid in clusters_to_keep:
            active_queue.append(cid)

        if batch_indices:
            batches.append(batch_indices)

    print(f"   📦 生成 {len(batches)} 个Batch，准备就绪！")
    return batches


def create_dynamic_sampler(dataset: Step1Dataset,
                           batch_size: int = 32,
                           max_clusters_per_batch: int = 5,
                           state_path: str = None,
                           round_idx: int = 1) -> List[List[int]]:
    """
    动态采样器（同样应用 Queue 优化）
    """
    # Round 1: 无 state，直接全量
    if round_idx <= 1 or state_path is None or not os.path.exists(state_path):
        print("   📦 Round 1 / 无 state，使用全量采样")
        return create_cluster_balanced_sampler(
            dataset, batch_size=batch_size,
            max_clusters_per_batch=max_clusters_per_batch
        )

    # ---- Round 2+: 读 state，按三区制过滤 ----
    print(f"   📦 Round {round_idx}: 读取 read_state.pt，按三区制采样...")
    state = torch.load(state_path, map_location='cpu')
    zone_ids_full = state['zone_ids']

    valid_indices = dataset.valid_indices
    all_labels = dataset.data_loader.clover_labels

    kept_indices = []

    # 参数：Zone I 抽样率
    ZONE1_SAMPLE_RATE = 0.20

    n_z1, n_z2, n_z3_dropped, n_z1_dropped = 0, 0, 0, 0

    for ds_idx, real_idx in enumerate(valid_indices):
        zone = int(zone_ids_full[real_idx])

        if zone == 3:
            n_z3_dropped += 1
            continue
        elif zone == 1:
            if np.random.random() < ZONE1_SAMPLE_RATE:
                kept_indices.append(ds_idx)
                n_z1 += 1
            else:
                n_z1_dropped += 1
        elif zone == 2:
            kept_indices.append(ds_idx)
            n_z2 += 1
        else:
            continue

    print(f"   📊 动态采样统计:")
    print(f"      Zone I  保留: {n_z1:>7d}  (丢弃 {n_z1_dropped})")
    print(f"      Zone II 保留: {n_z2:>7d}")
    print(f"      Zone III 丢弃:{n_z3_dropped:>7d}")
    print(f"      总保留:       {len(kept_indices)}")

    if len(kept_indices) == 0:
        print("   ⚠️ 动态采样后无数据，回退到全量采样")
        return create_cluster_balanced_sampler(
            dataset, batch_size=batch_size,
            max_clusters_per_batch=max_clusters_per_batch
        )

    # ---- 用保留的 idx 构建 cluster-balanced batches (Queue 优化版) ----
    cluster_to_indices = defaultdict(list)
    for ds_idx in kept_indices:
        real_idx = valid_indices[ds_idx]
        label = all_labels[real_idx]
        cluster_to_indices[label].append(ds_idx)

    for cid in cluster_to_indices:
        np.random.shuffle(cluster_to_indices[cid])

    cluster_ptrs = {cid: 0 for cid in cluster_to_indices}

    # ✅ 初始化队列
    cluster_ids = list(cluster_to_indices.keys())
    np.random.shuffle(cluster_ids)
    active_queue = deque(cluster_ids)

    batches = []
    while active_queue:
        num_sel = min(max_clusters_per_batch, len(active_queue))
        selected = []
        for _ in range(num_sel):
            selected.append(active_queue.popleft())

        batch = []
        per_cluster = max(1, batch_size // num_sel)

        clusters_to_keep = []

        for cid in selected:
            indices = cluster_to_indices[cid]
            ptr = cluster_ptrs[cid]
            rem = len(indices) - ptr
            take = min(per_cluster, rem)

            if take > 0:
                batch.extend(indices[ptr: ptr + take])
                cluster_ptrs[cid] += take
                if cluster_ptrs[cid] < len(indices):
                    clusters_to_keep.append(cid)

        for cid in clusters_to_keep:
            active_queue.append(cid)

        if batch:
            batches.append(batch)

    print(f"   📦 生成 {len(batches)} 个Batch（动态采样版）")
    return batches