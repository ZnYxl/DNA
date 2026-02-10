# models/step1_data.py
"""
修复清单:
  [FIX-#1]  新增 inference_mode，Step2 推理不再受 training_cap 限制
  [FIX-#8]  每轮采样种子加入 round_idx
  [OPT]     numpy 向量化 one-hot 编码 (5-10x 加速)
  [NEW]     GT 标签加载 (id20 序列匹配)
  [NEW]     training_cap 可通过 args 配置
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from collections import defaultdict

# ============================================================
# 高性能 One-Hot 编码 (numpy 向量化)
# ============================================================
_BASE_LUT = np.zeros(256, dtype=np.int64)
for _c, _i in [('A',0),('C',1),('G',2),('T',3),
               ('a',0),('c',1),('g',2),('t',3),
               ('N',0),('n',0)]:
    _BASE_LUT[ord(_c)] = _i


def seq_to_onehot(seq: str, max_len: int) -> torch.Tensor:
    """numpy 向量化版本，比逐字符循环快 5-10 倍"""
    L = min(len(seq), max_len)
    byte_arr = np.frombuffer(seq[:L].encode('ascii'), dtype=np.uint8)
    indices = _BASE_LUT[byte_arr]
    tensor = torch.zeros(max_len, 4)
    tensor[np.arange(L), indices] = 1.0
    return tensor


# ============================================================
# 数据加载器 (通用: Goldman / id20 / ERR036)
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
        self.gt_cluster_seqs: Dict[int, str] = {}

        self._load_all_data()

    def _load_all_data(self):
        print("\n" + "=" * 60)
        print("📂 [DataLoader] Loading Data...")
        print("=" * 60)

        read_path = os.path.join(self.feddna_dir, "read.txt")
        if not os.path.exists(read_path):
            raise FileNotFoundError(f"Missing: {read_path}")

        with open(read_path, 'r') as f:
            current_cluster = -1
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("====="):
                    current_cluster += 1
                else:
                    self.reads.append(line)
                    self.clover_labels.append(current_cluster)

        print(f"   ✅ Reads Loaded: {len(self.reads)}")
        print(f"   ✅ Clusters:     {current_cluster + 1}")

        # 加载 Refined Labels (Round 2+)
        if self.labels_path and os.path.exists(self.labels_path):
            print(f"   🔄 Loading Refined Labels...")
            try:
                refined = np.loadtxt(self.labels_path, dtype=int).tolist()
                if len(refined) == len(self.reads):
                    self.clover_labels = refined
                    noise_cnt = sum(1 for x in refined if x < 0)
                    print(f"   ✅ Labels Updated. Noise: {noise_cnt}")
                else:
                    print(f"   ⚠️ Length Mismatch ({len(refined)} vs {len(self.reads)}). Keeping original.")
            except Exception as e:
                print(f"   ⚠️ Failed to load labels: {e}")

        # 默认 GT = -1 (Goldman 模式)
        self.gt_labels = [-1] * len(self.reads)
        print(f"   ℹ️ GT Labels initialized to -1 (default)")

    def load_gt_tags(self, gt_tags_file: str):
        """
        从 GT 标签文件加载 (id20 格式: 每行 "tag_id sequence")
        通过序列前缀匹配 read.txt 中的 reads
        """
        if not gt_tags_file or not os.path.exists(gt_tags_file):
            print(f"   ⚠️ GT 文件不存在或为 None")
            return

        print(f"   📋 Loading GT tags: {os.path.basename(gt_tags_file)}")

        PREFIX_LEN = 80
        seq_to_tag: Dict[str, int] = {}
        total_lines = 0
        with open(gt_tags_file, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) < 2:
                    continue
                tag_id = int(parts[0])
                seq = parts[1].strip()
                key = seq[:PREFIX_LEN]
                seq_to_tag[key] = tag_id
                total_lines += 1

        matched = 0
        for i, read in enumerate(self.reads):
            key = read[:PREFIX_LEN]
            tag = seq_to_tag.get(key, -1)
            self.gt_labels[i] = tag
            if tag >= 0:
                matched += 1

        rate = matched / max(len(self.reads), 1) * 100
        print(f"      GT 条目: {total_lines}, 匹配: {matched}/{len(self.reads)} ({rate:.1f}%)")


# ============================================================
# Dataset (通用)
# ============================================================
class Step1Dataset(Dataset):
    def __init__(self, data_loader: CloverDataLoader, max_len: int = 150,
                 training_cap: int = 2000000,
                 inference_mode: bool = False,
                 round_idx: int = 1):
        """
        Args:
            training_cap:   训练时的样本上限
            inference_mode: [FIX-#1] True 时忽略 training_cap，使用全部有效样本
            round_idx:      [FIX-#8] 轮次索引，用于变化采样种子
        """
        self.data_loader = data_loader
        self.max_len = max_len

        full_valid_indices = [i for i, label in enumerate(data_loader.clover_labels) if label >= 0]
        total_valid = len(full_valid_indices)

        if inference_mode:
            self.valid_indices = full_valid_indices
            print(f"   📊 Inference Mode: {len(self.valid_indices)} samples (全量)")
        elif total_valid > training_cap:
            seed = 42 + round_idx
            rng = np.random.RandomState(seed)
            self.valid_indices = rng.choice(
                full_valid_indices, training_cap, replace=False
            ).tolist()
            print(f"   ✂️ Training: {total_valid} → {training_cap} (seed={seed})")
        else:
            self.valid_indices = full_valid_indices

        print(f"   📊 Dataset Size: {len(self.valid_indices)} samples")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        seq = self.data_loader.reads[real_idx]
        encoding = seq_to_onehot(seq, self.max_len)

        return {
            'encoding': encoding,
            'clover_label': self.data_loader.clover_labels[real_idx],
            'gt_label': self.data_loader.gt_labels[real_idx],
            'read_idx': real_idx,
        }


# ============================================================
# 采样器 (通用, 贪心填充版)
#
# [OPT] id20 有 756K 簇平均仅 2.6 reads/簇
#       旧版固定 max_clusters_per_batch=8 → batch≈21 → 79K batches → 极慢
#       新版贪心填充: 不断取簇直到填满 batch_size，大幅减少 batch 数
# ============================================================
def create_cluster_balanced_sampler(dataset: Step1Dataset,
                                    batch_size: int = 256,
                                    max_clusters_per_batch: int = 64):
    """
    贪心填充策略:
      1. 打乱簇顺序
      2. 逐个簇倒入当前 batch，直到 batch_size 满或达到 max_clusters 上限
      3. 大簇(>batch_size)单独切片
    """
    print("   🔨 构建采样器 (贪心填充)...")
    valid_indices = dataset.valid_indices
    all_labels = dataset.data_loader.clover_labels

    cluster_to_indices = defaultdict(list)
    for idx, real_idx in enumerate(valid_indices):
        label = all_labels[real_idx]
        cluster_to_indices[label].append(idx)

    # 过滤空簇和单条簇 (对比学习至少需要2条)
    valid_clusters = {cid: idxs for cid, idxs in cluster_to_indices.items()
                      if len(idxs) >= 2}
    singleton_indices = [idxs[0] for idxs in cluster_to_indices.values()
                         if len(idxs) == 1]

    cluster_ids = list(valid_clusters.keys())
    np.random.shuffle(cluster_ids)

    batches = []
    current_batch = []
    current_n_clusters = 0

    for cid in cluster_ids:
        idxs = valid_clusters[cid]

        # 大簇: 切片处理
        if len(idxs) > batch_size:
            # 先 flush 当前 batch
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_n_clusters = 0
            # 切片
            for i in range(0, len(idxs), batch_size):
                batches.append(idxs[i:i + batch_size])
            continue

        # 能放进当前 batch 吗？
        if (len(current_batch) + len(idxs) > batch_size or
                current_n_clusters >= max_clusters_per_batch):
            # flush
            if current_batch:
                batches.append(current_batch)
            current_batch = list(idxs)
            current_n_clusters = 1
        else:
            current_batch.extend(idxs)
            current_n_clusters += 1

    # flush 最后一个 batch
    if current_batch:
        batches.append(current_batch)

    # 单条 reads: 合并成大 batch (它们不参与对比但参与重建损失)
    if singleton_indices:
        for i in range(0, len(singleton_indices), batch_size):
            batches.append(singleton_indices[i:i + batch_size])

    # 统计
    sizes = [len(b) for b in batches]
    avg_size = sum(sizes) / max(len(sizes), 1)
    n_valid_c = len(valid_clusters)
    n_single  = len(singleton_indices)

    print(f"   📦 Batches: {len(batches)} (avg {avg_size:.0f} samples/batch)")
    print(f"      有效簇(≥2): {n_valid_c}, 单条簇: {n_single}")

    return batches


def create_dynamic_sampler(dataset, batch_size=256, max_clusters_per_batch=64,
                           state_path=None, round_idx=1):
    return create_cluster_balanced_sampler(dataset, batch_size, max_clusters_per_batch)