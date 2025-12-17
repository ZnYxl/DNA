# models/step1_data.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter


class CloverDataLoader:
    """
    适配你的数据格式的加载器
    ✅ 已修改支持迭代闭环：可以加载 refined_labels.txt
    """

    def __init__(self, experiment_dir: str, labels_path: str = None):
        """
        Args:
            experiment_dir: 实验目录
            labels_path: (可选) 上一轮生成的 refined_labels.txt 路径
                         如果不传，默认加载 03_FedDNA_In/read.txt 里的原始标签
        """
        self.experiment_dir = experiment_dir
        self.raw_dir = os.path.join(experiment_dir, "01_RawData")
        self.feddna_dir = os.path.join(experiment_dir, "03_FedDNA_In")
        self.labels_path = labels_path  # ✅ 保存外部标签路径

        # 数据存储
        self.reads: List[str] = []
        self.clover_labels: List[int] = []  # Clover聚类结果 (会被 refined labels 覆盖)
        self.gt_labels: List[int] = []  # Ground Truth标签
        self.gt_cluster_seqs: Dict[int, str] = {}  # GT簇的参考序列

        self._load_all_data()

    def _load_feddna_format(self) -> Tuple[List[str], List[int]]:
        """
        加载FedDNA格式数据 (read.txt)
        格式：按簇分组，用=======分隔
        """
        read_path = os.path.join(self.feddna_dir, "read.txt")

        if not os.path.exists(read_path):
            raise FileNotFoundError(f"找不到 read.txt: {read_path}")

        reads = []
        labels = []
        current_cluster = -1  # 从-1开始，第一个分隔符后变成0

        print(f"📂 加载FedDNA格式数据: {read_path}")

        with open(read_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 检测分隔符
                if line.startswith("====="):
                    current_cluster += 1
                else:
                    # 这是一个read序列
                    reads.append(line)
                    labels.append(current_cluster)

        print(f"   ✅ 加载 {len(reads)} 条reads，{current_cluster + 1} 个Clover簇")
        return reads, labels

    def _load_raw_reads(self) -> Dict[str, str]:
        """
        加载原始reads: raw_reads.txt
        格式: Read_ID \t Sequence
        """
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
        """
        加载Read级别GT: ground_truth_reads.txt
        格式: Read_ID \t Cluster_ID \t Ref_Seq \t Quality
        """
        gt_path = os.path.join(self.raw_dir, "ground_truth_reads.txt")

        if not os.path.exists(gt_path):
            print(f"   ⚠️ ground_truth_reads.txt 不存在")
            return {}

        gt_dict = {}
        with open(gt_path, 'r') as f:
            header = f.readline()  # 跳过表头
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    read_id = parts[0]
                    cluster_id = int(parts[1])
                    ref_seq = parts[2]
                    quality = parts[3]
                    gt_dict[read_id] = (cluster_id, ref_seq, quality)

        print(f"   ✅ 加载 {len(gt_dict)} 条Read-Level GT")
        return gt_dict

    def _load_cluster_gt(self) -> Dict[int, str]:
        """
        加载Cluster级别GT: ground_truth_clusters.txt
        格式: Cluster_ID \t Ref_Seq
        """
        gt_path = os.path.join(self.raw_dir, "ground_truth_clusters.txt")

        if not os.path.exists(gt_path):
            print(f"   ⚠️ ground_truth_clusters.txt 不存在")
            return {}

        gt_dict = {}
        with open(gt_path, 'r') as f:
            header = f.readline()  # 跳过表头
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        cluster_id = int(parts[0])
                        ref_seq = parts[1]
                        gt_dict[cluster_id] = ref_seq
                    except ValueError:
                        continue

        print(f"   ✅ 加载 {len(gt_dict)} 个Cluster GT序列")
        return gt_dict

    def _build_gt_mapping(self, feddna_reads: List[str],
                          raw_reads: Dict[str, str],
                          read_gt: Dict[str, Tuple[int, str, str]]) -> List[int]:
        """
        建立FedDNA reads到GT标签的映射
        通过序列匹配找到每个read对应的GT簇ID
        """
        # 反向映射：sequence -> read_id
        seq_to_id = {seq: rid for rid, seq in raw_reads.items()}

        gt_labels = []
        matched = 0

        for seq in feddna_reads:
            if seq in seq_to_id:
                read_id = seq_to_id[seq]
                if read_id in read_gt:
                    gt_cluster_id = read_gt[read_id][0]
                    gt_labels.append(gt_cluster_id)
                    matched += 1
                else:
                    gt_labels.append(-1)
            else:
                gt_labels.append(-1)

        print(f"   ✅ GT标签匹配: {matched}/{len(feddna_reads)} ({matched / len(feddna_reads) * 100:.1f}%)")
        return gt_labels

    def _load_all_data(self):
        """加载所有数据"""
        print("\n" + "=" * 60)
        print("📂 加载实验数据")
        print("=" * 60)

        # 1. 加载FedDNA格式的reads和Clover标签 (作为基础)
        self.reads, initial_labels = self._load_feddna_format()

        # =========================================================
        # ✅ 核心修改：尝试加载 Refined Labels (用于迭代训练)
        # =========================================================
        if self.labels_path and os.path.exists(self.labels_path):
            print(f"\n🔄 [Iterative] 正在加载 Refined Labels: {self.labels_path}")
            try:
                # 假设 refined_labels.txt 是纯数字，每行一个 label
                # 使用 numpy 读取，因为它是最稳健的
                refined_labels = np.loadtxt(self.labels_path, dtype=int).tolist()

                if len(refined_labels) == len(self.reads):
                    self.clover_labels = refined_labels
                    print(f"   ✅ 成功覆盖标签: {len(self.clover_labels)} 条")

                    # 统计一下变化 (监控迭代效果)
                    changes = sum(1 for x, y in zip(initial_labels, refined_labels) if x != y)
                    print(f"   📉 与初始Clover相比变化数: {changes} ({(changes/len(initial_labels))*100:.1f}%)")
                    
                    # 统计噪声比例
                    noise_count = sum(1 for l in refined_labels if l == -1)
                    print(f"   🗑️ 当前噪声Reads数: {noise_count} ({(noise_count/len(refined_labels))*100:.1f}%)")
                else:
                    print(f"   ❌ 标签数量不匹配! Reads: {len(self.reads)}, Labels: {len(refined_labels)}")
                    print("   ⚠️ 回退使用初始 Clover 标签")
                    self.clover_labels = initial_labels
            except Exception as e:
                print(f"   ❌ Refined Labels 加载失败: {e}")
                print("   ⚠️ 回退使用初始 Clover 标签")
                self.clover_labels = initial_labels
        else:
            # 默认情况：使用原始 Clover 标签
            self.clover_labels = initial_labels
            if self.labels_path:
                print(f"   ⚠️ 指定的标签文件不存在: {self.labels_path}，已回退到默认")

        # 2. 加载原始数据和GT
        raw_reads = self._load_raw_reads()
        read_gt = self._load_read_gt()
        self.gt_cluster_seqs = self._load_cluster_gt()

        # 3. 建立GT映射
        if raw_reads and read_gt:
            self.gt_labels = self._build_gt_mapping(self.reads, raw_reads, read_gt)
        else:
            self.gt_labels = [-1] * len(self.reads)
            print(f"   ⚠️ 无法建立GT映射，使用-1填充")

        print(f"\n📊 数据摘要:")
        print(f"   - 总reads: {len(self.reads)}")
        print(f"   - 当前使用簇数: {len(set(self.clover_labels))}")
        print(f"   - GT簇数: {len(self.gt_cluster_seqs)}")
        print(f"   - 序列长度范围: {min(len(r) for r in self.reads)} - {max(len(r) for r in self.reads)}")


def seq_to_onehot(seq: str, max_len: int = 150) -> torch.Tensor:
    """DNA序列转one-hot编码"""
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}  # N当作A处理

    # 填充或截断到最大长度
    seq_padded = seq.ljust(max_len, 'N')[:max_len]

    # 转换为索引
    indices = [base_to_idx.get(base.upper(), 0) for base in seq_padded]

    # 转one-hot
    onehot = torch.zeros(max_len, 4)
    for i, idx in enumerate(indices):
        onehot[i, idx] = 1.0

    return onehot


class Step1Dataset(Dataset):
    """
    步骤一的数据集
    """

    def __init__(self, data_loader: CloverDataLoader, max_len: int = 150):
        self.data_loader = data_loader
        self.max_len = max_len

        # 过滤掉噪声reads (Clover标签为-1的)
        # 注意：这里的 self.data_loader.clover_labels 可能已经是 refined 过的标签了
        self.valid_indices = [i for i, label in enumerate(data_loader.clover_labels) if label >= 0]

        print(f"📊 Dataset统计:")
        print(f"   - 有效reads (Label != -1): {len(self.valid_indices)}/{len(data_loader.reads)}")
        print(f"   - 噪声reads (被过滤): {len(data_loader.reads) - len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        seq = self.data_loader.reads[real_idx]
        clover_label = self.data_loader.clover_labels[real_idx]
        gt_label = self.data_loader.gt_labels[real_idx]

        # 序列编码
        encoding = seq_to_onehot(seq, self.max_len)

        return {
            'encoding': encoding,  # (L, 4)
            'clover_label': clover_label,  # Clover聚类标签 (可能是 refined 的)
            'gt_label': gt_label,  # Ground Truth标签
            'read_idx': real_idx,  # 原始索引
            'sequence': seq  # 原始序列
        }


def create_cluster_balanced_sampler(dataset: Step1Dataset,
                                    batch_size: int = 32,
                                    max_clusters_per_batch: int = 5) -> List[List[int]]:
    """
    创建簇平衡的batch采样器
    确保每个batch包含多个簇，但不会太多（避免内存爆炸）
    """
    # 按Clover标签分组
    cluster_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset[idx]
        cluster_label = item['clover_label']
        cluster_to_indices[cluster_label].append(idx)

    print(f"📊 簇分布 (Top 10):")
    cluster_sizes = [(cid, len(indices)) for cid, indices in cluster_to_indices.items()]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    for i, (cid, size) in enumerate(cluster_sizes[:10]):  # 显示前10个最大的簇
        print(f"   簇{cid}: {size} reads")
    if len(cluster_sizes) > 10:
        print(f"   ... 还有 {len(cluster_sizes) - 10} 个簇")

    # 生成batch
    batches = []
    cluster_ids = list(cluster_to_indices.keys())
    np.random.shuffle(cluster_ids)

    while cluster_ids:
        # 随机选择几个簇
        num_clusters = min(max_clusters_per_batch, len(cluster_ids))
        selected_clusters = np.random.choice(cluster_ids, size=num_clusters, replace=False)

        # 从选中的簇中采样reads
        batch_indices = []
        for cluster_id in selected_clusters:
            cluster_indices = cluster_to_indices[cluster_id]

            # 每个簇贡献的reads数量
            reads_per_cluster = batch_size // num_clusters
            sample_size = min(reads_per_cluster, len(cluster_indices))

            if sample_size > 0:
                sampled = np.random.choice(cluster_indices, size=sample_size, replace=False)
                batch_indices.extend(sampled)

                # 移除已使用的indices
                for idx in sampled:
                    cluster_to_indices[cluster_id].remove(idx)

        # 移除空簇
        cluster_ids = [cid for cid in cluster_ids if len(cluster_to_indices[cid]) > 0]

        if batch_indices:
            batches.append(batch_indices)

    print(f"📦 生成 {len(batches)} 个batch")
    return batches