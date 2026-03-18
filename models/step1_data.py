# models/step1_data.py (FIXED VERSION)
"""
修复清单:
  [FIX-#1]  新增 inference_mode，Step2 推理不再受 training_cap 限制
  [FIX-#8]  每轮采样种子加入 round_idx
  [FIX-P0]  Step1Dataset 加入 consensus_dict，__getitem__ 返回 consensus_target
  [FIX-P0]  Round 2+ 采样颗粒度改为"簇级"（困难簇100%，完美簇动态采样）
  [OPT]     numpy 向量化 one-hot 编码 (5-10x 加速)
  [NEW]     GT 标签加载 (id20 序列匹配)
  [NEW]     training_cap 可通过 args 配置
  
  [BUG-FIX-2025-03-16] 修复'N'碱基编码错误 (G老师发现)
                       - 使用虚拟通道技巧，确保'N'被编码为[0,0,0,0]
                       - 与模型mask机制完美联动
  [BUG-FIX-2025-03-16] 添加consensus_dict格式验证
  [BUG-FIX-2025-03-16] default_consensus使用时报警告
  
  [TUNING-2025-03-16]  采样策略优化（基于实验数据调整）
                       - Round 1: 强制全量训练（确保baseline稳定）
                       - 困难簇阈值: 10% (更聚焦真正需要改进的簇)
                       - 完美簇采样率: R2=40%, R3+=25% (防止灾难性遗忘)
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from collections import defaultdict
import warnings

# ============================================================
# 高性能 One-Hot 编码 (numpy 向量化) - FIXED
# ============================================================
# [BUG-FIX] G老师发现：使用虚拟通道处理'N'碱基，确保'N'被编码为[0,0,0,0]
# 原问题：'N'被错误映射到索引0（与'A'相同），导致模型产生虚假置信度
# 解决方案：使用第5通道作为虚拟通道，'N'的1.0写入第5通道，截断后前4通道全为0
_BASE_LUT = np.full(256, 4, dtype=np.int64)  # 默认索引4（虚拟通道）
for _c, _i in [('A',0),('C',1),('G',2),('T',3),
               ('a',0),('c',1),('g',2),('t',3)]:
    _BASE_LUT[ord(_c)] = _i
# 'N', 'n' 以及任何异常字符保持索引4


def seq_to_onehot(seq: str, max_len: int) -> torch.Tensor:
    """
    numpy 向量化版本，比逐字符循环快 5-10 倍
    
    [BUG-FIX 2025-03-16] 使用5通道技巧处理'N'碱基：
      - 正常碱基(A/C/G/T): 映射到索引0-3 → [1,0,0,0] / [0,1,0,0] / [0,0,1,0] / [0,0,0,1]
      - 未知碱基(N)或异常字符: 映射到索引4 → 第4通道置1
      - 返回前4个通道: 'N'位置自动变成[0,0,0,0]
      - 与step1_model.py的mask机制完美联动 (mask = consensus.sum(dim=-1) > 0)
    
    示例:
        seq = "ACGNT" → indices = [0,1,2,4,3]
        tensor[5通道] = [[1,0,0,0,0],  # A
                         [0,1,0,0,0],  # C
                         [0,0,1,0,0],  # G
                         [0,0,0,0,1],  # N (第4通道)
                         [0,0,0,1,0]]  # T
        返回前4通道:    [[1,0,0,0],    # A
                         [0,1,0,0],    # C
                         [0,0,1,0],    # G
                         [0,0,0,0],    # N ✓ 正确！
                         [0,0,0,1]]    # T
    """
    L = min(len(seq), max_len)
    byte_arr = np.frombuffer(seq[:L].encode('ascii'), dtype=np.uint8)
    indices = _BASE_LUT[byte_arr]
    
    # 开辟5个通道 (A,C,G,T,虚拟通道)
    tensor = torch.zeros(max_len, 5)
    tensor[np.arange(L), indices] = 1.0
    
    # 返回前4个通道：'N'的1.0在第4通道，截断后前4个通道全是0
    return tensor[:, :4]  # shape: (max_len, 4)


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
            raise FileNotFoundError(f"read.txt not found: {read_path}")

        # 解析 read.txt：预处理脚本格式为 "reads → 分隔符 → reads → 分隔符"
        # 即分隔符出现在每个簇的 reads 之后，第一个簇的 reads 前面没有分隔符
        # 因此 current_cluster 从 0 开始，遇到分隔符时切换到下一个簇
        current_cluster = 0
        with open(read_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("====="):
                    current_cluster += 1   # 分隔符 = 当前簇结束，下一个 read 属于新簇
                else:
                    self.reads.append(line)
                    self.clover_labels.append(current_cluster)

        # 最后一个分隔符之后 current_cluster 多加了 1，实际簇数是 current_cluster
        n_clusters = current_cluster   # 分隔符数量 == 簇数量（每簇结尾各一个）
        print(f"   ✅ Reads: {len(self.reads)}")
        print(f"   ✅ Clusters (from read.txt): {n_clusters}")

        # Round 2+：用 refined_labels 覆盖 Clover 初始标签
        if self.labels_path and os.path.exists(self.labels_path):
            refined = np.loadtxt(self.labels_path, dtype=int).tolist()
            if len(refined) == len(self.reads):
                self.clover_labels = refined
                noise_cnt = sum(1 for x in refined if x < 0)
                print(f"   ✅ Labels (refined): {self.labels_path}, noise={noise_cnt}")
            else:
                print(f"   ⚠️ refined_labels 长度不匹配 ({len(refined)} vs {len(self.reads)})，保留 Clover 标签")

        self.gt_labels = [-1] * len(self.reads)

        # 加载 ref.txt → Round 1 consensus_dict（tag多数投票结果，完全自监督）
        self.ref_seqs: Dict[int, str] = {}
        ref_path = os.path.join(self.feddna_dir, "ref.txt")
        if os.path.exists(ref_path):
            with open(ref_path, 'r') as rf:
                for cid, line in enumerate(rf):
                    seq = line.strip()
                    if seq:
                        self.ref_seqs[cid] = seq
            print(f"   ✅ ref.txt: {len(self.ref_seqs)} 条 reference 序列")
        else:
            print(f"   ⚠️ ref.txt 未找到: {ref_path}")

        valid = sum(1 for l in self.clover_labels if l >= 0)
        unique = len(set(l for l in self.clover_labels if l >= 0))
        print(f"   ✅ 有效 reads: {valid}, 唯一簇: {unique}")

    def load_gt_tags(self, gt_tags_file: str):
        """
        加载 GT 标签。文件格式: cluster_id<TAB>sequence
        策略: 先建 sequence→cluster_id 字典，再逐条 read 做精确序列匹配。
        """
        if not os.path.exists(gt_tags_file):
            print(f"   ⚠️ GT tags 文件不存在: {gt_tags_file}")
            return
        print(f"   📋 加载 GT tags: {gt_tags_file}")

        # 建序列 → cluster_id 字典
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

        # 逐条 read 查字典
        matched = 0
        for i, read in enumerate(self.reads):
            cluster_id = seq_to_cluster.get(read.upper())
            if cluster_id is not None:
                self.gt_labels[i] = cluster_id
                matched += 1

        rate = matched / max(len(self.reads), 1) * 100
        print(f"      GT 条目: {total_lines}, 匹配: {matched}/{len(self.reads)} ({rate:.1f}%)")


# ============================================================
# Dataset (通用)
# [FIX-P0] 加入 consensus_dict 支持
# [FIX-P0] Round 2+ 改为簇级采样
# [BUG-FIX] 添加格式验证和警告机制
# ============================================================
class Step1Dataset(Dataset):
    def __init__(self, data_loader: CloverDataLoader, max_len: int = 150,
                 training_cap: int = 99999000000,
                 inference_mode: bool = False,
                 round_idx: int = 1,
                 consensus_dict: Optional[Dict[int, torch.Tensor]] = None,
                 cluster_change_info: Optional[Dict[int, float]] = None,
                 cv_threshold: float = 0.3):
        """
        Args:
            training_cap:        训练时的样本上限（Round 2+有效，Round 1强制全量）
            inference_mode:      True 时忽略 training_cap，使用全部有效样本
            round_idx:           轮次索引，用于变化采样种子
            consensus_dict:      {cluster_id: Tensor(L, 4)} 每个簇的伪 reference one-hot
            cluster_change_info: {cluster_id: CV值} Round2+ 簇级采样依据
                                  困难簇 (CV >= cv_threshold): 100% reads（混簇，需重训）
                                  完美簇 (CV <  cv_threshold): R2=40%, R3+=25%（纯净簇，保结构）
            cv_threshold:        CV 困难/完美 分界线，默认 0.3（可调）
        """
        self.data_loader = data_loader
        self.max_len = max_len
        self.round_idx = round_idx
        self.consensus_dict = consensus_dict or {}
        self.cv_threshold = cv_threshold

        # [BUG-FIX] 验证consensus_dict格式
        if self.consensus_dict:
            self._validate_consensus_dict()

        # 默认 consensus: 全零向量（与padding一致，触发mask机制）
        # [BUG-FIX] 改为全零，而非全A，这样更安全
        self._default_consensus = torch.zeros(max_len, 4)
        self._default_consensus_used = False  # 追踪是否使用了default

        full_valid_indices = [i for i, label in enumerate(data_loader.clover_labels) if label >= 0]
        total_valid = len(full_valid_indices)

        if inference_mode:
            # Step 2 全量推理
            self.valid_indices = full_valid_indices
            print(f"   📊 Inference Mode: {len(self.valid_indices)} samples (全量)")

        elif round_idx == 1:
            # =====================================================
            # Round 1: 强制全量训练（忽略training_cap）
            # 确保baseline稳定，为后续迭代提供可靠起点
            # =====================================================
            self.valid_indices = full_valid_indices
            print(f"   📊 Round 1 (强制全量): {len(self.valid_indices)} samples")

        elif cluster_change_info is not None and round_idx >= 2:
            # =====================================================
            # Round 2+ 簇级采样 (Curriculum Learning)
            #
            # cluster_change_info 现在存储的是 CV（变异系数）:
            #   CV = std(strength) / mean(strength)  (来自 step2_runner)
            #
            # 困难簇 (CV >= cv_threshold): 混簇，strength 分散 → 100% reads
            # 完美簇 (CV <  cv_threshold): 纯净簇，结构稳定  → R2=40%, R3+=25%
            #
            # cv_threshold 默认 0.3，可在 dataset 构造时通过 cv_threshold 参数覆盖
            # =====================================================
            cv_threshold = getattr(self, 'cv_threshold', 0.3)
            print(f"   🔀 Round {round_idx} 簇级采样 (CV困难度, threshold={cv_threshold})...")
            rng = np.random.RandomState(42 + round_idx)

            cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
            for i in full_valid_indices:
                label = data_loader.clover_labels[i]
                if label >= 0:
                    cluster_to_indices[label].append(i)

            selected = []
            n_hard = n_easy = n_hard_reads = n_easy_reads = 0

            # 动态采样率：Round 2多采样（防遗忘），Round 3+稳定后降低
            easy_ratio = 0.80 if round_idx == 2 else 0.50

            for cluster_id, idxs in cluster_to_indices.items():
                cv = cluster_change_info.get(cluster_id, 0.0)
                if cv >= cv_threshold:  # 困难簇：CV高=混簇=需重点训练
                    selected.extend(idxs)
                    n_hard += 1
                    n_hard_reads += len(idxs)
                else:                   # 完美簇：CV低=纯净=采样保持结构
                    k = max(1, int(len(idxs) * easy_ratio))
                    chosen = rng.choice(idxs, k, replace=False).tolist()
                    selected.extend(chosen)
                    n_easy += 1
                    n_easy_reads += len(chosen)

            self.valid_indices = selected
            print(f"   ✅ 困难簇 (CV≥{cv_threshold}): {n_hard} 簇 × 100% = {n_hard_reads} reads")
            print(f"   ✅ 完美簇 (CV<{cv_threshold}): {n_easy} 簇 × {int(easy_ratio*100)}%  = {n_easy_reads} reads")
            print(f"   📊 Dataset Size: {len(self.valid_indices)} / {total_valid} samples")

        else:
            # Fallback: 如果Round 2+没有cluster_change_info，使用training_cap
            if total_valid > training_cap:
                seed = 42 + round_idx
                rng = np.random.RandomState(seed)
                self.valid_indices = rng.choice(
                    full_valid_indices, training_cap, replace=False
                ).tolist()
                print(f"   ✂️ Training (fallback): {total_valid} → {training_cap} (seed={seed})")
            else:
                self.valid_indices = full_valid_indices

            print(f"   📊 Dataset Size: {len(self.valid_indices)} samples")

    def _validate_consensus_dict(self):
        """[BUG-FIX] 验证consensus_dict的格式"""
        issues = []
        for cluster_id, consensus in list(self.consensus_dict.items())[:10]:  # 只检查前10个
            if not isinstance(consensus, torch.Tensor):
                issues.append(f"cluster {cluster_id}: 不是Tensor")
            elif consensus.shape != (self.max_len, 4):
                issues.append(f"cluster {cluster_id}: shape {consensus.shape} != ({self.max_len}, 4)")
            elif consensus.dim() != 2:
                issues.append(f"cluster {cluster_id}: 维度 {consensus.dim()} != 2")
        
        if issues:
            warnings.warn(
                f"⚠️ consensus_dict 格式问题:\n" + "\n".join(issues[:5]),
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

        # [FIX-P0] 提供 consensus_target
        if clover_label in self.consensus_dict:
            consensus_target = self.consensus_dict[clover_label]
        else:
            # [BUG-FIX] 使用default时报警告
            if not self._default_consensus_used:
                warnings.warn(
                    f"⚠️ cluster {clover_label} 缺失consensus，使用default（全零向量）",
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
# 采样器 (通用, 贪心填充版)
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

        if len(idxs) > batch_size:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_n_clusters = 0
            for i in range(0, len(idxs), batch_size):
                batches.append(idxs[i:i + batch_size])
            continue

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

    if singleton_indices:
        for i in range(0, len(singleton_indices), batch_size):
            batches.append(singleton_indices[i:i + batch_size])

    sizes = [len(b) for b in batches]
    avg_size = sum(sizes) / max(len(sizes), 1)
    n_valid_c = len(valid_clusters)
    n_single = len(singleton_indices)

    print(f"   📦 Batches: {len(batches)} (avg {avg_size:.0f} samples/batch)")
    print(f"      有效簇(≥2): {n_valid_c}, 单条簇: {n_single}")

    return batches


def create_dynamic_sampler(dataset, batch_size=256, max_clusters_per_batch=64,
                           state_path=None, round_idx=1):
    return create_cluster_balanced_sampler(dataset, batch_size, max_clusters_per_batch)