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
                       - 困难簇阈值: CV >= 0.3（变异系数）
                       - 完美簇采样率: R2=80%, R3+=50% (防止灾难性遗忘)

  [G老师-BUG-FIX-漏洞1] 静态采样 → 动态采样
                       - 原问题: Dataset.__init__固定valid_indices，5个epoch看同样数据
                       - 修复: Dataset保留全量索引，采样逻辑移入create_cluster_balanced_sampler
                       - 效果: 每个epoch重新随机抽取完美簇的50%，覆盖接近全量数据

  [G老师-BUG-FIX-漏洞2] 大簇零头单独成批 → 零头流入缓冲区
                       - 原问题: 300条reads的簇切割后余44条单独成一个batch，batch内无负样本
                       - 修复: 整块正常切，余数extend进current_batch与后续小簇混合
                       - 效果: 消除单簇batch，对比学习负样本始终存在

  [G老师-BUG-FIX-漏洞3] singleton扎堆末尾 → 均匀分散
                       - 原问题: 所有单条簇堆在每个epoch最后，导致连续多个batch对比Loss=0
                       - 修复: singleton随机打散后round-robin插入已有batch的空余槽位
                       - 效果: 梯度平滑，singleton作为负样本贡献排斥力
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
# [G老师-漏洞1-FIX] Dataset 保留全量索引，采样逻辑移入采样器，实现每epoch动态采样
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
        """
        Args:
            training_cap:        训练时的样本上限（Round 2+有效，Round 1强制全量）
            inference_mode:      True 时忽略 training_cap，使用全部有效样本（含label=-1）
            round_idx:           轮次索引，用于变化采样种子
            consensus_dict:      {cluster_id: Tensor(L, 4)} 每个簇的伪 reference one-hot
            cluster_change_info: {cluster_id: CV值} Round2+ 簇级采样依据
                                  困难簇 (CV >= cv_threshold): 100% reads（混簇，需重训）
                                  完美簇 (CV <  cv_threshold): R2=80%, R3+=50%（纯净簇，保结构）
            cv_threshold:        CV 困难/完美 分界线，默认 0.3（可调）
            max_reads_per_cluster: 训练时每簇最多采样的 reads 数，0=不限制（默认）。
                                  与 FedDNA 训练设计保持一致（FedDNA 用 5-30），
                                  建议设 50。仅影响训练采样，不影响 Step2 推理（全量）。

        [G老师-漏洞1-FIX]
            Round 2+ 时，Dataset 不再在 __init__ 里固定采样结果。
            self.valid_indices 保留全量有效 reads，
            cluster_change_info / cv_threshold / easy_ratio 存为实例变量，
            由 create_cluster_balanced_sampler 在每个 epoch 构建 batch 时动态采样。
            这样 5 个 epoch 内每次都是全新的随机子集，覆盖接近完整数据。
        """
        self.data_loader = data_loader
        self.max_len = max_len
        self.round_idx = round_idx
        self.consensus_dict = consensus_dict or {}
        self.cv_threshold = cv_threshold

        # [G老师-漏洞1-FIX] 把采样元信息存在 Dataset 上，供采样器每 epoch 读取
        self.cluster_change_info = cluster_change_info  # 采样器需要用
        self.easy_ratio = 0.80 if round_idx == 2 else 0.50  # Round2=80%, Round3+=50%
        self.max_reads_per_cluster = max_reads_per_cluster  # 簇内采样上限，0=不限

        # [BUG-FIX] 验证 consensus_dict 格式
        if self.consensus_dict:
            self._validate_consensus_dict()

        # 默认 consensus: 全零向量（与 padding 一致，触发 mask 机制）
        self._default_consensus = torch.zeros(max_len, 4)
        self._default_consensus_used = False

        full_valid_indices = [i for i, label in enumerate(data_loader.clover_labels) if label >= 0]
        total_valid = len(full_valid_indices)

        # inference_mode 需要包含 label=-1 的 reads（死数据复活入口）
        all_indices = list(range(len(data_loader.clover_labels)))

        if inference_mode:
            # Step 2 全量推理：包含 label=-1 的 reads（死数据复活入口）
            # 注意：step2_runner 里的 labels_tensor 构建、Zone 划分、质心计算
            #       内部均有 label >= 0 的合法性校验，放行 -1 不会引起崩溃。
            self.valid_indices = all_indices
            n_negative = len(all_indices) - total_valid
            print(f"   📊 Inference Mode (全量): {len(self.valid_indices)} samples "
                  f"(含 {n_negative} 条 label=-1 的待复活 reads)")

        elif round_idx == 1:
            # =====================================================
            # Round 1: 强制全量训练（忽略 training_cap）
            # 确保 baseline 稳定，为后续迭代提供可靠起点
            # =====================================================
            self.valid_indices = full_valid_indices
            print(f"   📊 Round 1 (强制全量): {len(self.valid_indices)} samples")

        elif cluster_change_info is not None and round_idx >= 2:
            # =====================================================
            # [G老师-漏洞1-FIX] Round 2+ 动态采样
            #
            # 不在这里固定采样！valid_indices 保留全量，
            # 采样逻辑交给 create_cluster_balanced_sampler 每 epoch 执行。
            # self.cluster_change_info / self.easy_ratio 已存好，采样器会读它们。
            #
            # [G老师-索引对齐-FIX]
            # full_valid_indices 是"永久全量池"，Sampler 每次从这里抽子集。
            # valid_indices 会被 Sampler 每 epoch 覆盖为本轮子集，
            # 确保 __getitem__(idx) 和 Sampler 发出的 idx 指向同一条 read。
            # =====================================================
            self.valid_indices      = full_valid_indices  # 初始值=全量，会被Sampler覆盖
            self.full_valid_indices = full_valid_indices  # 永久备份，Sampler每epoch从这里采
            print(f"   📊 Round {round_idx} (动态采样模式，全量索引保留): "
                  f"{len(self.valid_indices)} samples")
            print(f"      采样器将在每 epoch 执行: "
                  f"困难簇100% / 完美簇{int(self.easy_ratio*100)}% (CV阈值={cv_threshold})")

        else:
            # Fallback: 如果 Round 2+ 没有 cluster_change_info，使用 training_cap
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
        """[BUG-FIX] 验证 consensus_dict 的格式"""
        issues = []
        for cluster_id, consensus in list(self.consensus_dict.items())[:10]:
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
            # [BUG-FIX] 使用 default 时报警告
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
# [G老师-漏洞1-FIX] Round 2+ 动态采样：每次调用重新随机抽完美簇子集
# [G老师-漏洞2-FIX] 大簇零头流入 current_batch，不单独成批
# [G老师-漏洞3-FIX] singleton 均匀分散到已有 batch，不扎堆末尾
# ============================================================
def create_cluster_balanced_sampler(dataset: Step1Dataset,
                                    batch_size: int = 256,
                                    max_clusters_per_batch: int = 64):
    """
    贪心填充策略（含G老师三项修复）:

    1. [漏洞1修复] Round 2+ 动态采样
       - 每次调用从 dataset.valid_indices（全量）重新随机抽完美簇子集
       - 困难簇100%，完美簇按 dataset.easy_ratio 采样
       - 每个 epoch 调用一次，5 epoch = 5 套不同的随机子集

    2. [漏洞2修复] 大簇切割后余数流入缓冲区
       - 整块正常切，最后一块（零头）extend 进 current_batch 等待和其他簇混合
       - 消除单簇 batch，保证每个 batch 都有来自多个簇的负样本

    3. [漏洞3修复] singleton 均匀分散
       - 打散后 round-robin 插入已有 batch 的空余槽位
       - singleton 作为其他簇的负样本贡献排斥力梯度，不再造成 epoch 末尾梯度断层
    """
    print("   🔨 构建采样器 (贪心填充)...")

    all_labels = dataset.data_loader.clover_labels

    # ==================================================================
    # [漏洞1修复] Round 2+ 动态采样：每次重新随机抽完美簇子集
    # ==================================================================
    cluster_change_info = dataset.cluster_change_info
    if cluster_change_info is not None and dataset.round_idx >= 2:
        cv_threshold = dataset.cv_threshold
        easy_ratio   = dataset.easy_ratio
        rng = np.random.RandomState()   # 不固定种子 → 每 epoch 不同

        # [G老师-索引对齐-FIX] 从永久备份 full_valid_indices 读取全量池
        # 不能从 dataset.valid_indices 读——它已经被上一个 epoch 覆盖成子集了
        full_pool = getattr(dataset, 'full_valid_indices', dataset.valid_indices)

        # 全量池按簇分组
        cluster_to_full: Dict[int, List[int]] = defaultdict(list)
        for i in full_pool:
            label = all_labels[i]
            if label >= 0:
                cluster_to_full[label].append(i)

        # 按困难/完美采样，生成本 epoch 的实际训练集
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

        print(f"   🔀 本 epoch 动态采样: "
              f"困难簇 {n_hard} × 100% = {n_hard_reads} reads | "
              f"完美簇 {n_easy} × {int(easy_ratio*100)}% = {n_easy_reads} reads | "
              f"合计 {len(epoch_indices)}")

        # [G老师-索引对齐-FIX] 致命 Bug 修复：将 Dataset 当前索引同步为本轮子集
        # DataLoader 把 batch 里的 idx (0,1,2...) 传给 __getitem__，
        # __getitem__ 执行 self.valid_indices[idx]，
        # 所以 valid_indices 必须和 working_indices 完全一致，
        # 否则 idx=0 取到全量第0个而非本轮子集第0个 → 数据标签完全错位
        working_indices = epoch_indices
        dataset.valid_indices = working_indices   # ← 强制同步，消除张冠李戴
    else:
        # Round 1 / inference_mode / fallback：valid_indices 本身就是对的，无需同步
        working_indices = dataset.valid_indices

    # ==================================================================
    # 按簇分组（基于本 epoch 的 working_indices）
    # ==================================================================
    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, real_idx in enumerate(working_indices):
        label = all_labels[real_idx]
        cluster_to_indices[label].append(idx)

    # ==================================================================
    # 簇内采样上限：与 FedDNA 训练保持一致，每簇最多 max_reads_per_cluster 条
    # FedDNA 训练时每簇采样 5-30 条 reads，推理时用全量。
    # SSI-EC 加了对比学习需要更多正样本对，建议设 50。
    # 仅影响训练采样器，Step2 推理（inference_mode=True）不经过此路径。
    # 每 epoch 随机采样不同子集，多 epoch 后覆盖率趋近全量。
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
            print(f"   ✂️ 簇内采样上限: {max_rpc} reads/簇, "
                  f"截断 {n_capped} 个大簇")

    # 有效簇（≥2 条 reads）和 singleton 簇（1 条 read）
    valid_clusters = {cid: idxs for cid, idxs in cluster_to_indices.items()
                      if len(idxs) >= 2}
    singleton_indices = [idxs[0] for idxs in cluster_to_indices.values()
                         if len(idxs) == 1]

    cluster_ids = list(valid_clusters.keys())
    np.random.shuffle(cluster_ids)

    # ==================================================================
    # 贪心填充：逐簇塞入 current_batch
    # [漏洞2修复] 大簇零头 extend 进 current_batch，不单独成批
    # ==================================================================
    batches = []
    current_batch: List[int] = []
    current_n_clusters = 0

    for cid in cluster_ids:
        idxs = valid_clusters[cid]

        if len(idxs) > batch_size:
            # 先把 current_batch 封装
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_n_clusters = 0

            # 整块切割（每块恰好 batch_size）
            n_full = len(idxs) // batch_size
            for i in range(n_full):
                batches.append(idxs[i * batch_size:(i + 1) * batch_size])

            # [漏洞2修复] 零头 extend 进 current_batch，等后续小簇来混合
            remainder = idxs[n_full * batch_size:]
            if remainder:
                current_batch = list(remainder)
                current_n_clusters = 1   # 这个零头来自一个簇
            continue

        # 普通簇：放不下就先封装，再开新 batch
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
    # [漏洞3修复] singleton 均匀分散：round-robin 插入已有 batch 的空余槽位
    # 不再扎堆末尾，让 singleton 作为负样本均匀贡献排斥力梯度
    # ==================================================================
    if singleton_indices:
        np.random.shuffle(singleton_indices)
        overflow = []

        if len(batches) == 0:
            # 边界情况：整个 dataset 全是 singleton
            for i in range(0, len(singleton_indices), batch_size):
                batches.append(singleton_indices[i:i + batch_size])
        else:
            n_batches = len(batches)
            ptr = 0   # 轮询指针，在所有 batch 上 round-robin
            for s in singleton_indices:
                # 从 ptr 出发找到第一个有空位的 batch
                inserted = False
                for _ in range(n_batches):
                    b_idx = ptr % n_batches
                    ptr += 1
                    if len(batches[b_idx]) < batch_size:
                        batches[b_idx].append(s)
                        inserted = True
                        break
                if not inserted:
                    overflow.append(s)  # 所有 batch 都满了，溢出

            # 溢出的 singleton 打成小 batch（数量很少，影响可忽略）
            for i in range(0, len(overflow), batch_size):
                batches.append(overflow[i:i + batch_size])

        n_distributed = len(singleton_indices) - len(overflow) if singleton_indices else 0
        print(f"   🌶️  Singleton: {len(singleton_indices)} 条已均匀分散 "
              f"({n_distributed} 插入已有batch, {len(overflow) if singleton_indices else 0} 溢出成新batch)")

    # 打印统计
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