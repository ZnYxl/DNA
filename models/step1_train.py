# models/step1_train.py
"""
Step1 训练主程 (Universal Edition)

修复清单:
  [FIX-P0]  训练前预计算 consensus_dict (majority vote / 从 Step2 加载)
  [FIX-P0]  训练循环: 从 batch_data 取 consensus_target，传给 model.forward
  [FIX-P0]  Round 2+: 读取 cluster_change_info，传给 Step1Dataset 实现簇级采样
  [FIX-#7]  epoch 数量: R1=10, R2+=5 (与实际测试一致)
  [FIX-#6]  Visualizer args 安全获取 (getattr)
  [NEW]     training_cap / round_idx 传递给 Dataset
"""
import torch
import torch.optim as optim
import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np
from collections import defaultdict
from typing import Dict, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.step1_model import Step1EvidentialModel, load_pretrained_feddna
from models.step1_data  import CloverDataLoader, Step1Dataset, create_dynamic_sampler, seq_to_onehot
from models.step1_visualizer import Step1Visualizer


# ---------------------------------------------------------------------------
# Hot Start 超参
# ---------------------------------------------------------------------------
ROUND1_EPOCHS = 10
ROUND1_LR     = 1e-4
ROUND2_EPOCHS = 5
ROUND2_LR     = 1e-5


class ListBatchSampler:
    def __init__(self, batches):
        self.batches = batches
    def __iter__(self):
        return iter(self.batches)
    def __len__(self):
        return len(self.batches)


def train_step1(args):
    """步骤一训练主函数"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")

    round_idx    = getattr(args, 'round_idx', 1)
    prev_state   = getattr(args, 'prev_state', None)
    training_cap = getattr(args, 'training_cap', 2000000)

    # =====================================================================
    # 1. 加载数据
    # =====================================================================
    print("\n" + "=" * 60)
    print("📂 数据加载")
    print("=" * 60)

    labels_path = getattr(args, 'refined_labels', None)
    data_loader = CloverDataLoader(args.experiment_dir, labels_path=labels_path)

    # =====================================================================
    # 2. [FIX-P0] 计算或加载 consensus_dict
    # =====================================================================
    print("\n" + "=" * 60)
    print("🧬 Consensus 计算")
    print("=" * 60)

    consensus_path = getattr(args, 'consensus_path', None)
    consensus_dict: Dict[int, torch.Tensor] = {}

    if consensus_path and os.path.exists(consensus_path):
        # Round 2+: 从 Step 2 输出加载（strength 加权 consensus）
        print(f"   📂 加载 consensus from: {consensus_path}")
        consensus_dict = torch.load(consensus_path, map_location='cpu')
        print(f"   ✅ 加载 {len(consensus_dict)} 个簇的 consensus")
    else:
        # Round 1: 直接从 ref.txt 读（预处理脚本已做 Clover majority vote）
        # data_loader.ref_seqs = {cluster_id: ref_seq_str}，在 _load_all_data 里加载
        print(f"   📖 Round {round_idx}: 从 ref.txt 构建 consensus_dict ...")
        if not data_loader.ref_seqs:
            raise RuntimeError(
                "ref.txt 未加载！请检查 CloverDataLoader._load_all_data 是否正确读取 ref.txt"
            )
        for cid, seq in data_loader.ref_seqs.items():
            consensus_dict[cid] = seq_to_onehot(seq, args.max_length)  # (L, 4)
        print(f"   ✅ consensus_dict: {len(consensus_dict)} 个簇")

    # =====================================================================
    # 3. [FIX-P0] 读取 cluster_change_info（Round 2+ 簇级采样依据）
    # =====================================================================
    cluster_change_info = getattr(args, 'cluster_change_info', None)
    if cluster_change_info is not None:
        hard_count = sum(1 for v in cluster_change_info.values() if v >= 0.05)
        easy_count = len(cluster_change_info) - hard_count
        print(f"   📊 cluster_change_info: 困难簇={hard_count}, 完美簇={easy_count}")

    # =====================================================================
    # 4. 创建 Dataset（携带 consensus_dict + cluster_change_info）
    # =====================================================================
    print("\n" + "=" * 60)
    print("📦 Dataset 创建")
    print("=" * 60)

    dataset = Step1Dataset(
        data_loader,
        max_len=args.max_length,
        training_cap=training_cap,
        inference_mode=False,
        round_idx=round_idx,
        consensus_dict=consensus_dict,              # [FIX-P0]
        cluster_change_info=cluster_change_info,    # [FIX-P0]
    )

    # =====================================================================
    # 5. 创建模型
    # =====================================================================
    print("\n" + "=" * 60)
    print("🧠 模型创建")
    print("=" * 60)

    num_clover_clusters = len(set(l for l in data_loader.clover_labels if l >= 0))
    num_gt_clusters = len(getattr(data_loader, 'gt_cluster_seqs', {}))
    num_clusters    = max(num_clover_clusters, num_gt_clusters, args.min_clusters)

    model = Step1EvidentialModel(
        dim=args.dim,
        max_length=args.max_length,
        num_clusters=num_clusters,
        device=str(device),
        cl_mode=getattr(args, 'cl_mode', 'ours'),   # 消融实验 flag
    ).to(device)

    print(f"   模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   当前簇数: {num_clover_clusters}")
    print(f"   对比学习模式: {getattr(args, 'cl_mode', 'ours')}")

    # =====================================================================
    # 6. Hot Start 权重加载
    # =====================================================================
    print("\n" + "=" * 60)
    print(f"🔋 Hot Start (Round {round_idx})")
    print("=" * 60)

    if round_idx <= 1:
        feddna_ckpt = getattr(args, 'feddna_checkpoint', None)
        if feddna_ckpt and os.path.exists(feddna_ckpt):
            model = load_pretrained_feddna(model, feddna_ckpt, device)
        else:
            print(f"   ⚠️ 未找到预训练权重，使用随机初始化")
    else:
        prev_ckpt = getattr(args, 'prev_checkpoint', None)
        if prev_ckpt and os.path.exists(prev_ckpt):
            try:
                import torch.nn as nn_
                ckpt = torch.load(prev_ckpt, map_location=device)
                sd   = ckpt.get('model_state_dict', ckpt)

                if 'length_adapter.weight' in sd:
                    sh = sd['length_adapter.weight'].shape
                    if sh[0] == args.max_length:
                        model.length_adapter = nn_.Linear(sh[1], sh[0]).to(device)
                        print(f"   🔧 预初始化 length_adapter: {sh}")

                model.load_state_dict(sd, strict=False)
                print(f"   ✅ 加载上一轮权重成功")
            except Exception as e:
                print(f"   ⚠️ 加载失败: {e}，使用随机初始化")
        else:
            print(f"   ⚠️ 无上一轮 checkpoint，随机初始化")

    # =====================================================================
    # 7. 优化器 + 调度器
    # =====================================================================
    epochs = ROUND1_EPOCHS if round_idx <= 1 else ROUND2_EPOCHS
    lr     = ROUND1_LR     if round_idx <= 1 else ROUND2_LR

    print(f"\n   📐 训练超参: epochs={epochs}, lr={lr}")

    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # =====================================================================
    # 8. 训练循环
    # =====================================================================
    print("\n" + "=" * 60)
    print("🚀 开始训练")
    print("=" * 60)

    model.train()

    training_history = {
        'total_loss': [], 'avg_strength': [], 'high_conf_ratio': [],
        'contrastive_loss': [], 'reconstruction_loss': [], 'kl_loss': [],
        'u_epi_mean': [], 'u_ale_mean': [], 'queue_count': []
    }

    for epoch in range(epochs):
        start_time = time.time()

        batch_indices_list = create_dynamic_sampler(
            dataset,
            batch_size=args.batch_size,
            max_clusters_per_batch=args.max_clusters_per_batch,
            state_path=prev_state,
            round_idx=round_idx
        )

        batch_sampler = ListBatchSampler(batch_indices_list)
        train_loader  = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
            pin_memory=True
        )

        epoch_loss  = epoch_con = epoch_rec = epoch_kl = 0
        epoch_str   = epoch_hc = epoch_u_epi = epoch_u_ale = epoch_qc = 0
        epoch_w_cc  = epoch_w_da = epoch_cos_pos = epoch_cos_neg = 0
        epoch_probe_cnt = 0
        num_batches = 0
        total_batches = len(batch_indices_list)

        print(f"\n🔄 Epoch {epoch + 1}/{epochs} (共 {total_batches} Batches)")

        for i, batch_data in enumerate(train_loader):
            reads_batch      = batch_data['encoding'].to(device)
            labels_batch     = batch_data['clover_label'].to(device)
            # [FIX-P0] 从 dataset 取 consensus_target 并送到 GPU
            consensus_batch  = batch_data['consensus_target'].to(device)

            # [FIX-P0] 传入 consensus_target
            loss_dict, outputs = model(reads_batch, labels_batch, consensus_batch, epoch=epoch)

            optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss  += loss_dict['total'].item()
            epoch_con   += loss_dict['contrastive'].item()
            epoch_rec   += loss_dict['reconstruction'].item()
            epoch_kl    += loss_dict['kl_divergence'].item()
            epoch_str   += outputs['avg_strength']
            epoch_hc    += outputs['high_conf_ratio']
            epoch_u_epi += outputs.get('u_epi_mean', 0.0)
            epoch_u_ale += outputs.get('u_ale_mean', 0.0)
            epoch_qc    += outputs.get('queue_count', 0)

            # 累积探针（仅非 nan）
            wcc = outputs.get('w_clean_clean', float('nan'))
            wda = outputs.get('w_dirty_any',   float('nan'))
            cp  = outputs.get('cos_sim_pos',   float('nan'))
            cn  = outputs.get('cos_sim_neg',   float('nan'))
            if not (wcc != wcc):  # nan check
                epoch_w_cc += wcc; epoch_w_da += wda
                epoch_cos_pos += cp; epoch_cos_neg += cn
                epoch_probe_cnt += 1

            num_batches += 1

            if (i + 1) % 50 == 0:
                print(f"   [Batch {i+1}/{total_batches}] "
                      f"Loss: {loss_dict['total'].item():.4f} | "
                      f"Str: {outputs['avg_strength']:.1f} | "
                      f"U_epi: {outputs.get('u_epi_mean',0):.4f}",
                      end='\r')

        scheduler.step()
        epoch_time = time.time() - start_time
        avg = lambda x: x / max(num_batches, 1)

        training_history['total_loss'].append(avg(epoch_loss))
        training_history['contrastive_loss'].append(avg(epoch_con))
        training_history['reconstruction_loss'].append(avg(epoch_rec))
        training_history['kl_loss'].append(avg(epoch_kl))
        training_history['avg_strength'].append(avg(epoch_str))
        training_history['high_conf_ratio'].append(avg(epoch_hc))
        training_history['u_epi_mean'].append(avg(epoch_u_epi))
        training_history['u_ale_mean'].append(avg(epoch_u_ale))
        training_history['queue_count'].append(avg(epoch_qc))

        print(f"\n   ✅ Epoch {epoch+1} ({epoch_time:.1f}s) | "
              f"Loss: {avg(epoch_loss):.4f} | Str: {avg(epoch_str):.1f} | "
              f"Recon: {avg(epoch_rec):.4f} | U_epi: {avg(epoch_u_epi):.4f}")

        # 三个诊断探针汇报
        if epoch_probe_cnt > 0:
            pc = epoch_probe_cnt
            print(f"   🔬 探针 A | w(干净-干净): {epoch_w_cc/pc:.4f}  "
                  f"w(含脏-任意): {epoch_w_da/pc:.4f}  "
                  f"比值: {(epoch_w_cc/pc) / max(epoch_w_da/pc, 1e-6):.1f}x")
            print(f"   🔬 探针 B | cos_pos: {epoch_cos_pos/pc:.4f}  "
                  f"cos_neg: {epoch_cos_neg/pc:.4f}  "
                  f"margin: {(epoch_cos_pos-epoch_cos_neg)/pc:.4f}")

    # =====================================================================
    # 9. 保存 checkpoint
    # =====================================================================
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    checkpoint_path = os.path.join(models_dir, "step1_final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'training_history': training_history,
        'round_idx': round_idx,
    }, checkpoint_path)
    print(f"\n💾 Checkpoint 保存: {checkpoint_path}")

    # 可视化
    try:
        viz = Step1Visualizer(output_dir)
        viz.plot_training_losses(training_history)
        viz.plot_evidence_stats(training_history)
        viz.save_config(args)
    except Exception as e:
        print(f"   ⚠️ 可视化跳过: {e}")

    return checkpoint_path