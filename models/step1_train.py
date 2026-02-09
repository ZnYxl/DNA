# models/step1_train.py
"""
Step1 训练主程 (Universal Edition)

修复清单:
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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.step1_model import Step1EvidentialModel, load_pretrained_feddna
from models.step1_data  import CloverDataLoader, Step1Dataset, create_dynamic_sampler
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

    round_idx  = getattr(args, 'round_idx', 1)
    prev_state = getattr(args, 'prev_state', None)
    training_cap = getattr(args, 'training_cap', 2000000)

    # =====================================================================
    # 1. 加载数据
    # =====================================================================
    print("\n" + "=" * 60)
    print("📂 数据加载")
    print("=" * 60)

    labels_path = getattr(args, 'refined_labels', None)
    data_loader = CloverDataLoader(args.experiment_dir, labels_path=labels_path)

    # [FIX] 传递 training_cap 和 round_idx
    dataset = Step1Dataset(
        data_loader,
        max_len=args.max_length,
        training_cap=training_cap,
        inference_mode=False,
        round_idx=round_idx
    )

    # =====================================================================
    # 2. 创建模型
    # =====================================================================
    print("\n" + "=" * 60)
    print("🧠 模型创建")
    print("=" * 60)

    num_clover_clusters = len(set(data_loader.clover_labels))
    num_gt_clusters = len(getattr(data_loader, 'gt_cluster_seqs', {}))
    num_clusters    = max(num_clover_clusters, num_gt_clusters, args.min_clusters)

    model = Step1EvidentialModel(
        dim=args.dim,
        max_length=args.max_length,
        num_clusters=num_clusters,
        device=device
    ).to(device)

    print(f"   模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   当前簇数: {num_clover_clusters}")

    # =====================================================================
    # 3. Hot Start 权重加载
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
                ckpt = torch.load(prev_ckpt, map_location=device)
                sd   = ckpt.get('model_state_dict', ckpt)

                if 'length_adapter.weight' in sd:
                    import torch.nn as nn_
                    sh = sd['length_adapter.weight'].shape
                    if sh[0] == args.max_length:
                        model.length_adapter = nn_.Linear(sh[1], sh[0]).to(device)
                        print(f"   🔧 预初始化 length_adapter: {sh}")
                    else:
                        print(f"   ⚠️ 长度不匹配 ({sh[0]} vs {args.max_length})，重置")

                model.load_state_dict(sd, strict=False)
                print(f"   ✅ 加载上一轮权重成功")
            except Exception as e:
                print(f"   ⚠️ 加载失败: {e}，使用随机初始化")
        else:
            print(f"   ⚠️ 无上一轮 checkpoint，随机初始化")

    # =====================================================================
    # 4. 优化器 + 调度器
    # =====================================================================
    epochs = ROUND1_EPOCHS if round_idx <= 1 else ROUND2_EPOCHS
    lr     = ROUND1_LR     if round_idx <= 1 else ROUND2_LR

    print(f"\n   📐 训练超参: epochs={epochs}, lr={lr}")

    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # =====================================================================
    # 5. 训练循环
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

        epoch_loss = epoch_con = epoch_rec = epoch_kl = 0
        epoch_str = epoch_hc = epoch_u_epi = epoch_u_ale = epoch_qc = 0
        num_batches = 0
        total_batches = len(batch_indices_list)

        print(f"\n🔄 Epoch {epoch + 1}/{epochs} (共 {total_batches} Batches)")

        for i, batch_data in enumerate(train_loader):
            reads_batch  = batch_data['encoding'].to(device)
            labels_batch = batch_data['clover_label'].to(device)

            loss_dict, outputs = model(reads_batch, labels_batch, epoch=epoch)

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
              f"U_epi: {avg(epoch_u_epi):.4f}")

    # =====================================================================
    # 6. 保存
    # =====================================================================
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    final_path = os.path.join(args.output_dir, "models", "step1_final_model.pth")

    # [FIX-#6] 把 epochs 和 lr 写入 args 供 visualizer 使用
    args.epochs = epochs
    args.lr = lr

    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args)
    }, final_path)

    print(f"\n💾 模型已保存: {final_path}")

    # 可视化
    try:
        visualizer = Step1Visualizer(args.output_dir)
        visualizer.generate_all_outputs(training_history, model, args)
    except Exception as e:
        print(f"⚠️ 可视化生成跳过: {e}")

    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 1 Training')
    parser.add_argument('--experiment_dir',         type=str,   required=True)
    parser.add_argument('--feddna_checkpoint',      type=str,   default=None)
    parser.add_argument('--prev_checkpoint',        type=str,   default=None)
    parser.add_argument('--refined_labels',         type=str,   default=None)
    parser.add_argument('--prev_state',             type=str,   default=None)
    parser.add_argument('--round_idx',              type=int,   default=1)
    parser.add_argument('--dim',                    type=int,   default=256)
    parser.add_argument('--max_length',             type=int,   default=150)
    parser.add_argument('--min_clusters',           type=int,   default=50)
    parser.add_argument('--device',                 type=str,   default='cuda')
    parser.add_argument('--batch_size',             type=int,   default=64)
    parser.add_argument('--max_clusters_per_batch', type=int,   default=8)
    parser.add_argument('--weight_decay',           type=float, default=1e-5)
    parser.add_argument('--training_cap',           type=int,   default=2000000)
    parser.add_argument('--output_dir',             type=str,   default='./step1_results')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_step1(args)