#!/usr/bin/env python
"""
verify_v20A_active.py —— 5 秒验证 Jaccard mask 是否真的在生效

直接调用 patched 后的 model + dataset, 跑 1 个 batch,
打印 jaccard_masked 数量. 不动训练, 不写文件.

预期结果:
  - jaccard_masked > 0  ✅ patch 生效, v20.A 实验有效
  - jaccard_masked == 0 ⚠️ patch 没生效 (或 batch 内无 cross-cluster anchor)
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

p = argparse.ArgumentParser()
p.add_argument('--code_dir', default='/mnt/st_data/liangxinyi/code')
p.add_argument('--exp_dir', default='/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d_v20A')
p.add_argument('--n_batches', type=int, default=3)
p.add_argument('--batch_size', type=int, default=256)
p.add_argument('--jaccard_theta', type=float, default=0.18)
args = p.parse_args()

if args.code_dir not in sys.path:
    sys.path.insert(0, args.code_dir)

from models.step1_data import CloverDataLoader, Step1Dataset, seq_to_kmer_bitvec
from models.step1_model import Step1EvidentialModel

# ── 加载 R3 状态 (有 -1 reads, 最大化 cross-cluster pair 信号) ──
import glob
labels_files = sorted(
    glob.glob(os.path.join(args.exp_dir, '04_Iterative_Labels',
                            'refined_labels_*.txt')),
    key=os.path.getmtime)
consensus_files = sorted(
    glob.glob(os.path.join(args.exp_dir, '04_Iterative_Labels',
                            'consensus_dict_*.pt')),
    key=os.path.getmtime)
ckpts = sorted(
    glob.glob(os.path.join(args.exp_dir, 'results',
                            'iter_*_step1', 'models',
                            'step1_final_model.pth')),
    key=os.path.getmtime)

if not labels_files or not ckpts:
    print("❌ 缺少 R3 状态文件"); sys.exit(1)

r3_labels    = labels_files[-1]
r3_consensus = consensus_files[-1] if consensus_files else None
r3_ckpt      = ckpts[-1]
print(f"📂 labels:    {os.path.basename(r3_labels)}")
print(f"📂 consensus: {os.path.basename(r3_consensus) if r3_consensus else 'none'}")
print(f"📂 ckpt:      {os.path.basename(r3_ckpt)}")

# ── DataLoader + Dataset ────────────────────────────────
dl = CloverDataLoader(args.exp_dir, labels_path=r3_labels)

consensus_dict = torch.load(r3_consensus, map_location='cpu') if r3_consensus else {}
ds = Step1Dataset(dl, max_len=201, consensus_dict=consensus_dict, kmer_k=5)
print(f"📦 dataset: {len(ds)} samples")

loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, drop_last=True)

# ── 检查 batch 是否含 kmer_vec ────────────────────────
batch = next(iter(loader))
print(f"\n🔍 batch keys: {list(batch.keys())}")
if 'kmer_vec' not in batch:
    print("❌ batch 缺 kmer_vec ← Patch 1 (data.py) 没生效")
    sys.exit(1)
kv = batch['kmer_vec']
print(f"✅ kmer_vec shape: {tuple(kv.shape)} (expected ({args.batch_size}, 1024))")
print(f"   非零率: {(kv > 0).float().mean().item():.3f}")

# ── 加载 R3 model ────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(r3_ckpt, map_location=device)
step1_args = ckpt.get('args', {})
model = Step1EvidentialModel(
    dim=step1_args.get('dim', 256),
    max_length=step1_args.get('max_length', 201),
    num_clusters=max(50, len(consensus_dict)),
    device=str(device),
).to(device)

# 兼容 length_adapter 维度
sd = ckpt['model_state_dict']
if 'length_adapter.weight' in sd:
    sh = sd['length_adapter.weight'].shape
    if sh[1] == step1_args.get('max_length', 201) and \
       sh[0] == step1_args.get('max_length', 201):
        import torch.nn as nn
        model.length_adapter = nn.Linear(sh[1], sh[0]).to(device)
model.load_state_dict(sd, strict=False)
model.eval()
print(f"✅ model loaded")

# ── 跑 N 个 batch, 看 jaccard_masked 计数 ──────────────
print(f"\n🔬 跑 {args.n_batches} 个 batch (jaccard_theta={args.jaccard_theta}):\n")
total_jaccard, total_soft = 0, 0
for i, batch in enumerate(loader):
    if i >= args.n_batches: break
    reads = batch['encoding'].to(device)
    labels = batch['clover_label'].to(device)
    consensus = batch['consensus_target'].to(device)
    kmer_vec = batch['kmer_vec'].to(device)

    with torch.no_grad():
        loss_dict, outputs = model(
            reads, labels, consensus,
            epoch=0, round_idx=3,
            kmer_vec=kmer_vec, jaccard_theta=args.jaccard_theta,
        )
    nj = outputs.get('jaccard_masked', -1)
    ns = outputs.get('soft_neg_masked', -1)
    print(f"   Batch {i+1}: jaccard_masked={nj:>5}  soft_neg_masked={ns:>5}  "
          f"con_loss={loss_dict['contrastive'].item():.4f}")
    total_jaccard += nj if nj > 0 else 0
    total_soft += ns if ns > 0 else 0

print()
print("="*60)
if total_jaccard > 0:
    print(f"✅ Jaccard mask **生效** (累计屏蔽 {total_jaccard} pair)")
    print(f"   v20.A 实际有 active. 说明退化 ≠ patch 失效, 而是 mask 强度不够 / 方向不对")
elif total_jaccard == 0 and 'jaccard_masked' in outputs:
    print(f"⚠️ jaccard_masked 字段存在但=0")
    print(f"   Patch 接通了, 但 batch 内没有 jaccard>theta 的 cross-cluster pair")
    print(f"   原因: theta=0.18 在 batch_size=256 内罕有命中 (要 256 reads 中两条同源 GT)")
    print(f"   建议: 降低 theta 或增大 batch_size 让 mask 命中")
else:
    print(f"❌ outputs 没有 'jaccard_masked' 字段 ← Patch 2 没生效")