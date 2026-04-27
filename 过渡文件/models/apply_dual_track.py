#!/usr/bin/env python3
"""
apply_dual_track.py — G老师双轨制：纯净训练 + 全量评估

改动:
  step2_runner.py:
    - 轨道一: 纯净 consensus（排除 -1）→ .pt 文件 → 下一轮训练
    - 轨道二: 全量 consensus（回填 Clover）→ .fasta 文件 → 评估 SR

  main_loop.py:
    - 传递 feddna_checkpoint 给 step2（如果尚未添加）

  step2_runner.py import:
    - 添加 load_pretrained_feddna

用法:
  python apply_dual_track.py
"""
import os, shutil

CODE_DIR   = "/mnt/st_data/liangxinyi/code/models"
MAIN_LOOP  = os.path.join(CODE_DIR, "main_loop.py")
STEP2_FILE = os.path.join(CODE_DIR, "step2_runner.py")


def backup_and_replace(filepath, old, new, desc):
    with open(filepath, 'r') as f:
        content = f.read()
    if old not in content:
        if new.strip()[:50] in content:
            print(f"  ⏭️  已打过: {desc}")
            return True
        print(f"  ❌ 未找到目标: {desc}")
        print(f"     前60字符: {repr(old[:60])}")
        return False
    bak = filepath + '.bak_dual_track'
    if not os.path.exists(bak):
        shutil.copy2(filepath, bak)
        print(f"  💾 备份: {bak}")
    content = content.replace(old, new, 1)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  ✅ {desc}")
    return True


def main():
    print("=" * 60)
    print("  Apply 双轨制 Consensus Patch")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════
    # Patch 1: main_loop.py — feddna_checkpoint 传递
    # ══════════════════════════════════════════════════════════
    print("\n📝 Patch 1: main_loop.py")
    with open(MAIN_LOOP, 'r') as f:
        ml = f.read()
    if 'feddna_checkpoint=args.feddna_checkpoint,' in ml:
        print("  ⏭️  feddna_checkpoint 已传递")
    else:
        backup_and_replace(MAIN_LOOP,
            "            disable_merge=getattr(args, 'disable_merge', False),\n"
            "            ref_length=getattr(args, 'ref_length', None),\n"
            "        )",
            "            disable_merge=getattr(args, 'disable_merge', False),\n"
            "            ref_length=getattr(args, 'ref_length', None),\n"
            "            feddna_checkpoint=args.feddna_checkpoint,\n"
            "        )",
            "传递 feddna_checkpoint")

    # ══════════════════════════════════════════════════════════
    # Patch 2: step2_runner.py import
    # ══════════════════════════════════════════════════════════
    print("\n📝 Patch 2: step2_runner.py import")
    with open(STEP2_FILE, 'r') as f:
        s2 = f.read()
    if 'load_pretrained_feddna' not in s2:
        backup_and_replace(STEP2_FILE,
            "from models.step1_model import Step1EvidentialModel",
            "from models.step1_model import Step1EvidentialModel, load_pretrained_feddna",
            "添加 load_pretrained_feddna import")
    else:
        print("  ⏭️  已有 import")

    # ══════════════════════════════════════════════════════════
    # Patch 3: 双轨制 Consensus 核心替换
    # ══════════════════════════════════════════════════════════
    print("\n📝 Patch 3: 双轨制 Consensus")

    old_consensus = '''\
    # ── [v2-策略三] Zone III 软隔离: consensus 前临时恢复标签 ──
    # Zone III reads 的 label=-1 导致它们完全不参与 consensus。
    # 但 FedDNA ds_fusion 内部本身就按 evidence 加权，噪声 reads 天然低权。
    # 策略: 临时恢复 Zone III 的原始簇标签让它们参与 consensus 投票。
    # new_labels 本身不变（Zone III 仍为 -1），训练时不受影响。
    labels_for_consensus = new_labels.clone()
    if z3_indices is not None and len(z3_indices) > 0:
        valid_z3 = (z3_original_labels >= 0)
        labels_for_consensus[z3_indices[valid_z3]] = z3_original_labels[valid_z3]
        print(f"   🔄 Zone III 软参与 consensus: {int(valid_z3.sum())}/{len(z3_indices)} reads 临时恢复标签")

    new_labels_np_for_consensus = new_labels.cpu().numpy()
    new_labels_np = new_labels.cpu().numpy()  # 严格版: 训练/评估/保存用

    # [FIX-DECODE] 用 FedDNA ds_fusion 替换 majority-vote consensus
    from models.step2_decode import run_feddna_decode
    # 模型已在 model.cpu() 后，需要重新上 GPU
    model.to(device)
    consensus_dict = run_feddna_decode(
        model=model,
        data_loader=data_loader,
        new_labels_np=new_labels_np_for_consensus,  # [v2] 含 Zone III 软参与
        flat_real_indices=flat_real_indices,
        model_max_len=model_max_len,
        device=device,
        batch_size=getattr(args, 'batch_size', 512),
        ref_length=getattr(args, 'ref_length', None),   # [v5] 先验长度
    )
    model.cpu()
    torch.cuda.empty_cache()
    print(f"   ✅ 生成 {len(consensus_dict)} 个簇的 consensus")'''

    new_consensus = '''\
    # ══════════════════════════════════════════════════════════════
    # [G老师-双轨制] 训练与评估的 Consensus 物理隔离
    # ══════════════════════════════════════════════════════════════
    # 轨道一: 纯净版（排除 -1）→ .pt → 下一轮训练靶点
    # 轨道二: 全量版（回填 Clover）→ .fasta → 评估 SR
    # 两条轨道物理隔离，阻断脏数据跨轮次污染。
    # ══════════════════════════════════════════════════════════════

    new_labels_np = new_labels.cpu().numpy()  # 严格版: 训练/聚类评估/保存用（保持 -1）

    from models.step2_decode import run_feddna_decode

    # ── 构建 FedDNA 解码模型 ──
    _feddna_ckpt_path = getattr(args, 'feddna_checkpoint', None)
    _use_feddna_decode = False
    if _feddna_ckpt_path and os.path.exists(_feddna_ckpt_path):
        try:
            print(f"   🔧 构建 FedDNA 解码模型...")
            feddna_decode_model = Step1EvidentialModel(
                dim=model_dim, max_length=model_max_len,
                num_clusters=num_clusters, device=str(device)
            ).to(device)
            feddna_decode_model = load_pretrained_feddna(
                feddna_decode_model, _feddna_ckpt_path, device,
                max_length=model_max_len
            )
            feddna_decode_model.eval()
            _use_feddna_decode = True
            print(f"   ✅ FedDNA 解码模型就绪")
        except Exception as e:
            print(f"   ⚠️ FedDNA 解码模型失败: {e}，回退到 SSI-EC 模型")
    else:
        print(f"   ⚠️ 未提供 feddna_checkpoint，使用 SSI-EC 模型解码")

    _decode_model = feddna_decode_model if _use_feddna_decode else model
    _decode_model.to(device)

    # ── 轨道一: 纯净版 Consensus（训练专用，神圣不可侵犯） ──
    print(f"\\n   🔒 轨道一: 纯净版 Consensus（训练专用，排除 -1）")
    consensus_dict_for_train = run_feddna_decode(
        model=_decode_model,
        data_loader=data_loader,
        new_labels_np=new_labels_np,  # 严格版，-1 被 step2_decode 自动跳过
        flat_real_indices=flat_real_indices,
        model_max_len=model_max_len,
        device=device,
        batch_size=getattr(args, 'batch_size', 512),
        ref_length=getattr(args, 'ref_length', None),
    )
    print(f"   ✅ 纯净版: {len(consensus_dict_for_train)} 个簇")

    # ── 轨道二: 全量版 Consensus（评估专用，应收尽收） ──
    # Zone III 恢复原始标签 + 剩余 -1 回填 Clover，100% reads 参与
    labels_for_eval = new_labels.clone()

    # Zone III 软参与: 恢复原始簇标签
    if z3_indices is not None and len(z3_indices) > 0:
        valid_z3 = (z3_original_labels >= 0)
        labels_for_eval[z3_indices[valid_z3]] = z3_original_labels[valid_z3]
        print(f"   🔄 Zone III 恢复: {int(valid_z3.sum())}/{len(z3_indices)} reads")

    # 剩余 -1 回填 Clover 初始标签
    clover_labels_tensor = torch.tensor(
        [data_loader.clover_labels[flat_real_indices[i]] for i in range(len(labels_for_eval))],
        dtype=torch.long
    )
    still_noise = (labels_for_eval == -1)
    labels_for_eval[still_noise] = clover_labels_tensor[still_noise]
    n_restored = int(still_noise.sum().item())

    print(f"\\n   🔓 轨道二: 全量版 Consensus（评估专用，回填 {n_restored} 条 -1）")
    labels_for_eval_np = labels_for_eval.cpu().numpy()
    consensus_dict_for_eval = run_feddna_decode(
        model=_decode_model,
        data_loader=data_loader,
        new_labels_np=labels_for_eval_np,
        flat_real_indices=flat_real_indices,
        model_max_len=model_max_len,
        device=device,
        batch_size=getattr(args, 'batch_size', 512),
        ref_length=getattr(args, 'ref_length', None),
    )
    print(f"   ✅ 全量版: {len(consensus_dict_for_eval)} 个簇")

    # 释放解码模型
    if _use_feddna_decode:
        del feddna_decode_model
    _decode_model.cpu()
    model.cpu()
    torch.cuda.empty_cache()

    # 后续代码统一用 consensus_dict 指向训练版（保护训练目标）
    consensus_dict = consensus_dict_for_train'''

    backup_and_replace(STEP2_FILE, old_consensus, new_consensus,
                       "双轨制 Consensus 核心逻辑")

    # ══════════════════════════════════════════════════════════
    # Patch 4: FASTA 保存用全量版
    # ══════════════════════════════════════════════════════════
    print("\n📝 Patch 4: FASTA 保存用全量版")

    old_fasta = '''\
    try:
        save_consensus_fasta(
            consensus_dict, new_labels_np, flat_real_indices,
            data_loader, model_max_len, fasta_path
        )'''

    new_fasta = '''\
    try:
        save_consensus_fasta(
            consensus_dict_for_eval, labels_for_eval_np, flat_real_indices,
            data_loader, model_max_len, fasta_path
        )'''

    backup_and_replace(STEP2_FILE, old_fasta, new_fasta,
                       "FASTA 保存改用 consensus_dict_for_eval")

    print("\n" + "=" * 60)
    print("  ✅ 双轨制补丁完成")
    print("=" * 60)
    print("""
  物理隔离:
    .pt  文件 → consensus_dict_for_train (纯净, 排除-1)  → 下一轮训练
    .fasta 文件 → consensus_dict_for_eval  (全量, 回填Clover) → SR 评估

  确认清单:
    ✓ step1_train.py: rnnblock 不冻结
    ✓ step2_runner.py: Zone III 全轮次隔离 (无 if round_idx==1 条件)
    ✓ 双轨制: .pt 纯净 / .fasta 全量
""")


if __name__ == "__main__":
    main()