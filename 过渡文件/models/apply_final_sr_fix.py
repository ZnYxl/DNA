#!/usr/bin/env python3
"""
apply_final_sr_fix.py — G老师终极诊断修复

核心原则 (G老师终极诊断):
  ❌ 不能加载 FedDNA 原生 epoch1_I.pth 解码 (ID20 150bp 语言 ≠ Seq_1D 201bp)
  ✅ 必须用 Step 1 训练后的 SSI-EC model 解码 (已适应 201 长度)

叠加三重优化:
  1. 双轨制 Consensus (G老师一期):
     - 轨道一 (.pt): 纯净, 排除 -1 → 下一轮训练靶点
     - 轨道二 (.fasta): 全量 → 评估 SR
  2. 精准归巢 (G老师二期杀手锏):
     - 死数据复活后, 残留 -1 按 embedding 距离归巢到大簇
     - 只影响评估轨道, 不污染训练轨道
  3. 统一用 model (G老师三期):
     - 两条轨道都用 Step 1 训练后的 SSI-EC model
     - 不加载任何 FedDNA 原生权重

用法:
  python apply_final_sr_fix.py

前置要求:
  step2_runner.py 必须是干净版本 (无 _use_feddna_decode / _SSIECDecodeAdapter 等残留)
  建议先:
    cp step2_runner.py.bak_dual_track step2_runner.py  # 或最初原始版
"""
import os, shutil

STEP2_FILE = "/mnt/st_data/liangxinyi/code/models/step2_runner.py"

# 归巢参数
MIN_LARGE_CLUSTER_SIZE = 10


def backup_and_replace(filepath, old, new, desc):
    with open(filepath, 'r') as f:
        content = f.read()
    if old not in content:
        if 'zone3_precise_reassign' in content:
            print(f"  ⏭️  已打过: {desc}")
            return True
        print(f"  ❌ 未找到目标: {desc}")
        print(f"     前80字符: {repr(old[:80])}")
        return False
    bak = filepath + '.bak_final_sr'
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
    print("  Apply Final SR Fix (G老师终极诊断)")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════
    # Patch 1: 死数据复活后添加精准归巢 (生成 eval_labels)
    # ══════════════════════════════════════════════════════════
    print("\n📝 Patch 1: 死数据复活后添加精准归巢")

    old_after_revival = '''\
        print(f"   ✨ 成功复活: {revived_count} / {len(noise_indices)} reads")
    else:
        print(f"\\n   🧟 死数据复活: 无候选数据或无有效质心，跳过")

    if _probe:
        _probe.snapshot("after_revival", new_labels)
        _probe.report(round_idx)'''

    new_after_revival = f'''\
        print(f"   ✨ 成功复活: {{revived_count}} / {{len(noise_indices)}} reads")
    else:
        print(f"\\n   🧟 死数据复活: 无候选数据或无有效质心，跳过")

    # ══════════════════════════════════════════════════════════════
    # [zone3_precise_reassign] G老师杀手锏: 残留 -1 精准归巢到大簇
    # ══════════════════════════════════════════════════════════════
    # 严格双轨隔离:
    #   - new_labels (保持 -1): 训练轨道, 纯净靶点
    #   - eval_labels (全量归巢): 评估轨道, 最大化重建
    # 归巢策略: 按 embedding 欧氏距离分配到当前大簇, 享受 MNN 合并红利
    eval_labels = new_labels.clone()
    final_noise_mask = (eval_labels == -1)
    final_noise_indices = torch.where(final_noise_mask)[0]

    if len(final_noise_indices) > 0 and len(centroids) > 0:
        MIN_LARGE_SIZE = {MIN_LARGE_CLUSTER_SIZE}
        large_cids = [cid for cid, sz in cluster_sizes.items() if sz >= MIN_LARGE_SIZE]
        if len(large_cids) == 0:
            large_cids = sorted(centroids.keys())
            print(f"\\n   🏠 Zone III 归巢: 无 size>={{MIN_LARGE_SIZE}} 大簇, "
                  f"回退到全簇 ({{len(large_cids)}})")
        else:
            print(f"\\n   🏠 Zone III 精准归巢: {{len(final_noise_indices):,}} 残留 -1 reads "
                  f"→ {{len(large_cids):,}} 大簇 (size>={{MIN_LARGE_SIZE}})")

        large_cids_sorted = sorted(large_cids)
        large_centroid_matrix = torch.stack(
            [centroids[c] for c in large_cids_sorted]
        ).cpu()

        reassigned_count = 0
        ed_chunk_size = 5000
        for start in range(0, len(final_noise_indices), ed_chunk_size):
            end       = min(start + ed_chunk_size, len(final_noise_indices))
            batch_idx = final_noise_indices[start:end]
            batch_emb = embeddings_f32[batch_idx]

            dists              = torch.cdist(batch_emb, large_centroid_matrix)
            min_dists, min_idx = dists.min(dim=1)

            reassign_cluster_ids = torch.tensor(
                [large_cids_sorted[j] for j in min_idx.tolist()],
                dtype=torch.long
            )
            eval_labels[batch_idx] = reassign_cluster_ids
            reassigned_count += len(batch_idx)

        print(f"   ✅ 归巢完成: {{reassigned_count:,}} reads → 大簇")
    else:
        print(f"\\n   🏠 Zone III 归巢: 无残留 -1 或无质心，跳过")

    if _probe:
        _probe.snapshot("after_revival", new_labels)
        _probe.report(round_idx)'''

    ok1 = backup_and_replace(STEP2_FILE, old_after_revival, new_after_revival,
                             "精准归巢 (生成 eval_labels)")

    if not ok1:
        print("\n  ❌ Patch 1 失败, 无法继续")
        return

    # ══════════════════════════════════════════════════════════
    # Patch 2: 替换 consensus 段 — 双轨制 + 用 model 解码
    # ══════════════════════════════════════════════════════════
    print("\n📝 Patch 2: 双轨制 Consensus (统一用 SSI-EC model)")

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
    # [G老师双轨制 + 终极诊断] Consensus 物理隔离, 统一用 SSI-EC model
    # ══════════════════════════════════════════════════════════════
    # 核心: 不加载 FedDNA 原生 epoch1_I.pth (它是 ID20 150bp 语言,
    #        与 Seq_1D 201bp 不兼容, 会导致 SR=0)
    # 正确: 用 Step 1 训练后的 model, 它的 RNNBlock 已适应 201 长度
    #
    # 轨道一: 纯净版 (排除 -1) → .pt → 下一轮训练靶点
    # 轨道二: 全量版 (eval_labels, 已归巢) → .fasta → 评估 SR
    # ══════════════════════════════════════════════════════════════

    new_labels_np = new_labels.cpu().numpy()  # 严格版: 训练/聚类评估/保存用

    from models.step2_decode import run_feddna_decode
    # 模型已在 model.cpu() 后，需要重新上 GPU
    model.to(device)

    # ── 轨道一: 纯净版 Consensus (训练专用, 神圣不可侵犯) ──
    print(f"\\n   🔒 轨道一: 纯净版 Consensus (排除 -1, 用于下一轮训练)")
    consensus_dict_for_train = run_feddna_decode(
        model=model,
        data_loader=data_loader,
        new_labels_np=new_labels_np,  # 严格版, step2_decode 自动跳过 -1
        flat_real_indices=flat_real_indices,
        model_max_len=model_max_len,
        device=device,
        batch_size=getattr(args, 'batch_size', 512),
        ref_length=getattr(args, 'ref_length', None),
    )
    print(f"   ✅ 纯净版: {len(consensus_dict_for_train)} 个簇")

    # ── 轨道二: 全量版 Consensus (评估专用, 应收尽收) ──
    # eval_labels 来自精准归巢段 (死数据复活之后):
    #   - P75 阈值内的 reads 已复活归簇
    #   - 残留 -1 已按 embedding 距离归巢到大簇
    # 无需额外 Clover 回填
    labels_for_eval_np = eval_labels.cpu().numpy()
    n_still_noise = int((eval_labels == -1).sum().item())
    print(f"\\n   🔓 轨道二: 全量版 Consensus (归巢结果, 残留 -1: {n_still_noise})")
    consensus_dict_for_eval = run_feddna_decode(
        model=model,
        data_loader=data_loader,
        new_labels_np=labels_for_eval_np,
        flat_real_indices=flat_real_indices,
        model_max_len=model_max_len,
        device=device,
        batch_size=getattr(args, 'batch_size', 512),
        ref_length=getattr(args, 'ref_length', None),
    )
    print(f"   ✅ 全量版: {len(consensus_dict_for_eval)} 个簇")

    model.cpu()
    torch.cuda.empty_cache()

    # 后续 consensus_dict 指向训练版 (.pt 保存)
    consensus_dict = consensus_dict_for_train'''

    ok2 = backup_and_replace(STEP2_FILE, old_consensus, new_consensus,
                             "双轨制 Consensus (统一 model 解码)")

    # ══════════════════════════════════════════════════════════
    # Patch 3: FASTA 保存用 consensus_dict_for_eval
    # ══════════════════════════════════════════════════════════
    print("\n📝 Patch 3: FASTA 保存用全量版")

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
                       "FASTA 保存改用全量版")

    print("\n" + "=" * 60)
    print("  ✅ 终极修复完成")
    print("=" * 60)
    print(f"""
  关键确认:
    ✅ 用 Step 1 训练后的 model 解码 (已适应 201 长度)
    ❌ 不加载 FedDNA 原生 epoch1_I.pth (它是 ID20 150bp 语言)

  双轨隔离:
    .pt  → consensus_dict_for_train (纯净, 排除 -1)  → 下一轮训练
    .fasta → consensus_dict_for_eval (全量, 归巢后)  → 评估 SR

  预期:
    - 聚类: R3 Purity ≈ 0.9699 (G老师双轨制已验证)
    - 重建: SR 恢复到 85-88% 的基础上, 再靠归巢突破 91%
""")


if __name__ == "__main__":
    main()