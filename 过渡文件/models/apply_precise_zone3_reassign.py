#!/usr/bin/env python3
"""
apply_precise_zone3_reassign.py — Zone III 精准归巢（游子归巢策略）

核心思路（G老师的杀手锏建议）:
  不用 Clover 初始标签回填（会继承 Clover 的过分割错误），
  而是用当前最新合并优化后的"大簇"作为归巢目标。
  Zone III reads 按 embedding 欧氏距离分配到最近的大簇。

  小簇（< MIN_CLUSTER_SIZE）被排除在候选外，因为:
    1. 小簇质心不稳定，距离判定不可靠
    2. 小簇很多是 Clover 过分割产物，归巢到它们等于继承错误
    3. 大簇是 MNN 合并后的"引力井"，归巢到它们享受大簇红利

隔离原则（延续 G老师双轨制）:
  - 训练轨道 (.pt): new_labels 保持 -1, 不受精准分配影响
  - 评估轨道 (.fasta): labels_for_eval 使用精准分配的结果

改动位置:
  step2_runner.py:
    1. 在死数据复活之后、embeddings_f32 释放之前，
       对残留 -1 做"大簇强制分配"，存成 labels_for_eval_zone3
    2. 双轨制段用 labels_for_eval_zone3 替代 Clover 回填

用法:
  python apply_precise_zone3_reassign.py
"""
import os, shutil

STEP2_FILE = "/mnt/st_data/liangxinyi/code/models/step2_runner.py"

# 大簇阈值: 小于此值的簇不作为归巢目标
MIN_LARGE_CLUSTER_SIZE = 10


def backup_and_replace(filepath, old, new, desc):
    with open(filepath, 'r') as f:
        content = f.read()
    if old not in content:
        if 'zone3_precise_reassign' in content:
            print(f"  ⏭️  已打过: {desc}")
            return True
        print(f"  ❌ 未找到目标: {desc}")
        print(f"     片段前60字符: {repr(old[:60])}")
        return False
    bak = filepath + '.bak_precise_zone3'
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
    print("  Apply 精准 Zone III 归巢")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════
    # Patch 1: 在死数据复活之后、embeddings 释放之前，
    #          对残留 -1 做大簇强制分配
    # ══════════════════════════════════════════════════════════
    print("\n📝 Patch 1: 死数据复活后添加精准分配")

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
    # 死数据复活用了 P{{DELTA_P}} 阈值，严格筛选。
    # 对阈值外仍残留的 -1 reads，用更宽松的"最近大簇"策略兜底。
    # 关键: 只分配到大簇 (size >= {MIN_LARGE_CLUSTER_SIZE}), 避免继承 Clover 过分割错误。
    # 结果存成 eval_labels, 只影响评估轨道 (.fasta), 不影响训练 (.pt)。
    eval_labels = new_labels.clone()
    final_noise_mask = (eval_labels == -1)
    final_noise_indices = torch.where(final_noise_mask)[0]

    if len(final_noise_indices) > 0 and len(centroids) > 0:
        MIN_LARGE_SIZE = {MIN_LARGE_CLUSTER_SIZE}

        # 筛选大簇
        large_cids = [cid for cid, sz in cluster_sizes.items() if sz >= MIN_LARGE_SIZE]
        if len(large_cids) == 0:
            # 兜底: 如果没有大簇, 用全部簇
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

    backup_and_replace(STEP2_FILE, old_after_revival, new_after_revival,
                       "添加精准 Zone III 归巢逻辑")

    # ══════════════════════════════════════════════════════════
    # Patch 2: 双轨制段用 eval_labels 替代 Clover 回填
    # ══════════════════════════════════════════════════════════
    print("\n📝 Patch 2: 双轨制评估段使用精准分配结果")

    old_dual_eval = '''\
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
    labels_for_eval_np = labels_for_eval.cpu().numpy()'''

    new_dual_eval = '''\
    # ── 轨道二: 全量版 Consensus（评估专用，应收尽收） ──
    # 使用精准归巢后的标签 (eval_labels), 已在死数据复活段完成:
    #   - 死数据复活: 通过 P75 阈值的 reads 已归簇
    #   - 精准归巢: 残留 -1 按 embedding 距离分配到大簇
    # 这两步共同实现"游子归巢"，避免 Clover 过分割继承
    labels_for_eval = eval_labels.clone()
    n_still_noise = int((labels_for_eval == -1).sum().item())

    print(f"\\n   🔓 轨道二: 全量版 Consensus（评估专用）")
    print(f"      使用精准归巢结果, 残留 -1: {n_still_noise}")
    labels_for_eval_np = labels_for_eval.cpu().numpy()'''

    backup_and_replace(STEP2_FILE, old_dual_eval, new_dual_eval,
                       "双轨制评估段使用 eval_labels")

    print("\n" + "=" * 60)
    print("  ✅ 补丁完成")
    print("=" * 60)
    print(f"""
  核心改动:
    - 新增精准归巢段 (死数据复活后): 残留 -1 → 最近大簇
    - 大簇阈值: size >= {MIN_LARGE_CLUSTER_SIZE}
    - 训练轨道 (.pt): new_labels 保持 -1, 纯净靶点
    - 评估轨道 (.fasta): eval_labels 全量归巢, 最大化重建

  预期效果:
    - 大簇 reads 数增加 → 小簇数减少 → 大簇 SR ~99% 的红利被充分利用
    - 替代原来的 Clover 回填, 避免继承过分割错误
    - 总 SR 有望突破 91% Clover baseline

  调参空间:
    MIN_LARGE_CLUSTER_SIZE = {MIN_LARGE_CLUSTER_SIZE} (保守值)
    可在脚本顶部调整; 降到 5 会让更多小簇参与, 但有错配风险
""")


if __name__ == "__main__":
    main()