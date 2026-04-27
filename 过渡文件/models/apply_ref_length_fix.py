#!/usr/bin/env python3
"""
apply_ref_length_fix.py
========================
修复 ref_length 先验长度未生效的 Bug。

问题：
  main_loop.py 传了 --ref_length 196 给 step2_runner，
  step2_runner 传给了 run_feddna_decode，但 run_feddna_decode 声明了参数却没用。
  save_consensus_fasta 根本没接收 ref_length，用的是 read 长度众数。
  当簇内 deletion reads 占多数时，众数=195，consensus 被截短 1bp → ED=1。

修改文件：
  1. models/step2_decode.py — save_consensus_fasta 增加 ref_length 参数
  2. models/step2_runner.py — 调用时传入 ref_length

使用方法：
  cd /mnt/st_data/liangxinyi/code/
  python apply_ref_length_fix.py

⚠️ 注意：启动实验时必须传 --ref_length 196（Seq_1D 的 reference 长度）
"""
import os
import shutil
import sys

BASE_DIR = "/mnt/st_data/liangxinyi/code"
FILES = {
    "step2_decode":  os.path.join(BASE_DIR, "models", "step2_decode.py"),
    "step2_runner":  os.path.join(BASE_DIR, "models", "step2_runner.py"),
}


def apply_replacement(filepath, old_str, new_str, description):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    count = content.count(old_str)
    if count == 0:
        print(f"  ❌ 未找到: {description}")
        print(f"     文件: {filepath}")
        print(f"     搜索: {repr(old_str[:80])}...")
        return False
    if count > 1:
        print(f"  ⚠️ 出现 {count} 次 (预期 1 次): {description}")
        return False
    content = content.replace(old_str, new_str)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  ✅ {description}")
    return True


def main():
    print("=" * 70)
    print("  ref_length 先验长度修复补丁")
    print("=" * 70)

    for name, path in FILES.items():
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            sys.exit(1)

    print("\n📦 备份...")
    for name, path in FILES.items():
        bak = path + ".bak_reflen"
        shutil.copy2(path, bak)
        print(f"  💾 {os.path.basename(path)} → {os.path.basename(bak)}")

    # ══════════════════════════════════════════════════════════════
    # 修改 1: step2_decode.py — save_consensus_fasta 增加 ref_length
    # ══════════════════════════════════════════════════════════════
    print("\n🔧 修改 step2_decode.py ...")

    ok = apply_replacement(
        FILES["step2_decode"],
        # ── 旧函数签名 ──
        """def save_consensus_fasta(
    consensus_dict: Dict[int, torch.Tensor],
    new_labels_np: np.ndarray,
    flat_real_indices,
    data_loader,
    model_max_len: int,
    fasta_path: str,
):""",
        # ── 新函数签名 ──
        """def save_consensus_fasta(
    consensus_dict: Dict[int, torch.Tensor],
    new_labels_np: np.ndarray,
    flat_real_indices,
    data_loader,
    model_max_len: int,
    fasta_path: str,
    ref_length: int = None,
):""",
        "save_consensus_fasta 增加 ref_length 参数"
    )
    if not ok:
        sys.exit(1)

    # 修改截断逻辑：有先验时用先验，没有时用众数
    ok = apply_replacement(
        FILES["step2_decode"],
        """    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    with open(fasta_path, 'w') as ff:
        for cluster_id, one_hot in sorted(consensus_dict.items()):
            actual_len = cluster_actual_len.get(cluster_id, model_max_len)
            indices = one_hot[:actual_len].argmax(dim=-1).numpy()
            seq = ''.join(BASE_MAP[i] for i in indices)
            ff.write(f">cluster_{cluster_id}\\n{seq}\\n")

    print(f"   💾 FedDNA Consensus FASTA: {fasta_path}")""",
        """    # [ref_length-FIX] 有先验时统一用先验长度，没有时退回众数
    if ref_length is not None:
        print(f"   📏 使用先验 ref_length={ref_length} 截断 (覆盖 read 众数)")
        mode_mismatch = sum(1 for cid, ml in cluster_actual_len.items() if ml != ref_length)
        if mode_mismatch > 0:
            print(f"   ⚠️ {mode_mismatch} 个簇的 read 众数 ≠ ref_length (将被先验覆盖)")

    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    with open(fasta_path, 'w') as ff:
        for cluster_id, one_hot in sorted(consensus_dict.items()):
            if ref_length is not None:
                actual_len = ref_length
            else:
                actual_len = cluster_actual_len.get(cluster_id, model_max_len)
            indices = one_hot[:actual_len].argmax(dim=-1).numpy()
            seq = ''.join(BASE_MAP[i] for i in indices)
            ff.write(f">cluster_{cluster_id}\\n{seq}\\n")

    print(f"   💾 FedDNA Consensus FASTA: {fasta_path}")""",
        "截断逻辑: 有先验用先验, 无先验用众数"
    )
    if not ok:
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════
    # 修改 2: step2_runner.py — 调用时传入 ref_length
    # ══════════════════════════════════════════════════════════════
    print("\n🔧 修改 step2_runner.py ...")

    ok = apply_replacement(
        FILES["step2_runner"],
        """        save_consensus_fasta(
            consensus_dict_for_eval, labels_for_eval_np, flat_real_indices,
            data_loader, model_max_len, fasta_path
        )""",
        """        save_consensus_fasta(
            consensus_dict_for_eval, labels_for_eval_np, flat_real_indices,
            data_loader, model_max_len, fasta_path,
            ref_length=getattr(args, 'ref_length', None),
        )""",
        "save_consensus_fasta 传入 ref_length"
    )
    if not ok:
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  ✅ 补丁完成!")
    print("=" * 70)
    print()
    print("  修改文件:")
    for name, path in FILES.items():
        print(f"    {path}")
    print()
    print("  ⚠️  启动实验时务必加: --ref_length 196")
    print()
    print("  回滚:")
    for name, path in FILES.items():
        print(f"    cp {path}.bak_reflen {path}")


if __name__ == "__main__":
    main()