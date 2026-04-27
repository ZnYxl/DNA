#!/usr/bin/env python3
"""
rollback_feddna_native.py — 回退错误的 FedDNA 原生模型补丁

问题分析:
  项目设计哲学: SSI-EC 的 read 和 consensus 都 pad 到 max_length (201)，
                 不存在长度映射需求。ConMamba encoder 和 RNNBlock 都是
                 length-preserving / length-agnostic 的。

  apply_feddna_native.py 的错误:
    - 引入 FedDNA 原生 Model(noise_length=155, label_length=150)
    - 强制走 Linear(155, 150) 压缩
    - 下游 ds_fusion_masked 期望 pmask 长度 = 201
    - 人为制造长度冲突 → 崩溃

  正确的做法 (之前 [FIX-SR-v2] 已验证 SR=85%):
    - 用 Step1EvidentialModel(max_length=201) 构造
    - load_pretrained_feddna 加载: encoder ✓, rnnblock ✓
    - length_adapter 被跳过 (因为 shape 不匹配), 设为 None
    - 流水线全程 201 长度, 无需任何适配

用法:
  python rollback_feddna_native.py
"""
import os, shutil

STEP2_FILE = "/mnt/st_data/liangxinyi/code/models/step2_runner.py"
BACKUP_FILE = STEP2_FILE + ".bak_feddna_native"


def main():
    print("=" * 60)
    print("  Rollback apply_feddna_native.py")
    print("=" * 60)

    if not os.path.exists(BACKUP_FILE):
        print(f"\n  ❌ 找不到备份: {BACKUP_FILE}")
        print(f"     说明 apply_feddna_native.py 可能没有打上，或备份被删除")
        return

    # 把当前状态也备份一下，以防万一
    current_backup = STEP2_FILE + ".bak_before_rollback"
    if not os.path.exists(current_backup):
        shutil.copy2(STEP2_FILE, current_backup)
        print(f"  💾 当前状态备份: {current_backup}")

    # 恢复
    shutil.copy2(BACKUP_FILE, STEP2_FILE)
    print(f"  ✅ 已恢复 step2_runner.py 到 apply_feddna_native 之前的状态")

    # 验证恢复后的状态
    with open(STEP2_FILE, 'r') as f:
        content = f.read()

    print("\n  🔍 验证恢复后的状态:")
    checks = [
        ('双轨制 Consensus', '[G老师-双轨制]' in content or 'consensus_dict_for_train' in content),
        ('Step1EvidentialModel 解码', 'Step1EvidentialModel(' in content and '_use_feddna_decode' in content),
        ('load_pretrained_feddna 调用', 'load_pretrained_feddna(' in content),
        ('没有错误的 FedDNAModel', 'from models.Model import Model as FedDNAModel' not in content),
        ('没有错误的 _SSIECDecodeAdapter', '_SSIECDecodeAdapter' not in content),
        ('没有 noise_length 硬编码', 'noise_length=155' not in content),
    ]

    all_ok = True
    for name, ok in checks:
        symbol = '✅' if ok else '❌'
        print(f"     {symbol} {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  ✅ 状态正确，可以重跑实验")
        print("\n  预期:")
        print("    - 聚类: R3 Purity≈0.9699, Acc(γ=1.00)≈0.8506")
        print("    - 重建: SR≈85% (恢复 [FIX-SR-v2] 的效果)")
    else:
        print("  ⚠️ 状态不完全正确，建议手动检查 step2_runner.py")


if __name__ == "__main__":
    main()