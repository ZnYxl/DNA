#!/usr/bin/env python3
"""
apply_calibration_v2.py
========================
方案 1: rnnblock 校准阶段 (Calibration Phase)

修改文件:
  1. models/step1_train.py
     - import 增加 masked_bayes_risk
     - 主训练循环结束后、checkpoint 保存前插入 calibration phase

使用方法:
  cd /mnt/st_data/liangxinyi/code/
  python apply_calibration_v2.py

会自动备份 .bak 文件。
"""
import os
import shutil
import sys

# ──────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────
BASE_DIR = "/mnt/st_data/liangxinyi/code"
FILES = {
    "step1_train": os.path.join(BASE_DIR, "models", "step1_train.py"),
}


def apply_replacement(filepath, old_str, new_str, description):
    """单次精确替换，失败则报错退出"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    count = content.count(old_str)
    if count == 0:
        print(f"  ❌ 未找到目标字符串: {description}")
        print(f"     文件: {filepath}")
        print(f"     搜索: {repr(old_str[:80])}...")
        return False
    if count > 1:
        print(f"  ⚠️ 目标字符串出现 {count} 次 (预期 1 次): {description}")
        print(f"     文件: {filepath}")
        return False

    content = content.replace(old_str, new_str)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  ✅ {description}")
    return True


def main():
    print("=" * 70)
    print("  SSI-EC 方案1: rnnblock Calibration Phase 补丁")
    print("=" * 70)

    # ──────────────────────────────────────────────────────
    # 检查文件存在
    # ──────────────────────────────────────────────────────
    for name, path in FILES.items():
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            sys.exit(1)

    # ──────────────────────────────────────────────────────
    # 备份
    # ──────────────────────────────────────────────────────
    print("\n📦 备份原始文件...")
    for name, path in FILES.items():
        bak = path + ".bak_calib"
        shutil.copy2(path, bak)
        print(f"  💾 {os.path.basename(path)} → {os.path.basename(bak)}")

    # ──────────────────────────────────────────────────────
    # 修改 1: import 增加 masked_bayes_risk
    # ──────────────────────────────────────────────────────
    print("\n🔧 修改 step1_train.py ...")

    ok = apply_replacement(
        FILES["step1_train"],
        "from models.step1_model import Step1EvidentialModel, load_pretrained_feddna",
        "from models.step1_model import Step1EvidentialModel, load_pretrained_feddna, masked_bayes_risk",
        "import 增加 masked_bayes_risk"
    )
    if not ok:
        sys.exit(1)

    # ──────────────────────────────────────────────────────
    # 修改 2: 插入 Calibration Phase
    # ──────────────────────────────────────────────────────
    # 插入点: 主训练循环最后一行探针日志之后, checkpoint 保存之前
    OLD_ANCHOR = """            print(f"   🔬 探针 B | cos_pos: {cp_s}  cos_neg: {cn_s}  margin: {margin_s}")

    # =====================================================================
    # 9. 保存 checkpoint
    # ===================================================================="""

    CALIBRATION_BLOCK = '''            print(f"   🔬 探针 B | cos_pos: {cp_s}  cos_neg: {cn_s}  margin: {margin_s}")

    # =====================================================================
    # 8.5 Calibration Phase: rnnblock 校准
    # =====================================================================
    # 原理: 对比训练把 encoder 特征空间彻底重组 (cos_neg 0.98→0.04),
    # rnnblock 来不及完全适应最终特征空间。冻结 encoder, 只用 recon_loss
    # 微调 rnnblock, 让它在静止空间里专心学解码。
    # 类似 SimCLR/MoCo 做完预训练后 fine-tune linear probe 的标准做法。
    calib_epochs = getattr(args, 'calib_epochs', 3)
    calib_lr     = getattr(args, 'calib_lr', 2e-5)

    if calib_epochs > 0:
        print("\\n" + "=" * 60)
        print(f"🎯 Calibration Phase: rnnblock 校准 ({calib_epochs} epochs, lr={calib_lr})")
        print("=" * 60)

        # ── 冻结 Encoder + Length Adapter ──
        for p in model.encoder.parameters():
            p.requires_grad = False
        if model.length_adapter is not None:
            for p in model.length_adapter.parameters():
                p.requires_grad = False

        # ── 确保 rnnblock 可训练 ──
        calib_params = list(model.rnnblock.parameters())
        for p in calib_params:
            p.requires_grad = True

        trainable_count = sum(p.numel() for p in calib_params if p.requires_grad)
        frozen_count    = sum(p.numel() for p in model.parameters()) - trainable_count
        print(f"   🔒 冻结: {frozen_count:,} 参数 (Encoder + Length Adapter)")
        print(f"   🔓 可训练: {trainable_count:,} 参数 (RNNBlock)")

        # ── 全新 optimizer (丢弃主训练阶段的动量污染) ──
        calib_optimizer = optim.AdamW(calib_params, lr=calib_lr, weight_decay=1e-4)

        model.train()
        model.encoder.eval()  # BN 统计量不动, Dropout 关闭

        for calib_epoch in range(calib_epochs):
            calib_start = time.time()

            # 复用主训练的动态采样器
            calib_batches = create_dynamic_sampler(
                dataset,
                batch_size=args.batch_size,
                max_clusters_per_batch=args.max_clusters_per_batch,
                state_path=prev_state,
                round_idx=round_idx
            )
            calib_sampler = ListBatchSampler(calib_batches)
            calib_loader  = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=calib_sampler,
                num_workers=4,
                pin_memory=True
            )

            calib_loss_sum = 0
            calib_str_sum  = 0
            calib_n        = 0

            for i, batch_data in enumerate(calib_loader):
                reads_batch     = batch_data['encoding'].to(device)
                consensus_batch = batch_data['consensus_target'].to(device)

                # ── 只走 encoder → decoder → recon_loss ──
                embeddings, _ = model.encode_reads(reads_batch)
                evidence, strength, alpha = model.decode_to_evidence(embeddings)

                recon_loss = masked_bayes_risk(evidence, consensus_batch)

                calib_optimizer.zero_grad()
                recon_loss.backward()
                torch.nn.utils.clip_grad_norm_(calib_params, max_norm=1.0)
                calib_optimizer.step()

                calib_loss_sum += recon_loss.item()
                calib_str_sum  += strength.mean().item()
                calib_n        += 1

                if (i + 1) % 100 == 0:
                    print(f"   [Calib Batch {i+1}/{len(calib_batches)}] "
                          f"Recon: {recon_loss.item():.4f} | "
                          f"Str: {strength.mean().item():.1f}", end='\\r')

            calib_time = time.time() - calib_start
            avg_loss = calib_loss_sum / max(calib_n, 1)
            avg_str  = calib_str_sum / max(calib_n, 1)
            print(f"\\n   ✅ Calib Epoch {calib_epoch+1}/{calib_epochs} ({calib_time:.1f}s) | "
                  f"Recon: {avg_loss:.4f} | Str: {avg_str:.1f}")

        # ── 恢复全部参数为可训练状态 (不影响后续保存) ──
        for p in model.parameters():
            p.requires_grad = True

        print(f"   🎯 Calibration 完成")

    # =====================================================================
    # 9. 保存 checkpoint
    # ===================================================================='''

    ok = apply_replacement(
        FILES["step1_train"],
        OLD_ANCHOR,
        CALIBRATION_BLOCK,
        "插入 Calibration Phase (8.5节)"
    )
    if not ok:
        sys.exit(1)

    # ──────────────────────────────────────────────────────
    # 完成
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✅ 补丁应用完成!")
    print("=" * 70)
    print()
    print("  修改文件:")
    print(f"    {FILES['step1_train']}")
    print()
    print("  新增行为:")
    print("    - 主训练结束后, 冻结 Encoder, 纯 recon_loss 微调 RNNBlock 3 epochs")
    print("    - 可通过 args.calib_epochs / args.calib_lr 控制 (默认 3 epochs, lr=2e-5)")
    print("    - 设 calib_epochs=0 可跳过校准阶段")
    print()
    print("  回滚方法:")
    print(f"    cp {FILES['step1_train']}.bak_calib {FILES['step1_train']}")
    print()
    print("  下一步: 重新跑实验, 对比 reconstruction 指标")


if __name__ == "__main__":
    main()