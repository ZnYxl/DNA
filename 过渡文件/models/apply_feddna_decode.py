#!/usr/bin/env python3
"""
apply_feddna_decode.py — 解码时使用完整 FedDNA 模型

核心思路:
  - 聚类标签: 来自 SSI-EC 模型（encoder 经对比学习优化，purity=0.964）·
  - 共识解码: 用 FedDNA 原始完整模型（encoder + rnnblock 都是原始的）

  SSI-EC 告诉你哪些 reads 属于同一个簇，
  FedDNA 负责把同簇 reads 解码成 consensus。
  两者各做自己擅长的事，互不干扰。

涉及文件:
  1. main_loop.py   — step2_args 加入 feddna_checkpoint（如果尚未添加）
  2. step2_runner.py — 构建独立 FedDNA 模型用于解码

用法:
  python apply_feddna_decode.py
"""
import os
import shutil

# ============================================================
# 配置
# ============================================================
CODE_DIR = "/mnt/st_data/liangxinyi/code/models"
MAIN_LOOP = os.path.join(CODE_DIR, "main_loop.py")
STEP2_RUNNER = os.path.join(CODE_DIR, "step2_runner.py")


def patch_file(filepath, old_str, new_str, description):
    if not os.path.exists(filepath):
        print(f"  ❌ 文件不存在: {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if old_str not in content:
        # 检查是否已有新内容
        if new_str.strip()[:60] in content:
            print(f"  ⏭️  已打过补丁，跳过: {description}")
            return True
        print(f"  ⚠️ 未找到目标字符串: {description}")
        print(f"      预期前50字符: {repr(old_str[:80])}")
        return False

    bak = filepath + '.bak_feddna_decode'
    if not os.path.exists(bak):
        shutil.copy2(filepath, bak)
        print(f"  💾 备份: {bak}")

    content = content.replace(old_str, new_str, 1)
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  ✅ {description}")
    return True


def main():
    print("=" * 60)
    print("  Apply FedDNA decode patch")
    print("=" * 60)

    # ==================================================================
    # Patch 1: main_loop.py — 传递 feddna_checkpoint 给 step2
    # ==================================================================
    print("\n📝 Patch 1: main_loop.py — 传递 feddna_checkpoint")

    # 尝试两种可能的现有代码（有无之前的补丁）
    old_main_v1 = """\
            disable_merge=getattr(args, 'disable_merge', False),
            ref_length=getattr(args, 'ref_length', None),
        )"""

    old_main_v2 = """\
            disable_merge=getattr(args, 'disable_merge', False),
            ref_length=getattr(args, 'ref_length', None),
            feddna_checkpoint=args.feddna_checkpoint,
        )"""

    new_main = """\
            disable_merge=getattr(args, 'disable_merge', False),
            ref_length=getattr(args, 'ref_length', None),
            feddna_checkpoint=args.feddna_checkpoint,
        )"""

    with open(MAIN_LOOP, 'r') as f:
        ml_content = f.read()

    if 'feddna_checkpoint=args.feddna_checkpoint,' in ml_content:
        print("  ⏭️  feddna_checkpoint 已传递，跳过")
    else:
        patch_file(MAIN_LOOP, old_main_v1, new_main,
                   "step2_args 加入 feddna_checkpoint")

    # ==================================================================
    # Patch 2: step2_runner.py — 解码时用完整 FedDNA 模型
    # ==================================================================
    print("\n📝 Patch 2: step2_runner.py — FedDNA 完整模型解码")

    # 匹配当前的解码段（可能是原始版或上次 swap 补丁后的）
    # 先尝试匹配原始版
    old_decode_original = """\
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
    print(f"   ✅ 生成 {len(consensus_dict)} 个簇的 consensus")"""

    new_decode = """\
    # [FIX-DECODE] 用 FedDNA ds_fusion 替换 majority-vote consensus
    from models.step2_decode import run_feddna_decode

    # ── [FIX-SR-v2] 用完整 FedDNA 原始模型做 consensus 解码 ──
    # 原因: SSI-EC 训练改变了 encoder 的输出分布，与原始 rnnblock 不兼容。
    #        单独换 rnnblock 会导致 encoder↔decoder 不匹配（SR=0%）。
    # 方案: 构建独立的 FedDNA 模型（encoder + rnnblock 都是原始权重），
    #        只用于解码。聚类标签来自 SSI-EC，解码质量来自 FedDNA。
    _feddna_ckpt_path = getattr(args, 'feddna_checkpoint', None)
    _use_feddna_decode = False

    if _feddna_ckpt_path and os.path.exists(_feddna_ckpt_path):
        try:
            print(f"   🔧 [FIX-SR-v2] 构建 FedDNA 解码模型...")
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
            print(f"   ✅ FedDNA 解码模型就绪 (来自 {os.path.basename(_feddna_ckpt_path)})")
        except Exception as e:
            print(f"   ⚠️ FedDNA 解码模型构建失败: {e}，回退到 SSI-EC 模型")
    else:
        print(f"   ⚠️ 未提供 feddna_checkpoint，使用 SSI-EC 模型解码")

    _decode_model = feddna_decode_model if _use_feddna_decode else model
    _decode_model.to(device)

    consensus_dict = run_feddna_decode(
        model=_decode_model,
        data_loader=data_loader,
        new_labels_np=new_labels_np_for_consensus,  # [v2] 含 Zone III 软参与
        flat_real_indices=flat_real_indices,
        model_max_len=model_max_len,
        device=device,
        batch_size=getattr(args, 'batch_size', 512),
        ref_length=getattr(args, 'ref_length', None),   # [v5] 先验长度
    )

    # 释放 FedDNA 解码模型
    if _use_feddna_decode:
        del feddna_decode_model
    _decode_model.cpu()
    model.cpu()
    torch.cuda.empty_cache()
    print(f"   ✅ 生成 {len(consensus_dict)} 个簇的 consensus")"""

    # 检查是否需要回滚上一次 swap 补丁
    with open(STEP2_RUNNER, 'r') as f:
        s2r_content = f.read()

    if '[FIX-SR-v2]' in s2r_content:
        print("  ⏭️  已打过 v2 补丁，跳过")
    elif '[FIX-SR] 解码时 swap 回 FedDNA 原始 rnnblock' in s2r_content:
        # 上次 swap 补丁在，需要先恢复再打新补丁
        bak_file = STEP2_RUNNER + '.bak_rnnswap'
        if os.path.exists(bak_file):
            shutil.copy2(bak_file, STEP2_RUNNER)
            print("  🔄 回滚上次 rnnblock swap 补丁")
            # 重新读取
            with open(STEP2_RUNNER, 'r') as f:
                s2r_content = f.read()
            if old_decode_original in s2r_content:
                patch_file(STEP2_RUNNER, old_decode_original, new_decode,
                           "FedDNA 完整模型解码")
            else:
                print("  ❌ 回滚后仍无法找到目标字符串")
        else:
            print("  ❌ 找不到 .bak_rnnswap 备份文件")
    elif old_decode_original in s2r_content:
        patch_file(STEP2_RUNNER, old_decode_original, new_decode,
                   "FedDNA 完整模型解码")
    else:
        print("  ⚠️ 无法匹配现有解码段，请手动检查 step2_runner.py")

    # ==================================================================
    # Patch 3: 确保 step2_runner.py 有 load_pretrained_feddna 的 import
    # ==================================================================
    print("\n📝 Patch 3: 确保 import load_pretrained_feddna")

    with open(STEP2_RUNNER, 'r') as f:
        s2r_content = f.read()

    if 'load_pretrained_feddna' not in s2r_content:
        old_import = "from models.step1_model import Step1EvidentialModel"
        new_import = "from models.step1_model import Step1EvidentialModel, load_pretrained_feddna"
        patch_file(STEP2_RUNNER, old_import, new_import,
                   "添加 load_pretrained_feddna import")
    else:
        print("  ⏭️  load_pretrained_feddna 已导入")

    # ==================================================================
    # Patch 4: 确保 model_dim / num_clusters 在解码段可见
    # ==================================================================
    # model_dim 和 num_clusters 在 run_step2 函数开头已定义，
    # 解码段在同一函数内，无需额外处理。
    print("\n  ℹ️  model_dim / num_clusters 已在 run_step2 作用域内")

    print("\n" + "=" * 60)
    print("  ✅ 补丁完成")
    print("=" * 60)
    print("""
  使用说明:
  1. 确保 step1_train.py 用旧版配置（不冻 rnnblock）
  2. 确保 step2_runner.py 用全轮次 Zone III 隔离
  3. 重跑实验
  4. 预期: 聚类=v5水平 (Purity R3≈0.964), 重建 SR≈91%
""")


if __name__ == "__main__":
    main()