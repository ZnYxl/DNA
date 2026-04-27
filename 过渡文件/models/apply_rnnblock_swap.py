#!/usr/bin/env python3
"""
apply_rnnblock_swap.py — 解码时使用 FedDNA 原始 rnnblock

改动逻辑:
  训练时: rnnblock 正常参与训练（保持迭代引擎运转）
  解码时: 临时换入 FedDNA 原始 rnnblock 权重（保护重建质量）

  聚类只用 encoder pooled embedding，跟 rnnblock 无关。
  重建才用 rnnblock。两个任务在推理时完全可以拆开。

涉及文件:
  1. main_loop.py   — step2_args 加入 feddna_checkpoint
  2. step2_runner.py — 解码前 swap rnnblock，解码后 swap back

用法:
  python apply_rnnblock_swap.py

  会自动备份 .bak，然后原地修改。
"""
import os
import shutil

# ============================================================
# 配置: 代码路径
# ============================================================
CODE_DIR = "/mnt/st_data/liangxinyi/code/models"
MAIN_LOOP = os.path.join(CODE_DIR, "main_loop.py")
STEP2_RUNNER = os.path.join(CODE_DIR, "step2_runner.py")


def patch_file(filepath, old_str, new_str, description):
    """在文件中做字符串替换，带备份"""
    if not os.path.exists(filepath):
        print(f"  ❌ 文件不存在: {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if old_str not in content:
        print(f"  ⚠️ 未找到目标字符串，可能已经打过补丁: {description}")
        return False

    # 备份
    bak = filepath + '.bak_rnnswap'
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
    print("  Apply rnnblock swap patch")
    print("=" * 60)

    # ==================================================================
    # Patch 1: main_loop.py — 传递 feddna_checkpoint 给 step2
    # ==================================================================
    print("\n📝 Patch 1: main_loop.py")

    old_main = """\
            disable_merge=getattr(args, 'disable_merge', False),
            ref_length=getattr(args, 'ref_length', None),
        )"""

    new_main = """\
            disable_merge=getattr(args, 'disable_merge', False),
            ref_length=getattr(args, 'ref_length', None),
            feddna_checkpoint=args.feddna_checkpoint,
        )"""

    patch_file(MAIN_LOOP, old_main, new_main,
               "step2_args 加入 feddna_checkpoint")

    # ==================================================================
    # Patch 2: step2_runner.py — 解码时 swap rnnblock
    # ==================================================================
    print("\n📝 Patch 2: step2_runner.py")

    old_decode = """\
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
    # 模型已在 model.cpu() 后，需要重新上 GPU
    model.to(device)

    # ── [FIX-SR] 解码时 swap 回 FedDNA 原始 rnnblock ──
    # 原因: 对比学习的梯度会扰动 rnnblock，导致 SR 逐轮下降 (91%→84%)。
    # 但聚类只依赖 encoder 的 pooled embedding，跟 rnnblock 无关。
    # 所以: 训练时 rnnblock 正常参与（保持迭代引擎），
    #        解码时临时换入 FedDNA 原始权重（保护重建质量）。
    import copy
    _trained_rnnblock_sd = copy.deepcopy(model.rnnblock.state_dict())
    _feddna_ckpt_path = getattr(args, 'feddna_checkpoint', None)
    _swapped = False
    if _feddna_ckpt_path and os.path.exists(_feddna_ckpt_path):
        try:
            _feddna_sd = torch.load(_feddna_ckpt_path, map_location=device)
            if isinstance(_feddna_sd, dict) and 'model_state_dict' in _feddna_sd:
                _feddna_sd = _feddna_sd['model_state_dict']
            # 提取 rnnblock 权重 (兼容 FedDNA 的 key 格式)
            _rnn_sd = {}
            for k, v in _feddna_sd.items():
                if 'rnnblock' in k:
                    # FedDNA: "rnnblock.rnn.weight_ih_l0" → 直接用
                    # 或可能带前缀 "model.rnnblock.xxx" → 去掉 "model."
                    clean_k = k.replace('model.', '') if k.startswith('model.') else k
                    _rnn_sd[clean_k] = v
            if _rnn_sd:
                # 去掉 "rnnblock." 前缀以匹配 model.rnnblock.load_state_dict()
                _rnn_sd_clean = {k.replace('rnnblock.', ''): v for k, v in _rnn_sd.items()}
                model.rnnblock.load_state_dict(_rnn_sd_clean, strict=True)
                _swapped = True
                print(f"   🔄 [FIX-SR] 解码用 FedDNA 原始 rnnblock (来自 {os.path.basename(_feddna_ckpt_path)})")
            else:
                print(f"   ⚠️ FedDNA checkpoint 中未找到 rnnblock 权重，使用训练后的 rnnblock")
        except Exception as e:
            print(f"   ⚠️ 加载 FedDNA rnnblock 失败: {e}，使用训练后的 rnnblock")
    else:
        print(f"   ⚠️ 未提供 feddna_checkpoint，使用训练后的 rnnblock")

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

    # ── [FIX-SR] 恢复训练后的 rnnblock（不影响后续流程） ──
    if _swapped:
        model.rnnblock.load_state_dict(_trained_rnnblock_sd)
        print(f"   🔄 rnnblock 已恢复为训练后权重")
    del _trained_rnnblock_sd

    model.cpu()
    torch.cuda.empty_cache()
    print(f"   ✅ 生成 {len(consensus_dict)} 个簇的 consensus")"""

    patch_file(STEP2_RUNNER, old_decode, new_decode,
               "解码时 swap FedDNA 原始 rnnblock")

    print("\n" + "=" * 60)
    print("  ✅ 补丁完成")
    print("=" * 60)
    print("\n  注意: 此补丁需要配合旧版代码使用（不冻 rnnblock + 全轮次 Zone III 隔离）")
    print("  请确认 step1_train.py 中 rnnblock 没有被冻结")
    print("  重跑实验后对比 SR 是否恢复到 ~91%")


if __name__ == "__main__":
    main()