#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
  SSI-EC v17 Patch: FASTA 纯净轨道 (方案 B)
===============================================================================

目标:
  彻底切断归巢 backfill 对 FASTA 的污染. v15/v16 的 FASTA 一直用
  consensus_dict_for_eval (基于归巢后的 labels_for_eval_np), 后者含
  R1=15K / R2=31K / R3=47K 条 backfill 脏 reads, 越滚越脏.

  v17 新建一条"轨道三": MV on new_labels_np (严格 labels, 不接受 backfill).
  与 v16 的训练 target 同源 (都是 MV), 跟 encoder 状态无关.
  三条轨道并存, FASTA 只走轨道三.

改动 2 个文件:
  1. models/step2_runner.py
       在 save_consensus_fasta 调用前, 根据 args.fasta_source 分支:
         - 'mv_strict' (默认, v17 行为): 用 compute_mv_consensus(new_labels_np)
         - 'fusion_eval' (v16 fallback): 用 consensus_dict_for_eval + eval_labels

  2. models/main_loop.py
       新增 argparse: --fasta_source {mv_strict, fusion_eval}  default=mv_strict
       新增传递:       step2_args.fasta_source = args.fasta_source

前置条件:
  必须已经打过 v16 patch (apply_mv_target_v16.py). 本脚本会检查
  compute_mv_consensus 是否存在, 不存在则中止.

用法:
  cd /mnt/st_data/liangxinyi/code/models/
  python apply_fasta_clean_v17.py
  # 生成 .bak (v17) 备份, 与 v16 的 .bak 不冲突 (v16 用 .bak, v17 用 .bak_v17)

  回滚:
  mv models/step2_runner.py.bak_v17 models/step2_runner.py
  mv models/main_loop.py.bak_v17    models/main_loop.py

  启动 v17:
  python -m models.main_loop ... --consensus_source mv --fasta_source mv_strict \\
                                 --ref_length 196 --disable_merge

  对照实验 (退回 v16 FASTA 行为):
  python -m models.main_loop ... --consensus_source mv --fasta_source fusion_eval ...
"""
import os
import sys
import shutil


# ===========================================================================
# 配置
# ===========================================================================
def _resolve_models_dir():
    env = os.environ.get("SSIEC_MODELS_DIR")
    if env and env.strip():
        return env.strip()
    if os.path.isdir("models") and os.path.isfile("models/step2_runner.py"):
        return "models"
    if os.path.isfile("step2_runner.py") and os.path.isfile("main_loop.py"):
        return "."
    return "models"


MODELS_DIR = _resolve_models_dir()
FILES = {
    "step2_runner": os.path.join(MODELS_DIR, "step2_runner.py"),
    "main_loop":    os.path.join(MODELS_DIR, "main_loop.py"),
    "step2_decode": os.path.join(MODELS_DIR, "step2_decode.py"),  # 仅用于前置检查
}


# ===========================================================================
# 工具
# ===========================================================================
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def backup_v17(path):
    bak = path + ".bak_v17"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"   💾 备份 (v17): {bak}")
    else:
        print(f"   ⚠️ v17 备份已存在, 跳过: {bak}")

def replace_exact(content, old, new, file_tag):
    count = content.count(old)
    if count == 0:
        raise RuntimeError(
            f"[{file_tag}] 锚点未找到 (可能是 v16 patch 未打 或 版本不同):\n"
            f"{'-'*70}\n{old[:400]}...\n{'-'*70}"
        )
    if count > 1:
        raise RuntimeError(
            f"[{file_tag}] 锚点匹配到 {count} 处, 不唯一:\n"
            f"{'-'*70}\n{old[:400]}...\n{'-'*70}"
        )
    return content.replace(old, new, 1)


# ===========================================================================
# 前置检查: v16 patch 必须已打
# ===========================================================================
def check_v16_applied():
    print("🔍 前置检查: 确认 v16 patch 已打上...")

    decode = read_file(FILES["step2_decode"])
    if "def compute_mv_consensus(" not in decode:
        print("❌ compute_mv_consensus 不存在于 step2_decode.py")
        print("   请先运行 apply_mv_target_v16.py 再运行本脚本")
        sys.exit(1)
    print("   ✓ compute_mv_consensus 已就位")

    runner = read_file(FILES["step2_runner"])
    if "[v16 路径B] 训练靶子切换" not in runner:
        print("❌ step2_runner.py 缺少 v16 分支")
        print("   请先运行 apply_mv_target_v16.py")
        sys.exit(1)
    print("   ✓ v16 训练靶子分支已就位")

    main = read_file(FILES["main_loop"])
    if "--consensus_source" not in main:
        print("❌ main_loop.py 缺少 --consensus_source")
        print("   请先运行 apply_mv_target_v16.py")
        sys.exit(1)
    print("   ✓ --consensus_source argparse 已就位")


# ===========================================================================
# Patch 1: step2_runner.py 的 FASTA 保存逻辑
# ---------------------------------------------------------------------------
# 锚点: save_consensus_fasta 调用块, 替换成 fasta_source 分支
# ===========================================================================
STEP2_RUNNER_OLD = """    # [FIX-DECODE] 用 save_consensus_fasta 替换原 FASTA 保存逻辑
    from models.step2_decode import save_consensus_fasta
    fasta_dir = os.path.join(args.output_dir, "consensus")
    os.makedirs(fasta_dir, exist_ok=True)
    fasta_path = os.path.join(fasta_dir, f"consensus_{ts}.fasta")
    try:
        save_consensus_fasta(
            consensus_dict_for_eval, labels_for_eval_np, flat_real_indices,
            data_loader, model_max_len, fasta_path,
            ref_length=getattr(args, 'ref_length', None),
        )
        print(f"   💾 Fasta: {fasta_path}")
    except Exception as e:
        print(f"   ⚠️ Fasta 保存失败: {e}")
        fasta_path = None"""

STEP2_RUNNER_NEW = """    # ══════════════════════════════════════════════════════════════════
    # [v17 FASTA 纯净轨道] 方案 B: FASTA 的 consensus 与 labels 可选择
    # ══════════════════════════════════════════════════════════════════
    # v15/v16 的 FASTA 一直用 consensus_dict_for_eval (fusion on 归巢后
    # eval_labels), 后者含 R1=15K / R2=31K / R3=47K 条 backfill 脏 reads,
    # 越滚越脏, 是 EER 逐轮上升 & R0→R1 SR 倒退的直接原因.
    #
    # v17 新建轨道三: MV on new_labels_np (严格 labels, 零 backfill), 与 v16
    # 训练 target 同源 (都是 MV), 对 encoder 状态免疫, 不受 FedDNA
    # checkpoint 迁移不良影响.
    #
    # 默认 mv_strict (v17 行为), fusion_eval 退回 v16 行为做 A/B 对照.
    from models.step2_decode import save_consensus_fasta
    fasta_dir = os.path.join(args.output_dir, "consensus")
    os.makedirs(fasta_dir, exist_ok=True)
    fasta_path = os.path.join(fasta_dir, f"consensus_{ts}.fasta")

    _fasta_source = getattr(args, 'fasta_source', 'mv_strict')
    try:
        if _fasta_source == 'mv_strict':
            print(f"\\n   🧼 [v17] FASTA 纯净轨道: MV on 严格 labels "
                  f"(零 backfill 污染)")
            from models.step2_decode import compute_mv_consensus
            consensus_dict_for_fasta = compute_mv_consensus(
                data_loader=data_loader,
                new_labels_np=new_labels_np,
                flat_real_indices=flat_real_indices,
                model_max_len=model_max_len,
                ref_length=getattr(args, 'ref_length', None),
            )
            save_consensus_fasta(
                consensus_dict_for_fasta, new_labels_np, flat_real_indices,
                data_loader, model_max_len, fasta_path,
                ref_length=getattr(args, 'ref_length', None),
            )
        else:
            print(f"\\n   🩹 [v16 fallback] FASTA: fusion on 归巢 labels "
                  f"(backfill 污染)")
            save_consensus_fasta(
                consensus_dict_for_eval, labels_for_eval_np, flat_real_indices,
                data_loader, model_max_len, fasta_path,
                ref_length=getattr(args, 'ref_length', None),
            )
        print(f"   💾 Fasta: {fasta_path}")
    except Exception as e:
        print(f"   ⚠️ Fasta 保存失败: {e}")
        fasta_path = None"""


def patch_step2_runner():
    path = FILES["step2_runner"]
    print(f"\n📝 Patch 1/2: {path}")
    backup_v17(path)
    content = read_file(path)

    # 幂等检查
    if "[v17 FASTA 纯净轨道]" in content:
        print("   ⚠️ v17 FASTA 轨道已存在, 跳过")
        return

    new_content = replace_exact(content, STEP2_RUNNER_OLD, STEP2_RUNNER_NEW,
                                "step2_runner.py")
    write_file(path, new_content)
    print("   ✅ FASTA 保存逻辑替换为 mv_strict / fusion_eval 分支")


# ===========================================================================
# Patch 2: main_loop.py - 新增 argparse + 传递
# ===========================================================================
# 注意: v16 已经在 --freeze_consensus 后新增了 --consensus_source,
#       v17 要在 --consensus_source 之后再加 --fasta_source
# ===========================================================================
MAIN_LOOP_ARGPARSE_OLD = """    parser.add_argument('--consensus_source', type=str, default='mv',
                        choices=['mv', 'fusion'],
                        help='[v16 路径B] Round 2+ 训练靶子来源. '
                             'mv (默认) = majority vote (打破 encoder 自污染), '
                             'fusion = evidence fusion (v15 行为, 用于对照).')
    args = parser.parse_args()"""

MAIN_LOOP_ARGPARSE_NEW = """    parser.add_argument('--consensus_source', type=str, default='mv',
                        choices=['mv', 'fusion'],
                        help='[v16 路径B] Round 2+ 训练靶子来源. '
                             'mv (默认) = majority vote (打破 encoder 自污染), '
                             'fusion = evidence fusion (v15 行为, 用于对照).')
    parser.add_argument('--fasta_source', type=str, default='mv_strict',
                        choices=['mv_strict', 'fusion_eval'],
                        help='[v17 FASTA 纯净轨道] 导出 FASTA 的 consensus 来源. '
                             'mv_strict (默认) = MV on 严格 labels (零 backfill 污染), '
                             'fusion_eval = fusion on 归巢 labels (v16 行为, 用于对照).')
    args = parser.parse_args()"""


MAIN_LOOP_NAMESPACE_OLD = """            ref_length=getattr(args, 'ref_length', None),
            feddna_checkpoint=args.feddna_checkpoint,
            consensus_source=getattr(args, 'consensus_source', 'mv'),
        )"""

MAIN_LOOP_NAMESPACE_NEW = """            ref_length=getattr(args, 'ref_length', None),
            feddna_checkpoint=args.feddna_checkpoint,
            consensus_source=getattr(args, 'consensus_source', 'mv'),
            fasta_source=getattr(args, 'fasta_source', 'mv_strict'),
        )"""


def patch_main_loop():
    path = FILES["main_loop"]
    print(f"\n📝 Patch 2/2: {path}")
    backup_v17(path)
    content = read_file(path)

    # 幂等检查
    if "--fasta_source" in content and "fasta_source=getattr(args" in content:
        print("   ⚠️ --fasta_source 已存在, 跳过")
        return

    # (a) argparse
    if "--fasta_source" not in content:
        content = replace_exact(content, MAIN_LOOP_ARGPARSE_OLD,
                                MAIN_LOOP_ARGPARSE_NEW, "main_loop.py (argparse)")
        print("   ✅ argparse 新增 --fasta_source")
    else:
        print("   ⚠️ argparse 已有 --fasta_source, 跳过")

    # (b) Namespace 传递
    if "fasta_source=getattr(args" not in content:
        content = replace_exact(content, MAIN_LOOP_NAMESPACE_OLD,
                                MAIN_LOOP_NAMESPACE_NEW,
                                "main_loop.py (step2_args Namespace)")
        print("   ✅ step2_args Namespace 新增 fasta_source 传递")
    else:
        print("   ⚠️ Namespace 已有 fasta_source, 跳过")

    write_file(path, content)


# ===========================================================================
# 主入口
# ===========================================================================
def main():
    print("=" * 70)
    print("  SSI-EC v17 Patch: FASTA 纯净轨道 (方案 B)")
    print("=" * 70)

    # 文件存在性
    for tag, path in FILES.items():
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            print(f"   请在正确的工作目录下运行, 或 SSIEC_MODELS_DIR=... 指定")
            sys.exit(1)
        print(f"   ✓ 找到: {path}")

    # 前置检查 (v16 必须已打)
    check_v16_applied()

    # 执行 patch
    try:
        patch_step2_runner()
        patch_main_loop()
    except Exception as e:
        print(f"\n❌ Patch 失败: {e}")
        print("   请检查 .bak_v17 备份文件")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  ✅ v17 Patch 完成")
    print("=" * 70)
    print("""
启动 v17 实验 (推荐, 默认启用 mv_strict):

  cd /mnt/st_data/liangxinyi/code/
  python -m models.main_loop \\
      --experiment_dir /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/ \\
      --feddna_checkpoint /mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth \\
      --ref_length 196 \\
      --disable_merge \\
      --consensus_source mv \\
      --fasta_source mv_strict \\
      ...其他参数保持与 v16 一致...

启动后 R1 开始就应立刻看到:
  🧼 [v17] FASTA 纯净轨道: MV on 严格 labels (零 backfill 污染)

对照实验 (退回 v16 FASTA 行为):
  ... --fasta_source fusion_eval ...

判据 (v16 → v17 预期):
  1. R3 EER 应继续下降 (v16 R3=0.00995 → v17 目标 <0.007)
  2. R3 SR 应上升 (v16 R3=81.88% → v17 目标 >85%)
  3. R0→R1 倒退应减轻 (v16 R0=91.07% R1=89.78%, Δ=-1.29; v17 期望 Δ >= -0.5)
  4. Recall 可能微降 (FASTA 不含归巢 backfill 的簇, -1 簇不参与)
""")
    print("回滚命令 (仅撤销 v17 改动, 保留 v16):")
    for tag, path in FILES.items():
        if tag != "step2_decode":  # step2_decode 未动
            print(f"  mv {path}.bak_v17 {path}")


if __name__ == "__main__":
    main()