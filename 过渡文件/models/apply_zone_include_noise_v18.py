#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
  SSI-EC v18 Patch: Zone 判定全量参与 (根治 -1 单向流失)
===============================================================================

核心发现 (diagnose_zone3_tracking.py):
  - R1∩R2 Jaccard=0.001, R1∩R3 Jaccard=0.003
  - 每轮 Zone III 96%+ 是新人, 上轮 Zone III 几乎全换
  - 但 label=-1 存盘数 15k→30k→47k 单调累积

根因 (step2_refine.py L219):
  split_confidence_by_zone:
      valid = (labels >= 0)         ← -1 reads 被排除在 Zone 判定外
      ale_valid = u_ale[valid]      ← 只在 label≥0 上算阈值
      zone_ids[z3_mask & valid] = 3 ← 只给 label≥0 的 reads 打 Zone 标签

  流程:
    R1 推理 330788 → Zone 判定 330788 (都是 label≥0) → 隔离 16558 个 Zone III
    R2 推理 330788 (含 -1) → Zone 判定只看 315300 (正常) → 隔离 16969 个新 Zone III
        R1 遗留的 15488 个 -1 reads 根本不参与 R2 Zone 判定!!
        → 它们既没机会重回 Zone I/II, 只能靠 "死数据复活" (6% 救回率)
    R3 同样累积

  结果: 每轮 5% 的 reads 单向流失到 -1, 雪球滚到 R3 时累积 14% (47389/330788).
        这就是 Recall 94%→91%→88% 的根因.

v18 改动 (一行 + 开关):
  step2_refine.py L206 函数签名:
    def split_confidence_by_zone(u_epi, u_ale, labels):
  改成 (新增可选参数 include_noise):
    def split_confidence_by_zone(u_epi, u_ale, labels, include_noise=False):

  L219 的 valid:
    if include_noise:
        valid = torch.ones(N, dtype=torch.bool, device=device)
    else:
        valid = (labels >= 0)

  step2_runner.py L297 调用点:
    zone_ids, zone_stats = split_confidence_by_zone(u_epi, u_ale, labels_tensor)
  改成:
    zone_ids, zone_stats = split_confidence_by_zone(
        u_epi, u_ale, labels_tensor,
        include_noise=getattr(args, 'zone_include_noise', True))

  main_loop.py 新增:
    --zone_include_noise {True, False}   default=True

  默认 True (v18 行为, 全量参与 Zone 判定).
  False 退回 v17 行为做 A/B 对照.

预期效果:
  - -1 reads 现在参与 Zone 判定: 大部分会被判为 Zone II (甚至 Zone I)
  - Zone I reads 被赋予大簇标签 → -1 累积打破
  - R2/R3 Zone III 可能微涨 (含 -1 reads 的 u_ale 长尾), 但 -1 总量应递减
  - Recall 预期大幅回升

用法:
  cd /mnt/st_data/liangxinyi/code/models/
  python apply_zone_include_noise_v18.py

  启动 v18:
  python main_loop.py ... --consensus_source mv --fasta_source mv_strict \\
                          --zone_include_noise True

  回退 v17 (对照):
  python main_loop.py ... --zone_include_noise False

前置条件:
  必须已打 v16 + v17 patch.
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
    if os.path.isdir("models") and os.path.isfile("models/step2_refine.py"):
        return "models"
    if os.path.isfile("step2_refine.py") and os.path.isfile("step2_runner.py"):
        return "."
    return "models"


MODELS_DIR = _resolve_models_dir()
FILES = {
    "step2_refine": os.path.join(MODELS_DIR, "step2_refine.py"),
    "step2_runner": os.path.join(MODELS_DIR, "step2_runner.py"),
    "main_loop":    os.path.join(MODELS_DIR, "main_loop.py"),
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

def backup_v18(path):
    bak = path + ".bak_v18"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"   💾 备份 (v18): {bak}")
    else:
        print(f"   ⚠️ v18 备份已存在, 跳过: {bak}")

def replace_exact(content, old, new, file_tag):
    count = content.count(old)
    if count == 0:
        raise RuntimeError(
            f"[{file_tag}] 锚点未找到 (可能 v16/v17 未打或源文件变动):\n"
            f"{'-'*70}\n{old[:400]}\n{'-'*70}"
        )
    if count > 1:
        raise RuntimeError(
            f"[{file_tag}] 锚点匹配到 {count} 处, 不唯一:\n"
            f"{'-'*70}\n{old[:400]}\n{'-'*70}"
        )
    return content.replace(old, new, 1)


# ===========================================================================
# 前置检查
# ===========================================================================
def check_prereqs():
    print("🔍 前置检查: v16/v17 patch 已打, 未跑过 v18...")
    runner = read_file(FILES["step2_runner"])
    refine = read_file(FILES["step2_refine"])
    main   = read_file(FILES["main_loop"])

    if "[v16 路径B] 训练靶子切换" not in runner:
        print("❌ v16 未打, 请先跑 apply_mv_target_v16.py"); sys.exit(1)
    if "[v17 FASTA 纯净轨道]" not in runner:
        print("❌ v17 未打, 请先跑 apply_fasta_clean_v17.py"); sys.exit(1)
    print("   ✓ v16/v17 已就位")

    # v18 幂等检查
    if "include_noise=" in refine or "[v18 Zone 全量判定]" in refine:
        print("   ⚠️ step2_refine.py 的 v18 patch 已打过, 会跳过")
    if "--zone_include_noise" in main:
        print("   ⚠️ main_loop.py 的 v18 patch 已打过, 会跳过")


# ===========================================================================
# Patch 1: step2_refine.py — 函数签名 + valid 改动
# ===========================================================================
STEP2_REFINE_SIG_OLD = """def split_confidence_by_zone(u_epi, u_ale, labels):
    \"\"\"
    自适应三区制划分:
      第一刀: CDF 拐点 (Kneedle) → Zone III (U_ale 长尾噪声)
      第二刀: K=2 GMM            → Zone I / Zone II (U_epi 认知边界)

    [FIX-Bug#2] 新增安全阀: Zone III 实际比例超过 30% 时强制回退
    [FIX-Bug#6] 打印实际比例而非硬编码 \"≈10%\"
    \"\"\"
    N      = len(labels)
    device = labels.device
    zone_ids = torch.zeros(N, dtype=torch.long, device=device)

    valid   = (labels >= 0)
    n_valid = valid.sum().item()"""

STEP2_REFINE_SIG_NEW = """def split_confidence_by_zone(u_epi, u_ale, labels, include_noise=False):
    \"\"\"
    自适应三区制划分:
      第一刀: CDF 拐点 (Kneedle) → Zone III (U_ale 长尾噪声)
      第二刀: K=2 GMM            → Zone I / Zone II (U_epi 认知边界)

    [FIX-Bug#2] 新增安全阀: Zone III 实际比例超过 30% 时强制回退
    [FIX-Bug#6] 打印实际比例而非硬编码 \"≈10%\"

    [v18 Zone 全量判定]
      include_noise=False (默认, v17 及以前行为):
          Zone 判定只看 labels >= 0 的 reads.
          问题: -1 reads 被排除, 一旦标 -1 就失去被重新评估的机会,
                导致每轮 5% 的 reads 单向流失到 -1, 累积到 R3 时占 14%.
      include_noise=True (v18 行为):
          labels=-1 的 reads 也参与 Zone 判定. 它们的 u_ale 可能不长尾
          (不是 Zone III), 应该分到 Zone I/II 并通过质心匹配重获 label.
    \"\"\"
    N      = len(labels)
    device = labels.device
    zone_ids = torch.zeros(N, dtype=torch.long, device=device)

    if include_noise:
        # [v18] 全量参与 Zone 判定, -1 reads 也能被重新审判
        valid = torch.ones(N, dtype=torch.bool, device=device)
    else:
        # [v17] 只看正常 label reads (会导致 -1 累积)
        valid = (labels >= 0)
    n_valid = valid.sum().item()"""


def patch_step2_refine():
    path = FILES["step2_refine"]
    print(f"\n📝 Patch 1/3: {path}")
    backup_v18(path)
    content = read_file(path)

    if "[v18 Zone 全量判定]" in content:
        print("   ⚠️ v18 已打, 跳过")
        return

    new_content = replace_exact(content, STEP2_REFINE_SIG_OLD,
                                STEP2_REFINE_SIG_NEW, "step2_refine.py")
    write_file(path, new_content)
    print("   ✅ split_confidence_by_zone 加 include_noise 参数")


# ===========================================================================
# Patch 2: step2_runner.py — 调用点传参
# ===========================================================================
STEP2_RUNNER_CALL_OLD = """    zone_ids, zone_stats = split_confidence_by_zone(u_epi, u_ale, labels_tensor)
    _np_zone_ids = zone_ids.numpy().copy()"""

STEP2_RUNNER_CALL_NEW = """    # ══════════════════════════════════════════════════════════════
    # [v18 Zone 全量判定] 让 -1 reads 也参与 Zone 判定, 打破 -1 累积
    # ══════════════════════════════════════════════════════════════
    _zone_include_noise = getattr(args, 'zone_include_noise', True)
    if _zone_include_noise:
        _n_negone_pre = int((labels_tensor < 0).sum())
        print(f"   🔄 [v18] Zone 全量判定: {_n_negone_pre} 个 -1 reads 参与本轮 Zone 审判")
    zone_ids, zone_stats = split_confidence_by_zone(
        u_epi, u_ale, labels_tensor,
        include_noise=_zone_include_noise,
    )
    _np_zone_ids = zone_ids.numpy().copy()"""


def patch_step2_runner():
    path = FILES["step2_runner"]
    print(f"\n📝 Patch 2/3: {path}")
    backup_v18(path)
    content = read_file(path)

    if "[v18 Zone 全量判定]" in content:
        print("   ⚠️ v18 已打, 跳过")
        return

    new_content = replace_exact(content, STEP2_RUNNER_CALL_OLD,
                                STEP2_RUNNER_CALL_NEW, "step2_runner.py")
    write_file(path, new_content)
    print("   ✅ split_confidence_by_zone 调用点传 include_noise")


# ===========================================================================
# Patch 3: main_loop.py — argparse + Namespace
# ===========================================================================
MAIN_LOOP_ARGPARSE_OLD = """    parser.add_argument('--fasta_source', type=str, default='mv_strict',
                        choices=['mv_strict', 'fusion_eval'],
                        help='[v17 FASTA 纯净轨道] 导出 FASTA 的 consensus 来源. '
                             'mv_strict (默认) = MV on 严格 labels (零 backfill 污染), '
                             'fusion_eval = fusion on 归巢 labels (v16 行为, 用于对照).')
    args = parser.parse_args()"""

MAIN_LOOP_ARGPARSE_NEW = """    parser.add_argument('--fasta_source', type=str, default='mv_strict',
                        choices=['mv_strict', 'fusion_eval'],
                        help='[v17 FASTA 纯净轨道] 导出 FASTA 的 consensus 来源. '
                             'mv_strict (默认) = MV on 严格 labels (零 backfill 污染), '
                             'fusion_eval = fusion on 归巢 labels (v16 行为, 用于对照).')
    parser.add_argument('--zone_include_noise', type=lambda x: str(x).lower() == 'true',
                        default=True,
                        help='[v18 Zone 全量判定] 让 label=-1 reads 也参与 Zone 判定. '
                             'True (默认) = 全量 (打破 -1 单向流失), '
                             'False = 仅 label>=0 (v17 行为, 用于对照).')
    args = parser.parse_args()"""


MAIN_LOOP_NAMESPACE_OLD = """            ref_length=getattr(args, 'ref_length', None),
            feddna_checkpoint=args.feddna_checkpoint,
            consensus_source=getattr(args, 'consensus_source', 'mv'),
            fasta_source=getattr(args, 'fasta_source', 'mv_strict'),
        )"""

MAIN_LOOP_NAMESPACE_NEW = """            ref_length=getattr(args, 'ref_length', None),
            feddna_checkpoint=args.feddna_checkpoint,
            consensus_source=getattr(args, 'consensus_source', 'mv'),
            fasta_source=getattr(args, 'fasta_source', 'mv_strict'),
            zone_include_noise=getattr(args, 'zone_include_noise', True),
        )"""


def patch_main_loop():
    path = FILES["main_loop"]
    print(f"\n📝 Patch 3/3: {path}")
    backup_v18(path)
    content = read_file(path)

    if "--zone_include_noise" in content and "zone_include_noise=getattr(args" in content:
        print("   ⚠️ v18 已打, 跳过")
        return

    if "--zone_include_noise" not in content:
        content = replace_exact(content, MAIN_LOOP_ARGPARSE_OLD,
                                MAIN_LOOP_ARGPARSE_NEW,
                                "main_loop.py (argparse)")
        print("   ✅ argparse 新增 --zone_include_noise")
    if "zone_include_noise=getattr(args" not in content:
        content = replace_exact(content, MAIN_LOOP_NAMESPACE_OLD,
                                MAIN_LOOP_NAMESPACE_NEW,
                                "main_loop.py (Namespace)")
        print("   ✅ step2_args Namespace 新增 zone_include_noise")
    write_file(path, content)


# ===========================================================================
# 主入口
# ===========================================================================
def main():
    print("=" * 70)
    print("  SSI-EC v18 Patch: Zone 判定全量参与")
    print("=" * 70)
    for tag, path in FILES.items():
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            sys.exit(1)
        print(f"   ✓ 找到: {path}")

    check_prereqs()

    try:
        patch_step2_refine()
        patch_step2_runner()
        patch_main_loop()
    except Exception as e:
        print(f"\n❌ Patch 失败: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  ✅ v18 Patch 完成")
    print("=" * 70)
    print("""
启动 v18 实验 (推荐, 默认启用 --zone_include_noise True):

  cd /mnt/st_data/liangxinyi/code/
  python main_loop.py \\
      --experiment_dir /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/ \\
      --feddna_checkpoint /mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth \\
      --gt_tags_file /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/seq1d_tags_reads.txt \\
      --gt_refs_file /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/seq1d_refs.txt \\
      --max_iterations 3 \\
      --max_length 201 \\
      --target_clusters 11736 \\
      --cl_mode ours \\
      --ref_length 196 \\
      --primer_prefix 20 \\
      --primer_suffix 20 \\
      --disable_merge \\
      --consensus_source mv \\
      --fasta_source mv_strict \\
      --zone_include_noise True \\
      2>&1 | tee seq1d_thin_v18.log

R2 开始后立刻校验 (关键!):
  grep "\\[v18\\]" seq1d_thin_v18.log
  期望看到: 🔄 [v18] Zone 全量判定: XXXXX 个 -1 reads 参与本轮 Zone 审判
  (R2 里这个数应当约等于 R1 结束时存盘的 -1 reads 数, 约 15k)

判据 (v17 → v18 预期):
  1. R2 存盘 -1 数应当明显低于 v17 的 30,610
     (-1 reads 参与 Zone 后, 大部分会升级为 Zone II 并获质心标签)
  2. R3 存盘 -1 数应当稳定或递减, 而不是 v17 的 47,389
  3. R3 Recall  88.72% → ≥ 93%  (主要效果)
  4. R3 SR     85.80% → ≥ 90%  (受 Recall 带动)
  5. R2/R3 Zone III 可能微涨 (-1 reads 里的真脏 reads 进入 Zone III)

对照实验:
  --zone_include_noise False → 完全复现 v17 (应完全一致 SR=85.80%)
""")


if __name__ == "__main__":
    main()