#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
  SSI-EC v19 Patch: Rebirth — -1 reads 在 Zone I/II 重获 label
===============================================================================

核心发现 (diagnose_negone_zone_fate.py on v18 结果):

  R1→R2: 上轮 -1 reads 15,488 在 R2 的 Zone 分布:
         Zone I:    87  (0.6%)
         Zone II:   6,068 (39.2%)
         Zone III:  9,333 (60.2%)
         → Zone I/II 共 6,155 reads (39.7%), 仍是 -1 的 5,492
  R2→R3: 上轮 -1 reads 31,473 在 R3 的 Zone 分布:
         Zone I:    1,899 (6.0%)
         Zone II:   10,094 (32.1%)
         Zone III:  19,480 (61.9%)
         → Zone I/II 共 11,993 reads (38.1%), 仍是 -1 的 11,939

  关键观察:
  约 40% 的 -1 reads 被 encoder 重新判为 Zone I/II (= 可信), 但由于
  step2_runner 里 new_labels = labels_tensor.clone() (保持原 label),
  它们仍然是 -1. 死数据复活门限太严 (delta=0.16), 只救回 <1%.

  → 11,939 reads 是"隐藏的免费收益". 它们已经通过 Zone 判定 (u_epi 低),
    只是没有标签赋值环节.

v19 改动 (一段代码 + 一个开关):
  step2_runner.py 在 L403 (new_labels = labels_tensor.clone()) 之后,
  L416 (Zone III 隔离) 之前, 插入 Rebirth 逻辑:

    # [v19 Rebirth] -1 + Zone I/II → 最近质心赋 label
    rebirth_mode = getattr(args, 'rebirth_mode', 'nearest')
    if rebirth_mode != 'off':
        reborn_mask = (labels_tensor < 0) & ((zone_ids == 1) | (zone_ids == 2))
        # 对这批 reads 做最近邻匹配, 直接赋 label

  新增参数:
    --rebirth_mode {off, nearest, bounded}   default=nearest
      off     - 禁用 rebirth (退回 v18 行为)
      nearest - 无门限最近邻 (默认, 最激进)
      bounded - 用 delta * REBIRTH_BOUNDED_SCALE (1.5) 作门限

前置条件:
  必须已打 v16 + v17 + v18 patch.

用法:
  cd /mnt/st_data/liangxinyi/code/models/
  python apply_rebirth_v19.py

  启动 v19:
  python main_loop.py ... --consensus_source mv --fasta_source mv_strict \\
                          --zone_include_noise True --rebirth_mode nearest

  对照 (退回 v18):
  python main_loop.py ... --rebirth_mode off

预期效果 (基于诊断):
  - R2 被救回 reads: ~5,492 (v18 的 0.5% 复活率 → v19 接近 100%)
  - R3 被救回 reads: ~11,939
  - Recall 显著回升, SR 更接近 R0 baseline (91.07%) 并有望超过
  - Purity 可能略降 (误合并风险), 但 MV fusion 对少量污染鲁棒
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

def backup_v19(path):
    bak = path + ".bak_v19"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"   💾 备份 (v19): {bak}")
    else:
        print(f"   ⚠️ v19 备份已存在, 跳过: {bak}")

def replace_exact(content, old, new, file_tag):
    count = content.count(old)
    if count == 0:
        raise RuntimeError(
            f"[{file_tag}] 锚点未找到:\n{'-'*70}\n{old[:400]}\n{'-'*70}"
        )
    if count > 1:
        raise RuntimeError(
            f"[{file_tag}] 锚点匹配到 {count} 处, 不唯一:\n"
            f"{'-'*70}\n{old[:400]}\n{'-'*70}"
        )
    return content.replace(old, new, 1)


# ===========================================================================
# 前置检查: v16+v17+v18 patch 必须已打
# ===========================================================================
def check_prereqs():
    print("🔍 前置检查: v16/v17/v18 patch 已打...")
    runner = read_file(FILES["step2_runner"])
    main   = read_file(FILES["main_loop"])

    if "[v16 路径B] 训练靶子切换" not in runner:
        print("❌ v16 未打, 请先跑 apply_mv_target_v16.py"); sys.exit(1)
    if "[v17 FASTA 纯净轨道]" not in runner:
        print("❌ v17 未打, 请先跑 apply_fasta_clean_v17.py"); sys.exit(1)
    if "[v18 Zone 全量判定]" not in runner:
        print("❌ v18 未打, 请先跑 apply_zone_include_noise_v18.py"); sys.exit(1)
    print("   ✓ v16/v17/v18 已就位")


# ===========================================================================
# Patch 1: step2_runner.py — 插入 Rebirth 逻辑
# ---------------------------------------------------------------------------
# 锚点: L416 的 Zone III 隔离代码 (独特, 易匹配)
# 在其之前插入 Rebirth 逻辑
# ===========================================================================
STEP2_RUNNER_OLD = """    z3_count = int((zone_ids == 3).sum().item())
    # [v2-策略三] 记录 Zone III reads 的索引和原始标签，consensus 时临时恢复
    z3_indices = torch.where(zone_ids == 3)[0]
    z3_original_labels = labels_tensor[z3_indices].clone()
    new_labels[zone_ids == 3] = -1
    print(f"   🔒 Zone III 标签隔离: {z3_count} reads → -1 (保留原始标签供 consensus 软参与)")"""

STEP2_RUNNER_NEW = """    # ══════════════════════════════════════════════════════════════════
    # [v19 Rebirth] -1 reads 在 Zone I/II 通过最近邻重获 label
    # ══════════════════════════════════════════════════════════════════
    # 背景: v18 让 -1 reads 参与 Zone 判定, 诊断显示 ~40% 被判为 Zone I/II.
    # 但 new_labels = labels_tensor.clone() 只是保持原 label, 所以这些 reads
    # 仍然是 -1. 死数据复活门限太严 (delta=0.16), 只救回 <1%.
    # v19 在 Zone III 隔离之前, 给这批"已被 encoder 重新信任"的 -1 reads
    # 一次最近邻赋 label 的机会.
    _rebirth_mode = getattr(args, 'rebirth_mode', 'nearest')
    if _rebirth_mode != 'off' and len(centroids) > 0:
        reborn_mask = (labels_tensor < 0) & ((zone_ids == 1) | (zone_ids == 2))
        n_candidates = int(reborn_mask.sum().item())
        if n_candidates > 0:
            reborn_idx = torch.where(reborn_mask)[0]
            _cids_sorted = sorted(centroids.keys())
            _centroid_mat = torch.stack([centroids[c] for c in _cids_sorted]).cpu()

            # 分块算距离 (防 OOM)
            n_reborn_success = 0
            _chunk = 5000
            for _s in range(0, len(reborn_idx), _chunk):
                _e = min(_s + _chunk, len(reborn_idx))
                _batch_idx = reborn_idx[_s:_e]
                _batch_emb = embeddings_f32[_batch_idx]
                _dists = torch.cdist(_batch_emb, _centroid_mat)
                _min_d, _min_id = _dists.min(dim=1)

                if _rebirth_mode == 'bounded':
                    REBIRTH_BOUNDED_SCALE = 1.5
                    _hit = _min_d < delta * REBIRTH_BOUNDED_SCALE
                else:
                    # 'nearest' - 无门限, 无条件赋最近质心
                    _hit = torch.ones_like(_min_d, dtype=torch.bool)

                _positions = _batch_idx[_hit]
                _cluster_ids = torch.tensor(
                    [_cids_sorted[j] for j in _min_id[_hit].tolist()],
                    dtype=torch.long
                )
                new_labels[_positions] = _cluster_ids
                n_reborn_success += int(_hit.sum().item())

            print(f"   🌱 [v19 Rebirth] -1 reads 在 Zone I/II 重生 "
                  f"(mode={_rebirth_mode}): "
                  f"{n_reborn_success:,}/{n_candidates:,} 成功")
            del _centroid_mat
        else:
            print(f"   🌱 [v19 Rebirth] 无候选 (label=-1 且在 Zone I/II 的 reads: 0)")
    elif _rebirth_mode == 'off':
        print(f"   🌱 [v19 Rebirth] 已禁用 (--rebirth_mode off, 退回 v18 行为)")

    z3_count = int((zone_ids == 3).sum().item())
    # [v2-策略三] 记录 Zone III reads 的索引和原始标签，consensus 时临时恢复
    z3_indices = torch.where(zone_ids == 3)[0]
    z3_original_labels = labels_tensor[z3_indices].clone()
    new_labels[zone_ids == 3] = -1
    print(f"   🔒 Zone III 标签隔离: {z3_count} reads → -1 (保留原始标签供 consensus 软参与)")"""


def patch_step2_runner():
    path = FILES["step2_runner"]
    print(f"\n📝 Patch 1/2: {path}")
    backup_v19(path)
    content = read_file(path)

    if "[v19 Rebirth]" in content:
        print("   ⚠️ v19 已打, 跳过")
        return

    new_content = replace_exact(content, STEP2_RUNNER_OLD,
                                STEP2_RUNNER_NEW, "step2_runner.py")
    write_file(path, new_content)
    print("   ✅ Zone III 隔离之前插入 Rebirth 逻辑")


# ===========================================================================
# Patch 2: main_loop.py — argparse + Namespace
# ===========================================================================
MAIN_LOOP_ARGPARSE_OLD = """    parser.add_argument('--zone_include_noise', type=lambda x: str(x).lower() == 'true',
                        default=True,
                        help='[v18 Zone 全量判定] 让 label=-1 reads 也参与 Zone 判定. '
                             'True (默认) = 全量 (打破 -1 单向流失), '
                             'False = 仅 label>=0 (v17 行为, 用于对照).')
    args = parser.parse_args()"""

MAIN_LOOP_ARGPARSE_NEW = """    parser.add_argument('--zone_include_noise', type=lambda x: str(x).lower() == 'true',
                        default=True,
                        help='[v18 Zone 全量判定] 让 label=-1 reads 也参与 Zone 判定. '
                             'True (默认) = 全量 (打破 -1 单向流失), '
                             'False = 仅 label>=0 (v17 行为, 用于对照).')
    parser.add_argument('--rebirth_mode', type=str, default='nearest',
                        choices=['off', 'nearest', 'bounded'],
                        help='[v19 Rebirth] -1 reads 在 Zone I/II 重获 label 的方式. '
                             'nearest (默认) = 无门限最近邻, '
                             'bounded = 用 delta*1.5 作门限, '
                             'off = 禁用 (v18 行为, 用于对照).')
    args = parser.parse_args()"""


MAIN_LOOP_NAMESPACE_OLD = """            ref_length=getattr(args, 'ref_length', None),
            feddna_checkpoint=args.feddna_checkpoint,
            consensus_source=getattr(args, 'consensus_source', 'mv'),
            fasta_source=getattr(args, 'fasta_source', 'mv_strict'),
            zone_include_noise=getattr(args, 'zone_include_noise', True),
        )"""

MAIN_LOOP_NAMESPACE_NEW = """            ref_length=getattr(args, 'ref_length', None),
            feddna_checkpoint=args.feddna_checkpoint,
            consensus_source=getattr(args, 'consensus_source', 'mv'),
            fasta_source=getattr(args, 'fasta_source', 'mv_strict'),
            zone_include_noise=getattr(args, 'zone_include_noise', True),
            rebirth_mode=getattr(args, 'rebirth_mode', 'nearest'),
        )"""


def patch_main_loop():
    path = FILES["main_loop"]
    print(f"\n📝 Patch 2/2: {path}")
    backup_v19(path)
    content = read_file(path)

    if "--rebirth_mode" in content and "rebirth_mode=getattr(args" in content:
        print("   ⚠️ v19 已打, 跳过")
        return

    if "--rebirth_mode" not in content:
        content = replace_exact(content, MAIN_LOOP_ARGPARSE_OLD,
                                MAIN_LOOP_ARGPARSE_NEW,
                                "main_loop.py (argparse)")
        print("   ✅ argparse 新增 --rebirth_mode")
    if "rebirth_mode=getattr(args" not in content:
        content = replace_exact(content, MAIN_LOOP_NAMESPACE_OLD,
                                MAIN_LOOP_NAMESPACE_NEW,
                                "main_loop.py (Namespace)")
        print("   ✅ Namespace 新增 rebirth_mode")
    write_file(path, content)


# ===========================================================================
# 主入口
# ===========================================================================
def main():
    print("=" * 70)
    print("  SSI-EC v19 Patch: Rebirth (-1 reads 在 Zone I/II 重获 label)")
    print("=" * 70)
    for tag, path in FILES.items():
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            sys.exit(1)
        print(f"   ✓ 找到: {path}")

    check_prereqs()

    try:
        patch_step2_runner()
        patch_main_loop()
    except Exception as e:
        print(f"\n❌ Patch 失败: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  ✅ v19 Patch 完成")
    print("=" * 70)
    print("""
启动 v19 实验 (推荐, 默认 rebirth_mode=nearest):

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
      --rebirth_mode nearest \\
      2>&1 | tee seq1d_thin_v19.log

R2 开始后立刻校验 (最关键!):
  grep "\\[v19 Rebirth\\]" seq1d_thin_v19.log
  期望看到 (基于 v18 诊断):
    R1: 🌱 [v19 Rebirth] -1 reads 在 Zone I/II 重生 (mode=nearest): 0/0 (R1 无前置 -1)
    R2: 🌱 [v19 Rebirth] ... 成功 ~6,000/6,000   (v18 诊断 R2 里有 6,155 候选)
    R3: 🌱 [v19 Rebirth] ... 成功 ~12,000/12,000 (v18 诊断 R3 里有 11,993 候选)

判据 (v18 → v19 预期):
  1. R2/R3 被救回的 reads 数比 v18 大幅增加
  2. R3 Recall 88.72%(v17) 90.65%(v18) → ≥93% (v19 目标)
  3. R3 SR     85.80%(v17) 87.98%(v18) → ≥91% (接近/超过 R0 Clover baseline)
  4. R3 存盘 -1 数比 v18 明显下降
  5. Purity 可能微降 1-2pp (误合并风险), 但 MV fusion 对少量污染鲁棒
     如果 Purity 崩到 <90%, 说明 nearest 太激进, 切 --rebirth_mode bounded

对照实验:
  --rebirth_mode off      → 退回 v18 行为
  --rebirth_mode bounded  → 更保守, 用 delta*1.5 作门限
""")


if __name__ == "__main__":
    main()