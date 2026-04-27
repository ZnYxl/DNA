#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
  SSI-EC v16 Patch: MV Target (路径 B)
===============================================================================

改动目标:
  把 Round 2+ 的 Step1 训练靶子 (consensus_dict) 从
      "evidence fusion on strict labels"
  换成
      "majority vote on strict labels"
  打破 encoder 自污染闭环: MV 是硬投票, 与 encoder 状态无关,
  即使 encoder 学坏, target 也是干净的.

改动三个文件:
  1. models/step2_decode.py
       新增函数: compute_mv_consensus
         - 纯统计操作, 不跑 encoder/decoder
         - 输出格式与 run_feddna_decode 的 consensus_dict 完全一致
           (Dict[int, Tensor(L, 4)] one-hot, 与 FedDNA ds_fusion 输出对齐)
         - 投票门限 has_vote >= 50% * N, 与 ds_fusion_masked 保持一致

  2. models/step2_runner.py
       L589: consensus_dict = consensus_dict_for_train
       改成: 依据 args.consensus_source 切换 mv / fusion
       FASTA 轨道 (consensus_dict_for_eval) 完全不动, 确保 v15→v16 只有
       "训练 target" 这一个变量改动, 因果辨识干净.

  3. models/main_loop.py
       新增 argparse: --consensus_source {mv, fusion}   default=mv
       新增传递:       step2_args.consensus_source = args.consensus_source

用法:
  cd /mnt/st_data/liangxinyi/code/
  python apply_mv_target_v16.py
  # 会在每个被修改文件旁生成 xxx.py.bak

  回滚:
  mv models/step2_decode.py.bak models/step2_decode.py
  mv models/step2_runner.py.bak models/step2_runner.py
  mv models/main_loop.py.bak   models/main_loop.py

  启动 v16:
  python main_loop.py ... --consensus_source mv --ref_length 196 --disable_merge
"""
import os
import sys
import shutil

# ===========================================================================
# 配置
# ===========================================================================
def _resolve_models_dir():
    """自动探测 models 目录 (优先级: 环境变量 > ./models/ > . )"""
    env = os.environ.get("SSIEC_MODELS_DIR")
    if env and env.strip():
        return env.strip()
    # 若当前目录下存在 models/ 子目录, 用 models/
    if os.path.isdir("models") and os.path.isfile("models/step2_decode.py"):
        return "models"
    # 若脚本就放在 models/ 里 (当前目录能找到三个源文件), 用 .
    if os.path.isfile("step2_decode.py") and os.path.isfile("step2_runner.py") \
       and os.path.isfile("main_loop.py"):
        return "."
    # 默认回退
    return "models"


MODELS_DIR = _resolve_models_dir()
FILES = {
    "step2_decode": os.path.join(MODELS_DIR, "step2_decode.py"),
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

def backup(path):
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"   💾 备份: {bak}")
    else:
        print(f"   ⚠️ 备份已存在, 跳过: {bak}")

def replace_exact(content, old, new, file_tag):
    count = content.count(old)
    if count == 0:
        raise RuntimeError(
            f"[{file_tag}] 锚点未找到, 可能源文件已经被修改过或版本不同:\n"
            f"{'-'*70}\n{old[:300]}...\n{'-'*70}"
        )
    if count > 1:
        raise RuntimeError(
            f"[{file_tag}] 锚点匹配到 {count} 处, 不唯一, 请检查源文件:\n"
            f"{'-'*70}\n{old[:300]}...\n{'-'*70}"
        )
    return content.replace(old, new, 1)


# ===========================================================================
# Patch 1: step2_decode.py - 追加 compute_mv_consensus
# ===========================================================================
MV_FUNCTION_CODE = '''
# ---------------------------------------------------------------------------
# [v16 路径B] MV Consensus: pure majority vote, 不跑 encoder
# ---------------------------------------------------------------------------
def compute_mv_consensus(
    data_loader,
    new_labels_np,
    flat_real_indices,
    model_max_len: int,
    ref_length: int = None,
) -> Dict[int, torch.Tensor]:
    """
    Pure majority-vote consensus, 用于 Round 2+ 的 Step1 训练靶子.

    与 run_feddna_decode 的区别:
      - 不跑 encoder/decoder, 纯统计投票
      - 不受 encoder 状态影响 -> 打破 encoder 自污染闭环
      - 输出格式完全一致 (Dict[int, Tensor(L, 4)] one-hot), step1_train.py 零改动

    逻辑:
      对每个簇 (只处理 label >= 0 的 reads):
        counts[L, 4] = sum over reads of one_hot(seq)
        has_vote[L]  = (有效 read 数 >= 50% * N)      # 与 ds_fusion_masked 一致
        indices[L]   = counts.argmax(dim=-1)
        one_hot[L,4] = F.one_hot(indices, 4); one_hot[~has_vote] = 0

    Args:
        data_loader:       CloverDataLoader, 提供 data_loader.reads[real_idx]
        new_labels_np:     严格版 labels (numpy, -1 会被自动跳过)
        flat_real_indices: data_loader.reads 的真实索引映射
        model_max_len:     序列 one-hot 长度 (与 fusion 版对齐, 通常 201)
        ref_length:        先验长度, 仅用于日志提示, 实际形状仍为 model_max_len
                           (与 run_feddna_decode 的 consensus_dict 对齐)

    Returns:
        consensus_dict: {cluster_id: Tensor(model_max_len, 4)} one-hot
    """
    from models.step1_data import seq_to_onehot

    # 1. cluster_id -> real_idx 列表 (只收集 label >= 0)
    cluster_to_ridx: Dict[int, list] = defaultdict(list)
    for didx, label in enumerate(new_labels_np):
        if label >= 0:
            real_idx = flat_real_indices[didx]
            cluster_to_ridx[int(label)].append(real_idx)

    print(f"\\n🗳️  [v16 路径B] MV Consensus: {len(cluster_to_ridx)} 个簇")
    if ref_length is not None:
        print(f"   📏 ref_length={ref_length} (one-hot 形状仍为 L={model_max_len}, "
              f"截断在 save_consensus_fasta 处理)")

    consensus_dict: Dict[int, torch.Tensor] = {}
    skipped = 0

    # 2. 逐簇统计投票
    for cluster_id, ridx_list in cluster_to_ridx.items():
        n_reads = len(ridx_list)
        if n_reads < 1:
            skipped += 1
            continue

        # 收集 one-hot encoding + padding mask
        encodings = []
        padding_masks = []
        for ridx in ridx_list:
            seq = data_loader.reads[ridx]
            enc = seq_to_onehot(seq, model_max_len)   # (L, 4)
            encodings.append(enc)
            pmask = enc.sum(dim=-1) > 0               # (L,) bool
            padding_masks.append(pmask)

        enc_tensor   = torch.stack(encodings)          # (N, L, 4)
        pmask_tensor = torch.stack(padding_masks)      # (N, L)

        # 3. 统计投票
        counts   = enc_tensor.sum(dim=0)               # (L, 4)  每位置4碱基计数
        n_valid  = pmask_tensor.sum(dim=0).float()     # (L,)    每位置有效 read 数
        # has_vote: 投票门限与 ds_fusion_masked 一致 (防 1 条 insertion read 拉长)
        has_vote = (n_valid >= max(n_reads * 0.5, 1))  # (L,) bool

        # 4. argmax -> one_hot, padding 位置清零
        indices = counts.argmax(dim=-1)                # (L,)
        one_hot = F.one_hot(indices, num_classes=4).float()  # (L, 4)
        one_hot[~has_vote] = 0.0

        consensus_dict[cluster_id] = one_hot

        # 释放本簇的中间 tensor
        del enc_tensor, pmask_tensor, counts, n_valid, has_vote, indices, one_hot

    print(f"   ✅ 生成 {len(consensus_dict)} 个簇的 MV consensus "
          f"(跳过 {skipped} 个空簇)")
    return consensus_dict


'''

STEP2_DECODE_ANCHOR_OLD = """# ---------------------------------------------------------------------------
# 工具: 将 consensus_dict 保存为 FASTA
# ---------------------------------------------------------------------------
def save_consensus_fasta("""

STEP2_DECODE_ANCHOR_NEW = MV_FUNCTION_CODE + STEP2_DECODE_ANCHOR_OLD


def patch_step2_decode():
    path = FILES["step2_decode"]
    print(f"\n📝 Patch 1/3: {path}")
    backup(path)
    content = read_file(path)

    # 幂等检查
    if "def compute_mv_consensus(" in content:
        print(f"   ⚠️ compute_mv_consensus 已存在, 跳过")
        return

    new_content = replace_exact(
        content, STEP2_DECODE_ANCHOR_OLD, STEP2_DECODE_ANCHOR_NEW,
        "step2_decode.py"
    )
    write_file(path, new_content)
    print(f"   ✅ 新增函数 compute_mv_consensus")


# ===========================================================================
# Patch 2: step2_runner.py - 替换 consensus_dict 赋值
# ===========================================================================
STEP2_RUNNER_ANCHOR_OLD = """    # 后续 consensus_dict 指向训练版 (.pt 保存)
    consensus_dict = consensus_dict_for_train"""

STEP2_RUNNER_ANCHOR_NEW = """    # ══════════════════════════════════════════════════════════════
    # [v16 路径B] 训练靶子切换: fusion vs MV
    # ══════════════════════════════════════════════════════════════
    # G老师钦点改动: 把 R2+ 的 consensus_dict (训练靶子) 从 evidence
    # fusion 换成 majority vote, 打破 encoder 自污染闭环.
    # FASTA 轨道 (consensus_dict_for_eval) 完全不动, 确保 v15→v16 只有
    # "训练 target" 这一个变量改动, 因果辨识干净.
    _consensus_source = getattr(args, 'consensus_source', 'mv')
    if _consensus_source == 'mv':
        print(f"\\n   🗳️  [v16 路径B] 训练靶子: MV consensus "
              f"(打破 encoder 自污染闭环)")
        from models.step2_decode import compute_mv_consensus
        consensus_dict = compute_mv_consensus(
            data_loader=data_loader,
            new_labels_np=new_labels_np,
            flat_real_indices=flat_real_indices,
            model_max_len=model_max_len,
            ref_length=getattr(args, 'ref_length', None),
        )
    else:
        print(f"\\n   🧬 [v15 fallback] 训练靶子: Evidence fusion consensus")
        consensus_dict = consensus_dict_for_train"""


def patch_step2_runner():
    path = FILES["step2_runner"]
    print(f"\n📝 Patch 2/3: {path}")
    backup(path)
    content = read_file(path)

    # 幂等检查
    if "[v16 路径B] 训练靶子切换" in content:
        print(f"   ⚠️ v16 训练靶子切换已存在, 跳过")
        return

    new_content = replace_exact(
        content, STEP2_RUNNER_ANCHOR_OLD, STEP2_RUNNER_ANCHOR_NEW,
        "step2_runner.py"
    )
    write_file(path, new_content)
    print(f"   ✅ consensus_dict 赋值换成 mv/fusion 分支")


# ===========================================================================
# Patch 3: main_loop.py - 新增 argparse + 传递
# ===========================================================================
MAIN_LOOP_ARGPARSE_OLD = """    parser.add_argument('--freeze_consensus', action='store_true', default=False,
                        help='[实验2] 所有轮次的 Step1 训练目标始终用 ref.txt，'
                             '不用上一轮 Step2 产出的 consensus。用于诊断 B 层毒化。')
    args = parser.parse_args()"""

MAIN_LOOP_ARGPARSE_NEW = """    parser.add_argument('--freeze_consensus', action='store_true', default=False,
                        help='[实验2] 所有轮次的 Step1 训练目标始终用 ref.txt，'
                             '不用上一轮 Step2 产出的 consensus。用于诊断 B 层毒化。')
    parser.add_argument('--consensus_source', type=str, default='mv',
                        choices=['mv', 'fusion'],
                        help='[v16 路径B] Round 2+ 训练靶子来源. '
                             'mv (默认) = majority vote (打破 encoder 自污染), '
                             'fusion = evidence fusion (v15 行为, 用于对照).')
    args = parser.parse_args()"""


MAIN_LOOP_NAMESPACE_OLD = """            ref_length=getattr(args, 'ref_length', None),
            feddna_checkpoint=args.feddna_checkpoint,
        )"""

MAIN_LOOP_NAMESPACE_NEW = """            ref_length=getattr(args, 'ref_length', None),
            feddna_checkpoint=args.feddna_checkpoint,
            consensus_source=getattr(args, 'consensus_source', 'mv'),
        )"""


def patch_main_loop():
    path = FILES["main_loop"]
    print(f"\n📝 Patch 3/3: {path}")
    backup(path)
    content = read_file(path)

    # 幂等检查
    if "--consensus_source" in content and "consensus_source=getattr(args" in content:
        print(f"   ⚠️ --consensus_source 已存在, 跳过")
        return

    # (a) argparse
    if "--consensus_source" not in content:
        content = replace_exact(
            content, MAIN_LOOP_ARGPARSE_OLD, MAIN_LOOP_ARGPARSE_NEW,
            "main_loop.py (argparse)"
        )
        print(f"   ✅ argparse 新增 --consensus_source")
    else:
        print(f"   ⚠️ argparse 部分已有 --consensus_source, 跳过")

    # (b) Namespace 传递
    if "consensus_source=getattr(args" not in content:
        content = replace_exact(
            content, MAIN_LOOP_NAMESPACE_OLD, MAIN_LOOP_NAMESPACE_NEW,
            "main_loop.py (step2_args Namespace)"
        )
        print(f"   ✅ step2_args Namespace 新增 consensus_source 传递")
    else:
        print(f"   ⚠️ Namespace 部分已有 consensus_source, 跳过")

    write_file(path, content)


# ===========================================================================
# 主入口
# ===========================================================================
def main():
    print("=" * 70)
    print("  SSI-EC v16 Patch: MV Target (路径 B)")
    print("=" * 70)

    # 检查文件存在
    for tag, path in FILES.items():
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            print(f"   请在正确的工作目录下运行, 或设置环境变量:")
            print(f"   SSIEC_MODELS_DIR=/path/to/models python apply_mv_target_v16.py")
            sys.exit(1)
        print(f"   ✓ 找到: {path}")

    # 执行 patch
    try:
        patch_step2_decode()
        patch_step2_runner()
        patch_main_loop()
    except Exception as e:
        print(f"\n❌ Patch 失败: {e}")
        print(f"   请检查 .bak 备份文件, 必要时手动恢复")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  ✅ v16 Patch 完成")
    print("=" * 70)
    print("""
启动 v16 实验:

  cd /mnt/st_data/liangxinyi/code/
  python main_loop.py \\
      --experiment_dir /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/ \\
      --feddna_checkpoint /mnt/st_data/liangxinyi/code/result/FLDNA_I/I_1214234233/model/epoch1_I.pth \\
      --ref_length 196 \\
      --disable_merge \\
      --consensus_source mv \\
      ...其他参数保持与 v15 一致...

判据看三个信号 (详见 v16_transition.md):
  1. EER 逐轮趋势:     应当不再单调递增
  2. 纯簇失败 (pure):  应当明显减少 (v15 R3=763)
  3. 簇消失 (uncov):   可能附带改善, 也可能不变 (v15 R3=656)

对照实验 (可选):
  --consensus_source fusion   # 退回 v15 行为, 做 A/B 对照
""")
    print("回滚命令:")
    for tag, path in FILES.items():
        print(f"  mv {path}.bak {path}")


if __name__ == "__main__":
    main()