#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# spike_centroid_nozone.py
# 只读 spike: 测"质心计算不用 zone_ids"对 SR 的影响
# -----------------------------------------------------------------------------
# 目的:
#   当前 compute_centroids_weighted 用 zone_ids 排除 Zone III、对 Zone II 截断。
#   死数据复活/归巢/Rebirth 删除后, zone_ids 唯一实质用途只剩"喂质心排噪"。
#   本 spike 把质心换成"全 label>=0 read 按 strength 加权"(完全不看 zone),
#   跑三轮看 SR:
#     - SR 仍 0.9539/0.9662/0.9699 -> 质心排噪无贡献 -> zone_ids/delta/三区划分可全删 (路线一)
#     - SR 掉 -> 质心需要 zone 排噪 -> 保留 zone 喂质心 (路线二)
#
# 单变量保证: 唯一改动是 compute_centroids_weighted 的实现。其余流程(训练/拆分/
#   consensus/评估/clean_mode/拆分参数)全部走标准命令, 一字不变。
#
# 零污染: 不改任何源文件, 仅在内存里 monkey-patch 函数引用。跑完删本脚本即可。
#
# 用法 (参数与你的标准三轮命令完全一致, 直接照搬, 只是把 python main_loop.py
#       换成 python spike_centroid_nozone.py):
#
#   python spike_centroid_nozone.py \
#       --experiment_dir /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d/ \
#       --feddna_checkpoint .../epoch1_I.pth \
#       --gt_tags_file ... --gt_refs_file ... \
#       --max_iterations 3 --max_length 201 --target_clusters 11736 \
#       --cl_mode ours --ref_length 196 --primer_prefix 20 --primer_suffix 20 \
#       --disable_merge --consensus_source mv --fasta_source mv_strict \
#       --zone_include_noise True --rebirth_mode nearest \
#       --enable_split --split_tau 5 --split_min_size 6 --clean_mode \
#       2>&1 | tee spike_centroid_nozone.log
# =============================================================================
import os
import sys
import torch
from collections import defaultdict

# 确保能 import models.*
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_HERE, _PARENT):
    if p not in sys.path:
        sys.path.insert(0, p)


# -----------------------------------------------------------------------------
# 替换版质心: 全 label>=0 read, 纯 strength 加权, 不看 zone_ids
# 签名与原函数完全一致 (embeddings, labels, strength, zone_ids), zone_ids 收下但忽略,
# 保证调用方零改动。
# -----------------------------------------------------------------------------
def compute_centroids_nozone(embeddings, labels, strength, zone_ids):
    """[SPIKE] 不使用 zone_ids: 所有 label>=0 read 按 strength 加权算质心。"""
    centroids = {}
    cluster_sizes = {}

    label_to_idx = defaultdict(list)
    labels_cpu = labels.cpu().numpy()
    for i, l in enumerate(labels_cpu):
        if l >= 0:
            label_to_idx[int(l)].append(i)

    for k_int, idxs in label_to_idx.items():
        mask_idx = torch.tensor(idxs, dtype=torch.long, device=embeddings.device)
        emb = embeddings[mask_idx]
        w   = strength[mask_idx]

        w_sum = w.sum()
        if w_sum < 1e-10:
            centroids[k_int] = emb.mean(dim=0)
        else:
            centroids[k_int] = (emb * w.unsqueeze(1)).sum(dim=0) / w_sum

        cluster_sizes[k_int] = len(idxs)

    print(f"   📍 [SPIKE] 质心(无zone, 全read strength加权): {len(centroids)} 个簇")
    return centroids, cluster_sizes


def _apply_patch():
    """在 step2_refine 和 step2_runner 两处命名空间都替换, 确保 patch 生效。

    step2_runner 用 `from models.step2_refine import compute_centroids_weighted`
    把函数绑到了自己的命名空间, 所以必须 patch step2_runner.compute_centroids_weighted,
    光 patch step2_refine 的没用。两处都 patch 最稳。
    """
    import models.step2_refine as refine
    import models.step2_runner as runner

    refine.compute_centroids_weighted = compute_centroids_nozone
    runner.compute_centroids_weighted = compute_centroids_nozone

    # 校验: 确认 runner 命名空间里确实换成了 spike 版
    assert runner.compute_centroids_weighted is compute_centroids_nozone, \
        "patch 失败: runner.compute_centroids_weighted 未被替换"
    print("=" * 70)
    print("  🔧 [SPIKE] 已 monkey-patch compute_centroids_weighted -> 无zone版")
    print("     (step2_refine + step2_runner 两处命名空间均已替换)")
    print("     唯一变量: 质心不再用 zone_ids 排噪; 其余流程不变")
    print("=" * 70)


if __name__ == "__main__":
    _apply_patch()
    # patch 打好后, 调标准入口。main_loop() 内部自己 parse_args, 读 sys.argv,
    # 所以命令行参数照常透传。
    from models.main_loop import main_loop
    main_loop()